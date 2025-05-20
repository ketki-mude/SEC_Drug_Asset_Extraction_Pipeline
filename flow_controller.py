import os
import json
import logging
from datetime import datetime
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_s3_cache(ticker):
    """Check if analysis results already exist in S3"""
    try:
        from aws_storage import S3Storage
        s3 = S3Storage()
        
        # Check for JSON file
        json_key = f"output/{ticker}/{ticker}_products.json"
        
        if s3.file_exists(json_key):
            logging.info(f"Found existing analysis for {ticker} in S3")
            json_data = s3.get_file(json_key)
            if json_data:
                return {
                    "status": "success",
                    "data": json.loads(json_data),
                    "message": f"Retrieved cached analysis for {ticker}",
                    "cached": True
                }
    except Exception as e:
        logging.error(f"Error checking S3 cache: {str(e)}")
    
    return None

def process_ticker(ticker):
    """Main entry point with S3 cache check"""
    ticker = ticker.upper()
    logging.info(f"Processing request for {ticker}")
    
    # First check S3 cache
    cached_result = check_s3_cache(ticker)
    if cached_result:
        logging.info(f"Using cached analysis for {ticker}")
        return cached_result
    
    try:
        # Import components
        from s3_fetch_split import S3FileFetcher
        import chunking_upload_to_pinecone as chunking
        import RAG_GEMINI as rag
        import sec_filing_extractor as sec_extractor
        
        # Initialize chunker
        chunker = chunking.SemanticChunker()

        # 1. SEC Filing Extraction
        success = sec_extractor.process_ticker_consolidated(ticker)
        if not success:
            return {
                "status": "error",
                "message": f"Failed to extract SEC filings for {ticker}"
            }

        # 2. Fetch and Split
        fetcher = S3FileFetcher()
        sections = fetcher.fetch_and_split(ticker)
        if not sections:
            return {
                "status": "error",
                "message": f"No content sections found for {ticker}"
            }

        # 3. Process sections and upload to Pinecone
        num_chunks = chunker.process_sections(sections)
        if num_chunks == 0:
            return {
                "status": "error",
                "message": f"Failed to create chunks for {ticker}"
            }

        # Wait for Pinecone indexing
        time.sleep(5)

        # 4. RAG Analysis
        extractor = rag.ProductNameExtractor()
        chunks = extractor.retrieve_product_chunks(ticker, top_k=15)
        results = extractor.extract_product_names(chunks, ticker)
        
        detailed_results = extractor.analyze_detailed_products(
            ticker=ticker,
            top_k=15,
            output_json=None,
            output_table=None,
            upload_to_s3=False  # We'll handle S3 storage explicitly
        )
        
        if detailed_results and "products" in detailed_results:
            # Store results in S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{ticker}_products_{timestamp}"
            
            s3_success = extractor.save_to_s3(
                ticker=ticker,
                filename_base=filename_base,
                products=detailed_results["products"]
            )
            
            if s3_success:
                logging.info(f"Successfully stored {ticker} analysis in S3")
            else:
                logging.warning(f"Failed to store {ticker} analysis in S3")
            
            return {
                "status": "success",
                "data": detailed_results["products"],
                "message": f"Found {len(detailed_results['products'])} products and stored in S3",
                "cached": False,
                "s3_stored": s3_success
            }
        else:
            logging.error("Second pass failed or returned no products")
            return {
                "status": "error",
                "message": "Failed to get detailed product information"
            }

    except Exception as e:
        logging.error(f"Error processing {ticker}: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

# For direct testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python flow_controller.py TICKER")
        sys.exit(1)
        
    ticker = sys.argv[1].upper()
    result = process_ticker(ticker)
    
    print(f"Status: {result['status']}")
    if result["status"] == "success":
        print(f"Found {len(result['data'])} products")
    else:
        print(f"Error: {result['message']}")