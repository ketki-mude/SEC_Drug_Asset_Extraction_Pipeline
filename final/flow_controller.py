import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_ticker(ticker):
    """
    Main entry point to process a ticker symbol.
    Simplified pipeline with single check for merged files.
    
    Args:
        ticker (str): Company ticker symbol
        
    Returns:
        dict: Processed data or error message
    """
    ticker = ticker.upper()
    logging.info(f"Starting pipeline for ticker {ticker}")
    
    # Import necessary modules
    from aws_storage import S3Storage
    from s3_fetch_split import S3FileFetcher
    import chunking_upload_to_pincone as chunking
    import RAG_GEMINI as rag
    import sec_filing_extractor as sec_extractor
    
    # Initialize S3 fetcher to check for merged files
    s3 = S3Storage()
    fetcher = S3FileFetcher()
    
    # 1. Check if output JSON already exists (shortcut)
    try:
        # List files in output directory for this ticker
        response = s3.s3_client.list_objects_v2(
            Bucket=s3.bucket_name,
            Prefix=f"output/{ticker}/"
        )
        
        json_files = []
        if 'Contents' in response:
            json_files = [item for item in response['Contents'] 
                         if item['Key'].endswith('.json')]
        
        if json_files:
            # Sort by last modified to get the most recent
            json_files.sort(key=lambda x: x['LastModified'], reverse=True)
            latest_file = json_files[0]['Key']
            
            # Get the file content
            obj = s3.s3_client.get_object(
                Bucket=s3.bucket_name, 
                Key=latest_file
            )
            data = json.loads(obj['Body'].read().decode('utf-8'))
            
            logging.info(f"Found existing results for {ticker} at {latest_file}")
            return {
                "status": "success",
                "data": data
            }
    except Exception as e:
        logging.info(f"No existing output found: {str(e)}")
    
    # 2. Check if merged files exist
    merged_files_exist = False
    try:
        # Check for 8-K and 10-K merged files
        file_patterns = [f"{ticker}/8-K_merged.txt", f"{ticker}/10-K_merged.txt"]
        
        for pattern in file_patterns:
            response = s3.s3_client.list_objects_v2(
                Bucket=s3.bucket_name,
                Prefix=pattern
            )
            
            if 'Contents' in response and len(response['Contents']) > 0:
                logging.info(f"Found merged file {pattern} in S3")
                merged_files_exist = True
                break
    except Exception as e:
        logging.error(f"Error checking for merged files: {str(e)}")
    
    # 3. Process based on what we found
    if merged_files_exist:
        logging.info(f"Found merged files for {ticker}. Skipping SEC extraction step.")
        # Skip the SEC extraction step and continue with fetch and split
    else:
        logging.info(f"No merged files found for {ticker}. Running full pipeline.")
        # Run SEC filing extractor to get the raw files
        success = sec_extractor.process_ticker_consolidated(
            ticker=ticker,
            years_back=2,
            limit_per_form=5
        )
        
        if not success:
            logging.error(f"SEC filing extraction failed for {ticker}")
            return {
                "status": "error",
                "message": f"Failed to extract SEC filings for {ticker}"
            }
    
    # 4. Always run S3 fetch and split
    logging.info(f"Running S3 fetch and split for {ticker}")
    sections = fetcher.fetch_and_split(ticker)
    
    if not sections:
        logging.error(f"No sections found for {ticker}")
        return {
            "status": "error",
            "message": f"No content sections found for {ticker}"
        }
    
    # 5. Always run chunking and upload to Pinecone
    logging.info(f"Running chunking and upload to Pinecone for {ticker}")
    chunker = chunking.SemanticChunker()
    num_chunks = chunker.process_sections(sections)
    
    if num_chunks == 0:
        logging.error(f"No chunks created for {ticker}")
        return {
            "status": "error",
            "message": f"Failed to create embeddings for {ticker}"
        }
    
    # 6. Always run RAG processing
    logging.info(f"Running RAG processing for {ticker}")
    extractor = rag.ProductNameExtractor()
    
    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    filename_base = f"{ticker.lower()}_products_{timestamp}"
    
    # Run the detailed analysis with S3 upload
    results = extractor.analyze_detailed_products(
        ticker=ticker,
        top_k=15,
        output_json=None,  # Don't save locally
        output_table=None,
        upload_to_s3=True   # Upload to S3
    )
    
    if "error" in results:
        logging.error(f"Error in RAG processing: {results['error']}")
        return {
            "status": "error",
            "message": f"Error in analysis: {results['error']}"
        }
    
    if "products" in results:
        logging.info(f"Successfully processed {ticker}")
        return {
            "status": "success",
            "data": results["products"]
        }
    else:
        logging.error(f"No products found for {ticker}")
        return {
            "status": "error",
            "message": f"No products found for {ticker}"
        }

# For direct testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python flow_controller.py TICKER")
        sys.exit(1)
        
    ticker = sys.argv[1].upper()
    result = process_ticker(ticker)
    
    if result["status"] == "success":
        print(f"Successfully processed {ticker}")
        print(f"Found {len(result['data'])} products")
    else:
        print(f"Error: {result['message']}")





