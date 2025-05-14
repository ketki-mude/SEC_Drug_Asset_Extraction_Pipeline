import os
import json
import logging
import argparse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from aws_storage import S3Storage
import pandas as pd
from io import StringIO
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Load environment variables
load_dotenv()

class ProductNameExtractor:
    """Extract product names from SEC filings using Pinecone and Gemini"""
    
    def __init__(self):
        """Initialize connections to Pinecone and Gemini"""
        # Set up the embedding model
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        logging.info("Loaded embedding model: all-mpnet-base-v2")
        
        # Set up Pinecone
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = self.pc.Index(os.getenv('PINECONE_INDEX_NAME'))
        logging.info(f"Connected to Pinecone index: {os.getenv('PINECONE_INDEX_NAME')}")
        
        # Set up Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'gemini-pro'))
        logging.info(f"Connected to Gemini model: {os.getenv('GEMINI_MODEL', 'gemini-pro')}")
    
    def retrieve_product_chunks(self, ticker, top_k=15, form_types=None):
        """
        Retrieve chunks from Pinecone that likely contain product information
        
        Args:
            ticker (str): Company ticker symbol
            top_k (int): Number of results to return per query
            form_types (list): Optional list of form types to filter (e.g., ["10-K", "8-K"])
            
        Returns:
            list: List of text chunks and metadata from Pinecone
        """
        logging.info(f"Retrieving product information for {ticker}")
        
        # Product-related search queries
        product_queries = [
            "product names portfolio offerings",
            "product pipeline candidates",
            "products and services",
            "key products main products",
            "product line flagship products",
            "new products product releases",
            "proprietary products patented products"
        ]
        
        # Build metadata filter
        metadata_filter = {"ticker_name": ticker}
        if form_types:
            metadata_filter["form_type"] = {"$in": form_types}
        
        all_chunks = []
        
        # Perform multiple searches with different queries to ensure comprehensive coverage
        for query in product_queries:
            try:
                # Generate query embedding
                query_embedding = self.embedding_model.encode(query).tolist()
                
                # Perform the search
                response = self.index.query(
                    vector=query_embedding,
                    filter=metadata_filter,
                    top_k=top_k,
                    include_metadata=True
                )
                
                # Process results
                for match in response.matches:
                    if hasattr(match, 'metadata') and 'text' in match.metadata:
                        chunk_data = {
                            'text': match.metadata.get('text', ''),
                            'ticker': match.metadata.get('ticker_name', ''),
                            'filing_date': match.metadata.get('filing_date', ''),
                            'form_type': match.metadata.get('form_type', ''),
                            'score': match.score
                        }
                        all_chunks.append(chunk_data)
                
                logging.info(f"Query '{query}' returned {len(response.matches)} chunks")
                
            except Exception as e:
                logging.error(f"Error searching for '{query}': {str(e)}")
        
        # Remove duplicates (keeping highest scoring version of each chunk)
        unique_chunks = {}
        for chunk in all_chunks:
            text = chunk['text']
            if text not in unique_chunks or chunk['score'] > unique_chunks[text]['score']:
                unique_chunks[text] = chunk
        
        result_chunks = list(unique_chunks.values())
        logging.info(f"Retrieved {len(result_chunks)} unique chunks containing potential product information")
        
        return result_chunks
    
    def extract_product_names(self, chunks, ticker):
        """
        Use Gemini to extract product names from chunks
        
        Args:
            chunks (list): List of text chunks with metadata
            ticker (str): Company ticker symbol for context
            
        Returns:
            dict: Extracted product information
        """
        logging.info(f"Extracting product names from {len(chunks)} chunks")
        
        if not chunks:
            return {"products": [], "error": "No relevant chunks found"}
        
        # Combine chunks with metadata for context
        context = "\n\n".join([
            f"CHUNK {i+1} [From {chunk['form_type']} filed on {chunk['filing_date']}]:\n{chunk['text']}"
            for i, chunk in enumerate(chunks[:20])  # Limit to top 20 chunks to avoid token limits
        ])
        
        # Create a specialized prompt for product name extraction
        prompt = f"""
        Extract ONLY actual drug candidates, therapeutic programs, and commercial products mentioned in these SEC filing excerpts for {ticker}. 

        For biotechnology/pharmaceutical companies, the following ARE products or drug candidates:
        - Target genes being developed into therapies 
        - Molecular platforms when they are described as actual products/services
        - Therapeutic modalities when they have specific names and are being developed
        
        The following are NEVER products (MUST EXCLUDE THESE):
        - ANY financial instruments: shares, securities, warrants, stocks, offerings, notes, bonds, options
        - ANY corporate entities, subsidiaries, or partnerships
        - ANY facilities, buildings, or real estate
        - General technology platforms unless specifically named as commercial offerings
        - Generic research programs without specific identifiers
        
        For each product/candidate:
        1. Extract the EXACT product name or identifier (preserving any codes)
        2. Provide a brief description based ONLY on the text
        3. Include development stage information if available
        4. Include the target disease/condition if mentioned
        
        Biotechnology-specific instructions:
        - Single gene names ARE valid product candidates if they are being targeted for therapy
        - Include ALL development programs with specific identifiers, regardless of stage
        - Look for terms like "pipeline," "candidate," "program," "therapy" to identify products
        
        SEC filing excerpts:
        {context}
        
        Format your response as a structured JSON with this format:
        {{
          "products": [
            {{
              "name": "Product name exactly as written",
              "description": "Brief description based only on provided text",
              "status": "Development stage (Phase 1, preclinical, etc.) if mentioned, otherwise 'Unknown'",
              "target_indication": "Disease or condition targeted if mentioned",
              "confidence": "high/medium/low based on clarity in text"
            }}
          ]
        }}
        
        Only include the JSON in your response, no other text.
        """
        
        try:
            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Try to parse JSON from response
            try:
                # Remove markdown code block formatting if present
                clean_response = response_text
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0].strip()
                
                product_data = json.loads(clean_response)
                
                # Add metadata
                product_data["ticker"] = ticker
                product_data["chunks_processed"] = len(chunks)
                
                return product_data
                
            except json.JSONDecodeError as je:
                logging.error(f"Error parsing JSON from Gemini response: {str(je)}")
                return {
                    "products": [],
                    "error": "Failed to parse product data",
                    "raw_response": response_text
                }
                
        except Exception as e:
            logging.error(f"Error with Gemini extraction: {str(e)}")
            return {
                "products": [],
                "error": str(e)
            }
    
    def run(self, ticker, top_k=15, form_types=None, output_file=None):
        """
        Run the complete product extraction pipeline
        
        Args:
            ticker (str): Company ticker symbol
            top_k (int): Number of results per query
            form_types (list): Optional list of form types to filter
            output_file (str): Optional file path to save results
            
        Returns:
            dict: Extracted product information
        """
        # Retrieve relevant chunks
        chunks = self.retrieve_product_chunks(ticker, top_k, form_types)
        
        # Extract product names
        results = self.extract_product_names(chunks, ticker)
        
        # Save results if output file provided
        if output_file and results:
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logging.info(f"Saved results to {output_file}")
            except Exception as e:
                logging.error(f"Error saving results to {output_file}: {str(e)}")
        
        return results

    def extract_product_set(self, results):
        """Extract unique product names into a set for further processing"""
        product_set = set()
        
        if "products" in results:
            for product in results["products"]:
                if "name" in product:
                    # Filter out financial instruments that might have been misclassified
                    name = product["name"].lower()
                    if not any(term in name for term in ["shares", "securities", "warrants", "offering"]):
                        product_set.add(product["name"])
        
        logging.info(f"Extracted {len(product_set)} unique product names")
        return product_set

    def query_product_details(self, ticker, product_name, top_k=25):
        """Query Pinecone for detailed information about a specific product"""
        logging.info(f"Gathering details for product: {product_name}")
        
        # Create targeted queries for different aspects we need
        detail_queries = [
            f"{product_name} mechanism of action how works",
            f"{product_name} target molecule pathway",
            f"{product_name} indication disease condition",
            f"{product_name} preclinical animal study model",
            f"{product_name} clinical trial phase results",
            f"{product_name} development milestone upcoming"
        ]
        
        all_chunks = []
        
        # Query for each aspect
        for query in detail_queries:
            try:
                query_embedding = self.embedding_model.encode(query).tolist()
                
                metadata_filter = {"ticker_name": ticker}
                
                response = self.index.query(
                    vector=query_embedding,
                    filter=metadata_filter,
                    top_k=max(3, top_k // len(detail_queries)),  # Distribute top_k among queries
                    include_metadata=True
                )
                
                for match in response.matches:
                    if hasattr(match, 'metadata') and 'text' in match.metadata:
                        chunk = {
                            'text': match.metadata.get('text', ''),
                            'ticker': match.metadata.get('ticker_name', ''),
                            'filing_date': match.metadata.get('filing_date', ''),
                            'form_type': match.metadata.get('form_type', ''),
                            'score': match.score
                        }
                        all_chunks.append(chunk)
                        
            except Exception as e:
                logging.error(f"Error querying for {query}: {str(e)}")
        
        # Remove duplicates but keep highest scoring version
        unique_chunks = {}
        for chunk in all_chunks:
            text = chunk['text']
            if text not in unique_chunks or chunk['score'] > unique_chunks[text]['score']:
                unique_chunks[text] = chunk
        
        result_chunks = list(unique_chunks.values())
        # Sort by relevance score
        result_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        logging.info(f"Retrieved {len(result_chunks)} unique chunks for {product_name}")
        return result_chunks

    def extract_product_details(self, ticker, product_name, chunks):
        """Generate structured product details using Gemini"""
        logging.info(f"Extracting detailed information for {product_name}")
        
        if not chunks:
            return {
                "name": product_name,
                "mechanism_of_action": "",
                "target": "",
                "indication": "",
                "preclinical_data": [],
                "clinical_trials": [],
                "upcoming_milestones": [],
                "references": []
            }
        
        # Combine chunks with metadata for context
        context = "\n\n".join([
            f"CHUNK {i+1} [From {chunk['form_type']} filed on {chunk['filing_date']}]:\n{chunk['text']}"
            for i, chunk in enumerate(chunks[:15])  # Limit to avoid token limits
        ])
        
        # Create a detailed extraction prompt
        prompt = f"""
        Based on the SEC filing excerpts provided, extract detailed information about the product/candidate {product_name} from {ticker}.

        Return ONLY a JSON object with the following structure and fields:
        {{
          "name": "{product_name}",
          "mechanism_of_action": "Detailed description of how the product works (e.g., siRNA to silence gene expression)",
          "target": "Biological molecule or pathway targeted (e.g., INHBE, dystrophin pre-mRNA)",
          "indication": "Disease or condition being targeted (e.g., Duchenne muscular dystrophy)",
          "preclinical_data": [
            "Bullet-point summaries of key animal experiments or preclinical results"
          ],
          "clinical_trials": [
            "Structured information about each clinical trial including phase, participants, results if available"
          ],
          "upcoming_milestones": [
            "Expected future catalysts or events related to this product"
          ],
          "references": [
            "Document references where this information was found (e.g., '10-K filed on 2023-03-15')"
          ]
        }}

        For each field:
        - If no information is available, use an empty string or empty array
        - Include specific dates, numbers, and results when available
        - Be comprehensive but concise
        - Only include information that is explicitly mentioned in the text

        SEC filing excerpts about {product_name}:
        {context}
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Process the response to extract JSON
            clean_response = response_text
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0].strip()
            
            product_data = json.loads(clean_response)
            logging.info(f"Successfully extracted structured data for {product_name}")
            return product_data
            
        except Exception as e:
            logging.error(f"Error extracting product details: {str(e)}")
            return {
                "name": product_name,
                "mechanism_of_action": "",
                "target": "",
                "indication": "",
                "preclinical_data": [],
                "clinical_trials": [],
                "upcoming_milestones": [],
                "references": []
            }

    def create_product_table(self, product_details, output_file=None):
        """Generate a formatted table from product details"""
        try:
            import pandas as pd
            from tabulate import tabulate
            
            # Define columns to match the image
            columns = [
                "Name/Number", 
                "Mechanism of Action", 
                "Target", 
                "Indication", 
                "Animal Models / Preclinical Data", 
                "Clinical Trials",
                "Upcoming Milestones",
                "References"
            ]
            
            # Create rows for each product
            rows = []
            for product in product_details:
                # Format list fields as bullet points
                preclinical = product.get("preclinical_data", [])
                preclinical_formatted = "\n".join([f"• {item}" for item in preclinical]) if preclinical else ""
                
                clinical = product.get("clinical_trials", [])
                clinical_formatted = "\n".join([f"• {item}" for item in clinical]) if clinical else ""
                
                milestones = product.get("upcoming_milestones", [])
                milestones_formatted = "\n".join([f"• {item}" for item in milestones]) if milestones else ""
                
                references = product.get("references", [])
                references_formatted = "\n".join([f"• {item}" for item in references]) if references else ""
                
                row = [
                    product.get("name", ""),
                    product.get("mechanism_of_action", ""),
                    product.get("target", ""),
                    product.get("indication", ""),
                    preclinical_formatted,
                    clinical_formatted,
                    milestones_formatted,
                    references_formatted
                ]
                
                rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=columns)
            
            # Save to file if requested
            if output_file:
                if output_file.endswith('.csv'):
                    df.to_csv(output_file, index=False)
                elif output_file.endswith('.xlsx'):
                    df.to_excel(output_file, index=False)
                elif output_file.endswith('.html'):
                    df.to_html(output_file, index=False)
                else:
                    with open(output_file, 'w') as f:
                        f.write(tabulate(df, headers='keys', tablefmt='pipe'))
                
                logging.info(f"Saved product table to {output_file}")
            
            print("\nProduct Analysis Table:")
            print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
            
            return df
        except ImportError:
            logging.error("Required packages not found. Install with: pip install pandas tabulate openpyxl")
            print("Table generation failed. Install required packages with: pip install pandas tabulate openpyxl")
            return None

    def analyze_detailed_products(self, ticker, top_k=15, output_json=None, output_table=None, upload_to_s3=False):
        """Complete pipeline to extract product details and generate table"""
        logging.info(f"Starting detailed product analysis for {ticker}")
        
        # Step 1: Get initial product list
        chunks = self.retrieve_product_chunks(ticker, top_k)
        results = self.extract_product_names(chunks, ticker)
        
        # Step 2: Create set of product names
        product_set = self.extract_product_set(results)
        
        if not product_set:
            logging.warning(f"No products identified for {ticker}")
            return {
                "ticker": ticker,
                "products": [],
                "error": "No products identified"
            }
        
        # Step 3: Get detailed information for each product
        detailed_products = []
        
        for product_name in product_set:
            # Query Pinecone for this specific product
            product_chunks = self.query_product_details(ticker, product_name, top_k=30)
            
            # Extract structured data
            product_details = self.extract_product_details(ticker, product_name, product_chunks)
            
            detailed_products.append(product_details)
        
        # Step 4: Save detailed results locally if requested
        if output_json:
            try:
                with open(output_json, 'w') as f:
                    json.dump(detailed_products, f, indent=2)
                logging.info(f"Saved detailed results to local file: {output_json}")
            except Exception as e:
                logging.error(f"Error saving detailed results locally: {str(e)}")
        
        # Step 5: Generate table if requested
        if output_table:
            self.create_product_table(detailed_products, output_table)
        
        # Step 6: Save to S3 if requested
        if upload_to_s3:
            # Get base filename without extension
            if output_json:
                filename_base = os.path.splitext(os.path.basename(output_json))[0]
            else:
                filename_base = f"{ticker.lower()}_products_{datetime.now().strftime('%Y%m%d')}"
                
            s3_result = self.save_to_s3(ticker, filename_base, detailed_products)
            if s3_result:
                logging.info(f"Successfully saved all files to S3 for {ticker}")
            else:
                logging.warning(f"There were issues saving some files to S3 for {ticker}")
        
        return {
            "ticker": ticker,
            "products": detailed_products,
            "product_count": len(detailed_products)
        }

    def filter_products_for_json(self, products):
        """Remove products with any null/empty values for JSON output"""
        filtered_products = []
        
        for product in products:
            # Check if any field is empty/null
            has_empty_fields = (
                not product.get("name", "") or
                not product.get("mechanism_of_action", "") or
                not product.get("target", "") or
                not product.get("indication", "") or
                len(product.get("preclinical_data", [])) == 0 or
                len(product.get("clinical_trials", [])) == 0 or
                len(product.get("upcoming_milestones", [])) == 0 or
                len(product.get("references", [])) == 0
            )
            
            if not has_empty_fields:
                filtered_products.append(product)
        
        logging.info(f"Filtered products for JSON: {len(filtered_products)} of {len(products)} have complete data")
        return filtered_products

    def convert_to_markdown(self, products, ticker):
        """Convert product details to markdown format"""
        md = f"# Product Analysis for {ticker}\n\n"
        md += f"*Generated on: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        md += f"## Overview\n\n"
        md += f"This analysis found {len(products)} products/candidates for {ticker}.\n\n"
        
        for idx, product in enumerate(products, 1):
            md += f"## {idx}. {product.get('name', 'Unknown Product')}\n\n"
            
            md += f"**Mechanism of Action:** {product.get('mechanism_of_action', 'N/A')}\n\n"
            md += f"**Target:** {product.get('target', 'N/A')}\n\n"
            md += f"**Indication:** {product.get('indication', 'N/A')}\n\n"
            
            # Add preclinical data
            md += "### Animal Models / Preclinical Data\n\n"
            preclinical = product.get("preclinical_data", [])
            if preclinical:
                for item in preclinical:
                    md += f"* {item}\n"
            else:
                md += "No preclinical data found.\n"
            md += "\n"
            
            # Add clinical trials
            md += "### Clinical Trials\n\n"
            trials = product.get("clinical_trials", [])
            if trials:
                for trial in trials:
                    md += f"* {trial}\n"
            else:
                md += "No clinical trial data found.\n"
            md += "\n"
            
            # Add milestones
            md += "### Upcoming Milestones\n\n"
            milestones = product.get("upcoming_milestones", [])
            if milestones:
                for milestone in milestones:
                    md += f"* {milestone}\n"
            else:
                md += "No upcoming milestones found.\n"
            md += "\n"
            
            # Add references
            md += "### References\n\n"
            references = product.get("references", [])
            if references:
                for ref in references:
                    md += f"* {ref}\n"
            else:
                md += "No references found.\n"
            md += "\n---\n\n"
        
        return md

    def save_to_s3(self, ticker, filename_base, products):
        """Save product details to S3 in JSON and MD formats (no CSV)"""
        try:
            # Initialize S3Storage
            s3 = S3Storage()
            if not s3.check_connection():
                logging.error("Failed to connect to S3")
                return False
            
            # Create base folder path
            base_path = f"output/{ticker}"
            success = True
            
            # 1. Save JSON (filtered to remove products with null values)
            filtered_products = self.filter_products_for_json(products)
            json_key = f"{base_path}/{filename_base}.json"
            # The JSON upload is working fine, no changes needed
            json_result = s3.save_file(
                json_key, 
                json.dumps(filtered_products, indent=2),
                content_type="application/json"
            )
            if json_result:
                logging.info(f"Saved filtered JSON ({len(filtered_products)} complete products) to S3: {json_key}")
            else:
                logging.error(f"Failed to save JSON to S3: {json_key}")
                success = False
            
            # 2. Save Markdown (all products)
            try:
                md_content = self.convert_to_markdown(products, ticker)
                
                md_key = f"{base_path}/{filename_base}.md"
                md_result = s3.save_file(
                    md_key,
                    md_content,
                    content_type="text/markdown"
                )
                if md_result:
                    logging.info(f"Saved Markdown (all {len(products)} products) to S3: {md_key}")
                else:
                    logging.error(f"Failed to save Markdown to S3: {md_key}")
                    success = False
            except Exception as e:
                logging.error(f"Error processing Markdown for S3 upload: {str(e)}")
                success = False
            
            return success
                
        except Exception as e:
            logging.error(f"Error saving to S3: {str(e)}")
            return False


if __name__ == "__main__":
    from datetime import datetime  # Add this import
    
    parser = argparse.ArgumentParser(description="Extract product names from SEC filings")
    parser.add_argument("--ticker", required=True, help="Company ticker symbol")
    parser.add_argument("--top-k", type=int, default=15, help="Number of chunks to retrieve per query")
    parser.add_argument("--form-types", nargs="+", help="Optional list of form types to filter (e.g., 10-K 8-K)")
    parser.add_argument("--output", help="Output JSON file for saving the results")
    parser.add_argument("--detailed", action="store_true", help="Run detailed product analysis")
    parser.add_argument("--json-details", help="Output JSON file for detailed product information")
    parser.add_argument("--table", help="Output file for product table (csv, xlsx, html)")
    parser.add_argument("--s3-upload", action="store_true", help="Upload results to S3")
    
    args = parser.parse_args()
    
    extractor = ProductNameExtractor()
    
    if args.detailed or args.table or args.json_details:
        # Run detailed analysis pipeline
        extractor.analyze_detailed_products(
            ticker=args.ticker,
            top_k=args.top_k,
            output_json=args.json_details or args.output,
            output_table=args.table,
            upload_to_s3=args.s3_upload
        )
        print(f"\nCompleted detailed analysis for {args.ticker}")
    else:
        # This will use the existing run() method to extract basic product info
        results = extractor.run(
            ticker=args.ticker,
            top_k=args.top_k,
            form_types=args.form_types,
            output_file=args.output
        )
        
        # Print results
        print(f"\nExtracted Products for {args.ticker}:")
        print(json.dumps(results, indent=2))
        
        if "products" in results:
            print(f"\nFound {len(results['products'])} products")
            
            # Print a simple list for quick reference
            if results["products"]:
                print("\nProduct List:")
                for i, product in enumerate(results["products"], 1):
                    print(f"{i}. {product['name']} - {product.get('status', 'N/A')}")