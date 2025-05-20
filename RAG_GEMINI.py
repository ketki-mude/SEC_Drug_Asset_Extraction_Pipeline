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
            "investigational candidate",
            "pipeline candidate",
            "clinical candidate",
            "preclinical data",
            "development candidates",
            "clinical trial",
            "proof of mechanism"
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
        # Enhanced logging for product name extraction
        logging.info(f"Starting product name extraction for {ticker}")
        
        # Log chunk summary
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            logging.info(f"Chunk {i+1} preview: {chunk['text'][:200]}...")
        
        # Build context with logging
        logging.info("Building context from chunks...")
        context = "\n\n".join([
            f"CHUNK {i+1} [From {chunk['form_type']} filed on {chunk['filing_date']}]:\n{chunk['text']}"
            for i, chunk in enumerate(chunks[:20])
        ])
        logging.info(f"Built context with {len(context)} characters")
        
        # Create a specialized prompt for product name extraction
        prompt = f"""
            You are a biotech data extractor analyzing SEC filings for {ticker}.
            For each drug candidate or program mentioned, extract:
            1. Exact product name/identifier
            2. Mechanism of action (HOW it works)
            3. Indication (WHAT disease/condition it treats)
            4. Development status
            
            Only include products that have clear identifiers.
            
            SEC filing excerpts:
            {context}
            
            Return a JSON with this EXACT structure:
            {{
              "products": [
                {{
                  "name": "Product identifier (e.g. EDG-7500)",
                  "mechanism_of_action": "How the drug works (e.g. selective beta-1 adrenergic receptor antagonist)",
                  "indication": "Disease/condition being treated (e.g. supraventricular tachycardia)",
                  "status": "Development stage",
                  "confidence": "high/medium/low"
                }}
              ]
            }}
            
            Important:
            - For mechanism_of_action and indication, extract specific details from the text
            - Do NOT return "N/A" or empty strings unless truly no information exists
            - Include exact quotes from the text when possible
            - Consolidate information about the same product from different sections
        """
        
        # Add debug logging for response content
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            logging.debug(f"Full Gemini Response:\n{response_text}")
            
            # Clean and parse JSON
            clean_response = response_text
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0].strip()
            
            product_data = json.loads(clean_response)
            
            # Debug log each product's complete information
            for product in product_data.get("products", []):
                logging.debug(f"""
                Product Details:
                Name: {product.get('name')}
                MOA: {product.get('mechanism_of_action')}
                Indication: {product.get('indication')}
                Status: {product.get('status')}
                Confidence: {product.get('confidence')}
                """)
            
            return product_data
            
        except Exception as e:
            logging.error(f"Error in extract_product_names: {str(e)}")
            return {"products": []}
    
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
        """Extract unique product names with improved deduplication"""
        product_map = {}  # Use a map to track all variations and their information
        
        if "products" in results:
            for product in results["products"]:
                if "name" in product:
                    # Normalize the name
                    name = product["name"].strip().upper()
                    
                    # Handle common variations
                    name = name.replace('CANDIDATE', '').strip()
                    
                    # Skip financial instruments
                    if any(term in name.lower() for term in ["shares", "securities", "warrants", "offering"]):
                        continue
                    
                    # If this is a new product or has better information
                    if (name not in product_map or 
                        (not product_map[name].get("target_indication") and product.get("target_indication")) or
                        (not product_map[name].get("mechanism_of_action") and product.get("mechanism_of_action"))):
                        
                        product_map[name] = {
                            "name": name,
                            "target_indication": product.get("target_indication", ""),
                            "mechanism_of_action": product.get("mechanism_of_action", ""),
                            "status": product.get("status", ""),
                            "confidence": product.get("confidence", "low")
                        }
                        logging.debug(f"Added/Updated product: {name}")
        
        # Convert map to list of unique products
        unique_products = list(product_map.values())
        
        logging.info(f"Extracted {len(unique_products)} unique products after deduplication")
        for prod in unique_products:
            logging.info(f"Unique product: {prod['name']}")
        
        return unique_products

    def query_product_details(self, ticker, product_name, top_k=25):
        """Query Pinecone for detailed information about a specific product"""
        logging.info(f"Gathering details for product: {product_name}")
        
        # Create targeted queries for different aspects we need
        detail_queries = [
            f"{product_name} mechanism of action how works",
            f"{product_name} target molecule pathway",
            f"{product_name} indication or product candidates disease condition",
            f"{product_name} preclinical studies/data/development",
            f"{product_name} clinical trial phase results",
            f"{product_name} development milestone upcoming"
        ]
        
        all_chunks = []
        # Add debug logging
        logging.info(f"Searching with queries for {product_name}:")
 
        
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
        """Generate structured product details using Gemini with verification"""
        logging.info(f"Extracting detailed information for {product_name}")
        
        if not chunks:
            logging.warning(f"No verified information found for {product_name}")
            return None
        
        # Get verification confidence from chunks
        confidence = chunks[0].get('product_verification', 'low')
        
        # Organize chunks by query type
        organized_chunks = {
            'mechanism': [],
            'target': [],
            'indication': [],
            'preclinical': [],
            'clinical': [],
            'development': []
        }
        
        for chunk in chunks:
            query_type = chunk.get('query_type', '').lower()
            for category in organized_chunks:
                if category in query_type:
                    organized_chunks[category].append(chunk)
        
        # Create context with organized information
        context_sections = []
        for category, category_chunks in organized_chunks.items():
            if category_chunks:
                context_sections.append(f"\n{category.upper()} INFORMATION:")
                for i, chunk in enumerate(category_chunks[:3], 1):  # Top 3 chunks per category
                    context_sections.append(
                        f"[{chunk['form_type']} {chunk['filing_date']}] {chunk['text']}"
                    )
        
        context = "\n\n".join(context_sections)
        
        # Modified prompt for better verification
        prompt = f"""
        Analyze the following SEC filing excerpts about {product_name} from {ticker}.
        
        First, verify this is a legitimate drug/program and not a misidentified term.
        Then extract detailed information ONLY if verification passes.
        
        Current verification confidence: {confidence}
        
        Context:
        {context}
        
        Return a JSON object with this structure:
        {{
            "verified": true/false,
            "verification_notes": "Why you believe this is or isn't a real drug/program",
            "name": "{product_name}",
            "confidence": "{confidence}",
            "mechanism_of_action": "Detailed description if available",
            "target": "Specific molecular/biological target",
            "indication": "Disease/condition being targeted",
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
                "Only include: Section name, form type (10-K/8-K), and filing date in format: 'Section Name (Form Type, YYYY-MM-DD)'"
            ]
        }}
        
        For references, use this exact format:
        - "Pipeline Overview (10-K, 2024-03-15)"
        - "Clinical Development (8-K, 2024-01-20)"
        Important:
    - Only include references where {product_name} is specifically discussed
    - Add a brief note about what information was found in each reference
    - Sort references by date (newest first)
        
        Only include sections that directly discuss {product_name}.
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

    def validate_product_information(self, product):
        """Validate if product has sufficient drug/program-related information"""
        
        # Critical fields - need at least 2 of these 3
        critical_fields = {
            "name": "Product identifier",
            "mechanism_of_action": "Mechanism of action",
            "indication": "Disease/condition indication"
        }
        
        # Supporting fields - need at least 1 of these
        supporting_fields = {
            "target": "Molecular/pathway target",
            "preclinical_data": "Preclinical studies",
            "clinical_trials": "Clinical trials",
            "upcoming_milestones": "Development milestones",
            "references": "Source documents"
        }
        
        # Count valid critical fields
        critical_count = 0
        missing_critical = []
        for field, desc in critical_fields.items():
            value = product.get(field, '')
            if isinstance(value, str):
                # Check if value exists and isn't just a placeholder
                if value and not any(x in value.lower() for x in ['unknown', 'n/a', 'placeholder', 'requires']):
                    critical_count += 1
                else:
                    missing_critical.append(desc)
        
        # Count valid supporting fields
        supporting_count = 0
        missing_supporting = []
        for field, desc in supporting_fields.items():
            value = product.get(field)
            if value:
                if isinstance(value, list):
                    if len(value) > 0 and not all('unknown' in str(x).lower() for x in value):
                        supporting_count += 1
                        continue
                elif isinstance(value, str) and not any(x in value.lower() for x in ['unknown', 'n/a', 'placeholder', 'requires']):
                    supporting_count += 1
                    continue
            missing_supporting.append(desc)
        
        # Validation logic:
        # 1. Must have at least 2 critical fields
        # 2. Must have at least 1 supporting field
        is_valid = critical_count >= 2 and supporting_count >= 1
        
        # Build validation message
        validation_notes = []
        if critical_count < 2:
            validation_notes.append(f"Missing critical information: needs {2-critical_count} more of {', '.join(missing_critical)}")
        if supporting_count < 1:
            validation_notes.append(f"Missing supporting information: needs at least 1 of {', '.join(missing_supporting)}")
        
        logging.info(f"Validation results for {product.get('name', 'Unknown')}:")
        logging.info(f"Critical fields: {critical_count}/{len(critical_fields)}")
        logging.info(f"Supporting fields: {supporting_count}/{len(supporting_fields)}")
        
        return is_valid, validation_notes

    def analyze_detailed_products(self, ticker, top_k=15, output_json=None, output_table=None, upload_to_s3=False):
        """Complete pipeline with improved validation"""
        logging.info(f"Starting detailed product analysis for {ticker}")
        
        try:
            # Step 1: Initial product list
            chunks = self.retrieve_product_chunks(ticker, top_k)
            results = self.extract_product_names(chunks, ticker)
            product_set = self.extract_product_set(results)
            
            # Step 2: Get detailed information and validate
            valid_products = []
            invalid_products = []
            
            for product in product_set:
                logging.info(f"\nProcessing details for product: {product['name']}")
                
                # Query Pinecone for product details
                product_chunks = self.query_product_details(ticker, product['name'], top_k=30)
                logging.info(f"Found {len(product_chunks)} chunks for {product['name']}")
                
                if not product_chunks:
                    logging.warning(f"No detailed information found for {product['name']}")
                    invalid_products.append({"name": product['name'], "reason": "No supporting information found"})
                    continue
                
                # Extract structured data
                product_details = self.extract_product_details(ticker, product['name'], product_chunks)
                
                # Validate product information
                is_valid, missing_info = self.validate_product_information(product_details)
                
                if is_valid:
                    valid_products.append(product_details)
                    logging.info(f"✓ Validated {product['name']} - Added to results")
                else:
                    invalid_products.append({
                        "name": product['name'],
                        "reason": f"Insufficient information: {', '.join(missing_info)}"
                    })
                    logging.warning(f"✗ Rejected {product['name']} - Missing critical information")
            
            # Log validation results
            logging.info(f"\nValidation Summary for {ticker}:")
            logging.info(f"- Accepted Products: {len(valid_products)}")
            logging.info(f"- Rejected Products: {len(invalid_products)}")
            for invalid in invalid_products:
                logging.info(f"  ✗ {invalid['name']}: {invalid['reason']}")
            
            # Save results if requested
            if output_json:
                results_with_metadata = {
                    "ticker": ticker,
                    "analysis_date": datetime.now().isoformat(),
                    "valid_products": valid_products,
                    "rejected_products": invalid_products,
                    "validation_summary": {
                        "total_products": len(product_set),
                        "valid_products": len(valid_products),
                        "invalid_products": len(invalid_products)
                    }
                }
                
                with open(output_json, 'w') as f:
                    json.dump(results_with_metadata, f, indent=2)
                logging.info(f"Saved detailed results to: {output_json}")
            
            # Generate table if requested
            if output_table and valid_products:
                self.create_product_table(valid_products, output_table)
            
            return {
                "ticker": ticker,
                "products": valid_products,
                "product_count": len(valid_products),
                "validation": {
                    "rejected_count": len(invalid_products),
                    "rejected_products": invalid_products
                }
            }
            
        except Exception as e:
            logging.error(f"Error in detailed analysis: {str(e)}")
            return {
                "ticker": ticker,
                "products": [],
                "product_count": 0,
                "error": str(e)
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
        """Save product details to S3, overwriting existing files"""
        try:
            # Initialize S3Storage
            s3 = S3Storage()
            if not s3.check_connection():
                logging.error("Failed to connect to S3")
                return False
            
            # Create base folder path and fixed filenames (no timestamp)
            base_path = f"output/{ticker}"
            success = True
            
            # 1. Save JSON (overwrite if exists)
            filtered_products = self.filter_products_for_json(products)
            json_key = f"{base_path}/{ticker}_products.json"  # Fixed filename
            
            json_result = s3.save_file(
                json_key, 
                json.dumps(filtered_products, indent=2),
                content_type="application/json"
            )
            if json_result:
                logging.info(f"Saved/Updated JSON ({len(filtered_products)} products) to S3: {json_key}")
            else:
                logging.error(f"Failed to save JSON to S3: {json_key}")
                success = False
            
            # 2. Save Markdown (overwrite if exists)
            try:
                md_content = self.convert_to_markdown(products, ticker)
                md_key = f"{base_path}/{ticker}_products.md"  # Fixed filename
                
                md_result = s3.save_file(
                    md_key,
                    md_content,
                    content_type="text/markdown"
                )
                if md_result:
                    logging.info(f"Saved/Updated Markdown to S3: {md_key}")
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