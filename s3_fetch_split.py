import boto3
import os
import re
import logging
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class S3FileFetcher:
    """Class to fetch SEC filing files from S3 and split by date headers"""
    
    def __init__(self):
        """Initialize S3 client with credentials from environment variables"""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        self.bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        logging.info(f"Initialized S3 client for bucket: {self.bucket_name}")
    
    def check_ticker_files(self, ticker):
        """Check if files exist for the given ticker"""
        ticker = ticker.upper()
        try:
            # Check for 8-K merged file patterns only
            file_patterns = [
                f"{ticker}/8-K_merged.txt",   # Original expected pattern
                f"{ticker}/8K_Merged.txt",    # Pattern we're using in sec_exhibits_extractor.py
                f"{ticker}/8K_merged.txt",    # Possible variation
                f"{ticker}/8-K_Merged.txt"    # Possible variation
            ]
            
            existing_files = []
            for pattern in file_patterns:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=pattern
                )
                
                if 'Contents' in response and len(response['Contents']) > 0:
                    logging.info(f"Found file with pattern: {pattern}")
                    existing_files.append(pattern)
            
            # If nothing found with exact patterns, try a more general search for 8K only
            if not existing_files:
                logging.info(f"No exact file matches found, trying general ticker folder search")
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=f"{ticker}/"
                )
                
                if 'Contents' in response:
                    for item in response['Contents']:
                        key = item['Key']
                        # Look for 8K merged files only
                        if ('8k' in key.lower() or '8-k' in key.lower()) and ('merged' in key.lower() or 'consolidated' in key.lower()):
                            logging.info(f"Found 8K file: {key}")
                            existing_files.append(key)
        
            if existing_files:
                logging.info(f"Found {len(existing_files)} 8-K files for ticker {ticker}: {existing_files}")
                return existing_files
            else:
                logging.warning(f"No merged 8-K files found for ticker {ticker}")
                return []
                
        except Exception as e:
            logging.error(f"Error checking for ticker files: {str(e)}")
            return []
        
    def get_file_content(self, file_key):
        """Get file content directly from S3 without downloading to disk"""
        try:
            logging.info(f"Streaming content for {file_key} from S3")
            
            # Get the object from S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            
            # Read the content
            content = response['Body'].read().decode('utf-8')
            
            logging.info(f"Successfully read {len(content)} bytes from {file_key}")
            return content
        except Exception as e:
            logging.error(f"Error reading file {file_key}: {str(e)}")
            return None
    
    def split_content_by_date_headers(self, content, file_key):
        """Split content by date headers"""
        try:
            logging.info(f"Splitting content from {file_key} by date headers")
            
            # Pattern for date headers - now more flexible to match both formats
            header_pattern = r"={80,}\s*FILING:\s*([0-9A-Z-]+)\s*-\s*(\d{4}-\d{2}-\d{2})\s*\(Accession:\s*([0-9-]+)\)\s*\n={80,}"
            
            # Find all headers and their positions
            headers = list(re.finditer(header_pattern, content))
            
            # If no headers found using standard pattern, try an alternative
            if not headers:
                alt_pattern = r"={10,}\s*FILING:?\s*([0-9A-Z-]+)\s*-\s*(\d{4}-\d{2}-\d{2})\s*\(?(?:Accession:)?\s*([0-9-]+)\)?\s*\n={10,}"
                headers = list(re.finditer(alt_pattern, content))
            
            # If still no headers, create a single section with the entire content
            if not headers:
                logging.warning(f"No date headers found in {file_key}, treating entire file as one section")
                form_type = "8-K" if "8-K" in file_key or "8K" in file_key else "10-K"
                return [{
                    "date": "unknown",
                    "filing_type": form_type,
                    "accession": "unknown",
                    "text": content
                }]
            
            sections = []
            
            # Process each section
            for i, match in enumerate(headers):
                # Extract filing type, date and accession number
                filing_type = match.group(1).strip()
                date = match.group(2).strip()
                accession = match.group(3).strip()
                
                # Find section start (after header)
                start_pos = match.end()
                
                # Find section end (next header or end of file)
                if i < len(headers) - 1:
                    end_pos = headers[i+1].start()
                else:
                    end_pos = len(content)
                
                # Extract section text
                section_text = content[start_pos:end_pos].strip()
                
                # Check for exhibit markers
                exhibit_section = None
                
                # Look for exhibit markers like "== EX-99.1 =="
                exhibit_matches = re.finditer(r'==\s*(EX-99\.[12])\s*==', section_text)
                for ex_match in exhibit_matches:
                    exhibit_type = ex_match.group(1)
                    if not exhibit_section:
                        exhibit_section = exhibit_type
                
                # Only add non-empty sections
                if section_text:
                    section_data = {
                        "date": date,
                        "filing_type": filing_type,
                        "accession": accession,
                        "text": section_text
                    }
                    
                    # Add exhibit information if found
                    if exhibit_section:
                        section_data["exhibit_type"] = exhibit_section
                        
                    sections.append(section_data)
            
            logging.info(f"Split content from {file_key} into {len(sections)} date-based sections")
            return sections
        
        except Exception as e:
            logging.error(f"Error splitting content from {file_key}: {str(e)}")
            return []

    def fetch_and_split(self, ticker):
        """Main method to fetch and split files for a ticker without local downloads"""
        ticker = ticker.upper()
        
        # Check if files exist
        file_keys = self.check_ticker_files(ticker)
        
        if not file_keys:
            logging.warning(f"No files found for ticker {ticker}")
            # Make one more attempt with a broader search for 8K only
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=ticker
                )
                
                if 'Contents' in response:
                    # Look for 8K text files only
                    text_files = [item['Key'] for item in response['Contents'] 
                                 if item['Key'].endswith('.txt') and 
                                 ('8k' in item['Key'].lower() or '8-k' in item['Key'].lower())]
                    
                    if text_files:
                        logging.info(f"Found {len(text_files)} 8K text files: {text_files}")
                        file_keys = text_files
            except Exception as e:
                logging.error(f"Error in broader file search: {str(e)}")
        
        if not file_keys:
            return []
    
        all_sections = []
        
        for file_key in file_keys:
            content = self.get_file_content(file_key)
            if not content:
                continue
            
            sections = self.split_content_by_date_headers(content, file_key)
            
            # Add metadata (now only 8-K)
            for section in sections:
                section["file_type"] = "8-K"
                section["ticker"] = ticker
            
            all_sections.extend(sections)
        
        logging.info(f"Processed {len(all_sections)} total sections for {ticker}")
        return all_sections

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch and split SEC filings from S3")
    parser.add_argument("--ticker", required=True, help="Company ticker symbol")
    parser.add_argument("--output", help="Output JSON file for saving the split sections")
    parser.add_argument("--debug", action="store_true", help="Print extra debugging info")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    fetcher = S3FileFetcher()
    sections = fetcher.fetch_and_split(args.ticker)
    
    if args.output and sections:
        import json
        with open(args.output, 'w') as f:
            json.dump(sections, f, indent=2)
        logging.info(f"Saved {len(sections)} sections to {args.output}")
    
    print(f"Successfully processed {len(sections)} sections for {args.ticker}")