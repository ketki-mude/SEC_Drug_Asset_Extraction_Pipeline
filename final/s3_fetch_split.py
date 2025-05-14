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
            # Check for both 8-K and 10-K files
            file_patterns = [f"{ticker}/8-K_merged.txt", f"{ticker}/10-K_merged.txt"]
            
            existing_files = []
            for pattern in file_patterns:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=pattern
                )
                
                if 'Contents' in response:
                    existing_files.append(pattern)
            
            if existing_files:
                logging.info(f"Found {len(existing_files)} files for ticker {ticker}: {existing_files}")
                return existing_files
            else:
                logging.warning(f"No files found for ticker {ticker}")
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
            
            # Pattern for date headers
            header_pattern = r"={80,}\s*FILING:\s*([0-9A-Z-]+)\s*-\s*(\d{4}-\d{2}-\d{2})\s*\(Accession:\s*([0-9-]+)\)\s*\n={80,}"
            
            # Find all headers and their positions
            headers = list(re.finditer(header_pattern, content))
            
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
                
                # Only add non-empty sections
                if section_text:
                    sections.append({
                        "date": date,
                        "filing_type": filing_type,
                        "accession": accession,
                        "text": section_text
                    })
            
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
            return []
        
        all_sections = []
        
        for file_key in file_keys:
            # Stream content directly from S3
            content = self.get_file_content(file_key)
            
            if not content:
                continue
            
            # Split content by date headers
            sections = self.split_content_by_date_headers(content, file_key)
            
            # Add file type metadata
            file_type = "8-K" if "8-K" in file_key else "10-K"
            for section in sections:
                section["file_type"] = file_type
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
    
    args = parser.parse_args()
    
    fetcher = S3FileFetcher()
    sections = fetcher.fetch_and_split(args.ticker)
    
    if args.output and sections:
        import json
        with open(args.output, 'w') as f:
            json.dump(sections, f, indent=2)
        logging.info(f"Saved {len(sections)} sections to {args.output}")
    
    print(f"Successfully processed {len(sections)} sections for {args.ticker}")