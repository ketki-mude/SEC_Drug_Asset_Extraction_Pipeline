import requests
import json
import re
import time
import logging
import os
from datetime import datetime
from bs4 import BeautifulSoup
import html2text
from dotenv import load_dotenv
from biotech_validator import is_biotech_company, HEADERS
from aws_storage import S3Storage
from text_processor import SECTextProcessor

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("sec_exhibits_extractor.log"), logging.StreamHandler()]
)

# Configuration
USER_AGENT = "Ketki Mude mude.k@northeastern.edu"  # Replace with your info
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
DEFAULT_YEARS_BACK = 2

def get_sec_filings(ticker, cik, years_back=DEFAULT_YEARS_BACK, limit_per_form=None):
    """
    Get list of SEC filings for a company
    
    Args:
        ticker (str): Ticker symbol
        cik (str): Company CIK
        years_back (int): Number of years to look back
        limit_per_form (int, optional): Maximum filings per form type
    
    Returns:
        list: List of filing metadata dictionaries
    """
    # Calculate cutoff date
    current_year = datetime.now().year
    start_year = current_year - years_back
    cut_off_date = datetime(start_year, 1, 1).date()
    
    logging.info(f"Getting 8-K filings for {ticker} since {cut_off_date}")
    
    try:
        padded_cik = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
        response = requests.get(url, headers=HEADERS, timeout=30)
        
        if response.status_code != 200:
            return []
            
        data = response.json()
        recent = data["filings"]["recent"]
        filings = []
        
        # Process each filing
        for i, (form, date_str, acc_no) in enumerate(zip(
            recent["form"], 
            recent["filingDate"], 
            recent["accessionNumber"]
        )):
            # Only process 8-K filings (exact matches, not amendments)
            if form != "8-K":
                continue
            
            # Check if we've hit the limit for this form type
            if limit_per_form and len(filings) >= limit_per_form:
                break
            
            # Check if filing is within our date range
            filing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if filing_date < cut_off_date:
                continue
            
            # Add to our list
            filings.append({
                "ticker": ticker,
                "cik": padded_cik,
                "form_type": form,
                "filing_date": date_str,
                "accession_number": acc_no
            })
        
        logging.info(f"Found {len(filings)} 8-K filings for {ticker}")
        return filings
        
    except Exception as e:
        logging.error(f"Error getting filings: {str(e)}")
        return []

def extract_exhibits_from_filing(cik, acc_no):
    """
    Extract EX-99.1 and EX-99.2 exhibits from an 8-K filing
    
    Args:
        cik (str): Company CIK
        acc_no (str): Accession number
        
    Returns:
        dict: Dictionary with exhibit content or None if no exhibits found
    """
    try:
        # Format accession number for URL (remove dashes)
        acc_no_clean = acc_no.replace("-", "")
        
        # Download complete submission file
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{acc_no}.txt"
        logging.info(f"Downloading complete submission: {url}")
        
        # Respect SEC rate limit
        time.sleep(0.1)
        
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            logging.error(f"Failed to download complete submission: HTTP {response.status_code}")
            return None
            
        raw_text = response.text
        
        # Extract exhibits 99.1 and 99.2 from the complete submission
        exhibits = {}
        
        # First look for exhibit sections with <DOCUMENT> tags
        logging.info("Searching for EX-99.1 and EX-99.2 in complete submission file")
        
        # Pattern to find document type and text content
        doc_pattern = re.compile(r'<DOCUMENT>.*?<TYPE>(EX-99\.[12]).*?<TEXT>(.*?)</TEXT>.*?</DOCUMENT>', re.DOTALL | re.IGNORECASE)
        
        for match in doc_pattern.finditer(raw_text):
            exhibit_type = match.group(1).upper()  # Normalize to uppercase
            exhibit_content = match.group(2).strip()
            
            if exhibit_content:
                logging.info(f"Found {exhibit_type} in complete submission")
                
                # Process HTML content if needed
                if '<html' in exhibit_content.lower() or '<body' in exhibit_content.lower():
                    try:
                        # Use BeautifulSoup to extract text from HTML
                        soup = BeautifulSoup(exhibit_content, 'html.parser')
                        
                        # Remove script and style elements
                        for element in soup(["script", "style"]):
                            element.extract()
                        
                        # Get text content with preservation of some structure
                        extracted_text = soup.get_text(separator=' ', strip=True)
                        exhibit_content = extracted_text
                    except Exception as e:
                        logging.error(f"Error processing HTML content: {str(e)}")
                
                # Process the content to improve readability
                processed_content = SECTextProcessor.process_text(exhibit_content, "8-K")
                
                # Remove anything that looks like HTML/XML tags
                processed_content = re.sub(r'<[^>]+>', ' ', processed_content)
                
                # Remove problematic special characters like bullet points
                processed_content = processed_content.replace('●', '')
                processed_content = processed_content.replace('•', '')
                processed_content = processed_content.replace('✓', '')
                processed_content = processed_content.replace('★', '')
                processed_content = processed_content.replace('✱', '')
                
                # Normalize whitespace after all these removals
                processed_content = re.sub(r'\s+', ' ', processed_content).strip()
                
                exhibits[exhibit_type] = processed_content
        
        # If no exhibits found with first method, try alternative patterns
        if not exhibits:
            logging.info("No exhibits found with primary method, trying alternative patterns")
            
            # Try looking for HTML sections with exhibit markers
            for exhibit_num in ["1", "2"]:
                # Try different patterns to identify exhibits
                patterns = [
                    # Look for HTML documents with exhibit in filename
                    rf'<DOCUMENT>.*?<FILENAME>[^<]*EX-?99[._-]{exhibit_num}[^<]*\.html?.*?<TEXT>(.*?)</TEXT>.*?</DOCUMENT>',
                    # Look for HTML documents with exhibit in description
                    rf'<DOCUMENT>.*?<DESCRIPTION>[^<]*EXHIBIT\s+99\.{exhibit_num}.*?<TEXT>(.*?)</TEXT>.*?</DOCUMENT>',
                    # Look for exhibit markers in text
                    rf'EXHIBIT\s+99\.{exhibit_num}.*?\n+(.+?)(?=EXHIBIT\s+|\Z)',
                    # Look for Item 7.01 (for press releases often filed as 99.1)
                    r'Item 7\.01[^\n]+\n+(.+?)(?=Item \d|\Z)'
                ]
                
                # Try each pattern
                for pattern in patterns:
                    match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
                    if match:
                        exhibit_type = f"EX-99.{exhibit_num}"
                        exhibit_content = match.group(1).strip()
                        
                        if exhibit_content and len(exhibit_content) > 100:  # Ensure content is substantial
                            logging.info(f"Found {exhibit_type} using alternative pattern")
                            
                            # Process HTML content if needed
                            if '<html' in exhibit_content.lower() or '<body' in exhibit_content.lower():
                                try:
                                    # Use BeautifulSoup to extract text from HTML
                                    soup = BeautifulSoup(exhibit_content, 'html.parser')
                                    
                                    # Remove script and style elements
                                    for element in soup(["script", "style"]):
                                        element.extract()
                                    
                                    # Get text content with preservation of some structure
                                    extracted_text = soup.get_text(separator=' ', strip=True)
                                    exhibit_content = extracted_text
                                except Exception as e:
                                    logging.error(f"Error processing HTML content: {str(e)}")
                            
                            processed_content = SECTextProcessor.process_text(exhibit_content, "8-K")
                            
                            # Remove anything that looks like HTML/XML tags
                            processed_content = re.sub(r'<[^>]+>', ' ', processed_content)
                            
                            # Remove problematic special characters like bullet points
                            processed_content = processed_content.replace('●', '')
                            processed_content = processed_content.replace('•', '')
                            processed_content = processed_content.replace('✓', '')
                            processed_content = processed_content.replace('★', '')
                            processed_content = processed_content.replace('✱', '')
                            processed_content = processed_content.replace('*', '')
                            
                            # Normalize whitespace after all these removals
                            processed_content = re.sub(r'\s+', ' ', processed_content).strip()
                            
                            exhibits[exhibit_type] = processed_content
                            break
        
        return exhibits if exhibits else None
        
    except Exception as e:
        logging.error(f"Error extracting exhibits: {str(e)}")
        return None

def process_ticker_consolidated(ticker, years_back=DEFAULT_YEARS_BACK, limit_per_form=None):
    """
    Process 8-K filings for a ticker and extract EX-99.1 and EX-99.2 exhibits
    with duplicates removed, organized by filing date
    """
    ticker = ticker.upper()
    logging.info(f"Processing 8-K exhibits for: {ticker}")
    
    # Validate ticker is for a biotech company
    is_biotech, company_info = is_biotech_company(ticker)
    
    if not company_info:
        print(f"Error: Could not find information for ticker {ticker}")
        return False
    
    # Set up S3 storage
    s3_storage = S3Storage()
    if not s3_storage.check_connection():
        print("Error connecting to S3. Check your credentials in the .env file.")
        return False
    
    # Check if file already exists in S3
    s3_key = f"{ticker}/8K_Merged.txt"
    file_exists = False
    
    try:
        # Check if file exists directly using head_object
        response = s3_storage.s3_client.list_objects_v2(
            Bucket=s3_storage.bucket_name,
            Prefix=s3_key
        )
        
        file_exists = 'Contents' in response and len(response['Contents']) > 0
        
        if file_exists:
            logging.info(f"8K_Merged.txt already exists for {ticker} in S3. Skipping processing.")
            print(f"8K_Merged.txt already exists for {ticker} in S3. Skipping processing.")
            return True
    except Exception as e:
        logging.error(f"Error checking if file exists in S3: {str(e)}")
    
    # Get list of 8-K filings
    filings = get_sec_filings(ticker, company_info['cik'], years_back, limit_per_form)
    
    if not filings:
        print(f"No 8-K filings found for {ticker} in the last {years_back} years")
        return False
    
    print(f"Found {len(filings)} 8-K filings for {ticker} ({company_info['name']})")
    
    # Initialize tracking for all exhibits
    consolidated_text = ""
    all_statements = set()
    successful_filings = []
    filing_stats = []
    
    # Process chronologically (oldest first)
    filings.sort(key=lambda x: x["filing_date"])
    
    # Group by year
    filings_by_year = {}
    for filing in filings:
        year = filing["filing_date"].split("-")[0]
        if year not in filings_by_year:
            filings_by_year[year] = []
        filings_by_year[year].append(filing)
    
    # Process each filing year by year
    for year, year_filings in sorted(filings_by_year.items()):
        consolidated_text += f"\n\n{'='*80}\n"
        consolidated_text += f" 8-K EXHIBITS FROM {year} \n"
        consolidated_text += f"{'='*80}\n\n"
        
        for i, filing in enumerate(year_filings):
            cik = company_info['cik'].lstrip('0')  # Remove leading zeros
            acc_no = filing["accession_number"]
            filing_date = filing["filing_date"]
            
            print(f"Processing {i+1}/{len(year_filings)}: 8-K from {filing_date}")
            
            # Extract exhibits - this now downloads and extracts from the complete submission directly
            exhibits = extract_exhibits_from_filing(cik, acc_no)
            
            if not exhibits:
                print(f"  - No EX-99.1 or EX-99.2 exhibits found, skipping")
                continue
            
            # Add metadata section for this filing
            filing_section = f"\n\n{'='*80}\n"
            filing_section += f"FILING: 8-K - {filing_date} (Accession: {acc_no})\n"
            filing_section += f"{'='*80}\n\n"
            
            # Process each exhibit
            has_unique_content = False
            
            for exhibit_type, exhibit_content in sorted(exhibits.items()):
                # Extract statements for deduplication
                current_statements = extract_statements(exhibit_content)
                print(f"  - Found {len(current_statements)} statements in {exhibit_type}")
                
                # Find unique statements (not seen before)
                unique_statements = current_statements - all_statements
                print(f"  - {len(unique_statements)} are unique (not in previous filings)")
                
                if unique_statements:
                    filing_section += f"\n== {exhibit_type} ==\n\n"
                    filing_section += "\n".join(unique_statements)
                    filing_section += "\n\n"
                    
                    # Update our master set of statements
                    all_statements.update(unique_statements)
                    has_unique_content = True
            
            # Add this filing's section to consolidated text if it has unique content
            if has_unique_content:
                consolidated_text += filing_section
                successful_filings.append(filing)
                
                # Record statistics
                filing_stats.append({
                    "filing_date": filing_date,
                    "accession_number": acc_no,
                    "exhibits_found": len(exhibits),
                    "unique_statements": sum(len(extract_statements(content) - all_statements) for _, content in exhibits.items()),
                })
            else:
                print(f"  - No unique content found in exhibits, skipping")
            
            # Respect SEC rate limits
            time.sleep(1)
    
    # Generate statistics summary for logging
    if successful_filings:
        summary = f"CONSOLIDATED 8-K EXHIBITS FOR {ticker} ({company_info['name']})\n"
        summary += f"Generated on: {datetime.now().strftime('%Y-%m-%d')}\n"
        summary += f"Timespan: {filings[0]['filing_date']} to {filings[-1]['filing_date']}\n"
        summary += f"Total 8-K filings processed: {len(filings)}\n"
        summary += f"Total 8-K filings with unique exhibits: {len(successful_filings)}\n"
        summary += f"Total unique statements: {len(all_statements)}\n\n"
        
        # Print summary to console
        print("\n" + summary)
        
        # Save consolidated file to S3 only (not locally)
        s3_key = f"{ticker}/8K_Merged.txt"
        
        try:
            s3_storage.s3_client.put_object(
                Bucket=s3_storage.bucket_name,
                Key=s3_key,
                Body=(summary + consolidated_text).encode('utf-8')
            )
            
            print(f"Successfully uploaded consolidated 8-K exhibits to S3: {s3_key}")
            
            # Verify file was uploaded
            verify_response = s3_storage.s3_client.list_objects_v2(
                Bucket=s3_storage.bucket_name,
                Prefix=s3_key
            )
            
            file_verified = 'Contents' in verify_response and len(verify_response['Contents']) > 0
            if file_verified:
                logging.info(f"Verified {s3_key} was successfully uploaded to S3")
            else:
                logging.warning(f"Could not verify {s3_key} was uploaded to S3")
            
            return True
            
        except Exception as e:
            print(f"Failed to upload consolidated 8-K exhibits to S3: {str(e)}")
            logging.error(f"Failed to upload consolidated 8-K exhibits to S3: {str(e)}")
            return False
    else:
        print(f"No successful 8-K filings with EX-99.1 or EX-99.2 exhibits to consolidate")
    
    return False

def extract_statements(text):
    """
    Extract individual statements from text for deduplication
    
    Args:
        text (str): Text to process
        
    Returns:
        set: Set of statements for deduplication
    """
    # Split by sentence boundary
    # This could be improved with NLP libraries like spaCy for more accurate sentence detection
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Clean and filter sentences
    statements = set()
    for sentence in raw_sentences:
        # Basic cleaning
        clean = sentence.strip()
        
        # Filter too short statements
        if len(clean) < 15:  # Skip very short statements
            continue
            
        # Filter likely boilerplate
        if re.search(r'^\s*\(?[ixvcd]+\)?[\s.)]', clean, re.IGNORECASE):  # Skip list markers
            continue
            
        if any(boilerplate in clean.lower() for boilerplate in [
            "javascript required", "click here", "see note", "see item", 
            "pursuant to", "in accordance with", "for further information",
            "for more information", "for additional information",
            "exhibit", "edgar filing"
        ]):
            continue
        
        # Normalize to improve deduplication
        clean = re.sub(r'\s+', ' ', clean)  # Normalize whitespace
        clean = clean.replace(''', "'").replace(''', "'")  # Normalize apostrophes
        clean = clean.replace('"', '"').replace('"', '"')  # Normalize quotes
        
        # Add to set of statements
        statements.add(clean)
    
    return statements

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract EX-99.1 and EX-99.2 exhibits from 8-K filings")
    parser.add_argument("--ticker", help="Ticker symbol")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS_BACK,
                        help="Number of years to look back")
    parser.add_argument("--limit", type=int, help="Maximum 8-K filings to process")
    parser.add_argument("--force", action="store_true", 
                        help="Force processing even if file already exists in S3")
    
    args = parser.parse_args()
    
    # Get ticker either from args or prompt
    ticker = args.ticker if args.ticker else input("Enter ticker symbol: ").strip().upper()
    
    # If force flag is set, need to implement force refresh logic
    if args.force:
        # In this case, we would skip the S3 existence check
        print(f"Force processing enabled for {ticker}")
        # Implementation would need to be added to process_ticker_consolidated
    
    success = process_ticker_consolidated(
        ticker,
        years_back=args.years,
        limit_per_form=args.limit
    )
    
    if success:
        print(f"\nSuccessfully processed 8-K exhibits for {ticker}")
    else:
        print(f"\nFailed to process 8-K exhibits for {ticker}")