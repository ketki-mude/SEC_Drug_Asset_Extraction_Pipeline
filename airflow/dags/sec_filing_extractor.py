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
    handlers=[logging.FileHandler("sec_extractor.log"), logging.StreamHandler()]
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
    
    logging.info(f"Getting filings for {ticker} since {cut_off_date}")
    
    try:
        padded_cik = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
        response = requests.get(url, headers=HEADERS, timeout=30)
        
        if response.status_code != 200:
            return []
            
        data = response.json()
        recent = data["filings"]["recent"]
        form_counts = {}
        filings = []
        
        # Process each filing
        for i, (form, date_str, acc_no) in enumerate(zip(
            recent["form"], 
            recent["filingDate"], 
            recent["accessionNumber"]
        )):
            # Only process 10-K and 8-K filings (exact matches, not amendments)
            if form not in ["10-K", "8-K"]:
                continue
            
            # Check if we've hit the limit for this form type
            if limit_per_form:
                form_counts[form] = form_counts.get(form, 0) + 1
                if form_counts[form] > limit_per_form:
                    continue
            
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
        
        logging.info(f"Found {len(filings)} filings for {ticker}")
        return filings
        
    except Exception as e:
        logging.error(f"Error getting filings: {str(e)}")
        return []

def download_and_process_filing(cik, acc_no, form_type):
    """
    Download and process a filing from SEC EDGAR
    
    Args:
        cik (str): Company CIK
        acc_no (str): Accession number
        form_type (str): Form type (10-K, 8-K, etc.)
    
    Returns:
        str: Processed filing content as text or None if failed
    """
    try:
        # Format accession number for URL (remove dashes)
        acc_no_clean = acc_no.replace("-", "")
        
        # Construct URL
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{acc_no}.txt"
        logging.info(f"Downloading filing: {url}")
        
        # Respect SEC rate limit
        time.sleep(0.1)
        
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            logging.error(f"Failed to download filing: HTTP {response.status_code}")
            return None
            
        raw_text = response.text
        
        # Extract HTML portion
        html_pattern = re.compile(r'<(?:HTML|html)>.*?</(?:HTML|html)>', re.DOTALL)
        match = html_pattern.search(raw_text)
        
        if match:
            html_content = match.group(0)
        else:
            # Try alternate approach for finding HTML
            doc_pattern = re.compile(r'<DOCUMENT>.*?<TYPE>(?:HTML|html).*?</DOCUMENT>', re.DOTALL)
            match = doc_pattern.search(raw_text)
            
            if match:
                html_match = html_pattern.search(match.group(0))
                html_content = html_match.group(0) if html_match else raw_text
            else:
                html_content = raw_text
        
        # Clean the HTML content first
        html_content = SECTextProcessor.clean_html_content(html_content)
        
        # Convert HTML to text
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.extract()
            
            # Use html2text for better formatting
            converter = html2text.HTML2Text()
            converter.ignore_links = False
            converter.ignore_images = True
            converter.body_width = 0  # No wrapping
            
            text = converter.handle(str(soup))
        except Exception as e:
            logging.error(f"Error converting HTML: {str(e)}")
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
        
        # Further process the text to improve readability - passing form_type for special handling
        processed_text = SECTextProcessor.process_text(text, form_type)
        
        # Return the full text (with financial data)
        return processed_text
        
    except Exception as e:
        logging.error(f"Error processing filing: {str(e)}")
        return None

def save_filing_locally(ticker, form_type, acc_no, filing_date, content, filtered_content=None, base_dir="sec_filings_local"):
    """
    Save filing content to local storage
    
    Args:
        ticker (str): Ticker symbol
        form_type (str): Form type (10-K, 8-K, etc.)
        acc_no (str): Accession number
        filing_date (str): Filing date
        content (str): Original processed content
        filtered_content (str): Filtered content (biomedical only)
        base_dir (str): Base directory for local storage
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory structure
        ticker_dir = os.path.join(base_dir, ticker.upper())
        form_dir = os.path.join(ticker_dir, form_type)
        os.makedirs(form_dir, exist_ok=True)
        
        # Base filename
        base_filename = f"{filing_date}_{acc_no}"
        
        # Save original version
        original_path = os.path.join(form_dir, f"{base_filename}_original.txt")
        with open(original_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Save filtered version if provided
        if filtered_content:
            filtered_path = os.path.join(form_dir, f"{base_filename}_biomedical.txt")
            with open(filtered_path, 'w', encoding='utf-8') as f:
                f.write(filtered_content)
        
        return True
    except Exception as e:
        logging.error(f"Error saving filing locally: {str(e)}")
        return False

def process_ticker(ticker, years_back=DEFAULT_YEARS_BACK, limit_per_form=None, force_refresh=False):
    """
    Process all SEC filings for a ticker
    
    Args:
        ticker (str): Ticker symbol
        years_back (int): Number of years to look back
        limit_per_form (int, optional): Maximum filings per form type
        force_refresh (bool): Force refresh existing filings
    
    Returns:
        bool: True if successful, False otherwise
    """
    ticker = ticker.upper()
    logging.info(f"Processing ticker: {ticker}")
    
    # Validate ticker is for a biotech company
    is_biotech, company_info = is_biotech_company(ticker)
    
    if not company_info:
        print(f"Error: Could not find information for ticker {ticker}")
        return False
    
    if not is_biotech:
        print(f"Warning: {ticker} ({company_info['name']}) is NOT a biotech company (SIC: {company_info['sic']})")
        proceed = input(f"Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            return False
    
    # Set up S3 storage
    s3_storage = S3Storage()
    if not s3_storage.check_connection():
        print("Error connecting to S3. Check your credentials in the .env file.")
        return False
    
    # Get list of filings
    filings = get_sec_filings(ticker, company_info['cik'], years_back, limit_per_form)
    
    if not filings:
        print(f"No filings found for {ticker} in the last {years_back} years")
        return False
    
    print(f"Found {len(filings)} filings for {ticker} ({company_info['name']})")
    print(f"Storing in S3 bucket: {s3_storage.bucket_name}")
    
    # Process each filing
    successful = []
    errors = 0
    
    for i, filing in enumerate(filings):
        acc_no = filing["accession_number"]
        form_type = filing["form_type"]
        filing_date = filing["filing_date"]
        
        print(f"Processing {i+1}/{len(filings)}: {form_type} from {filing_date}")
        
        # Check if already exists in S3
        if not force_refresh and s3_storage.filing_exists(ticker, form_type, acc_no, filing_date, as_txt=True):
            print(f"  - Already exists in S3, skipping")
            successful.append(filing)
            continue
        
        # Download and process the filing - passing form_type for special handling
        processed_text = download_and_process_filing(company_info['cik'].lstrip('0'), acc_no, form_type)
        if not processed_text:
            print(f"  - Failed to process")
            continue
        
        # Save to S3 as TXT file
        if s3_storage.save_text_filing(ticker, form_type, acc_no, filing_date, processed_text):
            print(f"  - Saved to S3: {filing_date}_{acc_no}.txt")
            successful.append(filing)
        else:
            print(f"  - Failed to save to S3")
            errors += 1
            if errors >= 3:
                print("Too many S3 errors. Stopping processing.")
                break
        
        # Respect SEC rate limits
        time.sleep(1)
    
    # Save metadata as JSON
    if successful:
        s3_storage.save_metadata(ticker, successful)
    
    # Print summary
    print(f"\nSummary for {ticker}:")
    print(f"  - Total filings found: {len(filings)}")
    print(f"  - Successfully processed: {len(successful)}")
    
    # Count by form type
    form_counts = {}
    for filing in successful:
        form = filing["form_type"]
        if form not in form_counts:
            form_counts[form] = 0
        form_counts[form] += 1
    
    for form, count in form_counts.items():
        print(f"  - {form}: {count} filings")
        
    return len(successful) > 0

def process_ticker_consolidated(ticker, years_back=DEFAULT_YEARS_BACK, limit_per_form=None):
    """
    Process all SEC filings for a ticker and create consolidated documents
    with duplicates removed, separated by form type
    """
    ticker = ticker.upper()
    logging.info(f"Processing ticker for consolidated output: {ticker}")
    
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
    
    # Get list of filings
    filings = get_sec_filings(ticker, company_info['cik'], years_back, limit_per_form)
    
    if not filings:
        print(f"No filings found for {ticker} in the last {years_back} years")
        return False
    
    print(f"Found {len(filings)} filings for {ticker} ({company_info['name']})")
    
    # Group filings by form type
    form_types = {}
    for filing in filings:
        form_type = filing["form_type"]
        if form_type not in form_types:
            form_types[form_type] = []
        form_types[form_type].append(filing)
    
    print(f"Creating consolidated documents by form type: {', '.join(form_types.keys())}")
    
    # Process each form type separately
    success_count = 0
    
    for form_type, form_filings in form_types.items():
        print(f"\nProcessing {len(form_filings)} {form_type} filings...")
        
        # Initialize tracking for this form type
        consolidated_text = ""
        all_statements = set()
        successful_filings = []
        filing_stats = []
        
        # Process chronologically (oldest first)
        form_filings.sort(key=lambda x: x["filing_date"])
        
        for i, filing in enumerate(form_filings):
            acc_no = filing["accession_number"]
            filing_date = filing["filing_date"]
            
            print(f"Processing {i+1}/{len(form_filings)}: {form_type} from {filing_date}")
            
            # Download and process the filing
            processed_text = download_and_process_filing(company_info['cik'].lstrip('0'), acc_no, form_type)
            if not processed_text:
                print(f"  - Failed to process")
                continue
            
            # Extract statements for deduplication
            current_statements = extract_statements(processed_text)
            print(f"  - Found {len(current_statements)} statements")
            
            # Find unique statements (not seen before)
            unique_statements = current_statements - all_statements
            print(f"  - {len(unique_statements)} are unique (not in previous filings)")
            
            # Add metadata section for this filing
            filing_section = f"\n\n{'='*80}\n"
            filing_section += f"FILING: {form_type} - {filing_date} (Accession: {acc_no})\n"
            filing_section += f"{'='*80}\n\n"
            
            # Add unique content
            if unique_statements:
                filing_section += "\n".join(unique_statements)
                # Update our master set of statements
                all_statements.update(unique_statements)
            else:
                filing_section += f"[No unique content found in this filing]\n"
            
            # Add this filing's section to consolidated text
            if not consolidated_text:
                consolidated_text = filing_section
            else:
                consolidated_text = consolidated_text + "\n\n" + filing_section
            
            # Track successful processing
            successful_filings.append(filing)
            
            # Record statistics
            filing_stats.append({
                "filing_date": filing_date,
                "accession_number": acc_no,
                "total_statements": len(current_statements),
                "unique_statements": len(unique_statements),
                "duplicate_ratio": 1 - (len(unique_statements) / len(current_statements)) if current_statements else 0
            })
            
            # Respect SEC rate limits
            time.sleep(1)
        
        # Generate statistics summary for logging
        if successful_filings:
            summary = f"CONSOLIDATED {form_type} FILINGS FOR {ticker} ({company_info['name']})\n"
            summary += f"Generated on: {datetime.now().strftime('%Y-%m-%d')}\n"
            summary += f"Timespan: {form_filings[0]['filing_date']} to {form_filings[-1]['filing_date']}\n"
            summary += f"Total {form_type} filings processed: {len(successful_filings)}\n"
            summary += f"Total unique statements: {len(all_statements)}\n\n"
            
            summary += "FILING STATISTICS:\n"
            for stat in filing_stats:
                summary += f"- {form_type} ({stat['filing_date']}): "
                summary += f"{stat['unique_statements']} unique / {stat['total_statements']} total statements "
                summary += f"({stat['duplicate_ratio']*100:.1f}% duplicate content)\n"
            
            # Print summary to console
            print("\n" + summary)
            
            # Save consolidated file with the requested path format
            filename = f"{form_type}_merged.txt"
            
            # Save directly to ticker folder without form_type subfolder
            s3_key = f"{ticker}/{filename}"
            
            try:
                s3_storage.s3_client.put_object(
                    Bucket=s3_storage.bucket_name,
                    Key=s3_key,
                    Body=consolidated_text.encode('utf-8')
                )
                
                print(f"Successfully uploaded consolidated {form_type} filing to S3: {s3_key}")
                success_count += 1
                
            except Exception as e:
                print(f"Failed to upload consolidated {form_type} filing: {str(e)}")
        else:
            print(f"No successful {form_type} filings to consolidate")
    
    return success_count > 0

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
    
    parser = argparse.ArgumentParser(description="Extract SEC filings for biotech companies")
    parser.add_argument("--ticker", help="Ticker symbol")
    parser.add_argument("--mode", choices=["standard", "consolidated"], default="standard",
                        help="Processing mode: standard=individual files, consolidated=deduplicated version")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS_BACK,
                        help="Number of years to look back")
    parser.add_argument("--limit", type=int, help="Maximum filings per form type")
    parser.add_argument("--force", action="store_true", 
                        help="Force refresh existing filings")
    
    args = parser.parse_args()
    
    # Get ticker either from args or prompt
    ticker = args.ticker if args.ticker else input("Enter ticker symbol: ").strip().upper()
    
    if args.mode == "consolidated":
        success = process_ticker_consolidated(
            ticker,
            years_back=args.years,
            limit_per_form=args.limit
        )
    else:
        success = process_ticker(
            ticker,
            years_back=args.years,
            limit_per_form=args.limit,
            force_refresh=args.force
        )
    
    if success:
        print(f"\nSuccessfully processed SEC filings for {ticker}")
    else:
        print(f"\nFailed to process SEC filings for {ticker}")