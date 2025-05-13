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
DEFAULT_YEARS_BACK = 3

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
        
        # Just return the processed text
        return processed_text
        
    except Exception as e:
        logging.error(f"Error processing filing: {str(e)}")
        return None

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

if __name__ == "__main__":
    # Get ticker from user input
    ticker = input("Enter ticker symbol: ").strip().upper()
    
    # Process the ticker with default settings
    success = process_ticker(
        ticker,
        years_back=DEFAULT_YEARS_BACK,
        limit_per_form=None,
        force_refresh=False
    )
    
    if success:
        print(f"\nSuccessfully processed SEC filings for {ticker}")
    else:
        print(f"\nFailed to process SEC filings for {ticker}")