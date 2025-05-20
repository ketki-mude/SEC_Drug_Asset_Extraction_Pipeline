import requests
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sec_extractor.log"),
        logging.StreamHandler()
    ]
)

# Configuration
USER_AGENT = "Ketki Mude mude.k@northeastern.edu"  # Replace with your info
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate"
}

# Biotechnology SIC codes
SIC_BIOTECH = [2834, 2836]  # Pharmaceutical Preparations, Biological Products

def get_company_info(ticker):
    """
    Get company information including SIC code for a given ticker.
    
    Args:
        ticker (str): Company ticker symbol
        
    Returns:
        dict: Company information or None if not found
    """
    try:
        # Try to use ticker to get company info
        url = f"https://data.sec.gov/submissions/CIK{ticker}.json"
        
        response = requests.get(url, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract SIC code
            sic = data.get('sic')
            
            # Extract company name
            name = data.get('name', 'Unknown')
            
            # Extract CIK
            cik = str(data.get('cik', ''))
            
            if sic and sic.isdigit():
                return {
                    'ticker': ticker,
                    'cik': cik,
                    'name': name,
                    'sic': int(sic)
                }
            else:
                logging.warning(f"No SIC code found for {ticker}")
                return None
        else:
            # Try alternative approach
            logging.warning(f"Failed to get info for ticker {ticker} directly. Status code: {response.status_code}")
            return get_company_from_tickers_list(ticker)
    
    except Exception as e:
        logging.error(f"Error getting company info for {ticker}: {str(e)}")
        return None

def get_company_from_tickers_list(ticker):
    """
    Alternative method to get company info by checking the SEC's company tickers list.
    
    Args:
        ticker (str): Company ticker symbol
        
    Returns:
        dict: Company information or None if not found
    """
    try:
        # Use the SEC's company tickers list
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers=HEADERS, timeout=30)
        companies_data = response.json()
        
        # Find the company by ticker
        for _, company in companies_data.items():
            if company.get('ticker') == ticker:
                cik_str = str(company.get('cik_str'))
                
                # Now get the full company info using the CIK
                time.sleep(0.1)  # Respect SEC rate limit
                return get_company_by_cik(cik_str, ticker)
        
        logging.error(f"Company ticker {ticker} not found in SEC company tickers list")
        return None
                
    except Exception as e:
        logging.error(f"Error searching company tickers list for {ticker}: {str(e)}")
        return None

def get_company_by_cik(cik, ticker):
    """
    Get company information using its CIK.
    
    Args:
        cik (str): Company CIK
        ticker (str): Company ticker symbol
        
    Returns:
        dict: Company information or None if not found
    """
    try:
        # Pad CIK to 10 digits
        padded_cik = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
        
        response = requests.get(url, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract SIC code
            sic = data.get('sic')
            
            # Extract company name
            name = data.get('name', 'Unknown')
            
            if sic and sic.isdigit():
                return {
                    'ticker': ticker,
                    'cik': cik,
                    'name': name,
                    'sic': int(sic)
                }
        
        logging.error(f"Failed to get company info for CIK {cik}")
        return None
        
    except Exception as e:
        logging.error(f"Error getting company info for CIK {cik}: {str(e)}")
        return None

def is_biotech_company(ticker):
    """
    Check if a company is a biotech company by validating its SIC code.
    
    Args:
        ticker (str): Company ticker symbol
        
    Returns:
        tuple: (is_biotech (bool), company_info (dict) or None)
    """
    company_info = get_company_info(ticker)
    
    if not company_info:
        logging.error(f"Could not find information for ticker {ticker}")
        return False, None
    
    if company_info['sic'] in SIC_BIOTECH:
        logging.info(f"{ticker} ({company_info['name']}) is a biotech company with SIC {company_info['sic']}")
        return True, company_info
    else:
        logging.info(f"{ticker} ({company_info['name']}) is NOT a biotech company. SIC: {company_info['sic']}")
        return False, company_info

# For direct testing
if __name__ == "__main__":
    test_ticker = "ABBV"
    is_biotech, info = is_biotech_company(test_ticker)
    
    if info:
        print(f"Company: {info['name']} (Ticker: {info['ticker']}, CIK: {info['cik']})")
        print(f"SIC Code: {info['sic']}")
        print(f"Is Biotech: {'Yes' if is_biotech else 'No'}")
    else:
        print(f"Could not find information for ticker {test_ticker}")