import boto3
import json
import logging
import os
from datetime import datetime
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class S3Storage:
    """Class for handling S3 storage operations"""
    
    def __init__(self):
        """Initialize S3 client and set bucket name from environment"""
        # Load credentials from environment variables
        self.bucket_name = os.environ.get("AWS_S3_BUCKET_NAME", "secbiotechdata")
        self.region = os.environ.get("AWS_REGION", "us-east-1")
        self.access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        self.secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        self.s3_client = None
        self.setup_client()
    
    def setup_client(self):
        """Set up the S3 client with credentials from environment"""
        try:
            # Initialize with specific credentials if provided
            if self.access_key and self.secret_key:
                self.s3_client = boto3.client(
                    's3',
                    region_name=self.region,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key
                )
            else:
                # Fall back to default credentials (from ~/.aws/credentials or EC2 role)
                self.s3_client = boto3.client('s3', region_name=self.region)
                
            logging.info(f"Initialized S3 client for bucket '{self.bucket_name}'")
            return True
        except Exception as e:
            logging.error(f"Error setting up S3 client: {str(e)}")
            return False
    
    def check_connection(self):
        """Test S3 connection and bucket access"""
        if not self.s3_client:
            return False
            
        try:
            # Test if bucket exists and is accessible
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            # Test write permission with a small file
            test_key = f"test_access_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=test_key,
                Body="S3 access test"
            )
            
            # Clean up test file
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=test_key)
            
            logging.info(f"Successfully verified access to S3 bucket '{self.bucket_name}'")
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchBucket':
                logging.error(f"Bucket '{self.bucket_name}' does not exist")
            elif error_code in ['AccessDenied', 'Forbidden']:
                logging.error(f"Access denied to bucket '{self.bucket_name}'")
            else:
                logging.error(f"Error accessing bucket '{self.bucket_name}': {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Error checking S3 connection: {str(e)}")
            return False
    
    def get_file(self, key):
        """Get file content from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            return response['Body'].read().decode('utf-8')
        except Exception as e:
            logging.error(f"Error getting file from S3: {str(e)}")
            return None
    
    def file_exists(self, key):
        """Check if a file exists in S3"""
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=key
            )
            return True
        except:
            return False
    
    def save_file(self, key, data, content_type='text/plain'):
        """Save data to S3 bucket"""
        if not self.s3_client:
            return False
            
        try:
            if isinstance(data, dict) or isinstance(data, list):
                data = json.dumps(data, indent=2, ensure_ascii=False)
                
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                ContentType=content_type
            )
            
            logging.info(f"Saved to S3: {key}")
            return True
        except Exception as e:
            logging.error(f"Error saving to S3: {str(e)}")
            return False
    
    def filing_exists(self, ticker, form_type, acc_no, filing_date, as_txt=True):
        """Check if a filing exists in S3"""
        extension = ".txt" if as_txt else ".json"
        key = f"{ticker}/{form_type}/{filing_date}_{acc_no}{extension}"
        return self.file_exists(key)
    
    def save_text_filing(self, ticker, form_type, acc_no, filing_date, text_content):
        """Save a text filing to S3"""
        key = f"{ticker}/{form_type}/{filing_date}_{acc_no}.txt"
        return self.save_file(key, text_content, "text/plain")
    
    def save_filing(self, ticker, form_type, acc_no, filing_date, data, as_txt=False):
        """Save a filing to S3 with date in the filename"""
        if as_txt:
            return self.save_text_filing(ticker, form_type, acc_no, filing_date, data)
        else:
            key = f"{ticker}/{form_type}/{filing_date}_{acc_no}.json"
            return self.save_file(key, data, "application/json")
    
    def save_metadata(self, ticker, metadata):
        """Save metadata about processed filings"""
        if not metadata:
            return False
            
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        key = f"{ticker}/metadata-{timestamp}.json"
        return self.save_file(key, metadata, "application/json")
    
    def save_consolidated_filing(self, ticker, filename, content):
        """
        Save consolidated SEC filing to S3
        
        Args:
            ticker (str): Ticker symbol
            filename (str): Filename for the consolidated document
            content (str): Text content of consolidated filing
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create S3 key
            s3_key = f"{ticker.upper()}/consolidated/{filename}"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content.encode('utf-8')
            )
            
            logging.info(f"Saved consolidated filing to S3: {s3_key}")
            return True
        except Exception as e:
            logging.error(f"Error saving consolidated filing: {str(e)}")
            return False