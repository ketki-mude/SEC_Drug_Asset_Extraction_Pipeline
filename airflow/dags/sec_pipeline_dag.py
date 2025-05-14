import os
import logging
import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable

# Import our processing modules
from aws_storage import S3Storage
from pinecone import Pinecone
import importlib
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 5, 14),
}

# Create the DAG
dag = DAG(
    'ticker_sec_extraction',
    default_args=default_args,
    description='Process SEC filings for a given ticker symbol',
    schedule_interval=None,
    catchup=False,
    tags=['sec', 'biotech', 'gemini'],
)

def validate_ticker_exists(ticker):
    """Make sure the ticker is valid before proceeding"""
    from biotech_validator import is_biotech_company
    
    is_biotech, company_info = is_biotech_company(ticker)
    
    if not company_info:
        logging.error(f"Could not find information for ticker {ticker}")
        return False
    
    logging.info(f"Ticker {ticker} validated: {company_info['name']} (SIC: {company_info['sic']})")
    
    # Store company info for downstream tasks
    Variable.set(f"company_info_{ticker}", 
                 json.dumps(company_info),
                 serialize_json=True)
    
    return True

def check_paths(**context):
    """
    Check if results already exist, or determine processing path based on available data
    This is a branching function that will determine the next step in the pipeline
    """
    # Get the ticker from DAG run configuration
    ticker = context['dag_run'].conf['ticker']
    ticker = ticker.upper()
    logging.info(f"Checking processing path for ticker: {ticker}")
    
    # Initialize S3 storage client
    s3 = S3Storage()
    if not s3.check_connection():
        logging.error("Failed to connect to S3")
        return "notify_error"
    
    # STEP 1: Check if results already exist in S3
    result_path = f"output/{ticker}/{ticker.lower()}_products_"
    
    # List files in the output directory for this ticker
    try:
        response = s3.s3_client.list_objects_v2(
            Bucket=s3.bucket_name,
            Prefix=f"output/{ticker}/"
        )
        
        if 'Contents' in response:
            json_files = [item['Key'] for item in response['Contents'] if item['Key'].endswith('.json')]
            if json_files:
                # Results already exist, no processing needed
                logging.info(f"Results already exist for {ticker}: {json_files}")
                # Store the path for downstream tasks
                Variable.set(f"result_path_{ticker}", json_files[0])
                return "results_ready"
    except Exception as e:
        logging.error(f"Error checking S3 for results: {str(e)}")
    
    # STEP 2: Check if ticker exists in Pinecone
    try:
        # Initialize Pinecone client
        api_key = os.environ.get("PINECONE_API_KEY")
        pinecone = Pinecone(api_key=api_key)
        
        # Get indexes
        indexes = pinecone.list_indexes()
        
        if indexes:
            index_name = os.environ.get("PINECONE_INDEX_NAME", "sec-filings-index")
            if index_name in [idx.name for idx in indexes]:
                index = pinecone.Index(index_name)
                
                # Query with metadata filter for this ticker
                # Use a simple query with minimal vector to just check presence
                dummy_vector = [0.0] * 768  # Assuming 768-dim embeddings
                results = index.query(
                    vector=dummy_vector,
                    filter={"ticker_name": ticker},
                    top_k=1,
                    include_metadata=True
                )
                
                if hasattr(results, 'matches') and len(results.matches) > 0:
                    logging.info(f"Ticker {ticker} found in Pinecone")
                    return "run_gemini"
    except Exception as e:
        logging.error(f"Error checking Pinecone: {str(e)}")
    
    # STEP 3: Check if merged files exist on S3
    try:
        merged_8k_exists = s3.file_exists(f"{ticker}/8-K_merged.txt")
        merged_10k_exists = s3.file_exists(f"{ticker}/10-K_merged.txt")
        
        if merged_8k_exists or merged_10k_exists:
            logging.info(f"Found merged files for {ticker}")
            return "run_chunking"
    except Exception as e:
        logging.error(f"Error checking for merged files: {str(e)}")
    
    # If none of the above, start from scratch
    return "run_sec_extractor"

def run_sec_extractor(**context):
    """Run the SEC filing extractor"""
    ticker = context['dag_run'].conf['ticker']
    ticker = ticker.upper()
    
    logging.info(f"Running SEC filing extractor for {ticker}")
    
    # Import the module
    from sec_filing_extractor import process_ticker_consolidated
    
    # Process the ticker in consolidated mode
    success = process_ticker_consolidated(
        ticker=ticker,
        years_back=2,  # Get filings from past 2 years
        limit_per_form=5  # Limit to 5 filings per form type
    )
    
    if not success:
        raise Exception(f"Failed to extract SEC filings for {ticker}")
    
    return ticker

def run_s3_fetch_split(**context):
    """Fetch and split SEC filings from S3"""
    ticker = context['task_instance'].xcom_pull(task_ids='run_sec_extractor')
    ticker = ticker.upper()
    
    logging.info(f"Running S3 fetch and split for {ticker}")
    
    # Import the module
    from s3_fetch_split import S3FileFetcher
    
    # Initialize fetcher and process
    fetcher = S3FileFetcher()
    sections = fetcher.fetch_and_split(ticker)
    
    if not sections:
        raise Exception(f"No sections found for {ticker}")
    
    # Store sections count for logging
    Variable.set(f"sections_count_{ticker}", str(len(sections)))
    
    logging.info(f"Successfully processed {len(sections)} sections for {ticker}")
    return ticker

def run_chunking(**context):
    """Run semantic chunking and upload to Pinecone"""
    # Try to get ticker from previous task or from DAG run config
    ti = context['task_instance']
    ticker = ti.xcom_pull(task_ids='run_s3_fetch_split')
    if not ticker:
        ticker = context['dag_run'].conf['ticker']
    
    ticker = ticker.upper()
    logging.info(f"Running semantic chunking for {ticker}")
    
    # Import the module
    from chunking_upload_to_pincone import SemanticChunker
    
    # Initialize chunker and process
    chunker = SemanticChunker()
    num_records = chunker.process_ticker(ticker)
    
    if num_records == 0:
        raise Exception(f"No chunks created for {ticker}")
    
    logging.info(f"Successfully processed {num_records} chunks for {ticker}")
    return ticker

def run_gemini(**context):
    """Run Gemini RAG to extract product information"""
    # Try to get ticker from previous task or from DAG run config
    ti = context['task_instance']
    ticker = ti.xcom_pull(task_ids='run_chunking')
    if not ticker:
        ticker = context['dag_run'].conf['ticker']
    
    ticker = ticker.upper()
    logging.info(f"Running Gemini RAG for {ticker}")
    
    # Import the module
    from RAG_GEMINI import ProductNameExtractor
    
    # Initialize extractor and run detailed analysis
    try:
        extractor = ProductNameExtractor()
        filename_base = f"{ticker.lower()}_products_{datetime.now().strftime('%Y%m%d')}"
        
        # Run detailed analysis with S3 upload
        results = extractor.analyze_detailed_products(
            ticker=ticker,
            top_k=15,
            output_json=None,  # Don't save locally
            upload_to_s3=True  # Upload to S3
        )
        
        if "error" in results:
            raise Exception(f"Error in Gemini extraction: {results['error']}")
        
        # Store the expected S3 path for downstream tasks
        result_path = f"output/{ticker}/{filename_base}.json"
        Variable.set(f"result_path_{ticker}", result_path)
        
        logging.info(f"Successfully extracted {results.get('product_count', 0)} products for {ticker}")
        return ticker
    except Exception as e:
        logging.error(f"Error in Gemini extraction: {str(e)}")
        raise

def notify_completion(**context):
    """Notify that the pipeline is complete and results are ready"""
    ticker = context['dag_run'].conf['ticker']
    ticker = ticker.upper()
    
    # Get result path from earlier task
    result_path = Variable.get(f"result_path_{ticker}", default_var=None)
    
    if result_path:
        logging.info(f"Processing complete for {ticker}. Results available at: {result_path}")
        # You could also send a notification via email, Slack, etc.
    else:
        logging.warning(f"Processing complete for {ticker}, but result path not found.")
    
    # Return the result path for the Streamlit app to use
    return result_path

def handle_error(**context):
    """Handle any errors in the pipeline"""
    ticker = context['dag_run'].conf['ticker']
    ticker = ticker.upper()
    
    logging.error(f"Error processing {ticker}")
    # You could send a notification or alert here

# Define the tasks
start = DummyOperator(
    task_id='start',
    dag=dag,
)

validate_ticker = PythonOperator(
    task_id='validate_ticker',
    python_callable=validate_ticker_exists,
    op_kwargs={'ticker': '{{ dag_run.conf["ticker"] }}'},
    dag=dag,
)

check_processing_path = BranchPythonOperator(
    task_id='check_processing_path',
    python_callable=check_paths,
    provide_context=True,
    dag=dag,
)

run_sec_extractor = PythonOperator(
    task_id='run_sec_extractor',
    python_callable=run_sec_extractor,
    provide_context=True,
    dag=dag,
)

run_s3_fetch_split = PythonOperator(
    task_id='run_s3_fetch_split',
    python_callable=run_s3_fetch_split,
    provide_context=True,
    dag=dag,
)

run_chunking = PythonOperator(
    task_id='run_chunking',
    python_callable=run_chunking,
    provide_context=True,
    dag=dag,
)

run_gemini = PythonOperator(
    task_id='run_gemini',
    python_callable=run_gemini,
    provide_context=True,
    dag=dag,
)

results_ready = DummyOperator(
    task_id='results_ready',
    dag=dag,
)

notify_completion = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    provide_context=True,
    dag=dag,
    trigger_rule=TriggerRule.ONE_SUCCESS,  # Run if any upstream task succeeds
)

notify_error = PythonOperator(
    task_id='notify_error',
    python_callable=handle_error,
    provide_context=True,
    dag=dag,
    trigger_rule=TriggerRule.ONE_FAILED,  # Run if any upstream task fails
)

end = DummyOperator(
    task_id='end',
    dag=dag,
    trigger_rule=TriggerRule.ONE_SUCCESS,  # Only need one path to succeed
)

# Define the task dependencies
start >> validate_ticker >> check_processing_path

check_processing_path >> results_ready
check_processing_path >> run_gemini
check_processing_path >> run_chunking >> run_gemini
check_processing_path >> run_sec_extractor >> run_s3_fetch_split >> run_chunking

results_ready >> notify_completion
run_gemini >> notify_completion

notify_completion >> end

# Error handling
[validate_ticker, check_processing_path, run_sec_extractor, run_s3_fetch_split, run_chunking, run_gemini] >> notify_error
notify_error >> end