import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from biotech_validator import is_biotech_company

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="SEC Filing Drug Asset Analyzer", layout="wide")
st.title("SEC Filing Drug Asset Analyzer")
st.write("Welcome! Use this tool to extract drug candidates and therapeutic programs from SEC filings for biotechnology companies.")

# Sidebar instructions
st.sidebar.header("How to Use")
st.sidebar.info(
    """Enter a ticker symbol in the form below. For example:
- **WVE** (Wave Life Sciences)
- **DRNA** (Dicerna Pharmaceuticals)
- **ALNY** (Alnylam Pharmaceuticals)
"""
)

# Set up session state for tracking job status
if 'job_status' not in st.session_state:
    st.session_state.job_status = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Function to get results from S3
def get_results_from_s3(ticker):
    from aws_storage import S3Storage
    
    s3 = S3Storage()
    if not s3.check_connection():
        st.error("Failed to connect to S3")
        return None
    
    # Try to find the most recent result file
    try:
        response = s3.s3_client.list_objects_v2(
            Bucket=s3.bucket_name,
            Prefix=f"output/{ticker}/"
        )
        
        if 'Contents' in response:
            json_files = [item['Key'] for item in response['Contents'] if item['Key'].endswith('.json')]
            if json_files:
                # Sort by last modified date, newest first
                json_files.sort(key=lambda x: response['Contents'][next(i for i, item in enumerate(response['Contents']) if item['Key'] == x)]['LastModified'], reverse=True)
                
                # Get the newest file
                latest_file = json_files[0]
                
                # Get file content
                obj = s3.s3_client.get_object(Bucket=s3.bucket_name, Key=latest_file)
                data = json.loads(obj['Body'].read().decode('utf-8'))
                return data
    except Exception as e:
        st.error(f"Error retrieving results: {str(e)}")
    
    return None

# Form for triggering analysis
with st.form("ticker_form"):
    ticker = st.text_input("Enter Ticker Symbol:", placeholder="E.g., WVE, DRNA, ALNY").strip().upper()
    submitted = st.form_submit_button("Start Analysis")

# Process the ticker when form is submitted
if submitted and ticker:
    # First validate if this is a biotech company
    try:
        is_biotech, company_info = is_biotech_company(ticker)
        
        if not is_biotech or not company_info:
            st.error(f"{ticker} is not a recognized biotech organization. Please try a different ticker.")
        else:
            # Only proceed if it's a valid biotech company
            # Update session state
            st.session_state.ticker = ticker
            st.session_state.job_status = "processing"
            
            # Show company name from validation
            st.info(f"Processing SEC filings for {company_info.get('name', ticker)}...")
            
            # Show processing message
            with st.spinner(f"Analyzing filings..."):
                # Import and call the flow controller
                import flow_controller
                result = flow_controller.process_ticker(ticker)
                
                if result["status"] == "success":
                    st.session_state.results = result["data"]
                    st.session_state.job_status = "success"
                    st.success(f"Analysis complete! Found {len(result['data'])} products.")
                else:
                    st.session_state.job_status = "error"
                    st.error(f"Analysis failed: {result['message']}")
    except Exception as e:
        st.error(f"Error validating ticker {ticker}: {str(e)}")

# Display results if available
if st.session_state.results:
    st.header(f"Product Analysis Results for {st.session_state.ticker}")
    
    # Create a DataFrame for displaying in a table
    product_data = []
    for product in st.session_state.results:
        product_data.append({
            "Name/Number": product.get("name", ""),
            "Mechanism of Action": product.get("mechanism_of_action", ""),
            "Indication": product.get("indication", "")
        })
    
    if product_data:
        df = pd.DataFrame(product_data)
        st.dataframe(df, use_container_width=True)
        
        # Show detailed view for selected product
        st.subheader("Detailed Product Information")
        selected_product = st.selectbox(
            "Select a product to see detailed information:",
            [p["Name/Number"] for p in product_data]
        )
        
        if selected_product:
            # Find the selected product
            product = next((p for p in st.session_state.results if p.get("name") == selected_product), None)
            
            if product:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### {product.get('name')}")
                    st.markdown(f"**Mechanism of Action:** {product.get('mechanism_of_action', 'N/A')}")
                    st.markdown(f"**Target:** {product.get('target', 'N/A')}")
                    st.markdown(f"**Indication:** {product.get('indication', 'N/A')}")
                
                with col2:
                    # Display preclinical data
                    st.markdown("#### Preclinical Data")
                    preclinical = product.get("preclinical_data", [])
                    if preclinical:
                        for item in preclinical:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown("No preclinical data available")
                
                # Clinical trials
                st.markdown("#### Clinical Trials")
                trials = product.get("clinical_trials", [])
                if trials:
                    for trial in trials:
                        if isinstance(trial, dict):
                            # Format dictionary as a readable string
                            trial_name = trial.get('name', '')
                            phase = trial.get('phase', '')
                            participants = trial.get('participants', '')
                            results = trial.get('results', '')
                            control = trial.get('control', '')
                            
                            # Build a formatted string
                            trial_text = f"**{trial_name}**: A {phase} {control} trial in {participants}. "
                            if results:
                                trial_text += f"{results}"
                            
                            st.markdown(f"- {trial_text}")
                        else:
                            # Handle string format directly
                            st.markdown(f"- {trial}")
                else:
                    st.markdown("No clinical trial data available")
                
                # Upcoming milestones
                st.markdown("#### Upcoming Milestones")
                milestones = product.get("upcoming_milestones", [])
                if milestones:
                    for milestone in milestones:
                        st.markdown(f"- {milestone}")
                else:
                    st.markdown("No upcoming milestones available")
                
                # References
                st.markdown("#### References")
                references = product.get("references", [])
                if references:
                    for ref in references:
                        st.markdown(f"- {ref}")
                else:
                    st.markdown("No references available")
    else:
        st.warning(f"No product information found for {st.session_state.ticker}")

# Add a button to start a new analysis
if st.session_state.results:
    # Convert full results to CSV (including all details)
    full_data = pd.json_normalize(st.session_state.results)
    full_csv = full_data.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download complete product details (CSV)",
        data=full_csv,
        file_name=f"{st.session_state.ticker}_complete_data.csv",
        mime="text/csv",
    )

    if st.button("Start New Analysis"):
        st.session_state.job_status = None
        st.session_state.ticker = None
        st.session_state.results = None
        st.rerun()