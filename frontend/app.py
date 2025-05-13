import streamlit as st
import pandas as pd
import os
import sys
import traceback
from dotenv import load_dotenv

# Get the directory of the current file and its parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Make sure the current directory is in the path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="SEC Filing Drug Asset Analyzer",
    page_icon="💊",
    layout="wide"
)

# Import after environment variables are loaded
try:
    from backend.pinecone_manager import PineconeManager
    from backend.semantic_chunking import ST_AVAILABLE
    from backend.llm_processor import GeminiProcessor
except ImportError as e:
    st.error(f"Import error: {str(e)}")
    st.error("Make sure all Python files are in the same directory.")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .title {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #3498db;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        padding: 15px;
        margin-bottom: 15px;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin-bottom: 15px;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pinecone_manager():
    """Get or initialize PineconeManager with caching and error handling"""
    try:
        return PineconeManager()
    except Exception as e:
        st.error(f"Error initializing PineconeManager: {str(e)}")
        return None

@st.cache_resource
def get_gemini_processor():
    """Get or initialize GeminiProcessor with caching and error handling"""
    try:
        return GeminiProcessor()
    except Exception as e:
        st.error(f"Error initializing GeminiProcessor: {str(e)}")
        return None

def main():
    st.markdown("<h1 class='title'>SEC Filing Drug Asset Analyzer</h1>", unsafe_allow_html=True)
    
    st.markdown(
        """This tool analyzes SEC filings for biotech companies, extracts information 
        about drug assets, and creates structured tables summarizing their development 
        pipeline. Simply enter a ticker symbol to get started."""
    )
    
    # Show model information box
    with st.expander("About the Models Used", expanded=False):
        st.markdown("""
        **Semantic Chunking**: 
        - Using Sentence Transformers with the all-MiniLM-L6-v2 model for semantic text chunking
        - Fallback to structural chunking if not available
        
        **LLM Processing**:
        - Using Google's Gemini Pro model for summarizing drug information
        
        **Vector Storage**:
        - Using Pinecone serverless for vector database storage
        """)
        
        # Show warning if Sentence Transformers isn't available
        if not ST_AVAILABLE:
            st.warning("⚠️ Sentence Transformers is not installed. Using fallback chunking method.")
    
    # Sidebar for API keys and settings
    with st.sidebar:
        st.subheader("Configuration")
        
        if not os.environ.get("PINECONE_API_KEY"):
            pinecone_key = st.text_input("Pinecone API Key", type="password")
            if pinecone_key:
                os.environ["PINECONE_API_KEY"] = pinecone_key
        
        if not os.environ.get("GEMINI_API_KEY"):
            gemini_key = st.text_input("Google Gemini API Key", type="password")
            if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key
                os.environ["GEMINI_MODEL"] = "gemini-pro"
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            max_chunks = st.slider("Max chunks to process", min_value=5, max_value=50, value=20, step=5)
        
        st.markdown("---")
        st.markdown("### How It Works")
        st.markdown("""
        1. Retrieves SEC filings from S3
        2. Uses true semantic chunking to process text
        3. Stores in Pinecone vector database
        4. Analyzes with Google Gemini AI to extract drug info
        5. Generates structured drug tables
        """)
    
    # Initialize managers
    pinecone_manager = get_pinecone_manager()
    gemini_processor = get_gemini_processor()
    
    if not pinecone_manager or not gemini_processor:
        st.error("Failed to initialize required components.")
        st.stop()
    
    # Input form
    with st.form("ticker_form"):
        ticker = st.text_input("Enter Ticker Symbol:", "").strip().upper()
        submitted = st.form_submit_button("Analyze Drug Pipeline")
    
    if submitted and ticker:
        # Check for API keys
        if not os.environ.get("GEMINI_API_KEY"):
            st.error("Please enter your Google Gemini API key in the sidebar.")
            st.stop()
            
        if not os.environ.get("PINECONE_API_KEY"):
            st.error("Please enter your Pinecone API key in the sidebar.")
            st.stop()
        
        try:
            # Process the ticker with progress tracking
            progress_bar = st.progress(0)
            status = st.empty()
            
            # Check S3 connection
            status.markdown('<div class="info-box">Checking S3 connection...</div>', unsafe_allow_html=True)
            progress_bar.progress(10)
            
            if not pinecone_manager.s3_storage.check_connection():
                st.error("Failed to connect to S3. Check your AWS credentials.")
                st.stop()
                
            # Check if ticker exists in S3
            progress_bar.progress(20)
            
            filings = pinecone_manager.get_all_s3_filings(ticker)
            if not filings:
                st.error(f"No SEC filings found in S3 for ticker: {ticker}")
                st.stop()
                
            progress_bar.progress(30)
            status.markdown(f'<div class="info-box">Found {len(filings)} SEC filings for {ticker}</div>', unsafe_allow_html=True)
            
            # Check if ticker exists in Pinecone
            progress_bar.progress(40)
            status.markdown('<div class="info-box">Checking if ticker data exists in Pinecone...</div>', unsafe_allow_html=True)
            
            if not pinecone_manager.check_ticker_in_pinecone(ticker):
                status.markdown(f'<div class="info-box">Processing SEC filings for {ticker}...</div>', unsafe_allow_html=True)
                progress_bar.progress(50)
                
                # Process ticker and add to Pinecone
                success = pinecone_manager.process_ticker(ticker)
                
                progress_bar.progress(75)
                
                if not success:
                    st.error(f"Failed to process SEC filings for {ticker}")
                    st.stop()
                    
                status.markdown(f'<div class="success-box">Successfully processed SEC filings for {ticker}</div>', unsafe_allow_html=True)
            else:
                progress_bar.progress(60)
                status.markdown(f'<div class="success-box">Found existing data for {ticker} in Pinecone</div>', unsafe_allow_html=True)
                
            # Generate tables
            progress_bar.progress(80)
            status.markdown('<div class="info-box">Analyzing data and generating tables...</div>', unsafe_allow_html=True)
                
            # Get chunks from Pinecone
            chunks = pinecone_manager.query_chunks(ticker)
                
            if not chunks:
                st.error(f"No data found for {ticker} in Pinecone")
                st.stop()
                
            progress_bar.progress(90)
                
            # Limit chunks by setting
            chunks = chunks[:max_chunks]
                
            # Process chunks with Gemini
            result = gemini_processor.process_chunks(chunks, ticker)
                
            # Complete progress
            progress_bar.progress(100)
            status.empty()
                
            # Display results
            markdown_table = result["markdown"]
            csv_data = result["csv"]
            df = result["dataframe"]
                
            st.subheader(f"Drug Pipeline for {ticker}")
            st.markdown(markdown_table)
                
            # Display DataFrame if available
            if not df.empty:
                st.subheader("Data Table View")
                st.dataframe(df)
                
            # Download buttons
            st.subheader("Download Results")
                
            col1, col2 = st.columns(2)
                
            with col1:
                if csv_data:
                    st.download_button(
                        label="Download as CSV",
                        data=csv_data,
                        file_name=f"{ticker}_drug_pipeline.csv",
                        mime="text/csv"
                    )
                
            with col2:
                if markdown_table:
                    st.download_button(
                        label="Download as Markdown",
                        data=markdown_table,
                        file_name=f"{ticker}_drug_pipeline.md",
                        mime="text/markdown"
                    )
                
        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Error: {str(e)}")
            with st.expander("Error Details"):
                st.code(error_details)
                
    elif not submitted:
        # Show example when no ticker entered
        st.markdown("---")
        st.markdown("### Example")
        
        st.markdown("""
        Enter a ticker like `WVE` (Wave Life Sciences) to see information about their drug pipeline:
        
        - **WVE-006**: RNA editing therapeutic for AATD
        - **WVE-N531**: Exon skipping therapy for DMD
        - **WVE-003**: Allele-selective silencing for Huntington's Disease
        - **INHBE Program**: Treatment for metabolic disorders and obesity
        
        The system will analyze all available SEC filings, extract information about these 
        drug programs, and create structured tables with details on each asset.
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())