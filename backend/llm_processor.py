import os
import logging
import re
import pandas as pd
import traceback
from io import StringIO
from dotenv import load_dotenv

# Try to import the Google Generative AI library with error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    logging.error("Google Generative AI library not available. Install with: pip install google-generativeai")
    GEMINI_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("gemini_processor.log"), logging.StreamHandler()]
)

class GeminiProcessor:
    """Class for processing SEC filing data with Google Gemini to generate drug tables"""
    
    def __init__(self):
        """Initialize Gemini processor"""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = os.environ.get("GEMINI_MODEL", "gemini-pro")  # Simpler default model
        
        # Check if library is available
        if not GEMINI_AVAILABLE:
            logging.error("Google Generative AI library not available")
            self.model = None
            return
        
        # Configure Gemini
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                logging.info(f"Initialized Gemini with model: {self.model_name}")
            except Exception as e:
                logging.error(f"Error initializing Gemini: {str(e)}")
                self.model = None
        else:
            logging.error("Gemini API key not found")
            self.model = None
    
    def process_chunks(self, chunks, ticker):
        """
        Process chunks to extract drug program information using Gemini
        
        Args:
            chunks: List of text chunks from Pinecone
            ticker: Ticker symbol
            
        Returns:
            dict: Dictionary with output in different formats
        """
        try:
            if not chunks:
                return {
                    "markdown": f"No data found for ticker: {ticker}",
                    "csv": "",
                    "dataframe": pd.DataFrame()
                }
            
            if not GEMINI_AVAILABLE:
                return {
                    "markdown": "Error: Google Generative AI library not installed. Run: pip install google-generativeai",
                    "csv": "",
                    "dataframe": pd.DataFrame()
                }
            
            if not self.model:
                return {
                    "markdown": "Error: Gemini not properly initialized. Check your API key.",
                    "csv": "",
                    "dataframe": pd.DataFrame()
                }
            
            # Sort chunks by filing date (newest first) and limit
            chunks_to_process = sorted(
                chunks, 
                key=lambda x: x.get("filing_date", "0000-00-00"),
                reverse=True
            )
            
            # Limit to 20 chunks to avoid token limits
            chunks_to_process = chunks_to_process[:20]
            
            # Extract text from chunks
            combined_text = ""
            for chunk in chunks_to_process:
                chunk_text = chunk.get("text", "")
                if chunk_text:
                    # Add filing metadata and limit text size
                    filing_info = f"[Filing Date: {chunk.get('filing_date', 'Unknown')}, Form Type: {chunk.get('form_type', 'Unknown')}]\n"
                    combined_text += filing_info + chunk_text[:1500] + "\n\n"  # Limit each chunk
            
            # Ensure the text isn't too long (Gemini has token limits)
            max_text_length = 30000  # Conservative limit for Gemini
            if len(combined_text) > max_text_length:
                combined_text = combined_text[:max_text_length] + "...[text truncated]"
            
            # Create the prompt
            prompt = f"""
            Analyze the SEC filing data for {ticker} below and produce a structured table summarizing all drug/asset programs discovered.
            Each row = 1 drug/asset and include these columns:
            
            - Name/Number (e.g., WVE-N531)
            - Mechanism of Action (briefly explain how the drug works)
            - Target (biological molecule/pathway)
            - Indication (target disease/condition)
            - Animal Models / Preclinical Data (bullet-point summaries)
            - Clinical Trials (grouped bullets for each trial)
            - Upcoming Milestones (key expected events)
            - References (section headers, document types, filing dates)
            
            Here is the SEC filing data:
            {combined_text}
            
            Provide the table in both Markdown format and as a CSV.
            Format your response exactly like this:
            
            [MARKDOWN_TABLE]
            (your markdown table here)
            [/MARKDOWN_TABLE]
            
            [CSV_TABLE]
            (your CSV data here with header row)
            [/CSV_TABLE]
            
            Be comprehensive but concise. If information is missing for a column, note "Data not available".
            """
            
            # Call the Gemini API with error handling
            try:
                response = self.model.generate_content(prompt)
                result_text = response.text
            except Exception as api_error:
                logging.error(f"Error calling Gemini API: {str(api_error)}")
                # Try with a simplified prompt if the first attempt failed
                try:
                    simplified_prompt = f"Analyze this SEC filing data for {ticker} and list all drug programs mentioned: {combined_text[:15000]}"
                    response = self.model.generate_content(simplified_prompt)
                    result_text = response.text
                except Exception as retry_error:
                    logging.error(f"Error with simplified Gemini API call: {str(retry_error)}")
                    return {
                        "markdown": f"Failed to process data with Gemini API. Error: {str(api_error)}",
                        "csv": "",
                        "dataframe": pd.DataFrame()
                    }
            
            # Parse the response
            return self._parse_response(result_text, ticker)
            
        except Exception as e:
            error_details = traceback.format_exc()
            logging.error(f"Error processing chunks: {str(e)}\n{error_details}")
            return {
                "markdown": f"Error processing chunks: {type(e).__name__}. See logs for details.",
                "csv": "",
                "dataframe": pd.DataFrame()
            }
    
    def _parse_response(self, result_text, ticker):
        """Parse Gemini response to extract markdown and CSV"""
        try:
            # Extract the markdown table
            markdown_match = re.search(r'\[MARKDOWN_TABLE\](.*?)\[\/MARKDOWN_TABLE\]', result_text, re.DOTALL)
            markdown_table = markdown_match.group(1).strip() if markdown_match else ""
            
            # If no markdown format found, look for a markdown table
            if not markdown_table:
                table_match = re.search(r'(\|.*\|[\r\n]+\|[-|]+\|[\r\n]+(\|.*\|[\r\n]+)+)', result_text, re.DOTALL)
                if table_match:
                    markdown_table = table_match.group(1).strip()
            
            # Extract the CSV table
            csv_match = re.search(r'\[CSV_TABLE\](.*?)\[\/CSV_TABLE\]', result_text, re.DOTALL)
            csv_data = csv_match.group(1).strip() if csv_match else ""
            
            # If no CSV format found, try to convert markdown to CSV
            if not csv_data and markdown_table:
                try:
                    # Simple conversion from markdown to CSV
                    lines = markdown_table.strip().split('\n')
                    csv_lines = []
                    
                    for line in lines:
                        if line.startswith('|') and line.endswith('|'):
                            line = line.strip('|')
                            cells = [cell.strip() for cell in line.split('|')]
                            csv_line = []
                            for cell in cells:
                                # Escape quotes in cell content
                                cell = cell.replace('"', '""')
                                csv_line.append(f'"{cell}"')
                            csv_lines.append(','.join(csv_line))
                    
                    # Skip the separator line
                    if len(csv_lines) > 1:
                        csv_lines = [csv_lines[0]] + csv_lines[2:] if len(csv_lines) > 2 else csv_lines
                    
                    csv_data = '\n'.join(csv_lines)
                except Exception as e:
                    logging.error(f"Error converting markdown to CSV: {str(e)}")
            
            # Convert CSV to DataFrame
            df = pd.DataFrame()
            if csv_data:
                try:
                    df = pd.read_csv(StringIO(csv_data))
                except Exception as e:
                    logging.error(f"Error parsing CSV data: {str(e)}")
            
            # If nothing was extracted, use the full response
            if not markdown_table and not csv_data:
                markdown_table = f"## Drug Pipeline Analysis for {ticker}\n\n{result_text}"
            
            return {
                "markdown": markdown_table if markdown_table else "No structured data found in the response.",
                "csv": csv_data,
                "dataframe": df
            }
            
        except Exception as e:
            error_details = traceback.format_exc()
            logging.error(f"Error parsing response: {str(e)}\n{error_details}")
            
            # Return the raw text as markdown if parsing failed
            return {
                "markdown": f"## Raw Response for {ticker}\n\n{result_text[:2000]}...",
                "csv": "",
                "dataframe": pd.DataFrame()
            }