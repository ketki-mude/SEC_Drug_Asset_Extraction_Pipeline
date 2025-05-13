import re
import html
import logging
import unicodedata


class SECTextProcessor:
    """Class for processing and cleaning SEC filing text to improve readability"""

    @staticmethod
    def clean_html_content(html_text):
        """Clean HTML content by removing problematic elements and characters"""
        # Remove version info (like v3.19.3.a.u2)
        html_text = re.sub(r'v\d+\.\d+\.\d+\.?\w*\s+', '', html_text)
        
        # Remove HTML comments
        html_text = re.sub(r'<!--.*?-->', '', html_text, flags=re.DOTALL)
        
        return html_text

    @staticmethod
    def process_text(text, form_type=None):
        """
        Process extracted text to improve readability
        
        Args:
            text: Text to process
            form_type: Type of form (8-K, 10-K, etc.) for special handling
        """
        if not text:
            return text
            
        # Decode HTML entities
        text = html.unescape(text)
        
        # Extra aggressive cleaning for 8-K forms
        if form_type and "8-K" in form_type:
            # Remove all special symbols and formatting characters (keep only alphanumeric, basic punctuation)
            
            # First, remove common markdown/formatting patterns
            text = re.sub(r'\[([^\]]+)\]\(javascript:void\\\(0\\\);\)', r'\1', text)  # javascript links
            text = re.sub(r'\[X\]', '', text)  # checkbox markers
            text = re.sub(r'\[\\\+?[^\]]+\]', '', text)  # [+ Details], [\- Definition], etc.
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold text**
            
            # Remove all pipe characters and table formatting
            text = text.replace('|', ' ')
            text = re.sub(r'\n\s*[-+:|]+\s*\n', '\n', text)  # table separators
            
            # Remove heading markers while preserving text
            text = re.sub(r'#+\s+', '', text)  # Remove # and ## markers
            
            # Remove asterisks, bullets, and other special characters
            text = text.replace('*', ' ')
            text = text.replace('•', ' ')
            text = text.replace('✱', ' ')
            text = text.replace('★', ' ')
            text = text.replace('✓', ' ')
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
            
            # Remove excessive whitespace
            text = re.sub(r'\s{2,}', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
        else:
            # Standard processing for other forms (10-K, etc.)
            # Remove javascript void links
            text = re.sub(r'\[([^\]]+)\]\(javascript:void\\\(0\\\);\)', r'\1', text)
            
            # Better handling for tables - completely remove pipe characters and format as space-separated
            text = re.sub(r'\|\s*\|\s*\|', ' ', text)  # Remove triple+ pipes
            text = re.sub(r'\|\s*\|', ' ', text)       # Remove double pipes
            text = re.sub(r'\s*\|\s*', ' ', text)      # Replace single pipes with spaces
            text = text.replace('|', ' ')  # Remove any remaining pipes
            
            # Remove markdown table separators
            text = re.sub(r'\n\s*[-+:|]+\s*\n', '\n', text)
            
            # Fix bullet points
            text = re.sub(r'• \s+', '• ', text)
            
            # Fix numbered lists
            text = re.sub(r'\n(\d+)\.\s+', r'\n\1. ', text)
            
            # Remove heading markers while preserving text
            text = re.sub(r'##\s+', '', text)
            text = re.sub(r'#\s+', '', text)
            
            # Remove excessive asterisks (* * *)
            text = re.sub(r'\n\* \* \*\n', '\n', text)
            
            # Fix double/triple blank lines
            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
            
            # Remove references and publisher information
            text = re.sub(r'\[\\+ References\].*?-Publisher SEC.*?-Subsection.*?\n', '\n', text, flags=re.DOTALL)
            
            # Clean up double asterisks (bold formatting) but preserve normal punctuation
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            
            # Clean up any remaining table formatting
            text = re.sub(r'---\s*', '', text)
            
            # Remove X markers and metadata definitions
            text = re.sub(r'\[X\](\(javascript:void\\\(0\\\);\))?\s+---.*?\*\*Period Type:\*\* \| duration\s+', '', text, flags=re.DOTALL)
            text = re.sub(r'\*\*Name:\*\* \| dei_.*?\n', '\n', text)
            text = re.sub(r'\*\*Namespace Prefix:\*\* \| dei_.*?\n', '\n', text)
            text = re.sub(r'\*\*Data Type:\*\* \| .*?\n', '\n', text)
            text = re.sub(r'\*\*Balance Type:\*\* \| na.*?\n', '\n', text)
            
            # Remove definition blocks
            text = re.sub(r'\[\\- Definition\].*?\[\\+ Details\].*?(\*\*.*?\*\*)', r'\1', text, flags=re.DOTALL)
            
        # Common cleanup for all form types - these should be safe for all documents
        
        # Fix spacing after removing formatting
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Final cleanup of excess newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text

    @staticmethod
    def extract_better_sections(text, form_type):
        """Extract better sections from filing text based on form type"""
        sections = []
        
        # For 10-K filings, extract meaningful sections
        if form_type in ["10-K", "10-K/A"]:
            # Look for standard 10-K section patterns
            section_patterns = [
                # Main section headers (Part I, Part II, etc.)
                r'(PART\s+[IVX]+[A-Z]?)\s*[-–—]\s*([^\n]+)',
                # Item sections
                r'(ITEM\s+\d+[A-Z]?)[.\s]+([^\n]+)',
                r'(Item\s+\d+[A-Z]?)[.\s]+([^\n]+)'
            ]
            
            last_pos = 0
            all_matches = []
            
            for pattern in section_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    title = f"{match.group(1)}: {match.group(2).strip()}"
                    start_pos = match.start()
                    all_matches.append((start_pos, title))
            
            # Sort matches by position
            all_matches.sort()
            
            # Extract text between matched positions
            for i, (pos, title) in enumerate(all_matches):
                if i < len(all_matches) - 1:
                    next_pos = all_matches[i + 1][0]
                    section_text = text[pos:next_pos].strip()
                else:
                    section_text = text[pos:].strip()
                
                sections.append({
                    "title": title,
                    "text": section_text
                })
        
        # For 8-K filings, try to extract items
        elif form_type in ["8-K", "8-K/A"]:
            # Look for 8-K item patterns
            section_patterns = [
                r'(ITEM\s+\d+\.\d+)[.\s]+([^\n]+)',
                r'(Item\s+\d+\.\d+)[.\s]+([^\n]+)'
            ]
            
            all_matches = []
            
            for pattern in section_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    title = f"{match.group(1)}: {match.group(2).strip()}"
                    start_pos = match.start()
                    all_matches.append((start_pos, title))
            
            # Sort matches by position
            all_matches.sort()
            
            # Extract text between matched positions
            for i, (pos, title) in enumerate(all_matches):
                if i < len(all_matches) - 1:
                    next_pos = all_matches[i + 1][0]
                    section_text = text[pos:next_pos].strip()
                else:
                    section_text = text[pos:].strip()
                
                sections.append({
                    "title": title,
                    "text": section_text
                })
            
            # Also check for exhibits
            exhibit_pattern = r'(EXHIBIT\s+\d+\.\d+)[.\s]+([^\n]+)'
            for match in re.finditer(exhibit_pattern, text, re.IGNORECASE):
                title = f"{match.group(1)}: {match.group(2).strip()}"
                sections.append({
                    "title": title,
                    "text": match.group(0).strip()
                })
        
        # If no sections found, extract key information from filing
        if not sections:
            cleaned_title = "Document Summary"
            
            # For 8-K, try to extract important information
            if form_type in ["8-K", "8-K/A"]:
                # Try to find the event date or filing purpose
                event_date_match = re.search(r'(date of report|event date|event occurred).*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2}, \d{4})', 
                                            text, re.IGNORECASE)
                if event_date_match:
                    cleaned_title = f"8-K Event: {event_date_match.group(2)}"
                
                # Try to find purpose of filing
                purpose_match = re.search(r'(Item\s+\d+\.\d+).*?(results of operations|financial statements|material events|material information)', 
                                         text, re.IGNORECASE)
                if purpose_match:
                    cleaned_title = f"8-K Filing: {purpose_match.group(2).strip().capitalize()}"
            
            # For 10-K, extract key information
            elif form_type in ["10-K", "10-K/A"]:
                # Try to extract fiscal year or period
                fiscal_match = re.search(r'(fiscal year|annual report|period ending).*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2}, \d{4})', 
                                        text, re.IGNORECASE)
                if fiscal_match:
                    cleaned_title = f"10-K Annual Report: {fiscal_match.group(2)}"
                else:
                    cleaned_title = "10-K Annual Report"
            
            sections.append({
                "title": cleaned_title,
                "text": text
            })
        
        return sections

    @staticmethod
    def process_filing_content(filing_data):
        """Process filing content to improve readability"""
        try:
            if not filing_data:
                return filing_data
                
            metadata = filing_data.get("metadata", {})
            content = filing_data.get("content", {})
            
            form_type = metadata.get("form_type", "")
            full_text = content.get("full_text", "")
            
            # Process the full text
            cleaned_text = SECTextProcessor.process_text(full_text, form_type)
            
            # Extract better sections
            better_sections = SECTextProcessor.extract_better_sections(cleaned_text, form_type)
            
            # Update the content
            content["full_text"] = cleaned_text
            content["sections"] = better_sections
            
            # Update the filing data
            filing_data["content"] = content
            
            return filing_data
            
        except Exception as e:
            logging.error(f"Error processing filing content: {str(e)}")
            return filing_data  # Return original if processing fails