import re
import html
import logging

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
        
        # Common cleanup for all form types - remove problematic patterns
        
        # Remove sequences of dots (table of contents formatting)
        text = re.sub(r'\.{3,}', ' ', text)
        
        # Remove multiple dashes but preserve single dashes
        text = re.sub(r'-{2,}', ' ', text)
        
        # Remove multiple underscores
        text = re.sub(r'_{2,}', ' ', text)
        
        # Remove empty square brackets
        text = re.sub(r'\[\s*\]', '', text)
        
        # Remove javascript void links
        text = re.sub(r'\[([^\]]+)\]\(javascript:void\\\(0\\\);\)', r'\1', text)
        
        # Remove formatting patterns
        text = re.sub(r'\[X\]', '', text)  # checkbox markers
        text = re.sub(r'\[\\\+?[^\]]+\]', '', text)  # [+ Details], [\- Definition], etc.
        
        # Remove table formatting - completely remove pipe characters and format as space-separated
        text = re.sub(r'\|\s*\|\s*\|', ' ', text)  # Remove triple+ pipes
        text = re.sub(r'\|\s*\|', ' ', text)       # Remove double pipes
        text = re.sub(r'\s*\|\s*', ' ', text)      # Replace single pipes with spaces
        text = text.replace('|', ' ')              # Remove any remaining pipes
        
        # Remove markdown table separators
        text = re.sub(r'\n\s*[-+:|]+\s*\n', '\n', text)
        
        # Remove heading markers while preserving text
        text = re.sub(r'##\s+', '', text)
        text = re.sub(r'#\s+', '', text)
        
        # Clean up double asterisks (bold formatting) but preserve normal punctuation
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        
        # Remove excessive asterisks (* * *)
        text = re.sub(r'\n\* \* \*\n', '\n', text)
        text = re.sub(r'\*{2,}', '', text)
        
        # Remove special symbols that cause issues
        text = text.replace('✱', ' ')
        text = text.replace('★', ' ')
        text.replace('✓', ' ')
        
        # Handle bullets consistently
        text = text.replace('•', '* ')
        
        # Extra aggressive cleaning for 8-K forms
        if form_type and "8-K" in form_type:
            # Remove brackets
            text = re.sub(r'\[.*?\]', '', text)
            
            # Remove any remaining special characters
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
            
            # X markers and metadata definitions frequently appear in 8-K
            text = re.sub(r'\[X\](\(javascript:void\\\(0\\\);\))?\s+---.*?\*\*Period Type:\*\* \| duration\s+', '', text, flags=re.DOTALL)
            text = re.sub(r'\*\*Name:\*\* \| dei_.*?\n', '\n', text)
            text = re.sub(r'\*\*Namespace Prefix:\*\* \| dei_.*?\n', '\n', text)
            text = re.sub(r'\*\*Data Type:\*\* \| .*?\n', '\n', text)
            text = re.sub(r'\*\*Balance Type:\*\* \| na.*?\n', '\n', text)
        
        # Common final cleanup for all documents
        
        # Remove definition blocks
        text = re.sub(r'\[\\- Definition\].*?\[\\+ Details\].*?(\*\*.*?\*\*)', r'\1', text, flags=re.DOTALL)
        
        # Remove references and publisher information
        text = re.sub(r'\[\\+ References\].*?-Publisher SEC.*?-Subsection.*?\n', '\n', text, flags=re.DOTALL)
        
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

    @staticmethod
    def filter_financial_content(text, form_type=None):
        """
        Filter out financial data from text while preserving biomedical information.
        
        Args:
            text (str): Text to filter
            form_type (str, optional): Form type for specialized handling
            
        Returns:
            str: Filtered text with financial content removed
        """
        if not text:
            return text
            
        # Record original length for reporting
        original_length = len(text)
        
        # Patterns that identify financial sections to remove
        financial_section_patterns = [
            # Financial tables
            r'(?s)(CONSOLIDATED BALANCE SHEETS|CONSOLIDATED STATEMENTS OF OPERATIONS|FINANCIAL RESULTS).*?\n\n',
            r'(?s)(Assets|Liabilities.*?shareholders\' equity).*?(Total assets|Total liabilities)',
            r'(?s)(Revenue|Operating expenses:).*?(Net loss|Net income)',
            
            # Securities descriptions
            r'(?s)DESCRIPTION OF SECURITIES REGISTERED PURSUANT TO SECTION 12.*?EXCHANGE ACT OF 1934',
            r'(?s)The following description of our share capital.*?constitution.',
            
            # Stock and equity information
            r'(?s)EQUITY INCENTIVE PLAN.*?Eligible Award Recipients',
            r'(?s)UNDERWRITING AGREEMENT.*?Introductory',
            
            # Financial metrics
            r'(?s)(Cash and cash equivalents|Short-term investments).*?(Operating lease right-of-use assets)',
            r'(?s)FY\d{4} year-end cash total of.*?providing runway into',
            r'(?s)Cash and cash equivalents of \$\d+\.\d+ million',
            
            # Financial statements specific sections
            r'(?s)UNAUDITED CONSOLIDATED (BALANCE SHEETS|STATEMENTS OF OPERATIONS).*?\n\n',
            
            # Investor information
            r'(?s)Investor conference call and webcast.*?today',
            r'(?s)Investor Contact:.*?Media Contact:',
            r'(?s)To find out more, please visit.*?follow Wave on Twitter',
        ]
        
        # Forward-looking statements (standard boilerplate)
        forward_looking_patterns = [
            r'(?s)This document contains forward-looking statements.*?forward-looking statements in this',
            r'(?s)Forward-looking statements.*?similar expressions\.',
            r'(?s)The forward-looking statements in this presentation.*?risks and uncertainties\.',
        ]
        
        # Financial paragraph indicators (phrases commonly found in financial paragraphs)
        financial_paragraph_indicators = [
            r'\$\d+(\.\d+)? million', r'cash runway', r'operating expenses',
            r'revenue of \$\d+', r'net loss of \$\d+', r'expenses were \$\d+',
            r'income tax', r'balance sheet', r'cash equivalents',
            r'weighted-average', r'per share', r'stockholders', r'shareholders',
            r'total assets', r'total liabilities', r'additional paid-in capital',
            r'cash flows from', r'accumulated deficit', r'financing activities',
            r'accounting policies', r'GAAP', r'diluted earnings'
        ]
        
        # Biomedical terms that indicate content should be preserved
        biomedical_terms = [
            r'WVE-\d+', r'clinical trial', r'patient', r'drug candidate',
            r'AIMer', r'AATD', r'DMD', r'HD', r'RNA editing', r'disease',
            r'therapeutic', r'treatment', r'Phase \d[b/]?\d?[a]?', r'data showed', 
            r'results', r'FORWARD-\d+', r'SELECT-', r'RestorAATion',
            r'oligonucleotide', r'platform', r'PN-modified', r'exon skipping',
            r'protein', r'clinical-stage', r'dosing', r'CSF', r'efficacy',
            r'genetic medicines', r'siRNA', r'GalNAc', r'Huntington', r'splicing'
        ]
        
        # Check if document should be completely skipped
        skip_document_patterns = [
            r'DESCRIPTION OF SECURITIES REGISTERED PURSUANT TO SECTION 12',
            r'EQUITY INCENTIVE PLAN',
            r'UNDERWRITING AGREEMENT',
            r'AMENDMENT.*?OPEN MARKET SALE AGREEMENT',
            r'Document and Entity Information'
        ]
        
        for pattern in skip_document_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Check if it also contains biomedical keywords
                bio_count = sum(1 for term in biomedical_terms 
                              if re.search(term, text, re.IGNORECASE))
                
                # Skip if minimal biomedical content
                if bio_count < 3:
                    logging.info("Document appears to be primarily financial/legal - skipping")
                    return "[DOCUMENT SKIPPED - PRIMARILY FINANCIAL/LEGAL CONTENT]"
        
        # Remove entire financial sections
        for pattern in financial_section_patterns:
            text = re.sub(pattern, '[FINANCIAL_SECTION_REMOVED]\n\n', text, flags=re.IGNORECASE)
        
        # Remove forward-looking statements
        for pattern in forward_looking_patterns:
            text = re.sub(pattern, '[FORWARD_LOOKING_STATEMENTS_REMOVED]\n\n', text, flags=re.IGNORECASE)
        
        # Process paragraph by paragraph
        paragraphs = re.split(r'\n\s*\n', text)
        filtered_paragraphs = []
        
        for paragraph in paragraphs:
            # Skip placeholders we already inserted
            if '[FINANCIAL_SECTION_REMOVED]' in paragraph or '[FORWARD_LOOKING_STATEMENTS_REMOVED]' in paragraph:
                filtered_paragraphs.append(paragraph)
                continue
                
            # Skip very short paragraphs
            if len(paragraph.strip()) < 20:
                filtered_paragraphs.append(paragraph)
                continue
            
            # Count financial terms in paragraph
            financial_term_count = sum(1 for term in financial_paragraph_indicators 
                                     if re.search(term, paragraph, re.IGNORECASE))
            
            # Count biomedical terms
            bio_count = sum(1 for term in biomedical_terms 
                          if re.search(term, paragraph, re.IGNORECASE))
            
            # Keep if it has biomedical terms, remove if it has financial terms without biomedical terms
            if bio_count > 0 or financial_term_count < 2:
                filtered_paragraphs.append(paragraph)
            else:
                filtered_paragraphs.append('[FINANCIAL_PARAGRAPH_REMOVED]')
        
        filtered_text = '\n\n'.join(filtered_paragraphs)
        
        # Clean up multiple consecutive removal markers
        filtered_text = re.sub(r'(\[\w+_REMOVED\]\s*){2,}', 
                             r'[MULTIPLE_SECTIONS_REMOVED]\n\n', filtered_text)
        
        # Calculate reduction stats
        final_length = len(filtered_text)
        reduction_percent = ((original_length - final_length) / original_length) * 100
        logging.info(f"Reduced text by {reduction_percent:.1f}% " +
                    f"(from {original_length} to {final_length} characters)")
        
        return filtered_text