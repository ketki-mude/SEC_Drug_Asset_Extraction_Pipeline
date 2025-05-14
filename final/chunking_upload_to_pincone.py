import os
import re
import json
import uuid
import logging
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from s3_fetch_split import S3FileFetcher

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class SemanticChunker:
    """Class to perform semantic chunking and upload to Pinecone"""
    
    def __init__(self, similarity_threshold=0.75):
        """Initialize chunker with embedding model and Pinecone connection"""
        # Simply use all-mpnet-base-v2 model (reliable and works on Windows)
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        logging.info("Loaded embedding model: all-mpnet-base-v2")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Set up the index
        index_name = os.getenv('PINECONE_INDEX_NAME')
        
        # Check if index exists and create if needed (copied from pinecone_manager.py)
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            try:
                # Create index if it doesn't exist
                logging.info(f"Index '{index_name}' not found. Creating new index...")
                self.pc.create_index(
                    name=index_name,
                    dimension=768,  # all-mpnet-base-v2 dimensions
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": os.getenv('PINECONE_ENVIRONMENT')
                        },
                        "metric": "cosine"
                    }
                )
                logging.info(f"Created new Pinecone index: {index_name}")
            except Exception as e:
                logging.error(f"Failed to create index: {str(e)}")
                logging.info("Please create the index manually in the Pinecone Console")
                raise
        
        self.index = self.pc.Index(index_name)
        logging.info(f"Connected to Pinecone index: {index_name}")
        
        # Set similarity threshold for chunking
        self.similarity_threshold = similarity_threshold
    
    def semantic_chunking(self, text, similarity_threshold=None):
        """Improved semantic chunking with better paragraph handling"""
        # Use instance threshold if none provided
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        # Split text into paragraphs (more aggressively combine paragraphs)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 50]
        
        # If too many paragraphs, try to pre-combine some
        if len(paragraphs) > 100:
            logging.info(f"Too many paragraphs ({len(paragraphs)}), pre-combining some")
            combined_paragraphs = []
            current_combined = ""
            
            # Combine short paragraphs that likely belong together
            for p in paragraphs:
                if len(current_combined) < 500:  # Combine paragraphs until reasonable size
                    current_combined += " " + p if current_combined else p
                else:
                    combined_paragraphs.append(current_combined)
                    current_combined = p
                    
            # Add the last combined paragraph
            if current_combined:
                combined_paragraphs.append(current_combined)
                
            paragraphs = combined_paragraphs
            logging.info(f"Reduced to {len(paragraphs)} paragraphs")
        
        # Generate embeddings for paragraphs
        logging.info(f"Generating embeddings for {len(paragraphs)} paragraphs")
        embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in tqdm(range(0, len(paragraphs), batch_size), desc="Processing paragraph embeddings"):
            batch = paragraphs[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
        
        # Chunk paragraphs based on similarity
        chunks = []
        current_chunk = []
        current_chunk_text = ""
        
        for i, paragraph in tqdm(list(enumerate(paragraphs)), desc="Creating semantic chunks"):
            # If the current chunk is empty, add this paragraph
            if not current_chunk:
                current_chunk.append(i)
                current_chunk_text = paragraph
                continue
                
            # Calculate similarity with current chunk
            current_embedding = embeddings[i]
            chunk_embeddings = [embeddings[j] for j in current_chunk]
            chunk_embedding = np.mean(chunk_embeddings, axis=0)
            similarity = np.dot(current_embedding, chunk_embedding) / (np.linalg.norm(current_embedding) * np.linalg.norm(chunk_embedding))
            
            # If similar enough, add to current chunk, otherwise start new chunk
            if similarity > similarity_threshold:
                current_chunk.append(i)
                current_chunk_text += " " + paragraph
            else:
                chunks.append(current_chunk_text)
                current_chunk = [i]
                current_chunk_text = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk_text)
        
        logging.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def process_section(self, section):
        """Process a single section from the SEC filings"""
        # Apply semantic chunking
        chunks = self.semantic_chunking(section['text'])
        
        # Generate embeddings and prepare for Pinecone
        pinecone_records = []
        
        for chunk in tqdm(chunks, desc="Generating chunk embeddings"):
            # Skip huge chunks or break them down further
            if len(chunk.encode('utf-8')) > 38000:  # Keep ~2KB buffer under the 40KB limit
                logging.warning(f"Chunk too large ({len(chunk.encode('utf-8'))} bytes), breaking it down")
                # Split large chunks into smaller pieces (simple approach)
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk) if s.strip()]
                
                # Combine sentences into smaller chunks
                sub_chunks = []
                current_sub = ""
                
                for sentence in sentences:
                    if len((current_sub + " " + sentence).encode('utf-8')) < 38000:
                        current_sub += " " + sentence if current_sub else sentence
                    else:
                        if current_sub:
                            sub_chunks.append(current_sub)
                        current_sub = sentence
                
                if current_sub:
                    sub_chunks.append(current_sub)
                    
                # Process each sub-chunk
                for sub_chunk in sub_chunks:
                    self._add_chunk_to_records(sub_chunk, section, pinecone_records)
            else:
                # Process normal-sized chunk
                self._add_chunk_to_records(chunk, section, pinecone_records)
        
        return pinecone_records

    def _add_chunk_to_records(self, chunk_text, section, records):
        """Helper to add a chunk to records after checking size"""
        # Generate a unique ID
        chunk_id = f"{section['ticker']}_{section['filing_type']}_{section['date']}_{uuid.uuid4()}"
        
        # Generate embedding
        embedding = self.embedding_model.encode(chunk_text, show_progress_bar=False)
        
        # Create metadata
        metadata = {
            'filing_date': section['date'],
            'form_type': section['filing_type'],
            'text': chunk_text,
            'ticker_name': section['ticker'],
            'accession': section.get('accession', '')
        }
        
        # Check final size before adding
        metadata_size = len(json.dumps(metadata).encode('utf-8'))
        if metadata_size > 40000:
            # If still too large, truncate the text
            logging.warning(f"Metadata still too large ({metadata_size} bytes), truncating text")
            max_safe_length = int(len(chunk_text) * 0.9)  # Try 90% of current length
            metadata['text'] = chunk_text[:max_safe_length] + "..."
        
        # Add to records
        records.append({
            'id': chunk_id,
            'values': embedding.tolist(),
            'metadata': metadata
        })
    
    def upsert_to_pinecone(self, records):
        """
        Upsert records to Pinecone
        
        Args:
            records (list): List of records to upsert
            
        Returns:
            bool: Success or failure
        """
        try:
            # Upload in batches to avoid API limits
            batch_size = 100
            for i in tqdm(range(0, len(records), batch_size), desc="Upserting to Pinecone"):
                batch = records[i:i+batch_size]
                self.index.upsert(vectors=batch)
            
            logging.info(f"Successfully upserted {len(records)} records to Pinecone")
            return True
        
        except Exception as e:
            logging.error(f"Error upserting to Pinecone: {str(e)}")
            return False
    
    def process_sections(self, sections):
        """
        Process multiple sections and upload to Pinecone
        
        Args:
            sections (list): List of section data
            
        Returns:
            int: Number of records processed
        """
        all_records = []
        
        for section in tqdm(sections, desc="Processing sections"):
            records = self.process_section(section)
            all_records.extend(records)
        
        # Upsert all records to Pinecone
        if all_records:
            self.upsert_to_pinecone(all_records)
        
        return len(all_records)
    
    def process_ticker(self, ticker):
        """
        Process all filings for a ticker
        
        Args:
            ticker (str): Company ticker symbol
            
        Returns:
            int: Number of records processed
        """
        # Fetch and split filings
        fetcher = S3FileFetcher()
        sections = fetcher.fetch_and_split(ticker)
        
        if not sections:
            logging.warning(f"No sections found for ticker {ticker}")
            return 0
        
        # Process sections
        return self.process_sections(sections)

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic chunking and Pinecone upload for SEC filings")
    parser.add_argument("--ticker", required=True, help="Company ticker symbol")
    parser.add_argument("--sections", help="JSON file with pre-processed sections")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold for chunking (0.0-1.0)")
    parser.add_argument("--save", help="Save processed chunks to JSON file before uploading")
    
    args = parser.parse_args()
    
    chunker = SemanticChunker(similarity_threshold=args.threshold)
    
    # Either process a pre-processed sections file or fetch from S3
    if args.sections:
        try:
            with open(args.sections, 'r') as f:
                sections = json.load(f)
            
            num_records = chunker.process_sections(sections)
            print(f"Successfully processed {num_records} chunks for {len(sections)} sections")
        except Exception as e:
            logging.error(f"Error processing sections file: {str(e)}")
    else:
        num_records = chunker.process_ticker(args.ticker)
        print(f"Successfully processed {num_records} chunks for ticker {args.ticker}")
    #can run speratly using the following way if we want
    # python chunking_upload_to_pincone.py --ticker WVE