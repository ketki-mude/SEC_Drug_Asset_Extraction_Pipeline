import os
import json
import logging
import time
import uuid
import re
import traceback
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from .aws_storage import S3Storage
from .semantic_chunking import SemanticChunker, ST_AVAILABLE

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Sentence Transformers not installed. Using fallback embedding method.")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pinecone_manager.log"), logging.StreamHandler()]
)

class PineconeManager:
    """Class for managing Pinecone operations for SEC filings"""
    
    def __init__(self):
        """Initialize Pinecone manager with S3 storage and semantic chunking"""
        # Pinecone Configuration
        self.api_key = os.environ.get("PINECONE_API_KEY")
        self.environment = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.environ.get("PINECONE_INDEX_NAME", "sec-filings")
        
        # AWS S3 Integration
        self.s3_storage = S3Storage()
        
        # Semantic chunker
        self.chunker = SemanticChunker(max_chunk_size=1000, overlap=100)
        
        # Initialize embedding model
        self.embedding_model = None
        self.native_dim = 384  # Native dimension of all-MiniLM-L6-v2
        self.target_dim = 1536  # Target dimension for Pinecone (OpenAI compatible)
        
        if EMBEDDINGS_AVAILABLE:
            try:
                model_name = "all-MiniLM-L6-v2"
                self.embedding_model = SentenceTransformer(model_name)
                logging.info(f"Initialized embedding model: {model_name}")
            except Exception as e:
                logging.error(f"Error loading embedding model: {str(e)}")
                self.embedding_model = None
        
        # Maximum number of filings to process at once
        self.max_filings = 20
        
        # Initialize Pinecone
        self.pc = None
        self.index = None
        self.setup_pinecone()
    
    def setup_pinecone(self):
        """Set up Pinecone client and ensure index exists"""
        try:
            if not self.api_key:
                logging.error("Pinecone API key not found")
                return False
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists, create if not
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                logging.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.target_dim,  # Using target dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to initialize
                time.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logging.info(f"Connected to Pinecone index: {self.index_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error setting up Pinecone: {str(e)}")
            return False
    
    def create_embedding(self, text):
        """
        Create embedding for text and resize to target dimension
        
        Args:
            text: Text to embed
            
        Returns:
            list: Embedding vector with target dimension
        """
        if not text or len(text.strip()) < 10:
            # Random vector for empty text
            return list(np.random.normal(0, 0.01, self.target_dim))
        
        try:
            # Clean text
            text = self._prepare_text(text)
            
            # Use Sentence Transformers if available
            if EMBEDDINGS_AVAILABLE and self.embedding_model:
                # Generate native embedding
                native_embedding = self.embedding_model.encode(text)
                
                # Resize to target dimension
                return self._resize_vector(native_embedding)
            else:
                # Fallback method
                return self._create_deterministic_embedding(text)
        except Exception as e:
            logging.error(f"Error creating embedding: {str(e)}")
            return self._create_deterministic_embedding(text)
    
    def _prepare_text(self, text):
        """Clean text for embedding"""
        if not text:
            return ""
            
        # Remove special characters
        text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate very long texts
        if len(text) > 5000:
            text = text[:5000]
        
        return text.strip()
    
    def _resize_vector(self, vector):
        """
        Resize a vector from native dimension to target dimension
        
        Args:
            vector: Native dimension vector
            
        Returns:
            list: Resized vector with target dimension
        """
        vector = np.array(vector)
        
        # If dimensions match, return as is
        if len(vector) == self.target_dim:
            return vector.tolist()
            
        # If native is smaller than target
        if len(vector) < self.target_dim:
            # Calculate repeat count
            repeat_count = self.target_dim // len(vector)
            remainder = self.target_dim % len(vector)
            
            # Repeat the vector and add remainder
            resized = np.tile(vector, repeat_count)
            
            if remainder > 0:
                resized = np.concatenate([resized, vector[:remainder]])
                
            # Add small noise to make repeated sections slightly different
            noise = np.random.normal(0, 0.01, self.target_dim)
            resized = resized + noise
        else:
            # Truncate if native is larger than target
            resized = vector[:self.target_dim]
        
        # Normalize
        norm = np.linalg.norm(resized)
        if norm > 0:
            resized = resized / norm
            
        return resized.tolist()
    
    def _create_deterministic_embedding(self, text):
        """Generate a deterministic embedding based on text features"""
        # Generate a seed from text hash
        text_hash = sum(ord(c) for c in text[:100])
        np.random.seed(text_hash)
        
        # Create random vector
        vector = np.random.normal(0, 0.1, self.target_dim)
        
        # Add some text-based features
        # Text length feature
        length_factor = min(len(text) / 1000, 1.0)
        vector = vector * (0.8 + 0.4 * length_factor)
        
        # Text structure features
        paragraphs = text.count('\n\n') + 1
        paragraph_factor = min(paragraphs / 10, 1.0)
        vector[0:10] = vector[0:10] * (1.0 + 0.2 * paragraph_factor)
        
        # Content features
        feature_terms = {
            'clinical': 50, 'study': 51, 'drug': 52, 'treatment': 53,
            'patient': 54, 'disease': 55, 'trial': 56, 'phase': 57,
            'therapy': 58, 'product': 59, 'pipeline': 60, 'development': 61,
            'fda': 62, 'approval': 63, 'candidate': 64, 'target': 65,
            'protein': 66, 'cell': 67, 'gene': 68, 'dna': 69,
            'rna': 70, 'mutation': 71, 'receptor': 72, 'enzyme': 73,
            'antibody': 74, 'molecule': 75, 'compound': 76, 'mechanism': 77
        }
        
        # Boost features that are present
        for term, idx in feature_terms.items():
            if term in text.lower():
                vector[idx % self.target_dim] *= 1.3
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def process_ticker(self, ticker):
        """
        Process all filings for a ticker and store in Pinecone
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            bool: True if successful, False otherwise
        """
        ticker = ticker.upper()
        
        try:
            # Check if ticker already in Pinecone
            if self.check_ticker_in_pinecone(ticker):
                logging.info(f"Ticker {ticker} already exists in Pinecone")
                return True
            
            # Get all filings from S3
            filings = self.get_all_s3_filings(ticker)
            
            if not filings:
                logging.warning(f"No filings found for ticker: {ticker}")
                return False
            
            logging.info(f"Processing {len(filings)} filings for ticker: {ticker}")
            
            # Process each filing
            all_documents = []
            
            for filing in filings:
                try:
                    documents = self.process_filing(filing)
                    all_documents.extend(documents)
                except Exception as e:
                    logging.error(f"Error processing filing: {str(e)}")
                    continue  # Continue with next filing
            
            # Upload to Pinecone
            if all_documents:
                success = self.upsert_to_pinecone(all_documents)
                return success
            else:
                logging.warning(f"No documents generated for ticker: {ticker}")
                return False
                
        except Exception as e:
            logging.error(f"Error processing ticker {ticker}: {str(e)}")
            return False
    
    def get_all_s3_filings(self, ticker):
        """
        Get all filings for a ticker from S3
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            list: List of filing JSON objects
        """
        ticker = ticker.upper()
        all_filings = []
        
        # Check if ticker exists in S3
        if not self.s3_storage.check_connection():
            logging.error("S3 connection failed")
            return []
        
        # Get list of filings
        logging.info(f"Listing S3 files for ticker: {ticker}")
        filing_files = []
        
        # Use paginator to get all objects
        try:
            s3_client = self.s3_storage.s3_client
            paginator = s3_client.get_paginator('list_objects_v2')
            
            # Get all pages of results
            for page in paginator.paginate(Bucket=self.s3_storage.bucket_name, Prefix=f"{ticker}/"):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Only get 8-K and 10-K JSON files
                        if '/8-K/' in key or '/10-K/' in key:
                            if key.endswith('.json'):
                                filing_files.append(key)
            
            logging.info(f"Found {len(filing_files)} filing files for {ticker}")
            
            # Limit the number of filings to process
            if len(filing_files) > self.max_filings:
                logging.info(f"Limiting to {self.max_filings} filings for processing")
                filing_files = filing_files[:self.max_filings]
            
            # Download each filing
            for file_key in filing_files:
                try:
                    # Get object from S3
                    response = s3_client.get_object(
                        Bucket=self.s3_storage.bucket_name,
                        Key=file_key
                    )
                    
                    # Parse JSON
                    content = response['Body'].read().decode('utf-8')
                    filing_data = json.loads(content)
                    
                    all_filings.append(filing_data)
                    logging.info(f"Loaded filing: {file_key}")
                except Exception as e:
                    logging.error(f"Error loading filing {file_key}: {str(e)}")
        
        except Exception as e:
            logging.error(f"Error listing S3 files: {str(e)}")
        
        return all_filings
    
    def process_filing(self, filing_data):
        """
        Process a filing into semantic chunks with embeddings
        
        Args:
            filing_data: Filing JSON data
            
        Returns:
            list: List of document objects ready for Pinecone
        """
        try:
            metadata = filing_data.get("metadata", {})
            content = filing_data.get("content", {})
            
            # Get the full text
            full_text = content.get("full_text", "")
            
            if not full_text:
                logging.warning(f"Empty text for filing: {metadata.get('accession_number')}")
                return []
            
            # Use semantic chunker to create chunks
            chunks = self.chunker.chunk_text(full_text)
            
            # Prepare documents for Pinecone
            pinecone_docs = []
            
            for i, chunk_text in enumerate(chunks):
                try:
                    # Skip empty chunks
                    if not chunk_text:
                        continue
                        
                    # Generate embedding (resized to target dimension)
                    embedding = self.create_embedding(chunk_text)
                    
                    # Generate a unique ID
                    doc_id = f"{metadata.get('ticker')}_{metadata.get('filing_date')}_{metadata.get('form_type')}_{i}_{uuid.uuid4().hex[:8]}"
                    
                    # Limit text size to avoid metadata limits
                    truncated_text = chunk_text[:2000] if len(chunk_text) > 2000 else chunk_text
                    
                    # Create document
                    document = {
                        "id": doc_id,
                        "values": embedding,
                        "metadata": {
                            "ticker": metadata.get("ticker", "").upper(),
                            "cik": metadata.get("cik", ""),
                            "form_type": metadata.get("form_type", ""),
                            "filing_date": metadata.get("filing_date", ""),
                            "accession_number": metadata.get("accession_number", ""),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "text": truncated_text
                        }
                    }
                    
                    pinecone_docs.append(document)
                except Exception as chunk_error:
                    logging.error(f"Error processing chunk {i}: {str(chunk_error)}")
                    continue
            
            logging.info(f"Created {len(pinecone_docs)} chunks for filing: {metadata.get('accession_number')}")
            return pinecone_docs
            
        except Exception as e:
            logging.error(f"Error processing filing: {str(e)}")
            return []
    
    def upsert_to_pinecone(self, documents, batch_size=20):
        """
        Upload documents to Pinecone in batches
        
        Args:
            documents: List of documents to upload
            batch_size: Size of batches for uploading
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.index:
            logging.error("Pinecone index not initialized")
            return False
        
        try:
            total = len(documents)
            logging.info(f"Uploading {total} documents to Pinecone")
            
            for i in range(0, total, batch_size):
                batch = documents[i:i+batch_size]
                
                # Convert to Pinecone format
                vectors = []
                for doc in batch:
                    # Ensure text isn't too large for metadata
                    metadata = doc["metadata"].copy()
                    if "text" in metadata and len(metadata["text"]) > 2000:
                        metadata["text"] = metadata["text"][:2000]
                    
                    # Ensure values is the right dimension
                    values = doc["values"]
                    if len(values) != self.target_dim:
                        logging.warning(f"Vector dimension mismatch: {len(values)} vs {self.target_dim}")
                        continue
                    
                    vectors.append({
                        "id": doc["id"],
                        "values": values,
                        "metadata": metadata
                    })
                
                # Upsert to Pinecone
                if vectors:
                    self.index.upsert(vectors=vectors)
                
                logging.info(f"Uploaded batch {i//batch_size + 1}/{(total+batch_size-1)//batch_size}")
                
                # Respect API rate limits
                time.sleep(0.5)
            
            return True
            
        except Exception as e:
            logging.error(f"Error upserting to Pinecone: {str(e)}")
            return False
    
    def check_ticker_in_pinecone(self, ticker):
        """
        Check if ticker is already in Pinecone
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            bool: True if ticker exists, False otherwise
        """
        if not self.index:
            return False
        
        try:
            ticker = ticker.upper()
            
            # Query with filter
            response = self.index.query(
                vector=[0] * self.target_dim,  # Dummy vector for metadata filtering
                filter={"ticker": ticker},
                top_k=1,
                include_metadata=False
            )
            
            return len(response.matches) > 0
            
        except Exception as e:
            logging.error(f"Error checking ticker in Pinecone: {str(e)}")
            return False
    
    def query_chunks(self, ticker, top_k=1000):
        """
        Query Pinecone for chunks related to a ticker
        
        Args:
            ticker: Ticker symbol
            top_k: Maximum number of results
            
        Returns:
            list: List of chunks
        """
        if not self.index:
            logging.error("Pinecone index not initialized")
            return []
        
        try:
            ticker = ticker.upper()
            
            # Query with filter
            response = self.index.query(
                vector=[0] * self.target_dim,  # Dummy vector for metadata filtering
                filter={"ticker": ticker},
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract chunks
            chunks = []
            for match in response.matches:
                chunk = match.metadata
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logging.error(f"Error querying chunks: {str(e)}")
            return []