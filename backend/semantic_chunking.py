import re
import logging
import traceback
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logging.warning("Sentence Transformers not installed. Install with: pip install sentence-transformers")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("semantic_chunker.log"), logging.StreamHandler()]
)

class SemanticChunker:
    """Class for semantically chunking SEC filing text based on meaning and topic changes"""
    
    def __init__(self, max_chunk_size=1000, overlap=100, similarity_threshold=0.7):
        """
        Initialize semantic chunker with configurable parameters
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            similarity_threshold: Threshold for semantic similarity (0-1)
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        
        # Initialize embedding model
        self.embedding_model = None
        self.model_initialized = False
        
        # Try to initialize model (lazy loading)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        if self.model_initialized:
            return
            
        if ST_AVAILABLE:
            try:
                model_name = "all-MiniLM-L6-v2"  # Small, fast model
                self.embedding_model = SentenceTransformer(model_name)
                self.model_initialized = True
                logging.info(f"Initialized Sentence Transformer model: {model_name}")
            except Exception as e:
                logging.error(f"Error loading Sentence Transformer model: {str(e)}")
    
    def clean_text(self, text):
        """
        Clean text for processing
        
        Args:
            text: Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
        
        # Remove javascript void links
        text = re.sub(r'\[([^\]]+)\]\(javascript:void\\\(0\\\);\)', r'\1', text)
        
        # Clean up table formatting
        text = re.sub(r'\n\s*\|[-|]+\|\s*\n', '\n\n', text)
        text = re.sub(r'\|\s*[-:]+\s*\|', ' ', text)
        text = re.sub(r'\|\s*\|', ' ', text)
        
        # Fix excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def process_text(self, text):
        """
        Process and chunk text semantically
        
        Args:
            text: Text to process
            
        Returns:
            list: List of semantic chunks
        """
        return self.chunk_text(text)
    
    def chunk_text(self, text):
        """
        Split text into semantically meaningful chunks based on topic changes
        
        Args:
            text: Text to chunk
            
        Returns:
            list: List of semantic chunks
        """
        try:
            # Clean the text
            text = self.clean_text(text)
            
            # If text is empty or very small, return as is
            if not text or len(text) <= self.max_chunk_size:
                return [text] if text else []
            
            # First split by structural boundaries to get initial segments
            segments = self._split_by_structure(text)
            
            # If we can do semantic chunking and have enough segments
            if ST_AVAILABLE and self.embedding_model and len(segments) > 3:
                return self._semantic_chunking(segments)
            else:
                # Fall back to structural chunking
                return self._ensure_chunk_sizes(segments)
            
        except Exception as e:
            logging.error(f"Error in chunk_text: {str(e)}\n{traceback.format_exc()}")
            # Fallback to simple chunking
            return self._simple_chunk_text(text)
    
    def _split_by_structure(self, text):
        """
        Split text by structural elements to get initial segments
        
        Args:
            text: Text to split
            
        Returns:
            list: List of text segments
        """
        # Try to split by paragraphs first
        segments = re.split(r'\n\s*\n', text)
        
        # If segments are too large, split further by sentences
        smaller_segments = []
        for segment in segments:
            if len(segment) > self.max_chunk_size:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', segment)
                smaller_segments.extend(sentences)
            else:
                smaller_segments.append(segment)
        
        # Remove empty segments and very small segments
        segments = [seg.strip() for seg in smaller_segments if len(seg.strip()) > 10]
        
        return segments
    
    def _simple_chunk_text(self, text):
        """
        Fallback method to chunk text when semantic chunking fails
        
        Args:
            text: Text to chunk
            
        Returns:
            list: List of text chunks
        """
        # If text is small enough, return as a single chunk
        if len(text) <= self.max_chunk_size:
            return [text]
        
        # Split by character with overlap
        chunks = []
        for i in range(0, len(text), self.max_chunk_size - self.overlap):
            end = min(i + self.max_chunk_size, len(text))
            chunks.append(text[i:end])
            
            if end == len(text):
                break
        
        return chunks
    
    def _ensure_chunk_sizes(self, segments):
        """
        Combine segments into chunks that respect size limits
        
        Args:
            segments: List of text segments
            
        Returns:
            list: List of appropriately sized chunks
        """
        chunks = []
        current_chunk = ""
        
        for segment in segments:
            # If adding this segment would exceed max size
            if len(current_chunk) + len(segment) > self.max_chunk_size:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If segment itself is too large, split it
                if len(segment) > self.max_chunk_size:
                    # Split the segment by character
                    for i in range(0, len(segment), self.max_chunk_size - self.overlap):
                        end = min(i + self.max_chunk_size, len(segment))
                        chunks.append(segment[i:end])
                    current_chunk = ""
                else:
                    current_chunk = segment
            else:
                # Add segment to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + segment
                else:
                    current_chunk = segment
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _semantic_chunking(self, segments):
        """
        Combine segments into chunks based on semantic similarity
        
        Args:
            segments: List of text segments
            
        Returns:
            list: List of semantic chunks
        """
        if len(segments) <= 1:
            return segments
        
        # Create embeddings for each segment
        try:
            # Make sure model is initialized
            self._initialize_model()
            
            # Encode segments
            embeddings = self.embedding_model.encode(segments)
        except Exception as e:
            logging.error(f"Error creating embeddings: {str(e)}")
            return self._ensure_chunk_sizes(segments)
        
        # Find semantic boundaries (where topics change)
        boundaries = [0]  # Always include start
        
        for i in range(1, len(segments) - 1):
            # Calculate similarity with previous and next segment
            prev_similarity = cosine_similarity(
                [embeddings[i]], [embeddings[i-1]]
            )[0][0]
            
            next_similarity = cosine_similarity(
                [embeddings[i]], [embeddings[i+1]]
            )[0][0]
            
            # If similarity drops significantly, this is a topic boundary
            if ((prev_similarity > self.similarity_threshold and 
                 next_similarity < self.similarity_threshold) or
                (prev_similarity - next_similarity > 0.2)):
                boundaries.append(i)
                logging.debug(f"Topic boundary detected at segment {i}: prev_sim={prev_similarity:.2f}, next_sim={next_similarity:.2f}")
        
        boundaries.append(len(segments))  # Always include end
        
        # Create chunks based on identified boundaries
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for i in range(len(boundaries) - 1):
            # Combine all segments between boundaries
            boundary_text = "\n\n".join(segments[boundaries[i]:boundaries[i+1]])
            boundary_length = len(boundary_text)
            
            # If this semantic unit is too large by itself, split it
            if boundary_length > self.max_chunk_size:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_length = 0
                
                # Split large semantic unit into smaller chunks
                boundary_chunks = self._simple_chunk_text(boundary_text)
                chunks.extend(boundary_chunks)
            
            # If adding this semantic unit would exceed max size
            elif current_length + boundary_length > self.max_chunk_size:
                # Add current chunk and start a new one
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = boundary_text
                current_length = boundary_length
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + boundary_text
                    current_length += boundary_length + 4  # account for newlines
                else:
                    current_chunk = boundary_text
                    current_length = boundary_length
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        logging.info(f"Created {len(chunks)} semantic chunks from {len(segments)} segments")
        return chunks