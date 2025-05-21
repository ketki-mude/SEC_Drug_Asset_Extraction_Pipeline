# SEC Drug Asset Extraction Pipeline

This project implements an automated system for analyzing SEC filings to extract and structure information about drug development pipelines for biotech companies. The system combines RAG (Retrieval Augmented Generation) with Google's Gemini Pro LLM to process unstructured SEC filing data and generate comprehensive drug pipeline analyses.

## 📌 Project Resources

- **Streamlit App**: [Application Link](http://34.134.167.79:8501)
- **Documentation**: [Project Documentation](https://codelabs-preview.appspot.com/?file_id=1WCRLM8Uw9uFklAa_8tjpBmvVDP7hKa-UW-vBwFqVFvQ#3)

## 📌 Technologies Used

- Apache Airflow
- AWS S3
- Pinecone Vector DB
- Google Gemini Pro
- Streamlit
- Docker
- Python

## 📌 Project Flow

### Step 1: User Input
- User enters a ticker symbol via Streamlit interface
- System validates if the ticker represents a biotech company

### Step 2: Data Processing Pipeline
1. **Check Existing Results**
   - Query S3 for processed results
   - If found, display directly to user

2. **Vector Database Check**
   - Query Pinecone for existing vector data
   - If found, proceed to LLM extraction

3. **Raw Data Check**
   - Check S3 for raw SEC filings
   - Process to Pinecone if found
   - Otherwise, fetch new data from SEC EDGAR

4. **LLM Processing**
   - Extract drug information using Gemini Pro
   - Structure data into standardized format
   - Store results in S3

### Step 3: Output Generation
- Markdown formatted drug pipeline analysis
- Structured CSV data
- Interactive Streamlit display

## 📌 Key Components

### Frontend (Streamlit)
- Ticker input form
- Real-time processing status
- Results display with download options

### Backend Processing
1. **SEC Filing Extraction**
   - Fetches filings from EDGAR
   - Stores raw data in S3

2. **Vector Processing**
   - Semantic chunking of SEC texts
   - Vector embedding generation
   - Pinecone storage and retrieval

3. **LLM Analysis**
   - Context-aware prompting
   - Structured information extraction
   - Multi-format output generation

### Data Storage
- **S3**: Raw filings and processed results
- **Pinecone**: Vector embeddings for RAG
- **Local**: Temporary processing files

## 📌 Setup Instructions

1. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Configuration**
   Create a `.env` file with:
   ```env
   OPENAI_API_KEY=your_key
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_key
   PINECONE_API_KEY=your_key
   ```

3. **Running the Pipeline**

   # Launch Streamlit
   streamlit run app.py
   ```


