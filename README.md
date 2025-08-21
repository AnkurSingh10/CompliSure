# 📚 CompliSure: Intelligent Insurance and Legal Assistant

This project is a sophisticated Retrieval-Augmented Generation (RAG) API built with `FastAPI`. It uses `FAISS` for vector storage, `HuggingFaceEmbeddings` for creating embeddings, and hybrid retrieval combining semantic and keyword search. The server accepts document URLs and questions, processes multiple document formats, and returns AI-generated answers using Google `Gemini` API with advanced prompt engineering.

## 🗂️ Project Structure

```text
.
├── app.py                  # FastAPI app entry point with async lifespan management
├── utils/
│   ├── DocsLoader.py      # Multi-format document loader (PDF/DOCX/PPT/Excel/Images/TXT)
│   ├── Schemas.py         # Pydantic schemas (request/response models)
│   └── __init__.py        
├── .env                   # Environment variables (API keys)
├── Dockerfile             # Docker container configuration
├── docker-compose.yml     # Docker Compose orchestration
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── __pycache__/          # Python cache files
```

## 🚀 Features

- **Multi-format Document Processing**: Supports PDF, DOCX, PPT, Excel, JPG, PNG, TXT, and more
- **Hybrid Retrieval System**: Combines semantic search (FAISS) with keyword search (BM25)
- **Advanced AI Models**:
  - Embeddings: `BAAI/bge-base-en-v1.5` (HuggingFace)
  - LLM: `Gemini-2.5-Flash` (Google Generative AI)
  - Reranker: `BAAI/bge-reranker-base` (optional, commented for CPU optimization)
- **Document Caching**: In-memory vectorstore caching for faster responses
- **Authentication**: Bearer token-based API authentication
- **Async Processing**: Concurrent question processing with asyncio
- **Special Handlers**: Custom logic for specific document types and use cases
- **Chain-of-Thought Prompting**: Advanced prompt engineering for better responses

## 🔧 Setup Instructions

### Prerequisites

- Docker and Docker Compose installed
- Google Gemini API key

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 2. Create and configure the `.env` file

In the project root, create a `.env` file with your credentials:

```env
# Primary Gemini API key for main LLM operations
gemini_api_key3=your-google-gemini-api-key

# Secondary Gemini API key for document processing
gemini_api_key2=your-google-gemini-api-key

# Team API key for HackRX authentication
TEAM_API_KEY=your-team-api-token-by-hackrx
```

### 3. Build and run with Docker Compose

```bash
# Build and start the application
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

The application will be available at: [http://localhost:5000](http://localhost:5000)

### 4. Alternative: Direct Docker build

```bash
# Build the image
docker build -t policy-chatbot .

# Run the container
docker run -p 5000:7860 --env-file .env policy-chatbot
```

## 🧪 Running the Application

### Using Docker Compose (Recommended)

```bash
# Start the application
docker-compose up --build
```

The application will be available at: [http://localhost:5000](http://localhost:5000)

### Health Check

```bash
# Check if the API is running
curl http://localhost:5000/
```

## 📤 API Endpoints

### `POST /api/v1/hackrx/run`

**Headers:**

```http
Authorization: Bearer <your-team-api-key>
Content-Type: application/json
```

**Request:**

```json
{
  "documents": "https://example.com/mydoc.pdf",
  "questions": [
    "What is the document about?", 
    "What are the key policy terms?",
    "Summarize the coverage details."
  ]
}
```

**Response:**

```json
{
  "answers": [
    "The document discusses insurance policy terms and conditions...",
    "Key policy terms include premium payments, coverage limits...",
    "Coverage includes medical expenses up to $50,000..."
  ]
}
```

### `GET /`

Health check endpoint that returns API status.

## 🔑 Authentication

The API uses Bearer token authentication. Include your team API key in the Authorization header:

```bash
curl -X POST "http://localhost:5000/api/v1/hackrx/run" \
  -H "Authorization: Bearer your-team-api-key" \
  -H "Content-Type: application/json" \
  -d '{"documents": "https://example.com/doc.pdf", "questions": ["What is this about?"]}'
```

## 📦 Key Dependencies

### Core Framework

- `fastapi==0.115.13` - Modern web API framework
- `uvicorn[standard]==0.34.3` - ASGI server

### AI & Machine Learning

- `langchain==0.3.26` - LLM framework and chains
- `langchain-community==0.3.27` - Community integrations
- `langchain-google-genai==2.1.8` - Google Gemini AI integration
- `langchain-huggingface==0.3.1` - HuggingFace model integrations
- `sentence-transformers==4.1.0` - Embedding models
- `faiss-cpu==1.7.4` - Vector similarity search
- `rank-bm25==0.2.2` - Keyword search algorithm

### Document Processing

- `pymupdf==1.24.5` - PDF processing
- `python-docx==1.1.2` - DOCX file handling
- `python-pptx==0.6.23` - PowerPoint file processing
- `openpyxl==3.1.2` - Excel file handling
- `pytesseract==0.3.10` - OCR for image text extraction
- `pillow==10.3.0` - Image processing
- `unstructured[pytesseract]==0.12.5` - Advanced document parsing

### Utilities

- `requests==2.32.4` - HTTP client
- `beautifulsoup4==4.12.2` - HTML/XML parsing
- `pandas==2.2.2` - Data manipulation
- `numpy==1.26.4` - Numerical computing
- `scikit-learn==1.4.2` - Machine learning utilities

## 🏗️ Architecture

### Document Processing Pipeline

1. **Document Loading**: Multi-format document ingestion via URLs
2. **Text Extraction**: OCR, parsing, and text extraction from various formats
3. **Chunking**: Intelligent text splitting for optimal retrieval
4. **Embedding Generation**: Convert text chunks to vector embeddings
5. **Vector Storage**: Store embeddings in FAISS for fast similarity search

### Retrieval System

- **Hybrid Approach**: Combines semantic (FAISS) and keyword (BM25) search
- **Ensemble Retriever**: Weighted combination (70% semantic, 30% keyword)
- **Reranking**: Optional cross-encoder reranking for improved relevance
- **Caching**: In-memory vectorstore caching for performance

### AI Processing

- **Primary LLM**: Gemini-2.5-Flash for question answering
- **Secondary LLM**: Gemini-2.0-Flash for document processing
- **Advanced Prompting**: Chain-of-thought reasoning with critique and revision

## 🔧 Configuration

### Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `gemini_api_key3` | Primary Gemini API key for main operations | Yes |
| `gemini_api_key2` | Secondary Gemini API key for document processing | Yes |
| `TEAM_API_KEY` | HackRX authentication token | Yes |
| `PINECONE_API_KEY` | Pinecone API key (currently unused) | No |

### Docker Configuration

- **Port Mapping**: Host port 5000 → Container port 7860
- **Container Name**: `myapp`
- **Restart Policy**: `unless-stopped`
- **Environment**: Loaded from `.env` file

## 📊 Performance Features

- **Async Processing**: Concurrent handling of multiple questions
- **Model Preloading**: ML models loaded during application startup
- **Document Caching**: Vectorstores cached in memory for repeated requests
- **Efficient Retrieval**: Optimized search parameters (k=14 for dense, k=9 for sparse)

## 🚨 Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required environment variables are set
2. **Port Conflicts**: Change host port in `docker-compose.yml` if 5000 is occupied
3. **Memory Issues**: Large documents may require more RAM for processing
4. **Model Downloads**: First run downloads embedding models to `/tmp/e5-large-v2`

### Debugging

```bash
# View application logs
docker-compose logs -f app

# Check container status
docker-compose ps

# Rebuild after changes
docker-compose down && docker-compose up --build
```

## 🤝 Contributing

This project was developed for HackRX 6.0. The codebase includes advanced RAG techniques, hybrid retrieval systems, and optimized document processing for policy and insurance document analysis.
