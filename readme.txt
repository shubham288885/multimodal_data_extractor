# Document Search and Carbon Accounting System

## System Architecture Overview

This system is a comprehensive document processing and retrieval pipeline with carbon accounting capabilities. It combines document ingestion, vector search, and AI-powered question answering to provide intelligent responses to user queries.

### Key Components

![System Architecture](https://i.ibb.co/0JQnXXX/architecture.jpg)

#### 1. Retrieval Pipeline
- **User Query Processing**: Takes user questions and processes them through the system
- **NeMo Retriever Embedding**: Converts queries to vector embeddings using NVIDIA's embedding models
- **Vector Database**: Stores document embeddings in Milvus for efficient similarity search
- **NeMo Retriever Reranking**: Refines search results using NVIDIA's reranking models
- **LLM**: Generates comprehensive answers using DeepSeek-R1 model via NVIDIA NIMS

#### 2. Ingestion Pipeline
- **Document Processing**: Extracts text and visual elements from documents
- **Object Detection**: Identifies charts, tables, and other visual elements using YOLOX
- **Chart Extraction**: Processes charts using DeePlot and CACHED models
- **Table Extraction**: Extracts tabular data using PaddleOCR
- **Post-Processing**: Filters and chunks extracted data for efficient storage
- **Vector Storage**: Embeds and stores document content in Milvus

#### 3. Carbon Accounting
- The system includes specialized carbon accounting capabilities using the DeepSeek-R1 LLM
- Calculates Scope 3 greenhouse gas emissions for user-defined activities
- Provides structured JSON output with detailed emission calculations

## Setup Instructions

### Prerequisites
- Python 3.8+
- NVIDIA API keys for various models
- Milvus Cloud account

### Environment Setup

1. Clone the repository:
bash
git clone https://github.com/yourusername/document-search-system.git
cd document-search-system


2. Create and activate a virtual environment:
bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


3. Install dependencies:
bash
pip install -r requirements.txt


4. Create a `.env` file with your API keys and endpoints:
NVIDIA API Keys
NVIDIA_YOLOX_KEY=your-key-here
NVIDIA_DEPLOT_KEY=your-key-here
NVIDIA_EMBEDDING_KEY=your-key-here
NVIDIA_PADDLEOCR_KEY=your-key-here
NVIDIA_CACHED_KEY=your-key-here
NVIDIA_RERANK_KEY=your-key-here
NVIDIA_LLM_KEY=your-key-here
NVIDIA API Endpoints
NVIDIA_YOLOX_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvidia/nv-yolox-page-elements-v1
NVIDIA_DEPLOT_ENDPOINT=https://ai.api.nvidia.com/v1/vlm/google/deplot
NVIDIA_EMBEDDING_ENDPOINT=https://integrate.api.nvidia.com/v1
NVIDIA_PADDLEOCR_ENDPOINT=https://ai.api.nvidia.com/v1/cv/baidu/paddleocr
NVIDIA_CACHED_ENDPOINT=https://ai.api.nvidia.com/v1/cv/university-at-buffalo/cached
NVIDIA_RERANK_ENDPOINT=https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking
NVIDIA_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
Milvus Cloud Configuration
MILVUS_URI=your-milvus-uri
MILVUS_TOKEN=your-milvus-token


### Setting Up Milvus

1. Set up Milvus Cloud:
bash
python setup_milvus.py


This script will:
- Connect to your Milvus Cloud instance
- Create a collection named "vector_db" with the appropriate schema
- Set up indexing for efficient vector search
- Run a test query to verify the setup

### Running the System

#### 1. Validate API Keys and Endpoints
bash
python -c "from utils.api_validator import validate_api_keys, validate_endpoints; validate_api_keys(); validate_endpoints()"


#### 2. Ingest Documents

bash
python -m core.pipeline.ingestion_pipeline --document_path path/to/your/document.pdf


This will:
- Extract text and visual elements from the document
- Process charts and tables
- Generate embeddings
- Store the content in Milvus

#### 3. Launch the Streamlit App
bash
streamlit run app/frontend/streamlit_app.py


This will start the web interface where you can:
- Enter queries about your documents
- Get AI-generated answers based on document content
- View the relevant document sections that informed the answer

## Carbon Accounting Features

The system includes specialized carbon accounting capabilities:

1. **Process Identification**: Breaks down activities into emission-generating stages
2. **Emission Factor Selection**: Uses region-specific, industry-specific, or global average emission factors
3. **Calculation & Validation**: Computes emissions with detailed calculations
4. **Structured Output**: Returns JSON with granular data and plain-language summaries

To use the carbon accounting features, simply ask questions related to emissions calculations for specific activities or products.

## System Flow

1. **Document Ingestion**:
   - Documents are processed through the ingestion pipeline
   - Text and visual elements are extracted
   - Content is embedded and stored in Milvus

2. **Query Processing**:
   - User enters a question in the Streamlit interface
   - Query is converted to an embedding
   - Similar documents are retrieved from Milvus
   - Results are reranked for relevance
   - DeepSeek-R1 LLM generates a comprehensive answer

3. **Carbon Accounting**:
   - For emissions-related queries, the system uses specialized prompting
   - Calculations follow GHG Protocol and ISO standards
   - Results include detailed breakdowns of emission sources

## Troubleshooting

- **API Connection Issues**: Verify your API keys and endpoints in the .env file
- **Milvus Connection Problems**: Check your Milvus URI and token
- **Document Processing Errors**: Ensure documents are in supported formats (PDF)
- **Memory Issues**: For large documents, consider increasing your system's available memory

## Advanced Configuration

The system can be customized by modifying:

- `core/llm/answer_generator.py`: Adjust LLM parameters and prompting
- `core/pipeline/retrieval_pipeline.py`: Modify search parameters
- `core/vector_store/store.py`: Configure vector database settings
- `app/frontend/streamlit_app.py`: Customize the user interface

## License

[shubham288885]