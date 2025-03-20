# Multimodal PDF Extractor with Carbon Accounting

## System Architecture Overview

This system is a comprehensive document processing, retrieval, and carbon accounting solution. It combines document ingestion, vector search, and AI-powered question answering with greenhouse gas emissions calculation capabilities.

### Key Components

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
- **Activity Extraction**: Identifies emission-relevant activities from documents using DeepSeek-R1
- **Emission Factor Lookup**: Finds appropriate emission factors through semantic search
- **Emissions Calculation**: Calculates Scope 1, 2, and 3 emissions using DeepSeek-R1
- **Structured Output**: Generates detailed JSON reports with emissions breakdowns
- **Visualization**: Displays emissions results in an interactive UI

## Setup Instructions

### Prerequisites
- Python 3.8+
- NVIDIA API keys for various models
- Milvus Cloud account
- Access to Emission Factors API

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-pdf-extractor.git
cd multimodal-pdf-extractor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your `.env` file with your API keys and endpoints:
```
# NVIDIA API Keys
NVIDIA_YOLOX_KEY=your-key-here
NVIDIA_DEPLOT_KEY=your-key-here
# ... other API keys

# NVIDIA API Endpoints
NVIDIA_YOLOX_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvidia/nv-yolox-page-elements-v1
# ... other endpoints

# Milvus Cloud Configuration
MILVUS_URI=your-milvus-uri
MILVUS_TOKEN=your-milvus-token

# Emission Factors API
EMISSION_FACTORS_API_URL=http://localhost:8000/api/v1/emission-factors/search
```

### Setting Up Milvus

Set up Milvus Cloud:
```bash
python setup_milvus.py
```

### Running the System

1. Launch the Streamlit App:
```bash
streamlit run app/frontend/streamlit_app.py
```

## Features and Usage

### Document Search

1. Upload PDF documents
2. Enter natural language queries about the documents
3. View AI-generated answers based on document content

### Carbon Emissions Calculation

1. Upload bills or documents with emission-relevant activities
2. The system automatically extracts activities that could generate emissions
3. View detailed emissions breakdown by source and process
4. Download results in structured JSON format

## Emissions Calculation Methodology

The system follows these steps to calculate emissions:

1. **Document Analysis**: Extracts text and visual elements from documents
2. **Activity Identification**: Uses AI to identify emission-relevant activities
3. **Emission Factor Selection**: Finds appropriate emission factors via semantic search
4. **Calculation**: Computes emissions following GHG Protocol and IPCC guidelines
5. **Structured Output**: Provides detailed breakdown of emissions by source and process

### Emission Result Format

```json
{
  "activity_description": "Purchase of 20 kg plastic bags (20 km transport)",
  "emission_sources": [
    {
      "source": "production",
      "processes": [
        {
          "name": "polyethylene_production",
          "description": "Crude oil refining, polymerization, and bag manufacturing",
          "parameters": {
            "quantity": "20 kg",
            "emission_factor": "1.5 kg CO2e/kg (IPCC 2023, Plastics Manufacturing)",
            "calculation": "20 kg × 1.5 kg CO2e/kg = 30 kg CO2e",
            "total_emissions": 30.0
          }
        }
      ],
      "total_emissions": 32.3
    }
  ],
  "total_scope_3_emissions": 43.1,
  "assumptions": [
    "Defaulted to landfill disposal (emission factor: 0.1 kg CO2e/kg)."
  ],
  "data_sources": [
    "IPCC 2023: Plastics Production Emission Factors"
  ]
}
```

## Troubleshooting

- **API Connection Issues**: Verify your API keys and endpoints in the .env file
- **Milvus Connection Problems**: Check your Milvus URI and token
- **Document Processing Errors**: Ensure documents are in supported formats (PDF)
- **Emission Calculation Errors**: Check connection to the Emission Factors API 