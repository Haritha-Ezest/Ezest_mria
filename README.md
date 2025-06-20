# MRIA - Medical Records Insight Agent

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7+-orange.svg)](https://spacy.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

MRIA (Medical Records Insight Agent) is a comprehensive medical document processing and knowledge extraction system built with FastAPI. It provides a unified pipeline for medical document ingestion, processing, and graph-based knowledge representation with advanced medical entity recognition capabilities.

## 🏥 Enhanced Medical NER Agent

MRIA now features a **production-ready Enhanced Medical Entity Recognition Agent** that provides:

### 🎯 Comprehensive Medical Entity Extraction
- **Medical Conditions**: Diabetes, Hypertension, CAD, Pneumonia
- **Medications & Dosages**: Metformin 500mg, Lisinopril 10mg, Aspirin 81mg
- **Medical Procedures**: ECG, Cardiac catheterization, Chest X-ray, PCI
- **Laboratory Values**: HbA1c = 8.5%, Glucose = 185 mg/dL, Troponin levels
- **Vital Signs**: Blood pressure, Heart rate, Temperature, Respiratory rate
- **Anatomical References**: Left ventricle, Liver, Kidney, Cardiac arteries
- **Temporal Information**: Dosing frequencies, treatment durations, follow-up schedules

### 🧠 Advanced Medical NLP Models
- **scispaCy**: Scientific and biomedical text processing
- **BioBERT**: Biomedical domain-specific BERT model
- **Med7**: Medical named entity recognition model
- **Clinical BERT**: Clinical text understanding and analysis
- **Multi-model Ensemble**: Combines multiple models for enhanced accuracy

### 🔗 Medical Knowledge Base Integration
- **ICD-10**: International Classification of Diseases coding
- **RxNorm**: Standardized medication nomenclature
- **CPT**: Current Procedural Terminology codes
- **UMLS**: Unified Medical Language System (optional)
- **SNOMED CT**: Systematized Nomenclature of Medicine Clinical Terms (optional)

### ⚡ Processing Capabilities
- **Real-time Processing**: Fast mode for immediate results
- **Batch Processing**: Handle multiple documents efficiently
- **Medical Specialty Detection**: Automatic specialty classification
- **Confidence Scoring**: Quality assessment for extracted entities
- **Temporal Analysis**: Extract and structure temporal medical information
- **Entity Normalization**: Standardize medical terminology

## Architecture

The system follows a multi-stage medical document processing pipeline:

1. **Document Ingestion** - Upload and initial processing of medical documents
2. **OCR Processing** - Extract text from medical images, PDFs, and scanned documents
3. **Enhanced Medical NER** - Advanced medical entity recognition and classification
4. **Entity Linking** - Connect entities to medical knowledge bases (ICD-10, RxNorm, CPT)
5. **Document Chunking** - Split documents for vector storage optimization
6. **Graph Database Integration** - Store structured medical knowledge representations
7. **Supervisor Coordination** - Orchestrate the entire medical workflow

## Features

- 🏥 **Enhanced Medical NER Agent**: Advanced medical entity recognition with multi-model ensemble
- 🚀 **High Performance**: Built with FastAPI for maximum performance  
- 📄 **Multi-format Support**: Handle medical PDFs, images, and text documents
- 🔍 **Advanced OCR**: Tesseract-based text extraction from medical images
- 🧠 **Medical NLP Pipeline**: scispaCy, BioBERT, Med7, and Clinical BERT
- 📊 **Advanced Medical Knowledge Graph**: Enhanced Neo4j integration with comprehensive medical schema
- 🔗 **Knowledge Base Integration**: Real-time linking to ICD-10, SNOMED-CT, RxNorm, and clinical databases  
- ⏰ **Temporal Analysis**: Advanced timeline tracking and medical event correlation
- 🎯 **Clinical Insights**: Patient risk assessment and cross-population pattern analysis
- 🩺 **Clinical Decision Support**: Evidence-based recommendations and drug interaction checking
- 🎯 **Vector Storage**: ChromaDB and FAISS for semantic medical search
- 🔄 **Async Processing**: Non-blocking operations throughout
- 📝 **Comprehensive API**: RESTful endpoints with OpenAPI documentation
- 🛡️ **Production Ready**: Error handling, logging, and monitoring
- � **Advanced Analytics**: Patient timeline analysis, cross-patient patterns, and medical insights

## Technology Stack

- **Framework**: FastAPI 0.104.1+
- **Language**: Python 3.11+
- **Package Manager**: uv (ultra-fast Python package installer)
- **Database**: Neo4j 5.15.0+
- **Vector Store**: ChromaDB 0.4.22+, FAISS
- **OCR**: Tesseract, OpenCV
- **Medical NLP**: 
  - spaCy 3.7+ with scispaCy 0.5.3+
  - Transformers 4.35+ (BioBERT, Clinical BERT)
  - Med7 for medical entity recognition
  - Custom medical models and regex patterns
- **Server**: Uvicorn with async support
- **Code Quality**: Black, flake8, mypy for code formatting and quality

## Prerequisites

- Python 3.11 or higher
- [uv package manager](https://github.com/astral-sh/uv) (recommended)
- Neo4j database instance
- Redis (for caching)
- Tesseract OCR engine

### Installing uv

If you don't have `uv` installed, you can install it using:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

## Quick Start

### Option 1: Docker with UV (Recommended)

The fastest way to get started with MRIA is using Docker with UV optimization:

```bash
# Start development environment with UV
./scripts/docker-uv.sh dev start

# View logs
./scripts/docker-uv.sh dev logs

# Access the API
curl http://localhost:8000/health

# Stop the environment
./scripts/docker-uv.sh dev stop
```

**Services available:**
- MRIA API: http://localhost:8000
- Redis Commander: http://localhost:8081
- API Documentation: http://localhost:8000/docs

For detailed Docker setup instructions, see [DOCKER_UV_SETUP.md](DOCKER_UV_SETUP.md).

### Option 2: Local Development Setup

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd mria
```

#### 2. Automated Setup (Recommended)

Using UV (ultra-fast):
```bash
python setup_local_uv.py
```

Using standard pip:
```bash
python setup_local.py
```

#### 3. Manual Setup

Using `uv` (recommended):

```bash
# Create a virtual environment with Python 3.11+
uv venv --python 3.11

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

#### 4. Medical NER Models Setup

MRIA includes an automated setup script for medical NLP models:

```bash
# Activate your virtual environment first
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Run the medical models setup script
python setup_medical_ner_models.py
```

This script will install and configure:
- **scispaCy models**: Scientific biomedical text processing
- **Medical spaCy models**: en_core_sci_sm, en_core_med7_lg
- **Hugging Face models**: BioBERT, Clinical BERT, Med7
- **Required dependencies**: torch, transformers, scipy, scikit-learn

**Manual installation** (if needed):
```bash
# Install core medical NLP packages
pip install torch transformers scispacy scipy scikit-learn

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_sci_sm

# Install scispaCy models
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_med7_lg-0.5.3.tar.gz
```

#### 5. Environment Configuration

Create a `.env` file in the project root:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
LOG_LEVEL=info

# Security
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Redis Configuration
REDIS_URL=redis://localhost:6379

# API Keys (if required)
OPENAI_API_KEY=your_openai_key
LANGSMITH_API_KEY=your_langsmith_key
```

#### 6. Run the Application

**Using Docker (Recommended):**
```bash
# Development with UV optimization
./scripts/docker-uv.sh dev start

# Or using standard Docker
./scripts/start-docker.sh
```

**Using Local Python:**
```bash
# Development mode with auto-reload (UV)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using the start script
./scripts/start-dev-uv.sh

# Or using standard pip
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- **Application**: <http://localhost:8000>
- **Interactive API docs**: <http://localhost:8000/docs>
- **ReDoc documentation**: <http://localhost:8000/redoc>
- **Redis Commander** (Docker only): <http://localhost:8081>

## API Endpoints

### System Endpoints

- `GET /` - API information and welcome message
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health check with component status

### Core Functionality

- `POST /ingest/upload` - Upload and process documents
- `POST /ocr/process` - OCR processing for images and PDFs
- `POST /ner/extract` - Named entity recognition
- `POST /chunk/process` - Document chunking
- `GET /supervisor/status` - Workflow status and coordination

### Enhanced Knowledge Graph API

**Core Graph Operations:**
- `POST /graph/create` - Create new patient graph with enhanced medical attributes
- `PUT /graph/update/{patient_id}` - Update patient information and relationships
- `GET /graph/patient/{patient_id}` - Retrieve comprehensive patient graph
- `POST /graph/query` - Execute advanced Cypher queries with clinical context
- `GET /graph/schema` - Retrieve complete graph schema with constraints
- `GET /graph/info` - Get database statistics and health information

**Advanced Medical Analytics:**
- `POST /graph/temporal` - Create and manage temporal relationships between medical events
- `GET /graph/timeline/{patient_id}` - Generate detailed patient timeline with medical event correlation
- `GET /graph/patterns/{condition}` - Discover cross-patient patterns for specific medical conditions
- `POST /graph/knowledge/expand` - Expand knowledge base with external medical data integration
- `GET /graph/insights/{patient_id}` - Generate comprehensive patient insights and risk assessments

## Development

### Using uv for Development

```bash
# Install development dependencies
uv pip install -r requirements-dev.txt

# Run tests
uv run pytest

# Format code
uv run black .

# Lint code
uv run flake8 .

# Type checking
uv run mypy .
```

### Project Structure

```text
mria/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── common/
│   │   └── utils.py         # Shared utilities
│   ├── routers/             # API route handlers
│   │   ├── chunking.py
│   │   ├── graph.py
│   │   ├── ingestion.py
│   │   ├── ner.py
│   │   ├── ocr.py
│   │   └── supervisor.py
│   ├── schemas/             # Pydantic models
│   │   ├── chunking.py
│   │   ├── graph.py
│   │   ├── ingestion.py
│   │   ├── ner.py
│   │   ├── ocr.py
│   │   └── supervisor.py
│   └── services/            # Business logic
│       ├── chunker.py
│       ├── file_handler.py
│       ├── graph_client.py
│       ├── ner_processor.py
│       ├── ocr_processor.py
│       ├── supervisor.py
│       └── vector_store.py
├── tests/                   # Test files
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── .env.example            # Environment variables template
├── .gitignore
└── README.md
```

### Adding New Dependencies

With `uv`, adding new dependencies is fast and efficient:

```bash
# Add a new package
uv pip install package-name

# Add to requirements.txt
echo "package-name==version" >> requirements.txt

# Or install and add to requirements in one step
uv add package-name
```

## Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app

# Run specific test file
uv run pytest tests/test_ingestion.py

# Run with verbose output
uv run pytest -v
```

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Manual Deployment

```bash
# Install production dependencies
uv pip install -r requirements.txt --no-dev

# Run with production settings
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Performance Optimization

- **Async Operations**: All I/O operations are asynchronous
- **Connection Pooling**: Database connections are pooled
- **Caching**: Redis caching for frequently accessed data
- **Batch Processing**: Efficient batch operations for large datasets
- **Memory Management**: Streaming for large file processing

## Monitoring and Logging

- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Health Checks**: Comprehensive health monitoring endpoints
- **Metrics**: Request timing and performance metrics
- **Error Tracking**: Detailed error logging and stack traces

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Ensure all tests pass (`uv run pytest`)
6. Format code (`uv run black .`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:

- Create an issue in the GitHub repository
- Check the API documentation at `/docs`
- Review the health check endpoints for system status

## Changelog

### v0.1.0 (Current)

- Initial release with core pipeline functionality
- FastAPI-based REST API
- Multi-stage document processing
- Graph database integration
- Comprehensive error handling and logging
