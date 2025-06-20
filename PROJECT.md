# Medical Records Insight Agent (MRIA) - Project Documentation

## Overview

The Medical Records Insight Agent (MRIA) is an AI-powered agentic system designed to parse, understand, and extract actionable insights from diverse and unstructured medical records. The system supports multiple input formats including Electronic Health Records (EHRs), lab reports, radiology images, and handwritten prescriptions, enabling medical professionals to query, explore, and analyze patient data semantically across time and context.

## Project Information

- **Project Type**: Medical AI System / Document Processing Pipeline
- **Architecture**: Agent-Based Multi-Stage Processing
- **Framework**: FastAPI with Python 3.11+
- **Version**: 0.1.0
- **Date**: June 2025
- **License**: MIT

## Use Case

### Objective

Build an AI-powered agentic system capable of:

- Processing diverse and unstructured medical records
- Extracting structured medical knowledge and insights
- Enabling semantic queries across patient data
- Supporting medical professionals in data-driven decision making
- Providing temporal analysis of patient health progression

### Target Users

- **Medical Professionals**: Doctors, nurses, medical researchers
- **Healthcare Administrators**: Hospital administrators, clinic managers
- **Medical Researchers**: Clinical researchers, epidemiologists
- **Healthcare Data Analysts**: Medical data scientists, health informatics specialists

### Business Value

- **Improved Patient Care**: Faster access to comprehensive patient history
- **Efficiency Gains**: Reduced time spent searching through medical records
- **Data-Driven Insights**: Pattern recognition across patient populations
- **Compliance Support**: Structured data for regulatory reporting
- **Research Enablement**: Aggregated anonymized data for medical research

## System Architecture

The MRIA system follows an agent-based architecture with specialized components handling different aspects of medical record processing:

### Core Architectural Principles
- **Multi-Stage Pipeline**: Sequential processing with specialized agents
- **Microservices Pattern**: Modular design with clear separation of concerns
- **Async Processing**: Non-blocking operations for scalability
- **Event-Driven**: Agent coordination through workflow events
- **Graph-Based Knowledge**: Neo4j for complex medical relationships
- **Vector Storage**: Semantic search capabilities for medical content

### Technology Stack

#### Core Framework
- **FastAPI 0.104.1**: High-performance web framework for APIs
- **Python 3.11+**: Modern Python with type hints and async support
- **Uvicorn**: ASGI server with async capabilities
- **Pydantic 2.5.2**: Data validation and settings management

#### AI/ML Libraries
- **LangChain 0.1.0**: Agent orchestration and LLM integration
- **LangGraph 0.0.20**: Workflow management for agent coordination
- **Transformers 4.36.2**: State-of-the-art NLP models
- **SpaCy 3.7.2**: Industrial-strength NLP processing
- **Sentence-Transformers 2.2.2**: Semantic embeddings

#### Databases & Storage
- **Neo4j 5.15.0**: Graph database for medical knowledge representation
- **ChromaDB 0.4.22**: Vector database for semantic search
- **FAISS 1.7.4**: Efficient similarity search and clustering
- **Redis 5.0.1**: Caching and session management

#### Document Processing
- **Tesseract (pytesseract 0.3.10)**: OCR for text extraction
- **OpenCV 4.8.1.78**: Image processing and computer vision
- **PDF2Image 1.16.3**: PDF to image conversion
- **Pillow 10.1.0**: Python Imaging Library
- **python-magic 0.4.27**: File type detection

#### Data Processing
- **Pandas 2.1.4**: Data manipulation and analysis
- **NumPy 1.25.2**: Numerical computing
- **aiofiles 23.2.0**: Async file operations

## Agent Architecture

### 1. Supervisor Agent
**Role**: Orchestrates the entire pipeline and manages agent coordination

**Responsibilities**:
- Workflow orchestration for each new record upload
- Agent lifecycle management and task distribution
- Error handling, retry logic, and failure recovery
- Progress tracking and status reporting
- Resource allocation and load balancing

**Technologies Used**:
- LangChain/LangGraph for workflow management
- Redis for state management
- FastAPI for REST endpoints

**Key Endpoints**:
- `POST /supervisor/enqueue` - Queue new processing job
- `GET /supervisor/status/{job_id}` - Check job status
- `POST /supervisor/retry/{job_id}` - Retry failed jobs

### 2. OCR + Ingestion Agent
**Role**: Handles document upload and text extraction from various formats

**Input Types Supported**:
- Scanned PDFs (medical reports, prescriptions)
- JPG/PNG images (handwritten notes, lab results)
- DOCX documents (structured reports)
- Excel files (lab results, patient data)

**Processing Capabilities**:
- Multi-format file detection and validation
- OCR text extraction with positional data preservation
- Image preprocessing for better OCR accuracy
- Text cleaning and normalization
- Metadata extraction (creation date, source, etc.)

**Technologies Used**:
- Tesseract OCR engine
- OpenCV for image preprocessing
- PDF2Image for PDF processing
- python-magic for file type detection

**Key Endpoints**:
- `POST /ingest/upload` - Upload medical documents
- `GET /ingest/status/{upload_id}` - Check ingestion status
- `POST /ocr/extract` - Extract text from images/PDFs

### 3. Medical Entity Recognition (NER) Agent
**Role**: Identifies and extracts medical entities from processed text

**Medical Entities Extracted**:
- **Conditions/Diagnoses**: "Type 2 Diabetes Mellitus", "Hypertension"
- **Medications**: "Metformin 500mg", "Lisinopril 10mg"
- **Procedures**: "ECG", "Chest X-ray", "Blood glucose test"
- **Symptoms**: "Blurred vision", "Frequent urination", "Fatigue"
- **Lab Values**: "HbA1c = 7.8%", "Glucose = 165 mg/dL"
- **Anatomical References**: "Left ventricle", "Liver", "Kidney"
- **Temporal Information**: "6 months ago", "daily", "twice weekly"

**NLP Models Used**:
- **Medical-specific models**: scispaCy, BioBERT, Med7
- **Custom fine-tuned models**: Domain-specific entity recognition
- **Clinical BERT variants**: For medical context understanding

**Key Endpoints**:
- `POST /ner/extract` - Extract medical entities from text
- `GET /ner/entities/{doc_id}` - Retrieve extracted entities
- `POST /ner/validate` - Validate entity extraction results

### 4. Chunking & Timeline Structuring Agent
**Role**: Organizes extracted information into meaningful temporal structures

**Chunking Strategy**:
- **Visit-based chunking**: Group information by medical visits/encounters
- **Topic-based chunking**: Group related medical information together
- **Temporal chunking**: Organize information chronologically
- **Semantic chunking**: Use embeddings to group related content

**Timeline Structure**:
```
Patient Timeline:
├── Visit 1 (2024-01-15)
│   ├── Symptoms: ["fatigue", "increased thirst"]
│   ├── Tests: ["HbA1c: 8.2%", "Fasting glucose: 180 mg/dL"]
│   └── Diagnosis: ["Type 2 Diabetes Mellitus"]
├── Visit 2 (2024-03-15)
│   ├── Medications: ["Metformin 500mg BID"]
│   ├── Tests: ["HbA1c: 7.1%"]
│   └── Progress: ["Improved glucose control"]
└── Visit 3 (2024-06-15)
    ├── Tests: ["HbA1c: 6.8%"]
    └── Status: ["Target HbA1c achieved"]
```

**Key Endpoints**:
- `POST /chunk/process` - Process document into chunks
- `GET /chunk/timeline/{patient_id}` - Get patient timeline
- `POST /chunk/structure` - Create structured medical timeline

### 5. Knowledge Graph Builder Agent
**Role**: Converts structured medical data into Neo4j graph representations with advanced analytics capabilities

**Enhanced Graph Schema**:
```cypher
// Core Entities with Enhanced Attributes
(Patient:Person {
  id, name, dob, gender, mrn, phone, email, address,
  emergency_contact, insurance_info, preferred_language,
  created_at, updated_at, active_status
})

(Visit:MedicalEncounter {
  id, patient_id, date, type, location, provider, status,
  duration, chief_complaint, visit_notes, follow_up_required,
  created_at, updated_at
})

(Condition:MedicalCondition {
  id, name, icd_code, severity, status, onset_date, resolution_date,
  clinical_notes, family_history, chronic_flag, risk_factors,
  knowledge_base_id, external_refs, created_at, updated_at
})

(Medication:Drug {
  id, name, dosage, frequency, route, start_date, end_date,
  prescribing_provider, indication, side_effects, contraindications,
  drug_interactions, knowledge_base_id, external_refs,
  created_at, updated_at
})

(Test:LabTest {
  id, name, value, unit, reference_range, status, ordered_date,
  resulted_date, ordering_provider, lab_name, interpretation,
  critical_flag, knowledge_base_id, external_refs,
  created_at, updated_at
})

(Procedure:MedicalProcedure {
  id, name, cpt_code, description, date, provider, location,
  outcome, complications, follow_up_required, anesthesia_type,
  knowledge_base_id, external_refs, created_at, updated_at
})

// Enhanced Relationships with Temporal and Contextual Data
(Patient)-[:HAS_VISIT {relationship_type: "primary_care" | "specialist" | "emergency"}]->(Visit)
(Visit)-[:DIAGNOSED_WITH {confidence: float, date: datetime, provider: string}]->(Condition)
(Visit)-[:PRESCRIBED {indication: string, date: datetime, provider: string}]->(Medication)
(Visit)-[:PERFORMED {indication: string, date: datetime, provider: string}]->(Test)
(Visit)-[:UNDERWENT {indication: string, date: datetime, provider: string}]->(Procedure)
(Condition)-[:TREATED_WITH {start_date: datetime, end_date: datetime, effectiveness: string}]->(Medication)
(Test)-[:INDICATES {confidence: float, date: datetime, interpretation: string}]->(Condition)
(Condition)-[:FOLLOWS {temporal_relationship: "before" | "after" | "concurrent"}]->(Condition)
(Medication)-[:INTERACTS_WITH {interaction_type: string, severity: string}]->(Medication)
```

**Advanced Graph Operations**:
- **Enhanced Patient Record Management**: Create and update patient graphs with rich medical attributes
- **Temporal Relationship Analysis**: Track medical events across time with sophisticated temporal modeling
- **Cross-Patient Pattern Mining**: Identify patterns across patient populations for comparative analysis
- **Knowledge Base Integration**: Link medical entities to external knowledge bases and clinical databases
- **Patient Timeline Analysis**: Generate comprehensive patient health timelines with event correlation
- **Medical Insights Generation**: Advanced analytics for clinical decision support and risk assessment

**Enhanced API Endpoints**:

*Core Operations:*
- `POST /graph/create` - Create new patient graph with enhanced attributes
- `PUT /graph/update/{patient_id}` - Update patient information and relationships
- `GET /graph/patient/{patient_id}` - Retrieve comprehensive patient graph
- `POST /graph/query` - Execute advanced Cypher queries with clinical context
- `GET /graph/schema` - Retrieve complete graph schema with constraints
- `GET /graph/info` - Get database statistics and health information

*Advanced Analytics:*
- `POST /graph/temporal` - Create and manage temporal relationships between medical events
- `GET /graph/timeline/{patient_id}` - Generate detailed patient timeline with medical event correlation
- `GET /graph/patterns/{condition}` - Discover cross-patient patterns for specific medical conditions
- `POST /graph/knowledge/expand` - Expand knowledge base with external medical data integration
- `GET /graph/insights/{patient_id}` - Generate comprehensive patient insights and risk assessments

**Knowledge Base Integration**:
- External medical database linking (ICD-10, SNOMED-CT, RxNorm)
- Clinical decision support system integration
- Medical literature and research paper references
- Drug interaction and contraindication databases
- Medical device and equipment catalogs

### 6. Insight Agent
**Role**: Analyzes patterns and generates medical insights across patient data

**Analysis Capabilities**:
- **Treatment Progression Analysis**: Track medication effectiveness over time
- **Comparative Analysis**: Compare similar patients and outcomes
- **Risk Factor Identification**: Identify potential health risks
- **Treatment Response Patterns**: Analyze response to specific treatments
- **Population Health Insights**: Aggregate trends across patient groups

**Insight Types**:
- **Individual Patient Insights**: Personal health progression and predictions
- **Cohort Analysis**: Group-level patterns and trends
- **Treatment Efficacy**: Medication and procedure effectiveness
- **Risk Stratification**: Patient risk categorization
- **Clinical Decision Support**: Evidence-based recommendations

**Key Endpoints**:
- `POST /insights/generate/{patient_id}` - Generate patient insights
- `POST /insights/compare` - Compare multiple patients
- `GET /insights/population/{condition}` - Population-level insights
- `POST /insights/recommendations/{patient_id}` - Clinical recommendations

### 7. Query/Chat Agent
**Role**: Provides natural language interface for medical professionals

**Query Capabilities**:
- **Natural Language Processing**: Understand medical terminology and context
- **Multi-modal Queries**: Support text, voice, and structured queries
- **Contextual Understanding**: Maintain conversation context and medical history
- **Domain-specific Knowledge**: Leverage medical knowledge bases

**Supported Query Types**:
- Patient history summaries
- Treatment comparisons
- Medication interactions
- Lab result interpretations
- Population health queries
- Clinical decision support

**Example Queries**:
- "Summarize this patient's diabetes progression over the last year"
- "Compare this patient's HbA1c trends with similar patients"
- "What are the medication interactions for this patient's current prescriptions?"
- "Show me patients with similar presentations who responded well to treatment X"

**Key Endpoints**:
- `POST /chat/query` - Natural language query interface
- `GET /chat/history/{session_id}` - Retrieve conversation history
- `POST /chat/context` - Set conversation context
- `POST /chat/feedback` - Provide query feedback

## API Structure

### Base URL
```
http://localhost:8000
```

### Core Endpoints

#### System Endpoints
- `GET /` - Root endpoint with API information
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed system health status
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

#### Document Processing Pipeline
```
POST /ingest/upload → POST /ocr/extract → POST /ner/extract → 
POST /chunk/process → POST /graph/create → POST /insights/generate
```

#### Agent-Specific Endpoints
Each agent exposes RESTful endpoints following consistent patterns:
- `POST /{agent}/process` - Main processing endpoint
- `GET /{agent}/status/{id}` - Status checking
- `GET /{agent}/results/{id}` - Retrieve results
- `POST /{agent}/retry/{id}` - Retry operations

### Detailed Router Endpoints

#### Ingestion Router (`/ingest`)

**Upload Documents**
```http
POST /ingest/upload
Content-Type: multipart/form-data

Response:
{
  "message": "Medical documents uploaded successfully for processing",
  "upload_id": "upload_12345",
  "status": "received", 
  "files_count": 2,
  "patient_id": "patient_789",
  "next_step": "OCR extraction scheduled"
}
```

**Check Upload Status**
```http
GET /ingest/status/{upload_id}

Response:
{
  "message": "Upload processing completed successfully",
  "upload_id": "upload_12345",
  "status": "completed",
  "files_processed": 2,
  "processing_time": "45 seconds",
  "next_available": "OCR results ready for extraction"
}
```

**Get Upload Results**
```http
GET /ingest/results/{upload_id}

Response:
{
  "message": "Upload results retrieved successfully",
  "files": [
    {
      "filename": "lab_report.pdf",
      "file_id": "file_001", 
      "status": "extracted",
      "content_preview": "Patient: John Doe, HbA1c: 7.8%..."
    }
  ]
}
```

#### OCR Router (`/ocr`)

**Extract Text from Documents**
```http
POST /ocr/extract
Content-Type: application/json

Request:
{
  "document_id": "doc_123",
  "enhance_image": true,
  "language": "eng"
}

Response:
{
  "message": "OCR text extraction completed successfully",
  "document_id": "doc_123",
  "extracted_text": "LABORATORY REPORT\nPatient: John Doe\nHbA1c: 7.8%\nGlucose: 165 mg/dL...",
  "confidence_score": 0.96,
  "processing_time": "12 seconds",
  "pages_processed": 2,
  "next_step": "Ready for medical entity recognition"
}
```

**Get OCR Status**
```http
GET /ocr/status/{document_id}

Response:
{
  "message": "OCR processing status retrieved successfully", 
  "document_id": "doc_123",
  "status": "completed",
  "pages_total": 2,
  "pages_processed": 2,
  "confidence_average": 0.94,
  "issues_detected": []
}
```

#### NER Router (`/ner`)

**Extract Medical Entities**
```http
POST /ner/extract
Content-Type: application/json

Request:
{
  "text": "Patient diagnosed with Type 2 Diabetes. Prescribed Metformin 500mg twice daily.",
  "document_id": "doc_123"
}

Response:
{
  "message": "Medical entities extracted successfully",
  "document_id": "doc_123", 
  "entities": {
    "conditions": ["Type 2 Diabetes"],
    "medications": ["Metformin"],
    "dosages": ["500mg"],
    "frequencies": ["twice daily"],
    "procedures": [],
    "lab_values": []
  },
  "confidence_scores": {
    "conditions": 0.98,
    "medications": 0.95,
    "overall": 0.92
  },
  "processing_time": "3.2 seconds",
  "next_step": "Ready for timeline structuring"
}
```

**Get Entity Results**
```http
GET /ner/entities/{doc_id}

Response:
{
  "message": "Medical entities retrieved successfully",
  "document_id": "doc_123",
  "total_entities": 12,
  "entity_breakdown": {
    "conditions": 3,
    "medications": 2, 
    "lab_values": 4,
    "procedures": 2,
    "symptoms": 1
  },
  "extracted_at": "2024-06-15T10:30:00Z"
}
```

#### Chunking Router (`/chunk`)

**Process Document into Chunks**
```http
POST /chunk/process
Content-Type: application/json

Request:
{
  "document_id": "doc_123",
  "patient_id": "patient_789",
  "chunking_strategy": "visit_based"
}

Response:
{
  "message": "Document successfully chunked and structured",
  "document_id": "doc_123",
  "chunks_created": 3,
  "chunking_strategy": "visit_based",
  "timeline_events": [
    {
      "date": "2024-06-15",
      "type": "lab_results",
      "chunk_id": "chunk_001"
    },
    {
      "date": "2024-06-15", 
      "type": "prescription",
      "chunk_id": "chunk_002"
    }
  ],
  "processing_time": "8 seconds",
  "next_step": "Ready for graph database integration"
}
```

**Get Patient Timeline**
```http
GET /chunk/timeline/{patient_id}

Response:
{
  "message": "Patient timeline retrieved successfully",
  "patient_id": "patient_789",
  "timeline_span": "2024-01-15 to 2024-06-15",
  "total_visits": 3,
  "total_events": 12,
  "recent_activity": "Lab results updated 2 hours ago",
  "timeline_summary": "Diabetes management progression tracked over 5 months"
}
```

#### Graph Router (`/graph`)

**Create Patient Graph**
```http
POST /graph/create
Content-Type: application/json

Request:
{
  "patient_id": "patient_789",
  "visit_data": {
    "date": "2024-06-15",
    "diagnoses": ["Type 2 Diabetes"],
    "medications": ["Metformin 500mg"],
    "lab_results": [{"name": "HbA1c", "value": 7.8, "unit": "%"}]
  }
}

Response:
{
  "message": "Patient graph created successfully in Neo4j",
  "patient_id": "patient_789",
  "graph_nodes_created": 8,
  "relationships_created": 12,
  "graph_id": "graph_456",
  "cypher_operations": 5,
  "processing_time": "15 seconds",
  "next_step": "Graph ready for insights generation"
}
```

**Query Patient Graph** 
```http
GET /graph/patient/{patient_id}

Response:
{
  "message": "Patient graph data retrieved successfully",
  "patient_id": "patient_789",
  "graph_summary": {
    "total_visits": 3,
    "conditions": 2,
    "medications": 3,
    "lab_tests": 8,
    "procedures": 1
  },
  "last_updated": "2024-06-15T14:22:00Z",
  "graph_health": "complete"
}
```

**Execute Cypher Query**
```http
POST /graph/query
Content-Type: application/json

Request:
{
  "query": "MATCH (p:Patient)-[:HAS_VISIT]->(v:Visit)-[:HAS_TEST]->(t:Test) WHERE p.id = 'patient_789' RETURN t.name, t.value",
  "parameters": {}
}

Response:
{
  "message": "Cypher query executed successfully",
  "results": [
    {"t.name": "HbA1c", "t.value": 7.8},
    {"t.name": "Glucose", "t.value": 165}
  ],
  "execution_time": "45ms",
  "records_returned": 2,
  "query_performance": "optimal"
}
```

#### Supervisor Router (`/supervisor`)

**Enqueue Processing Job**
```http
POST /supervisor/enqueue
Content-Type: application/json

Request:
{
  "patient_id": "patient_789",
  "documents": ["doc_123", "doc_124"],
  "priority": "normal",
  "workflow_type": "full_pipeline"
}

Response:
{
  "message": "Processing job enqueued successfully",
  "job_id": "job_999",
  "status": "queued",
  "estimated_completion": "3-5 minutes",
  "workflow_steps": [
    "OCR extraction",
    "Medical NER",
    "Timeline structuring", 
    "Graph integration",
    "Insight generation"
  ],
  "queue_position": 2
}
```

**Check Job Status**
```http
GET /supervisor/status/{job_id}

Response:
{
  "message": "Job status retrieved successfully",
  "job_id": "job_999",
  "status": "processing",
  "current_step": "Medical NER",
  "progress": "60%",
  "steps_completed": 3,
  "steps_remaining": 2,
  "estimated_time_remaining": "90 seconds",
  "last_update": "2024-06-15T14:25:00Z"
}
```

**Retry Failed Job**
```http
POST /supervisor/retry/{job_id}

Response:
{
  "message": "Job retry initiated successfully",
  "job_id": "job_999",
  "retry_attempt": 2,
  "status": "retrying",
  "failure_reason": "Temporary OCR service timeout",
  "retry_strategy": "exponential_backoff",
  "next_attempt_in": "30 seconds"
}
```

## Example Workflow

### Complete Medical Record Processing Pipeline

#### End-to-End Workflow Overview
```
Doctor uploads a record → File Upload Service → OCR Processing → 
Medical NER → Timeline Structuring → Graph Database Update → 
Insight Generation → Query Interface Ready
```

### Detailed Workflow Steps

#### Step 1: Doctor uploads a record → File Upload Service
**Scenario**: Dr. Smith has a patient's lab results (PDF) and prescription (scanned image) that need to be processed.

**Action**: Doctor uses the web interface or mobile app to upload documents
```bash
# Multiple file upload with patient context
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@lab_results.pdf" \
  -F "files=@prescription_scan.jpg" \
  -F "patient_id=12345" \
  -F "visit_date=2024-06-15" \
  -F "document_types=lab_report,prescription" \
  -F "provider=Dr. Smith" \
  -F "clinic=General Medicine Clinic"
```

**File Upload Service Response**:
```json
{
  "upload_id": "upload_789",
  "status": "received",
  "message": "Documents uploaded successfully. Processing initiated.",
  "files_processed": [
    {
      "filename": "lab_results.pdf",
      "file_id": "file_001",
      "size": "2.4MB",
      "type": "application/pdf",
      "status": "queued_for_ocr"
    },
    {
      "filename": "prescription_scan.jpg", 
      "file_id": "file_002",
      "size": "1.8MB",
      "type": "image/jpeg",
      "status": "queued_for_ocr"
    }
  ],
  "patient_context": {
    "patient_id": "12345",
    "visit_date": "2024-06-15",
    "provider": "Dr. Smith"
  },
  "next_steps": [
    "OCR text extraction",
    "Medical entity recognition", 
    "Timeline integration"
  ],
  "estimated_completion": "2-3 minutes"
}
```

#### Status Tracking

```bash
# 2. Check processing status
curl "http://localhost:8000/ingest/status/550e8400-e29b-41d4-a716-446655440000"

# Response: Real-time progress
{
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing", 
  "current_step": "OCR Processing",
  "progress_percentage": 60.0,
  "files_completed": 1,
  "files_remaining": 1,
  "estimated_time_remaining": "90 seconds"
}
```

#### Results Retrieval

```bash
# 3. Get final results
curl "http://localhost:8000/ingest/results/550e8400-e29b-41d4-a716-446655440000"

# Response: Complete file information
{
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Upload results retrieved successfully",
  "files": [
    {
      "filename": "prescription.pdf",
      "file_id": "file_001",
      "status": "completed",
      "content_preview": "PRESCRIPTION\nPatient: John Doe\nMedication: Metformin 500mg..."
    }
  ],
  "available_actions": [
    "View extracted text",
    "Access medical entities", 
    "View patient timeline"
  ]
}
```

## Implementation Details

## Implementation Status

### ✅ Completed: File Upload Service ("Doctor uploads a record → File Upload Service")

**Implementation Status**: **FULLY IMPLEMENTED AND DOCUMENTED**

The complete "Doctor uploads a record → File Upload Service" workflow has been implemented with production-ready code, comprehensive documentation, and full integration with the MRIA pipeline. All components are tested and operational.

**What's Implemented:**

- ✅ Complete file upload REST API with FastAPI
- ✅ Secure file validation and storage service
- ✅ Comprehensive Pydantic data models and schemas
- ✅ Multi-format file support (PDF, images, documents)
- ✅ Patient-specific file organization and security
- ✅ Real-time status tracking and progress monitoring
- ✅ Background job queue integration for OCR processing
- ✅ Error handling, logging, and production-ready features
- ✅ Complete API documentation and usage examples
- ✅ Integration with downstream MRIA agents

**Files Created/Updated:**

```text
app/
├── routers/ingestion.py         # REST API endpoints (529 lines)
├── schemas/ingestion.py         # Pydantic models (168 lines)
├── services/file_handler.py     # File handling service (383 lines)
└── common/utils.py              # Utility functions
```

**Ready for Production**: The implementation includes all security, validation, monitoring, and error handling required for production deployment.

## Environment Configuration

### File Storage Environment Variables

The MRIA system uses environment variables to configure file storage locations, security settings, and processing parameters. All configuration is managed through a `.env` file in the project root.

#### Core Storage Configuration

```bash
# Base storage directory for all file operations
BASE_STORAGE_DIR=./storage

# Upload directories (relative to BASE_STORAGE_DIR)
UPLOAD_DIR=uploads/documents          # Main document uploads
TEMP_DIR=uploads/temp                 # Temporary processing files
PROCESSED_DIR=uploads/processed       # OCR-processed files
BACKUP_DIR=uploads/backup             # Backup copies
IMAGE_UPLOAD_DIR=uploads/images       # Image-specific uploads
IMAGE_PROCESSED_DIR=uploads/images/processed  # Processed images
```

#### Patient Directory Structure

```bash
# Patient-specific directory organization
PATIENT_DIR_STRUCTURE=patient_{patient_id}  # Directory naming pattern
USE_PATIENT_SUBDIRS=true                     # Enable patient isolation
```

**Storage Layout Example:**

```text
storage/
├── uploads/
│   ├── documents/
│   │   ├── patient_12345/           # Patient-specific directories
│   │   │   ├── uuid_20240615_prescription.pdf
│   │   │   └── uuid_20240615_lab_report.pdf
│   │   └── patient_67890/
│   ├── images/
│   │   ├── patient_12345/
│   │   │   └── uuid_20240615_xray.jpg
│   │   └── patient_67890/
│   ├── temp/                        # Temporary processing files
│   ├── processed/                   # OCR-processed content
│   └── backup/                      # Backup copies
```

#### File Size and Type Restrictions

```bash
# File size limits (in bytes)
MAX_FILE_SIZE=52428800              # 50MB default limit
MAX_PDF_SIZE=52428800               # 50MB for PDF files
MAX_IMAGE_SIZE=10485760             # 10MB for images
MAX_OFFICE_DOC_SIZE=26214400        # 25MB for office documents

# Allowed file types
ALLOWED_FILE_EXTENSIONS=pdf,txt,docx,doc,png,jpg,jpeg,tiff,tif,xls,xlsx
ALLOWED_MIME_TYPES=application/pdf,text/plain,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword,image/png,image/jpeg,image/tiff,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
```

#### Security and Retention Settings

```bash
# File retention and cleanup
FILE_RETENTION_DAYS=30              # Auto-delete files after 30 days
TEMP_FILE_RETENTION_HOURS=24        # Clean temp files after 24 hours
AUTO_CLEANUP_ENABLED=true           # Enable automatic cleanup
CLEANUP_SCHEDULE=0 2 * * *           # Daily at 2 AM (cron format)

# Security settings
ENABLE_VIRUS_SCAN=false             # Enable basic content scanning
VIRUS_SCAN_TIMEOUT=30               # Scan timeout in seconds
ENABLE_FILE_ENCRYPTION=false        # Enable file encryption at rest
ENCRYPTION_KEY_PATH=                # Path to encryption key file

# Storage quotas (in MB)
MAX_STORAGE_PER_PATIENT=1024        # 1GB per patient
MAX_TOTAL_STORAGE=10240             # 10GB total storage limit
STORAGE_WARNING_THRESHOLD=80        # Warning at 80% capacity
```

#### Configuration Module (`app/config.py`)

The configuration system uses Pydantic Settings for type-safe environment variable loading:

```python
from app.config import get_storage_config, get_patient_upload_dir

# Get storage configuration
storage_config = get_storage_config()
print(f"Base storage: {storage_config.base_storage_dir}")
print(f"Max file size: {storage_config.max_file_size} bytes")

# Get patient-specific directory
patient_upload_dir = get_patient_upload_dir("patient_12345")
print(f"Patient upload dir: {patient_upload_dir}")
```

#### Key Configuration Functions

```python
# Configuration access functions
get_storage_config() -> StorageConfig      # Get full storage config
get_patient_upload_dir(patient_id) -> Path # Patient upload directory
get_patient_image_dir(patient_id) -> Path  # Patient image directory
get_temp_dir() -> Path                      # Temporary files directory
get_processed_dir() -> Path                 # Processed files directory
get_file_size_limit(extension) -> int      # Size limit by file type
```

#### Environment Setup

1. **Copy Environment Template:**
   ```bash
   cp .env.example .env
   ```

2. **Update Storage Paths:**
   ```bash
   # Edit .env file
   BASE_STORAGE_DIR=./storage              # Development
   BASE_STORAGE_DIR=/var/lib/mria/storage  # Production
   ```

3. **Set Security Keys:**
   ```bash
   SECRET_KEY=your-secure-random-string-here
   NEO4J_PASSWORD=your-neo4j-password
   ```

4. **Configure Storage Limits:**
   ```bash
   # Adjust based on your requirements
   MAX_FILE_SIZE=104857600                 # 100MB
   MAX_STORAGE_PER_PATIENT=2048            # 2GB per patient
   ```

### Production Deployment

#### Storage Recommendations

**Development:**
```bash
BASE_STORAGE_DIR=./storage
MAX_FILE_SIZE=52428800                    # 50MB
MAX_STORAGE_PER_PATIENT=1024              # 1GB
```

**Production:**
```bash
BASE_STORAGE_DIR=/var/lib/mria/storage    # Persistent storage
MAX_FILE_SIZE=104857600                   # 100MB
MAX_STORAGE_PER_PATIENT=5120              # 5GB
ENABLE_FILE_ENCRYPTION=true               # Enable encryption
AUTO_CLEANUP_ENABLED=true                 # Enable cleanup
```

#### Security Considerations

1. **File Permissions:**
   ```bash
   # Set secure permissions on storage directory
   chmod 750 /var/lib/mria/storage
   chown mria:mria /var/lib/mria/storage
   ```

2. **Encryption Key Management:**
   ```bash
   # Generate encryption key
   openssl rand -base64 32 > /etc/mria/encryption.key
   chmod 600 /etc/mria/encryption.key
   
   # Set in environment
   ENCRYPTION_KEY_PATH=/etc/mria/encryption.key
   ```

3. **Backup Strategy:**
   ```bash
   # Enable backups
   BACKUP_DIR=uploads/backup
   FILE_RETENTION_DAYS=90                  # Longer retention
   ```

#### Health Monitoring

The storage configuration is exposed through the health endpoint:

```bash
curl http://localhost:8000/ingest/health
```

**Response includes:**
```json
{
  "service": "ingestion",
  "status": "healthy",
  "storage_accessible": true,
  "configuration": {
    "base_storage_dir": "storage",
    "max_file_size": 52428800,
    "supported_extensions": ["pdf", "txt", "docx", ...],
    "patient_subdirs_enabled": true,
    "auto_cleanup_enabled": true,
    "file_retention_days": 30
  }
}
```

## Implementation Status

### Query/Chat Agent

**Implementation Status**: ✅ **COMPLETED AND PRODUCTION READY**

The Query/Chat Agent has been fully implemented with comprehensive natural language processing capabilities, conversation management, and seamless integration with all existing MRIA components.

**What's Implemented:**
- ✅ Complete REST API with 6 primary endpoints
- ✅ Advanced medical query processing with 8 query types
- ✅ Natural language understanding for medical terminology
- ✅ Conversation session management and context persistence
- ✅ Integration with Graph Database, Insights Processor, and NER
- ✅ Medical entity extraction and intent detection
- ✅ Background task processing for metrics and feedback
- ✅ Comprehensive error handling and system monitoring
- ✅ Production-ready logging and security features

**Files Created:**
- `app/schemas/chat.py` - Complete Pydantic models (354 lines)
- `app/services/chat_processor.py` - Core chat engine (1,081 lines)  
- `app/routers/chat.py` - REST API endpoints (460 lines)
- `CHAT_AGENT_IMPLEMENTATION.md` - Complete documentation

**Validated Features:**
- ✅ Natural language query processing
- ✅ Medical terminology understanding
- ✅ Multi-modal query support (text-based)
- ✅ Contextual conversation management
- ✅ Domain-specific medical knowledge integration
- ✅ All 8 supported query types functional
- ✅ Integration testing completed successfully
