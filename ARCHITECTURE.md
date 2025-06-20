# MRIA System Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Agent-Based Architecture](#agent-based-architecture)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [Technology Stack](#technology-stack)
6. [Deployment Architecture](#deployment-architecture)
7. [Security Architecture](#security-architecture)
8. [Performance & Scalability](#performance--scalability)

## System Overview

The Medical Records Insight Agent (MRIA) is an AI-powered agentic system designed to process, analyze, and extract actionable insights from diverse medical records. The system transforms unstructured medical documents into a structured knowledge graph, enabling semantic queries and medical insights.

### Key Objectives
- **Document Intelligence**: Parse diverse medical document formats (PDFs, images, Excel, DOCX)
- **Medical NLP**: Extract structured medical entities and relationships
- **Knowledge Graph**: Build temporal, connected patient health graphs
- **Semantic Search**: Enable natural language queries across patient data
- **Insight Generation**: Provide medical analytics and pattern recognition

## High-Level Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        ST[Streamlit Web UI]
        API[FastAPI REST API]
    end
    
    subgraph "Application Layer"
        SUP[Supervisor Agent]
        OCR[OCR Agent]
        NER[NER Agent]
        CHK[Chunking Agent]
        GRA[Graph Agent]
        INS[Insight Agent]
        CHT[Chat Agent]
    end
    
    subgraph "Data Processing Layer"
        FS[File Storage]
        VS[Vector Store]
        REDIS[Redis Cache]
    end
    
    subgraph "Knowledge Layer"
        NEO4J[Neo4j Graph DB]
        CHROMA[ChromaDB Vector DB]
    end
    
    subgraph "External Services"
        AZURE[Azure OCR]
        GOOGLE[Google Vision]
        TESS[Tesseract OCR]
    end
    
    ST --> API
    API --> SUP
    SUP --> OCR
    SUP --> NER
    SUP --> CHK
    SUP --> GRA
    SUP --> INS
    SUP --> CHT
    
    OCR --> AZURE
    OCR --> GOOGLE
    OCR --> TESS
    
    FS --> OCR
    OCR --> VS
    NER --> VS
    CHK --> VS
    GRA --> NEO4J
    INS --> NEO4J
    CHT --> CHROMA
    
    SUP --> REDIS
    API --> REDIS
```

## Agent-Based Architecture

### Linear Workflow Pattern

The MRIA system implements a linear supervisor-orchestrated workflow where each agent returns control to the supervisor after completion:

```mermaid
graph LR
    A[Supervisor Entry] --> B(OCR Agent)
    B --> C[Supervisor]
    C --> D(NER Agent)
    D --> E[Supervisor]
    E --> F(Chunking Agent)
    F --> G[Supervisor]
    G --> H(Graph Agent)
    H --> I[Supervisor]
    I --> J(Insight Agent)
    J --> K[Supervisor]
    K --> L(Chat Agent)
    L --> M[Supervisor End]
    
    style A fill:#e1f5fe
    style C fill:#e1f5fe
    style E fill:#e1f5fe
    style G fill:#e1f5fe
    style I fill:#e1f5fe
    style K fill:#e1f5fe
    style M fill:#e1f5fe
```

### Agent Responsibilities

#### 1. Supervisor Agent
```python
# Key Features from supervisor.py
- LangChain/LangGraph workflow orchestration
- Redis-based job queue management
- State persistence with checkpointing
- Error handling and retry logic
- Real-time progress tracking
```

**Workflow Types**:
- `OCR_ONLY`: Text extraction only
- `OCR_TO_NER`: OCR + Medical entity recognition
- `COMPLETE_PIPELINE`: Full processing pipeline
- `DOCUMENT_TO_GRAPH`: Document to knowledge graph
- `INSIGHT_GENERATION`: Generate medical insights

#### 2. OCR + Ingestion Agent
**Input Formats**: PDF, DOCX, JPG, PNG, Excel, CSV
**OCR Providers**: 
- Tesseract (open source)
- Azure Form Recognizer (cloud)
- Google Vision API (cloud)

**Processing Pipeline**:
```
Document Upload → Format Detection → OCR Processing → Text Extraction → Quality Validation
```

#### 3. Medical NER Agent
**NLP Models**:
- scispaCy: Scientific/medical spaCy models
- BioBERT: Biomedical BERT
- Med7: Medical entity recognition
- Clinical BERT: Clinical domain specialization

**Entity Types Extracted**:
- **Conditions**: "Type 2 Diabetes Mellitus"
- **Medications**: "Metformin 500mg"
- **Procedures**: "Echocardiogram"
- **Symptoms**: "Chest pain", "Shortness of breath"
- **Lab Values**: "HbA1c = 7.8%", "BP = 140/90"
- **Anatomical**: "Left ventricle", "Coronary artery"

#### 4. Chunking & Timeline Agent
**Functionality**:
- Semantic document chunking
- Temporal event structuring
- Visit-based organization
- Chronological timeline creation

**Output Structure**:
```
Visit 1 (2024-01-15):
  ├── Lab Results: HbA1c, Glucose
  ├── Medications: Metformin prescribed
  └── Diagnosis: Type 2 DM confirmed

Visit 2 (2024-03-20):
  ├── Follow-up: BP monitoring
  ├── Medication adjustment
  └── Symptoms: Improved energy
```

#### 5. Knowledge Graph Builder Agent
**Graph Schema**:
```cypher
// Core node types
(Patient)-[:HAS_VISIT]->(Visit)
(Visit)-[:DIAGNOSED_WITH]->(Condition)
(Visit)-[:PRESCRIBED]->(Medication)
(Visit)-[:HAS_TEST]->(LabResult)
(Patient)-[:HAS_CONDITION]->(Condition)
(Medication)-[:TREATS]->(Condition)
(Medication)-[:INTERACTS_WITH]->(Medication)
```

**Relationship Types**:
- `HAS_VISIT`: Patient to visit connections
- `DIAGNOSED_WITH`: Visit to condition relationships
- `PRESCRIBED`: Medication prescriptions
- `HAS_TEST`: Lab test results
- `TREATS`: Drug-condition relationships
- `INTERACTS_WITH`: Drug interactions

#### 6. Insight Agent
**Analytics Capabilities**:
- Patient progression analysis
- Treatment effectiveness assessment
- Drug interaction detection
- Care gap identification
- Population health patterns

**Insight Categories**:
```python
insight_categories = [
    "patient_summary",
    "timeline_analysis", 
    "drug_interactions",
    "care_gaps",
    "treatment_effectiveness",
    "risk_assessment"
]
```

#### 7. Query/Chat Agent
**Query Types Supported**:
- Natural language medical questions
- Patient-specific queries
- Comparative analysis requests
- Population health queries
- Temporal analysis questions

**Example Queries**:
```
"What was the treatment history for patients diagnosed with type 2 diabetes in the last 6 months?"
"Summarize this patient's progression over the last year"
"Compare this patient's history with others having the same condition"
"Show me patients with similar symptoms to chest pain and shortness of breath"
```

## Data Flow Diagrams

### Document Processing Flow

```mermaid
flowchart TD
    subgraph "Input Layer"
        PDF[PDF Documents]
        IMG[Medical Images]
        DOC[DOCX Files]
        XLS[Excel Reports]
    end
    
    subgraph "Processing Pipeline"
        UP[Document Upload]
        OCR[OCR Processing]
        TXT[Text Extraction]
        NER[Medical NER]
        ENT[Entity Extraction]
        CHK[Document Chunking]
        TML[Timeline Creation]
        GRP[Graph Building]
        INS[Insight Generation]
    end
    
    subgraph "Storage Layer"
        FS[File Storage]
        VS[Vector Store]
        KG[Knowledge Graph]
        CACHE[Redis Cache]
    end
    
    subgraph "Query Layer"
        API[REST API]
        UI[Web Interface]
        CHAT[Chat Interface]
    end
    
    PDF --> UP
    IMG --> UP
    DOC --> UP
    XLS --> UP
    
    UP --> OCR
    OCR --> TXT
    TXT --> NER
    NER --> ENT
    ENT --> CHK
    CHK --> TML
    TML --> GRP
    GRP --> INS
    
    TXT --> VS
    ENT --> VS
    CHK --> VS
    GRP --> KG
    INS --> KG
    
    UP --> FS
    OCR --> CACHE
    
    KG --> API
    VS --> API
    API --> UI
    API --> CHAT
```

### Knowledge Graph Structure

```mermaid
graph TD
    subgraph "Patient Nodes"
        P1[Patient 12345]
        P2[Patient 67890]
    end
    
    subgraph "Visit Nodes"
        V1[Visit 2024-01-15]
        V2[Visit 2024-03-20]
        V3[Visit 2024-01-20]
    end
    
    subgraph "Medical Entities"
        C1[Type 2 DM]
        C2[Hypertension]
        M1[Metformin 500mg]
        M2[Lisinopril 10mg]
        L1[HbA1c: 7.8%]
        L2[BP: 140/90]
    end
    
    P1 -->|HAS_VISIT| V1
    P1 -->|HAS_VISIT| V2
    P2 -->|HAS_VISIT| V3
    
    V1 -->|DIAGNOSED_WITH| C1
    V1 -->|PRESCRIBED| M1
    V1 -->|HAS_TEST| L1
    
    V2 -->|DIAGNOSED_WITH| C2
    V2 -->|PRESCRIBED| M2
    V2 -->|HAS_TEST| L2
    
    M1 -->|TREATS| C1
    M2 -->|TREATS| C2
    
    style P1 fill:#e3f2fd
    style P2 fill:#e3f2fd
    style C1 fill:#fff3e0
    style C2 fill:#fff3e0
    style M1 fill:#e8f5e8
    style M2 fill:#e8f5e8
```

## Technology Stack

### Core Framework Stack
```yaml
Framework:
  - FastAPI: 0.104.1          # High-performance async web framework
  - Python: 3.11+            # Modern Python with type hints
  - Uvicorn: Latest          # ASGI server
  - Pydantic: 2.5.2          # Data validation

Agent Orchestration:
  - LangChain: 0.1.0         # Agent framework
  - LangGraph: 0.0.20        # Workflow management
  - Redis: 5.0.1             # Job queue and caching

AI/ML Stack:
  - Transformers: 4.36.2     # Hugging Face models
  - SpaCy: 3.7.2             # NLP processing
  - Sentence-Transformers: 2.2.2  # Embeddings
  - scikit-learn: 1.3.2      # ML utilities
```

### Database & Storage
```yaml
Knowledge Graph:
  - Neo4j: 5.15.0            # Graph database
  - neo4j-driver: 5.15.0     # Python driver

Vector Storage:
  - ChromaDB: 0.4.22         # Vector database
  - FAISS: 1.7.4             # Similarity search

Document Processing:
  - Tesseract: pytesseract 0.3.10  # OCR
  - OpenCV: 4.8.1.78         # Image processing
  - PDF2Image: 1.16.3        # PDF conversion
  - Pillow: 10.1.0           # Image handling
```

### External Integrations
```yaml
Cloud OCR Services:
  - Azure Form Recognizer    # Microsoft cognitive services
  - Google Vision API        # Google Cloud Vision
  - AWS Textract             # Amazon document analysis

Medical NLP Models:
  - scispaCy models          # Scientific NLP
  - BioBERT                  # Biomedical BERT
  - Clinical BERT            # Clinical domain
  - Med7                     # Medical NER
```

## Deployment Architecture

### Local Development Setup
```mermaid
graph TB
    subgraph "Development Environment"
        DEV[Developer Machine]
        DOCKER[Docker Containers]
        
        subgraph "Local Services"
            NEO4J_LOCAL[Neo4j Local]
            REDIS_LOCAL[Redis Local]
            API_LOCAL[FastAPI Dev Server]
            UI_LOCAL[Streamlit Dev]
        end
    end
    
    DEV --> DOCKER
    DOCKER --> NEO4J_LOCAL
    DOCKER --> REDIS_LOCAL
    DEV --> API_LOCAL
    DEV --> UI_LOCAL
```

### Production Deployment Options

#### Container-Based Deployment
```yaml
version: '3.8'
services:
  mria-api:
    image: mria:latest
    ports:
      - "8080:8080"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_URL=redis://redis:6379
    depends_on:
      - neo4j
      - redis
      
  neo4j:
    image: neo4j:5.15.0
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
      
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
      
  streamlit:
    image: mria-frontend:latest
    ports:
      - "8501:8501"
    depends_on:
      - mria-api
```



### HIPAA Compliance Considerations
- **Data Encryption**: All PHI encrypted at rest and in transit
- **Access Controls**: Role-based access with audit trails
- **Data Minimization**: Only necessary medical data processed
- **Audit Logging**: Comprehensive activity tracking
- **Backup & Recovery**: Secure data backup procedures

## Performance & Scalability

### Performance Characteristics
```yaml
Processing Throughput:
  - OCR Processing: ~2-5 seconds per page
  - NER Processing: ~1-3 seconds per document
  - Graph Updates: ~100-500 nodes/second
  - Query Response: <200ms average

Scalability Targets:
  - Concurrent Users: 100+
  - Documents/Day: 10,000+
  - Knowledge Graph: 1M+ nodes
  - Vector Storage: 100K+ embeddings
```

### Scaling Strategies
1. **Horizontal Scaling**: Multiple API instances behind load balancer
2. **Async Processing**: Non-blocking I/O operations
3. **Caching**: Redis for frequently accessed data
4. **Database Optimization**: Neo4j indexing and query optimization
5. **Resource Management**: CPU/memory allocation per agent
6. **Queue Management**: Priority-based job processing

---

*This architecture documentation is maintained as part of the MRIA project and should be updated as the system evolves.*
