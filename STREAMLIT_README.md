# MRIA Streamlit Application

This is the web interface for the **Medical Records Insight Agent (MRIA)** system - an AI-powered agentic system for parsing, understanding, and extracting actionable insights from medical records.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FastAPI backend running on port 8080
- Required services: Neo4j, Redis
- Install dependencies: `pip install -r streamlit_requirements.txt`

### Start the Application

1. **Start the FastAPI Backend:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
   ```

2. **Start the Streamlit Frontend:**
   ```bash
   streamlit run streamlit_app.py --server.port 8501
   ```

3. **Open your browser:**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8080
   - API Documentation: http://localhost:8080/docs

## 📋 Application Features

The Streamlit application provides a comprehensive web interface with the following pages:

### 🏠 Dashboard
- **Agent Status Monitoring**: Real-time health check of MRIA API (`/supervisor/health`)
- **Queue Metrics**: Live statistics showing total, running, completed, and failed jobs
- **System Health Indicators**: Visual status indicators for API and queue health
- **Queue Status**: Color-coded health status (Green: Healthy, Yellow: Warning, Red: Error)

### 📄 Document Upload
- **Multi-format Support**: PDF, DOCX, images (JPG, PNG), Excel/CSV files
- **Workflow Selection**: 
  - `complete_pipeline`: Full end-to-end processing
  - `ocr_only`: OCR extraction only
  - `ocr_to_ner`: OCR + Named Entity Recognition
  - `document_to_graph`: Document to Knowledge Graph
  - `insight_generation`: Generate medical insights
- **Priority Management**: Normal, High, Urgent, Low priority levels
- **Patient Context**: Optional patient ID association
- **File Information Display**: Shows selected files with size information

### 📊 Job Monitoring
- **Real-time Job Tracking**: Monitor document processing status via `/supervisor/status/{job_id}`
- **Job Progress Visualization**: Progress percentage and duration tracking
- **Status Overview**: Tabular view of all processing jobs
- **Auto-refresh Capability**: Manual refresh to update job statuses
- **Job History**: Track created time and processing duration

### 🕸️ Knowledge Graph Explorer
- **Patient-specific Graph Views**: Query patient knowledge graphs by ID
- **Custom Cypher Query Interface**: Execute custom Neo4j queries
- **Predefined Query Examples**:
  - Patient Visits: `MATCH (p:Patient)-[:HAS_VISIT]->(v:Visit)`
  - Patient Conditions: `MATCH (p:Patient)-[:DIAGNOSED_WITH]->(c:Condition)`
  - Visit Medications: `MATCH (v:Visit)-[:PRESCRIBED]->(m:Medication)`
  - Abnormal Lab Tests: `MATCH (p:Patient)-[:HAS_TEST]->(t:Test) WHERE t.abnormal_flag = true`
  - Drug Interactions: `MATCH (m1:Medication)-[:INTERACTS_WITH]-(m2:Medication)`
- **Graph Schema Reference**: Documentation of available node types and relationships
- **Query Result Display**: Tabular presentation of query results

### 💬 Medical Chat Interface
- **Natural Language Processing**: AI-powered medical query interface
- **Chat History**: Persistent conversation tracking in session state
- **Predefined Example Queries**:
  - Treatment history queries
  - Patient progression analysis
  - Comparative patient analysis
  - Medication pattern analysis
  - Symptom-based patient matching
- **Patient Context Options**: Query with specific patient focus
- **Response Context**: Include specific medical domains in responses

### 📈 Insights Dashboard
- **Patient-specific Analytics**: Generate insights for individual patients
- **Medical Pattern Analysis**: AI-powered analysis of patient data
- **Insight Generation**: On-demand insight creation via API calls
- **Patient Selection Interface**: Input field for patient ID selection
- **Progress Indicators**: Loading states during insight generation

### ⚙️ System Configuration
- **API Configuration**: 
  - Base URL configuration (default: `http://localhost:8080`)
  - API timeout settings
  - Connection testing functionality
- **Processing Defaults**:
  - Default workflow selection
  - Default priority settings
- **OCR Provider Selection**:
  - Tesseract (open source)
  - Azure Form Recognizer (with endpoint/key configuration)
  - Google Vision API (with service account configuration)
- **NER Model Configuration**: 
  - Multiple model selection: scispaCy, BioBERT, Med7, Clinical BERT
- **Database Configuration**:
  - Neo4j connection settings (URI, username, password)
  - Redis connection URL
- **Configuration Persistence**: Save settings to session state

## 🤖 Agent-Based Architecture

The system uses specialized AI agents for different processing stages:

1. **Supervisor Agent** - Orchestrates the entire pipeline
2. **OCR + Ingestion Agent** - Extracts text from documents
3. **Medical NER Agent** - Identifies medical entities
4. **Chunking & Timeline Agent** - Creates structured timelines
5. **Knowledge Graph Builder Agent** - Builds patient relationships
6. **Insight Agent** - Generates medical insights
7. **Query/Chat Agent** - Handles natural language queries

## 📊 Example Workflows

### Document Processing Pipeline
1. Upload medical document (PDF, image, etc.)
2. OCR Agent extracts text content
3. NER Agent identifies medical entities
4. Timeline Agent structures chronological events
5. Graph Builder creates knowledge relationships
6. Insight Agent generates analytical insights

### Medical Query Examples
- *"What was the treatment history for patients diagnosed with type 2 diabetes in the last 6 months?"*
- *"Summarize this patient's progression over the last year."*
- *"Compare this patient's history with others having the same condition."*
- *"What are the most common medications prescribed for hypertension?"*
- *"Find patients with similar symptoms to chest pain and shortness of breath."*

## 🛠️ Technical Implementation

### Application Architecture
- **Frontend Framework**: Streamlit with custom CSS styling
- **API Communication**: RESTful calls to FastAPI backend at `localhost:8080`
- **Session Management**: Streamlit session state for data persistence
- **Navigation**: Sidebar-based page routing system
- **Error Handling**: Comprehensive API error handling with user feedback

### Key Components
- **API Client**: `make_api_request()` function handles all backend communication
- **Health Monitoring**: `check_api_health()` for API status validation
- **File Upload**: Multi-file upload with format validation
- **Job Tracking**: Real-time status monitoring with auto-refresh
- **Query Interface**: Direct Cypher query execution against Neo4j
- **Configuration Management**: Persistent settings storage

### Dependencies (streamlit_requirements.txt)
- **Streamlit**: Web application framework
- **Requests**: HTTP client for API communication
- **Pandas**: Data manipulation and display
- **Plotly**: Interactive visualizations (subplots support)
- **Pathlib**: File path handling

### Page Structure
```python
pages = {
    "🏠 Dashboard": "dashboard",
    "📄 Document Upload": "upload", 
    "📊 Job Monitoring": "monitoring",
    "🕸️ Knowledge Graph": "graph",
    "💬 Medical Chat": "chat",
    "📈 Insights": "insights",
    "⚙️ Configuration": "config"
}
```

### Session State Variables
- `uploaded_files`: Track uploaded document files
- `processing_jobs`: Monitor active processing jobs
- `chat_history`: Maintain conversation history
- `patient_data`: Cache patient information
- `config`: Store application configuration

## 🔧 Configuration & Setup

### API Configuration
- **Base URL**: Default `http://localhost:8080` (configurable)
- **Health Check Endpoint**: `/supervisor/health`
- **Queue Status Endpoint**: `/supervisor/queue/status`
- **Job Status Endpoint**: `/supervisor/status/{job_id}`

### Application Settings
The application can be configured through the **Configuration** page:

- **API Connection**: URL, timeout settings, connection testing
- **Default Processing**: Workflow type and priority settings
- **OCR Providers**: Tesseract, Azure Form Recognizer, Google Vision API
- **NER Models**: scispaCy, BioBERT, Med7, Clinical BERT selection
- **Database Connections**: Neo4j and Redis configuration

### Environment Requirements
- **FastAPI Backend**: Must be running on configured port (default 8080)
- **Neo4j Database**: Required for knowledge graph functionality
- **Redis Server**: Required for job queue management
- **OCR Services**: Optional cloud services require API keys

## 📝 Usage Guide

### Getting Started
1. **Access Dashboard**: Navigate to http://localhost:8501 after starting the application
2. **Check System Status**: Verify API connection and queue health on the Dashboard
3. **Upload Documents**: Use the Document Upload page to process medical files
4. **Monitor Progress**: Track job status in the Job Monitoring section
5. **Explore Data**: Use Knowledge Graph and Chat interfaces to query processed data

### Step-by-Step Workflow
1. **Start with Dashboard**: Ensure all systems show green status indicators
2. **Upload Test Documents**: Begin with small files to verify the processing pipeline
3. **Select Appropriate Workflow**: Choose based on your processing needs:
   - Complete pipeline for full analysis
   - OCR-only for text extraction
   - Specific stages for targeted processing
4. **Monitor Job Progress**: Use the monitoring page to track processing status
5. **Query Results**: Use Chat interface for natural language queries or Graph explorer for Cypher queries
6. **Generate Insights**: Use Insights dashboard for patient-specific analytics

### Best Practices
- **File Size**: Start with smaller documents for initial testing
- **Patient IDs**: Associate documents with patient identifiers when available
- **Priority Settings**: Use appropriate priority levels based on urgency
- **Query Examples**: Start with predefined example queries before writing custom ones
- **Configuration**: Test API connections before processing documents

## 🚨 Troubleshooting

### Common Issues and Solutions

**"❌ MRIA API is not running" Error:**
- **Solution**: Start the FastAPI backend: `uvicorn app.main:app --host 0.0.0.0 --port 8080`
- **Check**: Verify Neo4j and Redis services are running
- **Port**: Ensure port 8080 is not being used by another application

**"❌ Could not connect to MRIA API" Error:**
- **Solution**: Check the API URL in Configuration page
- **Network**: Verify firewall settings allow connections to port 8080
- **Backend**: Confirm FastAPI service is healthy at http://localhost:8080/docs

**Document Upload Fails:**
- **File Format**: Verify file types are supported (PDF, DOCX, JPG, PNG, XLSX, CSV)
- **File Size**: Check available disk space and memory
- **OCR Dependencies**: Ensure OCR services are properly configured
- **API Response**: Check for error messages in the upload response

**Knowledge Graph Queries Return No Results:**
- **Data Processing**: Confirm documents have been fully processed through the pipeline
- **Graph Population**: Ensure NER and Graph Builder agents have completed successfully
- **Neo4j Connection**: Verify Neo4j database connection in Configuration
- **Query Syntax**: Validate Cypher query syntax for custom queries

**Chat Interface Not Responding:**
- **API Connection**: Check Dashboard for API health status
- **Processing Status**: Ensure documents have been processed and indexed
- **Patient Context**: Verify patient ID exists in the knowledge graph
- **Model Loading**: Check if NER models are properly loaded in the backend

**Job Monitoring Shows "Unknown" Status:**
- **Job ID**: Verify job IDs are valid and haven't expired
- **API Endpoint**: Check `/supervisor/status/{job_id}` endpoint availability
- **Refresh**: Use the refresh button to update job statuses
- **Backend Logs**: Check FastAPI backend logs for processing errors

### Performance Issues

**Slow Document Processing:**
- **File Size**: Large files take longer to process
- **OCR Provider**: Cloud OCR services may be slower than local Tesseract
- **System Resources**: Ensure adequate CPU and memory for processing
- **Queue Management**: Check job queue health and processing capacity

**Memory Issues:**
- **Large Files**: Process large documents in smaller batches
- **Session State**: Clear browser session state if application becomes sluggish
- **Backend Resources**: Monitor FastAPI backend memory usage

## 🔒 Security & Compliance

### Data Security
- **Patient Data Protection**: Ensure HIPAA compliance when processing medical records
- **Local Processing**: Data processed locally by default (no cloud transmission unless using cloud OCR)
- **API Security**: Secure API endpoints with appropriate authentication in production
- **Database Security**: Use strong passwords for Neo4j and Redis connections

### Configuration Security
- **API Keys**: Store OCR service credentials securely (avoid hardcoding)
- **Environment Variables**: Use environment variables for sensitive configurations
- **Network Security**: Consider VPN/firewall restrictions for production deployments
- **Access Control**: Implement user authentication for production environments

### Data Privacy
- **Session Isolation**: Each browser session maintains separate data
- **Temporary Storage**: Uploaded files are processed and can be configured for automatic cleanup
- **Graph Data**: Patient data stored in Neo4j should be properly secured and encrypted
- **Audit Trail**: Consider implementing audit logging for document processing activities

## 📞 Support & Development

### Getting Help
1. **System Logs**: Check terminal output for detailed error messages
2. **API Documentation**: Access FastAPI docs at http://localhost:8080/docs
3. **Configuration Validation**: Use the "Test API Connection" feature
4. **Job Status**: Monitor processing through the Job Monitoring interface

### Development & Debugging
- **Browser Console**: Check browser developer tools for JavaScript errors
- **Network Tab**: Monitor API requests and responses
- **Streamlit Logs**: Check terminal running Streamlit for Python errors
- **Backend Logs**: Monitor FastAPI uvicorn output for backend issues

### File Structure
```
streamlit_app.py              # Main Streamlit application
streamlit_requirements.txt    # Python dependencies for frontend
STREAMLIT_README.md          # This documentation file
```

### Key Functions
- `main()`: Application entry point with navigation
- `make_api_request()`: API communication handler
- `check_api_health()`: System health monitoring
- `display_agent_status()`: Dashboard functionality
- `upload_documents()`: File upload and processing
- `knowledge_graph_viewer()`: Graph exploration interface
- `medical_chat_interface()`: Natural language query interface

---

**MRIA - Medical Records Insight Agent**  
*AI-Powered Medical Data Analysis System*
