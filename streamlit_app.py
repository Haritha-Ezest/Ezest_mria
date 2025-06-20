#!/usr/bin/env python3
"""
MRIA (Medical Records Insight Agent) Streamlit Application

This application provides a comprehensive web interface for the AI-powered
medical records analysis system, supporting all agent-based workflows.
"""

import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="MRIA - Medical Records Insight Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8080"
AGENTS = {
    "supervisor": "Supervisor Agent",
    "ocr": "OCR + Ingestion Agent",
    "ner": "Medical NER Agent",
    "chunking": "Chunking & Timeline Agent",
    "graph": "Knowledge Graph Builder Agent",
    "insights": "Insight Agent",
    "chat": "Query/Chat Agent"
}

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_jobs' not in st.session_state:
    st.session_state.processing_jobs = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Optional[Dict]:
    """Make API request to MRIA backend"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            st.error(f"Unsupported method: {method}")
            return None
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Could not connect to MRIA API at {API_BASE_URL}")
        return None
    except Exception as e:
        st.error(f"❌ API request failed: {str(e)}")
        return None

def check_api_health() -> bool:
    """Check if the MRIA API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/supervisor/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def display_agent_status():
    """Display status of all agents"""
    st.subheader("🤖 Agent Status Dashboard")
    
    # Check API health
    api_healthy = check_api_health()
    
    if api_healthy:
        st.success("✅ MRIA API is running")
        
        # Get queue status
        queue_status = make_api_request("supervisor/queue/status")
        if queue_status:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Jobs", queue_status.get('total_jobs', 0))
            with col2:
                st.metric("Running Jobs", queue_status.get('running_jobs', 0))
            with col3:
                st.metric("Completed Jobs", queue_status.get('completed_jobs', 0))
            with col4:
                st.metric("Failed Jobs", queue_status.get('failed_jobs', 0))
                
            # Queue health indicator
            health = queue_status.get('queue_health', 'unknown')
            if health == 'healthy':
                st.success(f"🟢 Queue Status: {health.title()}")
            elif health == 'warning':
                st.warning(f"🟡 Queue Status: {health.title()}")
            else:
                st.error(f"🔴 Queue Status: {health.title()}")
    else:
        st.error("❌ MRIA API is not running. Please start the backend service.")
        st.info("Run: `uvicorn app.main:app --host 0.0.0.0 --port 8080`")

def upload_documents():
    """Document upload interface"""
    st.subheader("📄 Document Upload & Processing")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Medical Documents",
        type=['pdf', 'docx', 'jpg', 'jpeg', 'png', 'xlsx', 'csv'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, Images (JPG, PNG), Excel/CSV"
    )
    
    if uploaded_files:
        st.write(f"📁 Selected {len(uploaded_files)} file(s)")
        
        # Display file information
        for file in uploaded_files:
            st.write(f"- **{file.name}** ({file.size:,} bytes)")
        
        # Processing options
        st.subheader("Processing Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            workflow_type = st.selectbox(
                "Select Workflow",
                ["complete_pipeline", "ocr_only", "ocr_to_ner", "document_to_graph", "insight_generation"],
                help="Choose the processing workflow for your documents"
            )
            
        with col2:
            priority = st.selectbox(
                "Priority Level",
                ["normal", "high", "urgent", "low"],
                help="Set processing priority"
            )
        
        # Patient information
        st.subheader("Patient Information")
        patient_id = st.text_input("Patient ID (optional)", help="Enter patient identifier")
        
        # Process button
        if st.button("🚀 Start Processing", type="primary"):
            if check_api_health():
                # Process each file
                for file in uploaded_files:
                    # Save file temporarily
                    file_path = f"temp_{file.name}"
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    
                    # Create job request
                    job_data = {
                        "workflow_type": workflow_type,
                        "file_paths": [file_path],
                        "priority": priority,
                        "metadata": {
                            "original_filename": file.name,
                            "file_size": file.size,
                            "upload_time": datetime.now().isoformat()
                        }
                    }
                    
                    if patient_id:
                        job_data["patient_id"] = patient_id
                    
                    # Submit job
                    result = make_api_request("supervisor/enqueue", "POST", job_data)
                    
                    if result:
                        job_id = result.get('job_id')
                        st.success(f"✅ Job created for {file.name}: {job_id}")
                        st.session_state.processing_jobs.append({
                            'job_id': job_id,
                            'filename': file.name,
                            'status': 'queued',
                            'created_at': datetime.now()
                        })
                    else:
                        st.error(f"❌ Failed to process {file.name}")
                
                # Clean up temp files
                for file in uploaded_files:
                    file_path = f"temp_{file.name}"
                    if Path(file_path).exists():
                        Path(file_path).unlink()
                        
                st.rerun()
            else:
                st.error("❌ Cannot process files - API is not available")

def monitor_jobs():
    """Job monitoring interface"""
    st.subheader("📊 Job Monitoring")
    
    if not st.session_state.processing_jobs:
        st.info("No jobs to monitor. Upload documents to get started.")
        return
    
    # Refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    # Job status table
    jobs_data = []
    for job in st.session_state.processing_jobs:
        # Get job status
        status_result = make_api_request(f"supervisor/status/{job['job_id']}")
        
        if status_result:
            jobs_data.append({
                'Job ID': job['job_id'][:8] + "...",
                'Filename': job['filename'],
                'Status': status_result.get('status', 'unknown'),
                'Progress': f"{status_result.get('progress', 0)}%",
                'Created': job['created_at'].strftime("%H:%M:%S"),
                'Duration': str(datetime.now() - job['created_at']).split('.')[0]
            })
        else:
            jobs_data.append({
                'Job ID': job['job_id'][:8] + "...",
                'Filename': job['filename'],
                'Status': 'unknown',
                'Progress': "0%",
                'Created': job['created_at'].strftime("%H:%M:%S"),
                'Duration': str(datetime.now() - job['created_at']).split('.')[0]
            })
    
    # Display as dataframe
    if jobs_data:
        df = pd.DataFrame(jobs_data)
        st.dataframe(df, use_container_width=True)
        
        # Status summary
        status_counts = df['Status'].value_counts()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Completed", status_counts.get('completed', 0))
        with col2:
            st.metric("Running", status_counts.get('running', 0))
        with col3:
            st.metric("Failed", status_counts.get('failed', 0))

def knowledge_graph_viewer():
    """Knowledge graph visualization"""
    st.subheader("🕸️ Knowledge Graph Explorer")
    
    # Patient selection
    patient_id = st.text_input("Enter Patient ID", help="Enter patient ID to explore their knowledge graph")
    
    if patient_id:
        with st.spinner(f"Loading insights for patient {patient_id}..."):
            # Get patient insights data from the API endpoint
            insights_response = make_api_request(f"insights/generate/{patient_id}")
        
        if insights_response and "insights" in insights_response:
            insights = insights_response["insights"]
            st.success(f"✅ Found knowledge graph insights for patient {patient_id}")
            
            # Display API response information
            st.info(f"📊 {insights_response.get('message', 'Insights generated')} at {insights_response.get('timestamp', 'Unknown time')}")
            
            # Display overall patient information
            st.subheader("📊 Patient Overview")
            
            # Extract graph statistics
            graph_stats = insights.get("graph_statistics", {})
            
            # Show patient ID for confirmation
            st.write(f"**Patient ID:** {insights.get('patient_id', patient_id)}")
            st.write(f"**Insights Generated:** {insights.get('insights_generated_at', 'Unknown')}")
            
            # Display key metrics in an enhanced layout
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                visits_count = graph_stats.get("total_visits", 0)
                st.metric("Total Visits", visits_count, help="Number of medical visits recorded")
            with col2:
                conditions_count = graph_stats.get("total_conditions", 0)
                st.metric("Conditions", conditions_count, help="Number of diagnosed conditions")
            with col3:
                medications_count = graph_stats.get("total_medications", 0)
                st.metric("Medications", medications_count, help="Number of prescribed medications")
            with col4:
                tests_count = graph_stats.get("total_tests", 0)
                st.metric("Lab Tests", tests_count, help="Number of laboratory tests performed")
            with col5:
                procedures_count = graph_stats.get("total_procedures", 0)
                st.metric("Procedures", procedures_count, help="Number of medical procedures")
            
            # Create tabs for focused views
            tab1, tab2 = st.tabs([
                "📋 Graph Database Statistics",
                "🔍 Raw Insights Data"
            ])
            
            with tab1:
                st.subheader("Graph Database Statistics")
                
                if graph_stats:
                    # Create statistics visualization
                    stats_data = {
                        "Entity Type": ["Visits", "Conditions", "Medications", "Lab Tests", "Procedures"],
                        "Count": [
                            graph_stats.get("total_visits", 0),
                            graph_stats.get("total_conditions", 0),
                            graph_stats.get("total_medications", 0),
                            graph_stats.get("total_tests", 0),
                            graph_stats.get("total_procedures", 0)
                        ]
                    }
                    
                    stats_df = pd.DataFrame(stats_data)
                    
                    # Bar chart
                    fig = px.bar(
                        stats_df,
                        x="Entity Type",
                        y="Count",
                        title=f"Medical Entity Distribution for Patient {patient_id}",
                        color="Count",
                        color_continuous_scale="viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed statistics
                    st.subheader("Detailed Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Core Metrics:**")
                        for key, value in graph_stats.items():
                            if isinstance(value, (int, float)):
                                formatted_key = key.replace("_", " ").title()
                                st.metric(formatted_key, value)
                    
                    with col2:
                        st.write("**Derived Metrics:**")
                        # Calculate derived metrics
                        total_entities = sum(v for v in graph_stats.values() if isinstance(v, (int, float)))
                        st.metric("Total Medical Entities", total_entities)
                        
                        if graph_stats.get("total_visits", 0) > 0:
                            avg_conditions = graph_stats.get("total_conditions", 0) / graph_stats.get("total_visits", 1)
                            st.metric("Avg Conditions per Visit", f"{avg_conditions:.1f}")
                            
                            avg_medications = graph_stats.get("total_medications", 0) / graph_stats.get("total_visits", 1)
                            st.metric("Avg Medications per Visit", f"{avg_medications:.1f}")
                            
                            avg_tests = graph_stats.get("total_tests", 0) / graph_stats.get("total_visits", 1)
                            st.metric("Avg Tests per Visit", f"{avg_tests:.1f}")
                    
                    # Additional breakdown if available
                    if graph_stats.get("condition_counts"):
                        st.subheader("Condition Breakdown")
                        condition_data = graph_stats["condition_counts"]
                        conditions_df = pd.DataFrame(
                            list(condition_data.items()), 
                            columns=['Condition', 'Count']
                        )
                        conditions_df = conditions_df.sort_values('Count', ascending=False)
                        st.dataframe(conditions_df, use_container_width=True)
                    
                    if graph_stats.get("medication_counts"):
                        st.subheader("Medication Breakdown")
                        medication_data = graph_stats["medication_counts"]
                        medications_df = pd.DataFrame(
                            list(medication_data.items()), 
                            columns=['Medication', 'Count']
                        )
                        medications_df = medications_df.sort_values('Count', ascending=False)
                        st.dataframe(medications_df, use_container_width=True)
                else:
                    st.info("No graph statistics available for this patient")
            
            with tab2:
                st.subheader("Raw Insights Data")
                
                # Display raw JSON data
                st.json(insights)
                
                # Display API response metadata
                st.subheader("API Response Metadata")
                metadata = {
                    "Message": insights_response.get("message", "N/A"),
                    "Timestamp": insights_response.get("timestamp", "N/A"),
                    "Patient ID": patient_id,
                    "Insights Generated": insights.get("insights_generated_at", "N/A")
                }
                
                for key, value in metadata.items():
                    st.write(f"**{key}:** {value}")
        
        else:
            st.warning(f"❌ No knowledge graph insights found for patient {patient_id}")
            
            # Show debug information if response exists but no insights
            if insights_response:
                st.subheader("🔍 API Response Debug Information")
                
                # Show the full response structure
                st.write("**Full API Response:**")
                st.json(insights_response)
                
                # Check for specific error messages
                if "detail" in insights_response:
                    st.error(f"API Error: {insights_response['detail']}")
                  # Show what the endpoint should return
                st.info("""
                **Expected Response Structure:**
                ```json
                {
                    "message": "Patient insights generated successfully",
                    "insights": {
                        "patient_id": "...",
                        "graph_statistics": {...},
                        "timeline_analysis": {...},
                        "potential_drug_interactions": [...],
                        "care_gaps": [...],
                        "insights_generated_at": "..."
                    },
                    "timestamp": "..."
                }
                ```
                """)
            else:
                st.error("❌ Failed to get response from the API endpoint")
                st.info("Please check if the MRIA API server is running and accessible.")
    
      # Enhanced Query interface
    st.subheader("🔍 Graph Query Interface")
    
    st.write("Execute custom Cypher queries against the knowledge graph to explore patient data and relationships.")
    
    # Example queries with descriptions
    query_examples = {
        "Patient Visits": "MATCH (p:Patient)-[:HAS_VISIT]->(v:Visit) RETURN p.id as patient_id, v.date as visit_date, v.visit_type LIMIT 10",
        "Patient Conditions": "MATCH (p:Patient)-[:DIAGNOSED_WITH]->(c:Condition) RETURN p.id as patient_id, c.condition_name as condition LIMIT 10",
        "Visit Medications": "MATCH (v:Visit)-[:PRESCRIBED]->(m:Medication) RETURN v.date as visit_date, m.medication_name as medication, m.dosage LIMIT 10",
        "Abnormal Lab Tests": "MATCH (p:Patient)-[:HAS_TEST]->(t:Test) WHERE t.abnormal_flag = true RETURN p.id as patient_id, t.test_name, t.value, t.reference_range LIMIT 10",
        "Drug Interactions": "MATCH (m1:Medication)-[:INTERACTS_WITH]-(m2:Medication) RETURN m1.medication_name as drug1, m2.medication_name as drug2 LIMIT 10"
    }
    
    # Query selection
    selected_example = st.selectbox(
        "Select Example Query", 
        ["Custom Query"] + list(query_examples.keys()),
        help="Choose a predefined query or select 'Custom Query' to write your own"
    )
    
    # Set query text based on selection
    if selected_example == "Custom Query":
        default_query = ""
        st.info("💡 **Tip:** Use Cypher syntax to query the Neo4j graph database. Common patterns include MATCH, WHERE, RETURN clauses.")
    else:
        default_query = query_examples[selected_example]
        st.info(f"**Query Description:** {selected_example}")
    
    custom_query = st.text_area(
        "Cypher Query", 
        value=default_query, 
        height=120,
        help="Enter your Cypher query here. Be careful with queries that might return large datasets."
    )
    
    # Query execution
    col1, col2 = st.columns([1, 4])
    with col1:
        execute_button = st.button("🚀 Execute Query", type="primary")
    with col2:
        if custom_query:
            st.write(f"**Query Length:** {len(custom_query)} characters")
    
    if execute_button and custom_query:
        with st.spinner("Executing query..."):
            query_result = make_api_request("graph/query", "POST", {"query": custom_query})
        
        if query_result:
            # Check if query was successful
            if query_result.get('success', False):
                st.success("✅ Query executed successfully")
                
                # Display query metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Results Count", query_result.get('result_count', 0))
                with col2:
                    processing_time = query_result.get('processing_time', 0)
                    st.metric("Processing Time", f"{processing_time:.3f}s")
                with col3:
                    st.metric("Columns", len(query_result.get('columns', [])))
                
                # Display results
                results = query_result.get('results', [])
                if results:
                    st.subheader("Query Results")
                    
                    # Convert results to DataFrame for better display
                    try:
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        
                        # Download option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.warning(f"Could not convert results to table format: {e}")
                        st.json(results)
                else:
                    st.info("Query executed successfully but returned no results")
            else:
                st.error("❌ Query execution failed")
                error_message = query_result.get('message', 'Unknown error')
                st.error(f"Error: {error_message}")
                
                # Show query for debugging
                st.subheader("Debug Information")
                st.code(custom_query, language='cypher')
        else:
            st.error("❌ Failed to execute query - API request failed")
            st.info("Please check if the MRIA API server is running and accessible.")
    
    # Graph schema helper section
    st.subheader("📚 Graph Schema Reference")
    
    with st.expander("View Available Node Types and Relationships", expanded=False):
        schema_response = make_api_request("graph/schema")
        
        if schema_response:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Node Types:**")
                node_types = schema_response.get('node_types', [])
                for node_type in node_types:
                    st.write(f"• {node_type}")
            
            with col2:
                st.write("**Relationship Types:**")
                rel_types = schema_response.get('relationship_types', [])
                for rel_type in rel_types:
                    st.write(f"• {rel_type}")
            
            # Node properties
            st.write("**Node Properties:**")
            node_props = schema_response.get('node_properties', {})
            for node_type, properties in node_props.items():
                st.write(f"**{node_type}:** {', '.join(properties)}")
            
            # Relationship properties
            st.write("**Relationship Properties:**")
            rel_props = schema_response.get('relationship_properties', {})
            for rel_type, properties in rel_props.items():
                st.write(f"**{rel_type}:** {', '.join(properties)}")
        else:
            st.info("Could not load graph schema. Using default schema information.")
            
            # Default schema information
            st.write("**Common Node Types:**")
            st.write("• Patient, Visit, Condition, Medication, Test, Procedure")
            
            st.write("**Common Relationships:**")
            st.write("• HAS_VISIT, DIAGNOSED_WITH, PRESCRIBED, PERFORMED, UNDERWENT")


def medical_chat_interface():
    """Medical chat and query interface"""
    st.subheader("💬 Medical Query Interface")
    
    # Chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, chat in enumerate(st.session_state.chat_history):
            with st.expander(f"Query {i+1}: {chat['query'][:50]}..."):
                st.write(f"**Question:** {chat['query']}")
                st.write(f"**Answer:** {chat['response']}")
                st.write(f"**Time:** {chat['timestamp']}")
    
    # Query examples
    st.subheader("Example Queries")
    
    example_queries = [
        "What was the treatment history for patients diagnosed with type 2 diabetes in the last 6 months?",
        "Summarize this patient's progression over the last year.",
        "Compare this patient's history with others having the same condition.",
        "What are the most common medications prescribed for hypertension?",
        "Show me patients with similar symptoms to chest pain and shortness of breath.",
        "What lab values are trending upward for patient ID 12345?",
        "Find patients with medication adherence issues."
    ]
    
    selected_example = st.selectbox("Select Example Query", [""] + example_queries)
    
    # Query input
    user_query = st.text_area(
        "Enter your medical query:",
        value=selected_example,
        height=100,
        help="Ask questions about patient data, treatments, conditions, or medical insights"
    )
    
    # Additional context
    col1, col2 = st.columns(2)
    with col1:
        patient_context = st.text_input("Patient ID (optional)", help="Focus query on specific patient")
    with col2:
        time_range = st.selectbox("Time Range", ["All time", "Last 7 days", "Last 30 days", "Last 6 months", "Last year"])
    
    # Submit query
    if st.button("🔍 Submit Query", type="primary") and user_query:
        with st.spinner("Processing your query..."):
            # Prepare chat request
            chat_data = {
                "query": user_query,
                "context": {
                    "patient_id": patient_context if patient_context else None,
                    "time_range": time_range,
                    "include_similar_cases": True
                }
            }
            
            # Make API request
            response = make_api_request("chat/query", "POST", chat_data)
            
            if response:
                # Display response
                st.success("✅ Query processed successfully")
                
                answer = response.get('response', 'No response available')
                confidence = response.get('confidence', 0)
                sources = response.get('sources', [])
                
                # Main answer
                st.subheader("Response")
                st.write(answer)
                
                # Confidence and sources
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{confidence:.1%}")
                with col2:
                    st.metric("Sources", len(sources))
                
                # Sources detail
                if sources:
                    with st.expander("View Sources"):
                        for source in sources:
                            st.write(f"- {source}")
                
                # Save to chat history
                st.session_state.chat_history.append({
                    'query': user_query,
                    'response': answer,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'confidence': confidence,
                    'sources': sources
                })
                
                # Related insights
                insights = response.get('related_insights', [])
                if insights:
                    st.subheader("Related Insights")
                    for insight in insights:
                        st.info(f"💡 {insight}")
                        
            else:
                st.error("❌ Failed to process query")

def insights_dashboard():
    """Medical insights and analytics dashboard"""
    st.subheader("📈 Medical Insights Dashboard")
    
    # Patient selection section
    st.subheader("🔍 Patient Selection")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        patient_id = st.text_input(
            "Enter Patient ID", 
            placeholder="e.g., PAT001, PAT005, etc.",
            help="Enter the patient ID to generate insights for"
        )
    with col2:
        generate_insights = st.button("🚀 Generate Insights", type="primary")
      # Only proceed if patient_id is provided and button is clicked
    if patient_id and generate_insights:
        with st.spinner(f"Generating insights for patient {patient_id}..."):
            # Get insights data
            insights_data = make_api_request(f"insights/generate/{patient_id}")
        
        if insights_data and insights_data.get('insights'):
            insights = insights_data['insights']
            
            # Display patient information
            st.success(f"✅ Insights generated successfully for Patient {patient_id}")
            
            # Patient Overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Patient ID", patient_id)
            with col2:
                generation_time = insights.get('insights_generated_at', 'N/A')
                st.metric("Generated At", generation_time)
            with col3:
                processing_time = insights_data.get('processing_time', 0)
                st.metric("Processing Time", f"{processing_time:.2f}s" if processing_time else "N/A")
            
            # Create tabs for different insight categories
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Graph Statistics", 
                "📈 Timeline Analysis", 
                "💊 Drug Interactions", 
                "⚠️ Care Gaps",
                "🔍 Raw Data"
            ])
            
            with tab1:
                st.subheader("Graph Database Statistics")
                
                graph_stats = insights.get('graph_statistics', {})
                if graph_stats:
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Visits", graph_stats.get('total_visits', 0))
                    with col2:
                        st.metric("Conditions", graph_stats.get('total_conditions', 0))
                    with col3:
                        st.metric("Medications", graph_stats.get('total_medications', 0))
                    with col4:
                        st.metric("Lab Tests", graph_stats.get('total_tests', 0))
                    
                    # Additional statistics
                    if graph_stats.get('condition_counts'):
                        st.subheader("Condition Distribution")
                        condition_data = graph_stats['condition_counts']
                        
                        # Create DataFrame for visualization
                        conditions_df = pd.DataFrame(
                            list(condition_data.items()), 
                            columns=['Condition', 'Count']
                        )
                        conditions_df = conditions_df.sort_values('Count', ascending=False)
                        
                        # Display as chart
                        fig = px.bar(
                            conditions_df, 
                            x='Condition', 
                            y='Count', 
                            title=f"Medical Conditions for Patient {patient_id}"
                        )
                        fig.update_xaxis(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if graph_stats.get('medication_counts'):
                        st.subheader("Medication Usage")
                        medication_data = graph_stats['medication_counts']
                        
                        meds_df = pd.DataFrame(
                            list(medication_data.items()), 
                            columns=['Medication', 'Frequency']
                        )
                        meds_df = meds_df.sort_values('Frequency', ascending=False)
                        
                        fig = px.bar(
                            meds_df, 
                            x='Medication', 
                            y='Frequency', 
                            title=f"Medication History for Patient {patient_id}"
                        )
                        fig.update_xaxis(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No graph statistics available for this patient")
            
            with tab2:
                st.subheader("Timeline Analysis")
                
                timeline_analysis = insights.get('timeline_analysis', {})
                if timeline_analysis:
                    # Timeline summary
                    timeline_summary = timeline_analysis.get('summary', {})
                    if timeline_summary:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("First Visit", timeline_summary.get('first_visit_date', 'N/A'))
                        with col2:
                            st.metric("Last Visit", timeline_summary.get('last_visit_date', 'N/A'))
                        with col3:
                            st.metric("Total Visits", timeline_summary.get('total_visits', 0))
                    
                    # Timeline events
                    timeline_events = timeline_analysis.get('key_events', [])
                    if timeline_events:
                        st.subheader("Key Medical Events")
                        
                        for event in timeline_events:
                            with st.expander(f"{event.get('date', 'N/A')} - {event.get('event_type', 'Medical Event')}"):
                                st.write(f"**Date:** {event.get('date', 'N/A')}")
                                st.write(f"**Type:** {event.get('event_type', 'N/A')}")
                                st.write(f"**Description:** {event.get('description', 'N/A')}")
                                
                                if event.get('related_entities'):
                                    st.write(f"**Related:** {', '.join(event['related_entities'])}")
                    
                    # Visit trends
                    visit_trends = timeline_analysis.get('visit_trends', [])
                    if visit_trends:
                        st.subheader("Visit Frequency Over Time")
                        
                        # Convert to DataFrame and create visualization
                        visits_df = pd.DataFrame(visit_trends)
                        if not visits_df.empty and 'date' in visits_df.columns:
                            fig = px.line(
                                visits_df, 
                                x='date', 
                                y='visit_count',
                                title=f"Visit Frequency for Patient {patient_id}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No timeline analysis available for this patient")
            
            with tab3:
                st.subheader("Drug Interaction Analysis")
                
                drug_interactions = insights.get('potential_drug_interactions', [])
                if drug_interactions:
                    st.warning(f"⚠️ Found {len(drug_interactions)} potential drug interactions")
                    
                    for interaction in drug_interactions:
                        with st.expander(f"Interaction: {interaction.get('drug1', 'Unknown')} ↔ {interaction.get('drug2', 'Unknown')}"):
                            st.write(f"**Drug 1:** {interaction.get('drug1', 'N/A')}")
                            st.write(f"**Drug 2:** {interaction.get('drug2', 'N/A')}")
                            st.write(f"**Severity:** {interaction.get('severity', 'Unknown')}")
                            st.write(f"**Description:** {interaction.get('description', 'N/A')}")
                            
                            if interaction.get('recommendation'):
                                st.info(f"**Recommendation:** {interaction['recommendation']}")
                else:
                    st.success("✅ No potential drug interactions detected")
            
            with tab4:
                st.subheader("Care Gap Analysis")
                
                care_gaps = insights.get('care_gaps', [])
                if care_gaps:
                    st.warning(f"⚠️ Found {len(care_gaps)} potential care gaps")
                    
                    for gap in care_gaps:
                        gap_type = gap.get('gap_type', 'Unknown')
                        priority = gap.get('priority', 'Medium')
                        
                        # Color code by priority
                        if priority == 'High':
                            st.error(f"🔴 **{gap_type}**")
                        elif priority == 'Medium':
                            st.warning(f"🟡 **{gap_type}**")
                        else:
                            st.info(f"🔵 **{gap_type}**")
                        
                        st.write(f"**Description:** {gap.get('description', 'N/A')}")
                        st.write(f"**Recommendation:** {gap.get('recommendation', 'N/A')}")
                        st.write(f"**Priority:** {priority}")
                        st.write("---")
                else:
                    st.success("✅ No care gaps identified")
            
            with tab5:
                st.subheader("Raw Insights Data")
                
                # Display raw JSON data
                st.json(insights)
                
                # API response metadata
                st.subheader("API Response Metadata")
                metadata = {
                    "Message": insights_data.get("message", "N/A"),
                    "Timestamp": insights_data.get("timestamp", "N/A"),
                    "Patient ID": patient_id,
                    "Processing Time": f"{processing_time:.2f}s" if processing_time else "N/A"
                }
                
                for key, value in metadata.items():
                    st.write(f"**{key}:** {value}")
        
        elif insights_data:
            # API returned data but no insights
            st.warning(f"⚠️ No insights could be generated for patient {patient_id}")
            
            # Show debug information
            st.subheader("🔍 API Response Debug Information")
            st.json(insights_data)
            
            if "detail" in insights_data:
                st.error(f"API Error: {insights_data['detail']}")
        
        else:
            # API request failed
            st.error(f"❌ Failed to generate insights for patient {patient_id}")
            st.info("Please check if the MRIA API server is running and the patient ID exists.")
    
    elif patient_id and not generate_insights:
        st.info("👆 Click 'Generate Insights' to analyze the patient data")
    
    elif not patient_id:
        # Show general information when no patient is selected
        st.info("👆 Enter a Patient ID above to generate personalized medical insights")
        
        # Show sample insights and available features
        st.subheader("📋 Available Insight Categories")
        
        insight_categories = [
            {
                "title": "📊 Graph Statistics",
                "description": "Patient visit counts, condition distribution, medication usage patterns"
            },
            {
                "title": "📈 Timeline Analysis", 
                "description": "Medical event timeline, visit trends, treatment progression"
            },
            {
                "title": "💊 Drug Interactions",
                "description": "Potential medication interactions and safety recommendations"
            },
            {
                "title": "⚠️ Care Gaps",
                "description": "Missed appointments, overdue tests, treatment adherence issues"
            }
        ]
        
        for category in insight_categories:
            with st.expander(f"{category['title']}"):
                st.write(category['description'])
        
        # Show sample insights
        st.subheader("💡 Sample Medical Insights")
        
        sample_insights = [
            "📊 Patient has 12 total visits across 8 different conditions",
            "💊 Currently prescribed 5 medications with no detected interactions",
            "📈 Blood pressure readings show improving trend over last 6 months",
            "⚠️ Overdue for annual diabetes screening (last test 14 months ago)",
            "🔍 Similar treatment patterns found in 3 other patients with same condition"
        ]
        
        for insight in sample_insights:
            st.info(insight)

def system_configuration():
    """System configuration and settings"""
    st.subheader("⚙️ System Configuration")
    
    # API Configuration
    st.subheader("API Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        api_url = st.text_input("API Base URL", value=API_BASE_URL)
        
    with col2:
        api_timeout = st.number_input("API Timeout (seconds)", min_value=5, max_value=300, value=30)
    
    # Test API connection
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{api_url}/supervisor/health", timeout=api_timeout)
            if response.status_code == 200:
                st.success("✅ API connection successful")
            else:
                st.error(f"❌ API connection failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ API connection failed: {str(e)}")
    
    # Processing Configuration
    st.subheader("Processing Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        default_workflow = st.selectbox(
            "Default Workflow",
            ["complete_pipeline", "ocr_only", "ocr_to_ner", "document_to_graph", "insight_generation"]
        )
        
    with col2:
        default_priority = st.selectbox("Default Priority", ["normal", "high", "urgent", "low"])
    
    # OCR Configuration
    st.subheader("OCR Configuration")
    
    ocr_provider = st.selectbox("OCR Provider", ["tesseract", "azure_form_recognizer", "google_vision"])
    
    if ocr_provider == "azure_form_recognizer":
        azure_key = st.text_input("Azure Cognitive Services Key", type="password")
        azure_endpoint = st.text_input("Azure Endpoint")
        
    elif ocr_provider == "google_vision":
        google_credentials = st.text_area("Google Vision API Credentials (JSON)")
    
    # NER Configuration
    st.subheader("NER Configuration")
    
    ner_models = st.multiselect(
        "NER Models",
        ["scispacy", "biobert", "med7", "clinical_bert"],
        default=["scispacy", "med7"]
    )
    
    # Database Configuration
    st.subheader("Database Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        neo4j_username = st.text_input("Neo4j Username", value="neo4j")
        
    with col2:
        neo4j_password = st.text_input("Neo4j Password", type="password")
        redis_url = st.text_input("Redis URL", value="redis://localhost:6379")
    
    # Save configuration
    if st.button("Save Configuration", type="primary"):
        config = {
            "api_url": api_url,
            "api_timeout": api_timeout,
            "default_workflow": default_workflow,
            "default_priority": default_priority,
            "ocr_provider": ocr_provider,
            "ner_models": ner_models,
            "neo4j_uri": neo4j_uri,
            "neo4j_username": neo4j_username,
            "redis_url": redis_url
        }
        
        # Save to session state
        st.session_state.config = config
        st.success("✅ Configuration saved successfully")

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">🏥 MRIA - Medical Records Insight Agent</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    pages = {
        "🏠 Dashboard": "dashboard",
        "📄 Document Upload": "upload",
        "📊 Job Monitoring": "monitoring",
        "🕸️ Knowledge Graph": "graph",
        "💬 Medical Chat": "chat",
        "📈 Insights": "insights",
        "⚙️ Configuration": "config"
    }
    
    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
    page_key = pages[selected_page]
    
    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    # Check API status
    api_status = check_api_health()
    if api_status:
        st.sidebar.success("✅ API Online")
    else:
        st.sidebar.error("❌ API Offline")
    
    # Quick stats
    if api_status:
        queue_status = make_api_request("supervisor/queue/status")
        if queue_status:
            st.sidebar.metric("Active Jobs", queue_status.get('running_jobs', 0))
            st.sidebar.metric("Queue Health", queue_status.get('queue_health', 'unknown').title())
    
    # Page routing
    if page_key == "dashboard":
        display_agent_status()
        
    elif page_key == "upload":
        upload_documents()
        
    elif page_key == "monitoring":
        monitor_jobs()
        
    elif page_key == "graph":
        knowledge_graph_viewer()
        
    elif page_key == "chat":
        medical_chat_interface()
        
    elif page_key == "insights":
        insights_dashboard()
        
    elif page_key == "config":
        system_configuration()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>MRIA - Medical Records Insight Agent | AI-Powered Medical Data Analysis</p>
            <p>Built with Streamlit • Powered by FastAPI • Knowledge Graph with Neo4j</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
