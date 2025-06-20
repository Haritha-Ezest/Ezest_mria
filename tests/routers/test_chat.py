"""
Comprehensive tests for the chat router endpoints.

This module tests all endpoints in the chat router, including:
- Chat query processing with natural language understanding
- Conversation session management and context persistence
- Medical terminology processing and intent detection
- Integration with Graph Database and Insights components
- Error handling and validation for medical queries
- Background task processing for metrics and feedback
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch
from datetime import datetime, timedelta

from app.main import app
from app.schemas.chat import (
    ChatQueryResponse, ChatMessage,
    MessageRole, QueryType, ConversationHistory
)


class TestChatRouter:
    """Test class for chat router endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_chat_processor(self):
        """Mock chat processor for testing."""
        with patch('app.routers.chat.chat_processor') as mock:
            yield mock

    @pytest.fixture
    def sample_chat_request(self):
        """Sample chat query request."""
        return {
            "message": "Show me John Doe's diabetes progression over the last year",
            "patient_id": "patient_12345",
            "session_id": "session_789",
            "query_type": "patient_history",
            "context": {
                "medical_specialty": "endocrinology",
                "time_range": "last_12_months",
                "focus_conditions": ["diabetes"]
            }
        }

    @pytest.fixture
    def sample_medical_queries(self):
        """Sample medical queries for different scenarios."""
        return {
            "patient_history": "Summarize this patient's diabetes progression over the last year",
            "treatment_comparison": "Compare this patient's HbA1c trends with similar patients",
            "medication_interactions": "What are the medication interactions for this patient's current prescriptions?", 
            "lab_interpretation": "Interpret the latest lab results for patient ID 12345",
            "population_health": "Show me patients with similar presentations who responded well to metformin",
            "clinical_decision": "What treatment options would you recommend for this diabetic patient?",
            "risk_assessment": "Assess cardiovascular risk factors for this patient",
            "timeline_analysis": "Create a timeline of medical events for patient 12345"
        }

    def test_chat_query_success(self, client, mock_chat_processor, sample_chat_request):
        """Test successful chat query processing."""
        # Mock response
        mock_response = ChatQueryResponse(
            response="Based on the patient's medical history, John Doe's diabetes progression shows significant improvement. His HbA1c decreased from 8.2% to 6.8% over the past year following initiation of metformin therapy.",
            query_type=QueryType.PATIENT_HISTORY,
            confidence_score=0.95,
            sources=["Graph Database", "Patient Timeline"],
            entities_mentioned=["diabetes", "HbA1c", "metformin"],
            session_id="session_789",
            processing_time=2.3,
            recommendations=["Continue current medication regimen", "Schedule quarterly HbA1c monitoring"],
            metadata={
                "patient_records_accessed": 3,
                "time_range_analyzed": "2023-06-01 to 2024-06-01",
                "data_sources": ["lab_results", "prescriptions", "visit_notes"]
            }
        )
        
        mock_chat_processor.process_query.return_value = mock_response
        
        response = client.post("/chat/query", json=sample_chat_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "Based on the patient's medical history" in data["response"]
        assert data["confidence_score"] == 0.95
        assert data["query_type"] == "patient_history"
        assert len(data["entities_mentioned"]) == 3
        assert len(data["recommendations"]) == 2

    def test_chat_query_validation_error(self, client):
        """Test chat query with validation errors."""
        invalid_request = {
            "message": "",  # Empty message should fail validation
            "patient_id": "invalid_id_format"
        }
        
        response = client.post("/chat/query", json=invalid_request)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_chat_query_different_types(self, client, mock_chat_processor, sample_medical_queries):
        """Test different types of medical queries."""
        for query_type, message in sample_medical_queries.items():
            mock_response = ChatQueryResponse(
                response=f"Mock response for {query_type}",
                query_type=QueryType(query_type),
                confidence_score=0.9,
                sources=["Mock Source"],
                entities_mentioned=["test_entity"],
                session_id="test_session",
                processing_time=1.0
            )
            
            mock_chat_processor.process_query.return_value = mock_response
            
            request_data = {
                "message": message,
                "patient_id": "patient_123",
                "query_type": query_type
            }
            
            response = client.post("/chat/query", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["query_type"] == query_type
            assert "Mock response" in data["response"]

    @patch('app.routers.chat.chat_processor')
    def test_chat_query_processing_error(self, mock_processor, client, sample_chat_request):
        """Test chat query processing error handling."""
        mock_processor.process_query.side_effect = Exception("Processing failed")
        
        response = client.post("/chat/query", json=sample_chat_request)
        
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "Processing failed" in data["error"]

    def test_get_conversation_history_success(self, client, mock_chat_processor):
        """Test retrieving conversation history."""
        mock_history = ConversationHistory(
            session_id="session_789",
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content="Show me patient's diabetes status",
                    timestamp=datetime.now() - timedelta(minutes=5)
                ),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="The patient's latest HbA1c is 6.8%, showing good control.",
                    timestamp=datetime.now() - timedelta(minutes=4)
                )
            ],
            total_messages=2,
            session_start=datetime.now() - timedelta(hours=1),
            last_activity=datetime.now() - timedelta(minutes=4)
        )
        
        mock_chat_processor.get_conversation_history.return_value = mock_history
        
        response = client.get("/chat/history/session_789")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["session_id"] == "session_789"
        assert data["total_messages"] == 2
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

    def test_get_conversation_history_not_found(self, client, mock_chat_processor):
        """Test retrieving non-existent conversation history."""
        mock_chat_processor.get_conversation_history.return_value = None
        
        response = client.get("/chat/history/nonexistent_session")
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    def test_set_conversation_context_success(self, client, mock_chat_processor):
        """Test setting conversation context."""
        context_data = {
            "patient_id": "patient_12345",
            "medical_specialty": "cardiology",
            "focus_conditions": ["hypertension", "coronary_artery_disease"],
            "time_range": "last_6_months",
            "provider_id": "dr_smith_123"
        }
        
        mock_chat_processor.set_context.return_value = True
        
        response = client.post("/chat/context", json=context_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "Context set successfully" in data["message"]

    def test_provide_query_feedback_success(self, client, mock_chat_processor):
        """Test providing feedback for a query."""
        feedback_data = {
            "query_id": "query_456",
            "helpful": True,
            "accuracy_rating": 4,
            "feedback_text": "Very helpful summary of patient's condition",
            "suggestions": ["Include more recent lab values"]
        }
        
        mock_chat_processor.process_feedback.return_value = True
        
        response = client.post("/chat/feedback", json=feedback_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "Feedback received successfully" in data["message"]

    def test_get_chat_metrics_success(self, client, mock_chat_processor):
        """Test retrieving chat system metrics."""
        mock_metrics = {
            "total_queries": 1250,
            "successful_queries": 1180,
            "average_response_time": 2.3,
            "common_query_types": {
                "patient_history": 45,
                "medication_interactions": 20,
                "lab_interpretation": 15
            },
            "user_satisfaction": 4.2,
            "system_uptime": "99.8%",
            "last_updated": datetime.now().isoformat()
        }
        
        mock_chat_processor.get_metrics.return_value = mock_metrics
        
        response = client.get("/chat/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["metrics"]["total_queries"] == 1250
        assert data["metrics"]["successful_queries"] == 1180
        assert data["metrics"]["user_satisfaction"] == 4.2

    def test_get_chat_health_success(self, client, mock_chat_processor):
        """Test chat service health check."""
        mock_health = {
            "service": "chat",
            "status": "healthy",
            "database_connection": True,
            "nlp_models_loaded": True,
            "graph_integration": True,
            "response_time_avg": 2.1,
            "memory_usage": "45%",
            "active_sessions": 23
        }
        
        mock_chat_processor.get_health_status.return_value = mock_health
        
        response = client.get("/chat/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["service"] == "chat"
        assert data["status"] == "healthy"
        assert data["database_connection"] is True
        assert data["nlp_models_loaded"] is True
        assert data["graph_integration"] is True

    def test_chat_query_with_context(self, client, mock_chat_processor):
        """Test chat query with specific medical context."""
        request_with_context = {
            "message": "What's the latest cardiovascular risk assessment?",
            "patient_id": "patient_12345",
            "context": {
                "medical_specialty": "cardiology",
                "focus_conditions": ["coronary_artery_disease", "hypertension"],
                "time_range": "last_3_months",
                "include_family_history": True
            }
        }
        
        mock_response = ChatQueryResponse(
            response="Based on recent assessments, the patient has moderate cardiovascular risk with controlled hypertension and stable CAD.",
            query_type=QueryType.RISK_ASSESSMENT,
            confidence_score=0.88,
            sources=["Risk Calculator", "Recent Labs", "Echo Results"],
            entities_mentioned=["cardiovascular risk", "hypertension", "CAD"],
            session_id="new_session",
            processing_time=3.1,
            recommendations=["Continue ACE inhibitor", "Schedule stress test in 6 months"]
        )
        
        mock_chat_processor.process_query.return_value = mock_response
        
        response = client.post("/chat/query", json=request_with_context)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "cardiovascular risk" in data["response"]
        assert data["confidence_score"] == 0.88
        assert len(data["sources"]) == 3

    def test_batch_chat_queries(self, client, mock_chat_processor):
        """Test processing multiple chat queries in batch."""
        batch_queries = [
            {"message": "Patient's latest glucose levels?", "patient_id": "patient_001"},
            {"message": "Current medications for hypertension?", "patient_id": "patient_002"}, 
            {"message": "Risk factors for diabetes?", "patient_id": "patient_003"}
        ]
        
        # Mock batch processing
        mock_responses = [
            ChatQueryResponse(
                response="Latest glucose: 140 mg/dL",
                query_type=QueryType.LAB_INTERPRETATION,
                confidence_score=0.95,
                sources=["Lab Results"],
                entities_mentioned=["glucose"],
                session_id="batch_session_1",
                processing_time=1.2
            ),
            ChatQueryResponse(
                response="Current BP medications: Lisinopril 10mg daily",
                query_type=QueryType.MEDICATION_INTERACTIONS,
                confidence_score=0.92,
                sources=["Prescription Records"],
                entities_mentioned=["Lisinopril"],
                session_id="batch_session_2", 
                processing_time=1.5
            ),
            ChatQueryResponse(
                response="Risk factors include family history and obesity",
                query_type=QueryType.RISK_ASSESSMENT,
                confidence_score=0.87,
                sources=["Risk Assessment Tool"],
                entities_mentioned=["family history", "obesity"],
                session_id="batch_session_3",
                processing_time=1.8
            )
        ]
        
        mock_chat_processor.process_batch_queries.return_value = mock_responses
        
        response = client.post("/chat/batch", json={"queries": batch_queries})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["results"]) == 3
        assert all(result["success"] for result in data["results"])

    def test_long_running_query_timeout(self, client, mock_chat_processor):
        """Test handling of long-running queries with timeout."""
        mock_chat_processor.process_query.side_effect = asyncio.TimeoutError("Query timed out")
        
        timeout_request = {
            "message": "Perform comprehensive analysis of all patients with diabetes in the last decade",
            "patient_id": "all_patients",
            "timeout": 5  # 5 second timeout
        }
        
        response = client.post("/chat/query", json=timeout_request)
        
        assert response.status_code == 408  # Request Timeout
        data = response.json()
        assert data["success"] is False
        assert "timeout" in data["error"].lower()

    def test_medical_specialty_specific_queries(self, client, mock_chat_processor):
        """Test queries specific to different medical specialties."""
        specialties = {
            "cardiology": "Assess cardiac function based on latest echo",
            "endocrinology": "Evaluate thyroid function and diabetes control",
            "nephrology": "Review kidney function and dialysis needs",
            "oncology": "Analyze tumor markers and treatment response"
        }
        
        for specialty, query in specialties.items():
            mock_response = ChatQueryResponse(
                response=f"Specialized {specialty} analysis completed",
                query_type=QueryType.CLINICAL_DECISION,
                confidence_score=0.9,
                sources=[f"{specialty.title()} Guidelines"],
                entities_mentioned=[specialty],
                session_id=f"{specialty}_session",
                processing_time=2.0,
                metadata={"specialty": specialty}
            )
            
            mock_chat_processor.process_query.return_value = mock_response
            
            request_data = {
                "message": query,
                "patient_id": "patient_123",
                "context": {"medical_specialty": specialty}
            }
            
            response = client.post("/chat/query", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert specialty in data["response"]
            assert data["metadata"]["specialty"] == specialty
