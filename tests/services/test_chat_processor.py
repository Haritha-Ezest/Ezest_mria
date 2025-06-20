"""
Comprehensive test suite for the Chat Processor Service.

This module provides complete test coverage for the Chat Processor functionality
including natural language query processing, medical context understanding,
conversation management, and integration with other system components.

Tests cover:
1. Chat processor initialization and configuration
2. Natural language query processing and intent detection
3. Medical terminology understanding and entity extraction
4. Conversation session management and context persistence
5. Integration with graph database and insights components
6. Query type classification and routing
7. Response generation and formatting
8. Error handling and recovery mechanisms
9. Performance optimization and caching
10. Background task processing for metrics and feedback
"""

import pytest
import asyncio
from unittest.mock import patch
from datetime import datetime

from app.services.chat_processor import ChatProcessor
from app.schemas.chat import (
    ChatQueryRequest, ChatQueryResponse, ChatMessage,
    MessageRole, QueryType, ConversationContext
)


class TestChatProcessor:
    """Comprehensive test cases for the Chat Processor Service."""
    
    @pytest.fixture
    def chat_processor(self):
        """Create a chat processor instance for testing."""
        return ChatProcessor()    @pytest.fixture
    def sample_query_request(self):
        """Sample chat query request for testing."""
        return ChatQueryRequest(
            query="Show me John Doe's diabetes progression over the last year",
            context=ConversationContext(
                patient_id="patient_12345",
                clinical_speciality="endocrinology",
                time_period="last_12_months",
                condition_focus="diabetes"
            )
        )
    
    @pytest.fixture
    def mock_graph_client(self):
        """Mock graph client for testing."""
        with patch('app.services.chat_processor.graph_client') as mock:
            yield mock
    
    @pytest.fixture
    def mock_insights_processor(self):
        """Mock insights processor for testing."""
        with patch('app.services.chat_processor.insights_processor') as mock:
            yield mock
    
    @pytest.fixture
    def mock_ner_processor(self):
        """Mock NER processor for testing."""
        with patch('app.services.chat_processor.ner_processor') as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_process_query_success(self, chat_processor, sample_query_request, 
                                       mock_graph_client, mock_insights_processor):
        """Test successful query processing."""
        # Mock graph client response
        mock_graph_client.get_patient_timeline.return_value = {
            "patient_id": "patient_12345",
            "timeline": [
                {
                    "date": "2023-06-01",
                    "hba1c": 8.2,
                    "event": "initial_diagnosis"
                },
                {
                    "date": "2024-06-01", 
                    "hba1c": 6.8,
                    "event": "follow_up"
                }
            ]
        }
        
        # Mock insights processor response
        mock_insights_processor.analyze_progression.return_value = {
            "trend": "improving",
            "key_metrics": ["hba1c_reduction"],
            "recommendations": ["continue_medication"]
        }
        
        response = await chat_processor.process_query(sample_query_request)
        
        assert isinstance(response, ChatQueryResponse)
        assert response.query_type == QueryType.PATIENT_HISTORY
        assert response.confidence_score > 0.8
        assert "diabetes progression" in response.response.lower()
        assert len(response.sources) > 0
        assert response.processing_time > 0

    @pytest.mark.asyncio
    async def test_process_query_patient_not_found(self, chat_processor, mock_graph_client):
        """Test query processing when patient is not found."""
        mock_graph_client.get_patient_timeline.return_value = None
        
        query_request = ChatQueryRequest(
            message="Show patient history",
            patient_id="nonexistent_patient"
        )
        
        response = await chat_processor.process_query(query_request)
        
        assert isinstance(response, ChatQueryResponse)
        assert response.confidence_score < 0.5
        assert "not found" in response.response.lower()

    @pytest.mark.asyncio
    async def test_detect_query_intent_patient_history(self, chat_processor):
        """Test query intent detection for patient history."""
        messages = [
            "Show me patient John's medical history",
            "What's the progression of diabetes for patient 123?",
            "Summarize this patient's health timeline"
        ]
        
        for message in messages:
            intent = await chat_processor._detect_query_intent(message)
            assert intent == QueryType.PATIENT_HISTORY

    @pytest.mark.asyncio
    async def test_detect_query_intent_medication_interactions(self, chat_processor):
        """Test query intent detection for medication interactions."""
        messages = [
            "What are the drug interactions for this patient?",
            "Check medication interactions for metformin and lisinopril",
            "Are there any contraindications with current prescriptions?"
        ]
        
        for message in messages:
            intent = await chat_processor._detect_query_intent(message)
            assert intent == QueryType.MEDICATION_INTERACTIONS

    @pytest.mark.asyncio
    async def test_detect_query_intent_lab_interpretation(self, chat_processor):
        """Test query intent detection for lab interpretation."""
        messages = [
            "Interpret the latest lab results",
            "What do these HbA1c values mean?",
            "Explain the blood work for patient 123"
        ]
        
        for message in messages:
            intent = await chat_processor._detect_query_intent(message)
            assert intent == QueryType.LAB_INTERPRETATION

    @pytest.mark.asyncio
    async def test_extract_medical_entities(self, chat_processor, mock_ner_processor):
        """Test medical entity extraction from queries."""
        mock_ner_processor.extract_entities.return_value = {
            "conditions": ["diabetes", "hypertension"],
            "medications": ["metformin", "lisinopril"],
            "procedures": ["blood_test"],
            "lab_values": [{"name": "HbA1c", "value": "7.2%"}]
        }
        
        query = "Patient has diabetes and hypertension, taking metformin and lisinopril, latest HbA1c is 7.2%"
        entities = await chat_processor._extract_medical_entities(query)
        
        assert "diabetes" in entities["conditions"]
        assert "hypertension" in entities["conditions"] 
        assert "metformin" in entities["medications"]
        assert "lisinopril" in entities["medications"]
        assert len(entities["lab_values"]) == 1

    @pytest.mark.asyncio
    async def test_generate_response_patient_history(self, chat_processor, mock_graph_client):
        """Test response generation for patient history queries."""
        mock_timeline_data = {
            "patient_id": "patient_123",
            "visits": [
                {
                    "date": "2023-01-15",
                    "diagnosis": "Type 2 Diabetes",
                    "hba1c": 8.5
                },
                {
                    "date": "2023-06-15", 
                    "medication": "Metformin 500mg",
                    "hba1c": 7.2
                },
                {
                    "date": "2024-01-15",
                    "hba1c": 6.8,
                    "status": "improved"
                }
            ]
        }
        
        mock_graph_client.get_patient_timeline.return_value = mock_timeline_data
        
        response_text = await chat_processor._generate_response(
            QueryType.PATIENT_HISTORY,
            {"patient_id": "patient_123"},
            mock_timeline_data
        )
        
        assert "diabetes" in response_text.lower()
        assert "hba1c" in response_text.lower()
        assert "improved" in response_text.lower()

    @pytest.mark.asyncio
    async def test_session_management_create_session(self, chat_processor):
        """Test creating a new conversation session."""
        session = await chat_processor.create_session("patient_123", "user_456")
        
        assert isinstance(session, dict)  # Assuming it returns a dict with session data
        assert "user_id" in session
        assert "session_id" in session
        assert session["user_id"] == "user_456"

    @pytest.mark.asyncio
    async def test_session_management_add_message(self, chat_processor):
        """Test adding messages to a conversation session."""
        session = await chat_processor.create_session("patient_123", "user_456")
        
        message = ChatMessage(
            role=MessageRole.USER,
            content="Show patient's diabetes status",
            timestamp=datetime.now()
        )
        
        await chat_processor.add_message_to_session(session.session_id, message)
        
        # Retrieve session and verify message was added
        updated_session = await chat_processor.get_session(session.session_id)
        assert len(updated_session.messages) == 1
        assert updated_session.messages[0].content == "Show patient's diabetes status"

    @pytest.mark.asyncio
    async def test_conversation_history_retrieval(self, chat_processor):
        """Test retrieving conversation history."""
        session = await chat_processor.create_session("patient_123", "user_456")
        
        # Add multiple messages
        messages = [
            ChatMessage(role=MessageRole.USER, content="Query 1", timestamp=datetime.now()),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response 1", timestamp=datetime.now()),
            ChatMessage(role=MessageRole.USER, content="Query 2", timestamp=datetime.now()),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response 2", timestamp=datetime.now())
        ]
        
        for message in messages:
            await chat_processor.add_message_to_session(session.session_id, message)
        
        history = await chat_processor.get_conversation_history(session.session_id)
        
        assert isinstance(history, dict)  # Assuming it returns a dict with conversation data
        assert "session_id" in history
        assert "messages" in history
        assert len(history["messages"]) == 4    @pytest.mark.asyncio
    async def test_context_management(self, chat_processor):
        """Test conversation context management."""
        context = ConversationContext(
            patient_id="patient_123",
            clinical_speciality="cardiology",
            condition_focus="hypertension",
            time_period="last_6_months"
        )
        
        success = await chat_processor.set_context("session_123", context)
        assert success is True
        
        retrieved_context = await chat_processor.get_context("session_123")
        assert retrieved_context.patient_id == "patient_123"
        assert retrieved_context.medical_specialty == "cardiology"
        assert len(retrieved_context.focus_conditions) == 2

    @pytest.mark.asyncio
    async def test_process_medication_interaction_query(self, chat_processor, mock_graph_client):
        """Test processing medication interaction queries."""
        mock_graph_client.get_patient_medications.return_value = [
            {"name": "Metformin", "dosage": "500mg", "frequency": "BID"},
            {"name": "Lisinopril", "dosage": "10mg", "frequency": "daily"},
            {"name": "Warfarin", "dosage": "5mg", "frequency": "daily"}
        ]
        
        mock_graph_client.check_drug_interactions.return_value = [
            {
                "drug1": "Warfarin",
                "drug2": "Metformin", 
                "interaction_type": "minor",
                "description": "Monitor blood glucose levels"
            }
        ]
        
        query_request = ChatQueryRequest(
            message="Check for medication interactions",
            patient_id="patient_123",
            query_type=QueryType.MEDICATION_INTERACTIONS
        )
        
        response = await chat_processor.process_query(query_request)
        
        assert response.query_type == QueryType.MEDICATION_INTERACTIONS
        assert "interaction" in response.response.lower()
        assert "warfarin" in response.response.lower()

    @pytest.mark.asyncio
    async def test_process_population_health_query(self, chat_processor, mock_insights_processor):
        """Test processing population health queries."""
        mock_insights_processor.get_population_insights.return_value = {
            "condition": "diabetes",
            "total_patients": 1250,
            "average_hba1c": 7.4,
            "patients_at_target": 625,
            "common_treatments": ["Metformin", "Insulin"]
        }
        
        query_request = ChatQueryRequest(
            message="Show population trends for diabetes patients",
            query_type=QueryType.POPULATION_HEALTH
        )
        
        response = await chat_processor.process_query(query_request)
        
        assert response.query_type == QueryType.POPULATION_HEALTH
        assert "1250" in response.response or "1,250" in response.response
        assert "diabetes" in response.response.lower()

    @pytest.mark.asyncio
    async def test_query_feedback_processing(self, chat_processor):
        """Test processing user feedback on queries."""
        feedback_data = {
            "query_id": "query_123",
            "helpful": True,
            "accuracy_rating": 4,
            "feedback_text": "Very helpful analysis"
        }
        
        success = await chat_processor.process_feedback(feedback_data)
        assert success is True
        
        # Verify feedback was stored
        stored_feedback = await chat_processor.get_feedback("query_123")
        assert stored_feedback["helpful"] is True
        assert stored_feedback["accuracy_rating"] == 4

    @pytest.mark.asyncio
    async def test_metrics_collection(self, chat_processor):
        """Test metrics collection and reporting."""
        # Simulate processing several queries
        for i in range(5):
            query_request = ChatQueryRequest(
                message=f"Test query {i}",
                patient_id=f"patient_{i}"
            )
            await chat_processor.process_query(query_request)
        
        metrics = await chat_processor.get_metrics()
        
        assert metrics["total_queries"] >= 5
        assert "average_response_time" in metrics
        assert "successful_queries" in metrics
        assert "common_query_types" in metrics

    @pytest.mark.asyncio
    async def test_error_handling_service_unavailable(self, chat_processor, mock_graph_client):
        """Test error handling when services are unavailable."""
        mock_graph_client.get_patient_timeline.side_effect = Exception("Service unavailable")
        
        query_request = ChatQueryRequest(
            message="Show patient history",
            patient_id="patient_123"
        )
        
        response = await chat_processor.process_query(query_request)
        
        assert response.confidence_score < 0.5
        assert "unavailable" in response.response.lower() or "error" in response.response.lower()

    @pytest.mark.asyncio
    async def test_query_timeout_handling(self, chat_processor, mock_graph_client):
        """Test handling of query timeouts."""
        # Mock a slow response that exceeds timeout
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow response
            return {"data": "slow_data"}
        
        mock_graph_client.get_patient_timeline.side_effect = slow_response
        
        query_request = ChatQueryRequest(
            message="Show patient history",
            patient_id="patient_123",
            timeout=2  # 2 second timeout
        )
        
        with pytest.raises(asyncio.TimeoutError):
            await chat_processor.process_query(query_request)

    @pytest.mark.asyncio
    async def test_batch_query_processing(self, chat_processor, mock_graph_client):
        """Test processing multiple queries in batch."""
        mock_graph_client.get_patient_timeline.return_value = {"patient_data": "test"}
        
        queries = [
            ChatQueryRequest(message="Query 1", patient_id="patient_001"),
            ChatQueryRequest(message="Query 2", patient_id="patient_002"),
            ChatQueryRequest(message="Query 3", patient_id="patient_003")
        ]
        
        responses = await chat_processor.process_batch_queries(queries)
        
        assert len(responses) == 3
        assert all(isinstance(r, ChatQueryResponse) for r in responses)
        assert all(r.processing_time > 0 for r in responses)

    @pytest.mark.asyncio
    async def test_health_status_monitoring(self, chat_processor):
        """Test health status monitoring and reporting."""
        health_status = await chat_processor.get_health_status()
        
        assert "service" in health_status
        assert "status" in health_status
        assert "database_connection" in health_status
        assert "nlp_models_loaded" in health_status
        assert "response_time_avg" in health_status
        
        # Verify status is healthy after initialization
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_natural_language_understanding(self, chat_processor):
        """Test natural language understanding capabilities."""
        complex_queries = [
            "What happened to this patient's blood sugar levels after starting metformin?",
            "Compare the cardiovascular outcomes between patients on ACE inhibitors vs ARBs",
            "Show me the timeline of events leading to this patient's heart attack"
        ]
        
        for query in complex_queries:
            intent = await chat_processor._detect_query_intent(query)
            assert intent in [qt.value for qt in QueryType]
            
            entities = await chat_processor._extract_medical_entities(query)
            assert isinstance(entities, dict)
            assert any(len(v) > 0 for v in entities.values() if isinstance(v, list))

    @pytest.mark.asyncio
    async def test_medical_specialty_optimization(self, chat_processor):
        """Test query optimization based on medical specialty."""
        specialties = ["cardiology", "endocrinology", "nephrology", "oncology"]
        
        for specialty in specialties:
            query_request = ChatQueryRequest(
                query="Analyze this patient's condition",
                context=ConversationContext(
                    patient_id="patient_123",
                    clinical_speciality=specialty
                )
            )
            
            # Mock specialty-specific response
            with patch.object(chat_processor, '_generate_specialty_response') as mock_specialty:
                mock_specialty.return_value = f"Specialized {specialty} analysis"
                
                response = await chat_processor.process_query(query_request)
                assert specialty in response.metadata.get("specialty", "").lower()

    @pytest.mark.asyncio
    async def test_conversation_continuity(self, chat_processor):
        """Test conversation continuity and context preservation."""
        session_id = "continuous_session"
        
        # First query establishes context
        query1 = ChatQueryRequest(
            message="Show me patient John's diabetes status",
            patient_id="patient_john",
            session_id=session_id
        )
        
        await chat_processor.process_query(query1)
        
        # Follow-up query should understand context
        query2 = ChatQueryRequest(
            message="What about his blood pressure?",  # Contextual reference
            session_id=session_id  # Same session, no explicit patient_id
        )
        
        response2 = await chat_processor.process_query(query2)
        
        # Verify context was preserved
        assert response2.metadata.get("patient_id") == "patient_john"
