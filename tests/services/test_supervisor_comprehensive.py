"""
Comprehensive test suite for the Supervisor Agent.

This module provides complete test coverage for the Supervisor Agent functionality
including workflow orchestration, agent coordination, job management, and error handling.

Tests cover:
1. Supervisor Agent initialization and configuration
2. Workflow orchestration and state management
3. Agent coordination and task distribution
4. Job queuing and priority management
5. Error handling and recovery mechanisms
6. Progress tracking and monitoring
7. Redis integration and state persistence
8. LangGraph workflow execution
9. Agent communication and data flow
10. Performance and scalability testing
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from app.services.supervisor import LangChainSupervisor, SupervisorState, SupervisorCallbackHandler
from app.schemas.supervisor import (
    JobRequest, JobResponse, JobStatus, JobType, WorkflowType, 
    JobPriority, AgentType, QueueStatus, WorkflowConfiguration
)
from app.schemas.ocr import OCRResponse, OCRStatus
from app.schemas.ner import NERResponse, NERStatus
from app.schemas.chunking import ChunkingResponse, ChunkingStatus
from app.schemas.graph import GraphResponse, GraphStatus


class TestSupervisorAgent:
    """Comprehensive test cases for the Supervisor Agent."""
    
    @pytest.fixture
    async def supervisor(self):
        """Create a fully configured supervisor instance for testing."""
        with patch('redis.Redis') as mock_redis:
            # Mock Redis client with comprehensive functionality
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            # Mock Redis operations
            mock_redis_instance.lpush = AsyncMock(return_value=1)
            mock_redis_instance.rpop = AsyncMock(return_value=b'test_job_id')
            mock_redis_instance.hset = AsyncMock(return_value=1)
            mock_redis_instance.hgetall = AsyncMock(return_value={})
            mock_redis_instance.expire = AsyncMock(return_value=True)
            mock_redis_instance.ping = AsyncMock(return_value=True)
            
            supervisor = LangChainSupervisor(redis_url="redis://localhost:6379")
            supervisor.redis_client = mock_redis_instance
            supervisor.is_initialized = True
            
            # Mock workflow graph with enhanced functionality
            supervisor.workflow_graph = Mock()
            supervisor.workflow_graph.ainvoke = AsyncMock()
            supervisor.workflow_graph.stream = AsyncMock()
            
            # Mock checkpointer
            supervisor.checkpointer = Mock()
            supervisor.checkpointer.put = AsyncMock()
            supervisor.checkpointer.get = AsyncMock()
            
            return supervisor
    
    @pytest.fixture
    def complete_pipeline_job(self):
        """Create a complete pipeline job request."""
        return JobRequest(
            job_type=JobType.DOCUMENT_PROCESSING,
            workflow_type=WorkflowType.COMPLETE_PIPELINE,
            priority=JobPriority.HIGH,
            patient_id="patient_12345",
            document_ids=["doc_001", "doc_002"],
            file_paths=["/uploads/lab_report.pdf", "/uploads/prescription.png"],
            processing_config={
                "ocr_config": {
                    "engine": "tesseract",
                    "document_type": "lab_report",
                    "preprocessing": "medical_optimized",
                    "confidence_threshold": 0.8,
                    "extract_tables": True
                },
                "ner_config": {
                    "processing_mode": "medical",
                    "entity_linking": True,
                    "confidence_threshold": 0.7,
                    "extract_temporal": True
                },
                "chunking_config": {
                    "strategy": "medical_visits",
                    "chunk_size": 1000,
                    "overlap": 200,
                    "create_timeline": True
                },
                "graph_config": {
                    "create_patient_graph": True,
                    "link_knowledge_base": True,
                    "generate_insights": True
                }
            },
            metadata={
                "source": "medical_records",
                "department": "cardiology",
                "provider": "Dr. Smith",
                "urgency": "high"
            }
        )
    
    @pytest.fixture
    def sample_supervisor_state(self):
        """Create a sample supervisor state for testing."""
        return SupervisorState(
            job_id="test_job_123",
            status=JobStatus.RUNNING,
            current_agent_index=0,
            next_action=AgentType.OCR.value,
            file_paths=["/uploads/test_document.pdf"],
            processing_config={
                "ocr_config": {
                    "engine": "tesseract",
                    "document_type": "prescription"
                }
            },
            agent_results={},
            workflow_type=WorkflowType.COMPLETE_PIPELINE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"test": "data"}
        )

    # Test 1: Supervisor Agent Initialization and Configuration
    async def test_supervisor_initialization_success(self):
        """Test successful supervisor initialization with all components."""
        with patch('redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping = AsyncMock(return_value=True)
            
            supervisor = LangChainSupervisor(redis_url="redis://localhost:6379")
            
            # Test initialization
            await supervisor.initialize()
            
            # Assertions
            assert supervisor.is_initialized
            assert supervisor.redis_client is not None
            assert supervisor.workflow_graph is not None
            assert supervisor.checkpointer is not None
            assert supervisor.workflows is not None
            assert supervisor.agent_configs is not None
            
            # Verify Redis connection
            mock_redis_instance.ping.assert_called_once()

    async def test_supervisor_initialization_redis_failure(self):
        """Test supervisor initialization with Redis connection failure."""
        with patch('redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping = AsyncMock(side_effect=Exception("Redis connection failed"))
            
            supervisor = LangChainSupervisor(redis_url="redis://localhost:6379")
            
            # Test initialization failure
            with pytest.raises(Exception, match="Redis connection failed"):
                await supervisor.initialize()
    
    async def test_supervisor_workflow_configuration(self, supervisor):
        """Test supervisor workflow configuration and graph building."""
        # Test workflow configuration
        workflows = supervisor.get_available_workflows()
        
        assert WorkflowType.OCR_ONLY in workflows
        assert WorkflowType.OCR_TO_NER in workflows
        assert WorkflowType.COMPLETE_PIPELINE in workflows
        assert WorkflowType.BATCH_DOCUMENTS in workflows
        
        # Test agent configuration
        agent_configs = supervisor.get_agent_configurations()
        
        assert AgentType.OCR in agent_configs
        assert AgentType.NER in agent_configs
        assert AgentType.CHUNKING in agent_configs
        assert AgentType.GRAPH in agent_configs

    # Test 2: Job Management and Queuing
    async def test_enqueue_job_complete_pipeline(self, supervisor, complete_pipeline_job):
        """Test enqueuing a complete pipeline job with all configurations."""
        # Mock Redis operations
        supervisor.redis_client.lpush = AsyncMock(return_value=1)
        supervisor.redis_client.hset = AsyncMock(return_value=1)
        supervisor.redis_client.expire = AsyncMock(return_value=True)
        
        # Enqueue job
        job_response = await supervisor.enqueue_job(complete_pipeline_job)
        
        # Assertions
        assert isinstance(job_response, JobResponse)
        assert job_response.status == JobStatus.QUEUED
        assert job_response.job_type == JobType.DOCUMENT_PROCESSING
        assert job_response.workflow_type == WorkflowType.COMPLETE_PIPELINE
        assert job_response.priority == JobPriority.HIGH
        assert job_response.patient_id == "patient_12345"
        assert len(job_response.assigned_agents) == 4  # OCR, NER, Chunking, Graph
        assert job_response.created_at is not None
        assert job_response.estimated_duration > 0
        
        # Verify Redis operations
        supervisor.redis_client.lpush.assert_called_once()
        supervisor.redis_client.hset.assert_called()
        supervisor.redis_client.expire.assert_called()

    async def test_enqueue_job_priority_handling(self, supervisor):
        """Test job priority handling and queue management."""
        # Create jobs with different priorities
        high_priority_job = JobRequest(
            job_type=JobType.DOCUMENT_PROCESSING,
            workflow_type=WorkflowType.OCR_ONLY,
            priority=JobPriority.HIGH,
            document_ids=["urgent_doc"]
        )
        
        normal_priority_job = JobRequest(
            job_type=JobType.DOCUMENT_PROCESSING,
            workflow_type=WorkflowType.OCR_ONLY,
            priority=JobPriority.NORMAL,
            document_ids=["normal_doc"]
        )
        
        low_priority_job = JobRequest(
            job_type=JobType.DOCUMENT_PROCESSING,
            workflow_type=WorkflowType.OCR_ONLY,
            priority=JobPriority.LOW,
            document_ids=["low_doc"]
        )
        
        supervisor.redis_client.lpush = AsyncMock(return_value=1)
        supervisor.redis_client.hset = AsyncMock(return_value=1)
        supervisor.redis_client.expire = AsyncMock(return_value=True)
        
        # Enqueue jobs
        high_response = await supervisor.enqueue_job(high_priority_job)
        normal_response = await supervisor.enqueue_job(normal_priority_job)
        low_response = await supervisor.enqueue_job(low_priority_job)
        
        # Verify priority queue assignment
        assert high_response.priority == JobPriority.HIGH
        assert normal_response.priority == JobPriority.NORMAL
        assert low_response.priority == JobPriority.LOW
        
        # Verify Redis queue calls
        calls = supervisor.redis_client.lpush.call_args_list
        assert any("high" in str(call) for call in calls)
        assert any("normal" in str(call) for call in calls)
        assert any("low" in str(call) for call in calls)

    async def test_job_status_tracking(self, supervisor):
        """Test comprehensive job status tracking and updates."""
        job_id = str(uuid4())
        
        # Mock job data in various states
        queued_job_data = {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "assigned_agents": ["ocr", "ner", "chunking", "graph"]
        }
        
        running_job_data = {
            "job_id": job_id,
            "status": JobStatus.RUNNING.value,
            "progress": 25.0,
            "current_stage": "ocr_processing",
            "current_agent": "ocr",
            "agent_results": {"ocr": {"status": "in_progress"}}
        }
        
        completed_job_data = {
            "job_id": job_id,
            "status": JobStatus.COMPLETED.value,
            "progress": 100.0,
            "completed_at": datetime.now().isoformat(),
            "results": {"final_result": "success"}
        }
        
        # Test different job states
        supervisor.redis_client.hgetall = AsyncMock(return_value=queued_job_data)
        queued_response = await supervisor.get_job_status(job_id)
        assert queued_response.status == JobStatus.QUEUED
        assert queued_response.progress == 0.0
        
        supervisor.redis_client.hgetall = AsyncMock(return_value=running_job_data)
        running_response = await supervisor.get_job_status(job_id)
        assert running_response.status == JobStatus.RUNNING
        assert running_response.progress == 25.0
        assert running_response.current_stage == "ocr_processing"
        
        supervisor.redis_client.hgetall = AsyncMock(return_value=completed_job_data)
        completed_response = await supervisor.get_job_status(job_id)
        assert completed_response.status == JobStatus.COMPLETED
        assert completed_response.progress == 100.0

    # Test 3: Workflow Orchestration
    async def test_workflow_orchestration_complete_pipeline(self, supervisor, sample_supervisor_state):
        """Test complete pipeline workflow orchestration."""
        # Mock agent responses
        ocr_response = OCRResponse(
            document_id="test_doc",
            status=OCRStatus.COMPLETED,
            extracted_text="Sample medical text with diabetes diagnosis",
            confidence_score=0.95,
            processing_time=2.5,
            metadata={"pages": 1, "words": 50}
        )
        
        ner_response = NERResponse(
            document_id="test_doc",
            status=NERStatus.COMPLETED,
            entities=[
                {"text": "diabetes", "label": "CONDITION", "confidence": 0.92}
            ],
            processing_time=3.2,
            total_entities=15
        )
        
        chunking_response = ChunkingResponse(
            document_id="test_doc",
            status=ChunkingStatus.COMPLETED,
            chunks=[{"text": "chunk1", "metadata": {"visit": "2024-01-15"}}],
            timeline={"visits": [{"date": "2024-01-15", "events": ["diagnosis"]}]}
        )
        
        graph_response = GraphResponse(
            patient_id="patient_123",
            status=GraphStatus.COMPLETED,
            nodes_created=10,
            relationships_created=15,
            insights={"risk_factors": ["diabetes"]}
        )
        
        # Mock workflow execution
        supervisor.workflow_graph.ainvoke = AsyncMock(return_value={
            "status": JobStatus.COMPLETED,
            "agent_results": {
                "ocr": ocr_response.dict(),
                "ner": ner_response.dict(),
                "chunking": chunking_response.dict(),
                "graph": graph_response.dict()
            }
        })
        
        # Execute workflow
        result = await supervisor.execute_workflow(sample_supervisor_state)
        
        # Assertions
        assert result["status"] == JobStatus.COMPLETED
        assert "agent_results" in result
        assert "ocr" in result["agent_results"]
        assert "ner" in result["agent_results"]
        assert "chunking" in result["agent_results"]
        assert "graph" in result["agent_results"]

    async def test_workflow_conditional_routing(self, supervisor):
        """Test workflow conditional routing based on job type."""
        # Test OCR-only workflow
        ocr_only_state = SupervisorState(
            job_id="ocr_job",
            workflow_type=WorkflowType.OCR_ONLY,
            status=JobStatus.RUNNING,
            next_action=AgentType.OCR.value
        )
        
        # Test conditional routing
        next_node = supervisor._determine_next_agent(ocr_only_state)
        assert next_node == "ocr_agent"
        
        # Test OCR to NER workflow
        ocr_ner_state = SupervisorState(
            job_id="ocr_ner_job",
            workflow_type=WorkflowType.OCR_TO_NER,
            status=JobStatus.RUNNING,
            next_action=AgentType.NER.value
        )
        
        next_node = supervisor._determine_next_agent(ocr_ner_state)
        assert next_node == "ner_agent"
        
        # Test completion condition
        completed_state = SupervisorState(
            job_id="completed_job",
            status=JobStatus.COMPLETED
        )
        
        next_node = supervisor._determine_next_agent(completed_state)
        assert next_node == "END"

    # Test 4: Agent Coordination and Communication
    async def test_agent_coordination_data_flow(self, supervisor):
        """Test data flow between agents in the pipeline."""
        job_id = "test_job_456"
        
        # Mock OCR agent execution
        with patch.object(supervisor, '_call_ocr_service') as mock_ocr:
            mock_ocr.return_value = {
                "status": "completed",
                "extracted_text": "Patient has Type 2 Diabetes",
                "confidence": 0.95
            }
            
            # Mock NER agent execution
            with patch.object(supervisor, '_call_ner_service') as mock_ner:
                mock_ner.return_value = {
                    "status": "completed",
                    "entities": [
                        {"text": "Type 2 Diabetes", "label": "CONDITION", "confidence": 0.92}
                    ]
                }
                
                # Mock Chunking agent execution
                with patch.object(supervisor, '_call_chunking_service') as mock_chunking:
                    mock_chunking.return_value = {
                        "status": "completed",
                        "chunks": [{"text": "Patient has Type 2 Diabetes", "metadata": {}}]
                    }
                    
                    # Mock Graph agent execution
                    with patch.object(supervisor, '_call_graph_service') as mock_graph:
                        mock_graph.return_value = {
                            "status": "completed",
                            "nodes_created": 5,
                            "relationships_created": 3
                        }
                        
                        # Test agent coordination
                        state = SupervisorState(
                            job_id=job_id,
                            status=JobStatus.RUNNING,
                            next_action=AgentType.OCR.value,
                            file_paths=["/uploads/test.pdf"]
                        )
                        
                        # Execute OCR agent
                        updated_state = await supervisor._execute_ocr_agent(state)
                        assert "ocr" in updated_state.agent_results
                        assert updated_state.agent_results["ocr"]["status"] == "completed"
                        
                        # Execute NER agent
                        updated_state.next_action = AgentType.NER.value
                        updated_state.extracted_text = "Patient has Type 2 Diabetes"
                        updated_state = await supervisor._execute_ner_agent(updated_state)
                        assert "ner" in updated_state.agent_results
                        
                        # Execute Chunking agent
                        updated_state.next_action = AgentType.CHUNKING.value
                        updated_state = await supervisor._execute_chunking_agent(updated_state)
                        assert "chunking" in updated_state.agent_results
                        
                        # Execute Graph agent
                        updated_state.next_action = AgentType.GRAPH.value
                        updated_state = await supervisor._execute_graph_agent(updated_state)
                        assert "graph" in updated_state.agent_results

    async def test_agent_failure_handling(self, supervisor):
        """Test agent failure handling and recovery mechanisms."""
        job_id = "test_job_failure"
        
        # Mock OCR agent failure
        with patch.object(supervisor, '_call_ocr_service') as mock_ocr:
            mock_ocr.side_effect = Exception("OCR processing failed")
            
            state = SupervisorState(
                job_id=job_id,
                status=JobStatus.RUNNING,
                next_action=AgentType.OCR.value,
                file_paths=["/uploads/test.pdf"]
            )
            
            # Execute OCR agent with failure
            updated_state = await supervisor._execute_ocr_agent(state)
            
            # Verify failure handling
            assert updated_state.status == JobStatus.FAILED
            assert "error" in updated_state.agent_results.get("ocr", {})
            assert "OCR processing failed" in str(updated_state.agent_results["ocr"]["error"])

    # Test 5: Error Handling and Recovery
    async def test_error_handling_retry_logic(self, supervisor):
        """Test error handling with retry logic."""
        job_id = "test_job_retry"
        
        # Mock transient failure followed by success
        with patch.object(supervisor, '_call_ocr_service') as mock_ocr:
            mock_ocr.side_effect = [
                Exception("Temporary failure"),
                Exception("Temporary failure"),
                {"status": "completed", "extracted_text": "Success on third try"}
            ]
            
            state = SupervisorState(
                job_id=job_id,
                status=JobStatus.RUNNING,
                next_action=AgentType.OCR.value,
                file_paths=["/uploads/test.pdf"],
                retry_count=0,
                max_retries=3
            )
            
            # Execute with retry logic
            updated_state = await supervisor._execute_ocr_agent_with_retry(state)
            
            # Verify successful completion after retries
            assert updated_state.status != JobStatus.FAILED
            assert updated_state.retry_count == 2
            assert mock_ocr.call_count == 3

    async def test_job_timeout_handling(self, supervisor):
        """Test job timeout handling and cleanup."""
        job_id = "test_job_timeout"
        
        # Create a job that should timeout
        state = SupervisorState(
            job_id=job_id,
            status=JobStatus.RUNNING,
            created_at=datetime.now() - timedelta(hours=2),  # 2 hours ago
            timeout_minutes=60  # 1 hour timeout
        )
        
        # Test timeout detection
        is_timeout = supervisor._is_job_timeout(state)
        assert is_timeout
        
        # Test timeout handling
        await supervisor._handle_job_timeout(state)
        
        # Verify timeout status
        supervisor.redis_client.hset.assert_called_with(
            f"job:{job_id}",
            mapping={"status": JobStatus.TIMEOUT.value}
        )

    # Test 6: Performance and Monitoring
    async def test_performance_monitoring(self, supervisor):
        """Test performance monitoring and metrics collection."""
        job_id = "test_job_performance"
        
        # Mock performance metrics
        with patch.object(supervisor, '_collect_performance_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "processing_time": 45.2,
                "memory_usage": "256MB",
                "cpu_usage": "15%",
                "agent_timings": {
                    "ocr": 12.5,
                    "ner": 18.7,
                    "chunking": 8.3,
                    "graph": 5.7
                }
            }
            
            # Execute job with monitoring
            await supervisor._execute_job_with_monitoring(job_id)
            
            # Verify metrics collection
            mock_metrics.assert_called_once_with(job_id)

    async def test_queue_status_monitoring(self, supervisor):
        """Test queue status monitoring and health checks."""
        # Mock queue status data
        supervisor.redis_client.llen = AsyncMock(side_effect=[5, 3, 1])  # high, normal, low priority queues
        supervisor.redis_client.keys = AsyncMock(return_value=[
            b"job:running_1", b"job:running_2", b"job:completed_1"
        ])
        
        # Get queue status
        queue_status = await supervisor.get_queue_status()
        
        # Assertions
        assert isinstance(queue_status, QueueStatus)
        assert queue_status.high_priority_count == 5
        assert queue_status.normal_priority_count == 3
        assert queue_status.low_priority_count == 1
        assert queue_status.total_jobs == 9
        assert queue_status.running_jobs == 2

    async def test_health_check_comprehensive(self, supervisor):
        """Test comprehensive health check functionality."""
        # Mock health check responses
        supervisor.redis_client.ping = AsyncMock(return_value=True)
        
        with patch.object(supervisor, '_check_agent_health') as mock_agent_health:
            mock_agent_health.return_value = {
                "ocr_agent": {"status": "healthy", "response_time": 0.1},
                "ner_agent": {"status": "healthy", "response_time": 0.2},
                "chunking_agent": {"status": "healthy", "response_time": 0.15},
                "graph_agent": {"status": "healthy", "response_time": 0.3}
            }
            
            # Perform health check
            health_status = await supervisor.health_check()
            
            # Assertions
            assert health_status["status"] == "healthy"
            assert health_status["redis_connected"] is True
            assert "agents" in health_status
            assert len(health_status["agents"]) == 4

    # Test 7: Integration and End-to-End
    async def test_end_to_end_workflow_execution(self, supervisor, complete_pipeline_job):
        """Test complete end-to-end workflow execution."""
        # Mock all agent services
        with patch.object(supervisor, '_call_ocr_service') as mock_ocr, \
             patch.object(supervisor, '_call_ner_service') as mock_ner, \
             patch.object(supervisor, '_call_chunking_service') as mock_chunking, \
             patch.object(supervisor, '_call_graph_service') as mock_graph:
            
            # Configure mock responses
            mock_ocr.return_value = {
                "status": "completed",
                "extracted_text": "Patient John Doe has Type 2 Diabetes",
                "confidence": 0.95
            }
            
            mock_ner.return_value = {
                "status": "completed",
                "entities": [
                    {"text": "John Doe", "label": "PATIENT", "confidence": 0.98},
                    {"text": "Type 2 Diabetes", "label": "CONDITION", "confidence": 0.92}
                ]
            }
            
            mock_chunking.return_value = {
                "status": "completed",
                "chunks": [{"text": "Medical visit chunk", "metadata": {"visit_date": "2024-01-15"}}],
                "timeline": {"visits": [{"date": "2024-01-15", "diagnosis": "Type 2 Diabetes"}]}
            }
            
            mock_graph.return_value = {
                "status": "completed",
                "patient_id": "patient_12345",
                "nodes_created": 15,
                "relationships_created": 20,
                "insights": {"risk_factors": ["diabetes", "hypertension"]}
            }
            
            # Execute complete workflow
            supervisor.redis_client.lpush = AsyncMock(return_value=1)
            supervisor.redis_client.hset = AsyncMock(return_value=1)
            supervisor.redis_client.expire = AsyncMock(return_value=True)
            
            # Enqueue and process job
            job_response = await supervisor.enqueue_job(complete_pipeline_job)
            result = await supervisor.process_job(job_response.job_id)
            
            # Verify complete pipeline execution
            assert result["status"] == "completed"
            assert "ocr" in result["agent_results"]
            assert "ner" in result["agent_results"]
            assert "chunking" in result["agent_results"]
            assert "graph" in result["agent_results"]
            
            # Verify agent call sequence
            mock_ocr.assert_called_once()
            mock_ner.assert_called_once()
            mock_chunking.assert_called_once()
            mock_graph.assert_called_once()

    async def test_batch_processing_workflow(self, supervisor):
        """Test batch processing workflow for multiple documents."""
        # Create batch job request
        batch_job = JobRequest(
            job_type=JobType.BATCH_PROCESSING,
            workflow_type=WorkflowType.BATCH_DOCUMENTS,
            priority=JobPriority.NORMAL,
            file_paths=[
                "/uploads/report1.pdf",
                "/uploads/report2.pdf",
                "/uploads/report3.pdf"
            ],
            processing_config={
                "batch_size": 3,
                "parallel_processing": True
            }
        )
        
        # Mock batch processing
        with patch.object(supervisor, '_process_batch_documents') as mock_batch:
            mock_batch.return_value = {
                "status": "completed",
                "processed_documents": 3,
                "total_processing_time": 125.5,
                "results": [
                    {"document": "report1.pdf", "status": "completed"},
                    {"document": "report2.pdf", "status": "completed"},
                    {"document": "report3.pdf", "status": "completed"}
                ]
            }
            
            # Execute batch workflow
            supervisor.redis_client.lpush = AsyncMock(return_value=1)
            supervisor.redis_client.hset = AsyncMock(return_value=1)
            supervisor.redis_client.expire = AsyncMock(return_value=True)
            
            job_response = await supervisor.enqueue_job(batch_job)
            result = await supervisor.process_job(job_response.job_id)
            
            # Verify batch processing
            assert result["status"] == "completed"
            assert result["processed_documents"] == 3
            assert len(result["results"]) == 3

    # Test 8: Callback and Event System
    async def test_callback_system(self, supervisor):
        """Test callback system for progress updates and notifications."""
        callback_handler = SupervisorCallbackHandler()
        
        # Mock callback registration
        callback_function = AsyncMock()
        callback_handler.register_callback("job_progress", callback_function)
        
        # Test callback execution
        await callback_handler.trigger_callback("job_progress", {
            "job_id": "test_job",
            "progress": 50.0,
            "stage": "ner_processing"
        })
        
        # Verify callback execution
        callback_function.assert_called_once_with({
            "job_id": "test_job",
            "progress": 50.0,
            "stage": "ner_processing"
        })

    async def test_event_driven_processing(self, supervisor):
        """Test event-driven processing and notifications."""
        # Mock event system
        with patch.object(supervisor, '_publish_event') as mock_publish:
            # Process job with events
            await supervisor._process_job_with_events("test_job_123")
            
            # Verify event publishing
            expected_events = [
                "job_started",
                "ocr_completed",
                "ner_completed",
                "chunking_completed",
                "graph_completed",
                "job_completed"
            ]
            
            published_events = [call[0][0] for call in mock_publish.call_args_list]
            for event in expected_events:
                assert event in published_events

    # Test 9: Stress and Load Testing
    async def test_concurrent_job_processing(self, supervisor):
        """Test concurrent job processing and resource management."""
        # Create multiple concurrent jobs
        jobs = []
        for i in range(10):
            job = JobRequest(
                job_type=JobType.DOCUMENT_PROCESSING,
                workflow_type=WorkflowType.OCR_ONLY,
                priority=JobPriority.NORMAL,
                document_ids=[f"doc_{i}"]
            )
            jobs.append(job)
        
        # Mock Redis operations
        supervisor.redis_client.lpush = AsyncMock(return_value=1)
        supervisor.redis_client.hset = AsyncMock(return_value=1)
        supervisor.redis_client.expire = AsyncMock(return_value=True)
        
        # Process jobs concurrently
        tasks = []
        for job in jobs:
            task = asyncio.create_task(supervisor.enqueue_job(job))
            tasks.append(task)
        
        # Wait for all jobs to be enqueued
        responses = await asyncio.gather(*tasks)
        
        # Verify all jobs were processed
        assert len(responses) == 10
        for response in responses:
            assert response.status == JobStatus.QUEUED

    async def test_memory_management_large_documents(self, supervisor):
        """Test memory management with large documents."""
        # Create job with large document configuration
        large_doc_job = JobRequest(
            job_type=JobType.DOCUMENT_PROCESSING,
            workflow_type=WorkflowType.COMPLETE_PIPELINE,
            priority=JobPriority.NORMAL,
            file_paths=["/uploads/large_document.pdf"],
            processing_config={
                "memory_limit": "512MB",
                "chunk_size": 2048,
                "streaming_mode": True
            }
        )
        
        # Mock memory monitoring
        with patch.object(supervisor, '_monitor_memory_usage') as mock_memory:
            mock_memory.return_value = {
                "current_usage": "256MB",
                "peak_usage": "384MB",
                "limit": "512MB",
                "status": "within_limits"
            }
            
            # Process large document
            supervisor.redis_client.lpush = AsyncMock(return_value=1)
            supervisor.redis_client.hset = AsyncMock(return_value=1)
            supervisor.redis_client.expire = AsyncMock(return_value=True)
            
            job_response = await supervisor.enqueue_job(large_doc_job)
            
            # Verify memory management
            assert job_response.status == JobStatus.QUEUED
            mock_memory.assert_called()

    # Test 10: Configuration and Customization
    async def test_workflow_configuration_customization(self, supervisor):
        """Test workflow configuration and customization options."""
        # Test custom workflow configuration
        custom_config = WorkflowConfiguration(
            max_concurrent_jobs=5,
            job_timeout_minutes=120,
            retry_attempts=3,
            retry_delay_seconds=30,
            enable_callbacks=True,
            enable_monitoring=True,
            priority_weights={
                JobPriority.HIGH: 3,
                JobPriority.NORMAL: 2,
                JobPriority.LOW: 1
            }
        )
        
        # Apply configuration
        supervisor.configure_workflow(custom_config)
        
        # Verify configuration
        assert supervisor.max_concurrent_jobs == 5
        assert supervisor.job_timeout_minutes == 120
        assert supervisor.retry_attempts == 3
        assert supervisor.enable_callbacks is True

    async def test_agent_configuration_customization(self, supervisor):
        """Test agent configuration and customization."""
        # Test custom agent configurations
        custom_agent_configs = {
            AgentType.OCR: {
                "engine": "tesseract",
                "confidence_threshold": 0.8,
                "preprocessing": "medical_optimized"
            },
            AgentType.NER: {
                "models": ["scispacy", "biobert", "med7"],
                "entity_linking": True,
                "confidence_threshold": 0.7
            },
            AgentType.CHUNKING: {
                "strategy": "medical_visits",
                "chunk_size": 1000,
                "overlap": 200
            },
            AgentType.GRAPH: {
                "create_embeddings": True,
                "link_knowledge_base": True,
                "generate_insights": True
            }
        }
        
        # Apply agent configurations
        supervisor.configure_agents(custom_agent_configs)
        
        # Verify configurations
        assert supervisor.agent_configs[AgentType.OCR]["engine"] == "tesseract"
        assert supervisor.agent_configs[AgentType.NER]["entity_linking"] is True
        assert supervisor.agent_configs[AgentType.CHUNKING]["strategy"] == "medical_visits"
        assert supervisor.agent_configs[AgentType.GRAPH]["create_embeddings"] is True
