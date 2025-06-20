"""
Comprehensive test suite for supervisor schemas.

This module tests all Pydantic models in app.schemas.supervisor including:
- Job request and response models
- Workflow and agent type enums
- Data validation and serialization
- Edge cases and error handling
"""

import pytest
from datetime import datetime
from uuid import uuid4

from pydantic import ValidationError

from app.schemas.supervisor import (
    JobType, JobPriority, JobStatus, WorkflowType, AgentType,
    JobRequest, JobResponse, SupervisorState, AgentConfig,
    WorkflowConfig, JobMetrics, QueueStatus
)


class TestSupervisorEnums:
    """Test cases for supervisor enumeration types."""
    
    def test_job_type_enum_values(self):
        """Test JobType enum contains all expected values."""
        expected_values = {
            "document_processing", "ocr_extraction", "ner_analysis",
            "chunking", "graph_update", "insight_generation", "batch_processing"
        }
        
        actual_values = {item.value for item in JobType}
        assert actual_values == expected_values
    
    def test_job_priority_enum_values(self):
        """Test JobPriority enum contains all expected values."""
        expected_values = {"low", "normal", "high", "urgent"}
        
        actual_values = {item.value for item in JobPriority}
        assert actual_values == expected_values
    
    def test_job_status_enum_values(self):
        """Test JobStatus enum contains all expected values."""
        expected_values = {
            "queued", "running", "completed", "failed", "cancelled", "retry_pending"
        }
        
        actual_values = {item.value for item in JobStatus}
        assert actual_values == expected_values
    
    def test_workflow_type_enum_values(self):
        """Test WorkflowType enum contains all expected values."""
        expected_values = {
            "complete_pipeline", "ocr_only", "ocr_to_ner",
            "document_to_graph", "batch_documents"
        }
        
        actual_values = {item.value for item in WorkflowType}
        assert actual_values == expected_values
    
    def test_agent_type_enum_values(self):
        """Test AgentType enum contains all expected values."""
        expected_values = {
            "supervisor", "ocr", "ner", "chunking", "graph", "insight", "chat"
        }
        
        actual_values = {item.value for item in AgentType}
        assert actual_values == expected_values


class TestJobRequestModel:
    """Test cases for JobRequest Pydantic model."""
    
    def test_job_request_minimal_valid(self):
        """Test JobRequest with minimal required fields."""
        job_request = JobRequest(
            job_type=JobType.DOCUMENT_PROCESSING
        )
        
        assert job_request.job_type == JobType.DOCUMENT_PROCESSING
        assert job_request.workflow_type == WorkflowType.COMPLETE_PIPELINE  # Default
        assert job_request.priority == JobPriority.NORMAL  # Default
        assert job_request.document_ids == []  # Default
        assert job_request.file_paths == []  # Default
        assert job_request.processing_config == {}  # Default
        assert job_request.metadata == {}  # Default
    
    def test_job_request_all_fields(self):
        """Test JobRequest with all fields populated."""
        job_request = JobRequest(
            job_type=JobType.OCR_EXTRACTION,
            workflow_type=WorkflowType.OCR_TO_NER,
            priority=JobPriority.HIGH,
            patient_id="patient_123",
            document_ids=["doc_1", "doc_2"],
            file_paths=["/path/to/file1.pdf", "/path/to/file2.jpg"],
            processing_config={
                "ocr_config": {"engine": "tesseract", "language": "eng"},
                "ner_config": {"model": "medical_bert"}
            },
            metadata={
                "source": "test_system",
                "timestamp": "2024-01-15T10:30:00"
            }
        )
        
        assert job_request.job_type == JobType.OCR_EXTRACTION
        assert job_request.workflow_type == WorkflowType.OCR_TO_NER
        assert job_request.priority == JobPriority.HIGH
        assert job_request.patient_id == "patient_123"
        assert len(job_request.document_ids) == 2
        assert len(job_request.file_paths) == 2
        assert "ocr_config" in job_request.processing_config
        assert "source" in job_request.metadata
    
    def test_job_request_serialization(self):
        """Test JobRequest serialization to dict."""
        job_request = JobRequest(
            job_type=JobType.NER_ANALYSIS,
            priority=JobPriority.URGENT,
            patient_id="patient_456"
        )
        
        serialized = job_request.dict()
        
        assert serialized["job_type"] == "ner_analysis"
        assert serialized["priority"] == "urgent"
        assert serialized["patient_id"] == "patient_456"
    
    def test_job_request_json_serialization(self):
        """Test JobRequest JSON serialization."""
        job_request = JobRequest(
            job_type=JobType.CHUNKING,
            file_paths=["/test/file.pdf"]
        )
        
        json_str = job_request.json()
        
        assert "chunking" in json_str
        assert "/test/file.pdf" in json_str
    
    def test_job_request_validation_invalid_enum(self):
        """Test JobRequest validation with invalid enum values."""
        with pytest.raises(ValidationError) as exc_info:
            JobRequest(job_type="invalid_job_type")
        
        assert "job_type" in str(exc_info.value)


class TestJobResponseModel:
    """Test cases for JobResponse Pydantic model."""
    
    def test_job_response_creation(self):
        """Test JobResponse model creation and validation."""
        job_id = str(uuid4())
        created_at = datetime.now()
        
        job_response = JobResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            priority=JobPriority.HIGH,
            assigned_agents=[AgentType.OCR, AgentType.NER],
            created_at=created_at,
            estimated_completion=None,
            progress=0.0,
            current_stage="initialization",
            results={},
            error_details=None
        )
        
        assert job_response.job_id == job_id
        assert job_response.status == JobStatus.QUEUED
        assert job_response.priority == JobPriority.HIGH
        assert len(job_response.assigned_agents) == 2
        assert job_response.progress == 0.0
    
    def test_job_response_with_progress(self):
        """Test JobResponse with progress tracking."""
        job_response = JobResponse(
            job_id="test_job",
            status=JobStatus.RUNNING,
            priority=JobPriority.NORMAL,
            assigned_agents=[AgentType.OCR],
            created_at=datetime.now(),
            progress=65.5,
            current_stage="ocr_processing",
            results={"ocr": {"confidence": 0.94}}
        )
        
        assert job_response.progress == 65.5
        assert job_response.current_stage == "ocr_processing"
        assert "ocr" in job_response.results
    
    def test_job_response_with_error(self):
        """Test JobResponse with error details."""
        job_response = JobResponse(
            job_id="failed_job",
            status=JobStatus.FAILED,
            priority=JobPriority.NORMAL,
            assigned_agents=[AgentType.OCR],
            created_at=datetime.now(),
            progress=25.0,
            current_stage="ocr_processing",
            results={},
            error_details={
                "error_type": "ProcessingError",
                "error_message": "OCR extraction failed",
                "timestamp": "2024-01-15T10:30:00"
            }
        )
        
        assert job_response.status == JobStatus.FAILED
        assert job_response.error_details is not None
        assert job_response.error_details["error_type"] == "ProcessingError"


class TestSupervisorStateModel:
    """Test cases for SupervisorState model."""
    
    def test_supervisor_state_creation(self):
        """Test SupervisorState model creation."""
        state = SupervisorState(
            job_id="test_job_123",
            current_agent=AgentType.OCR,
            workflow_type=WorkflowType.OCR_TO_NER,
            status=JobStatus.RUNNING,
            progress=45.0,
            agent_results={
                "ocr": {"extracted_text": "Sample text", "confidence": 0.92}
            },
            errors=[],
            metadata={"start_time": "2024-01-15T10:00:00"}
        )
        
        assert state.job_id == "test_job_123"
        assert state.current_agent == AgentType.OCR
        assert state.workflow_type == WorkflowType.OCR_TO_NER
        assert state.progress == 45.0
        assert "ocr" in state.agent_results
    
    def test_supervisor_state_with_errors(self):
        """Test SupervisorState with error tracking."""
        state = SupervisorState(
            job_id="error_job",
            current_agent=AgentType.NER,
            workflow_type=WorkflowType.COMPLETE_PIPELINE,
            status=JobStatus.FAILED,
            progress=30.0,
            agent_results={"ocr": {"status": "completed"}},
            errors=[
                "OCR confidence below threshold",
                "NER model failed to load"
            ],
            metadata={"error_count": 2}
        )
        
        assert len(state.errors) == 2
        assert "OCR confidence below threshold" in state.errors
        assert state.metadata["error_count"] == 2


class TestAgentConfigModel:
    """Test cases for AgentConfig model."""
    
    def test_agent_config_creation(self):
        """Test AgentConfig model creation."""
        config = AgentConfig(
            agent_type=AgentType.OCR,
            enabled=True,
            timeout_seconds=300,
            retry_attempts=3,
            configuration={
                "engine": "tesseract",
                "languages": ["eng"],
                "confidence_threshold": 0.8
            }
        )
        
        assert config.agent_type == AgentType.OCR
        assert config.enabled is True
        assert config.timeout_seconds == 300
        assert config.retry_attempts == 3
        assert config.configuration["engine"] == "tesseract"
    
    def test_agent_config_defaults(self):
        """Test AgentConfig with default values."""
        config = AgentConfig(
            agent_type=AgentType.NER
        )
        
        assert config.enabled is True  # Default
        assert config.timeout_seconds == 300  # Default
        assert config.retry_attempts == 3  # Default
        assert config.configuration == {}  # Default


class TestWorkflowConfigModel:
    """Test cases for WorkflowConfig model."""
    
    def test_workflow_config_creation(self):
        """Test WorkflowConfig model creation."""
        config = WorkflowConfig(
            workflow_type=WorkflowType.COMPLETE_PIPELINE,
            enabled=True,
            agent_sequence=[
                AgentType.OCR,
                AgentType.NER,
                AgentType.CHUNKING,
                AgentType.GRAPH
            ],
            parallel_agents=[],
            configuration={
                "max_concurrent_jobs": 5,
                "enable_caching": True
            }
        )
        
        assert config.workflow_type == WorkflowType.COMPLETE_PIPELINE
        assert len(config.agent_sequence) == 4
        assert AgentType.OCR in config.agent_sequence
        assert config.configuration["max_concurrent_jobs"] == 5
    
    def test_workflow_config_with_parallel_agents(self):
        """Test WorkflowConfig with parallel agent execution."""
        config = WorkflowConfig(
            workflow_type=WorkflowType.BATCH_DOCUMENTS,
            enabled=True,
            agent_sequence=[AgentType.OCR, AgentType.NER],
            parallel_agents=[AgentType.NER, AgentType.CHUNKING],
            configuration={"batch_size": 10}
        )
        
        assert len(config.parallel_agents) == 2
        assert AgentType.NER in config.parallel_agents
        assert AgentType.CHUNKING in config.parallel_agents


class TestJobMetricsModel:
    """Test cases for JobMetrics model."""
    
    def test_job_metrics_creation(self):
        """Test JobMetrics model creation."""
        metrics = JobMetrics(
            job_id="metrics_job",
            start_time=datetime.now(),
            end_time=None,
            processing_time_seconds=0.0,
            agent_timings={
                "ocr": 45.2,
                "ner": 23.8
            },
            resource_usage={
                "memory_mb": 512,
                "cpu_percent": 75.5
            },
            throughput_metrics={
                "documents_processed": 1,
                "entities_extracted": 15
            }
        )
        
        assert metrics.job_id == "metrics_job"
        assert "ocr" in metrics.agent_timings
        assert metrics.resource_usage["memory_mb"] == 512
        assert metrics.throughput_metrics["entities_extracted"] == 15
    
    def test_job_metrics_with_completion(self):
        """Test JobMetrics with completion times."""
        start_time = datetime(2024, 1, 15, 10, 0, 0)
        end_time = datetime(2024, 1, 15, 10, 2, 30)
        
        metrics = JobMetrics(
            job_id="completed_job",
            start_time=start_time,
            end_time=end_time,
            processing_time_seconds=150.0,
            agent_timings={"ocr": 90.0, "ner": 60.0},
            resource_usage={},
            throughput_metrics={}
        )
        
        assert metrics.end_time == end_time
        assert metrics.processing_time_seconds == 150.0


class TestQueueStatusModel:
    """Test cases for QueueStatus model."""
    
    def test_queue_status_creation(self):
        """Test QueueStatus model creation."""
        status = QueueStatus(
            total_jobs=25,
            queued_jobs=8,
            running_jobs=3,
            completed_jobs=12,
            failed_jobs=2,
            average_processing_time=125.5,
            queue_health="healthy",
            last_updated=datetime.now()
        )
        
        assert status.total_jobs == 25
        assert status.queued_jobs == 8
        assert status.running_jobs == 3
        assert status.queue_health == "healthy"
        assert status.average_processing_time == 125.5
    
    def test_queue_status_health_calculation(self):
        """Test QueueStatus health status calculation."""
        # This would typically be a computed field
        status = QueueStatus(
            total_jobs=100,
            queued_jobs=80,  # High queue backlog
            running_jobs=2,
            completed_jobs=15,
            failed_jobs=3,
            average_processing_time=300.0,  # Slow processing
            queue_health="degraded",
            last_updated=datetime.now()
        )
        
        assert status.queue_health == "degraded"
        assert status.queued_jobs > status.running_jobs


class TestModelValidation:
    """Test cases for model validation and error handling."""
    
    def test_job_request_invalid_priority(self):
        """Test JobRequest validation with invalid priority."""
        with pytest.raises(ValidationError):
            JobRequest(
                job_type=JobType.DOCUMENT_PROCESSING,
                priority="invalid_priority"
            )
    
    def test_job_response_invalid_progress(self):
        """Test JobResponse validation with invalid progress value."""
        with pytest.raises(ValidationError):
            JobResponse(
                job_id="test",
                status=JobStatus.RUNNING,
                priority=JobPriority.NORMAL,
                assigned_agents=[AgentType.OCR],
                created_at=datetime.now(),
                progress=150.0  # Invalid - should be 0-100
            )
    
    def test_agent_config_negative_timeout(self):
        """Test AgentConfig validation with negative timeout."""
        with pytest.raises(ValidationError):
            AgentConfig(
                agent_type=AgentType.OCR,
                timeout_seconds=-10  # Invalid negative timeout
            )
    
    def test_supervisor_state_invalid_agent_type(self):
        """Test SupervisorState validation with invalid agent type."""
        with pytest.raises(ValidationError):
            SupervisorState(
                job_id="test",
                current_agent="invalid_agent",
                workflow_type=WorkflowType.OCR_ONLY,
                status=JobStatus.RUNNING,
                progress=50.0
            )


class TestModelSerialization:
    """Test cases for model serialization and deserialization."""
    
    def test_job_request_round_trip_serialization(self):
        """Test JobRequest serialization and deserialization."""
        original = JobRequest(
            job_type=JobType.DOCUMENT_PROCESSING,
            workflow_type=WorkflowType.COMPLETE_PIPELINE,
            priority=JobPriority.HIGH,
            patient_id="patient_789",
            file_paths=["/test/file.pdf"]
        )
        
        # Serialize to dict
        serialized = original.dict()
        
        # Deserialize back to model
        deserialized = JobRequest(**serialized)
        
        assert deserialized.job_type == original.job_type
        assert deserialized.workflow_type == original.workflow_type
        assert deserialized.priority == original.priority
        assert deserialized.patient_id == original.patient_id
        assert deserialized.file_paths == original.file_paths
    
    def test_job_response_json_round_trip(self):
        """Test JobResponse JSON serialization round trip."""
        original = JobResponse(
            job_id="json_test",
            status=JobStatus.COMPLETED,
            priority=JobPriority.NORMAL,
            assigned_agents=[AgentType.OCR, AgentType.NER],
            created_at=datetime.now(),
            progress=100.0,
            current_stage="completed",
            results={"final": "success"}
        )
        
        # Serialize to JSON
        json_str = original.json()
        
        # Deserialize from JSON
        deserialized = JobResponse.parse_raw(json_str)
        
        assert deserialized.job_id == original.job_id
        assert deserialized.status == original.status
        assert deserialized.progress == original.progress
        assert deserialized.results == original.results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
