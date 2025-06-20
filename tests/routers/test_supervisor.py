"""
Comprehensive test suite for supervisor router endpoints.

This module tests all REST API endpoints in app.routers.supervisor including:
- Job enqueuing and management
- Status monitoring and tracking
- Queue management operations
- Workflow configuration
- Error handling scenarios
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.main import app
from app.schemas.supervisor import (
    JobRequest, JobResponse, JobType, JobPriority, JobStatus,
    WorkflowType, AgentType, QueueStatus
)
from app.services.supervisor import LangChainSupervisor


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_supervisor():
    """Create mock supervisor service."""
    supervisor = AsyncMock(spec=LangChainSupervisor)
    return supervisor


class TestSupervisorInfoEndpoint:
    """Test cases for supervisor info endpoint."""
    
    def test_get_supervisor_info_success(self, client):
        """Test successful retrieval of supervisor service information."""
        response = client.get("/supervisor/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["service_name"] == "MRIA Supervisor Service"
        assert data["service_type"] == "workflow_orchestrator"
        assert data["status"] == "active"
        assert "capabilities" in data
        assert "workflow_orchestration" in data["capabilities"]
        assert "supported_workflows" in data
        assert "endpoints" in data
        assert "dependencies" in data
    
    def test_supervisor_info_contains_expected_endpoints(self, client):
        """Test that supervisor info contains expected endpoint information."""
        response = client.get("/supervisor/info")
        data = response.json()
        
        expected_endpoints = [
            "/supervisor/info",
            "/supervisor/enqueue", 
            "/supervisor/status/{job_id}",
            "/supervisor/queue/status",
            "/supervisor/workflows"
        ]
        
        for endpoint in expected_endpoints:
            assert endpoint in data["endpoints"]
    
    def test_supervisor_info_dependencies(self, client):
        """Test that supervisor info contains expected service dependencies."""
        response = client.get("/supervisor/info")
        data = response.json()
        
        expected_dependencies = ["ingestion", "ocr", "ner", "chunking", "graph"]
        
        for dependency in expected_dependencies:
            assert dependency in data["dependencies"]


class TestEnqueueJobEndpoint:
    """Test cases for job enqueuing endpoint."""
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_enqueue_job_success(self, mock_get_supervisor, client):
        """Test successful job enqueuing."""
        # Mock supervisor
        mock_supervisor = AsyncMock()
        mock_job_response = JobResponse(
            job_id="test_job_123",
            status=JobStatus.QUEUED,
            priority=JobPriority.HIGH,
            assigned_agents=[AgentType.OCR, AgentType.NER],
            created_at="2024-01-15T10:30:00",
            progress=0.0,
            current_stage="queued",
            results={}
        )
        mock_supervisor.enqueue_job.return_value = mock_job_response
        mock_get_supervisor.return_value = mock_supervisor
        
        # Test request
        job_request = {
            "job_type": "document_processing",
            "workflow_type": "complete_pipeline",
            "priority": "high",
            "patient_id": "patient_123",
            "file_paths": ["/test/document.pdf"]
        }
        
        response = client.post("/supervisor/enqueue", json=job_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["job_id"] == "test_job_123"
        assert data["status"] == "queued"
        assert data["priority"] == "high"
        assert len(data["assigned_agents"]) == 2
    
    def test_enqueue_job_missing_documents_error(self, client):
        """Test job enqueuing with missing documents/files."""
        job_request = {
            "job_type": "document_processing",
            "workflow_type": "complete_pipeline",
            "priority": "normal"
            # Missing document_ids and file_paths
        }
        
        response = client.post("/supervisor/enqueue", json=job_request)
        
        assert response.status_code == 400
        assert "document_id or file_path must be provided" in response.json()["detail"]
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_enqueue_job_supervisor_error(self, mock_get_supervisor, client):
        """Test job enqueuing when supervisor service fails."""
        # Mock supervisor to raise exception
        mock_supervisor = AsyncMock()
        mock_supervisor.enqueue_job.side_effect = Exception("Supervisor service unavailable")
        mock_get_supervisor.return_value = mock_supervisor
        
        job_request = {
            "job_type": "ocr_extraction",
            "file_paths": ["/test/document.pdf"]
        }
        
        response = client.post("/supervisor/enqueue", json=job_request)
        
        assert response.status_code == 500
        assert "Failed to enqueue job" in response.json()["detail"]
    
    def test_enqueue_job_invalid_job_type(self, client):
        """Test job enqueuing with invalid job type."""
        job_request = {
            "job_type": "invalid_job_type",
            "file_paths": ["/test/document.pdf"]
        }
        
        response = client.post("/supervisor/enqueue", json=job_request)
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_enqueue_job_with_processing_config(self, mock_get_supervisor, client):
        """Test job enqueuing with processing configuration."""
        mock_supervisor = AsyncMock()
        mock_job_response = JobResponse(
            job_id="config_job_456",
            status=JobStatus.QUEUED,
            priority=JobPriority.NORMAL,
            assigned_agents=[AgentType.OCR],
            created_at="2024-01-15T10:30:00",
            progress=0.0,
            current_stage="queued",
            results={}
        )
        mock_supervisor.enqueue_job.return_value = mock_job_response
        mock_get_supervisor.return_value = mock_supervisor
        
        job_request = {
            "job_type": "ocr_extraction",
            "file_paths": ["/test/medical_report.pdf"],
            "processing_config": {
                "ocr_config": {
                    "engine": "tesseract",
                    "language": "eng",
                    "confidence_threshold": 0.8
                }
            },
            "metadata": {
                "document_type": "lab_report",
                "source": "hospital_system"
            }
        }
        
        response = client.post("/supervisor/enqueue", json=job_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "config_job_456"


class TestJobStatusEndpoint:
    """Test cases for job status monitoring endpoint."""
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_get_job_status_success(self, mock_get_supervisor, client):
        """Test successful job status retrieval."""
        mock_supervisor = AsyncMock()
        mock_job_response = JobResponse(
            job_id="status_job_789",
            status=JobStatus.RUNNING,
            priority=JobPriority.HIGH,
            assigned_agents=[AgentType.OCR, AgentType.NER],
            created_at="2024-01-15T10:30:00",
            progress=45.5,
            current_stage="ner_processing",
            results={
                "ocr": {
                    "status": "completed",
                    "confidence": 0.94,
                    "extracted_text": "Sample medical text..."
                }
            }
        )
        mock_supervisor.get_job_status.return_value = mock_job_response
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.get("/supervisor/status/status_job_789")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["job_id"] == "status_job_789"
        assert data["status"] == "running"
        assert data["progress"] == 45.5
        assert data["current_stage"] == "ner_processing"
        assert "ocr" in data["results"]
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_get_job_status_not_found(self, mock_get_supervisor, client):
        """Test job status retrieval for non-existent job."""
        mock_supervisor = AsyncMock()
        mock_supervisor.get_job_status.return_value = None
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.get("/supervisor/status/nonexistent_job")
        
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_get_job_status_supervisor_error(self, mock_get_supervisor, client):
        """Test job status retrieval when supervisor service fails."""
        mock_supervisor = AsyncMock()
        mock_supervisor.get_job_status.side_effect = Exception("Database connection failed")
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.get("/supervisor/status/error_job")
        
        assert response.status_code == 500
        assert "Failed to retrieve job status" in response.json()["detail"]


class TestQueueStatusEndpoint:
    """Test cases for queue status monitoring endpoint."""
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_get_queue_status_success(self, mock_get_supervisor, client):
        """Test successful queue status retrieval."""
        mock_supervisor = AsyncMock()
        mock_queue_status = QueueStatus(
            total_jobs=50,
            queued_jobs=12,
            running_jobs=5,
            completed_jobs=30,
            failed_jobs=3,
            average_processing_time=125.5,
            queue_health="healthy",
            last_updated="2024-01-15T10:30:00"
        )
        mock_supervisor.get_queue_status.return_value = mock_queue_status
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.get("/supervisor/queue/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_jobs"] == 50
        assert data["queued_jobs"] == 12
        assert data["running_jobs"] == 5
        assert data["completed_jobs"] == 30
        assert data["failed_jobs"] == 3
        assert data["queue_health"] == "healthy"
        assert data["average_processing_time"] == 125.5
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_get_queue_status_with_filters(self, mock_get_supervisor, client):
        """Test queue status retrieval with time filters."""
        mock_supervisor = AsyncMock()
        mock_queue_status = QueueStatus(
            total_jobs=25,
            queued_jobs=8,
            running_jobs=2,
            completed_jobs=14,
            failed_jobs=1,
            average_processing_time=98.2,
            queue_health="healthy",
            last_updated="2024-01-15T10:30:00"
        )
        mock_supervisor.get_queue_status.return_value = mock_queue_status
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.get("/supervisor/queue/status?last_hours=24")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_jobs"] == 25


class TestWorkflowsEndpoint:
    """Test cases for workflow management endpoint."""
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_get_available_workflows_success(self, mock_get_supervisor, client):
        """Test successful retrieval of available workflows."""
        mock_supervisor = AsyncMock()
        mock_workflows = {
            "complete_pipeline": {
                "name": "Complete Document Processing Pipeline",
                "description": "Full pipeline from OCR to knowledge graph",
                "agents": ["ocr", "ner", "chunking", "graph"],
                "estimated_time": "5-10 minutes",
                "supported_formats": ["pdf", "jpg", "png", "docx"]
            },
            "ocr_only": {
                "name": "OCR Text Extraction Only",
                "description": "Extract text from documents without further processing",
                "agents": ["ocr"],
                "estimated_time": "1-2 minutes",
                "supported_formats": ["pdf", "jpg", "png", "tiff"]
            },
            "ocr_to_ner": {
                "name": "OCR + Medical Entity Recognition",
                "description": "Extract text and identify medical entities",
                "agents": ["ocr", "ner"],
                "estimated_time": "3-5 minutes",
                "supported_formats": ["pdf", "jpg", "png", "docx"]
            }
        }
        mock_supervisor.get_available_workflows.return_value = mock_workflows
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.get("/supervisor/workflows")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "complete_pipeline" in data
        assert "ocr_only" in data
        assert "ocr_to_ner" in data
        assert data["complete_pipeline"]["name"] == "Complete Document Processing Pipeline"
        assert len(data["complete_pipeline"]["agents"]) == 4


class TestJobRetryEndpoint:
    """Test cases for job retry endpoint."""
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_retry_job_success(self, mock_get_supervisor, client):
        """Test successful job retry."""
        mock_supervisor = AsyncMock()
        mock_job_response = JobResponse(
            job_id="retry_job_123",
            status=JobStatus.QUEUED,
            priority=JobPriority.HIGH,
            assigned_agents=[AgentType.OCR, AgentType.NER],
            created_at="2024-01-15T10:35:00",
            progress=0.0,
            current_stage="queued",
            results={}
        )
        mock_supervisor.retry_job.return_value = mock_job_response
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.post("/supervisor/retry/retry_job_123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["job_id"] == "retry_job_123"
        assert data["status"] == "queued"
        assert data["progress"] == 0.0
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_retry_job_not_found(self, mock_get_supervisor, client):
        """Test retry for non-existent job."""
        mock_supervisor = AsyncMock()
        mock_supervisor.retry_job.return_value = None
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.post("/supervisor/retry/nonexistent_job")
        
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_retry_job_already_running(self, mock_get_supervisor, client):
        """Test retry for job that is already running."""
        mock_supervisor = AsyncMock()
        mock_supervisor.retry_job.side_effect = ValueError("Job is already running")
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.post("/supervisor/retry/running_job")
        
        assert response.status_code == 400
        assert "already running" in response.json()["detail"]


class TestJobCancellationEndpoint:
    """Test cases for job cancellation endpoint."""
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_cancel_job_success(self, mock_get_supervisor, client):
        """Test successful job cancellation."""
        mock_supervisor = AsyncMock()
        mock_job_response = JobResponse(
            job_id="cancel_job_456",
            status=JobStatus.CANCELLED,
            priority=JobPriority.NORMAL,
            assigned_agents=[AgentType.OCR],
            created_at="2024-01-15T10:30:00",
            progress=25.0,
            current_stage="cancelled",
            results={"ocr": {"status": "cancelled"}}
        )
        mock_supervisor.cancel_job.return_value = mock_job_response
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.post("/supervisor/cancel/cancel_job_456")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["job_id"] == "cancel_job_456"
        assert data["status"] == "cancelled"
        assert data["current_stage"] == "cancelled"
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_cancel_job_already_completed(self, mock_get_supervisor, client):
        """Test cancellation of already completed job."""
        mock_supervisor = AsyncMock()
        mock_supervisor.cancel_job.side_effect = ValueError("Job already completed")
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.post("/supervisor/cancel/completed_job")
        
        assert response.status_code == 400
        assert "already completed" in response.json()["detail"]


class TestBatchOperationsEndpoint:
    """Test cases for batch operations endpoint."""
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_batch_enqueue_success(self, mock_get_supervisor, client):
        """Test successful batch job enqueuing."""
        mock_supervisor = AsyncMock()
        mock_batch_response = {
            "batch_id": "batch_789",
            "total_jobs": 3,
            "enqueued_jobs": 3,
            "failed_jobs": 0,
            "job_ids": ["job_1", "job_2", "job_3"]
        }
        mock_supervisor.enqueue_batch_jobs.return_value = mock_batch_response
        mock_get_supervisor.return_value = mock_supervisor
        
        batch_request = {
            "jobs": [
                {
                    "job_type": "document_processing",
                    "file_paths": ["/test/doc1.pdf"]
                },
                {
                    "job_type": "document_processing", 
                    "file_paths": ["/test/doc2.pdf"]
                },
                {
                    "job_type": "document_processing",
                    "file_paths": ["/test/doc3.pdf"]
                }
            ],
            "batch_config": {
                "max_concurrent": 2,
                "retry_failed": True
            }
        }
        
        response = client.post("/supervisor/batch/enqueue", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["batch_id"] == "batch_789"
        assert data["total_jobs"] == 3
        assert data["enqueued_jobs"] == 3
        assert len(data["job_ids"]) == 3


class TestHealthCheckEndpoint:
    """Test cases for supervisor health check endpoint."""
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_health_check_healthy(self, mock_get_supervisor, client):
        """Test health check when supervisor is healthy."""
        mock_supervisor = AsyncMock()
        mock_health = {
            "status": "healthy",
            "redis_connection": True,
            "workflow_engine": True,
            "agent_services": {
                "ocr": True,
                "ner": True,
                "chunking": True,
                "graph": True
            },
            "queue_size": 5,
            "active_jobs": 2
        }
        mock_supervisor.health_check.return_value = mock_health
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.get("/supervisor/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["redis_connection"] is True
        assert data["workflow_engine"] is True
        assert data["queue_size"] == 5
    
    @patch('app.routers.supervisor.get_supervisor')
    def test_health_check_degraded(self, mock_get_supervisor, client):
        """Test health check when supervisor is degraded."""
        mock_supervisor = AsyncMock()
        mock_health = {
            "status": "degraded",
            "redis_connection": True,
            "workflow_engine": True,
            "agent_services": {
                "ocr": True,
                "ner": False,  # NER service down
                "chunking": True,
                "graph": True
            },
            "queue_size": 15,  # High queue backlog
            "active_jobs": 1
        }
        mock_supervisor.health_check.return_value = mock_health
        mock_get_supervisor.return_value = mock_supervisor
        
        response = client.get("/supervisor/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "degraded"
        assert data["agent_services"]["ner"] is False
        assert data["queue_size"] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
