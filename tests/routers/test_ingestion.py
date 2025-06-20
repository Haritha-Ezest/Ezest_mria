"""
Comprehensive tests for the ingestion router endpoints.

This module tests all endpoints in the ingestion router, including:
- Service info endpoints
- File ingestion operations
- Batch processing
- Status monitoring
- Error handling and validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from io import BytesIO

from app.main import app


class TestIngestionRouter:
    """Test class for ingestion router endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_get_ingestion_info(self, client):
        """Test getting ingestion service information."""
        response = client.get("/ingestion/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "service_name" in data
        assert "service_type" in data
        assert "version" in data
        assert "capabilities" in data
        assert "supported_formats" in data

    @patch('app.services.file_handler.FileHandler.process_file')
    def test_ingest_file_success(self, mock_process, client):
        """Test successful file ingestion."""
        mock_process.return_value = {
            "file_id": "file_123",
            "filename": "test_document.pdf",
            "file_type": "pdf",
            "size": 1024,
            "pages": 3,
            "processing_status": "completed",
            "processing_time": 2.5,
            "extracted_text": "Sample document content...",
            "metadata": {
                "title": "Test Document",
                "author": "Test Author",
                "creation_date": "2023-01-15"
            }
        }

        # Create a fake PDF file
        pdf_data = BytesIO(b"fake_pdf_content")
        files = {"file": ("test_document.pdf", pdf_data, "application/pdf")}
        data = {"extract_text": True, "extract_metadata": True}
        
        response = client.post("/ingestion/ingest", files=files, data=data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["success"] is True
        assert result["file_id"] == "file_123"
        assert result["filename"] == "test_document.pdf"
        assert result["processing_status"] == "completed"

    def test_ingest_file_no_file(self, client):
        """Test file ingestion without file."""
        response = client.post("/ingestion/ingest")
        
        assert response.status_code == 422

    @patch('app.services.file_handler.FileHandler.batch_process')
    def test_batch_ingest_success(self, mock_batch, client):
        """Test successful batch file ingestion."""
        mock_batch.return_value = {
            "batch_id": "batch_456",
            "total_files": 3,
            "processed": 3,
            "failed": 0,
            "processing_time": 8.5,
            "results": [
                {
                    "file_id": "file_1",
                    "filename": "doc1.pdf",
                    "status": "completed",
                    "processing_time": 2.1
                },
                {
                    "file_id": "file_2",
                    "filename": "doc2.pdf",
                    "status": "completed",
                    "processing_time": 3.2
                },
                {
                    "file_id": "file_3",
                    "filename": "doc3.pdf",
                    "status": "completed",
                    "processing_time": 2.8
                }
            ]
        }

        # Create fake files
        files = [
            ("files", ("doc1.pdf", BytesIO(b"pdf_content_1"), "application/pdf")),
            ("files", ("doc2.pdf", BytesIO(b"pdf_content_2"), "application/pdf")),
            ("files", ("doc3.pdf", BytesIO(b"pdf_content_3"), "application/pdf"))
        ]
        
        response = client.post("/ingestion/batch-ingest", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["batch_id"] == "batch_456"
        assert data["total_files"] == 3
        assert data["processed"] == 3
        assert data["failed"] == 0

    @patch('app.services.file_handler.FileHandler.get_file_status')
    def test_get_file_status_success(self, mock_status, client):
        """Test getting file processing status."""
        file_id = "file_123"
        
        mock_status.return_value = {
            "file_id": "file_123",
            "filename": "test_document.pdf",
            "status": "completed",
            "progress": 100,
            "processing_time": 2.5,
            "created_at": "2023-01-15T10:00:00Z",
            "updated_at": "2023-01-15T10:02:30Z",
            "error_message": None
        }

        response = client.get(f"/ingestion/status/{file_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["file_id"] == "file_123"
        assert data["status"] == "completed"
        assert data["progress"] == 100

    def test_get_file_status_not_found(self, client):
        """Test getting status for non-existent file."""
        with patch('app.services.file_handler.FileHandler.get_file_status') as mock_status:
            mock_status.return_value = None
            
            response = client.get("/ingestion/status/nonexistent")
            
            assert response.status_code == 404

    @patch('app.services.file_handler.FileHandler.get_batch_status')
    def test_get_batch_status_success(self, mock_status, client):
        """Test getting batch processing status."""
        batch_id = "batch_456"
        
        mock_status.return_value = {
            "batch_id": "batch_456",
            "total_files": 5,
            "processed": 3,
            "failed": 1,
            "in_progress": 1,
            "status": "processing",
            "progress": 60,
            "created_at": "2023-01-15T10:00:00Z",
            "estimated_completion": "2023-01-15T10:15:00Z"
        }

        response = client.get(f"/ingestion/batch-status/{batch_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["batch_id"] == "batch_456"
        assert data["total_files"] == 5
        assert data["processed"] == 3
        assert data["progress"] == 60

    @patch('app.services.file_handler.FileHandler.get_file_content')
    def test_get_file_content_success(self, mock_content, client):
        """Test getting processed file content."""
        file_id = "file_123"
        
        mock_content.return_value = {
            "file_id": "file_123",
            "filename": "test_document.pdf",
            "content_type": "text",
            "content": "This is the extracted text content from the document...",
            "metadata": {
                "page_count": 3,
                "word_count": 250,
                "language": "en"
            },
            "processing_info": {
                "ocr_confidence": 0.95,
                "extraction_method": "tesseract"
            }
        }

        response = client.get(f"/ingestion/content/{file_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["file_id"] == "file_123"
        assert "extracted text content" in data["content"]
        assert data["metadata"]["page_count"] == 3

    def test_get_supported_formats(self, client):
        """Test getting supported file formats."""
        response = client.get("/ingestion/formats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "supported_formats" in data
        formats = data["supported_formats"]
        
        # Check common formats are present
        format_names = [fmt["name"] for fmt in formats]
        assert "PDF" in format_names
        assert "DOCX" in format_names
        assert "TXT" in format_names

    @patch('app.services.file_handler.FileHandler.delete_file')
    def test_delete_file_success(self, mock_delete, client):
        """Test successful file deletion."""
        file_id = "file_123"
        
        mock_delete.return_value = {
            "deleted": True,
            "file_id": "file_123",
            "cleanup_time": 0.1
        }

        response = client.delete(f"/ingestion/files/{file_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["deleted"] is True
        assert data["file_id"] == "file_123"

    @patch('app.services.file_handler.FileHandler.list_files')
    def test_list_files_success(self, mock_list, client):
        """Test getting file list."""
        mock_list.return_value = {
            "files": [
                {
                    "file_id": "file_1",
                    "filename": "doc1.pdf",
                    "status": "completed",
                    "created_at": "2023-01-15T10:00:00Z",
                    "size": 1024
                },
                {
                    "file_id": "file_2",
                    "filename": "doc2.pdf",
                    "status": "processing",
                    "created_at": "2023-01-15T10:05:00Z",
                    "size": 2048
                }
            ],
            "total_files": 2,
            "page": 1,
            "page_size": 10,
            "total_pages": 1
        }

        response = client.get("/ingestion/files?page=1&page_size=10")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["files"]) == 2
        assert data["total_files"] == 2

    def test_health_check(self, client):
        """Test ingestion service health check."""
        response = client.get("/ingestion/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "ingestion"
        assert "timestamp" in data

    @patch('app.services.file_handler.FileHandler.get_metrics')
    def test_get_ingestion_metrics(self, mock_metrics, client):
        """Test getting ingestion service metrics."""
        mock_metrics.return_value = {
            "total_files_processed": 1250,
            "total_batches_processed": 85,
            "avg_processing_time": 2.3,
            "success_rate": 0.96,
            "error_rate": 0.04,
            "storage_used_mb": 156.7,
            "files_by_type": {
                "pdf": 850,
                "docx": 250,
                "txt": 150
            }
        }

        response = client.get("/ingestion/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_files_processed"] == 1250
        assert data["success_rate"] == 0.96
        assert "files_by_type" in data

    def test_get_ingestion_config(self, client):
        """Test getting ingestion configuration."""
        response = client.get("/ingestion/config")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "max_file_size" in data
        assert "max_batch_size" in data
        assert "supported_formats" in data
        assert "processing_options" in data
