"""
Comprehensive tests for the OCR router endpoints.

This module tests all endpoints in the OCR router, including:
- Service info endpoints
- Image processing operations
- Document OCR
- Batch processing
- Error handling and validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from io import BytesIO

from app.main import app


class TestOCRRouter:
    """Test class for OCR router endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_image_data(self):
        """Sample image data for testing."""
        # Create a simple test image data
        return BytesIO(b"fake_image_data")

    def test_get_ocr_info(self, client):
        """Test getting OCR service information."""
        response = client.get("/ocr/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "service_name" in data
        assert "service_type" in data
        assert "version" in data
        assert "capabilities" in data

    @patch('app.services.ocr_processor.OCRProcessor.process_image')
    def test_process_image_success(self, mock_process, client):
        """Test successful image processing."""
        mock_process.return_value = {
            "text": "Patient John Doe\nDate: 2023-01-15\nDiagnosis: Hypertension",
            "confidence": 0.95,
            "processing_time": 1.2,
            "page_count": 1,
            "metadata": {
                "image_size": [800, 600],
                "dpi": 300,
                "color_mode": "RGB"
            }
        }

        # Create a fake image file
        image_data = BytesIO(b"fake_image_data")
        files = {"file": ("test_image.jpg", image_data, "image/jpeg")}
        
        response = client.post("/ocr/process-image", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "Patient John Doe" in data["text"]
        assert data["confidence"] == 0.95
        assert "processing_time" in data

    def test_process_image_no_file(self, client):
        """Test image processing without file."""
        response = client.post("/ocr/process-image")
        
        assert response.status_code == 422

    @patch('app.services.ocr_processor.OCRProcessor.process_document')
    def test_process_document_success(self, mock_process, client):
        """Test successful document processing."""
        mock_process.return_value = {
            "pages": [
                {
                    "page_number": 1,
                    "text": "Medical Report\nPatient: John Doe",
                    "confidence": 0.92,
                    "bounding_boxes": []
                },
                {
                    "page_number": 2,
                    "text": "Diagnosis: Hypertension\nTreatment: Medication",
                    "confidence": 0.88,
                    "bounding_boxes": []
                }
            ],
            "total_pages": 2,
            "full_text": "Medical Report\nPatient: John Doe\nDiagnosis: Hypertension\nTreatment: Medication",
            "processing_time": 2.5,
            "avg_confidence": 0.90
        }

        # Create a fake PDF file
        pdf_data = BytesIO(b"fake_pdf_data")
        files = {"file": ("test_document.pdf", pdf_data, "application/pdf")}
        
        response = client.post("/ocr/process-document", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["total_pages"] == 2
        assert len(data["pages"]) == 2
        assert "Medical Report" in data["full_text"]

    @patch('app.services.ocr_processor.OCRProcessor.batch_process')
    def test_batch_process_success(self, mock_batch, client):
        """Test successful batch processing."""
        mock_batch.return_value = {
            "results": [
                {
                    "filename": "doc1.jpg",
                    "text": "Document 1 content",
                    "confidence": 0.95,
                    "status": "success"
                },
                {
                    "filename": "doc2.jpg", 
                    "text": "Document 2 content",
                    "confidence": 0.88,
                    "status": "success"
                }
            ],
            "total_files": 2,
            "successful": 2,
            "failed": 0,
            "processing_time": 3.2
        }

        # Create fake image files
        files = [
            ("files", ("doc1.jpg", BytesIO(b"fake_image_1"), "image/jpeg")),
            ("files", ("doc2.jpg", BytesIO(b"fake_image_2"), "image/jpeg"))
        ]
        
        response = client.post("/ocr/batch-process", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["total_files"] == 2
        assert data["successful"] == 2
        assert data["failed"] == 0

    def test_get_supported_formats(self, client):
        """Test getting supported file formats."""
        response = client.get("/ocr/formats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "supported_formats" in data
        formats = data["supported_formats"]
        
        # Check common formats are present
        assert any("jpg" in fmt["extensions"] for fmt in formats)
        assert any("pdf" in fmt["extensions"] for fmt in formats)

    def test_health_check(self, client):
        """Test OCR service health check."""
        response = client.get("/ocr/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "ocr"
        assert "timestamp" in data

    @patch('app.services.ocr_processor.OCRProcessor.get_metrics')
    def test_get_ocr_metrics(self, mock_metrics, client):
        """Test getting OCR service metrics."""
        mock_metrics.return_value = {
            "total_images_processed": 850,
            "total_documents_processed": 320,
            "avg_processing_time": 1.8,
            "success_rate": 0.94,
            "error_rate": 0.06,
            "avg_confidence": 0.89
        }

        response = client.get("/ocr/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_images_processed"] == 850
        assert data["total_documents_processed"] == 320
        assert data["success_rate"] == 0.94

    def test_get_ocr_config(self, client):
        """Test getting OCR configuration."""
        response = client.get("/ocr/config")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "max_file_size" in data
        assert "supported_languages" in data
        assert "default_dpi" in data
        assert "preprocessing_options" in data
