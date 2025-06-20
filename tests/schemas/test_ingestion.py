"""
Comprehensive tests for the ingestion schemas.

This module tests all Pydantic models in the ingestion schemas, including:
- File ingestion models
- Batch processing models  
- Status and configuration models
- Field validation and constraints
"""

import pytest
from pydantic import ValidationError

from app.schemas.ingestion import (
    IngestionRequest, IngestionResponse, BatchIngestionRequest, BatchIngestionResponse,
    FileStatus, BatchStatus, IngestionConfig, FileMetadata, SupportedFormat
)


class TestSupportedFormat:
    """Test the SupportedFormat model."""

    def test_valid_format(self):
        """Test valid supported format."""
        format_info = SupportedFormat(
            name="PDF",
            extensions=[".pdf"],
            mime_types=["application/pdf"],
            max_size_mb=50,
            description="Portable Document Format",
            supports_ocr=True,
            supports_metadata_extraction=True
        )
        
        assert format_info.name == "PDF"
        assert ".pdf" in format_info.extensions
        assert "application/pdf" in format_info.mime_types
        assert format_info.max_size_mb == 50
        assert format_info.supports_ocr is True
        assert format_info.supports_metadata_extraction is True

    def test_invalid_max_size(self):
        """Test invalid max size validation."""
        with pytest.raises(ValidationError) as exc_info:
            SupportedFormat(
                name="PDF",
                extensions=[".pdf"],
                mime_types=["application/pdf"],
                max_size_mb=0  # Below minimum
            )
        assert "ensure this value is greater than 0" in str(exc_info.value)


class TestFileMetadata:
    """Test the FileMetadata model."""

    def test_valid_metadata(self):
        """Test valid file metadata."""
        metadata = FileMetadata(
            filename="test_document.pdf",
            size=1024,
            mime_type="application/pdf",
            created_at="2023-01-15T10:00:00Z",
            modified_at="2023-01-15T10:00:00Z",
            checksum="abc123def456",
            encoding="utf-8",
            language="en",
            page_count=3,
            word_count=250,
            title="Test Document",
            author="Test Author",
            subject="Medical Report",
            keywords=["medical", "report", "patient"]
        )
        
        assert metadata.filename == "test_document.pdf"
        assert metadata.size == 1024
        assert metadata.mime_type == "application/pdf"
        assert metadata.page_count == 3
        assert metadata.word_count == 250
        assert metadata.title == "Test Document"
        assert "medical" in metadata.keywords

    def test_metadata_minimal_fields(self):
        """Test metadata with only required fields."""
        metadata = FileMetadata(
            filename="simple.txt",
            size=512,
            mime_type="text/plain"
        )
        
        assert metadata.filename == "simple.txt"
        assert metadata.size == 512
        assert metadata.mime_type == "text/plain"
        assert metadata.keywords == []

    def test_invalid_size(self):
        """Test invalid file size validation."""
        with pytest.raises(ValidationError) as exc_info:
            FileMetadata(
                filename="test.txt",
                size=-1,  # Negative size
                mime_type="text/plain"
            )
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

    def test_invalid_page_count(self):
        """Test invalid page count validation."""
        with pytest.raises(ValidationError) as exc_info:
            FileMetadata(
                filename="test.pdf",
                size=1024,
                mime_type="application/pdf",
                page_count=0  # Below minimum
            )
        assert "ensure this value is greater than 0" in str(exc_info.value)


class TestIngestionConfig:
    """Test the IngestionConfig model."""

    def test_valid_config(self):
        """Test valid ingestion configuration."""
        config = IngestionConfig(
            extract_text=True,
            extract_metadata=True,
            extract_images=False,
            perform_ocr=True,
            language="en",
            max_file_size_mb=100,
            allowed_formats=[".pdf", ".docx", ".txt"],
            preprocessing=True,
            parallel_processing=True,
            max_workers=4
        )
        
        assert config.extract_text is True
        assert config.extract_metadata is True
        assert config.extract_images is False
        assert config.perform_ocr is True
        assert config.language == "en"
        assert config.max_file_size_mb == 100
        assert ".pdf" in config.allowed_formats
        assert config.parallel_processing is True
        assert config.max_workers == 4

    def test_config_defaults(self):
        """Test configuration with default values."""
        config = IngestionConfig()
        
        assert config.extract_text is True
        assert config.extract_metadata is False
        assert config.extract_images is False
        assert config.perform_ocr is True
        assert config.language == "auto"
        assert config.max_file_size_mb == 50
        assert config.parallel_processing is False
        assert config.max_workers == 1

    def test_invalid_max_file_size(self):
        """Test invalid max file size validation."""
        with pytest.raises(ValidationError) as exc_info:
            IngestionConfig(max_file_size_mb=0)  # Below minimum
        assert "ensure this value is greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            IngestionConfig(max_file_size_mb=1001)  # Above maximum
        assert "ensure this value is less than or equal to 1000" in str(exc_info.value)

    def test_invalid_max_workers(self):
        """Test invalid max workers validation."""
        with pytest.raises(ValidationError) as exc_info:
            IngestionConfig(max_workers=0)  # Below minimum
        assert "ensure this value is greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            IngestionConfig(max_workers=33)  # Above maximum  
        assert "ensure this value is less than or equal to 32" in str(exc_info.value)


class TestIngestionRequest:
    """Test the IngestionRequest model."""

    def test_valid_request(self):
        """Test valid ingestion request."""
        request = IngestionRequest(
            extract_text=True,
            extract_metadata=True,
            extract_images=False,
            perform_ocr=True,
            config=IngestionConfig(
                language="en",
                max_file_size_mb=25
            )
        )
        
        assert request.extract_text is True
        assert request.extract_metadata is True
        assert request.extract_images is False
        assert request.perform_ocr is True
        assert request.config.language == "en"
        assert request.config.max_file_size_mb == 25

    def test_request_defaults(self):
        """Test request with default values."""
        request = IngestionRequest()
        
        assert request.extract_text is True
        assert request.extract_metadata is False
        assert request.extract_images is False
        assert request.perform_ocr is True
        assert request.config.language == "auto"


class TestFileStatus:
    """Test the FileStatus model."""

    def test_valid_status(self):
        """Test valid file status."""
        status = FileStatus(
            file_id="file_123",
            filename="test_document.pdf",
            status="completed",
            progress=100,
            processing_time=2.5,
            created_at="2023-01-15T10:00:00Z",
            updated_at="2023-01-15T10:02:30Z",
            file_size=1024,
            extracted_text_length=500,
            metadata_extracted=True,
            ocr_performed=True,
            error_message=None
        )
        
        assert status.file_id == "file_123"
        assert status.filename == "test_document.pdf"
        assert status.status == "completed"
        assert status.progress == 100
        assert status.processing_time == 2.5
        assert status.file_size == 1024
        assert status.extracted_text_length == 500
        assert status.metadata_extracted is True
        assert status.ocr_performed is True
        assert status.error_message is None

    def test_status_with_error(self):
        """Test file status with error."""
        status = FileStatus(
            file_id="file_456",
            filename="corrupted_file.pdf",
            status="failed",
            progress=0,
            created_at="2023-01-15T10:00:00Z",
            updated_at="2023-01-15T10:01:00Z",
            error_message="File is corrupted and cannot be processed"
        )
        
        assert status.status == "failed"
        assert status.progress == 0
        assert "corrupted" in status.error_message

    def test_invalid_progress(self):
        """Test invalid progress validation."""
        with pytest.raises(ValidationError) as exc_info:
            FileStatus(
                file_id="file_123",
                filename="test.pdf",
                status="processing",
                progress=-1,  # Below minimum
                created_at="2023-01-15T10:00:00Z"
            )
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            FileStatus(
                file_id="file_123",
                filename="test.pdf",
                status="processing",
                progress=101,  # Above maximum
                created_at="2023-01-15T10:00:00Z"
            )
        assert "ensure this value is less than or equal to 100" in str(exc_info.value)

    def test_invalid_status_enum(self):
        """Test invalid status enum validation."""
        with pytest.raises(ValidationError) as exc_info:
            FileStatus(
                file_id="file_123",
                filename="test.pdf",
                status="invalid_status",  # Not in enum
                progress=50,
                created_at="2023-01-15T10:00:00Z"
            )
        assert "value is not a valid enumeration member" in str(exc_info.value)


class TestBatchStatus:
    """Test the BatchStatus model."""

    def test_valid_batch_status(self):
        """Test valid batch status."""
        status = BatchStatus(
            batch_id="batch_456",
            total_files=5,
            processed=3,
            failed=1,
            in_progress=1,
            status="processing",
            progress=60,
            created_at="2023-01-15T10:00:00Z",
            updated_at="2023-01-15T10:05:00Z",
            estimated_completion="2023-01-15T10:15:00Z",
            avg_processing_time=2.1
        )
        
        assert status.batch_id == "batch_456"
        assert status.total_files == 5
        assert status.processed == 3
        assert status.failed == 1
        assert status.in_progress == 1
        assert status.status == "processing"
        assert status.progress == 60
        assert status.avg_processing_time == 2.1

    def test_invalid_file_counts(self):
        """Test invalid file count validation."""
        with pytest.raises(ValidationError) as exc_info:
            BatchStatus(
                batch_id="batch_123",
                total_files=0,  # Below minimum
                processed=0,
                failed=0,
                in_progress=0,
                status="pending",
                progress=0,
                created_at="2023-01-15T10:00:00Z"
            )
        assert "ensure this value is greater than 0" in str(exc_info.value)


class TestIngestionResponse:
    """Test the IngestionResponse model."""

    def test_valid_response(self):
        """Test valid ingestion response."""
        response = IngestionResponse(
            success=True,
            file_id="file_123",
            filename="test_document.pdf",
            file_type="pdf",
            size=1024,
            processing_status="completed",
            processing_time=2.5,
            extracted_text="This is the extracted text content...",
            metadata=FileMetadata(
                filename="test_document.pdf",
                size=1024,
                mime_type="application/pdf",
                page_count=3,
                title="Test Document"
            ),
            ocr_performed=True,
            images_extracted=False,
            text_length=250
        )
        
        assert response.success is True
        assert response.file_id == "file_123"
        assert response.filename == "test_document.pdf"
        assert response.file_type == "pdf"
        assert response.size == 1024
        assert response.processing_status == "completed"
        assert "extracted text content" in response.extracted_text
        assert response.metadata.page_count == 3
        assert response.ocr_performed is True
        assert response.text_length == 250

    def test_error_response(self):
        """Test error ingestion response."""
        response = IngestionResponse(
            success=False,
            file_id="",
            filename="corrupted_file.pdf",
            file_type="pdf",
            size=0,
            processing_status="failed",
            processing_time=0.0,
            error_message="File processing failed due to corruption"
        )
        
        assert response.success is False
        assert response.processing_status == "failed"
        assert "corruption" in response.error_message


class TestBatchIngestionRequest:
    """Test the BatchIngestionRequest model."""

    def test_valid_batch_request(self):
        """Test valid batch ingestion request."""
        request = BatchIngestionRequest(
            extract_text=True,
            extract_metadata=True,
            parallel_processing=True,
            max_workers=4,
            config=IngestionConfig(
                language="en",
                max_file_size_mb=50
            )
        )
        
        assert request.extract_text is True
        assert request.extract_metadata is True
        assert request.parallel_processing is True
        assert request.max_workers == 4
        assert request.config.language == "en"

    def test_invalid_max_workers(self):
        """Test invalid max workers validation."""
        with pytest.raises(ValidationError) as exc_info:
            BatchIngestionRequest(max_workers=0)  # Below minimum
        assert "ensure this value is greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            BatchIngestionRequest(max_workers=17)  # Above maximum
        assert "ensure this value is less than or equal to 16" in str(exc_info.value)


class TestBatchIngestionResponse:
    """Test the BatchIngestionResponse model."""

    def test_valid_batch_response(self):
        """Test valid batch ingestion response."""
        response = BatchIngestionResponse(
            success=True,
            batch_id="batch_456",
            total_files=3,
            processed=3,
            failed=0,
            processing_time=8.5,
            results=[
                {
                    "file_id": "file_1",
                    "filename": "doc1.pdf",
                    "status": "completed",
                    "processing_time": 2.1,
                    "success": True
                },
                {
                    "file_id": "file_2",
                    "filename": "doc2.pdf",
                    "status": "completed",
                    "processing_time": 3.2,
                    "success": True
                },
                {
                    "file_id": "file_3",
                    "filename": "doc3.pdf",
                    "status": "completed",
                    "processing_time": 2.8,
                    "success": True
                }
            ],
            parallel_processed=True,
            avg_processing_time=2.7
        )
        
        assert response.success is True
        assert response.batch_id == "batch_456"
        assert response.total_files == 3
        assert response.processed == 3
        assert response.failed == 0
        assert len(response.results) == 3
        assert response.parallel_processed is True
        assert response.avg_processing_time == 2.7

    def test_batch_response_with_failures(self):
        """Test batch response with some failures."""
        response = BatchIngestionResponse(
            success=True,
            batch_id="batch_789",
            total_files=2,
            processed=1,
            failed=1,
            processing_time=3.0,
            results=[
                {
                    "file_id": "file_1",
                    "filename": "doc1.pdf",
                    "status": "completed",
                    "processing_time": 2.5,
                    "success": True
                },
                {
                    "file_id": "",
                    "filename": "corrupted.pdf",
                    "status": "failed",
                    "processing_time": 0.0,
                    "success": False,
                    "error_message": "File corrupted"
                }
            ]
        )
        
        assert response.success is True
        assert response.total_files == 2
        assert response.processed == 1
        assert response.failed == 1
        assert response.results[1]["success"] is False
        assert "corrupted" in response.results[1]["error_message"]
