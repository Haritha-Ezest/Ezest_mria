"""
Comprehensive tests for the OCR schemas.

This module tests all Pydantic models in the OCR schemas, including:
- Request and response models
- Field validation and constraints
- File format validation
- Processing options
"""

import pytest
from pydantic import ValidationError

from app.schemas.ocr import (
    OCRRequest, OCRResponse, BatchOCRRequest, BatchOCRResponse,
    OCRConfig, ProcessedPage, BoundingBox
)


class TestBoundingBox:
    """Test the BoundingBox model."""

    def test_valid_bounding_box(self):
        """Test valid bounding box."""
        bbox = BoundingBox(
            x=10,
            y=20,
            width=100,
            height=50,
            confidence=0.95
        )
        
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.confidence == 0.95

    def test_invalid_coordinates(self):
        """Test invalid coordinate validation."""
        with pytest.raises(ValidationError) as exc_info:
            BoundingBox(
                x=-1,  # Negative coordinate
                y=20,
                width=100,
                height=50,
                confidence=0.95
            )
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

    def test_invalid_dimensions(self):
        """Test invalid dimension validation."""
        with pytest.raises(ValidationError) as exc_info:
            BoundingBox(
                x=10,
                y=20,
                width=0,  # Zero width
                height=50,
                confidence=0.95
            )
        assert "ensure this value is greater than 0" in str(exc_info.value)

    def test_invalid_confidence(self):
        """Test invalid confidence validation."""
        with pytest.raises(ValidationError) as exc_info:
            BoundingBox(
                x=10,
                y=20,
                width=100,
                height=50,
                confidence=1.5  # Above 1
            )
        assert "ensure this value is less than or equal to 1" in str(exc_info.value)


class TestProcessedPage:
    """Test the ProcessedPage model."""

    def test_valid_page(self):
        """Test valid processed page."""
        page = ProcessedPage(
            page_number=1,
            text="This is extracted text from page 1.",
            confidence=0.92,
            width=800,
            height=600,
            dpi=300,
            bounding_boxes=[
                BoundingBox(
                    x=10,
                    y=20,
                    width=100,
                    height=20,
                    confidence=0.95
                )
            ],
            word_count=7,
            processing_time=1.2
        )
        
        assert page.page_number == 1
        assert "extracted text" in page.text
        assert page.confidence == 0.92
        assert page.width == 800
        assert page.height == 600
        assert page.dpi == 300
        assert len(page.bounding_boxes) == 1
        assert page.word_count == 7
        assert page.processing_time == 1.2

    def test_page_minimal_fields(self):
        """Test page with only required fields."""
        page = ProcessedPage(
            page_number=1,
            text="Simple text",
            confidence=0.88
        )
        
        assert page.page_number == 1
        assert page.text == "Simple text"
        assert page.confidence == 0.88
        assert page.bounding_boxes == []
        assert page.word_count is None

    def test_invalid_page_number(self):
        """Test invalid page number validation."""
        with pytest.raises(ValidationError) as exc_info:
            ProcessedPage(
                page_number=0,  # Below minimum
                text="text",
                confidence=0.9
            )
        assert "ensure this value is greater than 0" in str(exc_info.value)

    def test_invalid_confidence(self):
        """Test invalid confidence validation."""
        with pytest.raises(ValidationError) as exc_info:
            ProcessedPage(
                page_number=1,
                text="text",
                confidence=-0.1  # Below 0
            )
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)


class TestOCRConfig:
    """Test the OCRConfig model."""

    def test_valid_config(self):
        """Test valid OCR configuration."""
        config = OCRConfig(
            language="en",
            psm=6,
            oem=3,
            dpi=300,
            preprocessing=True,
            denoise=True,
            deskew=True,
            enhance_contrast=True,
            extract_tables=False,
            confidence_threshold=0.7
        )
        
        assert config.language == "en"
        assert config.psm == 6
        assert config.oem == 3
        assert config.dpi == 300
        assert config.preprocessing is True
        assert config.denoise is True
        assert config.deskew is True
        assert config.enhance_contrast is True
        assert config.extract_tables is False
        assert config.confidence_threshold == 0.7

    def test_default_config(self):
        """Test default OCR configuration."""
        config = OCRConfig()
        
        assert config.language == "en"
        assert config.psm == 6
        assert config.oem == 3
        assert config.dpi == 300
        assert config.preprocessing is True
        assert config.confidence_threshold == 0.5

    def test_invalid_psm(self):
        """Test invalid PSM validation."""
        with pytest.raises(ValidationError) as exc_info:
            OCRConfig(psm=-1)  # Below minimum
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            OCRConfig(psm=14)  # Above maximum
        assert "ensure this value is less than or equal to 13" in str(exc_info.value)

    def test_invalid_oem(self):
        """Test invalid OEM validation."""
        with pytest.raises(ValidationError) as exc_info:
            OCRConfig(oem=-1)  # Below minimum
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            OCRConfig(oem=4)  # Above maximum
        assert "ensure this value is less than or equal to 3" in str(exc_info.value)

    def test_invalid_dpi(self):
        """Test invalid DPI validation."""
        with pytest.raises(ValidationError) as exc_info:
            OCRConfig(dpi=49)  # Below minimum
        assert "ensure this value is greater than or equal to 50" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            OCRConfig(dpi=1201)  # Above maximum
        assert "ensure this value is less than or equal to 1200" in str(exc_info.value)


class TestOCRRequest:
    """Test the OCRRequest model."""

    def test_valid_request(self):
        """Test valid OCR request."""
        request = OCRRequest(
            extract_text=True,
            extract_metadata=True,
            extract_tables=False,
            return_confidence=True,
            return_bounding_boxes=True,
            config=OCRConfig(
                language="en",
                dpi=300,
                confidence_threshold=0.8
            )
        )
        
        assert request.extract_text is True
        assert request.extract_metadata is True
        assert request.extract_tables is False
        assert request.return_confidence is True
        assert request.return_bounding_boxes is True
        assert request.config.language == "en"
        assert request.config.dpi == 300

    def test_request_with_defaults(self):
        """Test OCR request with default values."""
        request = OCRRequest()
        
        assert request.extract_text is True
        assert request.extract_metadata is False
        assert request.extract_tables is False
        assert request.return_confidence is True
        assert request.return_bounding_boxes is False
        assert request.config.language == "en"


class TestOCRResponse:
    """Test the OCRResponse model."""

    def test_valid_response(self):
        """Test valid OCR response."""
        response = OCRResponse(
            success=True,
            text="Extracted text from document",
            confidence=0.94,
            processing_time=2.5,
            page_count=1,
            pages=[
                ProcessedPage(
                    page_number=1,
                    text="Extracted text from document",
                    confidence=0.94
                )
            ],
            metadata={
                "title": "Test Document",
                "author": "Test Author",
                "creation_date": "2023-01-15",
                "format": "PDF",
                "size": 1024
            },
            tables_extracted=False,
            bounding_boxes_included=False
        )
        
        assert response.success is True
        assert "Extracted text" in response.text
        assert response.confidence == 0.94
        assert response.processing_time == 2.5
        assert response.page_count == 1
        assert len(response.pages) == 1
        assert response.metadata["title"] == "Test Document"
        assert response.tables_extracted is False

    def test_error_response(self):
        """Test error OCR response."""
        response = OCRResponse(
            success=False,
            text="",
            confidence=0.0,
            processing_time=0.0,
            page_count=0,
            pages=[],
            error_message="OCR processing failed due to corrupted file."
        )
        
        assert response.success is False
        assert response.text == ""
        assert response.page_count == 0
        assert len(response.pages) == 0
        assert "OCR processing failed" in response.error_message

    def test_multi_page_response(self):
        """Test multi-page OCR response."""
        response = OCRResponse(
            success=True,
            text="Page 1 text\nPage 2 text",
            confidence=0.89,
            processing_time=4.2,
            page_count=2,
            pages=[
                ProcessedPage(
                    page_number=1,
                    text="Page 1 text",
                    confidence=0.92
                ),
                ProcessedPage(
                    page_number=2,
                    text="Page 2 text",
                    confidence=0.86
                )
            ]
        )
        
        assert response.success is True
        assert response.page_count == 2
        assert len(response.pages) == 2
        assert response.pages[0].page_number == 1
        assert response.pages[1].page_number == 2


class TestBatchOCRRequest:
    """Test the BatchOCRRequest model."""

    def test_valid_batch_request(self):
        """Test valid batch OCR request."""
        request = BatchOCRRequest(
            extract_text=True,
            extract_metadata=True,
            parallel_processing=True,
            max_workers=4,
            config=OCRConfig(
                language="en",
                confidence_threshold=0.8
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
            BatchOCRRequest(max_workers=0)  # Below minimum
        assert "ensure this value is greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            BatchOCRRequest(max_workers=17)  # Above maximum
        assert "ensure this value is less than or equal to 16" in str(exc_info.value)


class TestBatchOCRResponse:
    """Test the BatchOCRResponse model."""

    def test_valid_batch_response(self):
        """Test valid batch OCR response."""
        response = BatchOCRResponse(
            success=True,
            results=[
                {
                    "filename": "doc1.pdf",
                    "success": True,
                    "text": "Document 1 content",
                    "confidence": 0.94,
                    "processing_time": 2.1,
                    "page_count": 1
                },
                {
                    "filename": "doc2.pdf",
                    "success": True,
                    "text": "Document 2 content",
                    "confidence": 0.88,
                    "processing_time": 2.8,
                    "page_count": 2
                }
            ],
            total_files=2,
            successful=2,
            failed=0,
            total_processing_time=4.9,
            avg_processing_time=2.45,
            parallel_processed=True
        )
        
        assert response.success is True
        assert len(response.results) == 2
        assert response.total_files == 2
        assert response.successful == 2
        assert response.failed == 0
        assert response.total_processing_time == 4.9
        assert response.avg_processing_time == 2.45
        assert response.parallel_processed is True

    def test_batch_response_with_failures(self):
        """Test batch OCR response with some failures."""
        response = BatchOCRResponse(
            success=True,
            results=[
                {
                    "filename": "doc1.pdf",
                    "success": True,
                    "text": "Document 1 content",
                    "confidence": 0.94,
                    "processing_time": 2.1,
                    "page_count": 1
                },
                {
                    "filename": "doc2.pdf",
                    "success": False,
                    "text": "",
                    "confidence": 0.0,
                    "processing_time": 0.0,
                    "page_count": 0,
                    "error_message": "File corrupted"
                }
            ],
            total_files=2,
            successful=1,
            failed=1,
            total_processing_time=2.1,
            avg_processing_time=2.1
        )
        
        assert response.success is True
        assert response.total_files == 2
        assert response.successful == 1
        assert response.failed == 1
        assert response.results[1]["success"] is False
        assert "File corrupted" in response.results[1]["error_message"]
