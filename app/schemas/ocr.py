"""
Pydantic schemas for OCR (Optical Character Recognition) processing.

This module defines data models for OCR requests, responses, and configuration
for the medical document text extraction system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from pydantic import BaseModel, Field, validator


class OCREngine(str, Enum):
    """Supported OCR engines."""
    TESSERACT = "tesseract"
    AZURE_FORM_RECOGNIZER = "azure_form_recognizer"
    GOOGLE_VISION = "google_vision"
    AWS_TEXTRACT = "aws_textract"


class ImagePreprocessing(str, Enum):
    """Image preprocessing options."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MEDICAL_OPTIMIZED = "medical_optimized"


class DocumentType(str, Enum):
    """Medical document types for optimized processing."""
    GENERAL = "general"
    LAB_REPORT = "lab_report"
    PRESCRIPTION = "prescription"
    RADIOLOGY_REPORT = "radiology_report"
    DISCHARGE_SUMMARY = "discharge_summary"
    CLINICAL_NOTES = "clinical_notes"
    INSURANCE_FORM = "insurance_form"
    PATHOLOGY_REPORT = "pathology_report"


class OCRStatus(str, Enum):
    """OCR processing status."""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    EXTRACTING = "extracting"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BoundingBox(BaseModel):
    """Bounding box coordinates for text regions."""
    
    x: float = Field(..., ge=0, description="X coordinate (left)")
    y: float = Field(..., ge=0, description="Y coordinate (top)")
    width: float = Field(..., gt=0, description="Width of the bounding box")
    height: float = Field(..., gt=0, description="Height of the bounding box")
    
    def __str__(self) -> str:
        return f"BoundingBox(x={self.x}, y={self.y}, w={self.width}, h={self.height})"


class TextRegion(BaseModel):
    """Extracted text region with positional information."""
    
    text: str = Field(..., description="Extracted text content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")
    bounding_box: BoundingBox = Field(..., description="Text position coordinates")
    
    # Text properties
    font_size: Optional[float] = Field(None, description="Estimated font size")
    is_bold: Optional[bool] = Field(None, description="Whether text appears bold")
    is_italic: Optional[bool] = Field(None, description="Whether text appears italic")
    
    # Medical document specific
    is_header: bool = Field(default=False, description="Whether this is a header/title")
    is_value: bool = Field(default=False, description="Whether this is a numerical value")
    is_label: bool = Field(default=False, description="Whether this is a field label")


class OCRConfiguration(BaseModel):
    """Configuration for OCR processing."""
    
    # Engine selection
    engine: OCREngine = Field(default=OCREngine.TESSERACT, description="OCR engine to use")
    
    # Language settings
    languages: List[str] = Field(
        default=["eng"],
        description="Languages for OCR (ISO 639-3 codes)"
    )
    
    # Image preprocessing
    preprocessing: ImagePreprocessing = Field(
        default=ImagePreprocessing.MEDICAL_OPTIMIZED,
        description="Image preprocessing level"
    )
    
    # Document type optimization
    document_type: DocumentType = Field(
        default=DocumentType.GENERAL,
        description="Document type for optimized processing"
    )
    
    # Processing parameters
    dpi: int = Field(default=300, ge=150, le=600, description="Target DPI for processing")
    enhance_contrast: bool = Field(default=True, description="Apply contrast enhancement")
    correct_skew: bool = Field(default=True, description="Apply skew correction")
    remove_noise: bool = Field(default=True, description="Apply noise reduction")
    
    # Output options
    preserve_layout: bool = Field(default=True, description="Preserve document layout")
    extract_tables: bool = Field(default=True, description="Extract table structures")
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for text"
    )
    
    # Performance settings
    max_pages: Optional[int] = Field(None, gt=0, description="Maximum pages to process")
    timeout_seconds: int = Field(default=300, ge=30, description="Processing timeout")


class OCRRequest(BaseModel):
    """Request model for OCR text extraction."""
    
    # Document identification
    document_id: str = Field(..., description="Unique document identifier")
    file_path: Optional[str] = Field(None, description="Path to document file")
    file_url: Optional[str] = Field(None, description="URL to document file")
    
    # Patient and metadata
    patient_id: Optional[str] = Field(None, description="Associated patient ID")
    job_id: Optional[str] = Field(None, description="Parent job identifier")
    
    # Processing configuration
    config: OCRConfiguration = Field(
        default_factory=OCRConfiguration,
        description="OCR processing configuration"
    )
    
    # Priority and scheduling
    priority: str = Field(default="normal", description="Processing priority")
    callback_url: Optional[str] = Field(None, description="Callback URL for results")
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @validator('file_path', 'file_url')
    def validate_file_source(cls, v, values):
        """Ensure either file_path or file_url is provided."""
        if not any([v, values.get('file_path'), values.get('file_url')]):
            raise ValueError("Either file_path or file_url must be provided")
        return v


class OCRPage(BaseModel):
    """OCR results for a single page."""
    
    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    
    # Extracted content
    text: str = Field(..., description="Full page text content")
    text_regions: List[TextRegion] = Field(
        default_factory=list,
        description="Individual text regions with positions"
    )
    
    # Page metrics
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall page confidence")
    processing_time_seconds: float = Field(..., description="Page processing time")
    
    # Image properties
    image_dimensions: Tuple[int, int] = Field(..., description="Image width and height")
    image_dpi: int = Field(..., description="Image resolution")
    
    # Quality metrics
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Image quality score")
    issues_detected: List[str] = Field(
        default_factory=list,
        description="Quality issues detected"
    )
    
    # Medical document structure
    tables_detected: int = Field(default=0, description="Number of tables detected")
    has_handwriting: bool = Field(default=False, description="Handwriting detected")
    has_signatures: bool = Field(default=False, description="Signatures detected")


class OCRResponse(BaseModel):
    """Response model for OCR processing results."""
    
    # Request identification
    document_id: str = Field(..., description="Document identifier")
    job_id: Optional[str] = Field(None, description="Parent job identifier")
    
    # Processing status
    status: OCRStatus = Field(..., description="OCR processing status")
    
    # Results
    pages: List[OCRPage] = Field(
        default_factory=list,
        description="Per-page OCR results"
    )
    
    # Aggregated content
    full_text: str = Field(default="", description="Complete document text")
    
    # Metrics
    total_pages: int = Field(..., description="Total number of pages processed")
    pages_successful: int = Field(default=0, description="Successfully processed pages")
    pages_failed: int = Field(default=0, description="Failed pages")
    
    # Quality metrics
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    average_processing_time: float = Field(..., description="Average time per page")
    total_processing_time: float = Field(..., description="Total processing time")
    
    # Document analysis
    estimated_word_count: int = Field(..., description="Estimated word count")
    languages_detected: List[str] = Field(
        default_factory=list,
        description="Languages detected in document"
    )
    
    # Medical document insights
    document_type_detected: Optional[DocumentType] = Field(
        None, description="Detected document type"
    )
    medical_entities_preview: List[str] = Field(
        default_factory=list,
        description="Preview of potential medical entities"
    )
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    warnings: List[str] = Field(
        default_factory=list,
        description="Processing warnings"
    )
    
    # Timestamps
    started_at: datetime = Field(..., description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    
    # File information
    original_filename: Optional[str] = Field(None, description="Original file name")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    file_format: Optional[str] = Field(None, description="File format/MIME type")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OCRBatchRequest(BaseModel):
    """Request model for batch OCR processing."""
    
    batch_id: str = Field(..., description="Batch identifier")
    documents: List[OCRRequest] = Field(
        ..., min_items=1, description="Documents to process"
    )
    
    # Batch configuration
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_concurrent: int = Field(default=3, ge=1, le=10, description="Max concurrent jobs")
    
    # Global configuration
    global_config: Optional[OCRConfiguration] = Field(
        None, description="Global configuration for all documents"
    )
    
    # Callback settings
    batch_callback_url: Optional[str] = Field(None, description="Batch completion callback")
    individual_callbacks: bool = Field(
        default=False, description="Send callbacks for individual documents"
    )


class OCRBatchResponse(BaseModel):
    """Response model for batch OCR processing."""
    
    batch_id: str = Field(..., description="Batch identifier")
    status: OCRStatus = Field(..., description="Batch processing status")
    
    # Results
    results: List[OCRResponse] = Field(
        default_factory=list,
        description="Individual OCR results"
    )
    
    # Batch metrics
    total_documents: int = Field(..., description="Total documents in batch")
    completed_documents: int = Field(default=0, description="Completed documents")
    failed_documents: int = Field(default=0, description="Failed documents")
    
    # Timing
    batch_started_at: datetime = Field(..., description="Batch start time")
    batch_completed_at: Optional[datetime] = Field(None, description="Batch completion time")
    total_processing_time: float = Field(default=0.0, description="Total processing time")
    
    # Error summary
    error_summary: List[str] = Field(
        default_factory=list,
        description="Summary of errors encountered"
    )
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OCRHealthCheck(BaseModel):
    """Health check response for OCR service."""
    
    service_name: str = Field(default="OCR Service", description="Service name")
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    
    # Engine status
    available_engines: List[OCREngine] = Field(
        default_factory=list,
        description="Available OCR engines"
    )
    default_engine: OCREngine = Field(..., description="Default OCR engine")
    
    # Performance metrics
    average_response_time: float = Field(..., description="Average response time (seconds)")
    requests_processed: int = Field(..., description="Total requests processed")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate")
    
    # Resource usage
    cpu_usage: float = Field(..., description="Current CPU usage (%)")
    memory_usage: float = Field(..., description="Current memory usage (%)")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check time")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
