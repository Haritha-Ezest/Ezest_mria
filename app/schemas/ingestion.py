"""
Ingestion schemas for file upload and processing requests/responses.

This module defines the Pydantic models used for validating and serializing
data related to document ingestion and file upload operations in the MRIA system.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Supported document types for medical record processing."""
    
    LAB_REPORT = "lab_report"
    PRESCRIPTION = "prescription"
    MEDICAL_REPORT = "medical_report"
    RADIOLOGY = "radiology"
    DISCHARGE_SUMMARY = "discharge_summary"
    CLINICAL_NOTES = "clinical_notes"
    CONSENT_FORM = "consent_form"
    INSURANCE_CLAIM = "insurance_claim"
    OTHER = "other"


class FileStatus(str, Enum):
    """File processing status enumeration."""
    
    UPLOADED = "uploaded"
    VALIDATED = "validated"
    QUEUED_FOR_OCR = "queued_for_ocr"
    OCR_IN_PROGRESS = "ocr_in_progress"
    OCR_COMPLETED = "ocr_completed"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class ProcessedFileInfo(BaseModel):
    """Information about a processed file."""
    
    filename: str = Field(..., description="Original filename of the uploaded file")
    file_id: str = Field(..., description="Unique identifier for the file")
    size: str = Field(..., description="Human-readable file size (e.g., '2.4MB')")
    mime_type: str = Field(..., description="MIME type of the file")
    status: FileStatus = Field(..., description="Current processing status")
    content_preview: Optional[str] = Field(None, description="Preview of extracted content")
    confidence_score: Optional[float] = Field(None, description="OCR confidence score (0-1)")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class PatientContext(BaseModel):
    """Patient context information for file uploads."""
    
    patient_id: str = Field(..., description="Unique patient identifier")
    medical_record_number: Optional[str] = Field(None, description="Medical record number")
    visit_date: Optional[str] = Field(None, description="Visit date (YYYY-MM-DD)")
    provider: Optional[str] = Field(None, description="Healthcare provider name")
    clinic: Optional[str] = Field(None, description="Clinic or facility name")
    visit_type: Optional[str] = Field(None, description="Type of visit (routine, emergency, follow_up)")


class FileUploadRequest(BaseModel):
    """Request model for file upload operations."""
    
    patient_id: str = Field(..., description="Patient identifier for the uploaded files")
    document_types: List[DocumentType] = Field(..., description="Types of documents being uploaded")
    visit_date: Optional[str] = Field(None, description="Visit date (YYYY-MM-DD)")
    provider: Optional[str] = Field(None, description="Healthcare provider name")
    clinic: Optional[str] = Field(None, description="Clinic or facility name")
    priority: Optional[str] = Field("normal", description="Processing priority (low, normal, high, urgent)")
    notes: Optional[str] = Field(None, description="Additional notes or context")
    
    @validator('document_types')
    def validate_document_types(cls, v):
        """Ensure document types list is not empty."""
        if not v:
            raise ValueError("At least one document type must be specified")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority values."""
        if v not in ['low', 'normal', 'high', 'urgent']:
            raise ValueError("Priority must be one of: low, normal, high, urgent")
        return v


class FileUploadResponse(BaseModel):
    """Response model for successful file uploads."""
    
    upload_id: str = Field(..., description="Unique identifier for this upload batch")
    status: str = Field(..., description="Overall upload status")
    message: str = Field(..., description="Human-readable status message")
    files_processed: List[ProcessedFileInfo] = Field(..., description="Information about processed files")
    patient_context: PatientContext = Field(..., description="Patient context information")
    next_steps: List[str] = Field(..., description="Next processing steps")
    estimated_completion: str = Field(..., description="Estimated completion time")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UploadStatusResponse(BaseModel):
    """Response model for upload status queries."""
    
    upload_id: str = Field(..., description="Upload batch identifier")
    status: str = Field(..., description="Current processing status")
    message: str = Field(..., description="Status description")
    files_count: int = Field(..., description="Total number of files")
    files_completed: int = Field(..., description="Number of completed files")
    files_failed: int = Field(..., description="Number of failed files")
    processing_time: str = Field(..., description="Total processing time")
    progress_percentage: float = Field(..., description="Processing progress (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    error_details: Optional[List[str]] = Field(None, description="Error details if any")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UploadResultsResponse(BaseModel):
    """Response model for retrieving upload results."""
    
    upload_id: str = Field(..., description="Upload batch identifier")
    message: str = Field(..., description="Response message")
    files: List[ProcessedFileInfo] = Field(..., description="Detailed file information")
    patient_id: str = Field(..., description="Patient identifier")
    processing_summary: Dict[str, Any] = Field(..., description="Processing summary statistics")
    available_actions: List[str] = Field(..., description="Available next actions")
    created_at: datetime = Field(..., description="Upload creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }