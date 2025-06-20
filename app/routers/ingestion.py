"""
Ingestion router for document upload and file processing endpoints.

This module provides REST API endpoints for medical document ingestion,
including file uploads, status tracking, and results retrieval.
It implements the "Doctor uploads a record → File Upload Service" workflow.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import difflib  # Add import for fuzzy matching

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse

from app.schemas.ingestion import (
    FileUploadResponse, 
    UploadStatusResponse, 
    UploadResultsResponse,
    ProcessedFileInfo,    PatientContext,
    DocumentType,
    FileStatus
)
from app.services.file_handler import FileHandlerService, FileValidationError
from app.common.utils import format_processing_time
from app.config import get_storage_config


# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Initialize file handler service
file_handler = FileHandlerService()

# In-memory storage for upload tracking (in production, use Redis or database)
upload_tracking: Dict[str, Dict[str, Any]] = {}


async def queue_for_ocr_processing(upload_id: str, file_ids: List[str]) -> None:
    """
    Background task to queue files for OCR processing.
    
    Args:
        upload_id: Upload batch identifier
        file_ids: List of file IDs to process
    """
    try:
        # Update status to indicate OCR queuing
        if upload_id in upload_tracking:
            upload_tracking[upload_id]['status'] = 'ocr_queued'
            upload_tracking[upload_id]['current_step'] = 'OCR Processing'
            upload_tracking[upload_id]['updated_at'] = datetime.utcnow()
            
            # Update individual file statuses
            for file_info in upload_tracking[upload_id]['files']:
                if file_info['file_id'] in file_ids:
                    file_info['status'] = FileStatus.QUEUED_FOR_OCR.value
        
        # Here you would typically send files to OCR service queue
        # For now, we'll simulate the queuing process
        logger.info(f"Files queued for OCR processing: {file_ids}")
        
    except Exception as e:
        logger.error(f"Error queuing files for OCR: {str(e)}")


def get_closest_document_types(invalid_type: str, valid_types: List[str], cutoff: float = 0.6) -> List[str]:
    """
    Find the closest matching valid document types for a given invalid type using fuzzy matching.
    
    Args:
        invalid_type: The invalid document type string
        valid_types: List of valid document type strings
        cutoff: Similarity threshold (0-1)
        
    Returns:
        List of similar valid document types
    """
    return difflib.get_close_matches(invalid_type.lower(), [t.lower() for t in valid_types], n=3, cutoff=cutoff)


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    summary="Upload Medical Documents",
    description="Upload medical documents for processing. Supports multiple files and document types.",
    responses={
        200: {"description": "Files uploaded successfully"},
        400: {"description": "Invalid file format or validation error"},
        413: {"description": "File too large"},
        500: {"description": "Internal server error"}
    }
)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Medical document files to upload"),
    patient_id: str = Form(..., description="Patient identifier"),
    document_types: str = Form(..., description="Comma-separated document types"),
    visit_date: Optional[str] = Form(None, description="Visit date (YYYY-MM-DD)"),
    provider: Optional[str] = Form(None, description="Healthcare provider name"),
    clinic: Optional[str] = Form(None, description="Clinic or facility name"),
    visit_type: Optional[str] = Form("routine", description="Type of visit"),
    priority: str = Form("normal", description="Processing priority"),
    notes: Optional[str] = Form(None, description="Additional notes")
) -> FileUploadResponse:
    """
    Upload medical documents for processing.
    
    This endpoint implements the core "Doctor uploads a record → File Upload Service" workflow:
    1. Validates uploaded files for security and format compliance
    2. Stores files securely with metadata
    3. Creates processing job for downstream agents
    4. Returns upload confirmation with tracking information
    
    Args:
        background_tasks: FastAPI background tasks
        files: List of uploaded files
        patient_id: Patient identifier
        document_types: Comma-separated document types
        visit_date: Optional visit date
        provider: Optional provider name
        clinic: Optional clinic name
        visit_type: Type of medical visit
        priority: Processing priority level
        notes: Additional context notes
        
    Returns:
        FileUploadResponse with upload details and tracking information
        
    Raises:
        HTTPException: For validation errors or processing failures
    """
    logger.info(f"Processing file upload for patient {patient_id}: {len(files)} files")
    
    try:
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        # Parse document types
        doc_types = [dtype.strip() for dtype in document_types.split(',')]
        parsed_doc_types = []
        valid_types = [t.value for t in DocumentType]
        
        for doc_type in doc_types:
            try:
                # Try case-insensitive match first
                matched = False
                for valid_type in valid_types:
                    if doc_type.lower() == valid_type.lower():
                        parsed_doc_types.append(DocumentType(valid_type))
                        matched = True
                        break
                
                if not matched:
                    # If no case-insensitive match, try to use the enum directly
                    parsed_doc_types.append(DocumentType(doc_type))
                    
            except ValueError:
                # Find similar document types for helpful suggestions
                similar_types = get_closest_document_types(doc_type, valid_types)
                suggestion_msg = ""
                
                if similar_types:
                    suggestion_msg = f" Did you mean: {', '.join(similar_types)}?"
                
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid document type: {doc_type}. Valid document types are: {', '.join(valid_types)}.{suggestion_msg}"
                )
        
        # Validate number of files matches document types
        if len(files) != len(parsed_doc_types):
            raise HTTPException(
                status_code=400,
                detail=f"Number of files ({len(files)}) must match number of document types ({len(parsed_doc_types)})"
            )
        
        # Validate priority
        if priority not in ['low', 'normal', 'high', 'urgent']:
            raise HTTPException(
                status_code=400,
                detail="Priority must be one of: low, normal, high, urgent"
            )
        
        # Process each file
        processed_files = []
        file_ids = []
        
        for i, (file, doc_type) in enumerate(zip(files, parsed_doc_types)):
            try:
                # Store file securely
                file_metadata = await file_handler.store_file(file, patient_id, doc_type)
                
                # Get file information
                file_info = file_handler.get_file_info(file_metadata.stored_path)
                
                # Create processed file info
                processed_file = ProcessedFileInfo(
                    filename=file_metadata.filename,
                    file_id=file_metadata.file_id,
                    size=file_info.get('size_human', 'Unknown'),
                    mime_type=file_metadata.mime_type,
                    status=FileStatus.UPLOADED
                )
                
                processed_files.append(processed_file)
                file_ids.append(file_metadata.file_id)
                
                logger.info(f"File processed successfully: {file_metadata.filename} -> {file_metadata.file_id}")
                
            except FileValidationError as e:
                logger.error(f"File validation failed for {file.filename}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")
            except Exception as e:
                logger.error(f"File processing error for {file.filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")
        
        # Create patient context
        patient_context = PatientContext(
            patient_id=patient_id,
            visit_date=visit_date,
            provider=provider,
            clinic=clinic,
            visit_type=visit_type
        )
        
        # Store upload tracking information
        upload_tracking[upload_id] = {
            'upload_id': upload_id,
            'status': 'received',
            'patient_context': patient_context.dict(),
            'files': [pf.dict() for pf in processed_files],
            'priority': priority,
            'notes': notes,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'current_step': 'File Upload',
            'progress': 100.0,
            'total_files': len(files),
            'completed_files': len(files),
            'failed_files': 0
        }
        
        # Queue files for OCR processing in background
        background_tasks.add_task(queue_for_ocr_processing, upload_id, file_ids)
        
        # Prepare response
        response = FileUploadResponse(
            upload_id=upload_id,
            status="received",
            message="Medical documents uploaded successfully for processing",
            files_processed=processed_files,
            patient_context=patient_context,
            next_steps=[
                "OCR text extraction",
                "Medical entity recognition",
                "Timeline integration",
                "Graph database update"
            ],
            estimated_completion="2-3 minutes"
        )
        
        logger.info(f"Upload completed successfully: {upload_id} ({len(files)} files)")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in file upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during file upload"
        )


@router.get(
    "/status/{upload_id}",
    response_model=UploadStatusResponse,
    summary="Check Upload Status",
    description="Check the processing status of an uploaded document batch.",
    responses={
        200: {"description": "Status retrieved successfully"},
        404: {"description": "Upload not found"},
        500: {"description": "Internal server error"}
    }
)
async def get_upload_status(upload_id: str) -> UploadStatusResponse:
    """
    Check the processing status of an uploaded document batch.
    
    Args:
        upload_id: Unique upload identifier
        
    Returns:
        UploadStatusResponse with current processing status
        
    Raises:
        HTTPException: If upload not found or error occurs
    """
    logger.info(f"Checking status for upload: {upload_id}")
    
    try:
        # Check if upload exists
        if upload_id not in upload_tracking:
            raise HTTPException(
                status_code=404,
                detail=f"Upload not found: {upload_id}"
            )
        
        upload_info = upload_tracking[upload_id]
        
        # Calculate processing time
        created_at = upload_info['created_at']
        processing_time = format_processing_time(
            (datetime.utcnow() - created_at).total_seconds()
        )
        
        # Calculate progress
        total_files = upload_info['total_files']
        completed_files = upload_info['completed_files']
        progress = (completed_files / total_files) * 100.0 if total_files > 0 else 0.0
        
        response = UploadStatusResponse(
            upload_id=upload_id,
            status=upload_info['status'],
            message="Upload processing status retrieved successfully",
            files_count=total_files,
            files_completed=completed_files,
            files_failed=upload_info['failed_files'],
            processing_time=processing_time,
            progress_percentage=progress,
            current_step=upload_info.get('current_step'),
            error_details=upload_info.get('error_details')
        )
        
        logger.info(f"Status retrieved for upload {upload_id}: {upload_info['status']}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving upload status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving upload status"
        )


@router.get(
    "/results/{upload_id}",
    response_model=UploadResultsResponse,
    summary="Get Upload Results",
    description="Retrieve detailed results and file information for a completed upload.",
    responses={
        200: {"description": "Results retrieved successfully"},
        404: {"description": "Upload not found"},
        500: {"description": "Internal server error"}
    }
)
async def get_upload_results(upload_id: str) -> UploadResultsResponse:
    """
    Retrieve detailed results for a completed upload.
    
    Args:
        upload_id: Unique upload identifier
        
    Returns:
        UploadResultsResponse with detailed file information
        
    Raises:
        HTTPException: If upload not found or error occurs
    """
    logger.info(f"Retrieving results for upload: {upload_id}")
    
    try:
        # Check if upload exists
        if upload_id not in upload_tracking:
            raise HTTPException(
                status_code=404,
                detail=f"Upload not found: {upload_id}"
            )
        
        upload_info = upload_tracking[upload_id]
        
        # Convert file info to ProcessedFileInfo objects
        processed_files = [
            ProcessedFileInfo(**file_info) 
            for file_info in upload_info['files']
        ]
        
        # Create processing summary
        processing_summary = {
            'total_files': upload_info['total_files'],
            'completed_files': upload_info['completed_files'],
            'failed_files': upload_info['failed_files'],
            'processing_time': format_processing_time(
                (datetime.utcnow() - upload_info['created_at']).total_seconds()
            ),
            'status': upload_info['status'],
            'priority': upload_info['priority']
        }
        
        # Determine available actions based on status
        available_actions = []
        if upload_info['status'] == 'completed':
            available_actions = [
                "View extracted text",
                "Download processed files", 
                "View medical entities",
                "Access patient timeline"
            ]
        elif upload_info['status'] == 'failed':
            available_actions = [
                "Retry processing",
                "View error details",
                "Re-upload files"
            ]
        else:
            available_actions = [
                "Check processing status",
                "Cancel processing"
            ]
        
        response = UploadResultsResponse(
            upload_id=upload_id,
            message="Upload results retrieved successfully",
            files=processed_files,
            patient_id=upload_info['patient_context']['patient_id'],
            processing_summary=processing_summary,
            available_actions=available_actions,
            created_at=upload_info['created_at'],
            completed_at=upload_info.get('completed_at')
        )
        
        logger.info(f"Results retrieved for upload {upload_id}: {len(processed_files)} files")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving upload results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving upload results"
        )


@router.delete(
    "/cleanup/{upload_id}",
    summary="Cleanup Upload Data",
    description="Clean up temporary files and data for a completed upload.",
    responses={
        200: {"description": "Cleanup completed successfully"},
        404: {"description": "Upload not found"},
        500: {"description": "Internal server error"}
    }
)
async def cleanup_upload(upload_id: str) -> JSONResponse:
    """
    Clean up temporary files and data for a completed upload.
    
    Args:
        upload_id: Unique upload identifier
        
    Returns:
        JSONResponse confirming cleanup
        
    Raises:
        HTTPException: If upload not found or error occurs
    """
    logger.info(f"Cleaning up upload: {upload_id}")
    
    try:
        # Check if upload exists
        if upload_id not in upload_tracking:
            raise HTTPException(
                status_code=404,
                detail=f"Upload not found: {upload_id}"
            )
          # Remove from tracking
        upload_tracking.pop(upload_id)
        
        # Clean up temporary files (if any)
        cleanup_count = await file_handler.cleanup_temp_files()
        
        logger.info(f"Upload cleaned up: {upload_id} ({cleanup_count} temp files removed)")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Upload cleanup completed successfully",
                "upload_id": upload_id,
                "temp_files_cleaned": cleanup_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during upload cleanup: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error during upload cleanup"
        )


@router.get(
    "/health",
    summary="Check Ingestion Service Health",
    description="Health check endpoint for the ingestion service.",
    responses={
        200: {"description": "Service is healthy"},
        500: {"description": "Service is unhealthy"}
    }
)
async def health_check() -> JSONResponse:
    """
    Health check endpoint for the ingestion service.
    
    Returns:
        JSONResponse with service health status and configuration info    """
    try:
        storage_config = get_storage_config()
        
        # Check if storage directories are accessible
        storage_accessible = True
        try:
            # Check if main storage directories exist and are writable
            test_dirs = [
                storage_config.upload_dir,
                storage_config.temp_dir,
                storage_config.processed_dir
            ]
            
            for test_dir in test_dirs:
                if not test_dir.exists():
                    test_dir.mkdir(parents=True, exist_ok=True)
                
                # Test write permission
                test_file = test_dir / "health_check_test.tmp"
                test_file.write_text("test")
                test_file.unlink()
                
        except Exception as e:
            logger.warning(f"Storage accessibility check failed: {str(e)}")
            storage_accessible = False
        
        # Count temporary files
        temp_files_count = len(list(storage_config.temp_dir.glob('*'))) if storage_config.temp_dir.exists() else 0
        
        health_status = {
            "service": "ingestion",
            "status": "healthy" if storage_accessible else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "storage_accessible": storage_accessible,
            "active_uploads": len(upload_tracking),
            "temp_files": temp_files_count,
            "configuration": {
                "base_storage_dir": str(storage_config.base_storage_dir),
                "max_file_size": storage_config.max_file_size,
                "supported_extensions": storage_config.allowed_extensions,
                "patient_subdirs_enabled": storage_config.use_patient_subdirs,
                "auto_cleanup_enabled": storage_config.auto_cleanup_enabled,
                "file_retention_days": storage_config.file_retention_days
            },
            "supported_formats": list(file_handler.SUPPORTED_MIME_TYPES.keys())
        }
        
        status_code = 200 if storage_accessible else 500
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "service": "ingestion",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/info")
async def get_ingestion_info():
    """
    Get detailed information about the ingestion service capabilities and supported operations.
    
    Returns:
        JSONResponse: Service information, supported formats, and processing capabilities
    """
    return JSONResponse(
        content={
            "service_name": "MRIA Document Ingestion Service",
            "service_type": "file_processor",
            "version": "1.0.0",
            "status": "active",
            "capabilities": [
                "multi_format_upload",
                "file_validation",
                "metadata_extraction",
                "batch_processing",
                "progress_tracking",
                "error_handling",
                "storage_management",
                "virus_scanning"
            ],
            "supported_formats": [
                "application/pdf",
                "image/jpeg",
                "image/png",
                "image/tiff",
                "text/plain",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/csv",
                "application/json"
            ],
            "file_constraints": {
                "max_file_size": "50MB",
                "max_batch_size": "100 files",
                "supported_mime_types": 9,
                "virus_scan_enabled": True,
                "metadata_extraction": True
            },
            "processing_stages": [
                "upload_validation",
                "file_storage",
                "metadata_extraction",
                "virus_scanning",
                "format_conversion",
                "queue_for_processing",
                "status_tracking"
            ],
            "storage_locations": [
                "documents",
                "images",
                "processed",
                "temp",
                "backup"
            ],
            "integration_points": [
                "ocr_service",
                "supervisor_service",
                "vector_store",
                "file_system"
            ],
            "endpoints": [
                "/info",
                "/upload",
                "/status/{upload_id}",
                "/results/{upload_id}",
                "/batch-upload",
                "/health"
            ],
            "description": "Handles secure upload, validation, and initial processing of medical documents with comprehensive tracking and error handling"
        }
    )
