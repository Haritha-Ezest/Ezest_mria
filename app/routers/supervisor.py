"""
Supervisor router for orchestrating and managing document processing workflows.

This module provides REST API endpoints for supervising the complete document
processing pipeline, including job queuing, status monitoring, and workflow coordination.
"""

import os
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime

from app.schemas.supervisor import (
    JobRequest, JobResponse, JobStatus, QueueStatus, WorkflowType, WorkflowDefinition
)
from app.services.supervisor import get_supervisor, LangChainSupervisor
from app.common.utils import get_logger

# Configure logging
logger = get_logger(__name__)

# Create router instance
router = APIRouter(tags=["supervisor"])


@router.get("/info")
async def get_supervisor_info():
    """
    Get detailed information about the supervisor service capabilities and status.
    
    Returns:
        JSONResponse: Service information, capabilities, and current status
    """
    return JSONResponse(
        content={
            "service_name": "MRIA Supervisor Service",
            "service_type": "workflow_orchestrator",
            "version": "1.0.0",
            "status": "active",
            "capabilities": [
                "workflow_orchestration",
                "job_queue_management",
                "service_coordination",
                "status_monitoring",
                "error_handling",
                "batch_processing"
            ],
            "supported_workflows": [
                "document_ingestion_pipeline",
                "ocr_processing_chain",
                "ner_extraction_workflow",
                "chunking_and_analysis",
                "knowledge_graph_update"
            ],            "endpoints": [
                "/supervisor/info",
                "/supervisor/health",
                "/supervisor/debug/redis",
                "/supervisor/validate",
                "/supervisor/enqueue",
                "/supervisor/status/{job_id}",
                "/supervisor/job/{job_id}/metrics",
                "/supervisor/retry/{job_id}",
                "/supervisor/cancel/{job_id}",
                "/supervisor/queue/status",
                "/supervisor/workflows"
            ],
            "dependencies": ["ingestion", "ocr", "ner", "chunking", "graph"],
            "description": "Orchestrates and supervises the complete medical document processing pipeline using LangChain workflows"
        }
    )


@router.post("/enqueue", response_model=JobResponse)
async def enqueue_job(
    job_request: JobRequest,
    background_tasks: BackgroundTasks,
    supervisor: LangChainSupervisor = Depends(get_supervisor)
) -> JobResponse:
    """
    Enqueue a new job for processing in the medical document pipeline.
    
    This endpoint accepts job requests and adds them to the processing queue.
    The LangChain supervisor will orchestrate the workflow execution across
    the appropriate agents (OCR, NER, Chunking, Graph, etc.).
    
    Args:
        job_request: Job configuration and parameters
        background_tasks: FastAPI background tasks for async processing
        supervisor: Injected supervisor service instance
          Returns:
        JobResponse: Job status and tracking information
        
    Raises:
        HTTPException: If job enqueuing fails
    """
    try:
        # Validate job request
        if not job_request.document_ids and not job_request.file_paths:
            raise HTTPException(
                status_code=400,
                detail="At least one document_id or file_path must be provided"
            )
        
        # Additional validation for job request
        if job_request.max_retries < 0 or job_request.max_retries > 10:
            raise HTTPException(
                status_code=400,
                detail="max_retries must be between 0 and 10"
            )
            
        if job_request.timeout_seconds < 30 or job_request.timeout_seconds > 3600:
            raise HTTPException(
                status_code=400,
                detail="timeout_seconds must be between 30 and 3600"
            )
        
        # Validate workflow type compatibility
        if job_request.workflow_type not in WorkflowType:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid workflow_type: {job_request.workflow_type}"
            )
        
        # Validate file paths exist if provided
        if job_request.file_paths:
            for file_path in job_request.file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File path does not exist: {file_path}")
                    # Don't fail immediately, let the agents handle missing files
            
        # Enqueue the job
        job_response = await supervisor.enqueue_job(job_request)
        
        logger.info(f"Job {job_response.job_id} enqueued successfully with workflow {job_request.workflow_type}")
        
        # Add helpful message about processing time
        if job_response.processing_time_seconds is None:
            logger.debug(f"Job {job_response.job_id}: processing_time_seconds is null because job hasn't started yet")
        
        return job_response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to enqueue job: {str(e)}"
        )


@router.get("/status/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    supervisor: LangChainSupervisor = Depends(get_supervisor)
) -> JobResponse:
    """
    Get the current status and progress of a specific job.
    
    Args:
        job_id: Unique job identifier
        supervisor: Injected supervisor service instance
        
    Returns:
        JobResponse: Current job status, progress, and results
        
    Raises:
        HTTPException: If job not found or status retrieval fails
    """
    try:
        logger.info(f"Getting status for job {job_id}")
        logger.info(f"Supervisor instance: {type(supervisor).__name__}")
        logger.info(f"Redis client: {supervisor.redis_client}")
        
        job_status = await supervisor.get_job_status(job_id)
        
        if job_status is None:
            logger.error(f"Job {job_id} not found - supervisor returned None")
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
            
        logger.info(f"Successfully retrieved status for job {job_id}: {job_status.status}")
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve job status for {job_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve job status: {str(e)}"
        )


@router.get("/queue/status", response_model=QueueStatus)
async def get_queue_status(
    supervisor: LangChainSupervisor = Depends(get_supervisor)
) -> QueueStatus:
    """
    Get current status of the job processing queue.
    
    Returns information about queued jobs, processing metrics,
    system resource usage, and active workflows.
    
    Args:
        supervisor: Injected supervisor service instance
        
    Returns:
        QueueStatus: Queue metrics and system status
        
    Raises:
        HTTPException: If status retrieval fails
    """
    try:
        queue_status = await supervisor.get_queue_status()
        return queue_status
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve queue status: {str(e)}"
        )


@router.get("/workflows")
async def get_available_workflows():
    """
    Get list of available workflow types and their configurations.
    
    Returns:
        JSONResponse: Available workflows with descriptions and agent sequences
    """
    workflows = {
        WorkflowType.OCR_ONLY: {
            "name": "OCR Text Extraction",
            "description": "Extract text from documents using OCR",
            "agents": ["ocr"],
            "estimated_duration": "2-3 minutes",
            "use_case": "Simple text extraction from images or PDFs"
        },
        WorkflowType.OCR_TO_NER: {
            "name": "OCR and Medical Entity Recognition", 
            "description": "Extract text and identify medical entities",
            "agents": ["ocr", "ner"],
            "estimated_duration": "3-5 minutes",
            "use_case": "Text extraction with medical entity identification"
        },
        WorkflowType.COMPLETE_PIPELINE: {
            "name": "Complete Medical Document Processing",
            "description": "Full pipeline from OCR to knowledge graph",
            "agents": ["ocr", "ner", "chunking", "graph"],
            "estimated_duration": "5-8 minutes",
            "use_case": "Comprehensive medical document analysis and knowledge graph creation"
        },
        WorkflowType.DOCUMENT_TO_GRAPH: {
            "name": "Document to Knowledge Graph",
            "description": "Process documents and create knowledge graph entries",
            "agents": ["ocr", "ner", "chunking", "graph"],
            "estimated_duration": "6-10 minutes",
            "use_case": "Create structured knowledge representations from medical documents"
        },
        WorkflowType.BATCH_DOCUMENTS: {
            "name": "Batch Document Processing",
            "description": "Process multiple documents efficiently",
            "agents": ["ocr", "ner", "chunking", "graph"],
            "estimated_duration": "Variable based on batch size",
            "use_case": "Efficient processing of multiple medical documents"
        }
    }
    
    return JSONResponse(content={
        "available_workflows": workflows,
        "default_workflow": WorkflowType.COMPLETE_PIPELINE,
        "supported_document_types": [
            "PDF documents",
            "JPEG/PNG images", 
            "TIFF medical scans",
            "Handwritten prescriptions",
            "Lab reports",
            "Radiology reports",
            "Clinical notes"
        ]
    })


@router.post("/retry/{job_id}", response_model=JobResponse)
async def retry_job(
    job_id: str,
    supervisor: LangChainSupervisor = Depends(get_supervisor)
) -> JobResponse:
    """
    Retry a failed job with the same configuration.
    
    Args:
        job_id: Unique job identifier to retry
        supervisor: Injected supervisor service instance
        
    Returns:
        JobResponse: Status of the retried job
        
    Raises:
        HTTPException: If job not found, not retryable, or retry fails
    """
    try:
        # Get current job status
        job_status = await supervisor.get_job_status(job_id)
        
        if job_status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
            
        if job_status.status not in [JobStatus.FAILED, JobStatus.CANCELLED]:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not in a retryable state (current status: {job_status.status})"
            )
            
        if job_status.retry_count >= job_status.max_retries:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} has exceeded maximum retry attempts ({job_status.max_retries})"
            )
            
        # Retry the job through the supervisor
        retry_response = await supervisor.retry_job(job_id)
        
        return retry_response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retry job: {str(e)}"
        )


@router.delete("/cancel/{job_id}")
async def cancel_job(
    job_id: str,
    supervisor: LangChainSupervisor = Depends(get_supervisor)
):
    """
    Cancel a queued or running job.
    
    Args:
        job_id: Unique job identifier to cancel
        supervisor: Injected supervisor service instance
        
    Returns:
        JSONResponse: Cancellation status
        
    Raises:
        HTTPException: If job not found or cancellation fails
    """
    try:
        # Get current job status
        job_status = await supervisor.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
            
        if job_status.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} cannot be cancelled (current status: {job_status.status})"
            )
        
        try:
            # Cancel the job through the supervisor
            cancelled_job = await supervisor.cancel_job(job_id)
            
            return JSONResponse(content={
                "message": f"Job {job_id} has been cancelled",
                "job_id": cancelled_job.job_id,
                "status": cancelled_job.status.value,
                "cancelled_at": cancelled_job.cancelled_at.isoformat() if cancelled_job.cancelled_at else None,
                "cancellation_reason": "User requested cancellation"
            })
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to cancel job: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel job: {str(e)}"
        )


@router.post("/validate", response_model=dict)
async def validate_job_request(
    job_request: JobRequest
) -> dict:
    """
    Validate a job request without actually enqueueing it.
    
    This endpoint helps with debugging and testing job configurations.
    
    Args:
        job_request: Job configuration to validate
        
    Returns:
        dict: Validation results and warnings
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    try:
        # Basic validation
        if not job_request.document_ids and not job_request.file_paths:
            validation_results["errors"].append("At least one document_id or file_path must be provided")
            validation_results["valid"] = False
        
        # Parameter validation
        if job_request.max_retries < 0 or job_request.max_retries > 10:
            validation_results["errors"].append("max_retries must be between 0 and 10")
            validation_results["valid"] = False
            
        if job_request.timeout_seconds < 30 or job_request.timeout_seconds > 3600:
            validation_results["errors"].append("timeout_seconds must be between 30 and 3600")
            validation_results["valid"] = False
        
        # File path validation
        if job_request.file_paths:
            for file_path in job_request.file_paths:
                if not os.path.exists(file_path):
                    validation_results["warnings"].append(f"File path does not exist: {file_path}")
        
        # Workflow validation
        if job_request.workflow_type not in WorkflowType:
            validation_results["errors"].append(f"Invalid workflow_type: {job_request.workflow_type}")
            validation_results["valid"] = False
            
        validation_results["summary"] = f"Valid: {validation_results['valid']}, Warnings: {len(validation_results['warnings'])}, Errors: {len(validation_results['errors'])}"
        
        return validation_results
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation failed: {str(e)}"],
            "warnings": [],
            "summary": "Validation failed due to unexpected error"
        }


@router.get("/health")
async def health_check(
    supervisor: LangChainSupervisor = Depends(get_supervisor)
):
    """    Health check endpoint for the supervisor service.
    
    Returns:
        JSONResponse: Health status and system information
    """
    try:
        # Check supervisor status
        queue_status = await supervisor.get_queue_status()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service_name": "MRIA Supervisor Service",
            "version": "1.0.0",
            "queue_health": {
                "total_jobs": queue_status.total_jobs,
                "running_jobs": queue_status.running_jobs,
                "queued_jobs": queue_status.queued_jobs
            },
            "system_resources": {
                "cpu_usage_percent": getattr(queue_status, 'cpu_usage_percent', 0.0),
                "memory_usage_percent": getattr(queue_status, 'memory_usage_percent', 0.0)
            },
            "dependencies": {
                "redis": "connected",
                "agents": "available"
            }
        }
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            },
            status_code=503
        )


@router.get("/job/{job_id}/metrics")
async def get_job_metrics(
    job_id: str,
    supervisor: LangChainSupervisor = Depends(get_supervisor)
):
    """
    Get detailed metrics and timing information for a specific job.
    
    Args:
        job_id: Unique job identifier
        supervisor: Injected supervisor service instance
        
    Returns:
        JSONResponse: Detailed metrics including timing information
    """
    try:
        job_status = await supervisor.get_job_status(job_id)
        
        if job_status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        # Calculate various time metrics
        now = datetime.now()
        created_at = job_status.created_at
        started_at = job_status.started_at
        completed_at = job_status.completed_at
        
        # Calculate different time durations
        total_time_since_creation = (now - created_at).total_seconds()
        queue_time = (started_at - created_at).total_seconds() if started_at else None
        processing_time = job_status.processing_time_seconds
        
        # Estimate remaining time based on progress
        estimated_remaining_time = None
        if processing_time and job_status.progress > 0 and job_status.status == JobStatus.RUNNING:
            estimated_total_time = processing_time / (job_status.progress / 100)
            estimated_remaining_time = estimated_total_time - processing_time
        
        metrics = {
            "job_id": job_id,
            "status": job_status.status.value,
            "timing": {
                "created_at": created_at.isoformat(),
                "started_at": started_at.isoformat() if started_at else None,
                "completed_at": completed_at.isoformat() if completed_at else None,
                "total_time_since_creation_seconds": round(total_time_since_creation, 2),
                "queue_time_seconds": round(queue_time, 2) if queue_time else None,
                "processing_time_seconds": round(processing_time, 2) if processing_time else None,
                "estimated_remaining_time_seconds": round(estimated_remaining_time, 2) if estimated_remaining_time else None
            },
            "progress": {
                "percentage": job_status.progress,
                "current_stage": job_status.current_stage,
                "assigned_agents": job_status.assigned_agents,
                "completed_agents": int((job_status.progress / 100) * len(job_status.assigned_agents)) if job_status.progress > 0 else 0
            },
            "workflow": {
                "type": job_status.workflow_type.value,
                "total_agents": len(job_status.assigned_agents),
                "retry_count": job_status.retry_count,
                "max_retries": job_status.max_retries
            }
        }
        
        return JSONResponse(content=metrics)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve job metrics: {str(e)}"
        )


@router.get("/debug/redis")
async def debug_redis_status(
    supervisor: LangChainSupervisor = Depends(get_supervisor)
):
    """
    Debug endpoint to check Redis connection and job storage.
    """
    try:
        debug_info = {
            "redis_client_exists": supervisor.redis_client is not None,
            "redis_url": supervisor.redis_url
        }
        
        if supervisor.redis_client:
            # Test Redis connection
            try:
                ping_result = await supervisor.redis_client.ping()
                debug_info["redis_ping"] = ping_result
            except Exception as e:
                debug_info["redis_ping_error"] = str(e)
            
            # Get all job keys
            try:
                job_keys = await supervisor.redis_client.keys("job_status:*")
                debug_info["job_keys_found"] = len(job_keys)
                debug_info["job_keys"] = job_keys[:5]  # First 5 keys
            except Exception as e:
                debug_info["job_keys_error"] = str(e)
            
            # Get Redis info
            try:
                redis_info = await supervisor.redis_client.info()
                debug_info["redis_version"] = redis_info.get("redis_version")
                debug_info["connected_clients"] = redis_info.get("connected_clients")
                debug_info["db_keys"] = redis_info.get("db0", {}).get("keys", 0)
            except Exception as e:
                debug_info["redis_info_error"] = str(e)
        else:
            debug_info["error"] = "Redis client is None"
        
        return JSONResponse(content=debug_info)
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Debug failed: {str(e)}"},
            status_code=500
        )
