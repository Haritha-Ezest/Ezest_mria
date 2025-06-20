#!/usr/bin/env python3
"""
MRIA Supervisor Schema Definitions

This module defines the Pydantic models and enums used by the Medical Records Insight Agent (MRIA)
supervisor service for job orchestration, workflow management, and agent coordination.

The schemas in this module are used for:
1. Job request and response validation
2. API endpoint request/response models
3. Internal data structures for workflow management
4. Serialization/deserialization of job data
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from uuid import uuid4
from pydantic import BaseModel, Field, validator


class JobStatus(str, Enum):
    """
    Job execution status enumeration.
    
    Represents the current state of a job in the processing pipeline. Jobs progress through
    these states from QUEUED → RUNNING → COMPLETED (or FAILED). Jobs may also be CANCELLED
    by user request or moved to RETRY_PENDING if they encounter recoverable failures.
    """
    QUEUED = "queued"           # Job is waiting in queue to be processed
    RUNNING = "running"         # Job is currently being executed by one or more agents
    COMPLETED = "completed"     # Job has successfully finished all processing stages
    FAILED = "failed"           # Job encountered an unrecoverable error
    CANCELLED = "cancelled"     # Job was manually cancelled by user request
    RETRY_PENDING = "retry_pending"  # Job failed but will be automatically retried



class WorkflowType(str, Enum):
    """
    Available workflow types for medical document processing.
    
    Each workflow type represents a specific sequence of processing steps that will be applied
    to the input documents. Workflows range from simple OCR extraction to complete end-to-end
    processing with knowledge graph integration and insight generation.
    """
    OCR_ONLY = "ocr_only"                 # Text extraction only, no further processing
    OCR_TO_NER = "ocr_to_ner"             # Extract text and identify medical entities
    COMPLETE_PIPELINE = "complete_pipeline"  # Full end-to-end processing workflow
    DOCUMENT_TO_GRAPH = "document_to_graph"  # Process documents and create knowledge graph entries
    INSIGHT_GENERATION = "insight_generation"  # Generate medical insights from processed content


class JobType(str, Enum):
    """
    Types of jobs supported by the MRIA system.
    
    Each job type corresponds to a specific processing task category that can be executed
    independently or as part of a larger workflow. These types help the supervisor route
    tasks to the appropriate specialized agent.
    """
    DOCUMENT_PROCESSING = "document_processing"  # General document processing (may include multiple stages)
    DATA_INGESTION = "data_ingestion"      # Initial document intake and metadata extraction
    NER_ANALYSIS = "ner_analysis"          # Medical Named Entity Recognition processing
    CHUNKING = "chunking"                  # Document chunking and timeline creation
    GRAPH_UPDATE = "graph_update"          # Knowledge graph creation and relationship mapping
    INSIGHT_GENERATION = "insight_generation"  # Medical insight extraction and recommendation creation
    BATCH_PROCESSING = "batch_processing"  # Coordinated processing of multiple related documents




class JobPriority(str, Enum):
    """
    Job priority levels for queue management.
    
    Priority levels determine the order in which jobs are processed when multiple jobs
    are waiting in the queue. Higher priority jobs will be processed before lower priority
    jobs regardless of submission order.
    """
    LOW = "low"           # Background tasks, can be processed when system load is light
    NORMAL = "normal"     # Standard priority for most processing tasks
    HIGH = "high"         # Expedited processing for important medical documents
    URGENT = "urgent"     # Critical priority for emergency medical cases


class AgentType(str, Enum):
    """
    Types of specialized AI agents in the MRIA system.
    
    Each agent type represents a specialized component responsible for a specific aspect of
    medical document processing. Agents work together in orchestrated workflows to process
    documents and extract actionable insights.
    """
    SUPERVISOR = "supervisor"  # Orchestrates workflow and coordinates other agents
    OCR = "ocr"               # Optical Character Recognition for text extraction
    NER = "ner"               # Named Entity Recognition for medical entity extraction
    CHUNKING = "chunking"     # Document chunking and timeline creation
    GRAPH = "graph"           # Knowledge graph creation and relationship mapping
    INSIGHT = "insight"       # Medical insight extraction and recommendation generation
    CHAT = "chat"             # Natural language query interface for medical data


class JobRequest(BaseModel):
    """
    Request model for creating a new processing job.
    
    This model defines the structure and validation rules for job creation requests,
    including document sources, processing parameters, and metadata.
    """
    job_type: JobType = Field(
        default=JobType.DOCUMENT_PROCESSING,
        description="Type of processing job to execute"
    )
    workflow_type: WorkflowType = Field(
        default=WorkflowType.COMPLETE_PIPELINE,
        description="Processing workflow to apply to the documents"
    )
    patient_id: Optional[str] = Field(
        default=None,
        description="Optional patient identifier for medical context"
    )
    document_ids: List[str] = Field(
        default=[],
        description="List of document IDs already in the system to process"
    )
    file_paths: List[str] = Field(
        default=[],
        description="List of file paths to documents that need processing"
    )
    priority: JobPriority = Field(
        default=JobPriority.NORMAL,
        description="Priority level for job scheduling"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts if job fails"
    )
    timeout_seconds: int = Field(
        default=300,
        description="Maximum execution time in seconds before job times out"
    )
    processing_config: Dict[str, Any] = Field(
        default={},
        description="Additional configuration parameters for processing agents"
    )
    metadata: Dict[str, Any] = Field(
        default={},
        description="Optional metadata to associate with the job"
    )

    @validator('document_ids', 'file_paths')
    def validate_input_exists(cls, v, values):
        """Ensure at least one input source is provided."""
        if not any([v, values.get('document_ids', []), values.get('file_paths', [])]):
            raise ValueError("At least one document_id or file_path must be provided")
        return v




class JobResponse(BaseModel):
    """
    Response model for job status information.
    
    This model represents the current state of a job in the processing pipeline,
    including its progress, associated metadata, and any results or errors.
    Used for status monitoring and result retrieval.
    """
    job_id: str = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(..., description="Current execution status of the job")
    job_type: str = Field(..., description="Type of processing job")
    workflow_type: WorkflowType = Field(..., description="Workflow being applied to the documents")
    priority: JobPriority = Field(..., description="Priority level of the job")
    created_at: datetime = Field(..., description="Timestamp when job was created")
    started_at: Optional[datetime] = Field(None, description="Timestamp when job execution began")
    completed_at: Optional[datetime] = Field(None, description="Timestamp when job finished (successfully or not)")
    cancelled_at: Optional[datetime] = Field(None, description="Timestamp when job was cancelled (if applicable)")
    current_stage: Optional[str] = Field(None, description="Current processing stage if job is running")
    errors: List[str] = Field(default=[], description="List of error messages if job failed")
    max_retries: int = Field(..., description="Maximum number of retry attempts configured")
    retry_count: int = Field(default=0, description="Number of retry attempts made so far")
    assigned_agents: List[AgentType] = Field(default=[], description="Agents involved in processing this job")
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time in seconds (if completed)")
    results: Dict[str, Any] = Field(default={}, description="Results and output data from job execution")


class QueueStatus(BaseModel):
    """
    Status information about the job processing queue.
    
    This model provides system-wide metrics on the current state of the job queue,
    processing throughput, and overall health. Used for monitoring, diagnostics,
    and capacity planning.
    """
    total_jobs: int = Field(..., description="Total number of jobs in the system (all statuses)")
    queued_jobs: int = Field(..., description="Number of jobs waiting to be processed")
    running_jobs: int = Field(..., description="Number of jobs currently being processed")
    completed_jobs: int = Field(..., description="Number of jobs successfully completed")
    failed_jobs: int = Field(..., description="Number of jobs that failed (after retries)")
    average_processing_time: float = Field(..., description="Average processing time in seconds")
    queue_health: str = Field(
        ..., 
        description="Overall health status of the queue (healthy, warning, critical)"
    )


class WorkflowDefinition(BaseModel):
    """
    Definition of a processing workflow.
    
    This model represents a specific processing workflow configuration, including
    the sequence of agents involved, expected duration, and any specialized
    configuration parameters. Used for workflow orchestration and documentation.
    """
    workflow_id: str = Field(..., description="Unique identifier for the workflow")
    workflow_type: WorkflowType = Field(..., description="Type of workflow from predefined options")
    name: str = Field(..., description="Human-readable name of the workflow")
    description: str = Field(..., description="Detailed description of what the workflow does")
    agent_sequence: List[AgentType] = Field(
        ..., 
        description="Ordered sequence of agents that will process the documents"
    )
    estimated_duration_seconds: int = Field(
        ..., 
        description="Estimated processing time in seconds for typical documents"
    )
    configuration: Dict[str, Any] = Field(
        default={},
        description="Specific configuration parameters for this workflow"
    )


class BatchJobRequest(BaseModel):
    """
    Request model for batch job processing.
    
    This model defines the structure for requesting batch processing of multiple
    related documents. It allows for configuring concurrency, error handling,
    and priority settings for the entire batch operation.
    """
    
    batch_id: str = Field(
        default_factory=lambda: str(uuid4()), 
        description="Unique identifier for the batch operation"
    )
    jobs: List[JobRequest] = Field(
        ..., 
        min_items=1, 
        description="List of individual job requests to process as a batch"
    )
    
    # Batch configuration
    parallel_processing: bool = Field(
        default=True,
        description="Whether jobs can be processed in parallel or must be sequential"
    )
    max_concurrent_jobs: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of jobs to process simultaneously (when parallel)"
    )
    
    # Error handling
    fail_on_first_error: bool = Field(
        default=False,
        description="If True, the entire batch fails if any single job fails"
    )
    
    # Priority
    batch_priority: JobPriority = Field(
        default=JobPriority.NORMAL,
        description="Priority level applied to all jobs in this batch"
    )


class BatchJobResponse(BaseModel):
    """
    Response model for batch job requests.
    
    This model provides information about the status of a batch processing operation,
    including individual job IDs, completion statistics, and timing information.
    Used for batch status monitoring and result aggregation.
    """
    
    batch_id: str = Field(
        ..., 
        description="Unique identifier for the batch operation"
    )
    job_ids: List[str] = Field(
        ..., 
        description="List of individual job identifiers included in this batch"
    )
    
    # Status
    batch_status: JobStatus = Field(
        ..., 
        description="Overall status of the batch (derived from individual job statuses)"
    )
    jobs_completed: int = Field(
        default=0, 
        description="Number of successfully completed jobs in the batch"
    )
    jobs_failed: int = Field(
        default=0, 
        description="Number of failed jobs in the batch"
    )
    
    # Timestamps
    created_at: datetime = Field(
        ..., 
        description="Timestamp when the batch was created"
    )
    estimated_completion: Optional[datetime] = Field(
        None, 
        description="Estimated timestamp when all batch jobs will be completed"
    )
    
    class Config:
        """Pydantic configuration for serialization."""
        json_encoders = {
            datetime: lambda v: v.isoformat()  # Serialize datetime objects as ISO format strings
        }
