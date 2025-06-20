#!/usr/bin/env python3
"""
Minimal supervisor schemas to resolve import issue.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field,validator


class JobStatus(str, Enum):
    """Job execution status enumeration."""
    QUEUED = "queued"  
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY_PENDING = "retry_pending"



class WorkflowType(str, Enum):
    """Available workflow types."""
    OCR_ONLY = "ocr_only"
    OCR_TO_NER = "ocr_to_ner" 
    COMPLETE_PIPELINE = "complete_pipeline"
    DOCUMENT_TO_GRAPH = "document_to_graph"
    INSIGHT_GENERATION = "insight_generation"


class JobType(str, Enum):
    """Types of jobs."""
    DOCUMENT_PROCESSING = "document_processing"
    DATA_INGESTION = "data_ingestion"
    NER_ANALYSIS = "ner_analysis"
    CHUNKING = "chunking"
    GRAPH_UPDATE = "graph_update"
    INSIGHT_GENERATION = "insight_generation"
    BATCH_PROCESSING = "batch_processing"




class JobPriority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AgentType(str, Enum):
    """Types of agents."""
    
    SUPERVISOR = "supervisor"
    OCR = "ocr"
    NER = "ner"
    CHUNKING = "chunking"
    GRAPH = "graph"
    INSIGHT = "insight"
    CHAT = "chat"


class JobRequest(BaseModel):
    """Request model for creating a new job."""
    job_type: JobType = JobType.DOCUMENT_PROCESSING
    workflow_type: WorkflowType = WorkflowType.COMPLETE_PIPELINE
    patient_id: Optional[str] = None
    document_ids: List[str] = []
    file_paths: List[str] = []
    priority: JobPriority = JobPriority.NORMAL
    max_retries: int = 3
    timeout_seconds: int = 300
    processing_config: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    @validator('document_ids', 'file_paths')
    def validate_input_exists(cls, v, values):
        """Ensure at least one input source is provided."""
        if not any([v, values.get('document_ids', []), values.get('file_paths', [])]):
            raise ValueError("At least one document_id or file_path must be provided")
        return v




class JobResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: JobStatus
    job_type: str
    workflow_type: WorkflowType
    priority: JobPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_stage: Optional[str] = None
    errors: List[str] = []
    max_retries: int
    retry_count: int = 0
    assigned_agents: List[AgentType] = []
    processing_time_seconds: Optional[float] = None
    results: Dict[str, Any] = {}


class QueueStatus(BaseModel):
    """Status information about the job queue."""
    total_jobs: int
    queued_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    average_processing_time: float
    queue_health: str


class WorkflowDefinition(BaseModel):
    """Definition of a workflow."""
    workflow_id: str
    workflow_type: WorkflowType
    name: str
    description: str
    agent_sequence: List[AgentType]
    estimated_duration_seconds: int
    configuration: Dict[str, Any] = {}


class BatchJobRequest(BaseModel):
    """Request model for batch job processing."""
    
    batch_id: str = Field(default_factory=lambda: str(uuid4()), description="Batch identifier")
    jobs: List[JobRequest] = Field(..., min_items=1, description="List of jobs to process")
    
    # Batch configuration
    parallel_processing: bool = Field(
        default=True,
        description="Whether jobs can be processed in parallel"
    )
    max_concurrent_jobs: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent jobs"
    )
    
    # Error handling
    fail_on_first_error: bool = Field(
        default=False,
        description="Whether to stop batch on first job failure"
    )
    
    # Priority
    batch_priority: JobPriority = Field(
        default=JobPriority.NORMAL,
        description="Priority for the entire batch"
    )


class BatchJobResponse(BaseModel):
    """Response model for batch job requests."""
    
    batch_id: str = Field(..., description="Batch identifier")
    job_ids: List[str] = Field(..., description="Individual job identifiers")
    
    # Status
    batch_status: JobStatus = Field(..., description="Overall batch status")
    jobs_completed: int = Field(default=0, description="Number of completed jobs")
    jobs_failed: int = Field(default=0, description="Number of failed jobs")
    
    # Timestamps
    created_at: datetime = Field(..., description="Batch creation time")
    estimated_completion: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
