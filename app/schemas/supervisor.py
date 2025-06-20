#!/usr/bin/env python3
"""
Minimal supervisor schemas to resolve import issue.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job execution status enumeration."""
    QUEUED = "queued"  
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
    ANALYSIS = "analysis"


class JobPriority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AgentType(str, Enum):
    """Types of agents."""
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


# Test the classes
if __name__ == "__main__":
    print("Testing class creation...")
    req = JobRequest()
    print("JobRequest created successfully")
    print("Available classes:", [cls for cls in globals().keys() if not cls.startswith('_')])
