"""
Pydantic schemas for chunking and timeline structuring operations.

This module defines the data models used for request/response validation
in the chunking service, including medical timeline structures.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, validator


class ChunkingStrategy(str, Enum):
    """Available chunking strategies for medical documents."""
    VISIT_BASED = "visit_based"
    TOPIC_BASED = "topic_based"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"


class ChunkingConfig(BaseModel):
    """Configuration parameters for chunking operations."""
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    chunk_size: int = Field(default=1000, ge=100, le=4000, description="Target chunk size in characters")
    min_chunk_size: int = Field(default=100, ge=50, le=1000, description="Minimum chunk size in characters")
    overlap: int = Field(default=200, ge=0, description="Overlap between chunks in characters")
    preserve_medical_context: bool = Field(default=True, description="Keep medical context intact")
    include_metadata: bool = Field(default=True, description="Include chunk metadata")
    semantic_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Semantic similarity threshold")


class MedicalEntity(BaseModel):
    """Represents a medical entity found in text."""
    text: str = Field(description="Entity text")
    label: str = Field(description="Entity label/type")
    start: int = Field(description="Start position in text")
    end: int = Field(description="End position in text")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    normalized_form: Optional[str] = Field(default=None, description="Normalized medical term")


class MedicalChunk(BaseModel):
    """Represents a chunk of medical text with metadata."""
    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Chunk text content")
    start_index: int = Field(description="Start position in original document") 
    end_index: int = Field(description="End position in original document")
    word_count: int = Field(description="Number of words in chunk")
    char_count: int = Field(description="Number of characters in chunk")
    semantic_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Semantic coherence score")
    medical_entities: List[MedicalEntity] = Field(default_factory=list, description="Medical entities in chunk")
    chunk_type: str = Field(default="content", description="Type of chunk (header, content, conclusion)")
    visit_date: Optional[datetime] = Field(default=None, description="Associated visit date if detected")
    medical_topics: List[str] = Field(default_factory=list, description="Medical topics covered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TimelineEntry(BaseModel):
    """Represents a single entry in a patient's medical timeline."""
    date: datetime = Field(description="Date of medical event")
    visit_id: Optional[str] = Field(default=None, description="Visit identifier")
    symptoms: List[str] = Field(default_factory=list, description="Reported symptoms")
    tests: List[str] = Field(default_factory=list, description="Tests performed")
    medications: List[str] = Field(default_factory=list, description="Medications prescribed/mentioned")
    diagnoses: List[str] = Field(default_factory=list, description="Diagnoses made")
    procedures: List[str] = Field(default_factory=list, description="Procedures performed")
    progress_notes: List[str] = Field(default_factory=list, description="Progress notes")
    chunk_ids: List[str] = Field(default_factory=list, description="Associated chunk IDs")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Timeline entry confidence")


class PatientTimeline(BaseModel):
    """Complete medical timeline for a patient."""
    patient_id: str = Field(description="Patient identifier")
    timeline_entries: List[TimelineEntry] = Field(description="Chronological timeline entries")
    total_visits: int = Field(description="Total number of visits")
    date_range: Dict[str, datetime] = Field(description="Timeline date range")
    summary: Optional[str] = Field(default=None, description="Timeline summary")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Timeline metadata")

    @validator('timeline_entries')
    def sort_timeline_entries(cls, entries):
        """Sort timeline entries by date."""
        return sorted(entries, key=lambda x: x.date)
    def sort_timeline_entries(cls, entries):
        """Sort timeline entries by date."""
        return sorted(entries, key=lambda x: x.date)


class ChunkRequest(BaseModel):
    """Request model for chunking operations."""
    text: str = Field(description="Text to be chunked")
    patient_id: Optional[str] = Field(default=None, description="Patient identifier for timeline creation")
    document_type: Optional[str] = Field(default="medical_record", description="Type of medical document")
    config: ChunkingConfig = Field(default_factory=ChunkingConfig, description="Chunking configuration")
    create_timeline: bool = Field(default=False, description="Whether to create patient timeline")
    job_id: Optional[str] = Field(default=None, description="Associated job ID for tracking")


class ChunkResponse(BaseModel):
    """Response model for chunking operations."""
    job_id: Optional[str] = Field(default=None, description="Job ID for tracking")
    chunks: List[MedicalChunk] = Field(description="Generated chunks")
    total_chunks: int = Field(description="Total number of chunks")
    chunking_strategy: ChunkingStrategy = Field(description="Strategy used for chunking")
    processing_time: float = Field(description="Processing time in seconds")
    average_chunk_size: float = Field(description="Average chunk size in characters")
    medical_entities_found: int = Field(default=0, description="Total medical entities found")
    timeline_created: bool = Field(default=False, description="Whether timeline was created")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality assessment metrics")


class TimelineRequest(BaseModel):
    """Request model for timeline creation."""
    patient_id: str = Field(description="Patient identifier")
    chunks: List[MedicalChunk] = Field(description="Medical chunks to process")
    include_summary: bool = Field(default=True, description="Include timeline summary")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for timeline entries")


class TimelineResponse(BaseModel):
    """Response model for timeline operations."""
    patient_timeline: PatientTimeline = Field(description="Generated patient timeline")
    processing_time: float = Field(description="Processing time in seconds")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall timeline confidence")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")


class StructureRequest(BaseModel):
    """Request model for medical timeline structuring."""
    patient_id: str = Field(description="Patient identifier")
    raw_text: str = Field(description="Raw medical text to structure")
    existing_timeline: Optional[PatientTimeline] = Field(default=None, description="Existing timeline to update")
    merge_strategy: str = Field(default="chronological", description="Strategy for merging with existing timeline")


class StructureResponse(BaseModel):
    """Response model for medical timeline structuring."""
    structured_timeline: PatientTimeline = Field(description="Structured medical timeline")
    chunks_created: List[MedicalChunk] = Field(description="Chunks created during structuring")
    processing_summary: Dict[str, Any] = Field(description="Processing summary and statistics")
    quality_assessment: Dict[str, float] = Field(description="Quality assessment metrics")
