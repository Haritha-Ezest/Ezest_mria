"""
NER (Named Entity Recognition) Pydantic schemas.

This module defines the data models for NER API requests and responses,
including medical entity types, confidence scores, and processing options.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class EntityType(str, Enum):
    """Supported medical entity types for extraction."""
    # Basic entities
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    TIME = "TIME"

    # Medical entities
    MEDICATION = "MEDICATION"
    DISEASE = "DISEASE"
    CONDITION = "CONDITION"
    SYMPTOM = "SYMPTOM"
    PROCEDURE = "PROCEDURE"
    DOSAGE = "DOSAGE"
    FREQUENCY = "FREQUENCY"
    ANATOMICAL_STRUCTURE = "ANATOMICAL_STRUCTURE"
    LAB_VALUE = "LAB_VALUE"
    LAB_TEST = "LAB_TEST"
    MEDICAL_DEVICE = "MEDICAL_DEVICE"
    VITAL_SIGN = "VITAL_SIGN"
    ALLERGY = "ALLERGY"
    FAMILY_HISTORY = "FAMILY_HISTORY"
    SOCIAL_HISTORY = "SOCIAL_HISTORY"

    # Clinical specialties
    RADIOLOGY_FINDING = "RADIOLOGY_FINDING"
    PATHOLOGY_FINDING = "PATHOLOGY_FINDING"
    SURGICAL_PROCEDURE = "SURGICAL_PROCEDURE"
    DIAGNOSTIC_PROCEDURE = "DIAGNOSTIC_PROCEDURE"

    # Enhanced temporal entities
    TEMPORAL_DURATION = "TEMPORAL_DURATION"
    TEMPORAL_FREQUENCY = "TEMPORAL_FREQUENCY"
    TEMPORAL_TIME_OF_DAY = "TEMPORAL_TIME_OF_DAY"
    TEMPORAL_RELATIVE = "TEMPORAL_RELATIVE"

    # Enhanced medical entities
    ROUTE_OF_ADMINISTRATION = "ROUTE_OF_ADMINISTRATION"
    DRUG_FORM = "DRUG_FORM"
    SEVERITY = "SEVERITY"
    LATERALITY = "LATERALITY"  # left/right/bilateral
    NEGATION = "NEGATION"  # absent/denied/negative
    CERTAINTY = "CERTAINTY"  # possible/probable/definite


class ProcessingMode(str, Enum):
    """NER processing modes with different accuracy/speed tradeoffs."""
    FAST = "fast"           # Quick processing, lower accuracy
    STANDARD = "standard"   # Balanced processing
    ACCURATE = "accurate"   # Slower processing, higher accuracy
    MEDICAL = "medical"     # Medical-domain optimized


class LanguageCode(str, Enum):
    """Supported language codes for multilingual NER."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"


class MedicalEntity(BaseModel):
    """Individual medical entity with metadata."""
    text: str = Field(..., description="The extracted entity text")
    label: EntityType = Field(..., description="Entity type classification")
    start: int = Field(..., description="Start character position in text")
    end: int = Field(..., description="End character position in text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")

    # Advanced features
    normalized_text: Optional[str] = Field(None, description="Normalized/standardized entity text")
    entity_id: Optional[str] = Field(None, description="Linked entity identifier (UMLS, ICD-10, etc.)")
    concept_code: Optional[str] = Field(None, description="Medical concept code")
    semantic_type: Optional[str] = Field(None, description="Semantic type classification")
    context: Optional[str] = Field(None, description="Surrounding text context")
    
    # Lab value specific fields
    value: Optional[str] = Field(None, description="Extracted numeric value for lab results")
    reference_range: Optional[str] = Field(None, description="Reference range for lab values")

    # Relationships
    related_entities: Optional[List[str]] = Field(default_factory=list, description="Related entity IDs")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class EntityGroup(BaseModel):
    """Grouped entities by type with aggregated statistics."""
    entity_type: EntityType = Field(..., description="Entity type")
    entities: List[MedicalEntity] = Field(..., description="List of entities of this type")
    count: int = Field(..., description="Number of entities found")
    avg_confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence score")
    unique_values: int = Field(..., description="Number of unique entity values")


class NERRequest(BaseModel):
    """Request model for medical entity extraction."""
    text: str = Field(..., description="Text content to process for entity extraction")
    document_id: Optional[str] = Field(None, description="Optional document identifier for tracking")

    # Processing options
    processing_mode: ProcessingMode = Field(ProcessingMode.STANDARD, description="Processing mode")
    language: LanguageCode = Field(LanguageCode.ENGLISH, description="Text language")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")

    # Entity filtering
    entity_types: Optional[List[EntityType]] = Field(None, description="Specific entity types to extract")
    exclude_entity_types: Optional[List[EntityType]] = Field(None, description="Entity types to exclude")

    # Advanced options
    include_context: bool = Field(False, description="Include surrounding text context")
    enable_entity_linking: bool = Field(True, description="Enable medical concept linking")
    enable_normalization: bool = Field(True, description="Enable text normalization")
    max_entities: Optional[int] = Field(None, gt=0, description="Maximum number of entities to return")

    # Medical-specific options
    medical_specialty: Optional[str] = Field(None, description="Medical specialty context for optimization")
    patient_context: Optional[Dict[str, Any]] = Field(None, description="Patient context for better extraction")

    @validator('text')
    def validate_text_length(cls, v):
        """Validate text length for processing limits."""
        if len(v.strip()) == 0:
            raise ValueError("Text content cannot be empty")
        if len(v) > 1000000:  # 1MB text limit
            raise ValueError("Text content exceeds maximum length limit")
        return v.strip()


class ProcessingMetrics(BaseModel):
    """Processing performance and quality metrics."""
    processing_time: float = Field(..., description="Total processing time in seconds")
    text_length: int = Field(..., description="Input text length in characters")
    tokens_processed: int = Field(..., description="Number of tokens processed")
    entities_found: int = Field(..., description="Total number of entities found")
    avg_confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence score")
    model_version: str = Field(..., description="NER model version used")
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")


class NERResponse(BaseModel):
    """Response model for medical entity extraction results."""
    message: str = Field(..., description="Processing status message")
    document_id: Optional[str] = Field(None, description="Document identifier if provided")

    # Entity results
    entities: List[MedicalEntity] = Field(..., description="Extracted medical entities")
    entity_groups: List[EntityGroup] = Field(..., description="Entities grouped by type")
    total_entities: int = Field(..., description="Total number of entities extracted")

    # Confidence and quality metrics
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores by entity type")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall extraction confidence")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Text quality assessment score")

    # Processing information
    processing_metrics: ProcessingMetrics = Field(..., description="Processing performance metrics")
    processing_mode: ProcessingMode = Field(..., description="Processing mode used")
    language_detected: LanguageCode = Field(..., description="Detected or specified language")

    # Next steps
    next_step: str = Field(..., description="Suggested next processing step")
    recommendations: List[str] = Field(default_factory=list, description="Processing recommendations")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class BatchNERRequest(BaseModel):
    """Request model for batch entity extraction from multiple texts."""
    texts: List[str] = Field(..., description="List of text contents to process")
    document_ids: Optional[List[str]] = Field(None, description="Optional document identifiers")

    # Shared processing options
    processing_mode: ProcessingMode = Field(ProcessingMode.STANDARD, description="Processing mode")
    language: LanguageCode = Field(LanguageCode.ENGLISH, description="Text language")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")

    # Batch options
    batch_size: int = Field(10, gt=0, le=100, description="Batch processing size")
    parallel_processing: bool = Field(True, description="Enable parallel processing")

    @validator('texts')
    def validate_texts(cls, v):
        """Validate batch texts."""
        if len(v) == 0:
            raise ValueError("At least one text must be provided")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 texts allowed per batch")
        return v


class BatchNERResponse(BaseModel):
    """Response model for batch entity extraction results."""
    message: str = Field(..., description="Batch processing status message")
    total_documents: int = Field(..., description="Total number of documents processed")
    successful_extractions: int = Field(..., description="Number of successful extractions")
    failed_extractions: int = Field(..., description="Number of failed extractions")

    # Results
    results: List[NERResponse] = Field(..., description="Individual extraction results")

    # Batch metrics
    batch_processing_time: float = Field(..., description="Total batch processing time")
    avg_processing_time: float = Field(..., description="Average processing time per document")
    batch_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall batch confidence")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class EntityValidationRequest(BaseModel):
    """Request model for validating extracted entities."""
    document_id: str = Field(..., description="Document identifier")
    entities: List[MedicalEntity] = Field(..., description="Entities to validate")
    validation_rules: Optional[Dict[str, Any]] = Field(None, description="Custom validation rules")


class EntityValidationResponse(BaseModel):
    """Response model for entity validation results."""
    message: str = Field(..., description="Validation status message")
    document_id: str = Field(..., description="Document identifier")

    # Validation results
    valid_entities: List[MedicalEntity] = Field(..., description="Valid entities")
    invalid_entities: List[MedicalEntity] = Field(..., description="Invalid entities")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")

    # Statistics
    validation_score: float = Field(..., ge=0.0, le=1.0, description="Overall validation score")
    total_validated: int = Field(..., description="Total entities validated")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
