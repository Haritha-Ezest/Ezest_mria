"""
Graph schemas for medical knowledge graph operations.

This module defines Pydantic models for Neo4j graph database operations,
including nodes, relationships, and complex medical graph structures.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class NodeType(str, Enum):
    """Supported node types in the medical knowledge graph."""
    # Core entities as specified in requirements
    PATIENT = "Patient"
    VISIT = "Visit"  # MedicalEncounter
    CONDITION = "Condition"  # MedicalCondition
    MEDICATION = "Medication"  # Drug
    TEST = "Test"  # LabTest
    PROCEDURE = "Procedure"  # MedicalProcedure
    
    # Additional supporting entities
    SYMPTOM = "Symptom"
    PROVIDER = "Provider"
    FACILITY = "Facility"
    
    # Alternative labels for compatibility
    PERSON = "Person"  # Alias for Patient
    MEDICAL_ENCOUNTER = "MedicalEncounter"  # Alias for Visit
    MEDICAL_CONDITION = "MedicalCondition"  # Alias for Condition
    DRUG = "Drug"  # Alias for Medication
    LAB_TEST = "LabTest"  # Alias for Test
    MEDICAL_PROCEDURE = "MedicalProcedure"  # Alias for Procedure


class RelationshipType(str, Enum):
    """Supported relationship types in the medical knowledge graph."""
    HAS_VISIT = "HAS_VISIT"
    DIAGNOSED_WITH = "DIAGNOSED_WITH"
    PRESCRIBED = "PRESCRIBED"
    PERFORMED = "PERFORMED"
    UNDERWENT = "UNDERWENT"
    TREATED_WITH = "TREATED_WITH"
    INDICATES = "INDICATES"
    EXHIBITS_SYMPTOM = "EXHIBITS_SYMPTOM"
    CONTRAINDICATED_WITH = "CONTRAINDICATED_WITH"
    ALLERGIC_TO = "ALLERGIC_TO"
    RELATED_TO = "RELATED_TO"
    ADMINISTERED_BY = "ADMINISTERED_BY"
    TREATED_AT = "TREATED_AT"
    
    # Knowledge base and temporal relationships
    TEMPORAL_RELATIONSHIP = "TEMPORAL_RELATIONSHIP"
    KNOWLEDGE_RELATES_TO = "KNOWLEDGE_RELATES_TO"
    
    # Medical concept relationships
    SUPPORTS = "SUPPORTS"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    SUPERSEDES = "SUPERSEDES"
    DERIVED_FROM = "DERIVED_FROM"
    VALIDATES = "VALIDATES"
    RECOMMENDS = "RECOMMENDS"
    REQUIRES = "REQUIRES"


class GraphNode(BaseModel):
    """Base model for graph nodes."""
    id: str = Field(..., description="Unique identifier for the node")
    type: NodeType = Field(..., description="Type of the node")
    label: str = Field(..., description="Human-readable label")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class PatientNode(GraphNode):
    """Patient node schema (Person entity)."""
    type: NodeType = NodeType.PATIENT
    patient_id: str = Field(..., description="Unique patient identifier")
    name: Optional[str] = Field(None, description="Patient name")
    dob: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")
    gender: Optional[str] = Field(None, description="Patient gender (male/female/other)")
    mrn: Optional[str] = Field(None, description="Medical record number")
    
    # Additional patient attributes
    address: Optional[str] = Field(None, description="Patient address")
    phone: Optional[str] = Field(None, description="Patient phone number")
    email: Optional[str] = Field(None, description="Patient email")
    emergency_contact: Optional[str] = Field(None, description="Emergency contact information")
    insurance_info: Optional[Dict[str, Any]] = Field(None, description="Insurance information")


class VisitNode(GraphNode):
    """Medical visit/encounter node schema (MedicalEncounter entity)."""
    type: NodeType = NodeType.VISIT
    visit_id: str = Field(..., description="Visit identifier")
    date: str = Field(..., description="Visit date (YYYY-MM-DD)")
    visit_type: str = Field(..., description="Type of visit (routine/emergency/follow_up/consultation)")
    location: Optional[str] = Field(None, description="Visit location/facility")
    provider: Optional[str] = Field(None, description="Healthcare provider")
    
    # Additional visit attributes
    chief_complaint: Optional[str] = Field(None, description="Chief complaint")
    duration_minutes: Optional[int] = Field(None, description="Visit duration in minutes")
    visit_status: Optional[str] = Field(None, description="Visit status (scheduled/completed/cancelled)")
    notes: Optional[str] = Field(None, description="Visit notes")


class ConditionNode(GraphNode):
    """Medical condition node schema (MedicalCondition entity)."""
    type: NodeType = NodeType.CONDITION
    condition_name: str = Field(..., description="Condition name")
    icd_code: Optional[str] = Field(None, description="ICD-10 diagnostic code")
    severity: Optional[str] = Field(None, description="Condition severity (mild/moderate/severe)")
    status: Optional[str] = Field(None, description="Condition status (active/resolved/chronic)")
    onset_date: Optional[str] = Field(None, description="Condition onset date")
    
    # Additional condition attributes
    primary_diagnosis: Optional[bool] = Field(None, description="Is this a primary diagnosis")
    confidence_score: Optional[float] = Field(None, description="Diagnosis confidence score")
    differential_diagnoses: Optional[List[str]] = Field(None, description="Alternative diagnoses considered")
    diagnostic_criteria: Optional[str] = Field(None, description="Diagnostic criteria used")


class MedicationNode(GraphNode):
    """Medication node schema (Drug entity)."""
    type: NodeType = NodeType.MEDICATION
    medication_name: str = Field(..., description="Medication name")
    dosage: Optional[str] = Field(None, description="Medication dosage (e.g., 500mg)")
    frequency: Optional[str] = Field(None, description="Dosing frequency (e.g., twice daily, BID)")
    route: Optional[str] = Field(None, description="Route of administration (oral/IV/IM/topical)")
    rxnorm_code: Optional[str] = Field(None, description="RxNorm medication code")
    start_date: Optional[str] = Field(None, description="Medication start date")
    end_date: Optional[str] = Field(None, description="Medication end date")
    
    # Additional medication attributes
    strength: Optional[str] = Field(None, description="Medication strength")
    form: Optional[str] = Field(None, description="Medication form (tablet/liquid/injection)")
    generic_name: Optional[str] = Field(None, description="Generic medication name")
    brand_name: Optional[str] = Field(None, description="Brand medication name")
    indication: Optional[str] = Field(None, description="Indication for medication")
    contraindications: Optional[List[str]] = Field(None, description="Medication contraindications")
    side_effects: Optional[List[str]] = Field(None, description="Known side effects")


class TestNode(GraphNode):
    """Lab test/diagnostic test node schema (LabTest entity)."""
    type: NodeType = NodeType.TEST
    test_name: str = Field(..., description="Test name")
    value: Optional[Union[str, float, int]] = Field(None, description="Test value/result")
    unit: Optional[str] = Field(None, description="Test unit (mg/dL, %, mmHg)")
    reference_range: Optional[str] = Field(None, description="Normal reference range")
    status: Optional[str] = Field(None, description="Test status (ordered/completed/pending)")
    ordered_date: Optional[str] = Field(None, description="Test ordered date")
    result_date: Optional[str] = Field(None, description="Test result date")
    
    # Additional test attributes
    test_category: Optional[str] = Field(None, description="Test category (lab/imaging/diagnostic)")
    abnormal_flag: Optional[str] = Field(None, description="Abnormal result flag (high/low/critical)")
    interpretation: Optional[str] = Field(None, description="Test result interpretation")
    methodology: Optional[str] = Field(None, description="Test methodology used")
    ordering_provider: Optional[str] = Field(None, description="Provider who ordered the test")


class ProcedureNode(GraphNode):
    """Medical procedure node schema (MedicalProcedure entity)."""
    type: NodeType = NodeType.PROCEDURE
    procedure_name: str = Field(..., description="Procedure name")
    cpt_code: Optional[str] = Field(None, description="CPT procedure code")
    description: Optional[str] = Field(None, description="Procedure description")
    date: Optional[str] = Field(None, description="Procedure date")
    status: Optional[str] = Field(None, description="Procedure status (planned/completed/cancelled)")
    
    # Additional procedure attributes
    duration_minutes: Optional[int] = Field(None, description="Procedure duration in minutes")
    performing_provider: Optional[str] = Field(None, description="Provider performing procedure")
    location: Optional[str] = Field(None, description="Procedure location")
    indication: Optional[str] = Field(None, description="Indication for procedure")
    complications: Optional[List[str]] = Field(None, description="Procedure complications")
    outcomes: Optional[str] = Field(None, description="Procedure outcomes")


class GraphRelationship(BaseModel):
    """Base model for graph relationships."""
    id: str = Field(..., description="Unique relationship identifier")
    type: RelationshipType = Field(..., description="Relationship type")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship confidence")
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class PatientGraphRequest(BaseModel):
    """Request model for creating/updating patient graph."""
    patient_id: str = Field(..., description="Patient identifier")
    visit_data: Dict[str, Any] = Field(..., description="Visit data to process")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Entity relationships")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GraphQueryRequest(BaseModel):
    """Request model for Cypher queries."""
    query: str = Field(..., description="Cypher query string")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Result limit")
    
    @validator('query')
    def validate_query(cls, v):
        """Basic query validation."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        # Basic safety check - prevent destructive operations
        dangerous_keywords = ['DELETE', 'REMOVE', 'DROP', 'DETACH DELETE']
        query_upper = v.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise ValueError(f"Query contains dangerous keyword: {keyword}")
        return v


class GraphResponse(BaseModel):
    """Response model for graph operations."""
    message: str = Field(..., description="Operation result message")
    success: bool = Field(..., description="Operation success status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Response data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operation metadata")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)


class PatientGraphResponse(GraphResponse):
    """Response model for patient graph operations."""
    patient_id: str = Field(..., description="Patient identifier")
    nodes_created: int = Field(..., description="Number of nodes created")
    relationships_created: int = Field(..., description="Number of relationships created")
    graph_statistics: Dict[str, Any] = Field(default_factory=dict, description="Graph statistics")


class GraphQueryResponse(GraphResponse):
    """Response model for graph queries."""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Query results")
    columns: List[str] = Field(default_factory=list, description="Result columns")
    result_count: int = Field(..., description="Number of results")


class GraphStatistics(BaseModel):
    """Graph statistics model."""
    total_nodes: int = Field(..., description="Total number of nodes")
    total_relationships: int = Field(..., description="Total number of relationships")
    node_types: Dict[str, int] = Field(default_factory=dict, description="Count by node type")
    relationship_types: Dict[str, int] = Field(default_factory=dict, description="Count by relationship type")
    patients_count: int = Field(..., description="Number of patients")
    visits_count: int = Field(..., description="Number of visits")
    last_updated: datetime = Field(default_factory=datetime.now)


class GraphHealthCheck(BaseModel):
    """Graph database health check model."""
    status: str = Field(..., description="Health status")
    connection_status: str = Field(..., description="Database connection status")
    response_time: float = Field(..., description="Response time in seconds")
    database_info: Dict[str, Any] = Field(default_factory=dict, description="Database information")
    timestamp: datetime = Field(default_factory=datetime.now)


class GraphUpdateRequest(BaseModel):
    """Request model for graph updates."""
    operation: str = Field(..., description="Update operation type")
    data: Dict[str, Any] = Field(..., description="Update data")
    merge_strategy: str = Field("MERGE", description="Merge strategy for updates")
    
    @validator('operation')
    def validate_operation(cls, v):
        """Validate operation type."""
        allowed_operations = ['CREATE', 'UPDATE', 'MERGE', 'DELETE']
        if v.upper() not in allowed_operations:
            raise ValueError(f"Operation must be one of: {allowed_operations}")
        return v.upper()


class BatchGraphRequest(BaseModel):
    """Request model for batch graph operations."""
    operations: List[GraphUpdateRequest] = Field(..., description="List of operations")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch processing size")
    transaction_mode: bool = Field(default=True, description="Use transaction mode")


class GraphSchema(BaseModel):
    """Graph schema information."""
    node_types: List[str] = Field(..., description="Available node types")
    relationship_types: List[str] = Field(..., description="Available relationship types")
    node_properties: Dict[str, List[str]] = Field(default_factory=dict, description="Node properties by type")
    relationship_properties: Dict[str, List[str]] = Field(default_factory=dict, description="Relationship properties by type")
    constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Database constraints")
    indexes: List[Dict[str, Any]] = Field(default_factory=list, description="Database indexes")
