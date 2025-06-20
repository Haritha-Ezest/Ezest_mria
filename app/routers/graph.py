"""
Graph router for knowledge graph operations and relationship management.

This module provides REST API endpoints for building, updating, and querying
knowledge graphs from processed medical documents.
"""

import traceback
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse

from app.schemas.graph import (
    PatientGraphRequest, PatientGraphResponse, GraphQueryRequest, 
    GraphQueryResponse, GraphStatistics, GraphHealthCheck,
    BatchGraphRequest, GraphSchema, GraphResponse
)
from app.services.graph_client import Neo4jGraphClient, get_graph_client_from_config
from app.common.utils import get_logger

# Configure logging
logger = get_logger(__name__)

# Create router instance
router = APIRouter(tags=["graph"])



async def get_graph_client() -> Neo4jGraphClient:
    """Dependency to get Neo4j graph client with robust connection handling."""
    from app.config import get_settings
    from dotenv import load_dotenv
    import os
    
    # Ensure latest environment variables are loaded
    load_dotenv()
    
    # Get settings
    settings = get_settings()
    
    # Create a fresh client instance for each request to avoid connection issues
    neo4j_client = Neo4jGraphClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database
    )
    
    # Connect with explicit error handling
    try:
        await neo4j_client.connect()
        logger.info(f"Successfully connected to Neo4j for request")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j in dependency: {e}")
        # Try to clean up if connection failed
        if neo4j_client.driver:
            try:
                await neo4j_client.driver.close()
            except:
                pass
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {str(e)}"
        )
    
    return neo4j_client
@router.get("/info", response_model=Dict[str, Any])
async def get_graph_info():
    """
    Get detailed information about the enhanced knowledge graph service capabilities and schema.
    
    This endpoint provides comprehensive information about the graph service,
    including supported entities, relationship types, and advanced analytical operations.
    
    Returns:
        JSONResponse: Service information, graph schema, and operational details
    """
    return JSONResponse(
        content={
            "service_name": "Enhanced MRIA Knowledge Graph Service",
            "service_type": "medical_knowledge_graph_database",
            "version": "2.0.0",
            "status": "active",
            "description": "Advanced medical knowledge graph builder with temporal analysis and cross-patient insights",
            "capabilities": [
                "medical_entity_relationship_mapping",
                "knowledge_graph_construction", 
                "semantic_querying",
                "graph_traversal",
                "relationship_inference",
                "medical_ontology_integration",
                "temporal_relationship_analysis",
                "patient_timeline_construction",
                "cross_patient_pattern_analysis",
                "medical_knowledge_base_expansion",
                "drug_interaction_detection",
                "care_gap_analysis",
                "medical_concept_linking",
                "clinical_decision_support"
            ],
            "supported_entities": [
                # Core medical entities as per requirements
                "patient",  # Person
                "visit",    # MedicalEncounter  
                "condition", # MedicalCondition
                "medication", # Drug
                "test",      # LabTest
                "procedure", # MedicalProcedure
                
                # Supporting entities
                "symptom",
                "provider",
                "facility",
                "medical_concept"
            ],
            "relationship_types": [
                # Core relationships as per requirements
                "HAS_VISIT",
                "DIAGNOSED_WITH", 
                "PRESCRIBED",
                "PERFORMED",
                "UNDERWENT",
                "TREATED_WITH",
                "INDICATES",
                
                # Advanced relationship types
                "EXHIBITS_SYMPTOM",
                "CONTRAINDICATED_WITH",
                "ALLERGIC_TO",
                "RELATED_TO",
                "ADMINISTERED_BY",
                "TREATED_AT",
                "TEMPORAL_RELATIONSHIP",
                "KNOWLEDGE_RELATES_TO",
                "INTERACTS_WITH"
            ],
            "graph_statistics": {
                "total_nodes": 0,
                "total_relationships": 0,
                "entity_types": 10,
                "relationship_types": 16,
                "knowledge_base_concepts": 0
            },
            "advanced_capabilities": [
                "temporal_analysis",
                "patient_timeline_analysis", 
                "cross_patient_pattern_mining",
                "medical_knowledge_expansion",
                "drug_interaction_checking",
                "care_gap_identification",
                "clinical_insight_generation",
                "medical_concept_normalization",
                "entity_linking_to_knowledge_bases"
            ],
            "query_capabilities": [
                "cypher_queries",
                "pattern_matching",
                "path_finding",
                "subgraph_extraction",
                "similarity_search",
                "temporal_queries",
                "aggregation_queries",
                "cross_patient_analytics",
                "longitudinal_analysis"
            ],
            "knowledge_base_integration": [
                "UMLS",
                "SNOMED_CT", 
                "ICD_10",
                "RxNorm",
                "CPT",
                "medical_ontologies"
            ],
            "endpoints": [
                "/graph/info",
                "/graph/health", 
                "/graph/create",
                "/graph/update/{patient_id}",
                "/graph/patient/{patient_id}",
                "/graph/query",
                "/graph/schema",
                "/graph/statistics",
                "/graph/batch",
                "/graph/temporal",
                "/graph/timeline/{patient_id}",
                "/graph/patterns/{condition}",
                "/graph/knowledge/expand",
                "/graph/insights/{patient_id}"
            ],
            "graph_schema": {
                "core_entities": {
                    "Patient": "Person entity with demographics and identifiers",
                    "Visit": "MedicalEncounter entity for healthcare visits",
                    "Condition": "MedicalCondition entity for diagnoses",
                    "Medication": "Drug entity for medications",
                    "Test": "LabTest entity for laboratory and diagnostic tests",
                    "Procedure": "MedicalProcedure entity for medical procedures"
                },
                "core_relationships": {
                    "Patient-HAS_VISIT->Visit": "Patient has medical visits",
                    "Visit-DIAGNOSED_WITH->Condition": "Visit resulted in diagnosis",
                    "Visit-PRESCRIBED->Medication": "Visit resulted in medication prescription",
                    "Visit-PERFORMED->Test": "Visit included test performance",
                    "Visit-UNDERWENT->Procedure": "Visit included procedure",
                    "Condition-TREATED_WITH->Medication": "Condition treated with medication",
                    "Test-INDICATES->Condition": "Test result indicates condition"
                }
            }
        }
    )


@router.get("/health", response_model=GraphHealthCheck)
async def health_check(client: Neo4jGraphClient = Depends(get_graph_client)):
    """
    Perform health check on the graph database service.
    
    Returns:
        GraphHealthCheck: Health status and database connectivity information
    """
    try:
        logger.info("Starting graph health check...")
        health_status = await client.health_check()
        logger.info(f"Health check completed: {health_status.status}")
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Graph service health check failed: {str(e)}"
        )


@router.post("/create", response_model=PatientGraphResponse)
async def create_patient_graph(
    request: PatientGraphRequest,
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Create new patient graph from medical data.
    
    This endpoint processes medical entities and relationships to build
    a comprehensive knowledge graph for a patient's medical history.
    
    Args:
        request: Patient graph creation request with medical data
        
    Returns:
        PatientGraphResponse: Creation results with statistics
    """
    try:
        logger.info(f"Creating patient graph for patient: {request.patient_id}")
        
        response = await client.create_patient_graph(request)
        
        if response.success:
            logger.info(
                f"Patient graph created successfully for {request.patient_id}: "
                f"{response.nodes_created} nodes, {response.relationships_created} relationships"
            )
        else:
            logger.error(f"Failed to create patient graph: {response.message}")
            
        return response
        
    except Exception as e:
        logger.error(f"Error creating patient graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create patient graph: {str(e)}"
        )


@router.put("/update/{patient_id}", response_model=PatientGraphResponse)
async def update_patient_graph(
    patient_id: str,
    request: PatientGraphRequest,
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Update existing patient graph with new medical data.
    
    Args:
        patient_id: Patient identifier
        request: Updated medical data
        
    Returns:
        PatientGraphResponse: Update results with statistics
    """
    try:
        # Ensure patient_id matches request
        request.patient_id = patient_id
        
        logger.info(f"Updating patient graph for patient: {patient_id}")
        
        response = await client.create_patient_graph(request)  # Uses MERGE operations
        
        if response.success:
            logger.info(f"Patient graph updated successfully for {patient_id}")
        else:
            logger.error(f"Failed to update patient graph: {response.message}")
            
        return response
        
    except Exception as e:
        logger.error(f"Error updating patient graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update patient graph: {str(e)}"
        )


@router.get("/patient/{patient_id}", response_model=Dict[str, Any])
async def get_patient_graph(
    patient_id: str,
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Retrieve complete patient graph data.
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        Dict: Patient graph data with summary statistics
    """
    try:
        logger.info(f"Retrieving patient graph for: {patient_id}")
        
        # Ensure client is properly connected
        if not client.is_connected():
            await client.connect()
        
        # Get the patient graph data
        graph_data = await client.get_patient_graph(patient_id)
        
        # Check if patient was found
        if isinstance(graph_data, dict) and "error" in graph_data:
            logger.warning(f"Patient not found: {patient_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient graph not found: {patient_id}"
            )
        
        return {
            "message": "Patient graph data retrieved successfully",
            "patient_id": patient_id,
            "graph_data": graph_data,
            "last_updated": datetime.now().isoformat(),
            "graph_health": "complete"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving patient graph for {patient_id}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Ensure client connection is cleaned up on error
        try:
            if client and client.driver:
                await client.disconnect()
        except:
            pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve patient graph: {str(e)}"
        )


@router.post("/query", response_model=GraphQueryResponse)
async def execute_cypher_query(
    request: GraphQueryRequest,
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Execute Cypher query against the knowledge graph.
    
    Args:
        request: Cypher query request with parameters
        
    Returns:
        GraphQueryResponse: Query results with metadata
    """
    try:
        logger.info(f"Executing Cypher query: {request.query[:100]}...")
        
        # Validate query is not empty
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        # Execute the query with proper error handling and cleanup
        response = await client.execute_query(request)
        
        if response.success:
            logger.info(f"Query executed successfully, {response.result_count} results")
        else:
            logger.error(f"Query execution failed: {response.message}")
            
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        
        # Return a structured error response instead of raising HTTPException
        return GraphQueryResponse(
            message=f"Query execution failed: {str(e)}",
            success=False,
            data={"query": request.query, "parameters": request.parameters or {}},
            metadata={
                "error_type": type(e).__name__,
                "query_type": "cypher"
            },
            processing_time=0.0,
            timestamp=datetime.utcnow().isoformat(),
            results=[],
            columns=[],
            result_count=0
        )
    finally:
        # Ensure proper cleanup of the client connection
        try:
            if client and client.driver:
                await client.disconnect()
        except Exception as cleanup_error:
            logger.warning(f"Error during client cleanup: {cleanup_error}")


@router.get("/statistics", response_model=GraphStatistics)
async def get_graph_statistics(
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Get comprehensive graph database statistics.
    
    Returns:
        GraphStatistics: Detailed statistics about the knowledge graph
    """
    try:
        logger.info("Retrieving graph statistics")
        
        statistics = await client.get_graph_statistics()
        
        logger.info(
            f"Graph statistics: {statistics.total_nodes} nodes, "
            f"{statistics.total_relationships} relationships"
        )
        
        return statistics
        
    except Exception as e:
        logger.error(f"Error retrieving graph statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve graph statistics: {str(e)}"
        )


@router.get("/schema", response_model=GraphSchema)
async def get_graph_schema():
    """
    Get enhanced graph schema information including node types, relationships, and constraints.
    
    Returns:
        GraphSchema: Complete schema information for the medical knowledge graph
    """
    try:
        # Return enhanced schema based on medical domain requirements
        schema = GraphSchema(
            node_types=[
                "Patient", "Person",  # Core patient entity
                "Visit", "MedicalEncounter",  # Medical visits/encounters
                "Condition", "MedicalCondition",  # Medical conditions/diagnoses
                "Medication", "Drug",  # Medications and drugs
                "Test", "LabTest",  # Laboratory tests and diagnostic tests
                "Procedure", "MedicalProcedure",  # Medical procedures
                "Symptom", "Provider", "Facility",  # Supporting entities
                "MedicalConcept"  # Knowledge base concepts
            ],
            relationship_types=[
                "HAS_VISIT", "DIAGNOSED_WITH", "PRESCRIBED", "PERFORMED",
                "UNDERWENT", "TREATED_WITH", "INDICATES", "EXHIBITS_SYMPTOM",
                "CONTRAINDICATED_WITH", "ALLERGIC_TO", "RELATED_TO",
                "ADMINISTERED_BY", "TREATED_AT", "TEMPORAL_RELATIONSHIP",
                "KNOWLEDGE_RELATES_TO", "INTERACTS_WITH"
            ],
            node_properties={
                "Patient": ["id", "name", "dob", "gender", "mrn", "address", "phone", "email"],
                "Visit": ["id", "date", "visit_type", "location", "provider", "chief_complaint", "duration_minutes"],
                "Condition": ["id", "condition_name", "icd_code", "severity", "status", "onset_date", "confidence_score"],
                "Medication": ["id", "medication_name", "dosage", "frequency", "route", "rxnorm_code", "strength", "form"],
                "Test": ["id", "test_name", "value", "unit", "reference_range", "status", "test_category", "abnormal_flag"],
                "Procedure": ["id", "procedure_name", "cpt_code", "description", "date", "status", "duration_minutes"],
                "MedicalConcept": ["id", "name", "category", "description", "source", "confidence"]
            },
            relationship_properties={
                "HAS_VISIT": ["created_at", "confidence"],
                "DIAGNOSED_WITH": ["created_at", "confidence", "severity", "primary_diagnosis"],
                "PRESCRIBED": ["created_at", "confidence", "dosage", "frequency", "indication"],
                "PERFORMED": ["created_at", "confidence", "result_date", "abnormal_flag"],
                "UNDERWENT": ["created_at", "confidence", "procedure_date", "indication"],
                "TEMPORAL_RELATIONSHIP": ["temporal_order", "time_difference", "causality_confidence"],
                "KNOWLEDGE_RELATES_TO": ["relationship_type", "confidence"],
                "INTERACTS_WITH": ["interaction_type", "severity", "confidence"]
            },
            constraints=[
                {"type": "UNIQUE", "label": "Patient", "property": "id"},
                {"type": "UNIQUE", "label": "Patient", "property": "mrn"},
                {"type": "UNIQUE", "label": "Visit", "property": "id"},
                {"type": "UNIQUE", "label": "Condition", "property": "id"},
                {"type": "UNIQUE", "label": "Medication", "property": "id"},
                {"type": "UNIQUE", "label": "Test", "property": "id"},
                {"type": "UNIQUE", "label": "Procedure", "property": "id"},
                {"type": "UNIQUE", "label": "MedicalConcept", "property": "id"}
            ],
            indexes=[
                {"label": "Patient", "property": "name"},
                {"label": "Patient", "property": "dob"},
                {"label": "Visit", "property": "date"},
                {"label": "Visit", "property": "visit_type"},
                {"label": "Condition", "property": "condition_name"},
                {"label": "Condition", "property": "icd_code"},
                {"label": "Condition", "property": "status"},
                {"label": "Medication", "property": "medication_name"},
                {"label": "Medication", "property": "rxnorm_code"},
                {"label": "Test", "property": "test_name"},
                {"label": "Test", "property": "test_category"},
                {"label": "Procedure", "property": "procedure_name"},
                {"label": "Procedure", "property": "cpt_code"},
                {"label": "Entity", "property": "entity_id"},
                {"label": "Entity", "property": "concept_code"},
                {"label": "Entity", "property": "semantic_type"}
            ]
        )
        
        return schema
        
    except Exception as e:
        logger.error(f"Error retrieving graph schema: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve graph schema: {str(e)}"
        )


@router.post("/batch", response_model=GraphResponse)
async def batch_graph_operations(
    request: BatchGraphRequest,
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Execute batch graph operations for bulk updates.
    
    Args:
        request: Batch operations request
        
    Returns:
        GraphResponse: Batch operation results
    """
    try:
        logger.info(f"Executing {len(request.operations)} batch operations")
        
        start_time = datetime.now()
        processed_operations = 0
        errors = []
          # Process operations in batches
        for i in range(0, len(request.operations), request.batch_size):
            batch = request.operations[i:i + request.batch_size]
            
            for operation in batch:
                try:
                    # Process individual operation based on type
                    if operation.operation == "CREATE":
                        await _execute_create_operation(client, operation)
                        processed_operations += 1
                    elif operation.operation == "UPDATE":
                        await _execute_update_operation(client, operation)
                        processed_operations += 1
                    elif operation.operation == "MERGE":
                        await _execute_merge_operation(client, operation)
                        processed_operations += 1
                    else:
                        errors.append(f"Unsupported operation: {operation.operation}")
                        
                except Exception as e:
                    errors.append(f"Operation failed: {str(e)}")
                    logger.error(f"Batch operation failed: {e}", exc_info=True)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return GraphResponse(
            message=f"Batch operations completed: {processed_operations} processed, {len(errors)} errors",
            success=len(errors) == 0,
            data={
                "processed_operations": processed_operations,
                "total_operations": len(request.operations),
                "errors": errors
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error executing batch operations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute batch operations: {str(e)}"
        )


# New endpoints for advanced graph operations

@router.post("/temporal", response_model=Dict[str, Any])
async def create_temporal_relationship(
    request: Dict[str, Any],
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Create temporal relationships between medical entities.
    
    Args:
        request: Temporal relationship request containing patient_id, entity IDs, and temporal context
        
    Returns:
        Dict: Temporal relationship creation results
    """
    try:
        logger.info(f"Creating temporal relationship for patient: {request.get('patient_id')}")
        
        result = await client.create_temporal_relationship(
            patient_id=request["patient_id"],
            entity1_id=request["entity1_id"],
            entity2_id=request["entity2_id"],
            relationship_type=request["relationship_type"],
            temporal_context=request.get("temporal_context", {})
        )
        
        return {
            "message": "Temporal relationship created successfully",
            "patient_id": request["patient_id"],
            "relationship": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating temporal relationship: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create temporal relationship: {str(e)}"
        )


@router.get("/timeline/{patient_id}", response_model=Dict[str, Any])
async def get_patient_timeline(
    patient_id: str,
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Get comprehensive timeline analysis for a patient.
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        Dict: Patient timeline with medical events and analysis
    """
    try:
        logger.info(f"Analyzing timeline for patient: {patient_id}")
        
        timeline = await client.analyze_patient_timeline(patient_id)
        
        return {
            "message": "Patient timeline analyzed successfully",
            "timeline_data": timeline,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing patient timeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to analyze patient timeline: {str(e)}"
        )


@router.get("/patterns/{condition}", response_model=Dict[str, Any])
async def get_cross_patient_patterns(
    condition: str,
    limit: int = 100,
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Find patterns across patients with similar conditions.
    
    Args:
        condition: Medical condition name
        limit: Maximum number of patients to analyze
        
    Returns:
        Dict: Cross-patient pattern analysis results
    """
    try:
        logger.info(f"Analyzing cross-patient patterns for condition: {condition}")
        
        patterns = await client.find_cross_patient_patterns(condition, limit)
        
        return {
            "message": "Cross-patient patterns analyzed successfully",
            "pattern_data": patterns,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing cross-patient patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze patterns: {str(e)}"
        )


@router.post("/knowledge/expand", response_model=Dict[str, Any])
async def expand_knowledge_base(
    request: Dict[str, Any],
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Expand knowledge base with new medical concepts and relationships.
    
    Args:
        request: Knowledge expansion request with medical concepts
        
    Returns:
        Dict: Knowledge base expansion results
    """
    try:
        logger.info("Expanding knowledge base with new medical concepts")
        
        result = await client.expand_knowledge_base(
            medical_concepts=request.get("medical_concepts", [])
        )
        
        return {
            "message": "Knowledge base expanded successfully",
            "expansion_results": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error expanding knowledge base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to expand knowledge base: {str(e)}"
        )


@router.get("/insights/{patient_id}", response_model=Dict[str, Any])
async def get_patient_insights(
    patient_id: str,
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Generate comprehensive insights for a patient's medical graph.
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        Dict: Comprehensive patient insights including timeline, interactions, and care gaps
    """
    try:
        logger.info(f"Generating insights for patient: {patient_id}")
        
        insights = await client.get_patient_insights(patient_id)
        
        return {
            "message": "Patient insights generated successfully",
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating patient insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to generate patient insights: {str(e)}"
        )


# Helper endpoints for development and debugging

@router.delete("/reset", response_model=GraphResponse)
async def reset_graph_database(
    confirm: bool = False,
    client: Neo4jGraphClient = Depends(get_graph_client)
):
    """
    Reset the entire graph database (development only).
    
    WARNING: This will delete all data in the graph database.
    
    Args:
        confirm: Must be True to confirm the reset operation
        
    Returns:
        GraphResponse: Reset operation results
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset operation requires explicit confirmation (confirm=true)"
        )
    
    try:
        logger.warning("Resetting graph database - ALL DATA WILL BE DELETED")
        
        start_time = datetime.now()
        
        # Execute reset query directly without validation
        async with client.session() as session:
            result = await session.run("MATCH (n) DETACH DELETE n")
            summary = await result.consume()
            nodes_deleted = summary.counters.nodes_deleted
            relationships_deleted = summary.counters.relationships_deleted
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Graph database reset completed: {nodes_deleted} nodes and {relationships_deleted} relationships deleted")
        
        return GraphResponse(
            message=f"Graph database reset successfully: {nodes_deleted} nodes and {relationships_deleted} relationships deleted",
            success=True,
            data={
                "operation": "reset", 
                "status": "completed",
                "nodes_deleted": nodes_deleted,
                "relationships_deleted": relationships_deleted
            },
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error resetting graph database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset graph database: {str(e)}"
        )


async def _execute_create_operation(client: Neo4jGraphClient, operation) -> None:
    """Execute a CREATE operation on the graph database."""
    data = operation.data
    
    # Support both 'label' and 'node_type' for backwards compatibility
    node_type = data.get('label') or data.get('node_type')
    properties = data.get('properties', {})
    
    if not node_type:
        raise Exception("Either 'label' or 'node_type' must be specified in data")
    
    if not properties:
        raise Exception("Properties must be specified in data")
    
    # Map entity types to appropriate labels based on medical schema
    label_mapping = {
        'Patient': 'Patient',
        'Visit': 'Visit',
        'Condition': 'Condition',
        'Medication': 'Medication',
        'Test': 'Test',
        'LabTest': 'Test',  
        'Procedure': 'Procedure',
        'Symptom': 'Symptom',
        'Provider': 'Provider',
        'Facility': 'Facility',
        'MedicalConcept': 'MedicalConcept'
    }
    
    # Get all labels for this entity type
    primary_label = label_mapping.get(node_type, node_type)
    
    # Generate node ID if not provided
    node_id = properties.get('id') or f"{node_type}_{int(datetime.now().timestamp())}"
    
    # Ensure ID is in properties
    properties['id'] = node_id
    
    # Build property assignments
    property_assignments = []
    for key, value in properties.items():
        property_assignments.append(f"n.{key} = ${key}")
    
    # Add created timestamp
    property_assignments.append("n.created_at = datetime()")
    
    query = f"""
    CREATE (n:{primary_label})
    SET {', '.join(property_assignments)}
    RETURN n
    """
    
    logger.info(f"Executing create query: {query}")
    logger.info(f"With parameters: {properties}")
    
    # Execute the query
    result = await client.execute_query(GraphQueryRequest(query=query, parameters=properties))
    if not result.success:
        raise Exception(f"Failed to create node: {result.message}")
    
    logger.info(f"Successfully created {node_type} node with id={node_id}")


async def _execute_update_operation(client: Neo4jGraphClient, operation) -> None:
    """Execute an UPDATE operation on the graph database."""
    data = operation.data
    
    # Support multiple ways to identify the node to update
    node_id = data.get('id') or data.get('node_id')
    properties = data.get('properties', {})
    
    if not node_id:
        # Try to get ID from properties
        if 'id' in properties:
            node_id = properties['id']
        else:
            raise Exception("Node ID must be specified for update operation")
    
    if not properties:
        raise Exception("Properties must be specified for update operation")
    
    # Build SET clause for properties
    set_assignments = []
    for key, value in properties.items():
        if key != 'id':  # Don't update the ID field
            set_assignments.append(f"n.{key} = ${key}")
    
    # Add updated timestamp
    set_assignments.append("n.updated_at = datetime()")
    
    query = f"""
    MATCH (n {{id: $node_id}})
    SET {', '.join(set_assignments)}
    RETURN n
    """
    
    params = {'node_id': node_id, **properties}
    
    logger.info(f"Executing update query: {query}")
    logger.info(f"With parameters: {params}")
    
    # Execute the query
    result = await client.execute_query(GraphQueryRequest(query=query, parameters=params))
    if not result.success:
        raise Exception(f"Failed to update node: {result.message}")
    
    if result.result_count == 0:
        raise Exception(f"No node found with id: {node_id}")
    
    logger.info(f"Successfully updated node with id={node_id}")


async def _execute_merge_operation(client: Neo4jGraphClient, operation) -> None:
    """Execute a MERGE operation on the graph database."""
    data = operation.data
    
    # Support both 'label' and 'node_type' for backwards compatibility
    node_type = data.get('label') or data.get('node_type')
    properties = data.get('properties', {})
    
    if not node_type:
        raise Exception("Either 'label' or 'node_type' must be specified in data")
    
    if not properties:
        raise Exception("Properties must be specified in data")
    
    # Map entity types to appropriate labels based on medical schema
    label_mapping = {
        'Patient': 'Patient',
        'Visit': 'Visit',
        'Condition': 'Condition',
        'Medication': 'Medication',
        'Test': 'Test',
        'LabTest': 'Test',
        'Procedure': 'Procedure',
        'Symptom': 'Symptom',
        'Provider': 'Provider',
        'Facility': 'Facility',
        'MedicalConcept': 'MedicalConcept'
    }
    
    # Get all labels for this entity type
    primary_label = label_mapping.get(node_type, node_type)
    merge_key = data.get('merge_key', 'id')  # Default merge on 'id' field
    
    if merge_key not in properties:
        raise Exception(f"Merge key '{merge_key}' not found in properties")
    
    # Build property assignments for ON CREATE and ON MATCH
    create_assignments = []
    match_assignments = []
    
    for key, value in properties.items():
        if key == merge_key:
            continue  # Skip the merge key as it's already in the MERGE clause
        create_assignments.append(f"n.{key} = ${key}")
        match_assignments.append(f"n.{key} = ${key}")
    
    # Add timestamp fields
    create_assignments.append("n.created_at = datetime()")
    match_assignments.append("n.updated_at = datetime()")
    
    query = f"""
    MERGE (n:{primary_label} {{{merge_key}: ${merge_key}}})
    ON CREATE SET {', '.join(create_assignments)}
    ON MATCH SET {', '.join(match_assignments)}
    RETURN n
    """
    
    logger.info(f"Executing merge query: {query}")
    logger.info(f"With parameters: {properties}")
    
    # Execute the query
    result = await client.execute_query(GraphQueryRequest(query=query, parameters=properties))
    if not result.success:
        raise Exception(f"Failed to merge node: {result.message}")
    
    logger.info(f"Successfully merged {node_type} node with {merge_key}={properties[merge_key]}")
