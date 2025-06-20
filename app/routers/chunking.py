"""
Chunking router for document text segmentation and processing.

This module provides REST API endpoints for breaking down documents into
manageable chunks for further processing and analysis.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

from app.schemas.chunking import (
    ChunkRequest, ChunkResponse, TimelineRequest,
    StructureRequest, StructureResponse, ChunkingConfig, ChunkingStrategy
)
from app.services.chunker import chunker
from app.services.vector_store import vector_store

# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter(tags=["chunking"])


@router.get("/info")
async def get_chunking_info():
    """
    Get detailed information about the chunking service capabilities and configuration.
    
    Returns:
        JSONResponse: Service information, processing capabilities, and configuration details
    """
    return JSONResponse(
        content={
            "service_name": "MRIA Chunking Service",
            "service_type": "text_processor",
            "version": "1.0.0",
            "status": "active",
            "capabilities": [
                "document_segmentation",
                "semantic_chunking",
                "visit_based_chunking",
                "topic_based_chunking",
                "temporal_chunking",
                "timeline_creation",
                "medical_entity_extraction",
                "quality_assessment"
            ],
            "supported_formats": [
                "text/plain",
                "text/markdown",
                "application/pdf",
                "text/html",
                "application/msword"
            ],
            "chunking_strategies": [
                "visit_based",
                "topic_based", 
                "temporal",
                "semantic",
                "fixed_size",
                "sentence_boundary",
                "paragraph_boundary"
            ],
            "configuration": {
                "default_chunk_size": 1000,
                "default_overlap": 200,
                "max_chunk_size": 4000,
                "min_chunk_size": 100,
                "semantic_threshold": 0.7
            },            "endpoints": [
                "/chunk/info",
                "/chunk/process",
                "/chunk/timeline",
                "/chunk/timeline/{patient_id}",
                "/chunk/chunks/{patient_id}",
                "/chunk/structure",
                "/chunk/search",
                "/chunk/strategies",
                "/chunk/config"
            ],
            "description": "Intelligently segments medical documents into optimal chunks for processing and analysis, with timeline structuring capabilities"
        }
    )


@router.post("/process", response_model=ChunkResponse)
async def process_document_chunks(request: ChunkRequest):
    """
    Process document into chunks using specified strategy.
    
    Args:
        request: ChunkRequest containing text and configuration
        
    Returns:
        ChunkResponse: Generated chunks with metadata and quality metrics
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        logger.info(f"Processing document chunks for job {request.job_id} with strategy {request.config.strategy}")
        
        # Validate input
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text content is required")
        
        if len(request.text) > 1000000:  # 1MB limit
            raise HTTPException(status_code=400, detail="Text content too large (max 1MB)")
        
        # Process chunks
        response = await chunker.process_chunks(request)
        
        # Create timeline if requested
        if request.create_timeline and request.patient_id:
            timeline_request = TimelineRequest(
                patient_id=request.patient_id,
                chunks=response.chunks,
                include_summary=True,
                confidence_threshold=0.5
            )
            
            timeline_response = await chunker.create_patient_timeline(timeline_request)
            response.timeline_created = True
            response.quality_metrics["timeline_confidence"] = timeline_response.confidence_score
        
        logger.info(f"Successfully processed {response.total_chunks} chunks for job {request.job_id}")
        return response
        
    except Exception as e:
        logger.error(f"Chunk processing failed for job {request.job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk processing failed: {str(e)}")


@router.get("/timeline/{patient_id}")
async def get_patient_timeline(patient_id: str):
    """
    Get existing patient timeline by patient ID from vector database.
    
    This endpoint retrieves a patient's medical timeline from the vector database,
    including all visits, medical entities, and timeline metadata.
    
    Args:
        patient_id: Patient identifier (string)
        
    Returns:
        Dict: Patient timeline information with the following structure:
            - patient_id: Patient identifier
            - timeline_entries: List of chronological medical events
            - total_visits: Number of medical visits
            - date_range: Timeline start and end dates  
            - summary: Brief timeline summary
            - metadata: Additional timeline metadata
        
    Raises:
        HTTPException: 404 if patient timeline not found, 500 for retrieval errors
    """
    try:
        logger.info(f"Retrieving timeline for patient {patient_id} from vector database")
        
        # Validate patient_id
        if not patient_id or not patient_id.strip():
            raise HTTPException(status_code=400, detail="Patient ID is required")
        
        # Retrieve timeline from vector database
        timeline = await vector_store.get_patient_timeline(patient_id.strip())
        
        if not timeline:
            logger.warning(f"No timeline found for patient {patient_id} in vector database")
            raise HTTPException(
                status_code=404, 
                detail=f"Timeline not found for patient {patient_id}. Please ensure the patient has processed medical data."
            )
        
        # Log successful retrieval
        logger.info(f"Successfully retrieved timeline for patient {patient_id}: {timeline.total_visits} visits, "
                   f"date range: {timeline.date_range['start']} to {timeline.date_range['end']}")
          # Convert timeline to dictionary and add metadata
        timeline_data = timeline.dict()
        
        # Ensure datetime objects are properly serialized
        if timeline_data.get("date_range"):
            if "start" in timeline_data["date_range"] and timeline_data["date_range"]["start"]:
                timeline_data["date_range"]["start"] = timeline_data["date_range"]["start"].isoformat() if hasattr(timeline_data["date_range"]["start"], 'isoformat') else str(timeline_data["date_range"]["start"])
            if "end" in timeline_data["date_range"] and timeline_data["date_range"]["end"]:
                timeline_data["date_range"]["end"] = timeline_data["date_range"]["end"].isoformat() if hasattr(timeline_data["date_range"]["end"], 'isoformat') else str(timeline_data["date_range"]["end"])
        
        # Serialize timeline entries dates
        for entry in timeline_data.get("timeline_entries", []):
            if entry.get("date"):
                entry["date"] = entry["date"].isoformat() if hasattr(entry["date"], 'isoformat') else str(entry["date"])
        
        timeline_data["metadata"].update({
            "retrieved_at": datetime.now().isoformat(),
            "source": "vector_database",
            "vector_store_status": "active"
        })
        
        return JSONResponse(content=timeline_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404) without modification
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve timeline for patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Timeline retrieval failed: {str(e)}")


@router.post("/structure", response_model=StructureResponse)
async def create_structured_timeline(request: StructureRequest):
    """
    Create structured medical timeline from raw text.
    
    Args:
        request: StructureRequest with patient ID and raw text
        
    Returns:
        StructureResponse: Structured timeline with chunks and quality metrics
    """
    try:
        logger.info(f"Creating structured timeline for patient {request.patient_id}")
        
        # Validate input
        if not request.raw_text or not request.raw_text.strip():
            raise HTTPException(status_code=400, detail="Raw text is required")
        
        # First, chunk the raw text
        chunk_request = ChunkRequest(
            text=request.raw_text,
            patient_id=request.patient_id,
            config=ChunkingConfig(
                strategy=ChunkingStrategy.VISIT_BASED,
                preserve_medical_context=True,
                include_metadata=True
            ),
            create_timeline=False
        )
        
        chunk_response = await chunker.process_chunks(chunk_request)
        
        # Create timeline from chunks
        timeline_request = TimelineRequest(
            patient_id=request.patient_id,
            chunks=chunk_response.chunks,
            include_summary=True,
            confidence_threshold=0.5
        )
        
        timeline_response = await chunker.create_patient_timeline(timeline_request)
        
        # Merge with existing timeline if provided
        if request.existing_timeline:
            # Implement timeline merging logic here
            logger.info(f"Merging with existing timeline using strategy: {request.merge_strategy}")
            # For now, just use the new timeline
            structured_timeline = timeline_response.patient_timeline
        else:
            structured_timeline = timeline_response.patient_timeline
        
        # Calculate quality assessment
        quality_assessment = {
            "timeline_confidence": timeline_response.confidence_score,
            "chunk_quality": chunk_response.quality_metrics.get("coherence", 0.0),
            "entity_coverage": chunk_response.quality_metrics.get("entity_distribution", 0.0),
            "temporal_consistency": 0.85,  # Placeholder metric
            "overall_quality": (
                timeline_response.confidence_score + 
                chunk_response.quality_metrics.get("coherence", 0.0) + 
                chunk_response.quality_metrics.get("entity_distribution", 0.0)
            ) / 3.0
        }
        
        processing_summary = {
            "chunks_created": len(chunk_response.chunks),
            "timeline_entries": len(structured_timeline.timeline_entries),
            "processing_time": chunk_response.processing_time + timeline_response.processing_time,
            "strategy_used": chunk_request.config.strategy.value,
            "entities_extracted": chunk_response.medical_entities_found,            "warnings": timeline_response.warnings
        }
        
        response = StructureResponse(
            structured_timeline=structured_timeline,
            chunks_created=chunk_response.chunks,
            processing_summary=processing_summary,
            quality_assessment=quality_assessment
        )
        
        # Store timeline and chunks in vector database
        try:
            # Store the timeline
            timeline_stored = await vector_store.store_patient_timeline(structured_timeline)
            if timeline_stored:
                logger.info(f"Timeline stored in vector database for patient {request.patient_id}")
            else:
                logger.warning(f"Failed to store timeline in vector database for patient {request.patient_id}")
            
            # Store the chunks
            chunks_stored = await vector_store.store_medical_chunks(chunk_response.chunks, request.patient_id)
            if chunks_stored:
                logger.info(f"Chunks stored in vector database for patient {request.patient_id}")
            else:
                logger.warning(f"Failed to store chunks in vector database for patient {request.patient_id}")
                
        except Exception as e:
            logger.error(f"Error storing data in vector database: {e}")
            # Don't fail the request if storage fails
        
        logger.info(f"Successfully created structured timeline for patient {request.patient_id}")
        return response
        
    except Exception as e:
        logger.error(f"Timeline structuring failed for patient {request.patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Timeline structuring failed: {str(e)}")


@router.get("/strategies")
async def get_chunking_strategies():
    """
    Get available chunking strategies with descriptions.
    
    Returns:
        Dict: Available chunking strategies and their descriptions
    """
    return JSONResponse(
        content={
            "strategies": {
                "visit_based": {
                    "name": "Visit-based Chunking",
                    "description": "Groups information by medical visits/encounters",
                    "use_cases": ["Medical records with multiple visits", "Patient history analysis"],
                    "advantages": ["Maintains visit context", "Natural medical grouping"],
                    "best_for": "Multi-visit medical records"
                },
                "topic_based": {
                    "name": "Topic-based Chunking",
                    "description": "Groups related medical information together",
                    "use_cases": ["Medical research", "Topic-specific analysis"],
                    "advantages": ["Groups similar medical topics", "Good for thematic analysis"],
                    "best_for": "Medical literature and research documents"
                },
                "temporal": {
                    "name": "Temporal Chunking",
                    "description": "Organizes information chronologically",
                    "use_cases": ["Timeline analysis", "Disease progression tracking"],
                    "advantages": ["Chronological order", "Temporal relationships"],
                    "best_for": "Long-term patient monitoring"
                },
                "semantic": {
                    "name": "Semantic Chunking",
                    "description": "Uses embeddings to group related content",
                    "use_cases": ["Complex medical texts", "Semantic similarity analysis"],
                    "advantages": ["Intelligent content grouping", "Context-aware chunking"],
                    "best_for": "Complex medical documents with varied content"
                },
                "fixed_size": {
                    "name": "Fixed Size Chunking",
                    "description": "Divides text into fixed-size chunks with overlap",
                    "use_cases": ["Large document processing", "Uniform chunk sizes"],
                    "advantages": ["Predictable chunk sizes", "Simple processing"],
                    "best_for": "Large documents requiring uniform processing"
                },
                "sentence_boundary": {
                    "name": "Sentence Boundary Chunking",
                    "description": "Respects sentence boundaries while chunking",
                    "use_cases": ["Preserving sentence integrity", "Natural language processing"],
                    "advantages": ["Maintains sentence structure", "Natural boundaries"],
                    "best_for": "Documents where sentence integrity is important"
                },
                "paragraph_boundary": {
                    "name": "Paragraph Boundary Chunking",
                    "description": "Chunks text at paragraph boundaries",
                    "use_cases": ["Structured documents", "Maintaining paragraph context"],
                    "advantages": ["Preserves paragraph structure", "Natural content grouping"],
                    "best_for": "Well-structured medical documents"
                }
            },
            "default_strategy": "semantic",
            "recommended_strategies": {
                "medical_records": "visit_based",
                "research_papers": "topic_based",
                "patient_history": "temporal",
                "complex_documents": "semantic"
            }
        }
    )


@router.get("/config")
async def get_chunking_config():
    """
    Get current chunking configuration parameters.
    
    Returns:
        Dict: Current configuration parameters and their descriptions
    """
    return JSONResponse(
        content={
            "configuration": {                "chunk_size": {
                    "default": 1000,
                    "min": 100,
                    "max": 4000,
                    "description": "Target chunk size in characters"
                },
                "min_chunk_size": {
                    "default": 100,
                    "min": 50,
                    "max": 1000,
                    "description": "Minimum chunk size in characters"
                },
                "overlap": {
                    "default": 200,
                    "min": 0,
                    "max": 500,
                    "description": "Overlap between chunks in characters"
                },
                "semantic_threshold": {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "Semantic similarity threshold for grouping"
                },
                "preserve_medical_context": {
                    "default": True,
                    "description": "Keep medical context intact during chunking"
                },
                "include_metadata": {
                    "default": True,
                    "description": "Include chunk metadata in results"
                },
                "confidence_threshold": {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "Minimum confidence for timeline entries"
                }
            },
            "quality_metrics": {
                "coverage": "Percentage of original text covered by chunks",
                "coherence": "Average semantic coherence score of chunks",
                "entity_distribution": "How evenly medical entities are distributed",
                "timeline_confidence": "Confidence score for timeline creation"
            },
            "medical_entities": {
                "supported_types": ["date", "medication", "vitals", "lab_values", "symptoms", "diagnoses"],
                "extraction_methods": ["pattern_based", "nlp_based", "hybrid"],
                "confidence_scoring": "Based on extraction method and pattern matching"
            }
        }
    )


@router.get("/search")
async def search_similar_timelines(query: str, limit: int = 5):
    """
    Search for similar patient timelines using semantic search.
    
    Args:
        query: Search query text
        limit: Maximum number of results to return
        
    Returns:
        Dict: Similar timeline entries with similarity scores
    """
    try:
        logger.info(f"Searching for similar timelines with query: '{query}'")
        
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Search query is required")
        
        if limit < 1 or limit > 20:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 20")
        
        # Search for similar timelines
        similar_timelines = await vector_store.search_similar_timelines(query.strip(), limit)
        
        return JSONResponse(
            content={
                "query": query,
                "results_count": len(similar_timelines),
                "results": similar_timelines,
                "search_metadata": {
                    "embedding_model": "all-MiniLM-L6-v2",
                    "search_type": "semantic_similarity",
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


