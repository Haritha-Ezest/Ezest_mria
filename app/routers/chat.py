"""
Enhanced chat router with ChromaDB and Neo4j integration.

This router provides REST API endpoints for enhanced natural language medical queries
that leverage both ChromaDB vector store and Neo4j graph database for comprehensive
medical data retrieval and analysis.
"""
import uuid
from datetime import datetime
from typing import Dict, Optional, List

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks

from app.schemas.chat import (
    ChatQueryRequest, ChatQueryResponse,
    ConversationContextRequest, ConversationContextResponse,
    ChatFeedbackRequest, ChatFeedbackResponse,
    ConversationHistoryResponse,
    ChatMetrics, ChatSystemStatus
)
from app.schemas.graph import GraphQueryRequest
from app.services.chat_processor_enhanced import EnhancedChatProcessor, get_enhanced_chat_processor
from app.common.utils import get_logger

# Configure logging
logger = get_logger(__name__)

# Create enhanced router instance with original tags for compatibility
router = APIRouter(tags=["chat"])


@router.post("/query", response_model=ChatQueryResponse)
async def process_chat_query(
    request: ChatQueryRequest,
    background_tasks: BackgroundTasks,
    chat_processor: EnhancedChatProcessor = Depends(get_enhanced_chat_processor)
) -> ChatQueryResponse:
    """
    Process natural language medical query with enhanced data retrieval.
    
    This endpoint provides enhanced medical query processing by:
    - Retrieving patient timelines from ChromaDB vector store
    - Querying structured medical data from Neo4j graph database
    - Performing semantic search across similar patient cases
    - Combining and analyzing data from multiple sources
    - Generating comprehensive natural language responses
    
    Supports various medical query types:
    - Patient history summaries with timeline analysis
    - Population health queries with cohort analysis
    - Treatment comparisons across similar patients
    - Medication interaction analysis
    - Clinical decision support with evidence
    
    Args:
        request: Enhanced chat query request with user query and context
        background_tasks: FastAPI background tasks for async processing
        chat_processor: Enhanced chat processing service dependency
        
    Returns:
        ChatQueryResponse with comprehensive analysis from multiple data sources
        
    Raises:
        HTTPException: For invalid requests or processing errors
    """
    try:
        logger.info(f"Processing enhanced chat query: '{request.query[:100]}...' for session: {request.session_id}")
        
        # Validate query
        if len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query too short. Please provide a more detailed medical question."
            )
        
        # Validate context if provided
        if request.context and request.context.patient_id:
            logger.info(f"Processing query with patient context: {request.context.patient_id}")
        
        # Process the enhanced query
        response = await chat_processor.process_enhanced_query(request)
        
        # Log processing metrics in background
        background_tasks.add_task(
            _log_enhanced_query_metrics,
            query=request.query,
            session_id=response.session_id,
            processing_time=response.processing_time,
            confidence=response.confidence,
            sources_used=response.sources_used,
            results_count=len(response.results)
        )
        
        logger.info(
            f"Enhanced chat query processed - Session: {response.session_id}, "
            f"Confidence: {response.confidence:.2f}, Time: {response.processing_time:.2f}s, "
            f"Sources: {', '.join(response.sources_used)}, Results: {len(response.results)}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing enhanced chat query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process enhanced query: {str(e)}"
        )


@router.post("/patient-summary", response_model=ChatQueryResponse)
async def get_enhanced_patient_summary(
    patient_id: str,
    time_period: Optional[str] = None,
    include_similar_cases: bool = True,
    chat_processor: EnhancedChatProcessor = Depends(get_enhanced_chat_processor)
) -> ChatQueryResponse:
    """
    Get enhanced patient summary with data from ChromaDB and Neo4j.
    
    This endpoint provides comprehensive patient summaries by:
    - Retrieving patient timeline from ChromaDB
    - Querying detailed medical records from Neo4j
    - Finding similar patient cases for comparison
    - Analyzing treatment patterns and outcomes
    
    Args:
        patient_id: Patient identifier
        time_period: Optional time period filter (e.g., "last_6_months")
        include_similar_cases: Whether to include similar patient cases
        chat_processor: Enhanced chat processing service
        
    Returns:
        Comprehensive patient summary with timeline and comparative analysis
    """
    try:
        logger.info(f"Generating enhanced patient summary for: {patient_id}")
        
        # Create context for patient
        from app.schemas.chat import ConversationContext, ContextType
        context = ConversationContext(
            patient_id=patient_id,
            time_period=time_period,
            context_type=ContextType.PATIENT_FOCUSED
        )
        
        # Create query request
        query_text = f"Provide comprehensive summary for patient {patient_id}"
        if time_period:
            query_text += f" for the {time_period}"
        if include_similar_cases:
            query_text += " and compare with similar patients"
        
        request = ChatQueryRequest(
            query=query_text,
            context=context,
            session_id=str(uuid.uuid4()),
            include_sources=True,
            max_results=20
        )
        
        # Process with enhanced retrieval
        response = await chat_processor.process_enhanced_query(request)
        
        logger.info(f"Enhanced patient summary generated for: {patient_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating enhanced patient summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate patient summary: {str(e)}"
        )


@router.post("/population-analysis", response_model=ChatQueryResponse)
async def analyze_population_health(
    conditions: List[str],
    demographic_filters: Optional[Dict[str, str]] = None,
    include_trends: bool = True,
    chat_processor: EnhancedChatProcessor = Depends(get_enhanced_chat_processor)
) -> ChatQueryResponse:
    """
    Perform population health analysis using ChromaDB and Neo4j.
    
    This endpoint provides population-level medical analysis by:
    - Searching for patients with specified conditions in ChromaDB
    - Querying population statistics from Neo4j graph
    - Analyzing demographic patterns and trends
    - Comparing treatment outcomes across cohorts
    
    Args:
        conditions: List of medical conditions to analyze
        demographic_filters: Optional demographic filters
        include_trends: Whether to include trend analysis
        chat_processor: Enhanced chat processing service
        
    Returns:
        Comprehensive population health analysis
    """
    try:
        logger.info(f"Performing population analysis for conditions: {conditions}")
        
        # Create context for population analysis
        from app.schemas.chat import ConversationContext, ContextType
        context = ConversationContext(
            condition_focus=conditions[0] if conditions else None,
            context_type=ContextType.POPULATION_FOCUSED
        )
        
        # Create query request
        query_text = f"Analyze population health for patients with {', '.join(conditions)}"
        if demographic_filters:
            query_text += f" with demographics: {demographic_filters}"
        if include_trends:
            query_text += " including health trends and patterns"
        
        request = ChatQueryRequest(
            query=query_text,
            context=context,
            session_id=str(uuid.uuid4()),
            include_sources=True,
            max_results=50
        )
        
        # Process population analysis
        response = await chat_processor.process_enhanced_query(request)
        
        logger.info(f"Population analysis completed for: {conditions}")
        return response
        
    except Exception as e:
        logger.error(f"Error in population analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform population analysis: {str(e)}"
        )


@router.get("/data-sources/status")
async def get_data_sources_status(
    chat_processor: EnhancedChatProcessor = Depends(get_enhanced_chat_processor)
) -> Dict[str, str]:
    """
    Get status of integrated data sources (ChromaDB and Neo4j).
    
    Returns:
        Status information for ChromaDB and Neo4j connections
    """
    try:
        status_info = {
            "chromadb": "unknown",
            "neo4j": "unknown",
            "overall": "unknown"
        }
        
        # Check ChromaDB status
        try:
            # Try to access ChromaDB client
            if chat_processor.vector_store.client:
                status_info["chromadb"] = "connected"
            else:
                status_info["chromadb"] = "disconnected"
        except Exception as e:
            status_info["chromadb"] = f"error: {str(e)}"
        
        # Check Neo4j status
        try:            # Try to execute a simple query
            test_result = await chat_processor.graph_client.execute_query(
                GraphQueryRequest(
                    query="RETURN 1 as test",
                    parameters={}
                )
            )
            if test_result:
                status_info["neo4j"] = "connected"
            else:
                status_info["neo4j"] = "disconnected"
        except Exception as e:
            status_info["neo4j"] = f"error: {str(e)}"
        
        # Determine overall status
        if status_info["chromadb"] == "connected" and status_info["neo4j"] == "connected":
            status_info["overall"] = "healthy"
        elif "connected" in [status_info["chromadb"], status_info["neo4j"]]:
            status_info["overall"] = "partial"
        else:
            status_info["overall"] = "unhealthy"
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error checking data sources status: {str(e)}")
        return {
            "chromadb": "error",
            "neo4j": "error",
            "overall": "error",
            "error_message": str(e)
        }


@router.get("/health")
async def enhanced_chat_health_check() -> Dict[str, str]:
    """Health check for enhanced chat system."""
    return {
        "status": "healthy",
        "service": "enhanced-chat",
        "features": "chromadb,neo4j,semantic-search,graph-queries",
        "timestamp": datetime.utcnow().isoformat()
    }


# Background task functions
async def _log_enhanced_query_metrics(
    query: str,
    session_id: str,
    processing_time: float,
    confidence: float,
    sources_used: List[str],
    results_count: int
):
    """Log enhanced query processing metrics."""
    try:
        logger.info(
            f"Enhanced query metrics - Session: {session_id}, "
            f"Confidence: {confidence:.2f}, Time: {processing_time:.2f}s, "
            f"Sources: {', '.join(sources_used)}, Results: {results_count}"
        )
        # In production, this would store detailed metrics in analytics database
    except Exception as e:
        logger.error(f"Error logging enhanced query metrics: {str(e)}")
