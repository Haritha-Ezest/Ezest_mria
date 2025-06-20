"""
Insights router for medical analysis and pattern recognition.

This module provides REST API endpoints for generating medical insights,
analyzing treatment patterns, and providing clinical decision support.
"""

import traceback
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse

from app.schemas.insights import (
    PatientInsightRequest, PatientInsightResponse,
    PatientComparisonRequest, PatientComparisonResponse,
    PopulationInsightRequest, PopulationInsightResponse,
    ClinicalRecommendationRequest, ClinicalRecommendationResponse,
    InsightStatus, InsightMetrics, InsightType
)
from app.services.insights_processor import InsightsProcessor, get_insights_processor
from app.services.graph_client import Neo4jGraphClient
from app.common.utils import get_logger

# Configure logging
logger = get_logger(__name__)

# Create router instance
router = APIRouter(tags=["insights"])


async def get_graph_client() -> Neo4jGraphClient:
    """Dependency to get Neo4j graph client with robust connection handling."""
    from app.config import get_settings
    from dotenv import load_dotenv
    
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
    
    return neo4j_client


async def get_insights_service(
    graph_client: Neo4jGraphClient = Depends(get_graph_client)
) -> InsightsProcessor:
    """Dependency to get insights processor service."""
    return get_insights_processor(graph_client)


@router.post("/generate/{patient_id}", response_model=PatientInsightResponse)
async def generate_patient_insights(
    patient_id: str,
    request: PatientInsightRequest,
    insights_service: InsightsProcessor = Depends(get_insights_service)
) -> PatientInsightResponse:
    """
    Generate comprehensive insights for a specific patient.
    
    This endpoint analyzes patient data to generate various types of medical insights
    including treatment progression, risk assessment, and clinical recommendations.
    
    Args:
        patient_id: Unique identifier for the patient
        request: Patient insight request parameters
        insights_service: Insights processor service
        
    Returns:
        PatientInsightResponse containing generated insights
        
    Raises:
        HTTPException: If patient not found or insights generation fails
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Generating insights for patient {patient_id}")
        
        # Validate patient_id matches request
        if request.patient_id != patient_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Patient ID in URL does not match request body"
            )
          # Generate insights
        insights = await insights_service.generate_patient_insights(request)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate total number of insights across all types
        total_insights_count = 0
        for insight_type, insight_list in insights.items():
            if isinstance(insight_list, list):
                total_insights_count += len(insight_list)
            elif isinstance(insight_list, dict) and "error" not in insight_list:
                # Handle single insight objects
                total_insights_count += 1
        
        logger.info(f"Successfully generated {total_insights_count} insights across {len(insights)} insight types for patient {patient_id}")
        
        return PatientInsightResponse(
            message=f"Successfully generated {total_insights_count} insights for patient {patient_id}",
            patient_id=patient_id,
            insights=insights,
            generated_at=datetime.now(),
            processing_time=processing_time,
            insights_count=total_insights_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating insights for patient {patient_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate insights: {str(e)}"
        )


@router.post("/compare", response_model=PatientComparisonResponse)
async def compare_patients(
    request: PatientComparisonRequest,
    insights_service: InsightsProcessor = Depends(get_insights_service)
) -> PatientComparisonResponse:
    """
    Compare a primary patient with similar patients based on specified criteria.
    
    This endpoint finds patients with similar characteristics and compares their
    outcomes, treatments, and progression patterns.
    
    Args:
        request: Patient comparison request parameters
        insights_service: Insights processor service
        
    Returns:
        PatientComparisonResponse containing comparison results
        
    Raises:
        HTTPException: If comparison fails or patients not found
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Comparing patient {request.primary_patient_id} with similar patients")
        
        # Perform comparison analysis
        comparison_results = await insights_service.compare_patients(request)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Successfully compared patient {request.primary_patient_id} with {len(comparison_results.comparison_patients)} similar patients")
        
        return PatientComparisonResponse(
            message=f"Successfully compared patient {request.primary_patient_id} with {len(comparison_results.comparison_patients)} similar patients",
            primary_patient_id=request.primary_patient_id,
            comparison_results=comparison_results,
            generated_at=datetime.now(),
            processing_time=processing_time,
            patients_compared=len(comparison_results.comparison_patients)
        )
        
    except Exception as e:
        logger.error(f"Error comparing patients: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare patients: {str(e)}"
        )


@router.get("/population/{condition}", response_model=PopulationInsightResponse)
async def get_population_insights(
    condition: str,
    time_period: str = "1 year",
    age_min: int = None,
    age_max: int = None,
    gender: str = "any",
    include_trends: bool = True,
    insights_service: InsightsProcessor = Depends(get_insights_service)
) -> PopulationInsightResponse:
    """
    Generate population-level insights for a specific medical condition.
    
    This endpoint analyzes population trends, prevalence rates, treatment patterns,
    and outcomes for a specified medical condition.
    
    Args:
        condition: Medical condition to analyze
        time_period: Time period for analysis (default: "1 year")
        age_min: Minimum age for population filter
        age_max: Maximum age for population filter
        gender: Gender filter ("male", "female", "any")
        include_trends: Whether to include trend analysis
        insights_service: Insights processor service
        
    Returns:
        PopulationInsightResponse containing population insights
        
    Raises:
        HTTPException: If analysis fails or insufficient data
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Generating population insights for condition: {condition}")
        
        # Build demographic filters
        demographic_filters = {}
        if age_min is not None or age_max is not None:
            demographic_filters["age_range"] = [
                age_min or 0, 
                age_max or 120
            ]
        if gender != "any":
            demographic_filters["gender"] = gender
        
        # Create request
        request = PopulationInsightRequest(
            condition=condition,
            time_period=time_period,
            demographic_filters=demographic_filters if demographic_filters else None,
            include_trends=include_trends
        )
        
        # Generate population insights
        population_insights = await insights_service.generate_population_insights(request)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Successfully generated population insights for {condition} (population size: {population_insights.population_size})")
        
        return PopulationInsightResponse(
            message=f"Successfully generated population insights for {condition}",
            population_insights=population_insights,
            generated_at=datetime.now(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error generating population insights for {condition}: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate population insights: {str(e)}"
        )


@router.post("/recommendations/{patient_id}", response_model=ClinicalRecommendationResponse)
async def generate_clinical_recommendations(
    patient_id: str,
    request: ClinicalRecommendationRequest,
    insights_service: InsightsProcessor = Depends(get_insights_service)
) -> ClinicalRecommendationResponse:
    """
    Generate clinical decision support recommendations for a patient.
    
    This endpoint analyzes patient data to provide evidence-based clinical
    recommendations for treatment, monitoring, and care management.
    
    Args:
        patient_id: Unique identifier for the patient
        request: Clinical recommendation request parameters
        insights_service: Insights processor service
        
    Returns:
        ClinicalRecommendationResponse containing recommendations
        
    Raises:
        HTTPException: If patient not found or recommendation generation fails
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Generating clinical recommendations for patient {patient_id}")
        
        # Validate patient_id matches request
        if request.patient_id != patient_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Patient ID in URL does not match request body"
            )
        
        # Generate recommendations
        recommendations = await insights_service.generate_clinical_recommendations(request)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Successfully generated {len(recommendations)} clinical recommendations for patient {patient_id}")
        
        return ClinicalRecommendationResponse(
            message=f"Successfully generated {len(recommendations)} clinical recommendations for patient {patient_id}",
            patient_id=patient_id,
            recommendations=recommendations,
            generated_at=datetime.now(),
            processing_time=processing_time,
            recommendations_count=len(recommendations)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating clinical recommendations for patient {patient_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate clinical recommendations: {str(e)}"
        )


@router.get("/status/{job_id}", response_model=InsightStatus)
async def get_insight_status(
    job_id: str
) -> InsightStatus:
    """
    Get the status of an insight generation job.
    
    This endpoint provides real-time status updates for long-running
    insight generation processes.
    
    Args:
        job_id: Unique identifier for the insight generation job
        
    Returns:
        InsightStatus containing job progress and status
        
    Raises:
        HTTPException: If job not found
    """
    try:
        logger.info(f"Checking status for insight job {job_id}")
        
        # In a real implementation, this would check a job queue/database
        # For now, return a mock status
        return InsightStatus(
            job_id=job_id,
            status="completed",
            progress=100.0,
            current_step="Insight generation completed",
            estimated_completion=None,
            error_message=None,
            insights_generated=5,
            total_insights_requested=5
        )
        
    except Exception as e:
        logger.error(f"Error getting status for job {job_id}: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )


@router.get("/metrics", response_model=InsightMetrics)
async def get_insight_metrics() -> InsightMetrics:
    """
    Get metrics about insight generation performance and usage.
    
    This endpoint provides system-wide metrics about insight generation
    including performance statistics and usage patterns.
    
    Returns:
        InsightMetrics containing system metrics
    """
    try:
        logger.info("Retrieving insight generation metrics")
        
        # In a real implementation, this would query metrics from database/monitoring
        # For now, return mock metrics
        return InsightMetrics(
            total_insights_generated=1250,
            average_processing_time=3.5,
            success_rate=0.95,
            insights_by_type={
                InsightType.TREATMENT_PROGRESSION: 450,
                InsightType.RISK_ASSESSMENT: 350,
                InsightType.COMPARATIVE_ANALYSIS: 200,
                InsightType.CLINICAL_DECISION_SUPPORT: 150,
                InsightType.POPULATION_HEALTH: 100
            },
            average_confidence=0.82,
            high_confidence_rate=0.68,
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error retrieving insight metrics: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve insight metrics: {str(e)}"
        )


@router.post("/batch/generate")
async def generate_batch_insights(
    patient_ids: list[str],
    insight_types: list[InsightType],
    background_tasks: BackgroundTasks,
    time_period: str = "1 year"
) -> JSONResponse:
    """
    Generate insights for multiple patients in batch mode.
    
    This endpoint queues insight generation for multiple patients
    to be processed in the background.
    
    Args:
        patient_ids: List of patient IDs to process
        insight_types: Types of insights to generate
        background_tasks: FastAPI background tasks
        time_period: Time period for analysis
        
    Returns:
        JSONResponse with batch job information
    """
    try:
        logger.info(f"Queuing batch insight generation for {len(patient_ids)} patients")
        
        # Generate a batch job ID
        batch_job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(patient_ids)}"
        
        # In a real implementation, this would queue the jobs for background processing
        # For now, just return success
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "message": f"Batch insight generation queued for {len(patient_ids)} patients",
                "batch_job_id": batch_job_id,
                "patients_count": len(patient_ids),
                "insight_types": [it.value for it in insight_types],
                "status": "queued",
                "estimated_completion": "15-30 minutes"
            }
        )
        
    except Exception as e:
        logger.error(f"Error queuing batch insight generation: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue batch insight generation: {str(e)}"
        )


# Health check endpoint for insights service
@router.get("/health")
async def insights_health_check(
    insights_service: InsightsProcessor = Depends(get_insights_service)
) -> JSONResponse:
    """
    Health check endpoint for insights service.
    
    Returns:
        JSONResponse indicating service health status
    """
    try:
        # In a real implementation, this would check service dependencies
        return JSONResponse(
            content={
                "status": "healthy",
                "service": "insights",
                "timestamp": datetime.now().isoformat(),
                "dependencies": {
                    "neo4j": "connected",
                    "insights_processor": "operational"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Insights service health check failed: {str(e)}")
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "insights",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )
