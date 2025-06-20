"""
Insights schemas for medical analysis and pattern recognition.

This module defines Pydantic models for insights generation, patient analysis,
and medical pattern recognition across patient populations.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class InsightType(str, Enum):
    """Types of medical insights that can be generated."""
    TREATMENT_PROGRESSION = "treatment_progression"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    TREATMENT_RESPONSE = "treatment_response"
    POPULATION_HEALTH = "population_health"
    CLINICAL_DECISION_SUPPORT = "clinical_decision_support"
    COHORT_ANALYSIS = "cohort_analysis"
    TREATMENT_EFFICACY = "treatment_efficacy"
    RISK_STRATIFICATION = "risk_stratification"


class RiskLevel(str, Enum):
    """Risk levels for patient assessments."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(str, Enum):
    """Confidence levels for insights and recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TreatmentOutcome(str, Enum):
    """Treatment outcome categories."""
    IMPROVED = "improved"
    STABLE = "stable"
    DETERIORATED = "deteriorated"
    NO_CHANGE = "no_change"
    UNKNOWN = "unknown"


class MetricValue(BaseModel):
    """Represents a metric value with temporal context."""
    value: Union[float, int, str]
    unit: Optional[str] = None
    date: datetime
    reference_range: Optional[str] = None
    status: Optional[str] = None  # normal, abnormal, critical
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TreatmentProgressionInsight(BaseModel):
    """Insights about treatment progression over time."""
    patient_id: str
    condition: str
    medication: str
    start_date: datetime
    end_date: Optional[datetime] = None
    
    # Progression metrics
    key_metrics: List[MetricValue]
    trend: str  # improving, stable, declining
    effectiveness_score: float = Field(..., ge=0.0, le=1.0)
    
    # Analysis details
    baseline_values: Dict[str, Any]
    current_values: Dict[str, Any]
    improvement_percentage: Optional[float] = None
    time_to_response: Optional[int] = None  # days
    
    # Insights
    summary: str
    recommendations: List[str]
    confidence: ConfidenceLevel
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ComparativeAnalysisInsight(BaseModel):
    """Insights from comparing multiple patients."""
    primary_patient_id: str
    comparison_patients: List[str]
    condition: str
    
    # Comparison metrics
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    outcome_comparison: Dict[str, TreatmentOutcome]
    
    # Statistical analysis
    better_outcomes_count: int
    similar_outcomes_count: int
    worse_outcomes_count: int
    
    # Insights
    key_differences: List[str]
    success_factors: List[str]
    recommendations: List[str]
    confidence: ConfidenceLevel


class RiskFactorInsight(BaseModel):
    """Risk factor identification and assessment."""
    patient_id: str
    risk_factors: List[Dict[str, Any]]
    overall_risk_level: RiskLevel
    risk_score: float = Field(..., ge=0.0, le=1.0)
    
    # Risk breakdown
    modifiable_risks: List[str]
    non_modifiable_risks: List[str]
    
    # Predictions
    condition_probabilities: Dict[str, float]
    time_horizon: str  # "1 year", "5 years", etc.
    
    # Recommendations
    preventive_measures: List[str]
    monitoring_recommendations: List[str]
    confidence: ConfidenceLevel


class TreatmentResponsePattern(BaseModel):
    """Pattern analysis for treatment responses."""
    condition: str
    treatment: str
    patient_cohort_size: int
    
    # Response patterns
    response_rate: float = Field(..., ge=0.0, le=1.0)
    average_response_time: int  # days
    side_effect_rate: float = Field(..., ge=0.0, le=1.0)
    
    # Outcome distribution
    outcome_distribution: Dict[TreatmentOutcome, int]
    
    # Factors affecting response
    positive_predictors: List[str]
    negative_predictors: List[str]
    
    # Insights
    summary: str
    recommendations: List[str]
    confidence: ConfidenceLevel


class PopulationHealthInsight(BaseModel):
    """Population-level health insights and trends."""
    condition: str
    population_size: int
    time_period: str
    
    # Prevalence and trends
    prevalence_rate: float
    trend_direction: str  # increasing, stable, decreasing
    demographic_breakdown: Dict[str, Any]
    
    # Treatment patterns
    common_treatments: List[Dict[str, Any]]
    treatment_effectiveness: Dict[str, float]
    
    # Risk factors
    common_risk_factors: List[str]
    protective_factors: List[str]
    
    # Insights
    key_findings: List[str]
    public_health_recommendations: List[str]
    confidence: ConfidenceLevel


class ClinicalRecommendation(BaseModel):
    """Clinical decision support recommendation."""
    recommendation_id: str
    patient_id: str
    recommendation_type: str
    priority: str  # high, medium, low
    
    # Recommendation details
    title: str
    description: str
    rationale: str
    evidence_level: str  # A, B, C, D
    
    # Implementation
    suggested_actions: List[str]
    contraindications: List[str]
    monitoring_requirements: List[str]
    
    # Metadata
    confidence: ConfidenceLevel
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Request/Response Models

class PatientInsightRequest(BaseModel):
    """Request for generating patient-specific insights."""
    patient_id: str
    insight_types: List[InsightType]
    time_period: Optional[str] = "1 year"  # last 1 year, last 6 months, etc.
    include_predictions: bool = True
    include_recommendations: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "patient_12345",
                "insight_types": ["treatment_progression", "risk_assessment"],
                "time_period": "1 year",
                "include_predictions": True,
                "include_recommendations": True
            }
        }


class PatientInsightResponse(BaseModel):
    """Response containing patient insights."""
    message: str
    patient_id: str
    insights: Dict[str, Any]
    
    # Metadata
    generated_at: datetime
    processing_time: float
    insights_count: int
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PatientComparisonRequest(BaseModel):
    """Request for comparing multiple patients."""
    primary_patient_id: str
    comparison_criteria: Dict[str, Any]
    max_comparisons: int = Field(default=10, ge=1, le=50)
    include_demographics: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "primary_patient_id": "patient_12345",
                "comparison_criteria": {
                    "condition": "Type 2 Diabetes",
                    "age_range": [45, 65],
                    "gender": "any"
                },
                "max_comparisons": 10,
                "include_demographics": True
            }
        }


class PatientComparisonResponse(BaseModel):
    """Response containing patient comparison results."""
    message: str
    primary_patient_id: str
    comparison_results: ComparativeAnalysisInsight
    
    # Metadata
    generated_at: datetime
    processing_time: float
    patients_compared: int
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PopulationInsightRequest(BaseModel):
    """Request for population-level insights."""
    condition: str
    time_period: str = "1 year"
    demographic_filters: Optional[Dict[str, Any]] = None
    include_trends: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "condition": "Type 2 Diabetes",
                "time_period": "2 years",
                "demographic_filters": {
                    "age_range": [18, 80],
                    "gender": "any"
                },
                "include_trends": True
            }
        }


class PopulationInsightResponse(BaseModel):
    """Response containing population insights."""
    message: str
    population_insights: PopulationHealthInsight
    
    # Metadata
    generated_at: datetime
    processing_time: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ClinicalRecommendationRequest(BaseModel):
    """Request for clinical recommendations."""
    patient_id: str
    focus_areas: Optional[List[str]] = None  # specific conditions, treatments
    urgency_level: str = "routine"  # urgent, routine, preventive
    include_evidence: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "patient_12345",
                "focus_areas": ["diabetes_management", "cardiovascular_risk"],
                "urgency_level": "routine",
                "include_evidence": True
            }
        }


class ClinicalRecommendationResponse(BaseModel):
    """Response containing clinical recommendations."""
    message: str
    patient_id: str
    recommendations: List[ClinicalRecommendation]
    
    # Metadata
    generated_at: datetime
    processing_time: float
    recommendations_count: int
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class InsightStatus(BaseModel):
    """Status of insight generation process."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float = Field(..., ge=0.0, le=100.0)
    
    # Processing details
    current_step: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Results
    insights_generated: int = 0
    total_insights_requested: int = 0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class InsightMetrics(BaseModel):
    """Metrics about insight generation performance."""
    total_insights_generated: int
    average_processing_time: float
    success_rate: float = Field(..., ge=0.0, le=1.0)
    
    # Breakdown by type
    insights_by_type: Dict[InsightType, int]
    
    # Quality metrics
    average_confidence: float = Field(..., ge=0.0, le=1.0)
    high_confidence_rate: float = Field(..., ge=0.0, le=1.0)
    
    # System metrics
    last_updated: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
