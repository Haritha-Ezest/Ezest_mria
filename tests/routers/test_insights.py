"""
Comprehensive tests for the insights router endpoints.

This module tests all endpoints in the insights router, including:
- Patient insight generation and analysis
- Population health analysis and trends
- Treatment efficacy assessment
- Risk stratification and assessment
- Clinical decision support recommendations
- Comparative analysis between patients
- Integration with graph database and medical knowledge
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from datetime import datetime

from app.main import app


class TestInsightsRouter:
    """Test class for insights router endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_insights_processor(self):
        """Mock insights processor for testing."""
        with patch('app.routers.insights.insights_processor') as mock:
            yield mock

    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for insights testing."""
        return {
            "patient_id": "patient_12345",
            "demographics": {
                "age": 65,
                "gender": "male",
                "bmi": 32.5
            },
            "medical_history": [
                {
                    "condition": "Type 2 Diabetes",
                    "diagnosis_date": "2020-03-15",
                    "status": "active"
                },
                {
                    "condition": "Hypertension", 
                    "diagnosis_date": "2019-08-22",
                    "status": "controlled"
                }
            ],
            "medications": [
                {
                    "name": "Metformin",
                    "dosage": "500mg",
                    "frequency": "twice daily",
                    "start_date": "2020-03-15"
                },
                {
                    "name": "Lisinopril",
                    "dosage": "10mg", 
                    "frequency": "daily",
                    "start_date": "2019-08-22"
                }
            ],
            "lab_results": [
                {
                    "test": "HbA1c",
                    "value": 6.8,
                    "unit": "%",
                    "date": "2024-06-01",
                    "reference_range": "<7.0"
                },
                {
                    "test": "Blood Pressure",
                    "value": "135/82",
                    "unit": "mmHg",
                    "date": "2024-06-01"
                }
            ]
        }

    def test_generate_patient_insights_success(self, client, mock_insights_processor, sample_patient_data):
        """Test successful patient insight generation."""
        mock_insights = {
            "patient_id": "patient_12345",
            "insights_generated": datetime.now().isoformat(),
            "health_summary": {
                "overall_status": "stable_improvement",
                "key_conditions": ["Type 2 Diabetes", "Hypertension"],
                "treatment_response": "positive",
                "risk_level": "moderate"
            },
            "clinical_insights": [
                {
                    "category": "diabetes_management",
                    "insight": "Patient shows excellent diabetes control with HbA1c at target level",
                    "confidence": 0.95,
                    "supporting_data": ["HbA1c: 6.8%", "Medication adherence: High"]
                },
                {
                    "category": "cardiovascular_risk",
                    "insight": "Blood pressure slightly elevated despite medication",
                    "confidence": 0.88,
                    "supporting_data": ["BP: 135/82 mmHg", "Target: <130/80"]
                }
            ],
            "recommendations": [
                {
                    "priority": "high",
                    "category": "lifestyle",
                    "recommendation": "Consider dietary consultation for blood pressure management",
                    "rationale": "Current BP readings suggest need for additional lifestyle interventions"
                },
                {
                    "priority": "medium",
                    "category": "monitoring",
                    "recommendation": "Continue quarterly HbA1c monitoring",
                    "rationale": "Maintain current excellent diabetes control"
                }
            ],
            "predictive_analytics": {
                "diabetes_progression_risk": "low",
                "cardiovascular_event_risk": "moderate",
                "medication_adherence_prediction": "high",
                "hospitalization_risk_6m": "low"
            },
            "trend_analysis": {
                "hba1c_trend": "stable_optimal",
                "bp_trend": "slightly_increasing",
                "medication_effectiveness": "good"
            }
        }
        
        mock_insights_processor.generate_patient_insights.return_value = mock_insights
        
        response = client.post("/insights/generate/patient_12345")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["patient_id"] == "patient_12345"
        assert data["health_summary"]["overall_status"] == "stable_improvement"
        assert len(data["clinical_insights"]) == 2
        assert len(data["recommendations"]) == 2
        assert "diabetes_progression_risk" in data["predictive_analytics"]

    def test_generate_patient_insights_not_found(self, client, mock_insights_processor):
        """Test patient insights for non-existent patient."""
        mock_insights_processor.generate_patient_insights.return_value = None
        
        response = client.post("/insights/generate/nonexistent_patient")
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    def test_compare_patients_success(self, client, mock_insights_processor):
        """Test successful patient comparison."""
        comparison_request = {
            "patient_ids": ["patient_001", "patient_002", "patient_003"],
            "comparison_criteria": [
                "diabetes_control",
                "medication_response",
                "cardiovascular_risk"
            ],
            "time_period": "last_12_months"
        }
        
        mock_comparison = {
            "comparison_id": "comp_456",
            "patients_analyzed": 3,
            "comparison_date": datetime.now().isoformat(),
            "criteria_analyzed": ["diabetes_control", "medication_response", "cardiovascular_risk"],
            "comparative_results": {
                "diabetes_control": {
                    "patient_001": {"hba1c": 6.5, "control_level": "excellent"},
                    "patient_002": {"hba1c": 7.2, "control_level": "good"},
                    "patient_003": {"hba1c": 8.5, "control_level": "needs_improvement"}
                },
                "medication_response": {
                    "patient_001": {"response_score": 0.92, "adherence": "high"},
                    "patient_002": {"response_score": 0.78, "adherence": "moderate"},
                    "patient_003": {"response_score": 0.65, "adherence": "low"}
                },
                "cardiovascular_risk": {
                    "patient_001": {"risk_score": "low", "framingham_score": 8.2},
                    "patient_002": {"risk_score": "moderate", "framingham_score": 15.7},
                    "patient_003": {"risk_score": "high", "framingham_score": 22.8}
                }
            },
            "insights": [
                "Patient 001 shows superior diabetes control compared to cohort",
                "Medication adherence correlates strongly with treatment outcomes",
                "Patient 003 requires intensive intervention for cardiovascular risk reduction"
            ],
            "recommendations": {
                "patient_001": ["Maintain current regimen", "Continue monitoring"],
                "patient_002": ["Consider medication adjustment", "Improve adherence support"],
                "patient_003": ["Immediate intensive therapy", "Cardiovascular specialist referral"]
            }
        }
        
        mock_insights_processor.compare_patients.return_value = mock_comparison
        
        response = client.post("/insights/compare", json=comparison_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["patients_analyzed"] == 3
        assert len(data["criteria_analyzed"]) == 3
        assert "patient_001" in data["comparative_results"]["diabetes_control"]
        assert len(data["insights"]) == 3

    def test_population_health_insights_success(self, client, mock_insights_processor):
        """Test population health insights for specific condition."""
        mock_population_insights = {
            "condition": "diabetes",
            "analysis_date": datetime.now().isoformat(),
            "population_size": 1250,
            "demographic_breakdown": {
                "age_groups": {
                    "18-30": 45,
                    "31-50": 380,
                    "51-65": 520,
                    "65+": 305
                },
                "gender_distribution": {
                    "male": 675,
                    "female": 575
                }
            },
            "clinical_metrics": {
                "average_hba1c": 7.4,
                "patients_at_target": 625,
                "target_achievement_rate": 0.50,
                "common_medications": [
                    {"name": "Metformin", "usage_rate": 0.85},
                    {"name": "Insulin", "usage_rate": 0.32},
                    {"name": "Sulfonylureas", "usage_rate": 0.28}
                ]
            },
            "outcome_trends": {
                "hospitalization_rate": 0.08,
                "emergency_visits": 0.15,
                "medication_adherence": 0.72,
                "complication_rate": 0.23
            },
            "risk_stratification": {
                "low_risk": 425,
                "moderate_risk": 580,
                "high_risk": 245
            },
            "treatment_patterns": [
                "Metformin remains first-line therapy in 85% of cases",
                "Insulin usage increases significantly in patients over 65",
                "Combination therapy is common in moderate-to-high risk patients"
            ],
            "quality_indicators": {
                "annual_eye_exams": 0.78,
                "foot_exams": 0.65,
                "nephropathy_screening": 0.82,
                "lipid_monitoring": 0.89
            }
        }
        
        mock_insights_processor.get_population_insights.return_value = mock_population_insights
        
        response = client.get("/insights/population/diabetes")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["condition"] == "diabetes"
        assert data["population_size"] == 1250
        assert data["clinical_metrics"]["target_achievement_rate"] == 0.50
        assert len(data["treatment_patterns"]) == 3

    def test_clinical_recommendations_success(self, client, mock_insights_processor):
        """Test clinical recommendation generation."""
        mock_recommendations = {
            "patient_id": "patient_12345",
            "recommendation_date": datetime.now().isoformat(),
            "clinical_context": {
                "primary_conditions": ["Type 2 Diabetes", "Hypertension"],
                "current_medications": ["Metformin 500mg BID", "Lisinopril 10mg daily"],
                "recent_vitals": {"BP": "135/82", "BMI": 32.5, "HbA1c": 6.8}
            },
            "recommendations": [
                {
                    "category": "medication_adjustment",
                    "priority": "medium",
                    "recommendation": "Consider increasing Lisinopril to 20mg daily",
                    "rationale": "Blood pressure remains above target despite current therapy",
                    "evidence_level": "A",
                    "guideline_reference": "AHA/ACC 2017 Hypertension Guidelines",
                    "potential_benefits": ["Improved BP control", "Reduced cardiovascular risk"],
                    "monitoring_required": ["Serum creatinine", "Potassium levels"]
                },
                {
                    "category": "lifestyle_intervention",
                    "priority": "high",
                    "recommendation": "Structured weight management program",
                    "rationale": "BMI >30 contributes to both diabetes and hypertension management challenges",
                    "evidence_level": "A",
                    "potential_benefits": ["Improved glycemic control", "BP reduction", "Reduced medication needs"],
                    "implementation": ["Dietitian referral", "Exercise program", "Behavioral counseling"]
                },
                {
                    "category": "preventive_care",
                    "priority": "medium",
                    "recommendation": "Annual ophthalmologic examination",
                    "rationale": "Diabetic retinopathy screening per ADA guidelines",
                    "evidence_level": "A",
                    "next_due_date": "2024-03-15"
                }
            ],
            "care_gaps": [
                "Diabetic foot examination overdue (last: 2023-09-15)",
                "Nephropathy screening pending (microalbumin)",
                "Lipid panel due for annual monitoring"
            ],
            "drug_interactions": [],
            "contraindications": [],
            "follow_up_plan": {
                "next_visit": "3 months",
                "monitoring_labs": ["HbA1c", "Basic metabolic panel"],
                "vital_sign_tracking": ["Blood pressure", "Weight"]
            }
        }
        
        mock_insights_processor.generate_recommendations.return_value = mock_recommendations
        
        response = client.post("/insights/recommendations/patient_12345")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["patient_id"] == "patient_12345"
        assert len(data["recommendations"]) == 3
        assert len(data["care_gaps"]) == 3
        assert data["follow_up_plan"]["next_visit"] == "3 months"

    def test_risk_assessment_success(self, client, mock_insights_processor):
        """Test comprehensive risk assessment."""
        mock_risk_assessment = {
            "patient_id": "patient_12345",
            "assessment_date": datetime.now().isoformat(),
            "overall_risk_score": 0.35,
            "risk_category": "moderate",
            "risk_factors": {
                "modifiable": [
                    {"factor": "obesity", "impact": "high", "current_status": "BMI 32.5"},
                    {"factor": "hypertension", "impact": "high", "current_status": "135/82 mmHg"},
                    {"factor": "smoking", "impact": "high", "current_status": "former smoker"},
                    {"factor": "physical_inactivity", "impact": "medium", "current_status": "sedentary"}
                ],
                "non_modifiable": [
                    {"factor": "age", "impact": "medium", "current_status": "65 years"},
                    {"factor": "gender", "impact": "low", "current_status": "male"},
                    {"factor": "family_history", "impact": "medium", "current_status": "diabetes, CAD"}
                ]
            },
            "specific_risk_assessments": {
                "cardiovascular": {
                    "10_year_risk": 0.18,
                    "framingham_score": 15.2,
                    "risk_category": "moderate",
                    "primary_contributors": ["age", "hypertension", "diabetes"]
                },
                "diabetic_complications": {
                    "retinopathy_risk": "low",
                    "nephropathy_risk": "moderate", 
                    "neuropathy_risk": "low",
                    "macrovascular_risk": "moderate"
                },
                "hospitalization": {
                    "6_month_risk": 0.08,
                    "1_year_risk": 0.15,
                    "primary_predictors": ["uncontrolled_hypertension", "medication_adherence"]
                }
            },
            "intervention_priorities": [
                {
                    "priority": 1,
                    "intervention": "Blood pressure optimization",
                    "potential_risk_reduction": 0.15,
                    "evidence_level": "A"
                },
                {
                    "priority": 2,
                    "intervention": "Weight management",
                    "potential_risk_reduction": 0.12,
                    "evidence_level": "A"
                }
            ],
            "monitoring_recommendations": [
                "Monthly blood pressure checks until target achieved",
                "Quarterly HbA1c monitoring",
                "Annual comprehensive metabolic panel",
                "Biannual ophthalmologic examination"
            ]
        }
        
        mock_insights_processor.assess_patient_risk.return_value = mock_risk_assessment
        
        response = client.get("/insights/risk/patient_12345")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["patient_id"] == "patient_12345"
        assert data["overall_risk_score"] == 0.35
        assert data["risk_category"] == "moderate"
        assert len(data["risk_factors"]["modifiable"]) == 4
        assert len(data["intervention_priorities"]) == 2

    def test_treatment_efficacy_analysis(self, client, mock_insights_processor):
        """Test treatment efficacy analysis."""
        efficacy_request = {
            "treatment_type": "metformin",
            "patient_cohort": "type2_diabetes",
            "analysis_period": "2023-01-01_to_2024-06-01",
            "outcome_measures": ["hba1c_reduction", "weight_loss", "side_effects"]
        }
        
        mock_efficacy_analysis = {
            "treatment": "metformin",
            "analysis_period": "2023-01-01 to 2024-06-01",
            "patient_cohort": "type2_diabetes",
            "total_patients": 850,
            "efficacy_metrics": {
                "hba1c_reduction": {
                    "mean_reduction": 1.2,
                    "median_reduction": 1.0,
                    "patients_achieving_target": 425,
                    "target_achievement_rate": 0.50
                },
                "weight_loss": {
                    "mean_loss_kg": 3.5,
                    "patients_losing_weight": 680,
                    "significant_loss_rate": 0.35
                },
                "side_effects": {
                    "total_reported": 127,
                    "gastrointestinal": 95,
                    "discontinuation_rate": 0.08
                }
            },
            "subgroup_analysis": {
                "by_age": {
                    "under_50": {"efficacy": 0.62, "tolerability": 0.95},
                    "50_65": {"efficacy": 0.48, "tolerability": 0.92},
                    "over_65": {"efficacy": 0.41, "tolerability": 0.88}
                },
                "by_baseline_hba1c": {
                    "7_8": {"reduction": 0.8},
                    "8_9": {"reduction": 1.2},
                    "over_9": {"reduction": 1.8}
                }
            },
            "predictive_factors": [
                "Higher baseline HbA1c predicts greater reduction",
                "Younger patients show better overall response",
                "BMI >35 associated with enhanced weight loss"
            ],
            "recommendations": [
                "Consider dose escalation for patients with HbA1c >8.5%",
                "Monitor closely for GI side effects in first 3 months",
                "Excellent first-line choice for overweight patients"
            ]
        }
        
        mock_insights_processor.analyze_treatment_efficacy.return_value = mock_efficacy_analysis
        
        response = client.post("/insights/efficacy", json=efficacy_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["treatment"] == "metformin"
        assert data["total_patients"] == 850
        assert data["efficacy_metrics"]["hba1c_reduction"]["target_achievement_rate"] == 0.50
        assert len(data["predictive_factors"]) == 3

    def test_insights_health_check(self, client, mock_insights_processor):
        """Test insights service health check."""
        mock_health = {
            "service": "insights",
            "status": "healthy",
            "database_connection": True,
            "ml_models_loaded": True,
            "graph_integration": True,
            "cache_status": "operational",
            "last_model_update": "2024-06-01T10:00:00Z",
            "insights_generated_today": 347,
            "average_processing_time": 2.8
        }
        
        mock_insights_processor.get_health_status.return_value = mock_health
        
        response = client.get("/insights/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["service"] == "insights"
        assert data["status"] == "healthy"
        assert data["database_connection"] is True
        assert data["ml_models_loaded"] is True
        assert data["insights_generated_today"] == 347

    def test_batch_insights_generation(self, client, mock_insights_processor):
        """Test batch insights generation for multiple patients."""
        batch_request = {
            "patient_ids": ["patient_001", "patient_002", "patient_003"],
            "insight_types": ["health_summary", "risk_assessment", "recommendations"],
            "include_comparisons": True
        }
        
        mock_batch_results = {
            "batch_id": "batch_789",
            "processing_date": datetime.now().isoformat(),
            "patients_processed": 3,
            "insights_generated": [
                {
                    "patient_id": "patient_001",
                    "status": "completed",
                    "processing_time": 3.2,
                    "insights_count": 15
                },
                {
                    "patient_id": "patient_002", 
                    "status": "completed",
                    "processing_time": 2.8,
                    "insights_count": 12
                },
                {
                    "patient_id": "patient_003",
                    "status": "completed",
                    "processing_time": 4.1,
                    "insights_count": 18
                }
            ],
            "comparative_analysis": {
                "similar_patients": ["patient_001", "patient_002"],
                "outlier_patients": ["patient_003"],
                "common_patterns": ["diabetes_management", "hypertension_control"]
            },
            "summary_statistics": {
                "total_insights": 45,
                "average_processing_time": 3.4,
                "success_rate": 1.0
            }
        }
        
        mock_insights_processor.generate_batch_insights.return_value = mock_batch_results
        
        response = client.post("/insights/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["patients_processed"] == 3
        assert len(data["insights_generated"]) == 3
        assert data["summary_statistics"]["success_rate"] == 1.0
