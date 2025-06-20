"""
Comprehensive test suite for the Insights Processor Service.

This module provides complete test coverage for the Insights Processor functionality
including patient insight generation, population health analysis, treatment efficacy
assessment, risk stratification, and clinical decision support.

Tests cover:
1. Insights processor initialization and configuration
2. Patient insight generation and health summarization
3. Population health analysis and trend identification
4. Treatment efficacy assessment and outcome analysis
5. Risk stratification and assessment algorithms
6. Clinical decision support and recommendation generation
7. Comparative analysis between patients and cohorts
8. Integration with graph database and medical knowledge
9. Performance optimization and caching strategies
10. Error handling and recovery mechanisms
"""

import pytest
from unittest.mock import patch
from datetime import datetime

from app.services.insights_processor import InsightsProcessor


class TestInsightsProcessor:
    """Comprehensive test cases for the Insights Processor Service."""
    
    @pytest.fixture
    def insights_processor(self):
        """Create an insights processor instance for testing."""
        return InsightsProcessor()
    
    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for insights testing."""
        return {
            "patient_id": "patient_12345",
            "demographics": {
                "age": 65,
                "gender": "male",
                "bmi": 32.5,
                "ethnicity": "caucasian"
            },
            "medical_history": [
                {
                    "condition": "Type 2 Diabetes",
                    "icd_code": "E11.9",
                    "diagnosis_date": "2020-03-15",
                    "status": "active",
                    "severity": "moderate"
                },
                {
                    "condition": "Hypertension",
                    "icd_code": "I10",
                    "diagnosis_date": "2019-08-22",
                    "status": "controlled",
                    "severity": "mild"
                }
            ],
            "medications": [
                {
                    "name": "Metformin",
                    "dosage": "500mg",
                    "frequency": "twice daily",
                    "start_date": "2020-03-15",
                    "adherence_score": 0.95
                },
                {
                    "name": "Lisinopril",
                    "dosage": "10mg",
                    "frequency": "daily",
                    "start_date": "2019-08-22",
                    "adherence_score": 0.88
                }
            ],
            "lab_results": [
                {
                    "test": "HbA1c",
                    "value": 6.8,
                    "unit": "%",
                    "date": "2024-06-01",
                    "reference_range": "<7.0",
                    "status": "normal"
                },
                {
                    "test": "Fasting Glucose",
                    "value": 110,
                    "unit": "mg/dL",
                    "date": "2024-06-01",
                    "reference_range": "70-100",
                    "status": "slightly_elevated"
                }
            ],
            "vital_signs": [
                {
                    "type": "blood_pressure",
                    "systolic": 135,
                    "diastolic": 82,
                    "date": "2024-06-01"
                },
                {
                    "type": "weight",
                    "value": 95,
                    "unit": "kg",
                    "date": "2024-06-01"
                }
            ]
        }
    
    @pytest.fixture
    def mock_graph_client(self):
        """Mock graph client for testing."""
        with patch('app.services.insights_processor.graph_client') as mock:
            yield mock
    
    @pytest.fixture
    def mock_ai_recommendations(self):
        """Mock AI recommendations service for testing."""
        with patch('app.services.insights_processor.ai_recommendations') as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_generate_patient_insights_success(self, insights_processor, sample_patient_data, 
                                                   mock_graph_client, mock_ai_recommendations):
        """Test successful patient insight generation."""
        # Mock graph data
        mock_graph_client.get_patient_comprehensive_data.return_value = sample_patient_data
        mock_graph_client.get_patient_timeline.return_value = {
            "timeline_events": [
                {"date": "2020-03-15", "event": "diabetes_diagnosis"},
                {"date": "2024-06-01", "event": "hba1c_target_achieved"}
            ]
        }
        
        # Mock AI recommendations
        mock_ai_recommendations.generate_clinical_insights.return_value = {
            "recommendations": ["Continue current diabetes management"],
            "risk_factors": ["obesity", "family_history"],
            "predicted_outcomes": {"diabetes_control": "good"}
        }
        
        insights = await insights_processor.generate_patient_insights("patient_12345")
        
        assert insights is not None
        assert insights["patient_id"] == "patient_12345"
        assert "health_summary" in insights
        assert "clinical_insights" in insights
        assert "recommendations" in insights
        assert "predictive_analytics" in insights
        assert len(insights["clinical_insights"]) > 0

    @pytest.mark.asyncio
    async def test_generate_patient_insights_no_data(self, insights_processor, mock_graph_client):
        """Test insight generation when no patient data exists."""
        mock_graph_client.get_patient_comprehensive_data.return_value = None
        
        insights = await insights_processor.generate_patient_insights("nonexistent_patient")
        
        assert insights is None

    @pytest.mark.asyncio
    async def test_analyze_diabetes_progression(self, insights_processor, sample_patient_data):
        """Test diabetes progression analysis."""
        # Add historical HbA1c data
        hba1c_history = [
            {"date": "2020-03-15", "value": 8.5, "unit": "%"},
            {"date": "2020-09-15", "value": 7.8, "unit": "%"},
            {"date": "2021-03-15", "value": 7.2, "unit": "%"},
            {"date": "2024-06-01", "value": 6.8, "unit": "%"}
        ]
        
        progression_analysis = await insights_processor._analyze_diabetes_progression(
            sample_patient_data, hba1c_history
        )
        
        assert progression_analysis["trend"] == "improving"
        assert progression_analysis["total_reduction"] > 0
        assert progression_analysis["target_achieved"] is True
        assert "diabetes_control_excellent" in progression_analysis["status"]

    @pytest.mark.asyncio
    async def test_assess_cardiovascular_risk(self, insights_processor, sample_patient_data):
        """Test cardiovascular risk assessment."""
        risk_assessment = await insights_processor._assess_cardiovascular_risk(sample_patient_data)
        
        assert "risk_score" in risk_assessment
        assert "risk_category" in risk_assessment
        assert "contributing_factors" in risk_assessment
        assert risk_assessment["risk_score"] > 0
        assert risk_assessment["risk_category"] in ["low", "moderate", "high"]

    @pytest.mark.asyncio
    async def test_generate_clinical_recommendations(self, insights_processor, sample_patient_data, 
                                                   mock_ai_recommendations):
        """Test clinical recommendation generation."""
        mock_ai_recommendations.generate_clinical_recommendations.return_value = [
            {
                "category": "medication_adjustment",
                "recommendation": "Consider ACE inhibitor dose optimization",
                "priority": "medium",
                "evidence_level": "A"
            },
            {
                "category": "lifestyle_intervention",
                "recommendation": "Weight management program",
                "priority": "high",
                "evidence_level": "A"
            }
        ]
        
        recommendations = await insights_processor.generate_recommendations("patient_12345")
        
        assert recommendations is not None
        assert len(recommendations["recommendations"]) >= 2
        assert any(rec["category"] == "medication_adjustment" for rec in recommendations["recommendations"])
        assert any(rec["category"] == "lifestyle_intervention" for rec in recommendations["recommendations"])

    @pytest.mark.asyncio
    async def test_compare_patients_success(self, insights_processor, mock_graph_client):
        """Test successful patient comparison."""
        # Mock patient data for comparison
        mock_graph_client.get_patients_data.return_value = {
            "patient_001": {
                "hba1c": 6.5,
                "bp": "125/78",
                "medications": ["Metformin"],
                "adherence": 0.95
            },
            "patient_002": {
                "hba1c": 7.2,
                "bp": "135/85",
                "medications": ["Metformin", "Glipizide"],
                "adherence": 0.78
            },
            "patient_003": {
                "hba1c": 8.5,
                "bp": "145/92",
                "medications": ["Metformin", "Insulin"],
                "adherence": 0.65
            }
        }
        
        comparison_request = {
            "patient_ids": ["patient_001", "patient_002", "patient_003"],
            "comparison_criteria": ["diabetes_control", "medication_response"],
            "time_period": "last_12_months"
        }
        
        comparison = await insights_processor.compare_patients(comparison_request)
        
        assert comparison is not None
        assert comparison["patients_analyzed"] == 3
        assert "comparative_results" in comparison
        assert "insights" in comparison
        assert len(comparison["insights"]) > 0

    @pytest.mark.asyncio
    async def test_population_health_insights(self, insights_processor, mock_graph_client):
        """Test population health insights generation."""
        # Mock population data
        mock_graph_client.get_population_data.return_value = {
            "total_patients": 1250,
            "condition": "diabetes",
            "demographic_breakdown": {
                "age_groups": {"18-30": 45, "31-50": 380, "51-65": 520, "65+": 305},
                "gender": {"male": 675, "female": 575}
            },
            "clinical_metrics": {
                "average_hba1c": 7.4,
                "patients_at_target": 625,
                "common_medications": ["Metformin", "Insulin", "Sulfonylureas"]
            }
        }
        
        insights = await insights_processor.get_population_insights("diabetes")
        
        assert insights is not None
        assert insights["condition"] == "diabetes"
        assert insights["population_size"] == 1250
        assert "demographic_breakdown" in insights
        assert "clinical_metrics" in insights
        assert "outcome_trends" in insights

    @pytest.mark.asyncio
    async def test_treatment_efficacy_analysis(self, insights_processor, mock_graph_client):
        """Test treatment efficacy analysis."""
        # Mock treatment data
        mock_graph_client.get_treatment_outcomes.return_value = {
            "treatment": "metformin",
            "patient_cohort": "type2_diabetes",
            "total_patients": 850,
            "outcomes": {
                "hba1c_reduction": {"mean": 1.2, "median": 1.0},
                "weight_loss": {"mean": 3.5, "patients_affected": 680},
                "side_effects": {"total": 127, "discontinuation": 0.08}
            }
        }
        
        efficacy_request = {
            "treatment_type": "metformin",
            "patient_cohort": "type2_diabetes",
            "analysis_period": "2023-01-01_to_2024-06-01",
            "outcome_measures": ["hba1c_reduction", "weight_loss", "side_effects"]
        }
        
        analysis = await insights_processor.analyze_treatment_efficacy(efficacy_request)
        
        assert analysis is not None
        assert analysis["treatment"] == "metformin"
        assert analysis["total_patients"] == 850
        assert "efficacy_metrics" in analysis
        assert "subgroup_analysis" in analysis

    @pytest.mark.asyncio
    async def test_risk_stratification(self, insights_processor, sample_patient_data):
        """Test patient risk stratification."""
        risk_assessment = await insights_processor.assess_patient_risk("patient_12345")
        
        assert risk_assessment is not None
        assert "overall_risk_score" in risk_assessment
        assert "risk_category" in risk_assessment
        assert "risk_factors" in risk_assessment
        assert "specific_risk_assessments" in risk_assessment
        assert "intervention_priorities" in risk_assessment
        
        # Verify risk categories are valid
        assert risk_assessment["risk_category"] in ["low", "moderate", "high"]
        assert 0 <= risk_assessment["overall_risk_score"] <= 1

    @pytest.mark.asyncio
    async def test_predictive_analytics(self, insights_processor, sample_patient_data, mock_ai_recommendations):
        """Test predictive analytics capabilities."""
        mock_ai_recommendations.predict_outcomes.return_value = {
            "diabetes_progression_risk": 0.15,
            "cardiovascular_event_risk": 0.08,
            "hospitalization_risk_6m": 0.05,
            "medication_adherence_prediction": 0.92
        }
        
        predictions = await insights_processor._generate_predictive_analytics(sample_patient_data)
        
        assert predictions is not None
        assert "diabetes_progression_risk" in predictions
        assert "cardiovascular_event_risk" in predictions
        assert "hospitalization_risk_6m" in predictions
        assert all(isinstance(v, (int, float)) for v in predictions.values())

    @pytest.mark.asyncio
    async def test_trend_analysis(self, insights_processor):
        """Test medical trend analysis."""
        # Sample time series data
        time_series_data = {
            "hba1c": [
                {"date": "2023-01-01", "value": 8.5},
                {"date": "2023-04-01", "value": 7.8},
                {"date": "2023-07-01", "value": 7.2},
                {"date": "2024-01-01", "value": 6.9},
                {"date": "2024-06-01", "value": 6.8}
            ],
            "blood_pressure": [
                {"date": "2023-01-01", "systolic": 145, "diastolic": 95},
                {"date": "2024-06-01", "systolic": 135, "diastolic": 82}
            ]
        }
        
        trend_analysis = await insights_processor._analyze_trends(time_series_data)
        
        assert trend_analysis is not None
        assert "hba1c_trend" in trend_analysis
        assert "bp_trend" in trend_analysis
        assert trend_analysis["hba1c_trend"] in ["improving", "stable", "worsening"]

    @pytest.mark.asyncio
    async def test_care_gap_identification(self, insights_processor, sample_patient_data):
        """Test identification of care gaps."""
        care_gaps = await insights_processor._identify_care_gaps(sample_patient_data)
        
        assert isinstance(care_gaps, list)
        # Based on sample data, should identify some care gaps
        assert len(care_gaps) > 0
        
        # Verify care gap structure
        for gap in care_gaps:
            assert "category" in gap
            assert "description" in gap
            assert "priority" in gap

    @pytest.mark.asyncio
    async def test_medication_adherence_analysis(self, insights_processor, sample_patient_data):
        """Test medication adherence analysis."""
        adherence_analysis = await insights_processor._analyze_medication_adherence(sample_patient_data)
        
        assert adherence_analysis is not None
        assert "overall_adherence" in adherence_analysis
        assert "medication_specific" in adherence_analysis
        assert "adherence_factors" in adherence_analysis
        
        # Verify adherence scores are valid percentages
        assert 0 <= adherence_analysis["overall_adherence"] <= 1

    @pytest.mark.asyncio
    async def test_quality_metrics_calculation(self, insights_processor, sample_patient_data):
        """Test quality metrics calculation."""
        quality_metrics = await insights_processor._calculate_quality_metrics(sample_patient_data)
        
        assert quality_metrics is not None
        assert "diabetes_quality_indicators" in quality_metrics
        assert "preventive_care_compliance" in quality_metrics
        assert "guideline_adherence" in quality_metrics

    @pytest.mark.asyncio
    async def test_batch_insights_generation(self, insights_processor, mock_graph_client):
        """Test batch insights generation for multiple patients."""
        # Mock batch data
        mock_graph_client.get_patients_batch_data.return_value = {
            "patient_001": {"condition": "diabetes", "hba1c": 6.5},
            "patient_002": {"condition": "diabetes", "hba1c": 7.2},
            "patient_003": {"condition": "diabetes", "hba1c": 8.5}
        }
        
        batch_request = {
            "patient_ids": ["patient_001", "patient_002", "patient_003"],
            "insight_types": ["health_summary", "risk_assessment"],
            "include_comparisons": True
        }
        
        batch_results = await insights_processor.generate_batch_insights(batch_request)
        
        assert batch_results is not None
        assert batch_results["patients_processed"] == 3
        assert len(batch_results["insights_generated"]) == 3
        assert "comparative_analysis" in batch_results

    @pytest.mark.asyncio
    async def test_insights_caching(self, insights_processor, sample_patient_data, mock_graph_client):
        """Test insights caching functionality."""
        mock_graph_client.get_patient_comprehensive_data.return_value = sample_patient_data
        
        # First call should hit the database
        insights1 = await insights_processor.generate_patient_insights("patient_12345")
        
        # Second call should use cache
        insights2 = await insights_processor.generate_patient_insights("patient_12345")
        
        assert insights1 == insights2
        # Verify mock was called only once (first time)
        assert mock_graph_client.get_patient_comprehensive_data.call_count == 1

    @pytest.mark.asyncio
    async def test_error_handling_service_unavailable(self, insights_processor, mock_graph_client):
        """Test error handling when services are unavailable."""
        mock_graph_client.get_patient_comprehensive_data.side_effect = Exception("Database unavailable")
        
        insights = await insights_processor.generate_patient_insights("patient_12345")
        
        # Should handle gracefully and return None or error structure
        assert insights is None or "error" in insights

    @pytest.mark.asyncio
    async def test_performance_optimization(self, insights_processor, mock_graph_client):
        """Test performance optimization features."""
        # Mock data for performance testing
        mock_graph_client.get_patient_comprehensive_data.return_value = {
            "patient_id": "patient_perf_test",
            "large_dataset": list(range(1000))  # Simulate large dataset
        }
        
        start_time = datetime.now()
        insights = await insights_processor.generate_patient_insights("patient_perf_test")
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 5.0  # 5 seconds max
        assert insights is not None

    @pytest.mark.asyncio
    async def test_health_status_monitoring(self, insights_processor):
        """Test health status monitoring."""
        health_status = await insights_processor.get_health_status()
        
        assert health_status is not None
        assert "service" in health_status
        assert "status" in health_status
        assert "ml_models_loaded" in health_status
        assert "database_connection" in health_status
        assert health_status["service"] == "insights"

    @pytest.mark.asyncio
    async def test_insights_metrics_collection(self, insights_processor):
        """Test metrics collection for insights service."""
        # Generate some insights to create metrics
        with patch.object(insights_processor, 'generate_patient_insights') as mock_generate:
            mock_generate.return_value = {"patient_id": "test", "insights": []}
            
            await insights_processor.generate_patient_insights("test_patient")
            await insights_processor.generate_patient_insights("test_patient2")
        
        metrics = await insights_processor.get_metrics()
        
        assert metrics is not None
        assert "insights_generated_total" in metrics
        assert "average_processing_time" in metrics
        assert "success_rate" in metrics
