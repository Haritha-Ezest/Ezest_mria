"""
Comprehensive test suite for the AI Recommendations Service.

This module provides complete test coverage for the AI Recommendations functionality
including clinical decision support, treatment recommendations, risk assessment,
and integration with medical knowledge bases and clinical guidelines.

Tests cover:
1. AI recommendations service initialization and configuration
2. Clinical decision support and treatment recommendations
3. Drug interaction checking and contraindication analysis
4. Risk assessment and stratification algorithms
5. Evidence-based recommendation generation
6. Integration with medical knowledge bases and guidelines
7. Personalized treatment optimization
8. Population health recommendations
9. Performance optimization and caching
10. Error handling and fallback mechanisms
"""

import pytest
from unittest.mock import patch

from app.services.ai_recommendations import AIRecommendationsService


class TestAIRecommendationsService:
    """Comprehensive test cases for the AI Recommendations Service."""
    
    @pytest.fixture
    def ai_recommendations(self):
        """Create an AI recommendations service instance for testing."""
        return AIRecommendationsService()
    
    @pytest.fixture
    def sample_patient_profile(self):
        """Sample patient profile for recommendations testing."""
        return {
            "patient_id": "patient_12345",
            "age": 65,
            "gender": "male",
            "weight": 95,  # kg
            "height": 175,  # cm
            "bmi": 31.0,
            "allergies": ["penicillin", "shellfish"],
            "medical_history": [
                {
                    "condition": "Type 2 Diabetes",
                    "diagnosis_date": "2020-03-15",
                    "status": "active",
                    "severity": "moderate"
                },
                {
                    "condition": "Hypertension",
                    "diagnosis_date": "2019-08-22",
                    "status": "controlled",
                    "severity": "mild"
                },
                {
                    "condition": "Hyperlipidemia",
                    "diagnosis_date": "2021-01-10",
                    "status": "active",
                    "severity": "moderate"
                }
            ],
            "current_medications": [
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
            "recent_labs": [
                {
                    "test": "HbA1c",
                    "value": 7.8,
                    "unit": "%",
                    "date": "2024-06-01",
                    "reference_range": "<7.0"
                },
                {
                    "test": "LDL Cholesterol",
                    "value": 145,
                    "unit": "mg/dL",
                    "date": "2024-06-01",
                    "reference_range": "<100"
                }
            ],
            "vital_signs": {
                "blood_pressure": "135/82",
                "heart_rate": 78,
                "weight": 95,
                "bmi": 31.0
            }
        }

    @pytest.mark.asyncio
    async def test_generate_clinical_recommendations_success(self, ai_recommendations, sample_patient_profile):
        """Test successful generation of clinical recommendations."""
        with patch.object(ai_recommendations, '_analyze_patient_data') as mock_analyze:
            mock_analyze.return_value = {
                "risk_factors": ["obesity", "uncontrolled_diabetes", "elevated_ldl"],
                "treatment_gaps": ["statin_therapy", "diabetes_optimization"],
                "guideline_recommendations": ["ada_diabetes_guidelines", "acc_lipid_guidelines"]
            }
            
            recommendations = await ai_recommendations.generate_clinical_recommendations(sample_patient_profile)
            
            assert recommendations is not None
            assert "recommendations" in recommendations
            assert len(recommendations["recommendations"]) > 0
            
            # Should include medication recommendations
            med_recs = [r for r in recommendations["recommendations"] if r["category"] == "medication"]
            assert len(med_recs) > 0
            
            # Should include lifestyle recommendations
            lifestyle_recs = [r for r in recommendations["recommendations"] if r["category"] == "lifestyle"]
            assert len(lifestyle_recs) > 0

    @pytest.mark.asyncio
    async def test_check_drug_interactions(self, ai_recommendations):
        """Test drug interaction checking."""
        medications = [
            {"name": "Warfarin", "dosage": "5mg"},
            {"name": "Aspirin", "dosage": "81mg"},
            {"name": "Metformin", "dosage": "500mg"}
        ]
        
        with patch.object(ai_recommendations, '_get_drug_interaction_data') as mock_interactions:
            mock_interactions.return_value = [
                {
                    "drug1": "Warfarin",
                    "drug2": "Aspirin",
                    "interaction_type": "major",
                    "severity": "high",
                    "description": "Increased bleeding risk",
                    "clinical_significance": "Monitor INR closely"
                }
            ]
            
            interactions = await ai_recommendations.check_drug_interactions(medications)
            
            assert len(interactions) > 0
            assert interactions[0]["severity"] == "high"
            assert "bleeding" in interactions[0]["description"].lower()

    @pytest.mark.asyncio
    async def test_assess_contraindications(self, ai_recommendations, sample_patient_profile):
        """Test contraindication assessment for medications."""
        proposed_medication = {
            "name": "Metformin",
            "dosage": "1000mg",
            "frequency": "twice daily"
        }
        
        # Mock kidney function data that would contraindicate metformin
        with patch.object(ai_recommendations, '_get_patient_lab_values') as mock_labs:
            mock_labs.return_value = {
                "creatinine": 2.5,  # Elevated creatinine
                "egfr": 25  # Low eGFR
            }
            
            contraindications = await ai_recommendations.assess_contraindications(
                proposed_medication, 
                sample_patient_profile
            )
            
            assert len(contraindications) > 0
            # Should flag kidney function concern
            kidney_contraindications = [c for c in contraindications if "kidney" in c["reason"].lower() or "renal" in c["reason"].lower()]
            assert len(kidney_contraindications) > 0

    @pytest.mark.asyncio
    async def test_optimize_diabetes_treatment(self, ai_recommendations, sample_patient_profile):
        """Test diabetes treatment optimization recommendations."""
        optimization = await ai_recommendations.optimize_diabetes_treatment(sample_patient_profile)
        
        assert optimization is not None
        assert "current_hba1c" in optimization
        assert "target_hba1c" in optimization
        assert "recommendations" in optimization
        
        # Since HbA1c is 7.8% (above target), should recommend optimization
        assert optimization["current_hba1c"] == 7.8
        assert optimization["target_hba1c"] < 7.0
        assert len(optimization["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_cardiovascular_risk_assessment(self, ai_recommendations, sample_patient_profile):
        """Test cardiovascular risk assessment and recommendations."""
        cv_assessment = await ai_recommendations.assess_cardiovascular_risk(sample_patient_profile)
        
        assert cv_assessment is not None
        assert "risk_score" in cv_assessment
        assert "risk_category" in cv_assessment
        assert "contributing_factors" in cv_assessment
        assert "recommendations" in cv_assessment
        
        # Patient has multiple CV risk factors
        assert cv_assessment["risk_score"] > 0.1  # Should have elevated risk
        assert cv_assessment["risk_category"] in ["low", "moderate", "high"]

    @pytest.mark.asyncio
    async def test_medication_adherence_recommendations(self, ai_recommendations):
        """Test medication adherence improvement recommendations."""
        adherence_data = {
            "patient_id": "patient_12345",
            "medications": [
                {
                    "name": "Metformin",
                    "prescribed_frequency": "twice daily",
                    "actual_adherence": 0.65,  # Poor adherence
                    "barriers": ["forgetfulness", "side_effects"]
                }
            ]
        }
        
        adherence_recs = await ai_recommendations.generate_adherence_recommendations(adherence_data)
        
        assert adherence_recs is not None
        assert "recommendations" in adherence_recs
        assert len(adherence_recs["recommendations"]) > 0
        
        # Should address identified barriers
        rec_text = " ".join([r["recommendation"] for r in adherence_recs["recommendations"]])
        assert "reminder" in rec_text.lower() or "forgetfulness" in rec_text.lower()

    @pytest.mark.asyncio
    async def test_preventive_care_recommendations(self, ai_recommendations, sample_patient_profile):
        """Test preventive care recommendations."""
        preventive_recs = await ai_recommendations.generate_preventive_care_recommendations(sample_patient_profile)
        
        assert preventive_recs is not None
        assert "screenings" in preventive_recs
        assert "immunizations" in preventive_recs
        assert "counseling" in preventive_recs
        
        # For diabetic patient, should recommend diabetic-specific screenings
        diabetic_screenings = [s for s in preventive_recs["screenings"] 
                             if "diabetic" in s["name"].lower() or "retinal" in s["name"].lower()]
        assert len(diabetic_screenings) > 0

    @pytest.mark.asyncio
    async def test_dosage_optimization(self, ai_recommendations, sample_patient_profile):
        """Test medication dosage optimization."""
        medication = {
            "name": "Lisinopril",
            "current_dosage": "10mg",
            "frequency": "daily"
        }
        
        optimization = await ai_recommendations.optimize_dosage(medication, sample_patient_profile)
        
        assert optimization is not None
        assert "current_dosage" in optimization
        assert "recommended_dosage" in optimization
        assert "rationale" in optimization
        
        # Given BP is 135/82 (elevated), might recommend dose increase
        if optimization["recommended_dosage"] != optimization["current_dosage"]:
            assert "blood pressure" in optimization["rationale"].lower()

    @pytest.mark.asyncio
    async def test_evidence_based_recommendations(self, ai_recommendations, sample_patient_profile):
        """Test evidence-based recommendation generation."""
        with patch.object(ai_recommendations, '_get_clinical_guidelines') as mock_guidelines:
            mock_guidelines.return_value = {
                "ada_diabetes_guidelines": {
                    "hba1c_target": "<7.0%",
                    "first_line_therapy": "metformin",
                    "second_line_options": ["sulfonylurea", "dpp4_inhibitor", "sglt2_inhibitor"]
                }
            }
            
            recommendations = await ai_recommendations.generate_evidence_based_recommendations(
                condition="diabetes",
                patient_profile=sample_patient_profile
            )
            
            assert recommendations is not None
            assert "guidelines_consulted" in recommendations
            assert "evidence_level" in recommendations
            assert len(recommendations["recommendations"]) > 0
            
            # Should reference appropriate guidelines
            assert "ada" in recommendations["guidelines_consulted"][0].lower()

    @pytest.mark.asyncio
    async def test_personalized_treatment_recommendations(self, ai_recommendations, sample_patient_profile):
        """Test personalized treatment recommendations based on patient factors."""
        personalized_recs = await ai_recommendations.generate_personalized_recommendations(sample_patient_profile)
        
        assert personalized_recs is not None
        assert "patient_factors_considered" in personalized_recs
        assert "personalization_score" in personalized_recs
        assert "recommendations" in personalized_recs
        
        # Should consider patient-specific factors
        factors = personalized_recs["patient_factors_considered"]
        assert "age" in factors
        assert "comorbidities" in factors
        assert "current_medications" in factors

    @pytest.mark.asyncio
    async def test_drug_allergy_checking(self, ai_recommendations, sample_patient_profile):
        """Test checking for drug allergies and cross-reactions."""
        proposed_medication = {
            "name": "Amoxicillin",  # Penicillin-based antibiotic
            "dosage": "500mg"
        }
        
        allergy_check = await ai_recommendations.check_drug_allergies(
            proposed_medication,
            sample_patient_profile["allergies"]
        )
        
        assert allergy_check is not None
        assert "allergy_risk" in allergy_check
        assert "cross_reactions" in allergy_check
        
        # Should flag penicillin allergy risk
        assert allergy_check["allergy_risk"] == "high"
        assert len(allergy_check["cross_reactions"]) > 0

    @pytest.mark.asyncio
    async def test_population_health_recommendations(self, ai_recommendations):
        """Test population health recommendations."""
        population_data = {
            "condition": "diabetes",
            "total_patients": 1000,
            "demographics": {
                "age_distribution": {"18-30": 50, "31-50": 300, "51-65": 400, "65+": 250},
                "gender": {"male": 520, "female": 480}
            },
            "clinical_metrics": {
                "average_hba1c": 7.8,
                "patients_at_target": 400,
                "common_medications": ["metformin", "insulin", "sulfonylureas"]
            }
        }
        
        pop_recommendations = await ai_recommendations.generate_population_recommendations(population_data)
        
        assert pop_recommendations is not None
        assert "quality_improvement_opportunities" in pop_recommendations
        assert "population_interventions" in pop_recommendations
        assert "outcome_predictions" in pop_recommendations

    @pytest.mark.asyncio
    async def test_clinical_decision_support(self, ai_recommendations, sample_patient_profile):
        """Test comprehensive clinical decision support."""
        clinical_scenario = {
            "presenting_symptoms": ["chest_pain", "shortness_of_breath"],
            "duration": "2_hours",
            "severity": "moderate_to_severe",
            "associated_symptoms": ["diaphoresis", "nausea"]
        }
        
        decision_support = await ai_recommendations.provide_clinical_decision_support(
            clinical_scenario,
            sample_patient_profile
        )
        
        assert decision_support is not None
        assert "differential_diagnosis" in decision_support
        assert "recommended_workup" in decision_support
        assert "immediate_actions" in decision_support
        assert "risk_stratification" in decision_support

    @pytest.mark.asyncio
    async def test_medication_monitoring_recommendations(self, ai_recommendations, sample_patient_profile):
        """Test medication monitoring recommendations."""
        monitoring_recs = await ai_recommendations.generate_monitoring_recommendations(sample_patient_profile)
        
        assert monitoring_recs is not None
        assert "lab_monitoring" in monitoring_recs
        assert "clinical_monitoring" in monitoring_recs
        assert "follow_up_schedule" in monitoring_recs
        
        # For metformin, should recommend kidney function monitoring
        metformin_monitoring = [m for m in monitoring_recs["lab_monitoring"] 
                               if "metformin" in m.get("medication", "").lower()]
        assert len(metformin_monitoring) > 0

    @pytest.mark.asyncio
    async def test_recommendations_priority_scoring(self, ai_recommendations, sample_patient_profile):
        """Test priority scoring for recommendations."""
        recommendations = await ai_recommendations.generate_clinical_recommendations(sample_patient_profile)
        
        # All recommendations should have priority scores
        for rec in recommendations["recommendations"]:
            assert "priority" in rec
            assert rec["priority"] in ["low", "medium", "high", "urgent"]
            
        # High-priority recommendations should be related to patient safety or urgent clinical needs
        high_priority_recs = [r for r in recommendations["recommendations"] if r["priority"] == "high"]
        if high_priority_recs:
            assert any("safety" in r["rationale"].lower() or "urgent" in r["rationale"].lower() 
                      for r in high_priority_recs)

    @pytest.mark.asyncio
    async def test_recommendations_caching(self, ai_recommendations, sample_patient_profile):
        """Test caching of recommendations for performance."""
        # First call should generate recommendations
        recs1 = await ai_recommendations.generate_clinical_recommendations(sample_patient_profile)
        
        # Second call with same data should use cache
        recs2 = await ai_recommendations.generate_clinical_recommendations(sample_patient_profile)
        
        # Results should be identical when using cache
        assert recs1["recommendations"] == recs2["recommendations"]

    @pytest.mark.asyncio
    async def test_error_handling_invalid_patient_data(self, ai_recommendations):
        """Test error handling with invalid patient data."""
        invalid_patient_data = {
            "patient_id": "invalid",
            "age": -5,  # Invalid age
            "medical_history": "not_a_list"  # Invalid format
        }
        
        recommendations = await ai_recommendations.generate_clinical_recommendations(invalid_patient_data)
        
        # Should handle gracefully
        assert recommendations is not None
        assert "error" in recommendations or len(recommendations.get("recommendations", [])) == 0

    @pytest.mark.asyncio
    async def test_recommendations_health_check(self, ai_recommendations):
        """Test AI recommendations service health monitoring."""
        health_status = await ai_recommendations.get_health_status()
        
        assert health_status is not None
        assert "service" in health_status
        assert "status" in health_status
        assert "ai_models_loaded" in health_status
        assert "knowledge_base_connection" in health_status
        
        assert health_status["service"] == "ai_recommendations"
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_recommendations_metrics(self, ai_recommendations):
        """Test metrics collection for recommendations service."""
        # Generate some recommendations to create metrics
        sample_data = {"patient_id": "test", "age": 50, "medical_history": []}
        
        await ai_recommendations.generate_clinical_recommendations(sample_data)
        await ai_recommendations.generate_clinical_recommendations(sample_data)
        
        metrics = await ai_recommendations.get_metrics()
        
        assert metrics is not None
        assert "total_recommendations_generated" in metrics
        assert "average_processing_time" in metrics
        assert "success_rate" in metrics
        assert metrics["total_recommendations_generated"] >= 2
