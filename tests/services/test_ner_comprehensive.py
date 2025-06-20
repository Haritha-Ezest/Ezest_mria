"""
Comprehensive test suite for the Medical Entity Recognition (NER) Agent.

This module provides complete test coverage for the NER Agent functionality
including medical entity extraction, clinical context understanding, and knowledge integration.

Tests cover:
1. NER Agent initialization and model configuration
2. Medical entity extraction and classification
3. Clinical context understanding and relationships
4. Temporal information processing
5. Entity linking and knowledge base integration
6. Medical specialty optimization
7. Confidence scoring and validation
8. Batch processing and performance optimization
9. Error handling and recovery mechanisms
10. Integration with supervisor workflow
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.services.ner_processor import NERProcessor
from app.services.medical_patterns import MedicalPatternMatcher
from app.schemas.ner import (
    NERRequest, NERResponse, NERStatus, NERConfiguration,
    MedicalEntity, EntityType, ProcessingMode, TemporalEntity
)


class TestNERAgent:
    """Comprehensive test cases for the Medical Entity Recognition Agent."""
    
    @pytest.fixture
    def ner_processor(self):
        """Create a NER processor instance for testing."""
        return NERProcessor()
    
    @pytest.fixture
    def medical_pattern_matcher(self):
        """Create a medical pattern matcher for testing."""
        return MedicalPatternMatcher()
    
    @pytest.fixture
    def sample_medical_text(self):
        """Comprehensive sample medical text for testing."""
        return """
        PATIENT: John Doe
        DOB: 03/15/1958  
        MRN: 12345678
        DATE: 2024-01-15
        
        CHIEF COMPLAINT: Increased thirst and frequent urination for 3 months
        
        HISTORY OF PRESENT ILLNESS:
        This 65-year-old male presents with a 3-month history of polyuria, polydipsia, 
        and blurred vision. Patient reports drinking 6-8 glasses of water daily and 
        urinating every 2 hours. Weight loss of 15 pounds over the past 2 months.
        
        PAST MEDICAL HISTORY:
        - Hypertension diagnosed 5 years ago
        - Hyperlipidemia
        - Family history of diabetes mellitus
        
        MEDICATIONS:
        - Lisinopril 10mg daily
        - Atorvastatin 20mg at bedtime
        
        PHYSICAL EXAMINATION:
        Vital Signs: BP 140/90 mmHg, HR 78 bpm, Temp 98.6°F, RR 16/min
        BMI: 32 kg/m²
        
        LABORATORY RESULTS:
        - Fasting glucose: 185 mg/dL (normal: 70-100 mg/dL)
        - HbA1c: 8.2% (normal: <7.0%)
        - Total cholesterol: 220 mg/dL
        - LDL: 140 mg/dL
        - HDL: 35 mg/dL
        - Triglycerides: 280 mg/dL
        
        ASSESSMENT AND PLAN:
        1. Type 2 Diabetes Mellitus - newly diagnosed
           - Start Metformin 500mg twice daily
           - Diabetes education and lifestyle counseling
           - Follow-up in 3 months
        
        2. Hypertension - continue current therapy
           - Monitor blood pressure
        
        3. Dyslipidemia - optimize lipid management
           - Continue statin therapy
           - Recheck lipids in 6 weeks
        """
    
    @pytest.fixture
    def prescription_text(self):
        """Sample prescription text for testing."""
        return """
        Rx: Metformin 500mg
        Sig: Take one tablet by mouth twice daily with meals
        Disp: #60 tablets
        Refills: 5
        
        Rx: Lisinopril 10mg  
        Sig: Take one tablet by mouth daily
        Disp: #30 tablets
        Refills: 3
        
        Dr. Sarah Johnson, MD
        DEA: BJ1234567
        """
    
    @pytest.fixture
    def lab_report_text(self):
        """Sample lab report text for testing."""
        return """
        LABORATORY REPORT
        Patient: Jane Smith
        DOB: 07/22/1970
        Date Collected: 2024-01-10
        Date Reported: 2024-01-11
        
        LIPID PANEL:
        Total Cholesterol: 245 mg/dL (High) [Reference: <200 mg/dL]
        LDL Cholesterol: 165 mg/dL (High) [Reference: <100 mg/dL]
        HDL Cholesterol: 42 mg/dL (Low) [Reference: >40 mg/dL]
        Triglycerides: 189 mg/dL [Reference: <150 mg/dL]
        
        DIABETES PANEL:
        Glucose, Fasting: 155 mg/dL (High) [Reference: 70-100 mg/dL]
        Hemoglobin A1c: 7.8% (High) [Reference: <7.0%]
        
        KIDNEY FUNCTION:
        Creatinine: 1.1 mg/dL [Reference: 0.7-1.3 mg/dL]
        BUN: 18 mg/dL [Reference: 7-20 mg/dL]
        eGFR: >60 mL/min/1.73m²
        """

    # Test 1: NER Agent Initialization and Configuration
    async def test_ner_processor_initialization(self, ner_processor):
        """Test NER processor initialization with medical models."""
        assert ner_processor is not None
        assert hasattr(ner_processor, 'models')
        assert hasattr(ner_processor, 'entity_types')
        
        # Test available models
        available_models = ner_processor.get_available_models()
        expected_models = ['scispacy', 'biobert', 'med7', 'bio_clinical_bert']
        for model in expected_models:
            assert model in available_models

    async def test_medical_model_configuration(self, ner_processor):
        """Test medical NLP model configuration and switching."""
        # Test scispaCy configuration
        scispacy_config = NERConfiguration(
            processing_mode=ProcessingMode.MEDICAL,
            models=['scispacy'],
            entity_linking=True,
            confidence_threshold=0.8
        )
        
        ner_processor.configure_models(scispacy_config)
        assert 'scispacy' in ner_processor.active_models
        assert ner_processor.entity_linking_enabled is True
        
        # Test multi-model configuration
        multi_model_config = NERConfiguration(
            processing_mode=ProcessingMode.ENSEMBLE,
            models=['scispacy', 'biobert', 'med7'],
            confidence_threshold=0.75,
            consensus_threshold=0.6
        )
        
        ner_processor.configure_models(multi_model_config)
        assert len(ner_processor.active_models) == 3
        assert ner_processor.processing_mode == ProcessingMode.ENSEMBLE

    async def test_entity_type_configuration(self, ner_processor):
        """Test entity type configuration and customization."""
        # Test standard medical entity types
        standard_entities = ner_processor.get_supported_entity_types()
        expected_entities = [
            EntityType.CONDITION, EntityType.MEDICATION, EntityType.PROCEDURE,
            EntityType.SYMPTOM, EntityType.LAB_VALUE, EntityType.ANATOMICAL,
            EntityType.TEMPORAL, EntityType.DOSAGE, EntityType.FREQUENCY
        ]
        
        for entity_type in expected_entities:
            assert entity_type in standard_entities
        
        # Test custom entity configuration
        custom_entities = [
            EntityType.CONDITION, EntityType.MEDICATION, EntityType.LAB_VALUE
        ]
        
        ner_processor.configure_entity_types(custom_entities)
        assert set(ner_processor.target_entity_types) == set(custom_entities)

    # Test 2: Medical Entity Extraction and Classification
    async def test_comprehensive_entity_extraction(self, ner_processor, sample_medical_text):
        """Test comprehensive medical entity extraction from clinical text."""
        with patch.object(ner_processor, '_extract_with_scispacy') as mock_scispacy, \
             patch.object(ner_processor, '_extract_with_biobert') as mock_biobert, \
             patch.object(ner_processor, '_extract_with_med7') as mock_med7:
            
            # Mock model responses
            mock_scispacy.return_value = [
                MedicalEntity(text="Type 2 Diabetes Mellitus", label=EntityType.CONDITION, 
                            start=0, end=24, confidence=0.95),
                MedicalEntity(text="Hypertension", label=EntityType.CONDITION, 
                            start=25, end=37, confidence=0.92),
                MedicalEntity(text="Metformin", label=EntityType.MEDICATION, 
                            start=40, end=49, confidence=0.88)
            ]
            
            mock_biobert.return_value = [
                MedicalEntity(text="diabetes", label=EntityType.CONDITION, 
                            start=10, end=18, confidence=0.93),
                MedicalEntity(text="500mg", label=EntityType.DOSAGE, 
                            start=50, end=55, confidence=0.90)
            ]
            
            mock_med7.return_value = [
                MedicalEntity(text="twice daily", label=EntityType.FREQUENCY, 
                            start=60, end=71, confidence=0.85)
            ]
            
            # Test entity extraction
            request = NERRequest(
                document_id="comprehensive_test",
                text=sample_medical_text,
                configuration=NERConfiguration(
                    processing_mode=ProcessingMode.ENSEMBLE,
                    models=['scispacy', 'biobert', 'med7']
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify extraction results
            assert response.status == NERStatus.COMPLETED
            assert response.total_entities >= 5
            
            # Check for specific entity types
            entity_labels = [entity.label for entity in response.entities]
            assert EntityType.CONDITION in entity_labels
            assert EntityType.MEDICATION in entity_labels
            assert EntityType.DOSAGE in entity_labels

    async def test_prescription_entity_extraction(self, ner_processor, prescription_text):
        """Test entity extraction from prescription text."""
        with patch.object(ner_processor, '_extract_medication_entities') as mock_meds:
            mock_meds.return_value = [
                MedicalEntity(text="Metformin", label=EntityType.MEDICATION, 
                            start=4, end=13, confidence=0.95,
                            attributes={"drug_class": "biguanide", "indication": "diabetes"}),
                MedicalEntity(text="500mg", label=EntityType.DOSAGE, 
                            start=14, end=19, confidence=0.92),
                MedicalEntity(text="twice daily", label=EntityType.FREQUENCY, 
                            start=45, end=56, confidence=0.88),
                MedicalEntity(text="Lisinopril", label=EntityType.MEDICATION, 
                            start=80, end=90, confidence=0.94,
                            attributes={"drug_class": "ACE inhibitor", "indication": "hypertension"})
            ]
            
            request = NERRequest(
                document_id="prescription_test",
                text=prescription_text,
                configuration=NERConfiguration(
                    processing_mode=ProcessingMode.MEDICAL,
                    document_type="prescription",
                    extract_medications=True
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify prescription-specific extraction
            assert response.status == NERStatus.COMPLETED
            medications = [e for e in response.entities if e.label == EntityType.MEDICATION]
            dosages = [e for e in response.entities if e.label == EntityType.DOSAGE]
            frequencies = [e for e in response.entities if e.label == EntityType.FREQUENCY]
            
            assert len(medications) >= 2
            assert len(dosages) >= 1
            assert len(frequencies) >= 1

    async def test_lab_value_extraction(self, ner_processor, lab_report_text):
        """Test extraction of lab values and ranges from lab reports."""
        with patch.object(ner_processor, '_extract_lab_values') as mock_labs:
            mock_labs.return_value = [
                MedicalEntity(text="Total Cholesterol: 245 mg/dL", label=EntityType.LAB_VALUE, 
                            start=0, end=29, confidence=0.94,
                            attributes={"test_name": "Total Cholesterol", "value": 245, 
                                      "unit": "mg/dL", "status": "High", "reference": "<200 mg/dL"}),
                MedicalEntity(text="HbA1c: 7.8%", label=EntityType.LAB_VALUE, 
                            start=200, end=212, confidence=0.96,
                            attributes={"test_name": "HbA1c", "value": 7.8, 
                                      "unit": "%", "status": "High", "reference": "<7.0%"})
            ]
            
            request = NERRequest(
                document_id="lab_report_test",
                text=lab_report_text,
                configuration=NERConfiguration(
                    processing_mode=ProcessingMode.MEDICAL,
                    document_type="lab_report",
                    extract_lab_values=True,
                    parse_reference_ranges=True
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify lab value extraction
            assert response.status == NERStatus.COMPLETED
            lab_values = [e for e in response.entities if e.label == EntityType.LAB_VALUE]
            assert len(lab_values) >= 2
            
            # Check for parsed attributes
            cholesterol_entity = next(e for e in lab_values if "Cholesterol" in e.text)
            assert cholesterol_entity.attributes["value"] == 245
            assert cholesterol_entity.attributes["status"] == "High"

    # Test 3: Temporal Information Processing
    async def test_temporal_entity_extraction(self, ner_processor):
        """Test extraction of temporal information from medical text."""
        temporal_text = """
        Patient diagnosed with diabetes 3 months ago. Taking medication twice daily.
        Follow-up appointment scheduled in 6 weeks. Last HbA1c was 8.2% on January 15, 2024.
        Symptoms started 2 years ago and have been worsening over the past 6 months.
        """
        
        with patch.object(ner_processor, '_extract_temporal_entities') as mock_temporal:
            mock_temporal.return_value = [
                TemporalEntity(text="3 months ago", label=EntityType.TEMPORAL, 
                             start=35, end=47, confidence=0.92,
                             temporal_type="relative_past", normalized_value="-3 months"),
                TemporalEntity(text="twice daily", label=EntityType.FREQUENCY, 
                             start=70, end=81, confidence=0.88,
                             temporal_type="frequency", normalized_value="2/day"),
                TemporalEntity(text="January 15, 2024", label=EntityType.DATE, 
                             start=150, end=166, confidence=0.95,
                             temporal_type="absolute_date", normalized_value="2024-01-15")
            ]
            
            request = NERRequest(
                document_id="temporal_test",
                text=temporal_text,
                configuration=NERConfiguration(
                    extract_temporal=True,
                    normalize_temporal=True
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify temporal extraction
            temporal_entities = [e for e in response.entities if e.label in [EntityType.TEMPORAL, EntityType.DATE, EntityType.FREQUENCY]]
            assert len(temporal_entities) >= 3
            
            # Check temporal normalization
            relative_entity = next(e for e in temporal_entities if "months ago" in e.text)
            assert hasattr(relative_entity, 'normalized_value')
            assert relative_entity.normalized_value == "-3 months"

    async def test_temporal_relationship_detection(self, ner_processor):
        """Test detection of temporal relationships between medical events."""
        relationship_text = """
        Patient was diagnosed with hypertension in 2020. Started on Lisinopril immediately.
        Developed diabetes in 2023, 3 years after hypertension diagnosis.
        Metformin was added to the regimen after diabetes diagnosis.
        """
        
        with patch.object(ner_processor, '_extract_temporal_relationships') as mock_relations:
            mock_relations.return_value = [
                {
                    "event1": "hypertension diagnosis",
                    "event2": "diabetes diagnosis", 
                    "relationship": "before",
                    "duration": "3 years",
                    "confidence": 0.89
                },
                {
                    "event1": "diabetes diagnosis",
                    "event2": "Metformin started",
                    "relationship": "before",
                    "duration": "immediate",
                    "confidence": 0.92
                }
            ]
            
            request = NERRequest(
                document_id="relationship_test",
                text=relationship_text,
                configuration=NERConfiguration(
                    extract_temporal=True,
                    analyze_relationships=True
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify relationship detection
            assert hasattr(response, 'temporal_relationships')
            assert len(response.temporal_relationships) >= 2
            assert any(rel["relationship"] == "before" for rel in response.temporal_relationships)

    # Test 4: Entity Linking and Knowledge Base Integration
    async def test_medical_entity_linking(self, ner_processor):
        """Test linking medical entities to knowledge bases."""
        with patch.object(ner_processor, '_link_to_knowledge_bases') as mock_linking:
            mock_linking.return_value = [
                MedicalEntity(text="Type 2 Diabetes Mellitus", label=EntityType.CONDITION,
                            start=0, end=24, confidence=0.95,
                            knowledge_links={
                                "icd10": "E11",
                                "snomed": "44054006",
                                "umls": "C0011860",
                                "mesh": "D003924"
                            }),
                MedicalEntity(text="Metformin", label=EntityType.MEDICATION,
                            start=30, end=39, confidence=0.92,
                            knowledge_links={
                                "rxnorm": "6809",
                                "drugbank": "DB00331",
                                "atc": "A10BA02"
                            })
            ]
            
            request = NERRequest(
                document_id="linking_test",
                text="Patient has Type 2 Diabetes Mellitus, prescribed Metformin",
                configuration=NERConfiguration(
                    entity_linking=True,
                    knowledge_bases=["icd10", "snomed", "rxnorm", "umls"]
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify entity linking
            linked_entities = [e for e in response.entities if hasattr(e, 'knowledge_links') and e.knowledge_links]
            assert len(linked_entities) >= 2
            
            diabetes_entity = next(e for e in linked_entities if "Diabetes" in e.text)
            assert "icd10" in diabetes_entity.knowledge_links
            assert diabetes_entity.knowledge_links["icd10"] == "E11"

    async def test_medical_terminology_normalization(self, ner_processor):
        """Test normalization of medical terminology and abbreviations."""
        with patch.object(ner_processor, '_normalize_medical_terms') as mock_normalize:
            mock_normalize.return_value = [
                {
                    "original": "DM",
                    "normalized": "Diabetes Mellitus",
                    "expansion_type": "abbreviation",
                    "confidence": 0.95
                },
                {
                    "original": "HTN", 
                    "normalized": "Hypertension",
                    "expansion_type": "abbreviation",
                    "confidence": 0.92
                },
                {
                    "original": "MI",
                    "normalized": "Myocardial Infarction", 
                    "expansion_type": "abbreviation",
                    "confidence": 0.88
                }
            ]
            
            abbreviated_text = "Patient has DM and HTN. Previous MI in 2020."
            
            request = NERRequest(
                document_id="normalization_test",
                text=abbreviated_text,
                configuration=NERConfiguration(
                    normalize_terminology=True,
                    expand_abbreviations=True
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify normalization
            assert hasattr(response, 'normalizations')
            assert len(response.normalizations) >= 3
            assert any(norm["original"] == "DM" for norm in response.normalizations)

    # Test 5: Medical Specialty Optimization
    async def test_cardiology_specialty_optimization(self, ner_processor):
        """Test NER optimization for cardiology specialty."""
        cardiology_text = """
        Patient presents with chest pain and shortness of breath. 
        ECG shows ST elevation in leads V1-V4. Troponin I elevated at 15.2 ng/mL.
        Echocardiogram reveals EF 35% with anterior wall motion abnormality.
        Diagnosed with STEMI. Started on aspirin, clopidogrel, and atorvastatin.
        """
        
        with patch.object(ner_processor, '_optimize_for_cardiology') as mock_cardiology:
            mock_cardiology.return_value = [
                MedicalEntity(text="ST elevation", label=EntityType.FINDING,
                            start=50, end=62, confidence=0.94,
                            specialty_context="cardiology"),
                MedicalEntity(text="Troponin I", label=EntityType.LAB_VALUE,
                            start=80, end=90, confidence=0.96,
                            specialty_context="cardiology"),
                MedicalEntity(text="STEMI", label=EntityType.CONDITION,
                            start=200, end=205, confidence=0.98,
                            specialty_context="cardiology")
            ]
            
            request = NERRequest(
                document_id="cardiology_test",
                text=cardiology_text,
                configuration=NERConfiguration(
                    medical_specialty="cardiology",
                    specialty_optimization=True
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify cardiology optimization
            cardiology_entities = [e for e in response.entities 
                                 if hasattr(e, 'specialty_context') and e.specialty_context == "cardiology"]
            assert len(cardiology_entities) >= 3

    async def test_radiology_specialty_optimization(self, ner_processor):
        """Test NER optimization for radiology specialty."""
        radiology_text = """
        CT scan of the chest with contrast shows no acute pulmonary embolism.
        Bilateral lower lobe infiltrates consistent with pneumonia.
        No pleural effusion or pneumothorax identified.
        Heart size is normal. Aorta is unremarkable.
        """
        
        with patch.object(ner_processor, '_optimize_for_radiology') as mock_radiology:
            mock_radiology.return_value = [
                MedicalEntity(text="CT scan", label=EntityType.PROCEDURE,
                            start=0, end=7, confidence=0.95,
                            specialty_context="radiology"),
                MedicalEntity(text="pulmonary embolism", label=EntityType.CONDITION,
                            start=40, end=58, confidence=0.92,
                            specialty_context="radiology"),
                MedicalEntity(text="infiltrates", label=EntityType.FINDING,
                            start=80, end=91, confidence=0.88,
                            specialty_context="radiology")
            ]
            
            request = NERRequest(
                document_id="radiology_test",
                text=radiology_text,
                configuration=NERConfiguration(
                    medical_specialty="radiology",
                    specialty_optimization=True
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify radiology optimization
            radiology_entities = [e for e in response.entities 
                                if hasattr(e, 'specialty_context') and e.specialty_context == "radiology"]
            assert len(radiology_entities) >= 3

    # Test 6: Confidence Scoring and Validation
    async def test_confidence_scoring_mechanisms(self, ner_processor):
        """Test confidence scoring for extracted entities."""
        with patch.object(ner_processor, '_calculate_entity_confidence') as mock_confidence:
            mock_confidence.return_value = {
                "model_confidence": 0.92,
                "context_confidence": 0.88,
                "knowledge_base_confidence": 0.95,
                "final_confidence": 0.92
            }
            
            test_text = "Patient diagnosed with Type 2 Diabetes Mellitus"
            
            request = NERRequest(
                document_id="confidence_test",
                text=test_text,
                configuration=NERConfiguration(
                    confidence_threshold=0.8,
                    calculate_detailed_confidence=True
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify confidence scoring
            assert response.overall_confidence >= 0.8
            high_confidence_entities = [e for e in response.entities if e.confidence >= 0.9]
            assert len(high_confidence_entities) >= 1

    async def test_quality_assessment_metrics(self, ner_processor):
        """Test quality assessment metrics for NER results."""
        with patch.object(ner_processor, '_assess_extraction_quality') as mock_quality:
            mock_quality.return_value = {
                "entity_density": 0.15,  # entities per word
                "medical_terminology_ratio": 0.8,
                "context_coherence": 0.92,
                "extraction_completeness": 0.88,
                "overall_quality": 0.89
            }
            
            request = NERRequest(
                document_id="quality_test",
                text="High-quality medical text with proper terminology",
                configuration=NERConfiguration(
                    assess_quality=True,
                    quality_threshold=0.8
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify quality assessment
            assert hasattr(response, 'quality_metrics')
            assert response.quality_metrics["overall_quality"] >= 0.8

    # Test 7: Error Handling and Recovery
    async def test_model_failure_recovery(self, ner_processor):
        """Test recovery mechanisms when NER models fail."""
        with patch.object(ner_processor, '_extract_with_scispacy') as mock_scispacy:
            # Simulate model failure
            mock_scispacy.side_effect = Exception("scispaCy model failed to load")
            
            # Mock fallback pattern matching
            with patch.object(ner_processor, '_fallback_pattern_extraction') as mock_fallback:
                mock_fallback.return_value = [
                    MedicalEntity(text="diabetes", label=EntityType.CONDITION,
                                start=20, end=28, confidence=0.75,
                                extraction_method="pattern_fallback")
                ]
                
                request = NERRequest(
                    document_id="failure_test",
                    text="Patient has diabetes mellitus",
                    configuration=NERConfiguration(
                        models=['scispacy'],
                        enable_fallback=True
                    )
                )
                
                response = await ner_processor.process_text(request)
                
                # Verify fallback extraction
                assert response.status == NERStatus.COMPLETED
                assert len(response.entities) >= 1
                assert response.entities[0].extraction_method == "pattern_fallback"

    async def test_low_confidence_handling(self, ner_processor):
        """Test handling of low-confidence extractions."""
        with patch.object(ner_processor, '_extract_entities') as mock_extract:
            # Mock low-confidence entities
            mock_extract.return_value = [
                MedicalEntity(text="possible condition", label=EntityType.CONDITION,
                            start=0, end=18, confidence=0.45),
                MedicalEntity(text="definite medication", label=EntityType.MEDICATION,
                            start=20, end=39, confidence=0.95),
                MedicalEntity(text="uncertain finding", label=EntityType.SYMPTOM,
                            start=40, end=57, confidence=0.6)
            ]
            
            request = NERRequest(
                document_id="low_confidence_test",
                text="Text with uncertain medical entities",
                configuration=NERConfiguration(
                    confidence_threshold=0.7,
                    handle_low_confidence="flag"  # flag, discard, or manual_review
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify low-confidence handling
            high_confidence_entities = [e for e in response.entities if e.confidence >= 0.7]
            
            assert len(high_confidence_entities) >= 1
            assert hasattr(response, 'low_confidence_entities')
            assert len(response.low_confidence_entities) >= 1

    # Test 8: Batch Processing and Performance
    async def test_batch_text_processing(self, ner_processor):
        """Test batch processing of multiple medical texts."""
        batch_texts = [
            "Patient has Type 2 Diabetes Mellitus",
            "Prescribed Metformin 500mg twice daily", 
            "Blood pressure elevated at 140/90 mmHg",
            "HbA1c level is 7.8 percent"
        ]
        
        batch_requests = [
            NERRequest(
                document_id=f"batch_doc_{i}",
                text=text,
                configuration=NERConfiguration(processing_mode=ProcessingMode.FAST)
            ) for i, text in enumerate(batch_texts)
        ]
        
        with patch.object(ner_processor, '_process_batch_efficiently') as mock_batch:
            mock_batch.return_value = [
                NERResponse(
                    document_id=f"batch_doc_{i}",
                    status=NERStatus.COMPLETED,
                    entities=[
                        MedicalEntity(text="entity", label=EntityType.CONDITION, 
                                    start=0, end=6, confidence=0.9)
                    ],
                    processing_time=1.2
                ) for i in range(4)
            ]
            
            responses = await ner_processor.process_batch(batch_requests)
            
            # Verify batch processing
            assert len(responses) == 4
            for response in responses:
                assert response.status == NERStatus.COMPLETED
                assert response.processing_time < 5.0  # Efficient processing

    async def test_performance_optimization(self, ner_processor):
        """Test performance optimization for large texts."""
        large_text = "Large medical text. " * 1000  # Simulate large document
        
        with patch.object(ner_processor, '_optimize_for_large_text') as mock_optimize:
            mock_optimize.return_value = {
                "chunked_processing": True,
                "parallel_models": True,
                "memory_optimization": True,
                "processing_time": 8.5
            }
            
            request = NERRequest(
                document_id="performance_test",
                text=large_text,
                configuration=NERConfiguration(
                    processing_mode=ProcessingMode.OPTIMIZED,
                    chunk_large_texts=True,
                    parallel_processing=True
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify performance optimization
            assert response.processing_time < 10.0  # Should be optimized
            assert hasattr(response, 'optimization_applied')
            assert response.optimization_applied is True

    # Test 9: Integration with Supervisor
    async def test_supervisor_workflow_integration(self, ner_processor):
        """Test integration with supervisor workflow."""
        supervisor_callback = AsyncMock()
        ner_processor.set_supervisor_callback(supervisor_callback)
        
        with patch.object(ner_processor, '_extract_entities') as mock_extract:
            mock_extract.return_value = [
                MedicalEntity(text="diabetes", label=EntityType.CONDITION,
                            start=0, end=8, confidence=0.92)
            ]
            
            request = NERRequest(
                document_id="supervisor_integration_test",
                text="Patient has diabetes",
                job_id="supervisor_job_456",
                configuration=NERConfiguration(
                    notify_supervisor=True,
                    progress_callbacks=True
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify supervisor integration
            supervisor_callback.assert_called()
            assert response.job_id == "supervisor_job_456"
            assert response.status == NERStatus.COMPLETED

    # Test 10: Configuration and Customization
    async def test_custom_entity_patterns(self, ner_processor):
        """Test custom entity pattern configuration."""
        custom_patterns = {
            "CUSTOM_MEDICATION": [
                r"\b\w+ine\b",  # Medications ending in 'ine'
                r"\b\w+pril\b"  # ACE inhibitors ending in 'pril'
            ],
            "CUSTOM_LAB": [
                r"\b[A-Z]{2,4}\d*\s*[:=]\s*\d+\.?\d*\s*\w+/?\w*\b"
            ]
        }
        
        ner_processor.add_custom_patterns(custom_patterns)
        
        test_text = "Prescribed lisinopril. Lab shows TSH: 2.5 mU/L"
        
        with patch.object(ner_processor, '_extract_with_custom_patterns') as mock_custom:
            mock_custom.return_value = [
                MedicalEntity(text="lisinopril", label="CUSTOM_MEDICATION",
                            start=11, end=21, confidence=0.88),
                MedicalEntity(text="TSH: 2.5 mU/L", label="CUSTOM_LAB",
                            start=35, end=49, confidence=0.85)
            ]
            
            request = NERRequest(
                document_id="custom_patterns_test",
                text=test_text,
                configuration=NERConfiguration(
                    use_custom_patterns=True
                )
            )
            
            response = await ner_processor.process_text(request)
            
            # Verify custom pattern extraction
            custom_entities = [e for e in response.entities 
                             if e.label.startswith("CUSTOM_")]
            assert len(custom_entities) >= 2
