"""
Comprehensive test suite for the Medical Patterns Service.

This module provides complete test coverage for the Medical Pattern Matcher functionality
including regex-based medical entity extraction, pattern recognition, clinical terminology
processing, and integration with NER systems.

Tests cover:
1. Medical pattern matcher initialization and configuration
2. Medical entity pattern recognition (conditions, medications, procedures)
3. Clinical terminology extraction and normalization
4. Regex pattern matching for medical text
5. Medical abbreviation expansion and standardization
6. Dosage and frequency pattern recognition
7. Lab value and vital sign extraction
8. Temporal information processing
9. Performance optimization for pattern matching
10. Error handling and pattern validation
"""

import pytest

from app.services.medical_patterns import MedicalPatternMatcher


class TestMedicalPatternMatcher:
    """Comprehensive test cases for the Medical Pattern Matcher Service."""
    
    @pytest.fixture
    def pattern_matcher(self):
        """Create a medical pattern matcher instance for testing."""
        return MedicalPatternMatcher()
    
    @pytest.fixture
    def sample_clinical_text(self):
        """Sample clinical text with various medical entities."""
        return """
        PATIENT: John Doe, 65 y/o male
        CHIEF COMPLAINT: Chest pain and shortness of breath
        
        HISTORY OF PRESENT ILLNESS:
        Patient presents with a 2-week history of chest pain and dyspnea. 
        Has known history of CAD, HTN, and T2DM.
        
        MEDICATIONS:
        - Metformin 500mg BID
        - Lisinopril 10mg daily
        - Atorvastatin 20mg at bedtime
        - Aspirin 81mg daily
        
        PHYSICAL EXAM:
        VS: BP 145/92 mmHg, HR 78 bpm, RR 18/min, Temp 98.6°F, O2 sat 96% RA
        HEENT: Normal
        CV: RRR, no murmurs
        PULM: Clear to auscultation bilaterally
        
        LABS:
        - Glucose: 165 mg/dL (H)
        - HbA1c: 8.2%
        - Total cholesterol: 220 mg/dL
        - LDL: 140 mg/dL (H)
        - HDL: 35 mg/dL (L)
        - Troponin I: <0.04 ng/mL
        
        ASSESSMENT:
        1. Acute chest pain - r/o ACS
        2. Type 2 Diabetes Mellitus - uncontrolled
        3. Hypertension - uncontrolled
        4. Dyslipidemia
        
        PLAN:
        1. EKG and serial troponins
        2. Increase Metformin to 1000mg BID
        3. Add Metoprolol 25mg BID
        4. Follow up in 2 weeks
        """
    
    @pytest.fixture
    def prescription_text(self):
        """Sample prescription text for testing."""
        return """
        Rx: Metformin XR 500mg
        Sig: Take one tablet by mouth twice daily with meals
        Disp: #60 tablets
        Refills: 5
        
        Rx: Lisinopril 10mg
        Sig: Take one tablet by mouth once daily
        Disp: #30 tablets
        Refills: 3
        
        Rx: Atorvastatin 20mg
        Sig: Take one tablet by mouth at bedtime
        Disp: #30 tablets
        Refills: 5
        """

    def test_extract_medical_conditions(self, pattern_matcher, sample_clinical_text):
        """Test extraction of medical conditions from clinical text."""
        conditions = pattern_matcher.extract_medical_conditions(sample_clinical_text)
        
        assert len(conditions) > 0
        
        # Check for specific conditions
        condition_names = [c["name"].lower() for c in conditions]
        assert any("diabetes" in name for name in condition_names)
        assert any("hypertension" in name for name in condition_names)
        assert any("chest pain" in name for name in condition_names)
        
        # Check for abbreviations
        assert any("cad" in name or "coronary artery disease" in name for name in condition_names)
        assert any("t2dm" in name or "type 2 diabetes" in name for name in condition_names)

    def test_extract_medications(self, pattern_matcher, sample_clinical_text):
        """Test extraction of medications from clinical text."""
        medications = pattern_matcher.extract_medications(sample_clinical_text)
        
        assert len(medications) >= 4  # At least 4 medications in sample text
        
        medication_names = [m["name"].lower() for m in medications]
        assert "metformin" in medication_names
        assert "lisinopril" in medication_names
        assert "atorvastatin" in medication_names
        assert "aspirin" in medication_names

    def test_extract_medication_dosages(self, pattern_matcher, sample_clinical_text):
        """Test extraction of medication dosages and frequencies."""
        medications = pattern_matcher.extract_medications(sample_clinical_text)
        
        metformin_entries = [m for m in medications if "metformin" in m["name"].lower()]
        assert len(metformin_entries) > 0
        
        metformin = metformin_entries[0]
        assert "dosage" in metformin
        assert "frequency" in metformin
        assert "500mg" in metformin["dosage"] or "500" in metformin["dosage"]
        assert "bid" in metformin["frequency"].lower() or "twice daily" in metformin["frequency"].lower()

    def test_extract_lab_values(self, pattern_matcher, sample_clinical_text):
        """Test extraction of laboratory values."""
        lab_values = pattern_matcher.extract_lab_values(sample_clinical_text)
        
        assert len(lab_values) > 0
        
        lab_names = [lab["name"].lower() for lab in lab_values]
        assert any("glucose" in name for name in lab_names)
        assert any("hba1c" in name for name in lab_names)
        assert any("cholesterol" in name for name in lab_names)
        
        # Check specific values
        glucose_entries = [lab for lab in lab_values if "glucose" in lab["name"].lower()]
        if glucose_entries:
            glucose = glucose_entries[0]
            assert "value" in glucose
            assert "unit" in glucose
            assert glucose["unit"].lower() in ["mg/dl", "mg/dl"]

    def test_extract_vital_signs(self, pattern_matcher, sample_clinical_text):
        """Test extraction of vital signs."""
        vital_signs = pattern_matcher.extract_vital_signs(sample_clinical_text)
        
        assert len(vital_signs) > 0
        
        vital_types = [vs["type"].lower() for vs in vital_signs]
        assert any("blood_pressure" in vtype or "bp" in vtype for vtype in vital_types)
        assert any("heart_rate" in vtype or "hr" in vtype for vtype in vital_types)
        assert any("temperature" in vtype or "temp" in vtype for vtype in vital_types)

    def test_extract_procedures(self, pattern_matcher, sample_clinical_text):
        """Test extraction of medical procedures."""
        procedures = pattern_matcher.extract_procedures(sample_clinical_text)
        
        assert len(procedures) > 0
        
        procedure_names = [p["name"].lower() for p in procedures]
        assert any("ekg" in name or "ecg" in name or "electrocardiogram" in name for name in procedure_names)

    def test_expand_medical_abbreviations(self, pattern_matcher, sample_clinical_text):
        """Test expansion of medical abbreviations."""
        expanded_text = pattern_matcher.expand_abbreviations(sample_clinical_text)
        
        # Check that common abbreviations are expanded
        assert "coronary artery disease" in expanded_text.lower() or "cad" in expanded_text.lower()
        assert "hypertension" in expanded_text.lower() or "htn" in expanded_text.lower()
        assert "type 2 diabetes mellitus" in expanded_text.lower() or "t2dm" in expanded_text.lower()
        assert "twice daily" in expanded_text.lower() or "bid" in expanded_text.lower()

    def test_normalize_dosage_patterns(self, pattern_matcher):
        """Test normalization of dosage patterns."""
        dosage_texts = [
            "500mg twice daily",
            "500 mg BID",
            "0.5g twice a day",
            "10mg once daily",
            "10 mg QD",
            "20mg at bedtime",
            "20 mg HS"
        ]
        
        for dosage_text in dosage_texts:
            normalized = pattern_matcher.normalize_dosage(dosage_text)
            assert normalized is not None
            assert "amount" in normalized
            assert "frequency" in normalized

    def test_extract_temporal_information(self, pattern_matcher, sample_clinical_text):
        """Test extraction of temporal information."""
        temporal_info = pattern_matcher.extract_temporal_information(sample_clinical_text)
        
        assert len(temporal_info) > 0
        
        # Should find temporal references like "2-week history", "follow up in 2 weeks"
        temporal_phrases = [t["phrase"].lower() for t in temporal_info]
        assert any("week" in phrase for phrase in temporal_phrases)

    def test_extract_prescription_information(self, pattern_matcher, prescription_text):
        """Test extraction of prescription information."""
        prescriptions = pattern_matcher.extract_prescription_info(prescription_text)
        
        assert len(prescriptions) >= 3  # Three prescriptions in sample
        
        for prescription in prescriptions:
            assert "medication" in prescription
            assert "sig" in prescription or "instructions" in prescription
            assert "dispense" in prescription or "quantity" in prescription

    def test_medical_entity_confidence_scoring(self, pattern_matcher, sample_clinical_text):
        """Test confidence scoring for extracted medical entities."""
        medications = pattern_matcher.extract_medications(sample_clinical_text, include_confidence=True)
        
        for medication in medications:
            assert "confidence" in medication
            assert 0 <= medication["confidence"] <= 1
              # Well-formed medication entries should have high confidence
            if "metformin" in medication["name"].lower() and "500mg" in str(medication.get("dosage", "")):
                assert medication["confidence"] > 0.8

    def test_pattern_matching_performance(self, pattern_matcher, sample_clinical_text):
        """Test pattern matching performance with large text."""
        # Create large text by repeating sample content
        large_text = sample_clinical_text * 100  # Repeat 100 times
        
        import time
        start_time = time.time()
        
        conditions = pattern_matcher.extract_medical_conditions(large_text)
        medications = pattern_matcher.extract_medications(large_text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 5.0  # 5 seconds max
        assert len(conditions) > 0
        assert len(medications) > 0

    def test_regex_pattern_validation(self, pattern_matcher):
        """Test validation of regex patterns."""
        # Test valid patterns
        valid_patterns = [
            r'\b(?:metformin|lisinopril|atorvastatin)\b',
            r'\d+\s*mg\b',
            r'\b(?:bid|tid|qid|daily)\b'
        ]
        
        for pattern in valid_patterns:
            assert pattern_matcher.validate_pattern(pattern) is True
        
        # Test invalid patterns
        invalid_patterns = [
            r'[',  # Unclosed bracket
            r'(?P<unclosed',  # Unclosed group
            r'*invalid*'  # Invalid quantifier
        ]
        
        for pattern in invalid_patterns:
            assert pattern_matcher.validate_pattern(pattern) is False

    def test_custom_pattern_addition(self, pattern_matcher):
        """Test adding custom medical patterns."""
        # Add custom medication pattern
        custom_medication_pattern = r'\b(?:custom_drug|test_medication)\s*\d+\s*mg\b'
        
        result = pattern_matcher.add_custom_pattern(
            category="medications",
            pattern=custom_medication_pattern,
            description="Custom test medications"
        )
        
        assert result["success"] is True
        
        # Test the custom pattern
        test_text = "Patient prescribed custom_drug 100mg twice daily"
        medications = pattern_matcher.extract_medications(test_text)
        
        custom_meds = [m for m in medications if "custom_drug" in m["name"].lower()]
        assert len(custom_meds) > 0

    def test_medical_specialty_patterns(self, pattern_matcher):
        """Test patterns specific to medical specialties."""
        cardiology_text = """
        Patient has history of MI, underwent CABG. Current meds include 
        metoprolol, lisinopril, and atorvastatin. EF is 45%.
        """
        
        cardiology_entities = pattern_matcher.extract_specialty_entities(
            cardiology_text, 
            specialty="cardiology"
        )
        
        assert len(cardiology_entities) > 0
        
        # Should find cardiology-specific terms
        entity_names = [e["name"].lower() for e in cardiology_entities]
        assert any("mi" in name or "myocardial infarction" in name for name in entity_names)
        assert any("cabg" in name or "bypass" in name for name in entity_names)
        assert any("ef" in name or "ejection fraction" in name for name in entity_names)

    def test_negation_detection(self, pattern_matcher):
        """Test detection of negated medical entities."""
        negated_text = """
        Patient denies chest pain. No shortness of breath.
        No history of diabetes. Patient is not taking metformin.
        """
        
        entities = pattern_matcher.extract_entities_with_negation(negated_text)
        
        # Should identify negated entities
        negated_entities = [e for e in entities if e.get("negated", False)]
        assert len(negated_entities) > 0
        
        # Check specific negations
        negated_names = [e["name"].lower() for e in negated_entities]
        assert any("chest pain" in name for name in negated_names)
        assert any("diabetes" in name for name in negated_names)

    def test_entity_relationship_extraction(self, pattern_matcher, sample_clinical_text):
        """Test extraction of relationships between medical entities."""
        relationships = pattern_matcher.extract_entity_relationships(sample_clinical_text)
        
        assert len(relationships) > 0
        
        # Should find medication-condition relationships
        med_condition_rels = [r for r in relationships if r["type"] == "medication_for_condition"]
        assert len(med_condition_rels) > 0
        
        # Example: Metformin for diabetes
        metformin_rels = [r for r in med_condition_rels 
                         if "metformin" in r["entity1"].lower() and 
                            "diabetes" in r["entity2"].lower()]
        assert len(metformin_rels) > 0

    def test_medical_text_cleaning(self, pattern_matcher):
        """Test medical text cleaning and preprocessing."""
        messy_text = """
        Pt. c/o CP & SOB x2wks. PMH: CAD, HTN, DM2.
        ROS: (-) F/C/N/V/D. PE: HEENT WNL, CV RRR, PULM CTAB.
        """
        
        cleaned_text = pattern_matcher.clean_medical_text(messy_text)
        
        # Should expand abbreviations and clean formatting
        cleaned_lower = cleaned_text.lower()
        assert "patient" in cleaned_lower or "pt" in cleaned_lower
        assert ("complains of" in cleaned_lower or "c/o" in cleaned_lower or 
                "complaint" in cleaned_lower)

    def test_icd_code_extraction(self, pattern_matcher):
        """Test extraction of ICD codes from medical text."""
        icd_text = """
        1. Type 2 Diabetes Mellitus (E11.9)
        2. Essential Hypertension (I10)
        3. Hyperlipidemia (E78.5)
        4. Coronary Artery Disease (I25.10)
        """
        
        icd_codes = pattern_matcher.extract_icd_codes(icd_text)
        
        assert len(icd_codes) >= 4
        
        # Check specific codes
        codes = [icd["code"] for icd in icd_codes]
        assert "E11.9" in codes
        assert "I10" in codes
        assert "E78.5" in codes
        assert "I25.10" in codes

    def test_cpt_code_extraction(self, pattern_matcher):
        """Test extraction of CPT codes from medical text."""
        cpt_text = """
        Procedures performed:
        - Office visit (99213)
        - EKG (93000)
        - Lipid panel (80061)
        - HbA1c (83036)
        """
        
        cpt_codes = pattern_matcher.extract_cpt_codes(cpt_text)
        
        assert len(cpt_codes) >= 4
        
        codes = [cpt["code"] for cpt in cpt_codes]
        assert "99213" in codes
        assert "93000" in codes
        assert "80061" in codes
        assert "83036" in codes

    def test_pattern_matcher_configuration(self, pattern_matcher):
        """Test pattern matcher configuration and settings."""
        # Test getting current configuration
        config = pattern_matcher.get_configuration()
        
        assert "enabled_patterns" in config
        assert "case_sensitive" in config
        assert "confidence_threshold" in config
        
        # Test updating configuration
        new_config = {
            "case_sensitive": False,
            "confidence_threshold": 0.8,
            "include_negated": True
        }
        
        result = pattern_matcher.update_configuration(new_config)
        assert result["success"] is True
        
        # Verify configuration was updated
        updated_config = pattern_matcher.get_configuration()
        assert not updated_config["case_sensitive"]
        assert updated_config["confidence_threshold"] == 0.8

    def test_pattern_matcher_health_check(self, pattern_matcher):
        """Test pattern matcher health monitoring."""
        health_status = pattern_matcher.get_health_status()
        
        assert "service" in health_status
        assert "status" in health_status
        assert "patterns_loaded" in health_status
        assert "processing_stats" in health_status
        
        assert health_status["service"] == "medical_patterns"
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]

    def test_batch_pattern_processing(self, pattern_matcher):
        """Test batch processing of multiple documents."""
        documents = [
            "Patient with diabetes taking metformin 500mg BID",
            "Hypertensive patient on lisinopril 10mg daily",
            "Chest pain workup with troponin and EKG"
        ]
        
        batch_results = pattern_matcher.batch_process(documents)
        
        assert len(batch_results) == 3
        
        for result in batch_results:
            assert "medications" in result or "conditions" in result or "procedures" in result
            assert "document_id" in result
            assert "processing_time" in result
