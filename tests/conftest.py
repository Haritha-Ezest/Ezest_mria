"""
Test configuration and fixtures for the MRIA test suite.

This module provides common test configuration, fixtures, and utilities
used across all test modules in the MRIA system.
"""

import pytest
import asyncio
import tempfile
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

# Test configuration
TEST_CONFIG = {
    "redis_url": "redis://localhost:6379/1",  # Use database 1 for testing
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "test_password",
    "test_upload_dir": "/tmp/mria_test_uploads",
    "test_data_dir": "tests/data"
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_medical_text():
    """Comprehensive sample medical text for testing."""
    return """
    PATIENT: John Doe
    DOB: 03/15/1958  
    MRN: 12345678
    
    CHIEF COMPLAINT: Increased thirst and frequent urination for 3 months
    
    HISTORY OF PRESENT ILLNESS:
    This 65-year-old male presents with a 3-month history of polydipsia and polyuria.
    Patient reports drinking excessive amounts of water and urinating frequently,
    including nocturia 3-4 times per night. Associated symptoms include fatigue,
    blurred vision, and unintentional weight loss of 15 pounds over 2 months.
    
    PAST MEDICAL HISTORY:
    - Hypertension (diagnosed 2015)
    - Hyperlipidemia (diagnosed 2018)
    - Family history of Type 2 Diabetes (mother, maternal grandfather)
    
    MEDICATIONS:
    - Lisinopril 10mg daily
    - Atorvastatin 20mg daily
    
    PHYSICAL EXAMINATION:
    Vital Signs: BP 145/95 mmHg, HR 78 bpm, RR 16/min, Temp 98.6°F
    Weight: 220 lbs, Height: 5'10", BMI: 31.5 kg/m²
    General: Alert, oriented, appears tired
    HEENT: Fundoscopic exam shows mild diabetic retinopathy
    Cardiovascular: Regular rate and rhythm, no murmurs
    Pulmonary: Clear to auscultation bilaterally
    Abdomen: Soft, non-tender, no organomegaly
    Extremities: No pedal edema, good peripheral pulses
    
    LABORATORY RESULTS:
    Fasting glucose: 165 mg/dL (normal: 70-100 mg/dL)
    HbA1c: 8.2% (normal: <5.7%)
    Random glucose: 245 mg/dL
    Creatinine: 1.0 mg/dL (normal: 0.7-1.3 mg/dL)
    BUN: 18 mg/dL (normal: 7-20 mg/dL)
    Total cholesterol: 220 mg/dL
    LDL: 145 mg/dL
    HDL: 38 mg/dL
    Triglycerides: 185 mg/dL
    
    ASSESSMENT AND PLAN:
    1. Type 2 Diabetes Mellitus - New diagnosis
       - Start Metformin 500mg twice daily with meals
       - Diabetes education scheduled
       - Nutritionist referral
       - Home glucose monitoring
       - Follow-up in 3 months with repeat HbA1c
    
    2. Hypertension - Poorly controlled
       - Increase Lisinopril to 20mg daily
       - Continue monitoring
    
    3. Diabetic retinopathy - Mild
       - Ophthalmology referral for annual screening
       - Blood sugar control essential
    
    FOLLOW-UP:
    - Return visit in 3 months
    - Lab work: HbA1c, comprehensive metabolic panel
    - Blood pressure log review
    - Medication compliance assessment
    """


@pytest.fixture
def sample_entities():
    """Sample medical entities extracted from text."""
    from app.schemas.ner import MedicalEntity, EntityType
    
    return [
        # Conditions
        MedicalEntity(
            text="Type 2 Diabetes Mellitus",
            entity_type=EntityType.CONDITION,
            start_pos=1450,
            end_pos=1475,
            confidence=0.95,
            normalized_form="E11.9",
            knowledge_base_id="ICD10:E11.9"
        ),
        MedicalEntity(
            text="Hypertension",
            entity_type=EntityType.CONDITION,
            start_pos=650,
            end_pos=662,
            confidence=0.92,
            normalized_form="I10",
            knowledge_base_id="ICD10:I10"
        ),
        MedicalEntity(
            text="Diabetic retinopathy",
            entity_type=EntityType.CONDITION,
            start_pos=1750,
            end_pos=1770,
            confidence=0.89,
            normalized_form="E11.319",
            knowledge_base_id="ICD10:E11.319"
        ),
        
        # Medications
        MedicalEntity(
            text="Metformin",
            entity_type=EntityType.MEDICATION,
            start_pos=1500,
            end_pos=1509,
            confidence=0.94,
            normalized_form="6809",
            knowledge_base_id="RXNORM:6809"
        ),
        MedicalEntity(
            text="Lisinopril",
            entity_type=EntityType.MEDICATION,
            start_pos=720,
            end_pos=730,
            confidence=0.91,
            normalized_form="29046",
            knowledge_base_id="RXNORM:29046"
        ),
        
        # Symptoms
        MedicalEntity(
            text="increased thirst",
            entity_type=EntityType.SYMPTOM,
            start_pos=200,
            end_pos=216,
            confidence=0.88
        ),
        MedicalEntity(
            text="frequent urination",
            entity_type=EntityType.SYMPTOM,
            start_pos=221,
            end_pos=239,
            confidence=0.87
        ),
        MedicalEntity(
            text="blurred vision",
            entity_type=EntityType.SYMPTOM,
            start_pos=450,
            end_pos=464,
            confidence=0.85
        ),
        
        # Lab Values
        MedicalEntity(
            text="HbA1c 8.2%",
            entity_type=EntityType.LAB_VALUE,
            start_pos=1200,
            end_pos=1210,
            confidence=0.96,
            test_name="HbA1c",
            test_value="8.2",
            test_unit="%"
        ),
        MedicalEntity(
            text="Fasting glucose 165 mg/dL",
            entity_type=EntityType.LAB_VALUE,
            start_pos=1150,
            end_pos=1176,
            confidence=0.94,
            test_name="Fasting glucose",
            test_value="165",
            test_unit="mg/dL"
        ),
        
        # Dosages and Frequencies
        MedicalEntity(
            text="500mg",
            entity_type=EntityType.DOSAGE,
            start_pos=1510,
            end_pos=1515,
            confidence=0.90
        ),
        MedicalEntity(
            text="twice daily",
            entity_type=EntityType.FREQUENCY,
            start_pos=1516,
            end_pos=1527,
            confidence=0.88
        )
    ]


@pytest.fixture
def sample_patient_data():
    """Sample patient data for graph creation."""
    return {
        "patient_id": "patient_123",
        "patient_info": {
            "name": "John Doe",
            "dob": "1958-03-15",
            "gender": "male",
            "mrn": "12345678",
            "phone": "555-0123",
            "email": "john.doe@email.com",
            "address": "123 Main St, Anytown, ST 12345",
            "emergency_contact": "Jane Doe (wife) - 555-0124",
            "insurance_info": {
                "provider": "Blue Cross Blue Shield",
                "policy_number": "ABC123456789",
                "group_number": "GRP001"
            },
            "preferred_language": "English",
            "created_at": datetime.now().isoformat(),
            "active_status": True
        },
        "visits": [
            {
                "visit_id": "visit_001",
                "date": "2024-01-15",
                "type": "initial_consultation",
                "location": "Primary Care Clinic",
                "provider": "Dr. Sarah Smith",
                "status": "completed",
                "duration": 45,
                "chief_complaint": "increased thirst and frequent urination",
                "visit_notes": "Initial presentation of diabetes symptoms"
            },
            {
                "visit_id": "visit_002",
                "date": "2024-02-20",
                "type": "follow_up",
                "location": "Primary Care Clinic",
                "provider": "Dr. Sarah Smith",
                "status": "completed",
                "duration": 30,
                "chief_complaint": "diabetes management",
                "visit_notes": "Lab results review and treatment initiation"
            }
        ],
        "conditions": [
            {
                "condition_id": "cond_001",
                "name": "Type 2 Diabetes Mellitus",
                "icd_code": "E11.9",
                "severity": "moderate",
                "status": "active",
                "onset_date": "2024-02-20",
                "clinical_notes": "New diagnosis based on elevated HbA1c and symptoms",
                "family_history": True,
                "chronic_flag": True,
                "risk_factors": ["obesity", "family_history", "age"]
            },
            {
                "condition_id": "cond_002",
                "name": "Hypertension",
                "icd_code": "I10",
                "severity": "mild",
                "status": "active",
                "onset_date": "2015-06-15",
                "clinical_notes": "Well-controlled on medication",
                "chronic_flag": True
            }
        ],
        "medications": [
            {
                "medication_id": "med_001",
                "name": "Metformin",
                "dosage": "500mg",
                "frequency": "twice daily",
                "route": "oral",
                "start_date": "2024-02-20",
                "prescribing_provider": "Dr. Sarah Smith",
                "indication": "Type 2 Diabetes Mellitus",
                "side_effects": ["GI upset", "metallic taste"],
                "contraindications": ["kidney disease", "liver disease"]
            },
            {
                "medication_id": "med_002",
                "name": "Lisinopril",
                "dosage": "20mg",
                "frequency": "once daily",
                "route": "oral",
                "start_date": "2015-06-15",
                "prescribing_provider": "Dr. Sarah Smith",
                "indication": "Hypertension"
            }
        ],
        "lab_tests": [
            {
                "test_id": "lab_001",
                "name": "HbA1c",
                "value": "8.2",
                "unit": "%",
                "reference_range": "< 5.7%",
                "status": "abnormal",
                "ordered_date": "2024-02-15",
                "resulted_date": "2024-02-18",
                "ordering_provider": "Dr. Sarah Smith",
                "lab_name": "LabCorp",
                "interpretation": "Consistent with diabetes diagnosis",
                "critical_flag": False
            },
            {
                "test_id": "lab_002",
                "name": "Fasting Glucose",
                "value": "165",
                "unit": "mg/dL",
                "reference_range": "70-100 mg/dL",
                "status": "high",
                "ordered_date": "2024-02-15",
                "resulted_date": "2024-02-18",
                "ordering_provider": "Dr. Sarah Smith",
                "lab_name": "LabCorp",
                "critical_flag": False
            }
        ]
    }


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    mock_client = AsyncMock()
    mock_client.ping.return_value = True
    mock_client.lpush.return_value = 1
    mock_client.hset.return_value = 1
    mock_client.hgetall.return_value = {}
    mock_client.llen.return_value = 0
    mock_client.keys.return_value = []
    mock_client.expire.return_value = True
    return mock_client


@pytest.fixture
def mock_neo4j_session():
    """Mock Neo4j session for testing."""
    mock_session = AsyncMock()
    mock_result = Mock()
    mock_result.data.return_value = []
    mock_result.single.return_value = None
    mock_session.run.return_value = mock_result
    return mock_session


@pytest.fixture
def mock_spacy_model():
    """Mock spaCy NLP model for testing."""
    mock_nlp = Mock()
    mock_doc = Mock()
    mock_doc.ents = []
    mock_nlp.return_value = mock_doc
    return mock_nlp


# Test utilities
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_medical_text(condition: str = "diabetes", complexity: str = "simple") -> str:
        """Generate medical text for testing."""
        templates = {
            "diabetes": {
                "simple": "Patient has type 2 diabetes. Taking metformin 500mg twice daily.",
                "complex": """
                Patient John Doe presents with newly diagnosed Type 2 Diabetes Mellitus.
                Initial symptoms included polydipsia, polyuria, and blurred vision.
                Laboratory results: HbA1c 8.2%, fasting glucose 165 mg/dL.
                Started on Metformin 500mg twice daily with meals.
                Follow-up scheduled in 3 months for glucose monitoring.
                """
            },
            "hypertension": {
                "simple": "Patient has high blood pressure. On lisinopril 10mg daily.",
                "complex": """
                Patient diagnosed with essential hypertension.
                Blood pressure readings consistently elevated at 145/95 mmHg.
                Started on Lisinopril 10mg once daily.
                Lifestyle modifications recommended including diet and exercise.
                """
            }
        }
        
        return templates.get(condition, {}).get(complexity, "Sample medical text")
    
    @staticmethod
    def generate_patient_data(patient_id: str = None) -> Dict[str, Any]:
        """Generate patient data for testing."""
        if patient_id is None:
            patient_id = f"patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        return {
            "patient_id": patient_id,
            "name": "Test Patient",
            "dob": "1980-01-01",
            "gender": "male",
            "mrn": f"MRN_{patient_id}",
            "conditions": ["Type 2 Diabetes Mellitus"],
            "medications": ["Metformin 500mg twice daily"],
            "created_at": datetime.now().isoformat()
        }


# Test markers
pytestmark = [
    pytest.mark.asyncio,
]


# Custom test assertions
def assert_medical_entity_valid(entity):
    """Assert that a medical entity has valid structure."""
    assert entity.text is not None and len(entity.text) > 0
    assert entity.entity_type is not None
    assert entity.start_pos >= 0
    assert entity.end_pos > entity.start_pos
    assert 0.0 <= entity.confidence <= 1.0


def assert_timeline_event_valid(event):
    """Assert that a timeline event has valid structure."""
    assert event.event_id is not None
    assert event.date is not None
    assert event.event_type is not None
    assert event.description is not None
    assert hasattr(event, 'entities')


def assert_graph_node_valid(node):
    """Assert that a graph node has valid structure."""
    assert hasattr(node, 'id') or hasattr(node, 'node_id')
    assert hasattr(node, 'labels') or hasattr(node, 'node_type')
    assert hasattr(node, 'properties')


# Performance testing utilities
class PerformanceTimer:
    """Context manager for measuring test performance."""
    
    def __init__(self, test_name: str, max_duration: float = None):
        self.test_name = test_name
        self.max_duration = max_duration
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        print(f"Test '{self.test_name}' completed in {duration:.3f} seconds")
        
        if self.max_duration and duration > self.max_duration:
            pytest.fail(f"Test '{self.test_name}' exceeded maximum duration of {self.max_duration}s (took {duration:.3f}s)")
    
    @property
    def duration(self) -> float:
        """Get the duration of the test in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
