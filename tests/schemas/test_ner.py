"""
Comprehensive tests for the NER schemas.

This module tests all Pydantic models in the NER schemas, including:
- EntityType enum validation  
- Request and response models
- Field validation and constraints
- Entity validation models
"""

import pytest
from pydantic import ValidationError

from app.schemas.ner import (
    EntityType, NERRequest, NERResponse, BatchNERRequest, BatchNERResponse,
    EntityValidationRequest, EntityValidationResponse, ExtractedEntity
)


class TestEntityType:
    """Test the EntityType enum."""

    def test_valid_entity_types(self):
        """Test valid entity types."""
        valid_types = [
            "PERSON",
            "CONDITION", 
            "MEDICATION",
            "DATE",
            "TIME",
            "QUANTITY",
            "MEASUREMENT",
            "PROCEDURE",
            "ORGANIZATION",
            "LOCATION"
        ]
        
        for entity_type in valid_types:
            assert EntityType(entity_type) in EntityType

    def test_invalid_entity_type(self):
        """Test invalid entity type."""
        with pytest.raises(ValueError):
            EntityType("INVALID_TYPE")


class TestExtractedEntity:
    """Test the ExtractedEntity model."""

    def test_valid_entity(self):
        """Test valid extracted entity."""
        entity = ExtractedEntity(
            text="John Doe",
            label="PERSON",
            start=8,
            end=16,
            confidence=0.95,
            context="Patient John Doe was admitted",
            normalized_form="JOHN DOE",
            umls_code="C123456"
        )
        
        assert entity.text == "John Doe"
        assert entity.label == "PERSON"
        assert entity.start == 8
        assert entity.end == 16
        assert entity.confidence == 0.95
        assert entity.context == "Patient John Doe was admitted"
        assert entity.normalized_form == "JOHN DOE"
        assert entity.umls_code == "C123456"

    def test_entity_minimal_fields(self):
        """Test entity with only required fields."""
        entity = ExtractedEntity(
            text="hypertension",
            label="CONDITION",
            start=25,
            end=37,
            confidence=0.88
        )
        
        assert entity.text == "hypertension"
        assert entity.label == "CONDITION"
        assert entity.start == 25
        assert entity.end == 37
        assert entity.confidence == 0.88
        assert entity.context is None
        assert entity.normalized_form is None

    def test_invalid_confidence(self):
        """Test invalid confidence score validation."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractedEntity(
                text="test",
                label="CONDITION",
                start=0,
                end=4,
                confidence=-0.1  # Below 0
            )
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ExtractedEntity(
                text="test",
                label="CONDITION",
                start=0,
                end=4,
                confidence=1.5  # Above 1
            )
        assert "ensure this value is less than or equal to 1" in str(exc_info.value)

    def test_invalid_positions(self):
        """Test invalid start/end positions."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractedEntity(
                text="test",
                label="CONDITION",
                start=-1,  # Negative start
                end=4,
                confidence=0.9
            )
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ExtractedEntity(
                text="test",
                label="CONDITION",
                start=10,
                end=5,  # End before start
                confidence=0.9
            )
        assert "end position must be greater than start position" in str(exc_info.value)


class TestNERRequest:
    """Test the NERRequest model."""

    def test_valid_request(self):
        """Test valid NER request."""
        request = NERRequest(
            text="Patient John Doe has diabetes and takes metformin daily.",
            entity_types=[EntityType.PERSON, EntityType.CONDITION, EntityType.MEDICATION],
            include_confidence=True,
            include_context=True,
            context_window=20,
            confidence_threshold=0.8,
            language="en"
        )
        
        assert "diabetes" in request.text
        assert EntityType.PERSON in request.entity_types
        assert EntityType.CONDITION in request.entity_types
        assert EntityType.MEDICATION in request.entity_types
        assert request.include_confidence is True
        assert request.include_context is True
        assert request.context_window == 20
        assert request.confidence_threshold == 0.8
        assert request.language == "en"

    def test_request_with_defaults(self):
        """Test NER request with default values."""
        request = NERRequest(
            text="Simple medical text."
        )
        
        assert len(request.entity_types) > 0  # Should have default types
        assert request.include_confidence is True
        assert request.include_context is False
        assert request.context_window == 10
        assert request.confidence_threshold == 0.5
        assert request.language == "en"

    def test_invalid_text(self):
        """Test invalid text validation."""
        with pytest.raises(ValidationError) as exc_info:
            NERRequest(text="")  # Empty text
        assert "ensure this value has at least 1 characters" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            NERRequest(text="x" * 100001)  # Too long
        assert "ensure this value has at most 100000 characters" in str(exc_info.value)

    def test_invalid_confidence_threshold(self):
        """Test invalid confidence threshold validation."""
        with pytest.raises(ValidationError) as exc_info:
            NERRequest(
                text="test text",
                confidence_threshold=-0.1  # Below 0
            )
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            NERRequest(
                text="test text",
                confidence_threshold=1.1  # Above 1
            )
        assert "ensure this value is less than or equal to 1" in str(exc_info.value)

    def test_invalid_context_window(self):
        """Test invalid context window validation."""
        with pytest.raises(ValidationError) as exc_info:
            NERRequest(
                text="test text",
                context_window=0  # Below minimum
            )
        assert "ensure this value is greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            NERRequest(
                text="test text",
                context_window=101  # Above maximum
            )
        assert "ensure this value is less than or equal to 100" in str(exc_info.value)


class TestNERResponse:
    """Test the NERResponse model."""

    def test_valid_response(self):
        """Test valid NER response."""
        response = NERResponse(
            success=True,
            entities=[
                ExtractedEntity(
                    text="John Doe",
                    label="PERSON",
                    start=8,
                    end=16,
                    confidence=0.95
                ),
                ExtractedEntity(
                    text="diabetes",
                    label="CONDITION",
                    start=25,
                    end=33,
                    confidence=0.92
                )
            ],
            total_entities=2,
            processing_time=0.35,
            model_version="1.0.0",
            language="en",
            metadata={
                "input_length": 57,
                "entities_found": {"PERSON": 1, "CONDITION": 1}
            }
        )
        
        assert response.success is True
        assert len(response.entities) == 2
        assert response.total_entities == 2
        assert response.processing_time == 0.35
        assert response.model_version == "1.0.0"
        assert response.language == "en"

    def test_error_response(self):
        """Test error NER response."""
        response = NERResponse(
            success=False,
            entities=[],
            total_entities=0,
            error_message="NER processing failed due to invalid input."
        )
        
        assert response.success is False
        assert len(response.entities) == 0
        assert response.total_entities == 0
        assert "NER processing failed" in response.error_message


class TestBatchNERRequest:
    """Test the BatchNERRequest model."""

    def test_valid_batch_request(self):
        """Test valid batch NER request."""
        request = BatchNERRequest(
            texts=[
                "Patient Mary Smith has hypertension.",
                "Dr. Johnson prescribed antibiotics.",
                "Blood pressure is 140/90 mmHg."
            ],
            entity_types=[EntityType.PERSON, EntityType.CONDITION, EntityType.MEDICATION],
            include_confidence=True,
            parallel_processing=True,
            batch_size=10
        )
        
        assert len(request.texts) == 3
        assert EntityType.PERSON in request.entity_types
        assert request.include_confidence is True
        assert request.parallel_processing is True
        assert request.batch_size == 10

    def test_empty_texts_validation(self):
        """Test validation for empty texts list."""
        with pytest.raises(ValidationError) as exc_info:
            BatchNERRequest(texts=[])
        assert "ensure this value has at least 1 items" in str(exc_info.value)

    def test_too_many_texts_validation(self):
        """Test validation for too many texts."""
        with pytest.raises(ValidationError) as exc_info:
            BatchNERRequest(texts=["text"] * 1001)  # Above maximum
        assert "ensure this value has at most 1000 items" in str(exc_info.value)

    def test_invalid_batch_size(self):
        """Test invalid batch size validation."""
        with pytest.raises(ValidationError) as exc_info:
            BatchNERRequest(
                texts=["test text"],
                batch_size=0  # Below minimum
            )
        assert "ensure this value is greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            BatchNERRequest(
                texts=["test text"],
                batch_size=101  # Above maximum
            )
        assert "ensure this value is less than or equal to 100" in str(exc_info.value)


class TestBatchNERResponse:
    """Test the BatchNERResponse model."""

    def test_valid_batch_response(self):
        """Test valid batch NER response."""
        response = BatchNERResponse(
            success=True,
            results=[
                {
                    "text_index": 0,
                    "entities": [
                        ExtractedEntity(
                            text="Mary Smith",
                            label="PERSON",
                            start=8,
                            end=18,
                            confidence=0.94
                        )
                    ],
                    "processing_time": 0.12
                },
                {
                    "text_index": 1,
                    "entities": [
                        ExtractedEntity(
                            text="Dr. Johnson",
                            label="PERSON",
                            start=0,
                            end=11,
                            confidence=0.96
                        )
                    ],
                    "processing_time": 0.15
                }
            ],
            total_texts=2,
            total_entities=2,
            avg_processing_time=0.135,
            parallel_processed=True
        )
        
        assert response.success is True
        assert len(response.results) == 2
        assert response.total_texts == 2
        assert response.total_entities == 2
        assert response.avg_processing_time == 0.135
        assert response.parallel_processed is True


class TestEntityValidationRequest:
    """Test the EntityValidationRequest model."""

    def test_valid_validation_request(self):
        """Test valid entity validation request."""
        request = EntityValidationRequest(
            entities=[
                ExtractedEntity(
                    text="John Doe",
                    label="PERSON",
                    start=8,
                    end=16,
                    confidence=0.95
                ),
                ExtractedEntity(
                    text="diabetes",
                    label="CONDITION",
                    start=25,
                    end=33,
                    confidence=0.88
                )
            ],
            context="Patient John Doe has diabetes and requires treatment.",
            validation_rules=["medical_terminology", "person_names", "consistency_check"],
            strict_validation=True
        )
        
        assert len(request.entities) == 2
        assert "diabetes" in request.context
        assert "medical_terminology" in request.validation_rules
        assert request.strict_validation is True

    def test_empty_entities_validation(self):
        """Test validation for empty entities list."""
        with pytest.raises(ValidationError) as exc_info:
            EntityValidationRequest(
                entities=[],
                context="Some context"
            )
        assert "ensure this value has at least 1 items" in str(exc_info.value)


class TestEntityValidationResponse:
    """Test the EntityValidationResponse model."""

    def test_valid_validation_response(self):
        """Test valid entity validation response."""
        response = EntityValidationResponse(
            success=True,
            validated_entities=[
                {
                    "text": "John Doe",
                    "label": "PERSON",
                    "start": 8,
                    "end": 16,
                    "confidence": 0.95,
                    "validation_score": 0.98,
                    "is_valid": True,
                    "validation_details": {
                        "name_format_valid": True,
                        "context_appropriate": True
                    }
                }
            ],
            overall_validation_score=0.96,
            validation_summary={
                "total_entities": 1,
                "valid_entities": 1,
                "invalid_entities": 0,
                "validation_rules_applied": ["person_names", "context_check"]
            },
            processing_time=0.08
        )
        
        assert response.success is True
        assert len(response.validated_entities) == 1
        assert response.overall_validation_score == 0.96
        assert response.validation_summary["valid_entities"] == 1
        assert response.processing_time == 0.08
