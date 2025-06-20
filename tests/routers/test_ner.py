"""
Comprehensive tests for the NER router endpoints.

This module tests all endpoints in the NER router, including:
- Service info endpoints
- Entity extraction operations
- Batch processing
- Entity validation
- Error handling and validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app


class TestNERRouter:
    """Test class for NER router endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_ner_request(self):
        """Sample NER request data."""
        return {
            "text": "Patient John Doe was diagnosed with diabetes and prescribed metformin. Follow-up appointment scheduled for next week.",
            "entity_types": ["PERSON", "CONDITION", "MEDICATION", "DATE"],
            "include_confidence": True,
            "include_context": True,
            "language": "en"
        }

    @pytest.fixture
    def sample_batch_request(self):
        """Sample batch NER request data."""
        return {
            "texts": [
                "Patient Mary Smith has hypertension and takes lisinopril daily.",
                "Dr. Johnson prescribed antibiotics for the infection.",
                "The patient's blood pressure is 140/90 mmHg."
            ],
            "entity_types": ["PERSON", "CONDITION", "MEDICATION", "MEASUREMENT"],
            "include_confidence": True
        }

    def test_get_ner_info(self, client):
        """Test getting NER service information."""
        response = client.get("/ner/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["service_name"] == "MRIA Named Entity Recognition Service"
        assert data["service_type"] == "nlp_processor"
        assert data["version"] == "1.0.0"
        assert data["status"] == "active"
        assert "capabilities" in data
        assert "medical_entity_extraction" in data["capabilities"]
        assert "supported_entity_types" in data

    @patch('app.services.ner_processor.get_ner_processor')
    def test_extract_entities_success(self, mock_get_processor, client, sample_ner_request):
        """Test successful entity extraction."""
        # Mock NER processor
        mock_processor = MagicMock()
        mock_processor.extract_entities.return_value = {
            "entities": [
                {
                    "text": "John Doe",
                    "label": "PERSON",
                    "start": 8,
                    "end": 16,
                    "confidence": 0.95,
                    "context": "Patient John Doe was diagnosed"
                },
                {
                    "text": "diabetes",
                    "label": "CONDITION",
                    "start": 32,
                    "end": 40,
                    "confidence": 0.92,
                    "context": "diagnosed with diabetes and"
                },
                {
                    "text": "metformin",
                    "label": "MEDICATION",
                    "start": 55,
                    "end": 64,
                    "confidence": 0.98,
                    "context": "prescribed metformin. Follow-up"
                }
            ],
            "total_entities": 3,
            "processing_time": 0.25,
            "model_version": "1.0.0"
        }
        mock_get_processor.return_value = mock_processor

        response = client.post("/ner/extract", json=sample_ner_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["entities"]) == 3
        assert data["total_entities"] == 3
        assert "processing_time" in data
        
        # Check entity details
        entities = data["entities"]
        person_entity = next(e for e in entities if e["label"] == "PERSON")
        assert person_entity["text"] == "John Doe"
        assert person_entity["confidence"] == 0.95

    def test_extract_entities_invalid_request(self, client):
        """Test entity extraction with invalid request."""
        invalid_request = {
            "text": "",  # Empty text
            "entity_types": ["INVALID_TYPE"],
            "language": "invalid_lang"
        }

        response = client.post("/ner/extract", json=invalid_request)
        
        assert response.status_code == 422

    def test_extract_entities_missing_text(self, client):
        """Test entity extraction without required text field."""
        invalid_request = {
            "entity_types": ["PERSON"],
            "include_confidence": True
            # Missing 'text' field
        }

        response = client.post("/ner/extract", json=invalid_request)
        
        assert response.status_code == 422

    @patch('app.services.ner_processor.get_ner_processor')
    def test_extract_entities_processing_error(self, mock_get_processor, client, sample_ner_request):
        """Test entity extraction when processing fails."""
        mock_processor = MagicMock()
        mock_processor.extract_entities.side_effect = Exception("NER processing failed")
        mock_get_processor.return_value = mock_processor

        response = client.post("/ner/extract", json=sample_ner_request)
        
        assert response.status_code == 500
        assert "Entity extraction failed" in response.json()["detail"]

    @patch('app.services.ner_processor.get_ner_processor')
    def test_batch_extract_success(self, mock_get_processor, client, sample_batch_request):
        """Test successful batch entity extraction."""
        mock_processor = MagicMock()
        mock_processor.batch_extract_entities.return_value = {
            "results": [
                {
                    "text_index": 0,
                    "entities": [
                        {
                            "text": "Mary Smith",
                            "label": "PERSON",
                            "start": 8,
                            "end": 18,
                            "confidence": 0.94
                        },
                        {
                            "text": "hypertension",
                            "label": "CONDITION",
                            "start": 23,
                            "end": 35,
                            "confidence": 0.91
                        }
                    ]
                },
                {
                    "text_index": 1,
                    "entities": [
                        {
                            "text": "Dr. Johnson",
                            "label": "PERSON",
                            "start": 0,
                            "end": 11,
                            "confidence": 0.96
                        }
                    ]
                }
            ],
            "total_texts": 3,
            "total_entities": 3,
            "processing_time": 0.8
        }
        mock_get_processor.return_value = mock_processor

        response = client.post("/ner/batch-extract", json=sample_batch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["results"]) == 2
        assert data["total_texts"] == 3
        assert data["total_entities"] == 3

    def test_batch_extract_empty_texts(self, client):
        """Test batch extraction with empty texts list."""
        empty_request = {
            "texts": [],
            "entity_types": ["PERSON"]
        }

        response = client.post("/ner/batch-extract", json=empty_request)
        
        assert response.status_code == 422

    @patch('app.services.ner_processor.get_ner_processor')
    def test_validate_entities_success(self, mock_get_processor, client):
        """Test successful entity validation."""
        validation_request = {
            "entities": [
                {
                    "text": "John Doe",
                    "label": "PERSON",
                    "start": 8,
                    "end": 16,
                    "confidence": 0.95
                },
                {
                    "text": "diabetes",
                    "label": "CONDITION",
                    "start": 32,
                    "end": 40,
                    "confidence": 0.92
                }
            ],
            "context": "Patient John Doe was diagnosed with diabetes",
            "validation_rules": ["medical_terminology", "person_names"]
        }

        mock_processor = MagicMock()
        mock_processor.validate_entities.return_value = {
            "validated_entities": [
                {
                    "text": "John Doe",
                    "label": "PERSON",
                    "start": 8,
                    "end": 16,
                    "confidence": 0.95,
                    "validation_score": 0.98,
                    "is_valid": True
                },
                {
                    "text": "diabetes",
                    "label": "CONDITION",
                    "start": 32,
                    "end": 40,
                    "confidence": 0.92,
                    "validation_score": 0.96,
                    "is_valid": True
                }
            ],
            "overall_validation_score": 0.97,
            "validation_summary": {
                "total_entities": 2,
                "valid_entities": 2,
                "invalid_entities": 0
            }
        }
        mock_get_processor.return_value = mock_processor

        response = client.post("/ner/validate", json=validation_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["validated_entities"]) == 2
        assert data["overall_validation_score"] == 0.97
        assert data["validation_summary"]["valid_entities"] == 2

    def test_get_supported_entity_types(self, client):
        """Test getting supported entity types."""
        response = client.get("/ner/entity-types")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "entity_types" in data
        entity_types = data["entity_types"]
        
        # Check that expected types are present
        type_names = [et["name"] for et in entity_types]
        assert "PERSON" in type_names
        assert "CONDITION" in type_names
        assert "MEDICATION" in type_names
        assert "DATE" in type_names

    @patch('app.services.ner_processor.is_ner_processor_ready')
    def test_health_check_healthy(self, mock_ready, client):
        """Test NER service health check when healthy."""
        mock_ready.return_value = True

        response = client.get("/ner/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "ner"
        assert "timestamp" in data

    @patch('app.services.ner_processor.is_ner_processor_ready')
    def test_health_check_unhealthy(self, mock_ready, client):
        """Test NER service health check when unhealthy."""
        mock_ready.return_value = False

        response = client.get("/ner/health")
        
        assert response.status_code == 503
        data = response.json()
        
        assert data["status"] == "unhealthy"
        assert data["service"] == "ner"

    @patch('app.services.ner_processor.get_ner_processor')
    def test_get_ner_models(self, mock_get_processor, client):
        """Test getting available NER models."""
        mock_processor = MagicMock()
        mock_processor.get_available_models.return_value = {
            "models": [
                {
                    "name": "medical_ner_v1",
                    "version": "1.0.0",
                    "language": "en",
                    "entity_types": ["PERSON", "CONDITION", "MEDICATION"],
                    "status": "active"
                },
                {
                    "name": "general_ner_v1",
                    "version": "1.0.0", 
                    "language": "en",
                    "entity_types": ["PERSON", "ORG", "GPE"],
                    "status": "active"
                }
            ],
            "default_model": "medical_ner_v1"
        }
        mock_get_processor.return_value = mock_processor

        response = client.get("/ner/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert len(data["models"]) == 2
        assert data["default_model"] == "medical_ner_v1"

    @patch('app.services.ner_processor.get_ner_processor')
    def test_get_ner_metrics(self, mock_get_processor, client):
        """Test getting NER service metrics."""
        mock_processor = MagicMock()
        mock_processor.get_metrics.return_value = {
            "total_extractions": 5420,
            "avg_processing_time": 0.35,
            "success_rate": 0.97,
            "error_rate": 0.03,
            "entities_extracted": {
                "PERSON": 1250,
                "CONDITION": 890,
                "MEDICATION": 650,
                "DATE": 320
            }
        }
        mock_get_processor.return_value = mock_processor

        response = client.get("/ner/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_extractions"] == 5420
        assert data["avg_processing_time"] == 0.35
        assert data["success_rate"] == 0.97
        assert "entities_extracted" in data

    def test_get_ner_config(self, client):
        """Test getting NER configuration."""
        response = client.get("/ner/config")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "default_model" in data
        assert "supported_languages" in data
        assert "confidence_threshold" in data
        assert "max_text_length" in data
