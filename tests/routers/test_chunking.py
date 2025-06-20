"""
Comprehensive tests for the chunking router endpoints.

This module tests all endpoints in the chunking router, including:
- Service info endpoints
- Document chunking operations
- Timeline processing
- Structure analysis
- Error handling and validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.schemas.chunking import ChunkingStrategy


class TestChunkingRouter:
    """Test class for chunking router endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_chunk_request(self):
        """Sample chunk request data."""
        return {
            "text": "Patient John Doe visited on 2023-01-15. He complained of chest pain and shortness of breath. Physical examination revealed elevated blood pressure. ECG showed normal sinus rhythm. Patient was prescribed medication and advised follow-up.",
            "strategy": "semantic",
            "max_chunk_size": 500,
            "overlap": 50,
            "preserve_structure": True,
            "extract_entities": True
        }

    @pytest.fixture
    def sample_timeline_request(self):
        """Sample timeline request data."""
        return {
            "chunks": [
                {
                    "id": "chunk_1",
                    "text": "Patient visited on 2023-01-15 with chest pain.",
                    "metadata": {
                        "visit_date": "2023-01-15",
                        "chief_complaint": "chest pain"
                    }
                },
                {
                    "id": "chunk_2", 
                    "text": "Follow-up visit on 2023-02-01 showed improvement.",
                    "metadata": {
                        "visit_date": "2023-02-01",
                        "status": "improved"
                    }
                }
            ],
            "sort_by": "date",
            "include_metadata": True
        }

    def test_get_chunking_info(self, client):
        """Test getting chunking service information."""
        response = client.get("/chunking/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["service_name"] == "MRIA Chunking Service"
        assert data["service_type"] == "text_processor"
        assert data["version"] == "1.0.0"
        assert data["status"] == "active"
        assert "capabilities" in data
        assert "document_segmentation" in data["capabilities"]
        assert "semantic_chunking" in data["capabilities"]

    @patch('app.services.chunker.chunker.process_text')
    def test_chunk_text_success(self, mock_process, client, sample_chunk_request):
        """Test successful text chunking."""
        # Mock chunker response
        mock_process.return_value = {
            "chunks": [
                {
                    "id": "chunk_1",
                    "text": "Patient John Doe visited on 2023-01-15.",
                    "start_pos": 0,
                    "end_pos": 45,
                    "metadata": {
                        "entities": ["John Doe", "2023-01-15"],
                        "chunk_type": "visit_info"
                    }
                },
                {
                    "id": "chunk_2", 
                    "text": "He complained of chest pain and shortness of breath.",
                    "start_pos": 46,
                    "end_pos": 98,
                    "metadata": {
                        "entities": ["chest pain", "shortness of breath"],
                        "chunk_type": "symptoms"
                    }
                }
            ],
            "total_chunks": 2,
            "processing_time": 0.5,
            "strategy_used": "semantic"
        }

        response = client.post("/chunking/chunk", json=sample_chunk_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["chunks"]) == 2
        assert data["total_chunks"] == 2
        assert data["strategy_used"] == "semantic"
        assert "processing_time" in data

        # Verify chunker was called with correct parameters
        mock_process.assert_called_once()
        call_args = mock_process.call_args[0][0]
        assert call_args.text == sample_chunk_request["text"]
        assert call_args.strategy == ChunkingStrategy.SEMANTIC

    def test_chunk_text_invalid_request(self, client):
        """Test chunking with invalid request data."""
        invalid_request = {
            "text": "",  # Empty text
            "strategy": "invalid_strategy",
            "max_chunk_size": -1  # Invalid size
        }

        response = client.post("/chunking/chunk", json=invalid_request)
        
        assert response.status_code == 422
        assert "validation error" in response.json()["detail"][0]["type"]

    def test_chunk_text_missing_text(self, client):
        """Test chunking without required text field."""
        invalid_request = {
            "strategy": "semantic",
            "max_chunk_size": 500
            # Missing 'text' field
        }

        response = client.post("/chunking/chunk", json=invalid_request)
        
        assert response.status_code == 422

    @patch('app.services.chunker.chunker.process_text')
    def test_chunk_text_processing_error(self, mock_process, client, sample_chunk_request):
        """Test chunking when processing fails."""
        mock_process.side_effect = Exception("Processing failed")

        response = client.post("/chunking/chunk", json=sample_chunk_request)
        
        assert response.status_code == 500
        assert "Chunking processing failed" in response.json()["detail"]

    @patch('app.services.chunker.chunker.create_timeline')
    def test_create_timeline_success(self, mock_timeline, client, sample_timeline_request):
        """Test successful timeline creation."""
        mock_timeline.return_value = {
            "timeline": [
                {
                    "date": "2023-01-15",
                    "events": [
                        {
                            "chunk_id": "chunk_1",
                            "text": "Patient visited on 2023-01-15 with chest pain.",
                            "event_type": "visit",
                            "metadata": {
                                "chief_complaint": "chest pain"
                            }
                        }
                    ]
                },
                {
                    "date": "2023-02-01",
                    "events": [
                        {
                            "chunk_id": "chunk_2",
                            "text": "Follow-up visit on 2023-02-01 showed improvement.",
                            "event_type": "follow_up",
                            "metadata": {
                                "status": "improved"
                            }
                        }
                    ]
                }
            ],
            "total_events": 2,
            "date_range": {
                "start": "2023-01-15",
                "end": "2023-02-01"
            }
        }

        response = client.post("/chunking/timeline", json=sample_timeline_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["timeline"]) == 2
        assert data["total_events"] == 2
        assert "date_range" in data

    def test_create_timeline_empty_chunks(self, client):
        """Test timeline creation with empty chunks."""
        empty_request = {
            "chunks": [],
            "sort_by": "date"
        }

        response = client.post("/chunking/timeline", json=empty_request)
        
        assert response.status_code == 422

    @patch('app.services.chunker.chunker.analyze_structure')
    def test_analyze_structure_success(self, mock_analyze, client):
        """Test successful structure analysis."""
        request_data = {
            "chunks": [
                {
                    "id": "chunk_1",
                    "text": "Patient John Doe visited on 2023-01-15.",
                    "metadata": {"section": "visit_info"}
                },
                {
                    "id": "chunk_2",
                    "text": "Physical examination revealed elevated blood pressure.",
                    "metadata": {"section": "examination"}
                }
            ],
            "analysis_type": "medical_structure"
        }

        mock_analyze.return_value = {
            "structure": {
                "sections": [
                    {
                        "name": "visit_info",
                        "chunks": ["chunk_1"],
                        "confidence": 0.95
                    },
                    {
                        "name": "examination",
                        "chunks": ["chunk_2"],
                        "confidence": 0.88
                    }
                ],
                "relationships": [
                    {
                        "source": "chunk_1",
                        "target": "chunk_2",
                        "type": "temporal",
                        "confidence": 0.82
                    }
                ]
            },
            "quality_score": 0.91,
            "analysis_metadata": {
                "total_sections": 2,
                "avg_confidence": 0.915
            }
        }

        response = client.post("/chunking/analyze-structure", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "structure" in data
        assert data["quality_score"] == 0.91
        assert len(data["structure"]["sections"]) == 2

    def test_analyze_structure_invalid_type(self, client):
        """Test structure analysis with invalid analysis type."""
        request_data = {
            "chunks": [
                {
                    "id": "chunk_1",
                    "text": "Some text",
                    "metadata": {}
                }
            ],
            "analysis_type": "invalid_type"
        }

        response = client.post("/chunking/analyze-structure", json=request_data)
        
        assert response.status_code == 422

    def test_get_chunking_strategies(self, client):
        """Test getting available chunking strategies."""
        response = client.get("/chunking/strategies")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "strategies" in data
        strategies = data["strategies"]
        
        # Check that all expected strategies are present
        strategy_names = [s["name"] for s in strategies]
        assert "semantic" in strategy_names
        assert "visit_based" in strategy_names
        assert "topic_based" in strategy_names
        assert "temporal" in strategy_names

    def test_get_chunking_config(self, client):
        """Test getting chunking configuration."""
        response = client.get("/chunking/config")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "default_strategy" in data
        assert "max_chunk_size" in data
        assert "default_overlap" in data
        assert "supported_formats" in data

    @patch('app.services.chunker.chunker.update_config')
    def test_update_chunking_config(self, mock_update, client):
        """Test updating chunking configuration."""
        config_data = {
            "default_strategy": "semantic",
            "max_chunk_size": 1000,
            "default_overlap": 100,
            "preserve_structure": True
        }

        mock_update.return_value = True

        response = client.put("/chunking/config", json=config_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        mock_update.assert_called_once()

    def test_health_check(self, client):
        """Test chunking service health check."""
        response = client.get("/chunking/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "service" in data
        assert data["service"] == "chunking"

    @patch('app.services.chunker.chunker.get_metrics')
    def test_get_chunking_metrics(self, mock_metrics, client):
        """Test getting chunking service metrics."""
        mock_metrics.return_value = {
            "total_chunks_processed": 1250,
            "avg_processing_time": 0.75,
            "success_rate": 0.98,
            "error_rate": 0.02,
            "active_sessions": 3
        }

        response = client.get("/chunking/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_chunks_processed"] == 1250
        assert data["avg_processing_time"] == 0.75
        assert data["success_rate"] == 0.98
