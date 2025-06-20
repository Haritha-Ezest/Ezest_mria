"""
Comprehensive tests for the graph router endpoints.

This module tests all endpoints in the graph router, including:
- Service info endpoints
- Graph operations
- Node and relationship management
- Query processing
- Error handling and validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app


class TestGraphRouter:
    """Test class for graph router endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_graph_data(self):
        """Sample graph data for testing."""
        return {
            "nodes": [
                {
                    "id": "patient_1",
                    "type": "Patient",
                    "properties": {
                        "name": "John Doe",
                        "age": 45,
                        "patient_id": "P001"
                    }
                },
                {
                    "id": "condition_1",
                    "type": "Condition",
                    "properties": {
                        "name": "Hypertension",
                        "icd_code": "I10"
                    }
                }
            ],
            "relationships": [
                {
                    "id": "rel_1",
                    "source": "patient_1",
                    "target": "condition_1",
                    "type": "HAS_CONDITION",
                    "properties": {
                        "diagnosed_date": "2023-01-15",
                        "severity": "moderate"
                    }
                }
            ]
        }

    def test_get_graph_info(self, client):
        """Test getting graph service information."""
        response = client.get("/graph/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "service_name" in data
        assert "service_type" in data
        assert "version" in data
        assert "capabilities" in data
        assert "supported_node_types" in data

    @patch('app.services.graph_client.GraphClient.create_graph')
    def test_create_graph_success(self, mock_create, client, sample_graph_data):
        """Test successful graph creation."""
        mock_create.return_value = {
            "graph_id": "graph_123",
            "nodes_created": 2,
            "relationships_created": 1,
            "processing_time": 0.8,
            "status": "success"
        }

        response = client.post("/graph/create", json=sample_graph_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["graph_id"] == "graph_123"
        assert data["nodes_created"] == 2
        assert data["relationships_created"] == 1

    def test_create_graph_invalid_data(self, client):
        """Test graph creation with invalid data."""
        invalid_data = {
            "nodes": [],  # Empty nodes
            "relationships": [
                {
                    "source": "missing_node",
                    "target": "another_missing_node",
                    "type": "INVALID_REL"
                }
            ]
        }

        response = client.post("/graph/create", json=invalid_data)
        
        assert response.status_code == 422

    @patch('app.services.graph_client.GraphClient.query_graph')
    def test_query_graph_success(self, mock_query, client):
        """Test successful graph querying."""
        query_data = {
            "query": "MATCH (p:Patient)-[r:HAS_CONDITION]->(c:Condition) RETURN p, r, c",
            "parameters": {},
            "limit": 100
        }

        mock_query.return_value = {
            "results": [
                {
                    "p": {
                        "id": "patient_1",
                        "properties": {"name": "John Doe", "age": 45}
                    },
                    "r": {
                        "type": "HAS_CONDITION",
                        "properties": {"diagnosed_date": "2023-01-15"}
                    },
                    "c": {
                        "id": "condition_1",
                        "properties": {"name": "Hypertension"}
                    }
                }
            ],
            "total_results": 1,
            "execution_time": 0.15
        }

        response = client.post("/graph/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["results"]) == 1
        assert data["total_results"] == 1
        assert "execution_time" in data

    def test_query_graph_invalid_query(self, client):
        """Test graph querying with invalid query."""
        invalid_query = {
            "query": "",  # Empty query
            "parameters": {}
        }

        response = client.post("/graph/query", json=invalid_query)
        
        assert response.status_code == 422

    @patch('app.services.graph_client.GraphClient.add_node')
    def test_add_node_success(self, mock_add, client):
        """Test successful node addition."""
        node_data = {
            "type": "Medication",
            "properties": {
                "name": "Lisinopril",
                "dosage": "10mg",
                "rxcui": "12345"
            }
        }

        mock_add.return_value = {
            "node_id": "med_1",
            "status": "created",
            "processing_time": 0.1
        }

        response = client.post("/graph/nodes", json=node_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["node_id"] == "med_1"
        assert data["status"] == "created"

    @patch('app.services.graph_client.GraphClient.get_node')
    def test_get_node_success(self, mock_get, client):
        """Test successful node retrieval."""
        node_id = "patient_1"
        
        mock_get.return_value = {
            "id": "patient_1",
            "type": "Patient",
            "properties": {
                "name": "John Doe",
                "age": 45,
                "patient_id": "P001"
            },
            "relationships": [
                {
                    "id": "rel_1",
                    "type": "HAS_CONDITION",
                    "target": "condition_1",
                    "direction": "outgoing"
                }
            ]
        }

        response = client.get(f"/graph/nodes/{node_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["node"]["id"] == "patient_1"
        assert data["node"]["type"] == "Patient"
        assert len(data["node"]["relationships"]) == 1

    def test_get_node_not_found(self, client):
        """Test getting non-existent node."""
        with patch('app.services.graph_client.GraphClient.get_node') as mock_get:
            mock_get.return_value = None
            
            response = client.get("/graph/nodes/nonexistent")
            
            assert response.status_code == 404

    @patch('app.services.graph_client.GraphClient.add_relationship')
    def test_add_relationship_success(self, mock_add, client):
        """Test successful relationship addition."""
        rel_data = {
            "source": "patient_1",
            "target": "med_1",
            "type": "PRESCRIBED",
            "properties": {
                "prescribed_date": "2023-01-15",
                "prescriber": "Dr. Smith"
            }
        }

        mock_add.return_value = {
            "relationship_id": "rel_2",
            "status": "created",
            "processing_time": 0.05
        }

        response = client.post("/graph/relationships", json=rel_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["relationship_id"] == "rel_2"
        assert data["status"] == "created"

    @patch('app.services.graph_client.GraphClient.get_schema')
    def test_get_graph_schema(self, mock_schema, client):
        """Test getting graph schema."""
        mock_schema.return_value = {
            "node_types": [
                {
                    "type": "Patient",
                    "properties": ["name", "age", "patient_id"],
                    "count": 150
                },
                {
                    "type": "Condition",
                    "properties": ["name", "icd_code", "description"],
                    "count": 89
                }
            ],
            "relationship_types": [
                {
                    "type": "HAS_CONDITION",
                    "properties": ["diagnosed_date", "severity"],
                    "count": 245
                }
            ],
            "total_nodes": 239,
            "total_relationships": 245
        }

        response = client.get("/graph/schema")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "node_types" in data
        assert "relationship_types" in data
        assert data["total_nodes"] == 239
        assert data["total_relationships"] == 245

    @patch('app.services.graph_client.GraphClient.search_nodes')
    def test_search_nodes_success(self, mock_search, client):
        """Test successful node search."""
        search_data = {
            "query": "John",
            "node_types": ["Patient"],
            "properties": ["name"],
            "limit": 10
        }

        mock_search.return_value = {
            "results": [
                {
                    "id": "patient_1",
                    "type": "Patient",
                    "properties": {"name": "John Doe", "age": 45},
                    "score": 0.95
                },
                {
                    "id": "patient_5",
                    "type": "Patient", 
                    "properties": {"name": "John Smith", "age": 38},
                    "score": 0.88
                }
            ],
            "total_results": 2,
            "search_time": 0.12
        }

        response = client.post("/graph/search", json=search_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["results"]) == 2
        assert data["total_results"] == 2

    def test_health_check(self, client):
        """Test graph service health check."""
        response = client.get("/graph/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "graph"
        assert "timestamp" in data

    @patch('app.services.graph_client.GraphClient.get_metrics')
    def test_get_graph_metrics(self, mock_metrics, client):
        """Test getting graph service metrics."""
        mock_metrics.return_value = {
            "total_nodes": 1250,
            "total_relationships": 3480,
            "queries_executed": 850,
            "avg_query_time": 0.25,
            "cache_hit_rate": 0.78,
            "storage_size_mb": 45.2
        }

        response = client.get("/graph/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_nodes"] == 1250
        assert data["total_relationships"] == 3480
        assert data["avg_query_time"] == 0.25

    @patch('app.services.graph_client.GraphClient.delete_node')
    def test_delete_node_success(self, mock_delete, client):
        """Test successful node deletion."""
        node_id = "patient_1"
        
        mock_delete.return_value = {
            "deleted": True,
            "relationships_deleted": 3,
            "processing_time": 0.08
        }

        response = client.delete(f"/graph/nodes/{node_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["deleted"] is True
        assert data["relationships_deleted"] == 3
