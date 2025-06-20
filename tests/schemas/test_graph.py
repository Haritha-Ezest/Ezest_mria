"""
Comprehensive tests for the graph schemas.

This module tests all Pydantic models in the graph schemas, including:
- Graph node and relationship models
- Request and response models
- Field validation and constraints
- Query models and validation
"""

import pytest
from pydantic import ValidationError

from app.schemas.graph import (
    GraphNode, GraphRelationship, GraphData, GraphQuery, GraphResponse,
    NodeSearchRequest, NodeSearchResponse, GraphSchema, GraphMetrics
)


class TestGraphNode:
    """Test the GraphNode model."""

    def test_valid_node(self):
        """Test valid graph node."""
        node = GraphNode(
            id="patient_001",
            type="Patient",
            properties={
                "name": "John Doe",
                "age": 45,
                "patient_id": "P001",
                "status": "active"
            },
            labels=["Patient", "Person"],
            created_at="2023-01-15T10:00:00Z",
            updated_at="2023-01-15T10:00:00Z"
        )
        
        assert node.id == "patient_001"
        assert node.type == "Patient"
        assert node.properties["name"] == "John Doe"
        assert node.properties["age"] == 45
        assert "Patient" in node.labels
        assert "Person" in node.labels

    def test_node_minimal_fields(self):
        """Test node with only required fields."""
        node = GraphNode(
            id="simple_node",
            type="SimpleType"
        )
        
        assert node.id == "simple_node"
        assert node.type == "SimpleType"
        assert node.properties == {}
        assert node.labels == []

    def test_invalid_node_id(self):
        """Test invalid node ID validation."""
        with pytest.raises(ValidationError) as exc_info:
            GraphNode(
                id="",  # Empty ID
                type="Patient"
            )
        assert "ensure this value has at least 1 characters" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            GraphNode(
                id="x" * 101,  # Too long
                type="Patient"
            )
        assert "ensure this value has at most 100 characters" in str(exc_info.value)

    def test_invalid_node_type(self):
        """Test invalid node type validation."""
        with pytest.raises(ValidationError) as exc_info:
            GraphNode(
                id="test_node",
                type=""  # Empty type
            )
        assert "ensure this value has at least 1 characters" in str(exc_info.value)


class TestGraphRelationship:
    """Test the GraphRelationship model."""

    def test_valid_relationship(self):
        """Test valid graph relationship."""
        relationship = GraphRelationship(
            id="rel_001",
            source="patient_001",
            target="condition_001",
            type="HAS_CONDITION",
            properties={
                "diagnosed_date": "2023-01-15",
                "severity": "moderate",
                "confirmed": True
            },
            weight=1.0,
            created_at="2023-01-15T10:00:00Z"
        )
        
        assert relationship.id == "rel_001"
        assert relationship.source == "patient_001"
        assert relationship.target == "condition_001"
        assert relationship.type == "HAS_CONDITION"
        assert relationship.properties["severity"] == "moderate"
        assert relationship.weight == 1.0

    def test_relationship_minimal_fields(self):
        """Test relationship with only required fields."""
        relationship = GraphRelationship(
            source="node1",
            target="node2",
            type="CONNECTED_TO"
        )
        
        assert relationship.source == "node1"
        assert relationship.target == "node2"
        assert relationship.type == "CONNECTED_TO"
        assert relationship.properties == {}
        assert relationship.weight == 1.0

    def test_invalid_relationship_weight(self):
        """Test invalid relationship weight validation."""
        with pytest.raises(ValidationError) as exc_info:
            GraphRelationship(
                source="node1",
                target="node2",
                type="CONNECTED",
                weight=-0.1  # Below minimum
            )
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            GraphRelationship(
                source="node1",
                target="node2",
                type="CONNECTED",
                weight=1.1  # Above maximum
            )
        assert "ensure this value is less than or equal to 1" in str(exc_info.value)

    def test_self_referencing_relationship(self):
        """Test self-referencing relationship validation."""
        with pytest.raises(ValidationError) as exc_info:
            GraphRelationship(
                source="node1",
                target="node1",  # Same as source
                type="SELF_REF"
            )
        assert "source and target cannot be the same" in str(exc_info.value)


class TestGraphData:
    """Test the GraphData model."""

    def test_valid_graph_data(self):
        """Test valid graph data."""
        graph_data = GraphData(
            nodes=[
                GraphNode(
                    id="patient_1",
                    type="Patient",
                    properties={"name": "John Doe"}
                ),
                GraphNode(
                    id="condition_1",
                    type="Condition",
                    properties={"name": "Hypertension"}
                )
            ],
            relationships=[
                GraphRelationship(
                    source="patient_1",
                    target="condition_1",
                    type="HAS_CONDITION"
                )
            ],
            metadata={
                "source": "medical_records",
                "version": "1.0",
                "created_by": "import_system"
            }
        )
        
        assert len(graph_data.nodes) == 2
        assert len(graph_data.relationships) == 1
        assert graph_data.metadata["source"] == "medical_records"

    def test_empty_graph_data(self):
        """Test empty graph data."""
        graph_data = GraphData(
            nodes=[],
            relationships=[]
        )
        
        assert len(graph_data.nodes) == 0
        assert len(graph_data.relationships) == 0
        assert graph_data.metadata == {}

    def test_relationship_references_missing_node(self):
        """Test relationship referencing non-existent node."""
        with pytest.raises(ValidationError) as exc_info:
            GraphData(
                nodes=[
                    GraphNode(id="node1", type="Type1")
                ],
                relationships=[
                    GraphRelationship(
                        source="node1",
                        target="missing_node",  # Not in nodes
                        type="CONNECTED"
                    )
                ]
            )
        assert "relationship target 'missing_node' not found in nodes" in str(exc_info.value)


class TestGraphQuery:
    """Test the GraphQuery model."""

    def test_valid_query(self):
        """Test valid graph query."""
        query = GraphQuery(
            query="MATCH (p:Patient)-[r:HAS_CONDITION]->(c:Condition) RETURN p, r, c",
            parameters={
                "patient_id": "P001",
                "condition_type": "chronic"
            },
            limit=100,
            offset=0,
            timeout=30
        )
        
        assert "MATCH (p:Patient)" in query.query
        assert query.parameters["patient_id"] == "P001"
        assert query.limit == 100
        assert query.offset == 0
        assert query.timeout == 30

    def test_query_with_defaults(self):
        """Test query with default values."""
        query = GraphQuery(
            query="MATCH (n) RETURN n"
        )
        
        assert query.parameters == {}
        assert query.limit == 1000
        assert query.offset == 0
        assert query.timeout == 60

    def test_invalid_query(self):
        """Test invalid query validation."""
        with pytest.raises(ValidationError) as exc_info:
            GraphQuery(query="")  # Empty query
        assert "ensure this value has at least 1 characters" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            GraphQuery(query="x" * 10001)  # Too long
        assert "ensure this value has at most 10000 characters" in str(exc_info.value)

    def test_invalid_limit(self):
        """Test invalid limit validation."""
        with pytest.raises(ValidationError) as exc_info:
            GraphQuery(
                query="MATCH (n) RETURN n",
                limit=0  # Below minimum
            )
        assert "ensure this value is greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            GraphQuery(
                query="MATCH (n) RETURN n",
                limit=10001  # Above maximum
            )
        assert "ensure this value is less than or equal to 10000" in str(exc_info.value)


class TestGraphResponse:
    """Test the GraphResponse model."""

    def test_valid_response(self):
        """Test valid graph response."""
        response = GraphResponse(
            success=True,
            data={
                "nodes": [
                    {
                        "id": "patient_1",
                        "type": "Patient",
                        "properties": {"name": "John Doe"}
                    }
                ],
                "relationships": [
                    {
                        "source": "patient_1",
                        "target": "condition_1",
                        "type": "HAS_CONDITION"
                    }
                ]
            },
            total_results=1,
            execution_time=0.15,
            query_plan="NodeByLabelScan",
            cached=False,
            metadata={
                "query_hash": "abc123",
                "database_version": "4.4.0"
            }
        )
        
        assert response.success is True
        assert "nodes" in response.data
        assert response.total_results == 1
        assert response.execution_time == 0.15
        assert response.cached is False

    def test_error_response(self):
        """Test error graph response."""
        response = GraphResponse(
            success=False,
            data={},
            total_results=0,
            execution_time=0.0,
            error_message="Query execution failed: syntax error"
        )
        
        assert response.success is False
        assert response.data == {}
        assert response.total_results == 0
        assert "syntax error" in response.error_message


class TestNodeSearchRequest:
    """Test the NodeSearchRequest model."""

    def test_valid_search_request(self):
        """Test valid node search request."""
        request = NodeSearchRequest(
            query="John",
            node_types=["Patient", "Doctor"],
            properties=["name", "email"],
            fuzzy_search=True,
            limit=50,
            min_score=0.7
        )
        
        assert request.query == "John"
        assert "Patient" in request.node_types
        assert "name" in request.properties
        assert request.fuzzy_search is True
        assert request.limit == 50
        assert request.min_score == 0.7

    def test_search_request_defaults(self):
        """Test search request with defaults."""
        request = NodeSearchRequest(
            query="search term"
        )
        
        assert request.node_types == []
        assert request.properties == []
        assert request.fuzzy_search is False
        assert request.limit == 100
        assert request.min_score == 0.0

    def test_invalid_search_query(self):
        """Test invalid search query validation."""
        with pytest.raises(ValidationError) as exc_info:
            NodeSearchRequest(query="")  # Empty query
        assert "ensure this value has at least 1 characters" in str(exc_info.value)

    def test_invalid_min_score(self):
        """Test invalid minimum score validation."""
        with pytest.raises(ValidationError) as exc_info:
            NodeSearchRequest(
                query="test",
                min_score=-0.1  # Below 0
            )
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            NodeSearchRequest(
                query="test",
                min_score=1.1  # Above 1
            )
        assert "ensure this value is less than or equal to 1" in str(exc_info.value)


class TestNodeSearchResponse:
    """Test the NodeSearchResponse model."""

    def test_valid_search_response(self):
        """Test valid node search response."""
        response = NodeSearchResponse(
            success=True,
            results=[
                {
                    "node": GraphNode(
                        id="patient_1",
                        type="Patient",
                        properties={"name": "John Doe"}
                    ),
                    "score": 0.95,
                    "matched_properties": ["name"]
                },
                {
                    "node": GraphNode(
                        id="doctor_1",
                        type="Doctor",
                        properties={"name": "Dr. John Smith"}
                    ),
                    "score": 0.88,
                    "matched_properties": ["name"]
                }
            ],
            total_results=2,
            search_time=0.12,
            query_used="John",
            fuzzy_search_applied=True
        )
        
        assert response.success is True
        assert len(response.results) == 2
        assert response.total_results == 2
        assert response.search_time == 0.12
        assert response.fuzzy_search_applied is True
        assert response.results[0]["score"] == 0.95


class TestGraphSchema:
    """Test the GraphSchema model."""

    def test_valid_schema(self):
        """Test valid graph schema."""
        schema = GraphSchema(
            node_types=[
                {
                    "type": "Patient",
                    "properties": ["name", "age", "patient_id"],
                    "constraints": ["UNIQUE patient_id"],
                    "indexes": ["name", "patient_id"],
                    "count": 150
                },
                {
                    "type": "Condition",
                    "properties": ["name", "icd_code", "description"],
                    "constraints": [],
                    "indexes": ["name", "icd_code"],
                    "count": 89
                }
            ],
            relationship_types=[
                {
                    "type": "HAS_CONDITION",
                    "properties": ["diagnosed_date", "severity"],
                    "constraints": [],
                    "count": 245
                }
            ],
            total_nodes=239,
            total_relationships=245,
            database_info={
                "version": "4.4.0",
                "edition": "community",
                "storage_size_mb": 45.2
            }
        )
        
        assert len(schema.node_types) == 2
        assert len(schema.relationship_types) == 1
        assert schema.total_nodes == 239
        assert schema.total_relationships == 245
        assert schema.database_info["version"] == "4.4.0"


class TestGraphMetrics:
    """Test the GraphMetrics model."""

    def test_valid_metrics(self):
        """Test valid graph metrics."""
        metrics = GraphMetrics(
            total_nodes=1250,
            total_relationships=3480,
            node_types_count=8,
            relationship_types_count=12,
            queries_executed=850,
            avg_query_time=0.25,
            cache_hit_rate=0.78,
            storage_size_mb=45.2,
            memory_usage_mb=128.5,
            active_connections=3,
            uptime_seconds=86400
        )
        
        assert metrics.total_nodes == 1250
        assert metrics.total_relationships == 3480
        assert metrics.node_types_count == 8
        assert metrics.relationship_types_count == 12
        assert metrics.queries_executed == 850
        assert metrics.avg_query_time == 0.25
        assert metrics.cache_hit_rate == 0.78
        assert metrics.storage_size_mb == 45.2
