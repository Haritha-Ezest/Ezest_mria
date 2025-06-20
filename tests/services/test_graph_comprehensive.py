"""
Comprehensive test suite for the Knowledge Graph Builder Agent.

This module provides extensive testing coverage for the Neo4j-based Knowledge Graph Builder Agent,
including all core functionality, advanced analytics, error handling, and integration scenarios.

Test coverage includes:
- Patient graph creation and management
- Medical entity relationships and temporal modeling
- Cross-patient pattern analysis and insights generation
- Knowledge base integration (ICD-10, SNOMED-CT, RxNorm)
- Advanced graph queries and analytics
- Performance optimization and caching
- Error handling and recovery
- Integration with other MRIA agents
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

# MRIA imports
from app.services.graph_client import Neo4jGraphClient
from app.schemas.graph import (
    PatientGraphRequest, GraphQueryRequest,
    PatientTimelineResponse, GraphInsightResponse
)


class TestNeo4jGraphClientInitialization:
    """Test cases for Neo4j graph client initialization and configuration."""
    
    @pytest.fixture
    def mock_neo4j_driver(self):
        """Mock Neo4j driver for testing."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            mock_driver.return_value.close = AsyncMock()
            yield mock_driver
    
    def test_graph_client_initialization_success(self, mock_neo4j_driver):
        """Test successful graph client initialization."""
        client = Neo4jGraphClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test_password"
        )
        
        assert client.uri == "bolt://localhost:7687"
        assert client.user == "neo4j"
        assert client.password == "test_password"
        mock_neo4j_driver.assert_called_once()
    
    def test_graph_client_initialization_with_custom_config(self, mock_neo4j_driver):
        """Test graph client initialization with custom configuration."""
        client = Neo4jGraphClient(
            uri="neo4j://production:7687",
            user="admin",
            password="secure_password",
            database="medical_graph",
            max_connection_lifetime=300,
            max_connection_pool_size=100
        )
        
        assert client.database == "medical_graph"
        mock_neo4j_driver.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graph_client_connection_verification(self, mock_neo4j_driver):
        """Test graph client connection verification."""
        mock_session = AsyncMock()
        mock_neo4j_driver.return_value.session.return_value = mock_session
        mock_session.run.return_value.single.return_value = {"version": "5.15.0"}
        
        client = Neo4jGraphClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test_password"
        )
        
        # Test connection verification
        is_connected = await client.verify_connection()
        assert is_connected is True
        mock_session.run.assert_called_with("CALL dbms.components() YIELD name, versions RETURN name, versions[0] as version")
    
    @pytest.mark.asyncio
    async def test_graph_client_connection_failure(self, mock_neo4j_driver):
        """Test graph client connection failure handling."""
        mock_session = AsyncMock()
        mock_neo4j_driver.return_value.session.return_value = mock_session
        mock_session.run.side_effect = Exception("Connection failed")
        
        client = Neo4jGraphClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test_password"
        )
        
        is_connected = await client.verify_connection()
        assert is_connected is False


class TestPatientGraphCreation:
    """Test cases for patient graph creation and management."""
    
    @pytest.fixture
    def graph_client(self):
        """Create a mock graph client for testing."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            client = Neo4jGraphClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password"
            )
            client.driver = mock_driver.return_value
            client.session = mock_session
            return client
    
    @pytest.fixture
    def comprehensive_patient_request(self):
        """Comprehensive patient graph request for testing."""
        return PatientGraphRequest(
            patient_id="patient_123",
            visit_data={
                "date": "2024-01-15T10:30:00",
                "type": "initial_consultation",
                "location": "Primary Care Clinic",
                "provider": "Dr. Sarah Johnson",
                "visit_status": "completed",
                "chief_complaint": "increased thirst and frequent urination",
                "duration": 45,
                "follow_up_required": True
            },
            entities=[
                {
                    "text": "Type 2 Diabetes Mellitus",
                    "label": "CONDITION",
                    "start": 0,
                    "end": 25,
                    "confidence": 0.95,
                    "normalized_text": "Type 2 Diabetes Mellitus",
                    "entity_id": "condition_001",
                    "concept_code": "E11.9",
                    "semantic_type": "Disease or Syndrome",
                    "icd_code": "E11.9",
                    "knowledge_base_refs": ["ICD10:E11.9", "SNOMED:44054006"]
                },
                {
                    "text": "Metformin 500mg twice daily",
                    "label": "MEDICATION",
                    "start": 30,
                    "end": 57,
                    "confidence": 0.92,
                    "normalized_text": "Metformin",
                    "entity_id": "medication_001",
                    "concept_code": "6809",
                    "semantic_type": "Pharmacologic Substance",
                    "rxnorm_code": "6809",
                    "dosage": "500mg",
                    "frequency": "twice daily",
                    "knowledge_base_refs": ["RXNORM:6809"]
                },
                {
                    "text": "HbA1c 8.2%",
                    "label": "LAB_VALUE",
                    "start": 60,
                    "end": 71,
                    "confidence": 0.94,
                    "normalized_text": "Hemoglobin A1c",
                    "entity_id": "lab_001",
                    "test_name": "HbA1c",
                    "test_value": "8.2",
                    "test_unit": "%",
                    "reference_range": "< 7.0%",
                    "status": "abnormal"
                }
            ],
            relationships=[
                {
                    "source_entity": "condition_001",
                    "target_entity": "medication_001",
                    "relationship_type": "TREATED_WITH",
                    "confidence": 0.90,
                    "temporal_context": "concurrent",
                    "metadata": {
                        "start_date": "2024-01-15",
                        "indication": "glycemic control",
                        "prescribing_provider": "Dr. Sarah Johnson"
                    }
                },
                {
                    "source_entity": "lab_001",
                    "target_entity": "condition_001",
                    "relationship_type": "INDICATES",
                    "confidence": 0.93,
                    "temporal_context": "diagnostic",
                    "metadata": {
                        "interpretation": "elevated HbA1c consistent with diabetes diagnosis"
                    }
                }
            ],
            metadata={
                "job_id": "job_456",
                "processing_timestamp": "2024-01-15T11:00:00",
                "source": "comprehensive_medical_processing",
                "patient_demographics": {
                    "name": "John Smith",
                    "dob": "1970-05-20",
                    "gender": "male",
                    "mrn": "MRN123456",
                    "phone": "555-0123",
                    "emergency_contact": "Jane Smith (spouse) 555-0124"
                }
            }
        )
    
    @pytest.mark.asyncio
    async def test_create_comprehensive_patient_graph(self, graph_client, comprehensive_patient_request):
        """Test creation of comprehensive patient graph with all components."""
        # Mock successful graph creation
        mock_result = Mock()
        mock_result.single.return_value = {
            "patient_id": "patient_123",
            "nodes_created": 15,
            "relationships_created": 8,
            "properties_set": 45
        }
        graph_client.session.run.return_value = mock_result
        
        response = await graph_client.create_patient_graph(comprehensive_patient_request)
        
        assert response.success is True
        assert response.patient_id == "patient_123"
        assert response.nodes_created == 15
        assert response.relationships_created == 8
        assert "Patient graph created successfully" in response.message
        
        # Verify Cypher query was executed
        graph_client.session.run.assert_called()
        call_args = graph_client.session.run.call_args
        assert "CREATE (p:Patient" in call_args[0][0]
        assert "CREATE (v:Visit" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_update_patient_graph(self, graph_client):
        """Test updating existing patient graph with new information."""
        update_request = PatientGraphRequest(
            patient_id="patient_123",
            visit_data={
                "date": "2024-02-15T14:00:00",
                "type": "follow_up",
                "provider": "Dr. Sarah Johnson",
                "chief_complaint": "diabetes follow-up"
            },
            entities=[
                {
                    "text": "improved glucose control",
                    "label": "CLINICAL_NOTE",
                    "confidence": 0.88,
                    "entity_id": "note_001"
                }
            ]
        )
        
        mock_result = Mock()
        mock_result.single.return_value = {
            "patient_id": "patient_123",
            "nodes_created": 2,
            "relationships_created": 1,
            "properties_updated": 3
        }
        graph_client.session.run.return_value = mock_result
        
        response = await graph_client.update_patient_graph("patient_123", update_request)
        
        assert response.success is True
        assert response.patient_id == "patient_123"
        assert "Patient graph updated successfully" in response.message
    
    @pytest.mark.asyncio
    async def test_create_patient_graph_error_handling(self, graph_client, comprehensive_patient_request):
        """Test error handling during patient graph creation."""
        # Mock Neo4j error
        graph_client.session.run.side_effect = Exception("Neo4j connection failed")
        
        response = await graph_client.create_patient_graph(comprehensive_patient_request)
        
        assert response.success is False
        assert "Error creating patient graph" in response.message
        assert response.error_details is not None


class TestGraphQueries:
    """Test cases for graph querying and retrieval operations."""
    
    @pytest.fixture
    def graph_client(self):
        """Create a mock graph client for testing."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            client = Neo4jGraphClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password"
            )
            client.driver = mock_driver.return_value
            client.session = mock_session
            return client
    
    @pytest.mark.asyncio
    async def test_get_patient_graph_comprehensive(self, graph_client):
        """Test comprehensive patient graph retrieval."""
        # Mock comprehensive patient data
        mock_records = [
            Mock(data=lambda: {
                "patient": {
                    "id": "patient_123",
                    "name": "John Smith",
                    "dob": "1970-05-20",
                    "gender": "male",
                    "mrn": "MRN123456"
                },
                "visits": [
                    {
                        "id": "visit_1",
                        "date": "2024-01-15T10:30:00",
                        "type": "initial_consultation",
                        "provider": "Dr. Sarah Johnson"
                    }
                ],
                "conditions": [
                    {
                        "id": "condition_001",
                        "name": "Type 2 Diabetes Mellitus",
                        "icd_code": "E11.9",
                        "status": "active"
                    }
                ],
                "medications": [
                    {
                        "id": "medication_001",
                        "name": "Metformin",
                        "dosage": "500mg",
                        "frequency": "twice daily"
                    }
                ]
            })
        ]
        
        graph_client.session.run.return_value = mock_records
        
        patient_graph = await graph_client.get_patient_graph("patient_123")
        
        assert patient_graph is not None
        assert patient_graph["patient"]["id"] == "patient_123"
        assert len(patient_graph["conditions"]) == 1
        assert patient_graph["conditions"][0]["name"] == "Type 2 Diabetes Mellitus"
    
    @pytest.mark.asyncio
    async def test_execute_custom_cypher_query(self, graph_client):
        """Test execution of custom Cypher queries."""
        query_request = GraphQueryRequest(
            query="""
            MATCH (p:Patient)-[:HAS_VISIT]->(v:Visit)-[:DIAGNOSED_WITH]->(c:Condition)
            WHERE c.name = $condition_name
            RETURN p.name as patient_name, v.date as visit_date, c.severity as severity
            """,
            parameters={"condition_name": "Type 2 Diabetes Mellitus"},
            limit=100
        )
        
        mock_records = [
            Mock(data=lambda: {
                "patient_name": "John Smith",
                "visit_date": "2024-01-15",
                "severity": "moderate"
            }),
            Mock(data=lambda: {
                "patient_name": "Jane Doe",
                "visit_date": "2024-01-20",
                "severity": "mild"
            })
        ]
        
        graph_client.session.run.return_value = mock_records
        
        results = await graph_client.execute_query(query_request)
        
        assert len(results) == 2
        assert results[0]["patient_name"] == "John Smith"
        assert results[1]["patient_name"] == "Jane Doe"
        
        # Verify query execution
        graph_client.session.run.assert_called_with(
            query_request.query,
            query_request.parameters
        )
    
    @pytest.mark.asyncio
    async def test_get_graph_schema(self, graph_client):
        """Test retrieval of graph database schema."""
        mock_schema_data = [
            Mock(data=lambda: {
                "label": "Patient",
                "properties": ["id", "name", "dob", "gender", "mrn"],
                "constraints": ["UNIQUE (id)"]
            }),
            Mock(data=lambda: {
                "label": "Condition",
                "properties": ["id", "name", "icd_code", "status", "severity"],
                "constraints": ["UNIQUE (id)"]
            })
        ]
        
        graph_client.session.run.return_value = mock_schema_data
        
        schema = await graph_client.get_schema()
        
        assert len(schema) == 2
        assert schema[0]["label"] == "Patient"
        assert "id" in schema[0]["properties"]
    
    @pytest.mark.asyncio
    async def test_get_database_statistics(self, graph_client):
        """Test retrieval of database statistics and health information."""
        mock_stats = Mock(data=lambda: {
            "node_count": 1250,
            "relationship_count": 3750,
            "patient_count": 125,
            "visit_count": 380,
            "condition_count": 245,
            "medication_count": 180,
            "database_size": "15.2 MB",
            "last_updated": "2024-01-15T12:00:00"
        })
        
        graph_client.session.run.return_value = [mock_stats]
        
        stats = await graph_client.get_database_info()
        
        assert stats["node_count"] == 1250
        assert stats["relationship_count"] == 3750
        assert stats["patient_count"] == 125


class TestTemporalRelationships:
    """Test cases for temporal relationship modeling and analysis."""
    
    @pytest.fixture
    def graph_client(self):
        """Create a mock graph client for testing."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            client = Neo4jGraphClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password"
            )
            client.driver = mock_driver.return_value
            client.session = mock_session
            return client
    
    @pytest.mark.asyncio
    async def test_create_temporal_relationships(self, graph_client):
        """Test creation of temporal relationships between medical events."""
        temporal_data = {
            "patient_id": "patient_123",
            "relationships": [
                {
                    "source_event": "diagnosis_diabetes",
                    "target_event": "prescription_metformin",
                    "relationship_type": "FOLLOWED_BY",
                    "temporal_distance": "1 day",
                    "confidence": 0.95
                },
                {
                    "source_event": "lab_hba1c_initial",
                    "target_event": "diagnosis_diabetes",
                    "relationship_type": "PRECEDED_BY",
                    "temporal_distance": "2 days",
                    "confidence": 0.93
                }
            ]
        }
        
        mock_result = Mock()
        mock_result.single.return_value = {
            "relationships_created": 2,
            "temporal_edges_added": 2
        }
        graph_client.session.run.return_value = mock_result
        
        result = await graph_client.create_temporal_relationships(temporal_data)
        
        assert result["relationships_created"] == 2
        assert result["temporal_edges_added"] == 2
    
    @pytest.mark.asyncio
    async def test_analyze_patient_timeline(self, graph_client):
        """Test comprehensive patient timeline analysis."""
        mock_timeline_data = [
            Mock(data=lambda: {
                "event_date": "2024-01-10",
                "event_type": "lab_test",
                "event_description": "HbA1c test performed",
                "result_value": "8.2%",
                "significance": "elevated"
            }),
            Mock(data=lambda: {
                "event_date": "2024-01-15",
                "event_type": "diagnosis",
                "event_description": "Type 2 Diabetes Mellitus diagnosed",
                "provider": "Dr. Sarah Johnson",
                "significance": "new_diagnosis"
            }),
            Mock(data=lambda: {
                "event_date": "2024-01-15",
                "event_type": "prescription",
                "event_description": "Metformin 500mg twice daily prescribed",
                "provider": "Dr. Sarah Johnson",
                "significance": "treatment_initiation"
            })
        ]
        
        graph_client.session.run.return_value = mock_timeline_data
        
        timeline = await graph_client.analyze_patient_timeline("patient_123")
        
        assert isinstance(timeline, PatientTimelineResponse)
        assert timeline.patient_id == "patient_123"
        assert len(timeline.timeline_events) == 3
        assert timeline.timeline_events[0].event_type == "lab_test"
        assert timeline.timeline_events[1].event_type == "diagnosis"
    
    @pytest.mark.asyncio
    async def test_temporal_pattern_analysis(self, graph_client):
        """Test temporal pattern analysis across patient events."""
        pattern_query = {
            "pattern_type": "treatment_response",
            "condition": "Type 2 Diabetes Mellitus",
            "timeframe": "90 days",
            "metrics": ["hba1c", "glucose", "medication_adherence"]
        }
        
        mock_pattern_data = [
            Mock(data=lambda: {
                "patient_id": "patient_123",
                "baseline_hba1c": 8.2,
                "followup_hba1c": 7.1,
                "response_time_days": 75,
                "medication": "Metformin",
                "adherence_score": 0.9
            }),
            Mock(data=lambda: {
                "patient_id": "patient_456",
                "baseline_hba1c": 9.1,
                "followup_hba1c": 7.8,
                "response_time_days": 82,
                "medication": "Metformin",
                "adherence_score": 0.85
            })
        ]
        
        graph_client.session.run.return_value = mock_pattern_data
        
        patterns = await graph_client.analyze_temporal_patterns(pattern_query)
        
        assert len(patterns) == 2
        assert patterns[0]["response_time_days"] == 75
        assert patterns[1]["baseline_hba1c"] == 9.1


class TestCrossPatientAnalysis:
    """Test cases for cross-patient pattern analysis and insights."""
    
    @pytest.fixture
    def graph_client(self):
        """Create a mock graph client for testing."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            client = Neo4jGraphClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password"
            )
            client.driver = mock_driver.return_value
            client.session = mock_session
            return client
    
    @pytest.mark.asyncio
    async def test_discover_condition_patterns(self, graph_client):
        """Test discovery of patterns for specific medical conditions."""
        mock_pattern_data = [
            Mock(data=lambda: {
                "condition": "Type 2 Diabetes Mellitus",
                "patient_count": 125,
                "average_age_at_diagnosis": 58.3,
                "common_comorbidities": ["Hypertension", "Hyperlipidemia"],
                "most_effective_treatments": ["Metformin", "Lifestyle modification"],
                "average_hba1c_improvement": 1.2,
                "treatment_response_rate": 0.78
            })
        ]
        
        graph_client.session.run.return_value = mock_pattern_data
        
        patterns = await graph_client.discover_condition_patterns("Type 2 Diabetes Mellitus")
        
        assert patterns["condition"] == "Type 2 Diabetes Mellitus"
        assert patterns["patient_count"] == 125
        assert "Metformin" in patterns["most_effective_treatments"]
    
    @pytest.mark.asyncio
    async def test_compare_patient_cohorts(self, graph_client):
        """Test comparison between different patient cohorts."""
        cohort_comparison = {
            "cohort_a": {
                "criteria": "age >= 65 AND condition = 'Type 2 Diabetes Mellitus'",
                "label": "elderly_diabetics"
            },
            "cohort_b": {
                "criteria": "age < 65 AND condition = 'Type 2 Diabetes Mellitus'",
                "label": "younger_diabetics"
            },
            "comparison_metrics": ["treatment_response", "complication_rate", "medication_adherence"]
        }
        
        mock_comparison_data = [
            Mock(data=lambda: {
                "cohort": "elderly_diabetics",
                "patient_count": 78,
                "average_treatment_response": 0.72,
                "average_complication_rate": 0.15,
                "average_adherence": 0.88
            }),
            Mock(data=lambda: {
                "cohort": "younger_diabetics",
                "patient_count": 47,
                "average_treatment_response": 0.85,
                "average_complication_rate": 0.08,
                "average_adherence": 0.82
            })
        ]
        
        graph_client.session.run.return_value = mock_comparison_data
        
        comparison = await graph_client.compare_cohorts(cohort_comparison)
        
        assert len(comparison) == 2
        assert comparison[0]["cohort"] == "elderly_diabetics"
        assert comparison[1]["average_treatment_response"] == 0.85
    
    @pytest.mark.asyncio
    async def test_identify_similar_patients(self, graph_client):
        """Test identification of patients similar to a target patient."""
        similarity_criteria = {
            "target_patient_id": "patient_123",
            "similarity_factors": ["age", "gender", "primary_condition", "comorbidities"],
            "similarity_threshold": 0.8,
            "max_results": 10
        }
        
        mock_similar_patients = [
            Mock(data=lambda: {
                "patient_id": "patient_456",
                "similarity_score": 0.92,
                "matching_factors": ["age", "gender", "primary_condition"],
                "age": 54,
                "gender": "male",
                "primary_condition": "Type 2 Diabetes Mellitus"
            }),
            Mock(data=lambda: {
                "patient_id": "patient_789",
                "similarity_score": 0.87,
                "matching_factors": ["age", "primary_condition", "comorbidities"],
                "age": 56,
                "gender": "male",
                "primary_condition": "Type 2 Diabetes Mellitus"
            })
        ]
        
        graph_client.session.run.return_value = mock_similar_patients
        
        similar_patients = await graph_client.find_similar_patients(similarity_criteria)
        
        assert len(similar_patients) == 2
        assert similar_patients[0]["similarity_score"] == 0.92
        assert "Type 2 Diabetes Mellitus" in similar_patients[1]["primary_condition"]


class TestKnowledgeBaseIntegration:
    """Test cases for knowledge base integration (ICD-10, SNOMED-CT, RxNorm)."""
    
    @pytest.fixture
    def graph_client(self):
        """Create a mock graph client for testing."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            client = Neo4jGraphClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password"
            )
            client.driver = mock_driver.return_value
            client.session = mock_session
            return client
    
    @pytest.mark.asyncio
    async def test_expand_knowledge_base(self, graph_client):
        """Test expansion of knowledge base with external medical data."""
        knowledge_expansion = {
            "data_sources": ["ICD10", "SNOMED_CT", "RXNORM"],
            "entity_types": ["conditions", "medications", "procedures"],
            "update_mode": "incremental"
        }
        
        mock_expansion_result = Mock()
        mock_expansion_result.single.return_value = {
            "icd10_codes_added": 245,
            "snomed_concepts_added": 1250,
            "rxnorm_drugs_added": 380,
            "relationships_created": 875,
            "update_timestamp": "2024-01-15T12:00:00"
        }
        graph_client.session.run.return_value = mock_expansion_result
        
        result = await graph_client.expand_knowledge_base(knowledge_expansion)
        
        assert result["icd10_codes_added"] == 245
        assert result["snomed_concepts_added"] == 1250
        assert result["rxnorm_drugs_added"] == 380
    
    @pytest.mark.asyncio
    async def test_link_entities_to_knowledge_base(self, graph_client):
        """Test linking medical entities to external knowledge bases."""
        entity_linking = {
            "patient_id": "patient_123",
            "entities": [
                {
                    "entity_id": "condition_001",
                    "text": "Type 2 Diabetes Mellitus",
                    "entity_type": "condition",
                    "potential_codes": ["E11.9", "E11", "44054006"]
                },
                {
                    "entity_id": "medication_001",
                    "text": "Metformin",
                    "entity_type": "medication",
                    "potential_codes": ["6809", "metformin"]
                }
            ]
        }
        
        mock_linking_result = [
            Mock(data=lambda: {
                "entity_id": "condition_001",
                "icd10_code": "E11.9",
                "snomed_code": "44054006",
                "description": "Type 2 Diabetes Mellitus without complications"
            }),
            Mock(data=lambda: {
                "entity_id": "medication_001",
                "rxnorm_code": "6809",
                "generic_name": "Metformin",
                "drug_class": "Biguanides"
            })
        ]
        
        graph_client.session.run.return_value = mock_linking_result
        
        linked_entities = await graph_client.link_entities_to_knowledge_base(entity_linking)
        
        assert len(linked_entities) == 2
        assert linked_entities[0]["icd10_code"] == "E11.9"
        assert linked_entities[1]["rxnorm_code"] == "6809"
    
    @pytest.mark.asyncio
    async def test_validate_medical_codes(self, graph_client):
        """Test validation of medical codes against knowledge bases."""
        code_validation = {
            "codes_to_validate": [
                {"code": "E11.9", "system": "ICD10"},
                {"code": "6809", "system": "RXNORM"},
                {"code": "44054006", "system": "SNOMED_CT"}
            ]
        }
        
        mock_validation_result = [
            Mock(data=lambda: {
                "code": "E11.9",
                "system": "ICD10",
                "valid": True,
                "description": "Type 2 Diabetes Mellitus without complications"
            }),
            Mock(data=lambda: {
                "code": "6809",
                "system": "RXNORM",
                "valid": True,
                "description": "Metformin"
            }),
            Mock(data=lambda: {
                "code": "44054006",
                "system": "SNOMED_CT",
                "valid": True,
                "description": "Diabetes mellitus type 2"
            })
        ]
        
        graph_client.session.run.return_value = mock_validation_result
        
        validation_results = await graph_client.validate_medical_codes(code_validation)
        
        assert len(validation_results) == 3
        assert all(result["valid"] for result in validation_results)


class TestAdvancedGraphAnalytics:
    """Test cases for advanced graph analytics and insights generation."""
    
    @pytest.fixture
    def graph_client(self):
        """Create a mock graph client for testing."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            client = Neo4jGraphClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password"
            )
            client.driver = mock_driver.return_value
            client.session = mock_session
            return client
    
    @pytest.mark.asyncio
    async def test_generate_patient_insights(self, graph_client):
        """Test comprehensive patient insights generation."""
        mock_insights_data = Mock(data=lambda: {
            "patient_id": "patient_123",
            "risk_assessment": {
                "cardiovascular_risk": "moderate",
                "diabetic_complications_risk": "low",
                "medication_adherence_risk": "low"
            },
            "treatment_recommendations": [
                "Continue current Metformin therapy",
                "Monitor HbA1c every 3 months",
                "Consider lifestyle counseling"
            ],
            "predicted_outcomes": {
                "hba1c_target_achievement": 0.85,
                "complication_probability": 0.12,
                "treatment_response_likelihood": 0.88
            },
            "similar_patient_outcomes": {
                "patients_with_similar_profile": 45,
                "average_treatment_success": 0.82,
                "common_complications": ["Neuropathy", "Retinopathy"]
            }
        })
        
        graph_client.session.run.return_value = [mock_insights_data]
        
        insights = await graph_client.get_patient_insights("patient_123")
        
        assert isinstance(insights, GraphInsightResponse)
        assert insights.patient_id == "patient_123"
        assert insights.risk_assessment["cardiovascular_risk"] == "moderate"
        assert len(insights.treatment_recommendations) == 3
        assert insights.predicted_outcomes["hba1c_target_achievement"] == 0.85
    
    @pytest.mark.asyncio
    async def test_medication_interaction_analysis(self, graph_client):
        """Test medication interaction analysis across patient medications."""
        interaction_query = {
            "patient_id": "patient_123",
            "current_medications": ["Metformin", "Lisinopril", "Atorvastatin"],
            "proposed_medication": "Glyburide"
        }
        
        mock_interaction_data = [
            Mock(data=lambda: {
                "drug_a": "Metformin",
                "drug_b": "Glyburide",
                "interaction_severity": "moderate",
                "interaction_description": "May increase risk of hypoglycemia",
                "clinical_significance": "Monitor blood glucose closely"
            }),
            Mock(data=lambda: {
                "drug_a": "Lisinopril",
                "drug_b": "Glyburide",
                "interaction_severity": "minor",
                "interaction_description": "No significant interaction",
                "clinical_significance": "No specific monitoring required"
            })
        ]
        
        graph_client.session.run.return_value = mock_interaction_data
        
        interactions = await graph_client.analyze_medication_interactions(interaction_query)
        
        assert len(interactions) == 2
        assert interactions[0]["interaction_severity"] == "moderate"
        assert "hypoglycemia" in interactions[0]["interaction_description"]
    
    @pytest.mark.asyncio
    async def test_population_health_analytics(self, graph_client):
        """Test population health analytics and trend analysis."""
        population_query = {
            "population_criteria": "condition = 'Type 2 Diabetes Mellitus'",
            "analytics_timeframe": "2023-01-01 to 2024-01-01",
            "metrics": ["prevalence", "treatment_patterns", "outcomes"]
        }
        
        mock_population_data = Mock(data=lambda: {
            "total_population": 1250,
            "condition_prevalence": 0.12,
            "age_distribution": {
                "under_40": 15,
                "40_to_65": 78,
                "over_65": 32
            },
            "treatment_patterns": {
                "metformin_monotherapy": 0.65,
                "combination_therapy": 0.25,
                "insulin_therapy": 0.10
            },
            "outcome_metrics": {
                "average_hba1c_improvement": 1.3,
                "target_achievement_rate": 0.72,
                "complication_rate": 0.08
            }
        })
        
        graph_client.session.run.return_value = [mock_population_data]
        
        population_analytics = await graph_client.analyze_population_health(population_query)
        
        assert population_analytics["total_population"] == 1250
        assert population_analytics["condition_prevalence"] == 0.12
        assert population_analytics["outcome_metrics"]["target_achievement_rate"] == 0.72


class TestPerformanceAndOptimization:
    """Test cases for performance optimization and caching."""
    
    @pytest.fixture
    def graph_client(self):
        """Create a mock graph client for testing."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            client = Neo4jGraphClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password"
            )
            client.driver = mock_driver.return_value
            client.session = mock_session
            return client
    
    @pytest.mark.asyncio
    async def test_batch_graph_operations(self, graph_client):
        """Test batch processing of multiple graph operations."""
        batch_operations = [
            {
                "operation": "create_patient",
                "data": {"patient_id": "patient_001", "name": "John Doe"}
            },
            {
                "operation": "create_patient",
                "data": {"patient_id": "patient_002", "name": "Jane Smith"}
            },
            {
                "operation": "create_visit",
                "data": {"patient_id": "patient_001", "visit_date": "2024-01-15"}
            }
        ]
        
        mock_batch_result = Mock()
        mock_batch_result.single.return_value = {
            "operations_processed": 3,
            "nodes_created": 5,
            "relationships_created": 2,
            "processing_time_ms": 125
        }
        graph_client.session.run.return_value = mock_batch_result
        
        result = await graph_client.execute_batch_operations(batch_operations)
        
        assert result["operations_processed"] == 3
        assert result["nodes_created"] == 5
        assert result["processing_time_ms"] == 125
    
    @pytest.mark.asyncio
    async def test_query_performance_optimization(self, graph_client):
        """Test query performance optimization and indexing."""
        # Test index creation for performance optimization
        mock_index_result = Mock()
        mock_index_result.single.return_value = {
            "indexes_created": 3,
            "index_creation_time_ms": 89
        }
        graph_client.session.run.return_value = mock_index_result

        result = await graph_client.optimize_database_indexes()

        assert result["indexes_created"] == 3
        assert result["index_creation_time_ms"] == 89
    
    @pytest.mark.asyncio
    async def test_connection_pool_management(self, graph_client):
        """Test connection pool management and resource optimization."""
        # Test concurrent connections
        concurrent_tasks = []
        for i in range(10):
            task = graph_client.get_patient_graph(f"patient_{i}")
            concurrent_tasks.append(task)
        
        # Mock responses for concurrent queries
        mock_results = [{"patient_id": f"patient_{i}"} for i in range(10)]
        graph_client.session.run.return_value = mock_results
        
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        # Verify all queries completed successfully
        assert len(results) == 10
        assert all(not isinstance(result, Exception) for result in results)
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, graph_client):
        """Test memory usage optimization for large result sets."""
        # Test streaming large result sets
        large_query = {
            "query": "MATCH (p:Patient) RETURN p LIMIT 10000",
            "stream_results": True,
            "batch_size": 1000
        }
        
        mock_streaming_result = Mock()
        mock_streaming_result.data.return_value = [
            {"patient_id": f"patient_{i}"} for i in range(1000)
        ]
        graph_client.session.run.return_value = mock_streaming_result
        
        result_stream = await graph_client.execute_streaming_query(large_query)
        
        assert result_stream is not None
        # Verify streaming functionality
        batch_count = 0
        async for batch in result_stream:
            batch_count += 1
            assert len(batch) <= 1000
            if batch_count >= 10:  # Limit test execution
                break


class TestErrorHandlingAndRecovery:
    """Test cases for error handling and recovery scenarios."""
    
    @pytest.fixture
    def graph_client(self):
        """Create a mock graph client for testing."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            client = Neo4jGraphClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password"
            )
            client.driver = mock_driver.return_value
            client.session = mock_session
            return client
    
    @pytest.mark.asyncio
    async def test_connection_failure_recovery(self, graph_client):
        """Test recovery from Neo4j connection failures."""
        # Simulate connection failure followed by recovery
        graph_client.session.run.side_effect = [
            Exception("Connection lost"),  # First attempt fails
            Mock(data=lambda: {"patient_id": "patient_123"})  # Second attempt succeeds
        ]
        
        with patch.object(graph_client, 'reconnect', new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.return_value = True
            
            # Test automatic retry with recovery
            result = await graph_client.get_patient_graph_with_retry("patient_123", max_retries=2)
            
            assert result is not None
            assert result["patient_id"] == "patient_123"
            mock_reconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, graph_client):
        """Test transaction rollback on errors during graph operations."""
        # Mock transaction that fails partway through
        mock_transaction = AsyncMock()
        mock_transaction.run.side_effect = [
            Mock(),  # First operation succeeds
            Exception("Constraint violation")  # Second operation fails
        ]
        
        graph_client.session.begin_transaction.return_value = mock_transaction
        
        patient_request = PatientGraphRequest(
            patient_id="patient_123",
            visit_data={"date": "2024-01-15", "type": "consultation"},
            entities=[],
            relationships=[]
        )
        
        result = await graph_client.create_patient_graph_transactional(patient_request)
        
        assert result.success is False
        assert "transaction rolled back" in result.message.lower()
        mock_transaction.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalid_cypher_query_handling(self, graph_client):
        """Test handling of invalid Cypher queries."""
        invalid_query = GraphQueryRequest(
            query="INVALID CYPHER SYNTAX",
            parameters={}
        )
        
        graph_client.session.run.side_effect = Exception("Invalid syntax")
        
        result = await graph_client.execute_query(invalid_query)
        
        assert result is None or len(result) == 0
        # Verify error was logged (would need actual logging setup in real tests)
    
    @pytest.mark.asyncio
    async def test_data_consistency_validation(self, graph_client):
        """Test data consistency validation and error reporting."""
        inconsistent_data = PatientGraphRequest(
            patient_id="",  # Invalid empty patient ID
            visit_data={"date": "invalid-date"},  # Invalid date format
            entities=[
                {
                    "text": "",  # Empty entity text
                    "label": "INVALID_TYPE",  # Invalid entity type
                    "confidence": 1.5  # Invalid confidence score
                }
            ]
        )
        
        validation_errors = await graph_client.validate_patient_data(inconsistent_data)
        
        assert len(validation_errors) > 0
        assert any("patient_id" in error for error in validation_errors)
        assert any("date" in error for error in validation_errors)
        assert any("entity" in error for error in validation_errors)


class TestIntegrationWithMRIAAgents:
    """Test cases for integration with other MRIA agents."""
    
    @pytest.fixture
    def graph_client(self):
        """Create a mock graph client for testing."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            client = Neo4jGraphClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password"
            )
            client.driver = mock_driver.return_value
            client.session = mock_session
            return client
    
    @pytest.mark.asyncio
    async def test_integration_with_supervisor_agent(self, graph_client):
        """Test integration with Supervisor Agent workflow."""
        # Mock supervisor agent job data
        supervisor_job_data = {
            "job_id": "job_789",
            "patient_id": "patient_123",
            "workflow_type": "complete_pipeline",
            "agent_results": {
                "ocr": {"extracted_text": "Patient presents with diabetes symptoms..."},
                "ner": {"entities": [], "relationships": []},
                "chunking": {"chunks": [], "timeline": []}
            }
        }
        
        mock_integration_result = Mock()
        mock_integration_result.single.return_value = {
            "graph_created": True,
            "nodes_created": 12,
            "relationships_created": 8,
            "integration_status": "successful"
        }
        graph_client.session.run.return_value = mock_integration_result
        
        result = await graph_client.process_supervisor_job(supervisor_job_data)
        
        assert result["graph_created"] is True
        assert result["integration_status"] == "successful"
    
    @pytest.mark.asyncio
    async def test_integration_with_ner_agent(self, graph_client):
        """Test integration with NER Agent entity extraction."""
        ner_results = {
            "document_id": "doc_456",
            "patient_id": "patient_123",
            "entities": [
                {
                    "text": "Type 2 Diabetes Mellitus",
                    "label": "CONDITION",
                    "confidence": 0.95,
                    "concept_code": "E11.9"
                },
                {
                    "text": "Metformin 500mg",
                    "label": "MEDICATION",
                    "confidence": 0.92,
                    "concept_code": "6809"
                }
            ],
            "relationships": [
                {
                    "source": "Type 2 Diabetes Mellitus",
                    "target": "Metformin 500mg",
                    "relationship_type": "TREATED_WITH"
                }
            ]
        }
        
        mock_ner_integration = Mock()
        mock_ner_integration.single.return_value = {
            "entities_processed": 2,
            "relationships_created": 1,
            "knowledge_base_links": 2
        }
        graph_client.session.run.return_value = mock_ner_integration
        
        result = await graph_client.integrate_ner_results(ner_results)
        
        assert result["entities_processed"] == 2
        assert result["relationships_created"] == 1
    
    @pytest.mark.asyncio
    async def test_integration_with_chunking_agent(self, graph_client):
        """Test integration with Chunking Agent timeline data."""
        chunking_results = {
            "patient_id": "patient_123",
            "document_id": "doc_456",
            "chunks": [
                {
                    "chunk_id": "chunk_001",
                    "text": "Initial consultation findings...",
                    "visit_date": "2024-01-15",
                    "chunk_type": "visit_summary"
                },
                {
                    "chunk_id": "chunk_002",
                    "text": "Lab results and interpretation...",
                    "visit_date": "2024-01-15",
                    "chunk_type": "lab_results"
                }
            ],
            "timeline": [
                {
                    "event_date": "2024-01-15",
                    "event_type": "visit",
                    "event_description": "Initial diabetes consultation"
                }
            ]
        }
        
        mock_chunking_integration = Mock()
        mock_chunking_integration.single.return_value = {
            "chunks_processed": 2,
            "timeline_events_created": 1,
            "visit_structure_enhanced": True
        }
        graph_client.session.run.return_value = mock_chunking_integration
        
        result = await graph_client.integrate_chunking_results(chunking_results)
        
        assert result["chunks_processed"] == 2
        assert result["timeline_events_created"] == 1
        assert result["visit_structure_enhanced"] is True
    
    @pytest.mark.asyncio
    async def test_data_flow_validation(self, graph_client):
        """Test validation of data flow between agents."""
        agent_data_flow = {
            "ocr_output": {"text_quality": 0.94, "text_length": 2500},
            "ner_output": {"entity_count": 15, "confidence_avg": 0.89},
            "chunking_output": {"chunk_count": 8, "timeline_events": 5},
            "expected_graph_complexity": {"nodes": 20, "relationships": 12}
        }
        
        validation_result = await graph_client.validate_agent_data_flow(agent_data_flow)
        
        assert validation_result["data_flow_valid"] is True
        assert validation_result["consistency_score"] >= 0.8
        assert "recommendations" in validation_result


# Integration test combining multiple agent interactions
class TestEndToEndGraphWorkflow:
    """End-to-end test cases for complete graph workflow."""
    
    @pytest.fixture
    def graph_client(self):
        """Create a mock graph client for testing."""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            mock_session = AsyncMock()
            mock_driver.return_value.session.return_value = mock_session
            
            client = Neo4jGraphClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password"
            )
            client.driver = mock_driver.return_value
            client.session = mock_session
            return client
    
    @pytest.mark.asyncio
    async def test_complete_patient_processing_workflow(self, graph_client):
        """Test complete patient processing workflow from OCR to insights."""
        # Step 1: OCR results
        ocr_output = {
            "document_id": "doc_789",
            "extracted_text": "Patient John Smith presents with increased thirst, frequent urination...",
            "confidence": 0.94
        }
        
        # Step 2: NER results
        ner_output = {
            "entities": [
                {"text": "Type 2 Diabetes", "label": "CONDITION", "confidence": 0.95},
                {"text": "Metformin", "label": "MEDICATION", "confidence": 0.92}
            ]
        }
        
        # Step 3: Chunking results
        chunking_output = {
            "chunks": [{"chunk_id": "c1", "text": "...", "visit_date": "2024-01-15"}],
            "timeline": [{"event_date": "2024-01-15", "event_type": "diagnosis"}]
        }
        
        # Mock final graph creation
        mock_workflow_result = Mock()
        mock_workflow_result.single.return_value = {
            "workflow_completed": True,
            "patient_graph_created": True,
            "insights_generated": True,
            "total_processing_time_ms": 2500
        }
        graph_client.session.run.return_value = mock_workflow_result
        
        # Execute complete workflow
        workflow_data = {
            "patient_id": "patient_123",
            "ocr_results": ocr_output,
            "ner_results": ner_output,
            "chunking_results": chunking_output
        }
        
        result = await graph_client.execute_complete_workflow(workflow_data)
        
        assert result["workflow_completed"] is True
        assert result["patient_graph_created"] is True
        assert result["insights_generated"] is True
        assert result["total_processing_time_ms"] <= 5000  # Performance requirement


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
