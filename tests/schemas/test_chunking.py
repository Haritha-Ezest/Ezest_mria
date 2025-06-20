"""
Comprehensive tests for the chunking schemas.

This module tests all Pydantic models in the chunking schemas, including:
- ChunkingStrategy enum validation
- ChunkingConfig model validation
- Request and response models
- Field validation and constraints
"""

import pytest
from pydantic import ValidationError

from app.schemas.chunking import (
    ChunkingStrategy, ChunkingConfig, MedicalEntity, MedicalChunk,
    ChunkRequest, ChunkResponse, TimelineRequest,
    StructureRequest
)


class TestChunkingStrategy:
    """Test the ChunkingStrategy enum."""

    def test_valid_strategies(self):
        """Test valid chunking strategies."""
        valid_strategies = [
            "visit_based",
            "topic_based", 
            "temporal",
            "semantic",
            "fixed_size",
            "sentence_boundary",
            "paragraph_boundary"
        ]
        
        for strategy in valid_strategies:
            assert ChunkingStrategy(strategy) in ChunkingStrategy

    def test_invalid_strategy(self):
        """Test invalid chunking strategy."""
        with pytest.raises(ValueError):
            ChunkingStrategy("invalid_strategy")


class TestChunkingConfig:
    """Test the ChunkingConfig model."""

    def test_valid_config(self):
        """Test valid chunking configuration."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=1500,
            min_chunk_size=200,
            overlap=100,
            preserve_medical_context=True,
            include_metadata=True,
            semantic_threshold=0.8
        )
        
        assert config.strategy == ChunkingStrategy.SEMANTIC
        assert config.chunk_size == 1500
        assert config.min_chunk_size == 200
        assert config.overlap == 100
        assert config.preserve_medical_context is True
        assert config.include_metadata is True
        assert config.semantic_threshold == 0.8

    def test_default_config(self):
        """Test default chunking configuration."""
        config = ChunkingConfig()
        
        assert config.strategy == ChunkingStrategy.SEMANTIC
        assert config.chunk_size == 1000
        assert config.min_chunk_size == 100
        assert config.overlap == 200
        assert config.preserve_medical_context is True
        assert config.include_metadata is True
        assert config.semantic_threshold == 0.7

    def test_invalid_chunk_size(self):
        """Test invalid chunk size validation."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(chunk_size=50)  # Below minimum
        assert "ensure this value is greater than or equal to 100" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(chunk_size=5000)  # Above maximum
        assert "ensure this value is less than or equal to 4000" in str(exc_info.value)

    def test_invalid_min_chunk_size(self):
        """Test invalid minimum chunk size validation."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(min_chunk_size=30)  # Below minimum
        assert "ensure this value is greater than or equal to 50" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(min_chunk_size=1500)  # Above maximum
        assert "ensure this value is less than or equal to 1000" in str(exc_info.value)

    def test_invalid_overlap(self):
        """Test invalid overlap validation."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(overlap=-10)  # Negative overlap
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

    def test_invalid_semantic_threshold(self):
        """Test invalid semantic threshold validation."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(semantic_threshold=-0.1)  # Below 0
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(semantic_threshold=1.5)  # Above 1
        assert "ensure this value is less than or equal to 1" in str(exc_info.value)


class TestMedicalEntity:
    """Test the MedicalEntity model."""

    def test_valid_entity(self):
        """Test valid medical entity."""
        entity = MedicalEntity(
            text="hypertension",
            label="CONDITION",
            start=25,
            end=37,
            confidence=0.95,
            normalized_form="Essential hypertension"
        )
        
        assert entity.text == "hypertension"
        assert entity.label == "CONDITION"
        assert entity.start == 25
        assert entity.end == 37
        assert entity.confidence == 0.95
        assert entity.normalized_form == "Essential hypertension"

    def test_entity_without_normalized_form(self):
        """Test medical entity without normalized form."""
        entity = MedicalEntity(
            text="John Doe",
            label="PERSON",
            start=0,
            end=8,
            confidence=0.98
        )
        
        assert entity.normalized_form is None

    def test_invalid_confidence(self):
        """Test invalid confidence score validation."""
        with pytest.raises(ValidationError) as exc_info:
            MedicalEntity(
                text="test",
                label="CONDITION", 
                start=0,
                end=4,
                confidence=-0.1  # Below 0
            )
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            MedicalEntity(
                text="test",
                label="CONDITION",
                start=0,
                end=4,
                confidence=1.5  # Above 1
            )
        assert "ensure this value is less than or equal to 1" in str(exc_info.value)


class TestMedicalChunk:
    """Test the MedicalChunk model."""

    def test_valid_chunk(self):
        """Test valid medical chunk."""
        chunk = MedicalChunk(
            chunk_id="chunk_001",
            text="Patient John Doe was diagnosed with hypertension.",
            start_position=0,
            end_position=49,
            chunk_type="visit_summary",
            entities=[
                MedicalEntity(
                    text="John Doe",
                    label="PERSON",
                    start=8,
                    end=16,
                    confidence=0.98
                ),
                MedicalEntity(
                    text="hypertension",
                    label="CONDITION",
                    start=37,
                    end=49,
                    confidence=0.95
                )
            ],
            metadata={
                "visit_date": "2023-01-15",
                "section": "diagnosis"
            },
            semantic_similarity=0.85,
            quality_score=0.92
        )
        
        assert chunk.chunk_id == "chunk_001"
        assert "John Doe" in chunk.text
        assert chunk.start_position == 0
        assert chunk.end_position == 49
        assert chunk.chunk_type == "visit_summary"
        assert len(chunk.entities) == 2
        assert chunk.semantic_similarity == 0.85
        assert chunk.quality_score == 0.92

    def test_chunk_without_optional_fields(self):
        """Test medical chunk without optional fields."""
        chunk = MedicalChunk(
            chunk_id="chunk_002",
            text="Simple text chunk."
        )
        
        assert chunk.entities == []
        assert chunk.metadata == {}
        assert chunk.semantic_similarity is None
        assert chunk.quality_score is None


class TestChunkRequest:
    """Test the ChunkRequest model."""

    def test_valid_request(self):
        """Test valid chunk request."""
        request = ChunkRequest(
            text="Patient visited on 2023-01-15 with chest pain. Examination revealed elevated BP.",
            strategy=ChunkingStrategy.SEMANTIC,
            max_chunk_size=500,
            overlap=50,
            preserve_structure=True,
            extract_entities=True,
            include_timeline=False
        )
        
        assert "chest pain" in request.text
        assert request.strategy == ChunkingStrategy.SEMANTIC
        assert request.max_chunk_size == 500
        assert request.overlap == 50
        assert request.preserve_structure is True
        assert request.extract_entities is True
        assert request.include_timeline is False

    def test_request_with_defaults(self):
        """Test chunk request with default values."""
        request = ChunkRequest(
            text="Simple medical text."
        )
        
        assert request.strategy == ChunkingStrategy.SEMANTIC
        assert request.max_chunk_size == 1000
        assert request.overlap == 200
        assert request.preserve_structure is True
        assert request.extract_entities is True
        assert request.include_timeline is False

    def test_invalid_text(self):
        """Test invalid text validation."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkRequest(text="")  # Empty text
        assert "ensure this value has at least 1 characters" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ChunkRequest(text="x" * 50001)  # Too long
        assert "ensure this value has at most 50000 characters" in str(exc_info.value)


class TestChunkResponse:
    """Test the ChunkResponse model."""

    def test_valid_response(self):
        """Test valid chunk response."""
        response = ChunkResponse(
            success=True,
            chunks=[
                MedicalChunk(
                    chunk_id="chunk_1",
                    text="First chunk text.",
                    start_position=0,
                    end_position=17
                ),
                MedicalChunk(
                    chunk_id="chunk_2", 
                    text="Second chunk text.",
                    start_position=18,
                    end_position=36
                )
            ],
            total_chunks=2,
            strategy_used=ChunkingStrategy.SEMANTIC,
            processing_time=0.5,
            timeline_created=False,
            metadata={
                "model_version": "1.0.0",
                "language": "en"
            }
        )
        
        assert response.success is True
        assert len(response.chunks) == 2
        assert response.total_chunks == 2
        assert response.strategy_used == ChunkingStrategy.SEMANTIC
        assert response.processing_time == 0.5
        assert response.timeline_created is False

    def test_error_response(self):
        """Test error chunk response."""
        response = ChunkResponse(
            success=False,
            chunks=[],
            total_chunks=0,
            error_message="Processing failed due to invalid input."
        )
        
        assert response.success is False
        assert len(response.chunks) == 0
        assert response.total_chunks == 0
        assert "Processing failed" in response.error_message


class TestTimelineRequest:
    """Test the TimelineRequest model."""

    def test_valid_timeline_request(self):
        """Test valid timeline request."""
        request = TimelineRequest(
            chunks=[
                MedicalChunk(
                    chunk_id="chunk_1",
                    text="Visit on 2023-01-15",
                    metadata={"visit_date": "2023-01-15"}
                ),
                MedicalChunk(
                    chunk_id="chunk_2",
                    text="Follow-up on 2023-02-01",
                    metadata={"visit_date": "2023-02-01"}
                )
            ],
            sort_by="date",
            include_metadata=True,
            group_by_visit=True
        )
        
        assert len(request.chunks) == 2
        assert request.sort_by == "date"
        assert request.include_metadata is True
        assert request.group_by_visit is True

    def test_empty_chunks_validation(self):
        """Test validation for empty chunks list."""
        with pytest.raises(ValidationError) as exc_info:
            TimelineRequest(chunks=[])
        assert "ensure this value has at least 1 items" in str(exc_info.value)


class TestStructureRequest:
    """Test the StructureRequest model."""

    def test_valid_structure_request(self):
        """Test valid structure analysis request."""
        request = StructureRequest(
            chunks=[
                MedicalChunk(
                    chunk_id="chunk_1",
                    text="Patient information",
                    metadata={"section": "demographics"}
                )
            ],
            analysis_type="medical_structure",
            include_relationships=True,
            confidence_threshold=0.8
        )
        
        assert len(request.chunks) == 1
        assert request.analysis_type == "medical_structure"
        assert request.include_relationships is True
        assert request.confidence_threshold == 0.8

    def test_invalid_analysis_type(self):
        """Test invalid analysis type validation."""
        with pytest.raises(ValidationError) as exc_info:
            StructureRequest(
                chunks=[MedicalChunk(chunk_id="test", text="test")],
                analysis_type="invalid_type"
            )
        assert "analysis_type must be one of" in str(exc_info.value)
