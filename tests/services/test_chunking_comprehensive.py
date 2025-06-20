"""
Comprehensive test suite for the Chunking & Timeline Structuring Agent.

This module provides complete test coverage for the Chunking Agent functionality
including document chunking, timeline creation, and medical information structuring.

Tests cover:
1. Chunking Agent initialization and strategy configuration
2. Document chunking with different strategies
3. Medical visit-based chunking and organization
4. Timeline creation and temporal structuring
5. Semantic chunking with medical embeddings
6. Chunk quality assessment and validation
7. Medical entity extraction from chunks
8. Timeline visualization and analysis
9. Integration with NER results and knowledge graphs
10. Performance optimization and batch processing
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.services.chunker import ChunkingProcessor
from app.schemas.chunking import (
    ChunkingRequest, ChunkingResponse, ChunkingStatus, ChunkingConfiguration,
    ChunkingStrategy, ChunkMetadata, TimelineEvent, PatientTimeline, MedicalChunk
)


class TestChunkingAgent:
    """Comprehensive test cases for the Chunking & Timeline Structuring Agent."""
    
    @pytest.fixture
    def chunking_processor(self):
        """Create a chunking processor instance for testing."""
        return ChunkingProcessor()
    
    @pytest.fixture
    def sample_medical_document(self):
        """Comprehensive sample medical document for chunking tests."""
        return """
        PATIENT: John Doe
        DOB: 03/15/1958
        MRN: 12345678
        
        VISIT 1 - 2024-01-15
        ==================
        CHIEF COMPLAINT: Increased thirst and frequent urination for 3 months
        
        HISTORY OF PRESENT ILLNESS:
        This 65-year-old male presents with a 3-month history of polyuria, polydipsia, 
        and blurred vision. Patient reports drinking 6-8 glasses of water daily and 
        urinating every 2 hours. Weight loss of 15 pounds over the past 2 months.
        
        PHYSICAL EXAMINATION:
        Vital Signs: BP 140/90 mmHg, HR 78 bpm, Temp 98.6°F, RR 16/min
        BMI: 32 kg/m²
        
        LABORATORY RESULTS:
        - Fasting glucose: 185 mg/dL (normal: 70-100 mg/dL)
        - HbA1c: 8.2% (normal: <7.0%)
        - Total cholesterol: 220 mg/dL
        
        ASSESSMENT AND PLAN:
        1. Type 2 Diabetes Mellitus - newly diagnosed
           - Start Metformin 500mg twice daily
           - Diabetes education and lifestyle counseling
           - Follow-up in 3 months
        
        VISIT 2 - 2024-04-20
        ==================
        CHIEF COMPLAINT: Follow-up for diabetes management
        
        HISTORY OF PRESENT ILLNESS:
        Patient returns for diabetes follow-up. Reports good compliance with Metformin.
        Polyuria and polydipsia have improved significantly. Weight stable.
        Home glucose monitoring shows readings 120-150 mg/dL.
        
        PHYSICAL EXAMINATION:
        Vital Signs: BP 135/88 mmHg, HR 76 bpm, Weight 185 lbs (down 10 lbs)
        
        LABORATORY RESULTS:
        - HbA1c: 7.1% (improved from 8.2%)
        - Fasting glucose: 135 mg/dL
        
        ASSESSMENT AND PLAN:
        1. Type 2 Diabetes Mellitus - improving control
           - Continue Metformin 500mg twice daily
           - Goal HbA1c <7%
           - Follow-up in 3 months
        
        VISIT 3 - 2024-07-15
        ==================
        CHIEF COMPLAINT: Routine diabetes follow-up
        
        LABORATORY RESULTS:
        - HbA1c: 6.8% (target achieved!)
        - Fasting glucose: 118 mg/dL
        
        ASSESSMENT AND PLAN:
        1. Type 2 Diabetes Mellitus - excellent control
           - Continue current regimen
           - Annual follow-up
        """
    
    @pytest.fixture
    def ner_results_sample(self):
        """Sample NER results for integration testing."""
        return {
            "entities": [
                {"text": "Type 2 Diabetes Mellitus", "label": "CONDITION", "start": 100, "end": 124, "confidence": 0.95},
                {"text": "Metformin", "label": "MEDICATION", "start": 200, "end": 209, "confidence": 0.92},
                {"text": "500mg", "label": "DOSAGE", "start": 210, "end": 215, "confidence": 0.88},
                {"text": "HbA1c", "label": "LAB_VALUE", "start": 300, "end": 305, "confidence": 0.94},
                {"text": "8.2%", "label": "VALUE", "start": 307, "end": 311, "confidence": 0.90}
            ],
            "temporal_entities": [
                {"text": "2024-01-15", "label": "DATE", "start": 50, "end": 60, "confidence": 0.98},
                {"text": "3 months", "label": "DURATION", "start": 150, "end": 158, "confidence": 0.85},
                {"text": "twice daily", "label": "FREQUENCY", "start": 220, "end": 231, "confidence": 0.87}
            ]
        }

    # Test 1: Chunking Agent Initialization and Configuration
    async def test_chunking_processor_initialization(self, chunking_processor):
        """Test chunking processor initialization with default configuration."""
        assert chunking_processor is not None
        assert hasattr(chunking_processor, 'chunking_strategies')
        assert hasattr(chunking_processor, 'embedding_model')
        
        # Test available strategies
        available_strategies = chunking_processor.get_available_strategies()
        expected_strategies = [
            ChunkingStrategy.VISIT_BASED, ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.TEMPORAL, ChunkingStrategy.TOPIC_BASED,
            ChunkingStrategy.FIXED_SIZE, ChunkingStrategy.SENTENCE_BASED,
            ChunkingStrategy.MEDICAL_SECTIONS
        ]
        
        for strategy in expected_strategies:
            assert strategy in available_strategies

    async def test_chunking_strategy_configuration(self, chunking_processor):
        """Test configuration of different chunking strategies."""
        # Test visit-based chunking configuration
        visit_config = ChunkingConfiguration(
            strategy=ChunkingStrategy.VISIT_BASED,
            chunk_size=1000,
            overlap=200,
            preserve_medical_sections=True,
            extract_visit_metadata=True
        )
        
        chunking_processor.configure_strategy(visit_config)
        assert chunking_processor.current_strategy == ChunkingStrategy.VISIT_BASED
        assert chunking_processor.chunk_size == 1000
        assert chunking_processor.overlap == 200
        
        # Test semantic chunking configuration
        semantic_config = ChunkingConfiguration(
            strategy=ChunkingStrategy.SEMANTIC,
            similarity_threshold=0.7,
            min_chunk_size=200,
            max_chunk_size=2000,
            use_medical_embeddings=True
        )
        
        chunking_processor.configure_strategy(semantic_config)
        assert chunking_processor.current_strategy == ChunkingStrategy.SEMANTIC
        assert chunking_processor.similarity_threshold == 0.7

    async def test_medical_section_identification(self, chunking_processor):
        """Test identification of medical document sections."""
        medical_sections = chunking_processor.identify_medical_sections(
            "CHIEF COMPLAINT: Chest pain\nHISTORY: Patient reports...\nPHYSICAL EXAM: Normal\nASSESSMENT: Stable"
        )
        
        expected_sections = ["CHIEF COMPLAINT", "HISTORY", "PHYSICAL EXAM", "ASSESSMENT"]
        for section in expected_sections:
            assert section in [s['type'] for s in medical_sections]

    # Test 2: Visit-Based Chunking
    async def test_visit_based_chunking(self, chunking_processor, sample_medical_document):
        """Test visit-based chunking of medical documents."""
        with patch.object(chunking_processor, '_extract_visit_chunks') as mock_visits:
            mock_visits.return_value = [
                MedicalChunk(
                    chunk_id="visit_1",
                    text="VISIT 1 - 2024-01-15\nCHIEF COMPLAINT: Increased thirst...",
                    metadata=ChunkMetadata(
                        visit_date="2024-01-15",
                        visit_type="initial_consultation",
                        sections=["chief_complaint", "history", "physical_exam", "assessment"],
                        entities_count=15,
                        confidence=0.92
                    ),
                    start_index=0,
                    end_index=1500
                ),
                MedicalChunk(
                    chunk_id="visit_2",
                    text="VISIT 2 - 2024-04-20\nCHIEF COMPLAINT: Follow-up...",
                    metadata=ChunkMetadata(
                        visit_date="2024-04-20",
                        visit_type="follow_up",
                        sections=["chief_complaint", "history", "physical_exam", "assessment"],
                        entities_count=12,
                        confidence=0.89
                    ),
                    start_index=1500,
                    end_index=2800
                )
            ]
            
            request = ChunkingRequest(
                document_id="visit_test",
                text=sample_medical_document,
                configuration=ChunkingConfiguration(
                    strategy=ChunkingStrategy.VISIT_BASED,
                    extract_visit_metadata=True
                )
            )
            
            response = await chunking_processor.process_document(request)
            
            # Verify visit-based chunking
            assert response.status == ChunkingStatus.COMPLETED
            assert len(response.chunks) >= 2
            
            # Check visit metadata
            visit_chunks = [chunk for chunk in response.chunks if chunk.metadata.visit_date]
            assert len(visit_chunks) >= 2
            assert any(chunk.metadata.visit_date == "2024-01-15" for chunk in visit_chunks)
            assert any(chunk.metadata.visit_date == "2024-04-20" for chunk in visit_chunks)

    async def test_visit_timeline_creation(self, chunking_processor, sample_medical_document):
        """Test creation of patient timeline from visit chunks."""
        with patch.object(chunking_processor, '_create_patient_timeline') as mock_timeline:
            mock_timeline.return_value = PatientTimeline(
                patient_id="patient_12345",
                timeline_events=[
                    TimelineEvent(
                        date="2024-01-15",
                        event_type="diagnosis",
                        description="Type 2 Diabetes Mellitus - newly diagnosed",
                        entities=["Type 2 Diabetes Mellitus", "Metformin"],
                        confidence=0.95
                    ),
                    TimelineEvent(
                        date="2024-04-20",
                        event_type="follow_up",
                        description="Diabetes follow-up - improving control",
                        entities=["HbA1c: 7.1%", "Metformin"],
                        confidence=0.92
                    ),
                    TimelineEvent(
                        date="2024-07-15",
                        event_type="follow_up",
                        description="Routine diabetes follow-up - excellent control",
                        entities=["HbA1c: 6.8%"],
                        confidence=0.88
                    )
                ],
                total_visits=3,
                date_range={"start": "2024-01-15", "end": "2024-07-15"}
            )
            
            request = ChunkingRequest(
                document_id="timeline_test",
                text=sample_medical_document,
                configuration=ChunkingConfiguration(
                    strategy=ChunkingStrategy.VISIT_BASED,
                    create_timeline=True
                )
            )
            
            response = await chunking_processor.process_document(request)
            
            # Verify timeline creation
            assert hasattr(response, 'timeline')
            assert response.timeline.total_visits == 3
            assert len(response.timeline.timeline_events) == 3
            assert response.timeline.timeline_events[0].event_type == "diagnosis"

    # Test 3: Semantic Chunking
    async def test_semantic_chunking_with_embeddings(self, chunking_processor):
        """Test semantic chunking using medical embeddings."""
        medical_text = """
        Patient presents with diabetes symptoms. Blood sugar elevated.
        History of hypertension and family diabetes. Patient on ACE inhibitor.
        Physical exam shows BMI 32 and elevated blood pressure.
        Lab results confirm diabetes with HbA1c 8.2 percent.
        Treatment plan includes Metformin and lifestyle changes.
        Follow-up scheduled for diabetes management review.
        """
        
        with patch.object(chunking_processor, '_compute_semantic_similarity') as mock_similarity:
            mock_similarity.return_value = [
                {"chunk_1": 0, "chunk_2": 1, "similarity": 0.85},  # Diabetes symptoms + elevated sugar
                {"chunk_1": 2, "chunk_2": 3, "similarity": 0.78},  # Physical + lab results
                {"chunk_1": 4, "chunk_2": 5, "similarity": 0.72}   # Treatment + follow-up
            ]
            
            with patch.object(chunking_processor, '_create_semantic_chunks') as mock_chunks:
                mock_chunks.return_value = [
                    MedicalChunk(
                        chunk_id="semantic_1",
                        text="Patient presents with diabetes symptoms. Blood sugar elevated.",
                        metadata=ChunkMetadata(
                            topic="diabetes_symptoms",
                            semantic_similarity=0.85,
                            medical_concepts=["diabetes", "blood sugar"]
                        )
                    ),
                    MedicalChunk(
                        chunk_id="semantic_2", 
                        text="Lab results confirm diabetes with HbA1c 8.2 percent.",
                        metadata=ChunkMetadata(
                            topic="diagnostic_results",
                            semantic_similarity=0.78,
                            medical_concepts=["lab results", "HbA1c", "diabetes"]
                        )
                    )
                ]
                
                request = ChunkingRequest(
                    document_id="semantic_test",
                    text=medical_text,
                    configuration=ChunkingConfiguration(
                        strategy=ChunkingStrategy.SEMANTIC,
                        similarity_threshold=0.7,
                        use_medical_embeddings=True
                    )
                )
                
                response = await chunking_processor.process_document(request)
                
                # Verify semantic chunking
                assert response.status == ChunkingStatus.COMPLETED
                assert len(response.chunks) >= 2
                
                # Check semantic metadata
                for chunk in response.chunks:
                    assert hasattr(chunk.metadata, 'semantic_similarity')
                    assert chunk.metadata.semantic_similarity >= 0.7

    async def test_medical_concept_clustering(self, chunking_processor):
        """Test clustering of medical concepts for semantic chunking."""
        with patch.object(chunking_processor, '_cluster_medical_concepts') as mock_cluster:
            mock_cluster.return_value = {
                "diabetes_cluster": ["diabetes", "HbA1c", "glucose", "metformin"],
                "cardiovascular_cluster": ["hypertension", "blood pressure", "lisinopril"],
                "symptom_cluster": ["thirst", "urination", "blurred vision"]
            }
            
            text = "Patient has diabetes with high glucose and elevated HbA1c. Also has hypertension."
            
            clusters = await chunking_processor._cluster_medical_concepts(text)
            
            assert "diabetes_cluster" in clusters
            assert "diabetes" in clusters["diabetes_cluster"]
            assert "hypertension" in clusters["cardiovascular_cluster"]

    # Test 4: Temporal Chunking
    async def test_temporal_chunking_with_dates(self, chunking_processor):
        """Test temporal chunking based on dates and time periods."""
        temporal_text = """
        January 2024: Patient diagnosed with Type 2 diabetes.
        February 2024: Started on Metformin therapy.
        March 2024: First follow-up visit, HbA1c improving.
        June 2024: Routine check-up, excellent control achieved.
        """
        
        with patch.object(chunking_processor, '_extract_temporal_chunks') as mock_temporal:
            mock_temporal.return_value = [
                MedicalChunk(
                    chunk_id="temporal_1",
                    text="January 2024: Patient diagnosed with Type 2 diabetes.",
                    metadata=ChunkMetadata(
                        temporal_period="2024-01",
                        event_type="diagnosis",
                        date_extracted="2024-01-15"
                    )
                ),
                MedicalChunk(
                    chunk_id="temporal_2",
                    text="February 2024: Started on Metformin therapy.",
                    metadata=ChunkMetadata(
                        temporal_period="2024-02",
                        event_type="treatment_start",
                        date_extracted="2024-02-01"
                    )
                )
            ]
            
            request = ChunkingRequest(
                document_id="temporal_test",
                text=temporal_text,
                configuration=ChunkingConfiguration(
                    strategy=ChunkingStrategy.TEMPORAL,
                    extract_dates=True,
                    sort_chronologically=True
                )
            )
            
            response = await chunking_processor.process_document(request)
            
            # Verify temporal chunking
            assert response.status == ChunkingStatus.COMPLETED
            assert len(response.chunks) >= 2
            
            # Check temporal ordering
            dates = [chunk.metadata.date_extracted for chunk in response.chunks]
            assert dates == sorted(dates)  # Should be chronologically ordered

    async def test_timeline_event_extraction(self, chunking_processor):
        """Test extraction of timeline events from temporal chunks."""
        with patch.object(chunking_processor, '_extract_timeline_events') as mock_events:
            mock_events.return_value = [
                TimelineEvent(
                    date="2024-01-15",
                    event_type="diagnosis",
                    description="Type 2 Diabetes Mellitus diagnosed",
                    entities=["Type 2 Diabetes Mellitus"],
                    severity="high",
                    confidence=0.95
                ),
                TimelineEvent(
                    date="2024-02-01",
                    event_type="medication_start",
                    description="Metformin therapy initiated", 
                    entities=["Metformin", "500mg", "twice daily"],
                    severity="medium",
                    confidence=0.92
                )
            ]
            
            events = await chunking_processor._extract_timeline_events("Medical text with events")
            
            assert len(events) == 2
            assert events[0].event_type == "diagnosis"
            assert events[1].event_type == "medication_start"
            assert all(event.confidence >= 0.9 for event in events)

    # Test 5: Integration with NER Results
    async def test_chunk_ner_integration(self, chunking_processor, ner_results_sample):
        """Test integration of chunking with NER results."""
        with patch.object(chunking_processor, '_enrich_chunks_with_entities') as mock_enrich:
            mock_enrich.return_value = [
                MedicalChunk(
                    chunk_id="enriched_1",
                    text="Patient diagnosed with Type 2 Diabetes Mellitus",
                    metadata=ChunkMetadata(
                        entities=[
                            {"text": "Type 2 Diabetes Mellitus", "label": "CONDITION", "confidence": 0.95}
                        ],
                        entity_density=0.2,  # entities per word
                        medical_relevance=0.9
                    ),
                    ner_entities=ner_results_sample["entities"]
                )
            ]
            
            request = ChunkingRequest(
                document_id="ner_integration_test",
                text="Patient diagnosed with Type 2 Diabetes Mellitus",
                ner_results=ner_results_sample,
                configuration=ChunkingConfiguration(
                    strategy=ChunkingStrategy.ENTITY_BASED,
                    integrate_ner_results=True
                )
            )
            
            response = await chunking_processor.process_document(request)
            
            # Verify NER integration
            assert response.status == ChunkingStatus.COMPLETED
            assert hasattr(response.chunks[0], 'ner_entities')
            assert len(response.chunks[0].ner_entities) >= 1
            assert response.chunks[0].metadata.entity_density > 0

    async def test_entity_based_chunking(self, chunking_processor, ner_results_sample):
        """Test entity-based chunking using NER results."""
        with patch.object(chunking_processor, '_create_entity_chunks') as mock_entity_chunks:
            mock_entity_chunks.return_value = [
                MedicalChunk(
                    chunk_id="entity_1",
                    text="Type 2 Diabetes Mellitus diagnosed. Started Metformin 500mg.",
                    metadata=ChunkMetadata(
                        primary_entities=["Type 2 Diabetes Mellitus", "Metformin"],
                        entity_types=["CONDITION", "MEDICATION"],
                        chunk_theme="diabetes_management"
                    )
                ),
                MedicalChunk(
                    chunk_id="entity_2",
                    text="HbA1c level 8.2% indicates poor control.",
                    metadata=ChunkMetadata(
                        primary_entities=["HbA1c", "8.2%"],
                        entity_types=["LAB_VALUE", "VALUE"],
                        chunk_theme="laboratory_results"
                    )
                )
            ]
            
            request = ChunkingRequest(
                document_id="entity_chunking_test",
                text="Medical text with entities",
                ner_results=ner_results_sample,
                configuration=ChunkingConfiguration(
                    strategy=ChunkingStrategy.ENTITY_BASED,
                    group_related_entities=True
                )
            )
            
            response = await chunking_processor.process_document(request)
            
            # Verify entity-based chunking
            assert len(response.chunks) == 2
            assert response.chunks[0].metadata.chunk_theme == "diabetes_management"
            assert response.chunks[1].metadata.chunk_theme == "laboratory_results"

    # Test 6: Chunk Quality Assessment
    async def test_chunk_quality_assessment(self, chunking_processor):
        """Test quality assessment of generated chunks."""
        with patch.object(chunking_processor, '_assess_chunk_quality') as mock_quality:
            mock_quality.return_value = {
                "coherence_score": 0.87,
                "completeness_score": 0.92,
                "medical_relevance": 0.89,
                "information_density": 0.75,
                "overall_quality": 0.86
            }
            
            test_chunk = MedicalChunk(
                chunk_id="quality_test",
                text="Patient with diabetes, prescribed Metformin 500mg twice daily",
                metadata=ChunkMetadata()
            )
            
            quality_metrics = await chunking_processor._assess_chunk_quality(test_chunk)
            
            assert quality_metrics["overall_quality"] >= 0.8
            assert quality_metrics["medical_relevance"] >= 0.8
            assert quality_metrics["coherence_score"] >= 0.8

    async def test_chunk_validation_and_filtering(self, chunking_processor):
        """Test validation and filtering of low-quality chunks."""
        chunks = [
            MedicalChunk(chunk_id="good_chunk", text="High quality medical content with clear entities",
                        metadata=ChunkMetadata(quality_score=0.92)),
            MedicalChunk(chunk_id="poor_chunk", text="Low quality fragmented text",
                        metadata=ChunkMetadata(quality_score=0.45)),
            MedicalChunk(chunk_id="average_chunk", text="Average quality medical text",
                        metadata=ChunkMetadata(quality_score=0.75))
        ]
        
        with patch.object(chunking_processor, '_filter_chunks_by_quality') as mock_filter:
            mock_filter.return_value = [
                chunks[0],  # Keep high quality
                chunks[2]   # Keep average quality
                # Filter out poor quality
            ]
            
            filtered_chunks = await chunking_processor._filter_chunks_by_quality(
                chunks, quality_threshold=0.7
            )
            
            assert len(filtered_chunks) == 2
            assert all(chunk.metadata.quality_score >= 0.7 for chunk in filtered_chunks)

    # Test 7: Medical Section Processing
    async def test_medical_section_chunking(self, chunking_processor):
        """Test chunking based on medical document sections."""
        section_text = """
        CHIEF COMPLAINT:
        Chest pain and shortness of breath.
        
        HISTORY OF PRESENT ILLNESS:
        Patient reports acute onset chest pain...
        
        PHYSICAL EXAMINATION:
        Vital signs stable. Heart rate regular...
        
        ASSESSMENT AND PLAN:
        1. Acute coronary syndrome - rule out MI
        2. Order cardiac enzymes and ECG
        """
        
        with patch.object(chunking_processor, '_chunk_by_medical_sections') as mock_sections:
            mock_sections.return_value = [
                MedicalChunk(
                    chunk_id="section_chief_complaint",
                    text="CHIEF COMPLAINT:\nChest pain and shortness of breath.",
                    metadata=ChunkMetadata(
                        section_type="chief_complaint",
                        section_priority="high",
                        medical_concepts=["chest pain", "shortness of breath"]
                    )
                ),
                MedicalChunk(
                    chunk_id="section_assessment",
                    text="ASSESSMENT AND PLAN:\n1. Acute coronary syndrome - rule out MI",
                    metadata=ChunkMetadata(
                        section_type="assessment_plan",
                        section_priority="critical",
                        medical_concepts=["acute coronary syndrome", "MI"]
                    )
                )
            ]
            
            request = ChunkingRequest(
                document_id="section_test",
                text=section_text,
                configuration=ChunkingConfiguration(
                    strategy=ChunkingStrategy.MEDICAL_SECTIONS,
                    preserve_section_headers=True,
                    extract_section_metadata=True
                )
            )
            
            response = await chunking_processor.process_document(request)
            
            # Verify section-based chunking
            assert len(response.chunks) >= 2
            section_types = [chunk.metadata.section_type for chunk in response.chunks]
            assert "chief_complaint" in section_types
            assert "assessment_plan" in section_types

    # Test 8: Performance and Optimization
    async def test_large_document_chunking_performance(self, chunking_processor):
        """Test performance optimization for large medical documents."""
        large_document = "Large medical document. " * 5000  # Simulate large document
        
        with patch.object(chunking_processor, '_optimize_large_document_processing') as mock_optimize:
            mock_optimize.return_value = {
                "processing_time": 12.5,
                "chunks_created": 45,
                "optimization_applied": True,
                "memory_efficient": True
            }
            
            request = ChunkingRequest(
                document_id="large_doc_test",
                text=large_document,
                configuration=ChunkingConfiguration(
                    strategy=ChunkingStrategy.OPTIMIZED,
                    optimize_for_large_docs=True,
                    streaming_mode=True
                )
            )
            
            response = await chunking_processor.process_document(request)
            
            # Verify performance optimization
            assert response.processing_time < 15.0  # Should be optimized
            assert hasattr(response, 'optimization_applied')
            assert response.optimization_applied is True

    async def test_batch_document_chunking(self, chunking_processor):
        """Test batch processing of multiple documents."""
        batch_documents = [
            "Document 1: Patient with diabetes",
            "Document 2: Patient with hypertension", 
            "Document 3: Patient with both conditions"
        ]
        
        batch_requests = [
            ChunkingRequest(
                document_id=f"batch_doc_{i}",
                text=doc,
                configuration=ChunkingConfiguration(strategy=ChunkingStrategy.FAST)
            ) for i, doc in enumerate(batch_documents)
        ]
        
        with patch.object(chunking_processor, '_process_batch_efficiently') as mock_batch:
            mock_batch.return_value = [
                ChunkingResponse(
                    document_id=f"batch_doc_{i}",
                    status=ChunkingStatus.COMPLETED,
                    chunks=[
                        MedicalChunk(
                            chunk_id=f"batch_chunk_{i}",
                            text=doc,
                            metadata=ChunkMetadata()
                        )
                    ],
                    processing_time=2.1
                ) for i, doc in enumerate(batch_documents)
            ]
            
            responses = await chunking_processor.process_batch(batch_requests)
            
            # Verify batch processing
            assert len(responses) == 3
            for response in responses:
                assert response.status == ChunkingStatus.COMPLETED
                assert response.processing_time < 5.0

    # Test 9: Error Handling and Recovery
    async def test_chunking_error_handling(self, chunking_processor):
        """Test error handling in chunking process."""
        with patch.object(chunking_processor, '_create_chunks') as mock_create:
            # Simulate chunking failure
            mock_create.side_effect = Exception("Chunking algorithm failed")
            
            request = ChunkingRequest(
                document_id="error_test",
                text="Problem document",
                configuration=ChunkingConfiguration(
                    strategy=ChunkingStrategy.VISIT_BASED,
                    enable_fallback=True
                )
            )
            
            with patch.object(chunking_processor, '_fallback_chunking') as mock_fallback:
                mock_fallback.return_value = [
                    MedicalChunk(
                        chunk_id="fallback_chunk",
                        text="Problem document",
                        metadata=ChunkMetadata(chunking_method="fallback")
                    )
                ]
                
                response = await chunking_processor.process_document(request)
                
                # Verify fallback handling
                assert response.status == ChunkingStatus.COMPLETED
                assert response.chunks[0].metadata.chunking_method == "fallback"

    async def test_invalid_document_handling(self, chunking_processor):
        """Test handling of invalid or corrupted documents."""
        invalid_request = ChunkingRequest(
            document_id="invalid_test",
            text="",  # Empty text
            configuration=ChunkingConfiguration(strategy=ChunkingStrategy.VISIT_BASED)
        )
        
        response = await chunking_processor.process_document(invalid_request)
        
        # Verify error handling
        assert response.status == ChunkingStatus.FAILED
        assert "empty" in response.error_message.lower() or "invalid" in response.error_message.lower()

    # Test 10: Integration with Supervisor
    async def test_supervisor_integration(self, chunking_processor):
        """Test integration with supervisor workflow."""
        supervisor_callback = AsyncMock()
        chunking_processor.set_supervisor_callback(supervisor_callback)
        
        with patch.object(chunking_processor, '_create_chunks') as mock_create:
            mock_create.return_value = [
                MedicalChunk(
                    chunk_id="supervisor_chunk",
                    text="Medical content for supervisor",
                    metadata=ChunkMetadata()
                )
            ]
            
            request = ChunkingRequest(
                document_id="supervisor_integration_test",
                text="Medical document for processing",
                job_id="supervisor_job_789",
                configuration=ChunkingConfiguration(
                    strategy=ChunkingStrategy.VISIT_BASED,
                    notify_supervisor=True
                )
            )
            
            response = await chunking_processor.process_document(request)
            
            # Verify supervisor integration
            supervisor_callback.assert_called()
            assert response.job_id == "supervisor_job_789"
            assert response.status == ChunkingStatus.COMPLETED

    async def test_downstream_integration_preparation(self, chunking_processor):
        """Test preparation of chunks for downstream agents (Graph Builder)."""
        with patch.object(chunking_processor, '_prepare_for_graph_builder') as mock_prepare:
            mock_prepare.return_value = {
                "graph_ready_chunks": [
                    {
                        "chunk_id": "graph_chunk_1",
                        "entities": ["Type 2 Diabetes", "Metformin"],
                        "relationships": [{"entity1": "Patient", "relation": "has_condition", "entity2": "Type 2 Diabetes"}],
                        "temporal_data": {"visit_date": "2024-01-15"}
                    }
                ],
                "patient_metadata": {
                    "patient_id": "patient_123",
                    "visit_count": 3,
                    "date_range": {"start": "2024-01-15", "end": "2024-07-15"}
                }
            }
            
            request = ChunkingRequest(
                document_id="graph_prep_test",
                text="Medical document with timeline",
                configuration=ChunkingConfiguration(
                    strategy=ChunkingStrategy.VISIT_BASED,
                    prepare_for_graph=True
                )
            )
            
            response = await chunking_processor.process_document(request)
            
            # Verify graph preparation
            assert hasattr(response, 'graph_ready_data')
            assert "graph_ready_chunks" in response.graph_ready_data
            assert "patient_metadata" in response.graph_ready_data
