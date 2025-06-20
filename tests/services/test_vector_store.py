"""
Comprehensive test suite for the Vector Store Service.

This module provides complete test coverage for the Vector Store functionality
including document embedding, semantic search, vector operations, and integration
with ChromaDB and FAISS for medical document processing.

Tests cover:
1. Vector store initialization and configuration
2. Document embedding and storage operations
3. Semantic search and similarity matching
4. Vector operations and indexing
5. Medical document vectorization
6. Query processing and retrieval
7. Performance optimization and caching
8. Integration with ChromaDB and FAISS
9. Error handling and recovery mechanisms
10. Health monitoring and metrics collection
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.services.vector_store import VectorStore


class TestVectorStore:
    """Comprehensive test cases for the Vector Store Service."""
    
    @pytest.fixture
    def vector_store(self):
        """Create a vector store instance for testing."""
        return VectorStore()
    
    @pytest.fixture
    def sample_medical_documents(self):
        """Sample medical documents for testing."""
        return [
            {
                "document_id": "doc_001",
                "content": "Patient presents with Type 2 Diabetes Mellitus. HbA1c is 8.2%. Started on Metformin 500mg twice daily.",
                "metadata": {
                    "patient_id": "patient_12345",
                    "document_type": "clinical_note",
                    "date": "2024-01-15",
                    "provider": "Dr. Smith"
                }
            },
            {
                "document_id": "doc_002", 
                "content": "Laboratory results show elevated glucose levels at 185 mg/dL. Recommend dietary counseling and medication adjustment.",
                "metadata": {
                    "patient_id": "patient_12345",
                    "document_type": "lab_report",
                    "date": "2024-01-20",
                    "provider": "Lab Services"
                }
            },
            {
                "document_id": "doc_003",
                "content": "Follow-up visit shows improved glycemic control. HbA1c decreased to 7.1%. Continue current therapy.",
                "metadata": {
                    "patient_id": "patient_12345",
                    "document_type": "follow_up_note",
                    "date": "2024-04-15",
                    "provider": "Dr. Smith"
                }
            }
        ]
    
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        return [
            np.random.rand(384).tolist(),  # ChromaDB typical dimension
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist()
        ]
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client for testing."""
        with patch('app.services.vector_store.chromadb') as mock:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock.Client.return_value = mock_client
            yield mock_client, mock_collection
    
    @pytest.fixture
    def mock_faiss_index(self):
        """Mock FAISS index for testing."""
        with patch('app.services.vector_store.faiss') as mock:
            mock_index = MagicMock()
            mock.IndexFlatIP.return_value = mock_index
            yield mock_index

    @pytest.mark.asyncio
    async def test_vector_store_initialization(self, vector_store, mock_chroma_client):
        """Test vector store initialization."""
        mock_client, mock_collection = mock_chroma_client
        
        await vector_store.initialize()
        
        assert vector_store.is_initialized is True
        mock_client.get_or_create_collection.assert_called()

    @pytest.mark.asyncio
    async def test_add_documents_success(self, vector_store, sample_medical_documents, 
                                       mock_chroma_client, sample_embeddings):
        """Test successful addition of documents to vector store."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock embedding generation
        with patch.object(vector_store, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = sample_embeddings
            
            result = await vector_store.add_documents(sample_medical_documents)
            
            assert result["success"] is True
            assert result["documents_added"] == 3
            mock_collection.add.assert_called_once()
            
            # Verify correct data passed to ChromaDB
            call_args = mock_collection.add.call_args
            assert len(call_args.kwargs["documents"]) == 3
            assert len(call_args.kwargs["embeddings"]) == 3
            assert len(call_args.kwargs["ids"]) == 3

    @pytest.mark.asyncio
    async def test_add_documents_empty_list(self, vector_store):
        """Test adding empty document list."""
        result = await vector_store.add_documents([])
        
        assert result["success"] is True
        assert result["documents_added"] == 0

    @pytest.mark.asyncio
    async def test_semantic_search_success(self, vector_store, mock_chroma_client):
        """Test successful semantic search."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock ChromaDB query response
        mock_collection.query.return_value = {
            "ids": [["doc_001", "doc_003"]],
            "distances": [[0.15, 0.28]],
            "documents": [["Patient with diabetes...", "Follow-up visit..."]],
            "metadatas": [[
                {"patient_id": "patient_12345", "document_type": "clinical_note"},
                {"patient_id": "patient_12345", "document_type": "follow_up_note"}
            ]]
        }
        
        # Mock embedding generation for query
        with patch.object(vector_store, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = [np.random.rand(384).tolist()]
            
            results = await vector_store.semantic_search(
                query="diabetes progression and treatment response",
                limit=2
            )
            
            assert len(results) == 2
            assert results[0]["document_id"] == "doc_001"
            assert results[0]["similarity_score"] > 0.7  # High similarity (low distance)
            assert "patient_id" in results[0]["metadata"]

    @pytest.mark.asyncio
    async def test_semantic_search_with_filters(self, vector_store, mock_chroma_client):
        """Test semantic search with metadata filters."""
        mock_client, mock_collection = mock_chroma_client
        
        mock_collection.query.return_value = {
            "ids": [["doc_001"]],
            "distances": [[0.12]],
            "documents": [["Filtered document content"]],
            "metadatas": [[{"patient_id": "patient_12345", "document_type": "clinical_note"}]]
        }
        
        with patch.object(vector_store, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = [np.random.rand(384).tolist()]
            
            results = await vector_store.semantic_search(
                query="diabetes treatment",
                filters={"patient_id": "patient_12345", "document_type": "clinical_note"},
                limit=5
            )
            
            assert len(results) == 1
            # Verify filters were applied
            mock_collection.query.assert_called_with(
                query_embeddings=[mock_embed.return_value[0]],
                n_results=5,
                where={"patient_id": "patient_12345", "document_type": "clinical_note"}
            )

    @pytest.mark.asyncio
    async def test_get_similar_documents(self, vector_store, mock_chroma_client):
        """Test finding similar documents by document ID."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock getting document by ID
        mock_collection.get.return_value = {
            "ids": ["doc_001"],
            "embeddings": [np.random.rand(384).tolist()],
            "documents": ["Reference document content"],
            "metadatas": [{"patient_id": "patient_12345"}]
        }
        
        # Mock similarity search
        mock_collection.query.return_value = {
            "ids": [["doc_002", "doc_003"]],
            "distances": [[0.18, 0.32]],
            "documents": [["Similar doc 1", "Similar doc 2"]],
            "metadatas": [[{"similarity": "high"}, {"similarity": "medium"}]]
        }
        
        similar_docs = await vector_store.get_similar_documents("doc_001", limit=2)
        
        assert len(similar_docs) == 2
        assert similar_docs[0]["similarity_score"] > similar_docs[1]["similarity_score"]

    @pytest.mark.asyncio
    async def test_update_document(self, vector_store, mock_chroma_client):
        """Test updating an existing document."""
        mock_client, mock_collection = mock_chroma_client
        
        updated_document = {
            "document_id": "doc_001",
            "content": "Updated: Patient shows excellent diabetes control with HbA1c at 6.5%",
            "metadata": {
                "patient_id": "patient_12345",
                "document_type": "clinical_note",
                "date": "2024-06-15",
                "updated": True
            }
        }
        
        with patch.object(vector_store, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = [np.random.rand(384).tolist()]
            
            result = await vector_store.update_document(updated_document)
            
            assert result["success"] is True
            assert result["document_id"] == "doc_001"
            mock_collection.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document(self, vector_store, mock_chroma_client):
        """Test deleting a document from vector store."""
        mock_client, mock_collection = mock_chroma_client
        
        result = await vector_store.delete_document("doc_001")
        
        assert result["success"] is True
        assert result["document_id"] == "doc_001"
        mock_collection.delete.assert_called_with(ids=["doc_001"])

    @pytest.mark.asyncio
    async def test_delete_documents_by_patient(self, vector_store, mock_chroma_client):
        """Test deleting all documents for a specific patient."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock getting patient documents
        mock_collection.get.return_value = {
            "ids": ["doc_001", "doc_002", "doc_003"]
        }
        
        result = await vector_store.delete_documents_by_patient("patient_12345")
        
        assert result["success"] is True
        assert result["documents_deleted"] == 3
        mock_collection.delete.assert_called()

    @pytest.mark.asyncio
    async def test_batch_operations(self, vector_store, sample_medical_documents, 
                                  mock_chroma_client, sample_embeddings):
        """Test batch operations for better performance."""
        mock_client, mock_collection = mock_chroma_client
        
        # Test batch addition
        with patch.object(vector_store, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = sample_embeddings
            
            result = await vector_store.batch_add_documents(
                sample_medical_documents, 
                batch_size=2
            )
            
            assert result["success"] is True
            assert result["total_documents"] == 3
            assert result["batches_processed"] == 2  # 2 docs + 1 doc

    @pytest.mark.asyncio
    async def test_embedding_generation(self, vector_store):
        """Test embedding generation for medical text."""
        medical_texts = [
            "Patient diagnosed with Type 2 Diabetes",
            "HbA1c levels showing improvement",
            "Metformin therapy well tolerated"
        ]
        
        with patch('app.services.vector_store.SentenceTransformer') as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = np.random.rand(3, 384)
            mock_model.return_value = mock_instance
            
            embeddings = await vector_store._generate_embeddings(medical_texts)
            
            assert len(embeddings) == 3
            assert all(len(emb) == 384 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_medical_text_preprocessing(self, vector_store):
        """Test preprocessing of medical texts before embedding."""
        raw_text = "Pt. w/ T2DM, HbA1c = 8.2%, started Metformin 500mg BID."
        
        processed_text = await vector_store._preprocess_medical_text(raw_text)
        
        # Should expand abbreviations and normalize
        assert "Patient" in processed_text or "patient" in processed_text
        assert "Type 2 Diabetes" in processed_text or "diabetes" in processed_text
        assert "twice daily" in processed_text or "BID" in processed_text

    @pytest.mark.asyncio
    async def test_collection_management(self, vector_store, mock_chroma_client):
        """Test vector store collection management."""
        mock_client, mock_collection = mock_chroma_client
        
        # Test creating collection
        collection_name = "test_medical_collection"
        result = await vector_store.create_collection(collection_name)
        
        assert result["success"] is True
        assert result["collection_name"] == collection_name
        
        # Test listing collections
        mock_client.list_collections.return_value = [
            MagicMock(name="medical_documents"),
            MagicMock(name="test_medical_collection")
        ]
        
        collections = await vector_store.list_collections()
        assert len(collections) == 2

    @pytest.mark.asyncio
    async def test_advanced_search_with_reranking(self, vector_store, mock_chroma_client):
        """Test advanced search with result reranking."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock initial search results
        mock_collection.query.return_value = {
            "ids": [["doc_001", "doc_002", "doc_003"]],
            "distances": [[0.2, 0.3, 0.4]],
            "documents": [["Doc 1 content", "Doc 2 content", "Doc 3 content"]],
            "metadatas": [[{}, {}, {}]]
        }
        
        with patch.object(vector_store, '_rerank_results') as mock_rerank:
            mock_rerank.return_value = [
                {"document_id": "doc_003", "rerank_score": 0.95},
                {"document_id": "doc_001", "rerank_score": 0.87},
                {"document_id": "doc_002", "rerank_score": 0.72}
            ]
            
            results = await vector_store.advanced_search(
                query="diabetes treatment outcomes",
                rerank=True,
                limit=3
            )
            
            # Results should be reordered by rerank score
            assert results[0]["document_id"] == "doc_003"
            assert results[0]["rerank_score"] == 0.95

    @pytest.mark.asyncio
    async def test_hybrid_search(self, vector_store, mock_chroma_client):
        """Test hybrid search combining semantic and keyword search."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock semantic search results
        mock_collection.query.return_value = {
            "ids": [["doc_001", "doc_002"]],
            "distances": [[0.15, 0.25]],
            "documents": [["Semantic result 1", "Semantic result 2"]],
            "metadatas": [[{"type": "semantic"}, {"type": "semantic"}]]
        }
        
        # Mock keyword search results
        with patch.object(vector_store, '_keyword_search') as mock_keyword:
            mock_keyword.return_value = [
                {"document_id": "doc_003", "keyword_score": 0.8, "content": "Keyword result 1"},
                {"document_id": "doc_001", "keyword_score": 0.6, "content": "Overlap result"}
            ]
            
            results = await vector_store.hybrid_search(
                query="diabetes metformin treatment",
                semantic_weight=0.7,
                keyword_weight=0.3,
                limit=3
            )
            
            assert len(results) <= 3
            # Should combine and deduplicate results
            assert any(r["document_id"] == "doc_001" for r in results)  # Overlap case

    @pytest.mark.asyncio
    async def test_vector_store_statistics(self, vector_store, mock_chroma_client):
        """Test vector store statistics and metrics."""
        mock_client, mock_collection = mock_chroma_client
        
        mock_collection.count.return_value = 1250
        mock_client.get_settings.return_value = {"anonymized_telemetry": False}
        
        stats = await vector_store.get_statistics()
        
        assert stats["total_documents"] == 1250
        assert "collections" in stats
        assert "storage_usage" in stats
        assert "average_embedding_dimension" in stats

    @pytest.mark.asyncio
    async def test_vector_store_health_check(self, vector_store, mock_chroma_client):
        """Test vector store health monitoring."""
        mock_client, mock_collection = mock_chroma_client
        mock_collection.count.return_value = 100  # Healthy document count
        
        health_status = await vector_store.get_health_status()
        
        assert health_status["service"] == "vector_store"
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]
        assert "database_connection" in health_status
        assert "embedding_service" in health_status
        assert "response_time" in health_status

    @pytest.mark.asyncio
    async def test_error_handling_connection_failure(self, vector_store):
        """Test error handling when database connection fails."""
        with patch('app.services.vector_store.chromadb') as mock_chroma:
            mock_chroma.Client.side_effect = Exception("Connection failed")
            
            result = await vector_store.initialize()
            
            assert result["success"] is False
            assert "connection" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_error_handling_embedding_failure(self, vector_store, sample_medical_documents, 
                                                   mock_chroma_client):
        """Test error handling when embedding generation fails."""
        mock_client, mock_collection = mock_chroma_client
        
        with patch.object(vector_store, '_generate_embeddings') as mock_embed:
            mock_embed.side_effect = Exception("Embedding service unavailable")
            
            result = await vector_store.add_documents(sample_medical_documents)
            
            assert result["success"] is False
            assert "embedding" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_performance_optimization_caching(self, vector_store):
        """Test performance optimization through caching."""
        # Test embedding caching
        text = "Patient with diabetes taking metformin"
        
        with patch.object(vector_store, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = [np.random.rand(384).tolist()]
            
            # First call should generate embedding
            embedding1 = await vector_store._get_cached_embedding(text)
            
            # Second call should use cache
            embedding2 = await vector_store._get_cached_embedding(text)
            
            assert np.array_equal(embedding1, embedding2)
            # Mock should be called only once
            assert mock_embed.call_count == 1

    @pytest.mark.asyncio
    async def test_vector_operations(self, vector_store):
        """Test vector mathematical operations."""
        vector1 = np.random.rand(384)
        vector2 = np.random.rand(384) 
        vector3 = np.random.rand(384)
        
        # Test cosine similarity
        similarity = await vector_store._calculate_cosine_similarity(vector1, vector2)
        assert -1 <= similarity <= 1
        
        # Test vector addition/averaging
        averaged = await vector_store._average_vectors([vector1, vector2, vector3])
        assert len(averaged) == 384
        
        # Test finding closest vectors
        query_vector = np.random.rand(384)
        candidate_vectors = [vector1, vector2, vector3]
        
        closest_idx = await vector_store._find_closest_vector(query_vector, candidate_vectors)
        assert 0 <= closest_idx < len(candidate_vectors)

    @pytest.mark.asyncio
    async def test_medical_entity_aware_search(self, vector_store, mock_chroma_client):
        """Test search that's aware of medical entities."""
        mock_client, mock_collection = mock_chroma_client
        
        # Mock NER extraction
        with patch('app.services.vector_store.ner_processor') as mock_ner:
            mock_ner.extract_entities.return_value = {
                "conditions": ["diabetes", "hypertension"],
                "medications": ["metformin"],
                "procedures": []
            }
            
            mock_collection.query.return_value = {
                "ids": [["doc_001"]],
                "distances": [[0.1]],
                "documents": [["Enhanced search result"]],
                "metadatas": [[{"entities_matched": ["diabetes", "metformin"]}]]
            }
            
            results = await vector_store.entity_aware_search(
                query="Patient with diabetes on metformin therapy",
                boost_entity_matches=True
            )
            
            assert len(results) == 1
            assert "entities_matched" in results[0]["metadata"]
