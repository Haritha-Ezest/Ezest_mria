"""
Vector store service for embedding storage and retrieval of medical timeline data.

This module provides functionality to store and retrieve patient timeline data
using vector embeddings for semantic search and similarity matching.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from app.schemas.chunking import PatientTimeline, TimelineEntry, MedicalChunk

# Configure logging
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store service for medical timeline data storage and retrieval.
    
    Uses ChromaDB for vector storage with sentence transformers for embeddings.
    Stores patient timelines with semantic search capabilities.
    """
    
    def __init__(self, persist_directory: str = "./storage/chroma_db"):
        """Initialize vector store with ChromaDB client."""
        self.persist_directory = persist_directory
        self.client = None
        self.embeddings_model = None
        self.timeline_collection = None
        self.chunks_collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collections."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialize embedding model
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create or get collections
            self.timeline_collection = self.client.get_or_create_collection(
                name="patient_timelines",
                metadata={
                    "description": "Patient medical timelines with semantic search",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            self.chunks_collection = self.client.get_or_create_collection(
                name="medical_chunks",
                metadata={
                    "description": "Medical document chunks with embeddings",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def store_patient_timeline(self, timeline: PatientTimeline) -> bool:
        """
        Store patient timeline in vector database.
        
        Args:
            timeline: PatientTimeline object to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create embeddings for timeline summary and entries
            timeline_text = self._prepare_timeline_text(timeline)
            embeddings = self.embeddings_model.encode([timeline_text])
            
            # Prepare metadata
            metadata = {
                "patient_id": timeline.patient_id,
                "total_visits": timeline.total_visits,
                "date_range_start": timeline.date_range["start"].isoformat(),
                "date_range_end": timeline.date_range["end"].isoformat(),
                "created_at": datetime.now().isoformat(),
                "summary": timeline.summary or "",
                "timeline_type": timeline.metadata.get("timeline_type", "medical_visits")
            }
            
            # Store in vector database
            self.timeline_collection.upsert(
                ids=[f"timeline_{timeline.patient_id}"],
                embeddings=embeddings.tolist(),
                documents=[json.dumps(timeline.dict(), default=str)],
                metadatas=[metadata]
            )
            
            # Store individual timeline entries for granular search
            for i, entry in enumerate(timeline.timeline_entries):
                entry_text = self._prepare_entry_text(entry)
                entry_embedding = self.embeddings_model.encode([entry_text])
                
                entry_metadata = {
                    "patient_id": timeline.patient_id,
                    "entry_index": i,
                    "visit_date": entry.date.isoformat(),
                    "visit_id": entry.visit_id or f"visit_{i}",
                    "confidence": entry.confidence,
                    "symptoms_count": len(entry.symptoms),
                    "tests_count": len(entry.tests),
                    "medications_count": len(entry.medications),
                    "diagnoses_count": len(entry.diagnoses)
                }
                
                self.timeline_collection.upsert(
                    ids=[f"entry_{timeline.patient_id}_{i}"],
                    embeddings=entry_embedding.tolist(),
                    documents=[json.dumps(entry.dict(), default=str)],
                    metadatas=[entry_metadata]
                )
            
            logger.info(f"Successfully stored timeline for patient {timeline.patient_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store timeline for patient {timeline.patient_id}: {e}")
            return False
    
    async def get_patient_timeline(self, patient_id: str) -> Optional[PatientTimeline]:
        """
        Retrieve patient timeline from vector database.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            PatientTimeline object or None if not found
        """
        try:
            # Query for the main timeline
            results = self.timeline_collection.get(
                ids=[f"timeline_{patient_id}"],
                include=["documents", "metadatas"]
            )
            
            if not results["ids"]:
                logger.info(f"No timeline found for patient {patient_id}")
                return None
            
            # Parse the timeline data
            timeline_data = json.loads(results["documents"][0])
            
            # Convert back to PatientTimeline object
            timeline = PatientTimeline(**timeline_data)
            
            logger.info(f"Retrieved timeline for patient {patient_id}")
            return timeline
            
        except Exception as e:
            logger.error(f"Failed to retrieve timeline for patient {patient_id}: {e}")
            return None
    
    async def store_medical_chunks(self, chunks: List[MedicalChunk], patient_id: str) -> bool:
        """
        Store medical chunks in vector database.
        
        Args:
            chunks: List of MedicalChunk objects
            patient_id: Patient identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not chunks:
                return True
            
            # Prepare data for batch insertion
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                # Create embedding for chunk text
                chunk_embedding = self.embeddings_model.encode([chunk.text])
                
                # Prepare metadata
                metadata = {
                    "patient_id": patient_id,
                    "chunk_id": chunk.chunk_id,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "word_count": chunk.word_count,
                    "char_count": chunk.char_count,
                    "chunk_type": chunk.chunk_type,
                    "visit_date": chunk.visit_date.isoformat() if chunk.visit_date else None,
                    "medical_topics": json.dumps(chunk.medical_topics),
                    "entities_count": len(chunk.medical_entities),
                    "semantic_score": chunk.semantic_score or 0.0
                }
                
                ids.append(f"chunk_{patient_id}_{chunk.chunk_id}")
                embeddings.append(chunk_embedding[0].tolist())
                documents.append(chunk.text)
                metadatas.append(metadata)
            
            # Store in vector database
            self.chunks_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully stored {len(chunks)} chunks for patient {patient_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunks for patient {patient_id}: {e}")
            return False
    
    async def search_similar_timelines(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar patient timelines using semantic search.
        
        Args:
            query_text: Search query
            limit: Maximum number of results
            
        Returns:
            List of similar timeline entries with similarity scores
        """
        try:
            # Create embedding for query
            query_embedding = self.embeddings_model.encode([query_text])
            
            # Search in timeline collection
            results = self.timeline_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            similar_timelines = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            )):
                similar_timelines.append({
                    "patient_id": metadata["patient_id"],
                    "similarity_score": 1 - distance,  # Convert distance to similarity
                    "metadata": metadata,
                    "timeline_data": json.loads(doc)
                })
            
            logger.info(f"Found {len(similar_timelines)} similar timelines for query")
            return similar_timelines
            
        except Exception as e:
            logger.error(f"Failed to search similar timelines: {e}")
            return []
    
    def _prepare_timeline_text(self, timeline: PatientTimeline) -> str:
        """Prepare timeline text for embedding."""
        text_parts = []
        
        if timeline.summary:
            text_parts.append(timeline.summary)
        
        for entry in timeline.timeline_entries:
            entry_text = self._prepare_entry_text(entry)
            text_parts.append(entry_text)
        
        return " ".join(text_parts)
    
    def _prepare_entry_text(self, entry: TimelineEntry) -> str:
        """Prepare timeline entry text for embedding."""
        text_parts = [
            f"Date: {entry.date.strftime('%Y-%m-%d')}",
            f"Symptoms: {', '.join(entry.symptoms)}",
            f"Tests: {', '.join(entry.tests)}",
            f"Medications: {', '.join(entry.medications)}",
            f"Diagnoses: {', '.join(entry.diagnoses)}",
            f"Procedures: {', '.join(entry.procedures)}",
            f"Notes: {', '.join(entry.progress_notes)}"
        ]
        
        return " ".join(text_parts)
    
    async def list_patient_timelines(self, limit: int = 10, offset: int = 0) -> List[PatientTimeline]:
        """
        List all patient timelines with pagination.
        
        Args:
            limit: Maximum number of timelines to return
            offset: Number of records to skip
            
        Returns:
            List[PatientTimeline]: List of patient timelines
        """
        try:
            if not self.timeline_collection:
                logger.warning("Timeline collection not available")
                return []
            
            # Get all timeline records with pagination
            results = self.timeline_collection.get(
                limit=limit,
                offset=offset,
                include=['metadatas', 'documents']
            )
            
            timelines = []
            if results and results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    try:
                        # Reconstruct timeline from metadata
                        timeline_data = json.loads(metadata.get('timeline_data', '{}'))
                        if timeline_data:
                            timeline = PatientTimeline(**timeline_data)
                            timelines.append(timeline)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse timeline data: {e}")
                        continue
            
            logger.info(f"Retrieved {len(timelines)} timelines with limit={limit}, offset={offset}")
            return timelines
            
        except Exception as e:
            logger.error(f"Failed to list patient timelines: {e}")
            return []
    
    async def count_patient_timelines(self) -> int:
        """
        Count total number of patient timelines in the database.
        
        Returns:
            int: Total count of patient timelines
        """
        try:
            if not self.timeline_collection:
                logger.warning("Timeline collection not available")
                return 0
            
            # Get count of all records
            results = self.timeline_collection.count()
            logger.info(f"Total patient timelines count: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to count patient timelines: {e}")
            return 0
    
    async def delete_patient_timeline(self, patient_id: str) -> bool:
        """
        Delete a patient's timeline from the vector database.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            if not self.timeline_collection:
                logger.warning("Timeline collection not available")
                return False
            
            # Delete timeline record
            self.timeline_collection.delete(
                where={"patient_id": patient_id}
            )
            
            logger.info(f"Successfully deleted timeline for patient {patient_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete timeline for patient {patient_id}: {e}")
            return False
    
    async def get_patient_chunks(self, patient_id: str, limit: int = 20, offset: int = 0) -> List[MedicalChunk]:
        """
        Retrieve medical chunks for a specific patient with pagination.
        
        Args:
            patient_id: Patient identifier
            limit: Maximum number of chunks to return
            offset: Number of chunks to skip
            
        Returns:
            List[MedicalChunk]: List of medical chunks for the patient
        """
        try:
            if not self.chunks_collection:
                logger.warning("Chunks collection not available")
                return []
            
            # Query chunks for the patient
            results = self.chunks_collection.get(
                where={"patient_id": patient_id},
                limit=limit,
                offset=offset,
                include=['metadatas', 'documents']
            )
            
            chunks = []
            if results and results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    try:
                        # Reconstruct chunk from metadata
                        chunk_data = json.loads(metadata.get('chunk_data', '{}'))
                        if chunk_data:
                            chunk = MedicalChunk(**chunk_data)
                            chunks.append(chunk)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse chunk data: {e}")
                        continue
            
            logger.info(f"Retrieved {len(chunks)} chunks for patient {patient_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks for patient {patient_id}: {e}")
            return []
    
    async def count_patient_chunks(self, patient_id: str) -> int:
        """
        Count total number of chunks for a specific patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            int: Total count of chunks for the patient
        """
        try:
            if not self.chunks_collection:
                logger.warning("Chunks collection not available")
                return 0
            
            # Get count of chunks for the patient
            results = self.chunks_collection.get(
                where={"patient_id": patient_id},
                include=[]  # Only count, don't retrieve data
            )
            
            count = len(results['ids']) if results and results['ids'] else 0
            logger.info(f"Total chunks for patient {patient_id}: {count}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to count chunks for patient {patient_id}: {e}")
            return 0
    
    async def delete_patient_chunks(self, patient_id: str) -> bool:
        """
        Delete all chunks for a specific patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            if not self.chunks_collection:
                logger.warning("Chunks collection not available")
                return False
            
            # Delete all chunks for the patient
            self.chunks_collection.delete(
                where={"patient_id": patient_id}
            )
            
            logger.info(f"Successfully deleted chunks for patient {patient_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete chunks for patient {patient_id}: {e}")
            return False
    

# Create global vector store instance
vector_store = VectorStore()
