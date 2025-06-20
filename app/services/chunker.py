"""
Medical document chunking and timeline structuring service.

This module provides intelligent chunking of medical documents with support for
visit-based, topic-based, temporal, and semantic chunking strategies. It also
creates structured medical timelines from processed chunks.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from app.schemas.chunking import (
    ChunkingStrategy, ChunkingConfig, MedicalChunk, MedicalEntity,
    TimelineEntry, PatientTimeline, ChunkRequest, ChunkResponse,
    TimelineRequest, TimelineResponse
)
from app.services.medical_patterns import MedicalPatternExtractor

# Configure logging
logger = logging.getLogger(__name__)


class MedicalChunker:
    """
    Intelligent medical document chunker with timeline structuring capabilities.
    
    Supports multiple chunking strategies optimized for medical content:
    - Visit-based: Groups by medical encounters
    - Topic-based: Groups by medical topics
    - Temporal: Chronological organization    - Semantic: Embedding-based similarity grouping
    """
    
    def __init__(self):
        """Initialize the medical chunker with required models."""
        self.embeddings_model = None
        self.nlp_model = None
        self.pattern_extractor = MedicalPatternExtractor()
        self._initialize_models()
        
        # Medical patterns for entity extraction
        self.medical_patterns = {
            'date': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
            ],
            'medication': [
                r'\b\w+\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|cc|units?)\b',
                r'\b(?:metformin|insulin|lisinopril|amlodipine|atorvastatin|levothyroxine)\b'
            ],
            'vitals': [
                r'\b(?:BP|Blood Pressure):?\s*\d{2,3}[/\\]\d{2,3}\b',
                r'\b(?:HR|Heart Rate):?\s*\d{2,3}\s*(?:bpm)?\b',
                r'\b(?:Temp|Temperature):?\s*\d{2,3}(?:\.\d)?\s*°?[FC]?\b'
            ],
            'lab_values': [
                r'\b(?:HbA1c|A1C):?\s*\d+\.\d+%?\b',
                r'\b(?:glucose|sugar):?\s*\d{2,3}\s*(?:mg/dL)?\b',
                r'\b(?:cholesterol|LDL|HDL):?\s*\d{2,3}\s*(?:mg/dL)?\b'            ]
        }
        
    def _initialize_models(self):
        """Initialize NLP and embedding models with comprehensive fallback handling."""
        try:
            # Load sentence transformer for semantic embeddings
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
            
            # Load spaCy model for medical NLP with multiple fallback options
            self.nlp_model = self._load_spacy_model()
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.embeddings_model = None
            self.nlp_model = None
    
    def _load_spacy_model(self):
        """Load spaCy model with fallback hierarchy."""
        # Priority order: medical model -> standard model -> minimal model -> None
        model_options = [
            ("en_core_sci_sm", "Medical spaCy model (SciSpaCy)"),
            ("en_core_web_sm", "Standard English spaCy model"),
            ("en_core_web_lg", "Large English spaCy model"),
            ("en_core_web_md", "Medium English spaCy model")
        ]
        
        for model_name, description in model_options:
            try:
                nlp = spacy.load(model_name)
                logger.info(f"{description} loaded successfully")
                return nlp
            except OSError:
                logger.debug(f"{description} not available, trying next option...")
                continue
            except Exception as e:
                logger.warning(f"Error loading {description}: {e}")
                continue
        
        # If no spaCy model is available, log warning and return None
        logger.warning("No spaCy models available - using pattern-based extraction only")
        logger.info("To install spaCy models, run: python setup_spacy_models.py")
        return None
    
    async def process_chunks(self, request: ChunkRequest) -> ChunkResponse:
        """
        Process text into medical chunks based on the specified strategy.
        
        Args:
            request: ChunkRequest containing text and configuration
            
        Returns:
            ChunkResponse with generated chunks and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract medical entities first
            entities = await self._extract_medical_entities(request.text)
            
            # Choose chunking strategy
            if request.config.strategy == ChunkingStrategy.VISIT_BASED:
                chunks = await self._chunk_by_visits(request.text, entities, request.config)
            elif request.config.strategy == ChunkingStrategy.TOPIC_BASED:
                chunks = await self._chunk_by_topics(request.text, entities, request.config)
            elif request.config.strategy == ChunkingStrategy.TEMPORAL:
                chunks = await self._chunk_by_temporal(request.text, entities, request.config)
            elif request.config.strategy == ChunkingStrategy.SEMANTIC:
                chunks = await self._chunk_by_semantic(request.text, entities, request.config)
            else:
                chunks = await self._chunk_fixed_size(request.text, entities, request.config)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(chunks, request.text)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ChunkResponse(
                job_id=request.job_id,
                chunks=chunks,
                total_chunks=len(chunks),
                chunking_strategy=request.config.strategy,
                processing_time=processing_time,
                average_chunk_size=sum(c.char_count for c in chunks) / len(chunks) if chunks else 0,
                medical_entities_found=sum(len(c.medical_entities) for c in chunks),
                timeline_created=False,  # Will be set if timeline is created                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise
    
    async def _extract_medical_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text using enhanced hybrid approach."""
        entities = []
        
        # Use enhanced pattern-based extraction as primary method
        pattern_entities = self.pattern_extractor.extract_entities(text)
        for entity_data in pattern_entities:
            entities.append(MedicalEntity(
                text=entity_data['text'],
                label=entity_data['label'],
                start=entity_data['start'],
                end=entity_data['end'],
                confidence=entity_data['confidence']
            ))
        
        # NLP-based extraction if model is available (as supplementary)
        if self.nlp_model:
            try:
                doc = self.nlp_model(text)
                for ent in doc.ents:
                    # Filter for medical entities with expanded medical labels
                    medical_labels = {
                        'PERSON': 'person', 'DATE': 'date', 'CARDINAL': 'number', 
                        'QUANTITY': 'measurement', 'ORG': 'organization',
                        'DISEASE': 'conditions', 'SYMPTOM': 'symptoms', 'MEDICATION': 'medication',
                        'DOSAGE': 'dosage', 'CONDITION': 'conditions', 'PROCEDURE': 'procedures'
                    }
                    
                    if ent.label_ in medical_labels:
                        entities.append(MedicalEntity(
                            text=ent.text,
                            label=medical_labels[ent.label_],
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.9  # NLP-based confidence
                        ))
                        
                logger.info(f"✅ spaCy NLP extraction found {len([e for e in entities if e.confidence == 0.9])} additional entities")
            except Exception as e:
                logger.warning(f"NLP entity extraction failed: {e}, using pattern-based only")
        else:
            logger.info("ℹ️  Using enhanced pattern-based entity extraction (spaCy not available)")
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        return sorted(entities, key=lambda x: x.start)
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        return sorted(entities, key=lambda x: x.start)
    
    def _deduplicate_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove duplicate entities with overlap resolution."""
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: x.start)
        
        deduplicated = [entities[0]]
        
        for entity in entities[1:]:
            last_entity = deduplicated[-1]
            
            # Check for overlap
            if entity.start < last_entity.end:
                # Choose entity with higher confidence
                if entity.confidence > last_entity.confidence:
                    deduplicated[-1] = entity
            else:
                deduplicated.append(entity)
        
        return deduplicated
    
    async def _chunk_by_visits(self, text: str, entities: List[MedicalEntity], config: ChunkingConfig) -> List[MedicalChunk]:
        """Chunk text by medical visits/encounters."""
        chunks = []
        
        # Find visit indicators
        visit_patterns = [
            r'visit\s+(?:date|on)',
            r'appointment\s+(?:date|on)',
            r'encounter\s+(?:date|on)',
            r'seen\s+on',
            r'follow-up\s+(?:visit|appointment)'
        ]
        
        visit_boundaries = []
        for pattern in visit_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                visit_boundaries.append(match.start())
        
        # Add document boundaries
        visit_boundaries = [0] + sorted(visit_boundaries) + [len(text)]
        
        # Create chunks for each visit section
        for i in range(len(visit_boundaries) - 1):
            start = visit_boundaries[i]
            end = visit_boundaries[i + 1]
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) > config.min_chunk_size:
                # Extract entities for this chunk
                chunk_entities = [e for e in entities if start <= e.start < end]
                
                # Detect visit date
                visit_date = self._extract_visit_date(chunk_text)
                
                chunks.append(MedicalChunk(
                    chunk_id=f"visit_chunk_{i+1}",
                    text=chunk_text,
                    start_index=start,
                    end_index=end,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    semantic_score=0.85,  # High coherence for visit-based chunks
                    medical_entities=chunk_entities,
                    chunk_type="visit",
                    visit_date=visit_date,
                    medical_topics=self._extract_medical_topics(chunk_text),
                    metadata={"visit_number": i+1}
                ))
        
        return chunks
    
    async def _chunk_by_topics(self, text: str, entities: List[MedicalEntity], config: ChunkingConfig) -> List[MedicalChunk]:
        """Chunk text by medical topics."""
        # Define medical topic keywords
        topic_keywords = {
            'symptoms': ['pain', 'ache', 'fever', 'nausea', 'fatigue', 'shortness of breath'],
            'medications': ['prescribed', 'medication', 'drug', 'therapy', 'treatment'],
            'tests': ['test', 'lab', 'blood work', 'x-ray', 'scan', 'examination'],
            'diagnosis': ['diagnosis', 'condition', 'disease', 'disorder', 'syndrome'],
            'procedures': ['procedure', 'surgery', 'operation', 'intervention']
        }
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentence_topics = []
        
        # Classify each sentence by topic
        for sentence in sentences:
            sentence_topic_scores = {}
            for topic, keywords in topic_keywords.items():
                score = sum(1 for keyword in keywords if keyword.lower() in sentence.lower())
                sentence_topic_scores[topic] = score
            
            # Assign to highest scoring topic
            best_topic = max(sentence_topic_scores, key=sentence_topic_scores.get)
            sentence_topics.append((sentence, best_topic))
        
        # Group sentences by topic
        topic_groups = defaultdict(list)
        for sentence, topic in sentence_topics:
            topic_groups[topic].append(sentence)
        
        # Create chunks for each topic
        chunks = []
        
        for topic, topic_sentences in topic_groups.items():
            if not topic_sentences:
                continue
                
            chunk_text = '. '.join(topic_sentences).strip()
            if len(chunk_text) > config.min_chunk_size:
                # Find chunk position in original text
                chunk_start = text.find(topic_sentences[0])
                chunk_end = chunk_start + len(chunk_text)
                
                # Extract entities for this chunk
                chunk_entities = [e for e in entities if chunk_start <= e.start < chunk_end]
                
                chunks.append(MedicalChunk(
                    chunk_id=f"topic_chunk_{topic}_{len(chunks)+1}",
                    text=chunk_text,
                    start_index=chunk_start,
                    end_index=chunk_end,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    semantic_score=0.8,
                    medical_entities=chunk_entities,
                    chunk_type="topic",
                    medical_topics=[topic],
                    metadata={"primary_topic": topic}
                ))
        
        return chunks
    
    async def _chunk_by_temporal(self, text: str, entities: List[MedicalEntity], config: ChunkingConfig) -> List[MedicalChunk]:
        """Chunk text by temporal/chronological order."""
        # Extract dates and their positions
        date_entities = [e for e in entities if e.label == 'date']
        
        if not date_entities:
            # Fallback to paragraph-based chunking
            return await self._chunk_by_paragraphs(text, entities, config)
        
        # Sort by date position
        date_entities.sort(key=lambda x: x.start)
        
        chunks = []
        
        # Create temporal boundaries
        boundaries = [0] + [e.start for e in date_entities] + [len(text)]
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) > config.min_chunk_size:
                # Extract entities for this chunk
                chunk_entities = [e for e in entities if start <= e.start < end]
                
                # Parse date if available
                chunk_date = None
                if i < len(date_entities):
                    chunk_date = self._parse_date(date_entities[i].text)
                
                chunks.append(MedicalChunk(
                    chunk_id=f"temporal_chunk_{i+1}",
                    text=chunk_text,
                    start_index=start,
                    end_index=end,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    semantic_score=0.75,
                    medical_entities=chunk_entities,
                    chunk_type="temporal",
                    visit_date=chunk_date,
                    medical_topics=self._extract_medical_topics(chunk_text),                    metadata={"temporal_order": i+1}
                ))
        
        return chunks
    
    async def _chunk_by_semantic(self, text: str, entities: List[MedicalEntity], config: ChunkingConfig) -> List[MedicalChunk]:
        """Chunk text by semantic similarity using embeddings."""
        if not self.embeddings_model:
            logger.warning("Embeddings model not available, falling back to fixed size chunking")
            return await self._chunk_fixed_size(text, entities, config)
        
        # For medical reports, use more flexible text segmentation
        # Split by lines first, then by sentences for better medical document handling
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # If we have structured lines (like lab reports), group them semantically
        if len(lines) > 1:
            text_segments = lines
        else:
            # Fallback to sentence splitting for narrative text
            text_segments = re.split(r'[.!?]+', text)
            text_segments = [s.strip() for s in text_segments if s.strip() and len(s.strip()) > 10]
        
        if len(text_segments) < 2:
            logger.info("Insufficient text segments for semantic chunking, using fixed size")
            return await self._chunk_fixed_size(text, entities, config)
        
        # Generate embeddings for text segments
        try:
            embeddings = self.embeddings_model.encode(text_segments)
        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}, falling back to fixed size chunking")
            return await self._chunk_fixed_size(text, entities, config)
        
        # Calculate semantic similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
          # Group segments by semantic similarity with adjusted threshold
        # For medical lab reports, use a more lenient threshold
        base_threshold = config.semantic_threshold
        if len(text_segments) > 10:  # Likely a structured medical report
            effective_threshold = max(0.3, base_threshold - 0.3)
        else:
            effective_threshold = max(0.4, base_threshold - 0.2)
            
        logger.info(f"Using semantic threshold: {effective_threshold} (original: {base_threshold})")
        
        segment_groups = []
        used_segments = set()
        
        for i, segment in enumerate(text_segments):
            if i in used_segments:
                continue
                
            # Find similar segments
            similar_indices = [i]
            for j in range(i + 1, len(text_segments)):
                if j not in used_segments and similarity_matrix[i][j] > effective_threshold:
                    similar_indices.append(j)
                    used_segments.add(j)
            
            used_segments.add(i)
            segment_groups.append([text_segments[idx] for idx in similar_indices])
        
        # Create chunks from segment groups
        chunks = []
        
        for group_idx, segment_group in enumerate(segment_groups):
            # Reconstruct chunk text preserving original formatting
            if len(lines) > 1 and segment_group == lines:  # Line-based segmentation
                chunk_text = '\n'.join(segment_group).strip()
            else:  # Sentence-based segmentation
                chunk_text = '. '.join(segment_group).strip()
              # Check minimum chunk size - be more lenient for structured medical documents
            effective_min_size = config.min_chunk_size
            if '\n' in text and len(text_segments) > 5:  # Structured document
                effective_min_size = min(config.min_chunk_size, 60)  # Lower minimum for structured docs
            
            if len(chunk_text) < effective_min_size:
                logger.debug(f"Skipping chunk (too small): {len(chunk_text)} < {effective_min_size}")
                continue
            
            # Find chunk boundaries in original text
            first_segment = segment_group[0]
            last_segment = segment_group[-1]
            
            chunk_start = text.find(first_segment)
            if chunk_start == -1:
                continue  # Skip if segment not found
                
            # Find end position more accurately
            chunk_end = chunk_start + len(chunk_text)
            
            # Adjust end position to match the actual last segment
            last_segment_end = text.find(last_segment, chunk_start) + len(last_segment)
            if last_segment_end > chunk_end:
                chunk_end = last_segment_end
            
            # Extract entities for this chunk
            chunk_entities = [e for e in entities if chunk_start <= e.start < chunk_end]
            
            # Calculate semantic coherence score
            if len(segment_group) > 1:
                try:
                    group_embeddings = self.embeddings_model.encode(segment_group)
                    semantic_score = np.mean(cosine_similarity(group_embeddings))
                except:
                    semantic_score = 0.7  # Default score if calculation fails
            else:
                semantic_score = 1.0
            
            chunks.append(MedicalChunk(
                chunk_id=f"semantic_chunk_{group_idx+1}",
                text=chunk_text,
                start_index=chunk_start,
                end_index=chunk_end,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                semantic_score=float(semantic_score),
                medical_entities=chunk_entities,
                chunk_type="semantic",
                medical_topics=self._extract_medical_topics(chunk_text),
                metadata={"semantic_group": group_idx+1, "segment_count": len(segment_group)}
            ))
        
        return chunks
    
    async def _chunk_fixed_size(self, text: str, entities: List[MedicalEntity], config: ChunkingConfig) -> List[MedicalChunk]:
        """Chunk text using fixed size with overlap."""
        chunks = []
        chunk_size = config.chunk_size
        overlap = config.overlap
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            
            if len(chunk_text.strip()) < config.min_chunk_size:
                break
            
            # Extract entities for this chunk
            chunk_entities = [e for e in entities if i <= e.start < i + len(chunk_text)]
            
            chunks.append(MedicalChunk(
                chunk_id=f"fixed_chunk_{len(chunks)+1}",
                text=chunk_text.strip(),
                start_index=i,
                end_index=i + len(chunk_text),
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                semantic_score=0.6,  # Lower coherence for fixed-size chunks
                medical_entities=chunk_entities,
                chunk_type="fixed",
                medical_topics=self._extract_medical_topics(chunk_text),
                metadata={"chunk_number": len(chunks)+1}
            ))
        
        return chunks
    
    async def _chunk_by_sentences(self, text: str, entities: List[MedicalEntity], config: ChunkingConfig) -> List[MedicalChunk]:
        """Chunk text by sentence boundaries."""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_length + len(sentence) > config.chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = '. '.join(current_chunk)
                chunk_end = start_pos + len(chunk_text)
                
                chunk_entities = [e for e in entities if start_pos <= e.start < chunk_end]
                
                chunks.append(MedicalChunk(
                    chunk_id=f"sentence_chunk_{len(chunks)+1}",
                    text=chunk_text,
                    start_index=start_pos,
                    end_index=chunk_end,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    semantic_score=0.75,
                    medical_entities=chunk_entities,
                    chunk_type="sentence",
                    medical_topics=self._extract_medical_topics(chunk_text),
                    metadata={"sentence_count": len(current_chunk)}
                ))
                
                start_pos = chunk_end
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = '. '.join(current_chunk)
            chunk_end = start_pos + len(chunk_text)
            
            chunk_entities = [e for e in entities if start_pos <= e.start < chunk_end]
            
            chunks.append(MedicalChunk(
                chunk_id=f"sentence_chunk_{len(chunks)+1}",
                text=chunk_text,
                start_index=start_pos,
                end_index=chunk_end,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                semantic_score=0.75,
                medical_entities=chunk_entities,
                chunk_type="sentence",
                medical_topics=self._extract_medical_topics(chunk_text),
                metadata={"sentence_count": len(current_chunk)}
            ))
        
        return chunks
    
    async def _chunk_by_paragraphs(self, text: str, entities: List[MedicalEntity], config: ChunkingConfig) -> List[MedicalChunk]:
        """Chunk text by paragraph boundaries."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_pos = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) < config.min_chunk_size:
                current_pos += len(paragraph) + 2  # +2 for \n\n
                continue
            
            # Extract entities for this paragraph
            para_end = current_pos + len(paragraph)
            chunk_entities = [e for e in entities if current_pos <= e.start < para_end]
            
            chunks.append(MedicalChunk(
                chunk_id=f"paragraph_chunk_{para_idx+1}",
                text=paragraph,
                start_index=current_pos,
                end_index=para_end,
                word_count=len(paragraph.split()),
                char_count=len(paragraph),
                semantic_score=0.8,  # Good coherence for paragraph chunks
                medical_entities=chunk_entities,
                chunk_type="paragraph",
                medical_topics=self._extract_medical_topics(paragraph),
                metadata={"paragraph_number": para_idx+1}
            ))
            
            current_pos = para_end + 2
        
        return chunks
    
    def _extract_visit_date(self, text: str) -> Optional[datetime]:
        """Extract visit date from chunk text."""
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._parse_date(match.group(1))
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string into datetime object."""
        date_formats = [
            '%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y',
            '%Y-%m-%d', '%B %d, %Y', '%b %d, %Y',
            '%B %d %Y', '%b %d %Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_medical_topics(self, text: str) -> List[str]:
        """Extract medical topics from text."""
        medical_topics = []
        
        # Topic keyword mapping
        topic_keywords = {
            'cardiology': ['heart', 'cardiac', 'chest pain', 'blood pressure', 'hypertension'],
            'diabetes': ['diabetes', 'glucose', 'insulin', 'HbA1c', 'blood sugar'],
            'respiratory': ['lung', 'breathing', 'cough', 'asthma', 'pneumonia'],
            'neurology': ['headache', 'migraine', 'seizure', 'stroke', 'neurological'],
            'gastroenterology': ['stomach', 'digestion', 'nausea', 'abdominal', 'gastric'],
            'orthopedics': ['bone', 'joint', 'fracture', 'arthritis', 'orthopedic']
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                medical_topics.append(topic)
        
        return medical_topics
    
    def _calculate_quality_metrics(self, chunks: List[MedicalChunk], original_text: str) -> Dict[str, float]:
        """Calculate quality metrics for chunking results."""
        if not chunks:
            return {"coverage": 0.0, "coherence": 0.0, "entity_distribution": 0.0}
        
        # Coverage: percentage of original text covered by chunks
        covered_chars = sum(c.char_count for c in chunks)
        coverage = covered_chars / len(original_text) if original_text else 0.0
        
        # Coherence: average semantic score of chunks
        coherence = np.mean([c.semantic_score or 0.0 for c in chunks])
        
        # Entity distribution: how evenly entities are distributed across chunks
        entity_counts = [len(c.medical_entities) for c in chunks]
        entity_distribution = 1.0 - (np.std(entity_counts) / (np.mean(entity_counts) + 1e-6))
        
        return {
            "coverage": float(coverage),
            "coherence": float(coherence),
            "entity_distribution": float(entity_distribution),
            "avg_chunk_size": float(np.mean([c.char_count for c in chunks])),
            "chunk_size_std": float(np.std([c.char_count for c in chunks]))
        }
    
    async def create_patient_timeline(self, request: TimelineRequest) -> TimelineResponse:
        """
        Create a structured patient timeline from medical chunks.
        
        Args:
            request: TimelineRequest with patient ID and chunks
            
        Returns:
            TimelineResponse with structured timeline
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Group chunks by visit date
            date_groups = defaultdict(list)
            
            for chunk in request.chunks:
                if chunk.visit_date:
                    date_key = chunk.visit_date.date()
                    date_groups[date_key].append(chunk)
                else:
                    # Try to extract date from chunk
                    extracted_date = self._extract_visit_date(chunk.text)
                    if extracted_date:
                        date_key = extracted_date.date()
                        date_groups[date_key].append(chunk)
                    else:
                        # Use a default date for undated chunks
                        date_groups[datetime.now().date()].append(chunk)
            
            # Create timeline entries
            timeline_entries = []
            
            for visit_date, chunks in date_groups.items():
                # Extract medical information from chunks
                symptoms = []
                tests = []
                medications = []
                diagnoses = []
                procedures = []
                progress_notes = []
                chunk_ids = [chunk.chunk_id for chunk in chunks]
                
                # Aggregate information from all chunks for this date
                for chunk in chunks:
                    # Extract structured information based on entities and text
                    chunk_info = self._extract_structured_info(chunk)
                    symptoms.extend(chunk_info.get('symptoms', []))
                    tests.extend(chunk_info.get('tests', []))
                    medications.extend(chunk_info.get('medications', []))
                    diagnoses.extend(chunk_info.get('diagnoses', []))
                    procedures.extend(chunk_info.get('procedures', []))
                    progress_notes.extend(chunk_info.get('progress_notes', []))
                
                # Remove duplicates
                symptoms = list(set(symptoms))
                tests = list(set(tests))
                medications = list(set(medications))
                diagnoses = list(set(diagnoses))
                procedures = list(set(procedures))
                
                # Calculate confidence based on entity confidence and chunk quality
                confidence = np.mean([
                    np.mean([e.confidence for e in chunk.medical_entities]) if chunk.medical_entities else 0.5
                    for chunk in chunks
                ])
                
                timeline_entry = TimelineEntry(
                    date=datetime.combine(visit_date, datetime.min.time()),
                    visit_id=f"visit_{visit_date.isoformat()}",
                    symptoms=symptoms,
                    tests=tests,
                    medications=medications,
                    diagnoses=diagnoses,
                    procedures=procedures,
                    progress_notes=progress_notes,
                    chunk_ids=chunk_ids,
                    confidence=float(confidence)
                )
                
                # Only include entries that meet confidence threshold
                if timeline_entry.confidence >= request.confidence_threshold:
                    timeline_entries.append(timeline_entry)
            
            # Create patient timeline
            if timeline_entries:
                sorted_entries = sorted(timeline_entries, key=lambda x: x.date)
                date_range = {
                    "start": sorted_entries[0].date,
                    "end": sorted_entries[-1].date
                }
                
                # Generate summary if requested
                summary = None
                if request.include_summary:
                    summary = self._generate_timeline_summary(sorted_entries)
                
                patient_timeline = PatientTimeline(
                    patient_id=request.patient_id,
                    timeline_entries=sorted_entries,
                    total_visits=len(sorted_entries),
                    date_range=date_range,
                    summary=summary,
                    metadata={
                        "total_chunks_processed": len(request.chunks),
                        "confidence_threshold": request.confidence_threshold,
                        "creation_date": datetime.now().isoformat()
                    }
                )
            else:
                # Empty timeline
                patient_timeline = PatientTimeline(
                    patient_id=request.patient_id,
                    timeline_entries=[],
                    total_visits=0,
                    date_range={"start": datetime.now(), "end": datetime.now()},
                    summary="No timeline entries met the confidence threshold."
                )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            overall_confidence = np.mean([entry.confidence for entry in timeline_entries]) if timeline_entries else 0.0
            
            warnings = []
            if len(timeline_entries) < len(date_groups):
                warnings.append(f"Some timeline entries were filtered out due to low confidence (threshold: {request.confidence_threshold})")
            
            return TimelineResponse(
                patient_timeline=patient_timeline,
                processing_time=processing_time,
                confidence_score=float(overall_confidence),
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Timeline creation failed: {e}")
            raise
    
    def _extract_structured_info(self, chunk: MedicalChunk) -> Dict[str, List[str]]:
        """Extract structured medical information from a chunk."""
        info = {
            'symptoms': [],
            'tests': [],
            'medications': [],
            'diagnoses': [],
            'procedures': [],
            'progress_notes': []
        }
        
        text = chunk.text.lower()
        
        # Symptom keywords
        symptom_keywords = ['pain', 'ache', 'fever', 'nausea', 'fatigue', 'dizzy', 'shortness of breath']
        for keyword in symptom_keywords:
            if keyword in text:
                # Extract sentence containing the symptom
                sentences = chunk.text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        info['symptoms'].append(sentence.strip())
                        break
        
        # Test keywords
        test_keywords = ['blood test', 'x-ray', 'mri', 'ct scan', 'ultrasound', 'lab', 'hba1c']
        for keyword in test_keywords:
            if keyword in text:
                # Extract test results
                for entity in chunk.medical_entities:
                    if entity.label in ['lab_values', 'vitals']:
                        info['tests'].append(entity.text)
        
        # Medication extraction from entities
        for entity in chunk.medical_entities:
            if entity.label == 'medication':
                info['medications'].append(entity.text)
        
        # Look for diagnosis patterns
        diagnosis_patterns = [
            r'diagnosed with ([^.]+)',
            r'diagnosis:?\s*([^.]+)',
            r'condition:?\s*([^.]+)'
        ]
        
        for pattern in diagnosis_patterns:
            matches = re.findall(pattern, chunk.text, re.IGNORECASE)
            info['diagnoses'].extend(matches)
        
        # Look for procedure patterns
        procedure_patterns = [
            r'procedure:?\s*([^.]+)',
            r'surgery:?\s*([^.]+)',
            r'operation:?\s*([^.]+)'
        ]
        
        for pattern in procedure_patterns:
            matches = re.findall(pattern, chunk.text, re.IGNORECASE)
            info['procedures'].extend(matches)
        
        # Progress notes - look for assessment and plan sections
        if any(keyword in text for keyword in ['assessment', 'plan', 'progress', 'follow-up']):
            info['progress_notes'].append(chunk.text[:200] + '...' if len(chunk.text) > 200 else chunk.text)
        
        return info
    
    def _generate_timeline_summary(self, timeline_entries: List[TimelineEntry]) -> str:
        """Generate a summary of the patient timeline."""
        if not timeline_entries:
            return "No medical timeline available."
        
        total_visits = len(timeline_entries)
        date_range = f"{timeline_entries[0].date.strftime('%Y-%m-%d')} to {timeline_entries[-1].date.strftime('%Y-%m-%d')}"
        
        # Count unique items across all visits
        all_diagnoses = set()
        all_medications = set()
        all_symptoms = set()
        
        for entry in timeline_entries:
            all_diagnoses.update(entry.diagnoses)
            all_medications.update(entry.medications)
            all_symptoms.update(entry.symptoms)
        
        summary_parts = [
            f"Patient medical timeline spanning {date_range} with {total_visits} visit(s)."
        ]
        
        if all_diagnoses:
            summary_parts.append(f"Key diagnoses: {', '.join(list(all_diagnoses)[:3])}.")
        
        if all_medications:
            summary_parts.append(f"Medications: {', '.join(list(all_medications)[:3])}.")
        
        if all_symptoms:
            summary_parts.append(f"Primary symptoms: {', '.join(list(all_symptoms)[:3])}.")
        
        return ' '.join(summary_parts)


# Global chunker instance
chunker = MedicalChunker()
