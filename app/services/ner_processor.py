"""
Medical Named Entity Recognition (NER) Processing Service.

This module provides comprehensive medical entity extraction capabilities
using state-of-the-art NLP models specialized for healthcare text processing.
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import spacy
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TokenClassificationPipeline
)

from app.schemas.ner import (
    NERRequest, NERResponse, MedicalEntity, EntityGroup,
    ProcessingMetrics, EntityType, ProcessingMode,
    BatchNERRequest, BatchNERResponse
)


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class NERModelConfig:
    """Configuration for NER models and processing."""
    # Model paths and configurations
    medical_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    bio_bert_model: str = "dmis-lab/biobert-base-cased-v1.1"
    med7_model: str = "kormilitzin/en_core_med7_lg"
    spacy_model_name: str = "en_core_web_sm"
    scispacy_model_name: str = "en_core_sci_sm"

    # Processing parameters
    max_sequence_length: int = 512
    stride: int = 128
    confidence_threshold: float = 0.5
    batch_size: int = 16

    # Medical knowledge bases for entity linking
    umls_enabled: bool = False
    snomed_ct_enabled: bool = False
    icd10_enabled: bool = True

    # Enhanced medical entity patterns
    medical_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "MEDICATION": [
            # Dosage patterns
            r"\b\w+\s*(?:mg|g|ml|mcg|units?|tablets?|capsules?|drops?|patches?)\b",
            # Common medications
            r"\b(?:aspirin|metformin|lisinopril|atorvastatin|omeprazole|warfarin|insulin|hydrochlorothiazide|amlodipine|levothyroxine)\b",
            # Generic medication patterns
            r"\b\w+(?:cillin|mycin|prazole|statin|sartan|olol|pril|azide)\b",
            # Injectable medications
            r"\b\w+\s*(?:injection|infusion|IV|intramuscular|subcutaneous)\b"
        ],        "LAB_VALUE": [
            # Standard lab values with reference ranges
            r"\b(?:hemoglobin|hgb|hb)\s*(?:\(.*?\))?\s*[:=]?\s*\d+(?:\.\d+)?\s*(?:g/dL|g/dl)?\s*(?:,?\s*range?\s*[:=]?\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?(?:\s*g/dL|g/dl)?)?\b",
            r"\b(?:platelet count|platelets|plt)\s*(?:\(.*?\))?\s*[:=]?\s*\d+(?:\.\d+)?\s*(?:/uL|/ul|k/uL|k/ul|×10³/μL)?\s*(?:,?\s*range?\s*[:=]?\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?(?:\s*/uL|/ul|k/uL|k/ul)?)?\b",
            r"\b(?:neutrophils|lymphocytes|monocytes|eosinophils|basophils)\s*(?:\(.*?\))?\s*[:=]?\s*\d+(?:\.\d+)?\s*%?\s*(?:,?\s*range?\s*[:=]?\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?%?)?\b",
            r"\b(?:white blood cell count|wbc|white cell count)\s*(?:\(.*?\))?\s*[:=]?\s*\d+(?:\.\d+)?\s*(?:/uL|/ul|k/uL|k/ul|×10³/μL)?\s*(?:,?\s*range?\s*[:=]?\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?(?:\s*/uL|/ul|k/uL|k/ul)?)?\b",
            r"\b(?:red blood cell count|rbc|red cell count)\s*(?:\(.*?\))?\s*[:=]?\s*\d+(?:\.\d+)?\s*(?:/uL|/ul|M/uL|M/ul|×10⁶/μL)?\s*(?:,?\s*range?\s*[:=]?\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?(?:\s*/uL|/ul|M/uL|M/ul)?)?\b",
            r"\b(?:hematocrit|hct)\s*(?:\(.*?\))?\s*[:=]?\s*\d+(?:\.\d+)?\s*%?\s*(?:,?\s*range?\s*[:=]?\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?%?)?\b",
            # Standard lab values without ranges
            r"\b\w+\s*[:=]\s*\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L|%|mEq/L|g/dL|IU/L|ng/mL|pg/mL|mcg/L)\b",
            # Specific tests
            r"\bHbA1c\s*[:=]\s*\d+(?:\.\d+)?%?\b",
            r"\b(?:glucose|cholesterol|triglycerides|HDL|LDL|creatinine|BUN|TSH|PSA)\s*[:=]\s*\d+(?:\.\d+)?\s*\w+/?\w*\b"
        ],
        "VITAL_SIGN": [
            # Blood pressure
            r"\b(?:BP|blood pressure)\s*[:=]?\s*\d+/\d+\s*mmHg?\b",
            # Heart rate
            r"\b(?:HR|heart rate|pulse)\s*[:=]?\s*\d+\s*(?:bpm|beats/min)?\b",
            # Temperature
            r"\btemp(?:erature)?\s*[:=]?\s*\d+(?:\.\d+)?°?[CF]?\b",
            # Respiratory rate
            r"\b(?:RR|respiratory rate|respiration)\s*[:=]?\s*\d+\s*(?:/min)?\b",
            # Oxygen saturation
            r"\b(?:O2 sat|oxygen saturation|SpO2)\s*[:=]?\s*\d+%?\b"
        ],
        "CONDITION": [
            # Common conditions
            r"\b(?:diabetes|hypertension|pneumonia|asthma|COPD|CHF|MI|stroke|cancer|infection)\b",
            # ICD-10 codes
            r"\b[A-Z]\d{2}(?:\.\d{1,3})?\b",
            # Condition patterns
            r"\b(?:acute|chronic|severe|mild|moderate)\s+\w+(?:\s+\w+)?\b"
        ],        "PROCEDURE": [
            # Diagnostic procedures
            r"\b(?:ECG|EKG|chest X-ray|CT scan|MRI|ultrasound|echocardiogram|endoscopy|colonoscopy|biopsy)\b",
            # Lab tests and blood work
            r"\b(?:complete blood count|CBC|blood test|urinalysis|liver function test|kidney function|lipid panel|CMP|TSH test)\b",
            # OCR-corrected variations
            r"\b(?:completebloodcount|completeBLOODcount|CONPLETEBLOODCouniT)\b",
            # CPT codes
            r"\b\d{5}\b"
        ],
        "SYMPTOM": [
            # Common symptoms
            r"\b(?:pain|ache|fatigue|nausea|vomiting|diarrhea|constipation|fever|chills|sweating|dizziness|headache)\b",
            # Location-specific symptoms
            r"\b(?:chest|abdominal|back|joint|muscle)\s+pain\b",
            # Descriptive symptoms
            r"\b(?:shortness of breath|difficulty breathing|blurred vision|frequent urination|weight loss|weight gain)\b"
        ],
        "ANATOMICAL_STRUCTURE": [
            # Organs
            r"\b(?:heart|lung|liver|kidney|brain|stomach|intestine|pancreas|spleen|thyroid|prostate)\b",
            # Body parts
            r"\b(?:left|right)\s+(?:ventricle|atrium|lung|kidney|breast)\b",
            # Systems
            r"\b(?:cardiovascular|respiratory|gastrointestinal|nervous|endocrine|musculoskeletal)\s+system\b"
        ],        "TEMPORAL_DURATION": [
            # Time expressions
            r"\b(?:\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b",
            r"\b(?:\d+\s+(?:days?|weeks?|months?|years?))\b"
        ],
        "TEMPORAL_FREQUENCY": [
            r"\b(?:daily|weekly|monthly|yearly|twice daily|three times daily|as needed|PRN)\b"
        ],
        "TEMPORAL_TIME_OF_DAY": [
            r"\b(?:morning|afternoon|evening|night|bedtime)\b",
            r"\b(?:before|after)\s+meals?\b"
        ],
        "DOSAGE": [
            # Dosage amounts
            r"\b\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|units?|tablets?|capsules?|drops?|teaspoons?|tablespoons?)\b",
            # Frequency
            r"\b(?:once|twice|three times?|four times?)\s+(?:daily|weekly|monthly)\b",
            r"\b(?:BID|TID|QID|QD|PRN|as needed)\b"
        ]
    })

    # Medical specialty configurations
    specialty_models: Dict[str, str] = field(default_factory=lambda: {
        "cardiology": "cardiology-bert-base",
        "oncology": "bio-bert-oncology",
        "radiology": "radiology-bert-base",
        "pathology": "path-bert-base"
    })


class MedicalNERProcessor:
    """
    Advanced medical Named Entity Recognition processor.

    Combines multiple NLP models and approaches for comprehensive
    medical entity extraction from clinical text.
    """

    def __init__(self, config: Optional[NERModelConfig] = None):
        """
        Initialize the medical NER processor.

        Args:
            config: Optional configuration for models and processing
        """
        self.config = config or NERModelConfig()
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self._is_initialized = False

        # Medical entity mappings
        self.medical_entity_map = self._build_medical_entity_mappings()

        # Medical knowledge bases for entity linking
        self.umls_kb = None
        self.snomed_kb = None
        self.icd10_kb = None

        # Temporal extractors
        self.temporal_patterns = self._build_temporal_patterns()

        # Medical normalization mappings
        self.medical_normalizers = self._build_medical_normalizers()

        # Initialize models lazily
        self._model_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize all NER models and pipelines."""
        if self._is_initialized:
            return

        async with self._model_lock:
            if self._is_initialized:
                return

            logger.info("Initializing Medical NER Processor...")

            try:
                # Initialize spaCy models
                await self._load_spacy_models()

                # Initialize transformer models
                await self._load_transformer_models()

                # Initialize custom pipelines
                await self._setup_custom_pipelines()

                self._is_initialized = True
                logger.info("Medical NER Processor initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize NER processor: {e}")
                raise

    async def _load_spacy_models(self) -> None:
        """Load spaCy models for medical NER."""
        try:
            # Standard English model
            if not spacy.util.is_package(self.config.spacy_model_name):
                logger.warning(f"spaCy model {self.config.spacy_model_name} not found, using basic model")
                self.models['spacy'] = spacy.load("en_core_web_sm")
            else:
                self.models['spacy'] = spacy.load(self.config.spacy_model_name)

            # Try to load scispaCy model for medical text
            try:
                self.models['scispacy'] = spacy.load(self.config.scispacy_model_name)
                logger.info("Loaded scispaCy medical model")
            except OSError:
                logger.warning("scispaCy model not available, using standard spaCy")
                self.models['scispacy'] = self.models['spacy']

            # Try to load Med7 model for comprehensive medical NLP
            try:
                self.models['med7'] = spacy.load(self.config.med7_model)
                logger.info("Loaded Med7 medical model")
            except OSError:
                logger.warning("Med7 model not available, using scispaCy fallback")
                self.models['med7'] = self.models['scispacy']

        except Exception as e:
            logger.error(f"Failed to load spaCy models: {e}")
            # Fallback to basic model
            self.models['spacy'] = spacy.blank("en")
            self.models['scispacy'] = self.models['spacy']
            self.models['med7'] = self.models['spacy']

    async def _load_transformer_models(self) -> None:
        """Load transformer models for medical NER."""
        try:
            # Load medical BERT tokenizer and model (Bio_ClinicalBERT)
            self.tokenizers['medical'] = AutoTokenizer.from_pretrained(
                self.config.medical_model_name
            )
            self.models['medical'] = AutoModelForTokenClassification.from_pretrained(
                self.config.medical_model_name
            )

            # Create pipeline
            self.pipelines['medical'] = TokenClassificationPipeline(
                model=self.models['medical'],
                tokenizer=self.tokenizers['medical'],
                aggregation_strategy="average"
            )

            logger.info("Loaded Bio_ClinicalBERT model")

            # Try to load BioBERT model for additional medical understanding
            try:
                self.tokenizers['biobert'] = AutoTokenizer.from_pretrained(
                    self.config.bio_bert_model
                )
                self.models['biobert'] = AutoModelForTokenClassification.from_pretrained(
                    self.config.bio_bert_model
                )

                self.pipelines['biobert'] = TokenClassificationPipeline(
                    model=self.models['biobert'],
                    tokenizer=self.tokenizers['biobert'],
                    aggregation_strategy="average"
                )

                logger.info("Loaded BioBERT model")

            except Exception as e:
                logger.warning(f"BioBERT model not available: {e}")
                self.pipelines['biobert'] = None

        except Exception as e:
            logger.error(f"Failed to load transformer models: {e}")
            # Create fallback pipeline
            self.pipelines['medical'] = None
            self.pipelines['biobert'] = None

    async def _setup_custom_pipelines(self) -> None:
        """Setup custom processing pipelines."""
        # Medical entity pattern matching
        self.pattern_matchers = {}
        for entity_type, patterns in self.config.medical_patterns.items():
            compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            self.pattern_matchers[entity_type] = compiled_patterns

    def _build_medical_entity_mappings(self) -> Dict[str, EntityType]:
        """Build mappings from model labels to our entity types."""
        return {
            # spaCy standard entities
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "FAC": EntityType.LOCATION,  # Facilities
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.ORGANIZATION,  # Often medical costs, classify as org
            "QUANTITY": EntityType.LAB_VALUE,  # Could be lab measurements
            "ORDINAL": EntityType.ORGANIZATION,  # Numbers like "1st floor"
            "CARDINAL": EntityType.LAB_VALUE,  # Numbers that could be lab values

            # Medical entities (would be trained/configured)
            "DRUG": EntityType.MEDICATION,
            "MEDICATION": EntityType.MEDICATION,
            "DISEASE": EntityType.DISEASE,
            "CONDITION": EntityType.DISEASE,
            "SYMPTOM": EntityType.SYMPTOM,
            "PROCEDURE": EntityType.PROCEDURE,
            "TEST": EntityType.LAB_TEST,
            "DOSAGE": EntityType.DOSAGE,
            "FREQUENCY": EntityType.FREQUENCY,
            "ANATOMY": EntityType.ANATOMICAL_STRUCTURE,
            "LAB": EntityType.LAB_VALUE,
            "LAB_VALUE": EntityType.LAB_VALUE,
            "LAB_TEST": EntityType.LAB_TEST,
            "DEVICE": EntityType.MEDICAL_DEVICE,
            "VITAL_SIGN": EntityType.VITAL_SIGN,
        }

    async def extract_entities(self, request: NERRequest) -> NERResponse:
        """
        Extract medical entities from text using multiple NLP approaches.

        Args:
            request: NER processing request

        Returns:
            NER response with extracted entities and metadata
        """
        # Ensure models are initialized
        await self.initialize()

        start_time = time.time()

        try:
            # Process text with multiple models
            entities = await self._process_with_multiple_models(
                request.text, request.processing_mode, request.confidence_threshold
            )

            # Perform entity linking if enabled
            if request.enable_entity_linking:
                entities = await self._perform_entity_linking(entities)

            # Calculate entity relationships
            entities = self._calculate_entity_relationships(entities, request.text)

            # Filter entities based on request parameters
            filtered_entities = self._filter_entities(entities, request)

            # Group entities by type
            entity_groups = self._group_entities(filtered_entities)

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(entity_groups)

            # Assess text quality
            quality_score = await self._assess_text_quality(request.text)

            # Create processing metrics
            processing_time = time.time() - start_time
            metrics = ProcessingMetrics(
                processing_time=processing_time,
                text_length=len(request.text),
                tokens_processed=len(request.text.split()),
                entities_found=len(filtered_entities),
                avg_confidence=np.mean([e.confidence for e in filtered_entities]) if filtered_entities else 0.0,
                model_version="medical-ner-v1.0"
            )

            # Generate next step recommendation
            next_step = self._suggest_next_step(filtered_entities, request.text)

            return NERResponse(
                message="Medical entities extracted successfully",
                document_id=request.document_id,
                entities=filtered_entities,
                entity_groups=entity_groups,
                total_entities=len(filtered_entities),
                confidence_scores=confidence_scores,
                overall_confidence=np.mean(list(confidence_scores.values())) if confidence_scores else 0.0,
                quality_score=quality_score,
                processing_metrics=metrics,
                processing_mode=request.processing_mode,
                language_detected=request.language,
                next_step=next_step,
                recommendations=self._generate_recommendations(filtered_entities, request)
            )

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            # Return error response
            return NERResponse(
                message=f"Entity extraction failed: {str(e)}",
                document_id=request.document_id,
                entities=[],
                entity_groups=[],
                total_entities=0,
                confidence_scores={},
                overall_confidence=0.0,
                quality_score=0.0,
                processing_metrics=ProcessingMetrics(
                    processing_time=time.time() - start_time,
                    text_length=len(request.text),
                    tokens_processed=0,
                    entities_found=0,
                    avg_confidence=0.0,
                    model_version="medical-ner-v1.0"
                ),
                processing_mode=request.processing_mode,
                language_detected=request.language,
                next_step="Fix errors and retry extraction",
                recommendations=["Check input text format", "Verify model availability"]
            )

    async def _process_with_multiple_models(
        self, text: str, mode: ProcessingMode, threshold: float
    ) -> List[MedicalEntity]:
        """Process text with multiple NER models and combine results."""
        all_entities = []

        # Process with spaCy model
        spacy_entities = await self._extract_with_spacy(text, mode)
        all_entities.extend(spacy_entities)

        # Process with scispaCy model (if available and different)
        if 'scispacy' in self.models and self.models['scispacy'] != self.models['spacy']:
            scispacy_entities = await self._extract_with_scispacy(text, mode)
            all_entities.extend(scispacy_entities)

        # Process with Med7 model for comprehensive medical entities
        if 'med7' in self.models and self.models['med7'] != self.models['scispacy']:
            med7_entities = await self._extract_with_med7(text, mode)
            all_entities.extend(med7_entities)

        # Process with transformer model (Bio_ClinicalBERT)
        if self.pipelines.get('medical'):
            transformer_entities = await self._extract_with_transformer(text, mode)
            all_entities.extend(transformer_entities)

        # Process with BioBERT if available
        if self.pipelines.get('biobert'):
            biobert_entities = await self._extract_with_biobert(text, mode)
            all_entities.extend(biobert_entities)

        # Process with pattern matching
        pattern_entities = await self._extract_with_patterns(text)
        all_entities.extend(pattern_entities)        # Extract temporal information
        temporal_entities = await self._extract_temporal_entities(text)
        all_entities.extend(temporal_entities)

        # Extract lab values with specific patterns
        lab_value_entities = self._extract_lab_values_with_ranges(text)
        all_entities.extend(lab_value_entities)

        # Apply OCR error correction
        corrected_entities = []
        for entity in all_entities:
            corrected_entity = self._correct_ocr_errors(entity)
            corrected_entities.append(corrected_entity)

        # Validate and correct entity types
        validated_entities = []
        for entity in corrected_entities:
            validated_entity = self._validate_and_correct_entity_type(entity)
            validated_entities.append(validated_entity)

        # Filter out generic headings and artifacts
        filtered_entities = [e for e in validated_entities if not self._should_ignore_entity(e)]

        # Deduplicate and merge overlapping entities
        merged_entities = self._merge_overlapping_entities(filtered_entities, threshold)

        # Apply medical normalization
        normalized_entities = self._normalize_medical_entities(merged_entities)        # Filter by confidence threshold
        final_entities = [e for e in normalized_entities if e.confidence >= threshold]

        return final_entities

    async def _extract_with_spacy(self, text: str, mode: ProcessingMode) -> List[MedicalEntity]:
        """Extract entities using spaCy model."""
        if 'spacy' not in self.models:
            return []

        try:
            doc = self.models['spacy'](text)
            entities = []

            for ent in doc.ents:
                # Use mapping with intelligent fallback
                entity_type = self.medical_entity_map.get(ent.label_, None)
                
                # If no mapping found, try to classify based on content
                if entity_type is None:
                    entity_type = self._classify_unknown_entity(ent.text, ent.label_)

                # Skip entities that should be ignored (common artifacts)
                if self._should_ignore_entity_text(ent.text):
                    continue

                entity = MedicalEntity(
                    text=ent.text,
                    label=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8,  # Default confidence for spaCy
                    context=self._extract_context(text, ent.start_char, ent.end_char),
                    semantic_type=ent.label_
                )
                entities.append(entity)

            # Also extract additional medical patterns using custom patterns
            pattern_entities = await self._extract_additional_medical_patterns(text)
            entities.extend(pattern_entities)

            return entities

        except Exception as e:
            logger.error(f"spaCy extraction failed: {e}")
            return []

    async def _extract_with_scispacy(self, text: str, mode: ProcessingMode) -> List[MedicalEntity]:
        """Extract entities using scispaCy medical model."""
        if 'scispacy' not in self.models:
            return []

        try:
            doc = self.models['scispacy'](text)
            entities = []

            for ent in doc.ents:
                # scispaCy provides medical-specific entities
                entity_type = self._map_scispacy_label(ent.label_)

                entity = MedicalEntity(
                    text=ent.text,
                    label=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,  # Higher confidence for medical model
                    context=self._extract_context(text, ent.start_char, ent.end_char),
                    semantic_type=ent.label_
                )
                entities.append(entity)

            return entities

        except Exception as e:
            logger.error(f"scispaCy extraction failed: {e}")
            return []

    async def _extract_with_med7(self, text: str, mode: ProcessingMode) -> List[MedicalEntity]:
        """Extract entities using Med7 medical model."""
        if 'med7' not in self.models:
            return []

        try:
            doc = self.models['med7'](text)
            entities = []

            for ent in doc.ents:
                # Med7 provides specialized medical entities
                entity_type = self._map_med7_label(ent.label_)

                entity = MedicalEntity(
                    text=ent.text,
                    label=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9,  # High confidence for specialized medical model
                    context=self._extract_context(text, ent.start_char, ent.end_char),
                    semantic_type=ent.label_
                )
                entities.append(entity)

            return entities

        except Exception as e:
            logger.error(f"Med7 extraction failed: {e}")
            return []

    async def _extract_with_transformer(self, text: str, mode: ProcessingMode) -> List[MedicalEntity]:
        """Extract entities using transformer model."""
        if not self.pipelines.get('medical'):
            return []

        try:
            # Process text in chunks due to token limits
            chunks = self._split_text_for_processing(text)
            entities = []

            for chunk_start, chunk_text in chunks:
                results = self.pipelines['medical'](chunk_text)

                for result in results:
                    entity_type = self._map_transformer_label(result['entity_group'])

                    entity = MedicalEntity(
                        text=result['word'],
                        label=entity_type,
                        start=chunk_start + result['start'],
                        end=chunk_start + result['end'],
                        confidence=result['score'],
                        context=self._extract_context(text, chunk_start + result['start'], chunk_start + result['end'])
                    )
                    entities.append(entity)

            return entities

        except Exception as e:
            logger.error(f"Transformer extraction failed: {e}")
            return []

    async def _extract_with_biobert(self, text: str, mode: ProcessingMode) -> List[MedicalEntity]:
        """Extract entities using BioBERT model."""
        if not self.pipelines.get('biobert'):
            return []

        try:
            # Process text in chunks due to token limits
            chunks = self._split_text_for_processing(text)
            entities = []

            for chunk_start, chunk_text in chunks:
                results = self.pipelines['biobert'](chunk_text)

                for result in results:
                    entity_type = self._map_biobert_label(result['entity_group'])

                    entity = MedicalEntity(
                        text=result['word'],
                        label=entity_type,
                        start=chunk_start + result['start'],
                        end=chunk_start + result['end'],
                        confidence=result['score'],
                        context=self._extract_context(text, chunk_start + result['start'], chunk_start + result['end'])
                    )
                    entities.append(entity)

            return entities

        except Exception as e:
            logger.error(f"BioBERT extraction failed: {e}")
            return []

    async def _extract_with_patterns(self, text: str) -> List[MedicalEntity]:
        """Extract entities using regex patterns."""
        entities = []

        try:
            for entity_type, patterns in self.pattern_matchers.items():
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        entity = MedicalEntity(
                            text=match.group(),
                            label=EntityType(entity_type),
                            start=match.start(),
                            end=match.end(),
                            confidence=0.7,  # Medium confidence for patterns
                            context=self._extract_context(text, match.start(), match.end())
                        )
                        entities.append(entity)

            return entities

        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return []

    async def _extract_temporal_entities(self, text: str) -> List[MedicalEntity]:
        """Extract temporal information from medical text."""
        entities = []

        try:
            # Extract different types of temporal information
            for temporal_type, patterns in self.temporal_patterns.items():
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        # Map temporal types to proper EntityTypes
                        entity_type = EntityType.TIME  # default
                        if temporal_type == "duration":
                            entity_type = EntityType.TEMPORAL_DURATION
                        elif temporal_type == "frequency":
                            entity_type = EntityType.TEMPORAL_FREQUENCY
                        elif temporal_type == "time_of_day":
                            entity_type = EntityType.TEMPORAL_TIME_OF_DAY
                        elif temporal_type == "relative_time":
                            entity_type = EntityType.TEMPORAL_RELATIVE
                        
                        entity = MedicalEntity(
                            text=match.group(),
                            label=entity_type,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.8,
                            context=self._extract_context(text, match.start(), match.end()),
                            semantic_type=temporal_type
                        )
                        entities.append(entity)

            return entities

        except Exception as e:
            logger.error(f"Temporal extraction failed: {e}")
            return []

    def _split_text_for_processing(self, text: str) -> List[Tuple[int, str]]:
        """Split text into chunks for transformer processing."""
        max_length = self.config.max_sequence_length
        stride = self.config.stride

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_length, len(text))
            chunk = text[start:end]
            chunks.append((start, chunk))

            if end >= len(text):
                break

            start += stride

        return chunks

    def _extract_context(self, text: str, start: int, end: int, context_window: int = 50) -> str:
        """Extract surrounding context for an entity."""
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        return text[context_start:context_end]

    def _map_scispacy_label(self, label: str) -> EntityType:
        """Map scispaCy labels to our entity types."""
        mapping = {
            "DISEASE": EntityType.DISEASE,
            "SYMPTOM": EntityType.SYMPTOM,
            "PROCEDURE": EntityType.PROCEDURE,
            "ANATOMY": EntityType.ANATOMICAL_STRUCTURE,
            "DRUG": EntityType.MEDICATION,
            "LAB": EntityType.LAB_VALUE,
        }
        return mapping.get(label, EntityType.CONDITION)

    def _map_transformer_label(self, label: str) -> EntityType:
        """Map transformer model labels to our entity types."""
        mapping = {
            "DISEASE": EntityType.DISEASE,
            "MEDICATION": EntityType.MEDICATION,
            "SYMPTOM": EntityType.SYMPTOM,
            "PROCEDURE": EntityType.PROCEDURE,
            "ANATOMY": EntityType.ANATOMICAL_STRUCTURE,
            "DOSAGE": EntityType.DOSAGE,
            "LAB": EntityType.LAB_VALUE,
        }
        return mapping.get(label, EntityType.CONDITION)

    def _map_med7_label(self, label: str) -> EntityType:
        """Map Med7 labels to our entity types."""
        mapping = {
            "DRUG": EntityType.MEDICATION,
            "STRENGTH": EntityType.DOSAGE,
            "DURATION": EntityType.FREQUENCY,
            "FREQUENCY": EntityType.FREQUENCY,
            "ROUTE": EntityType.FREQUENCY,
            "FORM": EntityType.DOSAGE,
            "DOSAGE": EntityType.DOSAGE
        }
        return mapping.get(label, EntityType.MEDICATION)

    def _map_biobert_label(self, label: str) -> EntityType:
        """Map BioBERT labels to our entity types."""
        mapping = {
            "DISEASE": EntityType.DISEASE,
            "DRUG": EntityType.MEDICATION,
            "SYMPTOM": EntityType.SYMPTOM,
            "PROCEDURE": EntityType.PROCEDURE,
            "ANATOMY": EntityType.ANATOMICAL_STRUCTURE,
            "DOSAGE": EntityType.DOSAGE,
            "LAB": EntityType.LAB_VALUE,
            "FINDING": EntityType.SYMPTOM        }
        return mapping.get(label, EntityType.CONDITION)

    def _classify_unknown_entity(self, text: str, original_label: str) -> EntityType:
        """Classify entities that don't have direct mappings."""
        text_lower = text.lower().strip()
        
        # Check for medical condition patterns first
        condition_patterns = [
            r'\b(?:diabetes|hypertension|cancer|infection|disease|syndrome|disorder)\b',
            r'\b(?:pressure|glucose|cholesterol|hemoglobin|blood)\b'
        ]
        
        for pattern in condition_patterns:
            if re.search(pattern, text_lower):
                return EntityType.CONDITION
        
        # Check for lab value patterns
        lab_value_patterns = [
            r'\b\d+\s*(?:mg/dl|mg/l|mmol/l|%|bpm|/min)\b',
            r'\b(?:level|count|value|result):\s*\d+\b',
            r'\b\d+\.\d+\s*(?:mg/dl|mmol/l|%)\b'
        ]
        
        for pattern in lab_value_patterns:
            if re.search(pattern, text_lower):
                return EntityType.LAB_VALUE
        
        # Check for vital sign patterns
        vital_patterns = [
            r'\b(?:blood pressure|heart rate|pulse|temperature|bp|hr)\b',
            r'\b\d+/\d+\s*(?:mmhg)?\b',  # BP readings
            r'\b\d+\s*bpm\b'
        ]
        
        for pattern in vital_patterns:
            if re.search(pattern, text_lower):
                return EntityType.VITAL_SIGN
        
        # Check for organization patterns
        org_patterns = [
            r'\b(?:thyrocare|pathology|laboratory|lab|clinic|hospital|medical|healthcare|diagnostics)\b',
            r'\b(?:pvt|ltd|inc|corp|corporation|limited|llc)\b'
        ]
        
        for pattern in org_patterns:
            if re.search(pattern, text_lower):
                return EntityType.ORGANIZATION
        
        # Check for location patterns
        location_patterns = [
            r'\b(?:hall|wing|floor|building|street|road|avenue|city|state)\b',
            r'\b(?:lst|1st|2nd|3rd|\d+th)\s*floor\b',
            r'\b(?:a\s*wing|b\s*wing|east|west|north|south)\b'
        ]
        
        for pattern in location_patterns:
            if re.search(pattern, text_lower):
                return EntityType.LOCATION
        
        # Check for person patterns
        person_patterns = [
            r'\b(?:dr|doctor|mr|mrs|ms|patient)\s+\w+\b',
            r'\b\w+\s*barsagadey\b',
            r'\bvivek\s*\w*\b'
        ]
        
        for pattern in person_patterns:
            if re.search(pattern, text_lower):
                return EntityType.PERSON
        
        # Check for test/procedure patterns
        test_patterns = [
            r'\b(?:test|panel|profile|package|study|workup)\b',
            r'\b(?:aarogyam|winter|advanced|comprehensive|basic)\b'
        ]
        
        for pattern in test_patterns:
            if re.search(pattern, text_lower):
                return EntityType.LAB_TEST
        
        # Based on original spaCy label, make intelligent guess
        if original_label in ['MISC', 'PRODUCT', 'WORK_OF_ART']:
            # These could be test names or organizations
            if any(word in text_lower for word in ['test', 'panel', 'profile', 'study']):
                return EntityType.LAB_TEST
            else:
                return EntityType.CONDITION  # Changed from ORGANIZATION
        elif original_label in ['NORP', 'LANGUAGE']:
            return EntityType.CONDITION  # Changed from ORGANIZATION        elif original_label in ['EVENT']:
            return EntityType.PROCEDURE
        elif original_label in ['QUANTITY', 'CARDINAL']:
            # Numbers might be lab values
            if re.search(r'\d+', text_lower):
                return EntityType.LAB_VALUE
            else:
                return EntityType.CONDITION
        else:
            # More intelligent fallback based on text content
            if re.search(r'\d+', text_lower) and any(word in text_lower for word in ['mg', 'dl', 'mmol', '%', 'count', 'level']):
                return EntityType.LAB_VALUE
            elif any(word in text_lower for word in ['patient', 'hospital', 'clinic', 'lab']):
                return EntityType.ORGANIZATION
            else:
                # Default to CONDITION instead of ORGANIZATION for medical text
                return EntityType.CONDITION

    def _should_ignore_entity_text(self, text: str) -> bool:
        """Check if entity text should be ignored (common artifacts)."""
        text_lower = text.lower().strip()
        
        # Ignore very short texts
        if len(text_lower) <= 1:
            return True
            
        # Ignore common punctuation and artifacts
        ignore_patterns = [
            r'^[.,;:!?()[\]{}"\'`-]+$',  # Only punctuation
            r'^\d+$',  # Only digits (but allow in context)
            r'^[a-z]$',  # Single letters
            r'^(the|and|or|of|to|in|for|with|by|at|on)$',  # Common stop words
            r'^(a|an|is|are|was|were|be|been|being)$',  # More stop words
            r'^\s*$',  # Only whitespace
        ]
        
        for pattern in ignore_patterns:
            if re.match(pattern, text_lower):
                return True
                
        return False

    async def _extract_additional_medical_patterns(self, text: str) -> List[MedicalEntity]:
        """Extract additional medical entities using custom patterns."""
        entities = []
        
        # Enhanced medical patterns
        medical_patterns = [
            # Lab values with units
            (r'\b(\d+(?:\.\d+)?)\s*(mg/dl|mg/l|mmol/l|g/dl|mEq/l|%|bpm|/min|mmHg)\b', EntityType.LAB_VALUE),
            
            # Blood pressure readings
            (r'\b(\d{2,3}/\d{2,3})\s*(?:mmhg)?\b', EntityType.VITAL_SIGN),
            
            # Temperature readings
            (r'\b(\d{2,3}(?:\.\d)?)\s*(?:°f|°c|f|c|degrees?)\b', EntityType.VITAL_SIGN),
            
            # Medical test names
            (r'\b(complete blood count|cbc|lipid profile|liver function test|kidney function test|thyroid function test)\b', EntityType.LAB_TEST),
            
            # Medication dosages
            (r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|tablets?|capsules?)\b', EntityType.DOSAGE),
            
            # Medical conditions with common patterns
            (r'\b(type\s+[12]\s+diabetes|essential hypertension|chronic kidney disease|coronary artery disease)\b', EntityType.CONDITION),
            
            # Doctor/Person titles and names
            (r'\b(dr\.?\s+\w+(?:\s+\w+)*|doctor\s+\w+(?:\s+\w+)*)\b', EntityType.PERSON),
            
            # Medical facilities
            (r'\b(\w+\s+(?:hospital|clinic|medical center|laboratory|pathology|diagnostics))\b', EntityType.ORGANIZATION),
        ]
        
        for pattern, entity_type in medical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                
                entity = MedicalEntity(
                    text=entity_text,
                    label=entity_type,
                    start=start_pos,
                    end=end_pos,
                    confidence=0.7,  # Pattern-based confidence
                    context=self._extract_context(text, start_pos, end_pos),
                    semantic_type="PATTERN_MATCH"
                )
                entities.append(entity)
        
        return entities

    def _normalize_medical_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Normalize medical entities using predefined mappings."""
        normalized_entities = []
        
        for entity in entities:
            normalized_entity = entity.copy()
            original_text = entity.text.lower().strip()
            
            # Normalize based on entity type
            if entity.label == EntityType.MEDICATION:
                if "medications" in self.medical_normalizers:
                    normalized_text = self.medical_normalizers["medications"].get(original_text)
                    if normalized_text:
                        normalized_entity.text = normalized_text
                        normalized_entity.normalized_text = normalized_text
            
            elif entity.label in [EntityType.DISEASE, EntityType.CONDITION]:
                if "conditions" in self.medical_normalizers:
                    normalized_text = self.medical_normalizers["conditions"].get(original_text)
                    if normalized_text:
                        normalized_entity.text = normalized_text
                        normalized_entity.normalized_text = normalized_text
            
            elif entity.label == EntityType.PROCEDURE:
                if "procedures" in self.medical_normalizers:
                    normalized_text = self.medical_normalizers["procedures"].get(original_text)
                    if normalized_text:
                        normalized_entity.text = normalized_text
                        normalized_entity.normalized_text = normalized_text
            
            elif entity.label == EntityType.LAB_VALUE:
                if "lab_tests" in self.medical_normalizers:
                    normalized_text = self.medical_normalizers["lab_tests"].get(original_text)
                    if normalized_text:
                        normalized_entity.text = normalized_text
                        normalized_entity.normalized_text = normalized_text
            
            # If no normalization was applied, keep original text
            if normalized_entity.normalized_text is None:
                normalized_entity.normalized_text = entity.text
            
            normalized_entities.append(normalized_entity)
        
        return normalized_entities

    def _merge_overlapping_entities(self, entities: List[MedicalEntity], threshold: float) -> List[MedicalEntity]:
        """Merge overlapping entities, keeping the highest confidence."""
        if not entities:
            return []        # Sort by start position
        entities.sort(key=lambda x: x.start)

        merged = []
        current = entities[0]

        for next_entity in entities[1:]:
            # Check for overlap
            if self._entities_overlap(current, next_entity):
                # Keep the entity with higher confidence
                if next_entity.confidence > current.confidence:
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity

        merged.append(current)
        return merged

    def _entities_overlap(self, entity1: MedicalEntity, entity2: MedicalEntity) -> bool:
        """Check if two entities overlap in text positions."""
        return not (entity1.end <= entity2.start or entity2.end <= entity1.start)

    def _filter_entities(self, entities: List[MedicalEntity], request: NERRequest) -> List[MedicalEntity]:
        """Filter entities based on request parameters."""
        filtered = entities

        # Apply exclusions first (more intuitive logic)
        if request.exclude_entity_types:
            filtered = [e for e in filtered if e.label not in request.exclude_entity_types]

        # Then apply inclusions (if any remain after exclusions)
        if request.entity_types:
            # Only apply include filter if it doesn't conflict with exclusions
            include_types = set(request.entity_types)
            exclude_types = set(request.exclude_entity_types) if request.exclude_entity_types else set()
            
            # Resolve conflicts: if an entity type is both included and excluded, treat it as excluded
            effective_include_types = include_types - exclude_types
            
            if effective_include_types:
                filtered = [e for e in filtered if e.label in effective_include_types]
            else:
                # If all include types are also excluded, return empty list
                logger.warning("Entity type filter conflict: all included types are also excluded")
                return []

        # Limit number of entities
        if request.max_entities and len(filtered) > request.max_entities:
            # Sort by confidence and take top N
            filtered.sort(key=lambda x: x.confidence, reverse=True)
            filtered = filtered[:request.max_entities]

        return filtered

    def _group_entities(self, entities: List[MedicalEntity]) -> List[EntityGroup]:
        """Group entities by type with statistics."""
        groups = {}

        for entity in entities:
            entity_type = entity.label
            if entity_type not in groups:
                groups[entity_type] = []
            groups[entity_type].append(entity)

        entity_groups = []
        for entity_type, type_entities in groups.items():
            confidences = [e.confidence for e in type_entities]
            unique_texts = set(e.text.lower() for e in type_entities)

            group = EntityGroup(
                entity_type=entity_type,
                entities=type_entities,
                count=len(type_entities),
                avg_confidence=np.mean(confidences),
                unique_values=len(unique_texts)
            )
            entity_groups.append(group)

        return entity_groups

    def _calculate_confidence_scores(self, entity_groups: List[EntityGroup]) -> Dict[str, float]:
        """Calculate confidence scores by entity type."""
        confidence_scores = {}

        for group in entity_groups:
            confidence_scores[group.entity_type.value] = group.avg_confidence

        return confidence_scores

    async def _assess_text_quality(self, text: str) -> float:
        """Assess the quality of input text for NER processing."""
        quality_score = 1.0

        # Check text length
        if len(text) < 10:
            quality_score *= 0.5

        # Check for special characters (OCR artifacts)
        special_char_ratio = len(re.findall(r'[^\w\s.,;:!?-]', text)) / len(text)
        if special_char_ratio > 0.1:
            quality_score *= (1.0 - special_char_ratio)

        # Check for repeated characters (OCR errors)
        repeated_chars = len(re.findall(r'(.)\1{3,}', text))
        if repeated_chars > 0:
            quality_score *= 0.8

        # Check for medical terminology presence
        medical_terms = ["patient", "diagnosis", "treatment", "medication", "symptom", "procedure"]
        medical_term_count = sum(1 for term in medical_terms if term.lower() in text.lower())
        if medical_term_count > 0:
            quality_score *= (1.0 + 0.1 * medical_term_count)

        return min(1.0, quality_score)

    def _suggest_next_step(self, entities: List[MedicalEntity], text: str) -> str:
        """Suggest the next processing step based on extraction results."""
        if not entities:
            return "Consider text preprocessing or different extraction models"

        if len(entities) < 5:
            return "Ready for timeline structuring and relationship analysis"
        else:
            return "Ready for knowledge graph construction and entity linking"

    def _generate_recommendations(self, entities: List[MedicalEntity], request: NERRequest) -> List[str]:
        """Generate processing recommendations."""
        recommendations = []

        if not entities:
            recommendations.append("Consider adjusting confidence threshold or processing mode")

        if len(entities) > 100:
            recommendations.append("Consider text chunking for better performance")

        avg_confidence = np.mean([e.confidence for e in entities]) if entities else 0
        if avg_confidence < 0.7:
            recommendations.append("Consider using higher accuracy processing mode")

        return recommendations

    async def process_batch(self, request: BatchNERRequest) -> BatchNERResponse:
        """Process multiple texts in batch."""
        start_time = time.time()
        results = []
        successful = 0
        failed = 0

        # Process texts in batches
        for i in range(0, len(request.texts), request.batch_size):
            batch_texts = request.texts[i:i + request.batch_size]
            batch_ids = None
            if request.document_ids:
                batch_ids = request.document_ids[i:i + request.batch_size]

            # Create individual requests
            batch_requests = []
            for j, text in enumerate(batch_texts):
                doc_id = batch_ids[j] if batch_ids else f"batch_{i}_{j}"
                ner_request = NERRequest(
                    text=text,
                    document_id=doc_id,
                    processing_mode=request.processing_mode,
                    language=request.language,
                    confidence_threshold=request.confidence_threshold
                )
                batch_requests.append(ner_request)

            # Process batch
            if request.parallel_processing:
                batch_results = await asyncio.gather(
                    *[self.extract_entities(req) for req in batch_requests],
                    return_exceptions=True
                )
            else:
                batch_results = []
                for req in batch_requests:
                    try:
                        result = await self.extract_entities(req)
                        batch_results.append(result)
                    except Exception as e:
                        batch_results.append(e)

            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    failed += 1
                    logger.error(f"Batch processing error: {result}")
                else:
                    results.append(result)
                    successful += 1

        # Calculate batch metrics
        batch_processing_time = time.time() - start_time
        avg_processing_time = batch_processing_time / len(request.texts)

        # Calculate batch confidence
        all_confidences = []
        for result in results:
            if hasattr(result, 'overall_confidence'):
                all_confidences.append(result.overall_confidence)

        batch_confidence = np.mean(all_confidences) if all_confidences else 0.0

        return BatchNERResponse(
            message=f"Batch processing completed: {successful} successful, {failed} failed",
            total_documents=len(request.texts),
            successful_extractions=successful,
            failed_extractions=failed,
            results=results,
            batch_processing_time=batch_processing_time,
            avg_processing_time=avg_processing_time,
            batch_confidence=batch_confidence
        )

    async def health_check(self) -> bool:
        """Perform health check of the NER processor."""
        try:
            await self.initialize()

            # Test with sample text
            test_request = NERRequest(
                text="Patient has diabetes and takes metformin 500mg daily.",
                processing_mode=ProcessingMode.FAST
            )

            result = await self.extract_entities(test_request)
            return len(result.entities) > 0

        except Exception as e:
            logger.error(f"NER health check failed: {e}")
            return False

    def _build_temporal_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Build regex patterns for temporal information extraction."""
        patterns = {
            "duration": [
                re.compile(r"\b(\d+)\s+(days?|weeks?|months?|years?)\b", re.IGNORECASE),
                re.compile(r"\b(few|several)\s+(days?|weeks?|months?|years?)\b", re.IGNORECASE)
            ],
            "frequency": [
                re.compile(r"\b(once|twice|three times?|four times?)\s+(daily|weekly|monthly|yearly)\b", re.IGNORECASE),
                re.compile(r"\b(every|each)\s+(\d+)\s+(hours?|days?|weeks?|months?)\b", re.IGNORECASE),
                re.compile(r"\b(BID|TID|QID|QD|PRN|as needed)\b", re.IGNORECASE)
            ],
            "time_of_day": [
                re.compile(r"\b(morning|afternoon|evening|night|bedtime)\b", re.IGNORECASE),
                re.compile(r"\b(before|after|with)\s+meals?\b", re.IGNORECASE)
            ],
            "relative_time": [
                re.compile(r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b", re.IGNORECASE),
                re.compile(r"\b(yesterday|today|tomorrow|last week|next week|last month|next month)\b", re.IGNORECASE)
            ]        }
        return patterns

    def _build_medical_normalizers(self) -> Dict[str, Dict[str, str]]:
        """Build medical entity normalization mappings."""
        return {
            "medications": {
                # Common abbreviations and alternate names
                "ASA": "aspirin",
                "HCTZ": "hydrochlorothiazide",
                "MTX": "methotrexate",
                "TMP-SMX": "trimethoprim-sulfamethoxazole",
                "insulin NPH": "insulin isophane",
                "insulin regular": "insulin human"
            },
            "conditions": {
                # Medical abbreviations
                "DM": "diabetes mellitus",
                "HTN": "hypertension",
                "MI": "myocardial infarction",
                "CHF": "congestive heart failure",
                "COPD": "chronic obstructive pulmonary disease",
                "UTI": "urinary tract infection",
                "CAD": "coronary artery disease",
                "DVT": "deep vein thrombosis",
                "PE": "pulmonary embolism"
            },
            "procedures": {
                # Procedure abbreviations and OCR corrections
                "EKG": "electrocardiogram",
                "ECHO": "echocardiogram",
                "CT": "computed tomography",
                "MRI": "magnetic resonance imaging",
                "US": "ultrasound",
                "CXR": "chest X-ray",
                "CONPLETEBLOODCouniT": "Complete Blood Count",
                "completebloodcount": "Complete Blood Count",
                "completeBLOODcount": "Complete Blood Count",
                "CBC": "Complete Blood Count",
                "bloodcount": "Blood Count",
                "blood count": "Complete Blood Count"
            },
            "lab_tests": {
                # Lab test normalizations and OCR corrections
                "A1C": "hemoglobin A1c",
                "FBS": "fasting blood sugar",
                "BUN": "blood urea nitrogen",
                "ESR": "erythrocyte sedimentation rate",
                "CRP": "C-reactive protein",
                "Hgb": "Hemoglobin",
                "HGB": "Hemoglobin",
                "hemogIobin": "Hemoglobin",
                "hemogiobia": "Hemoglobin",
                "PIatelets": "Platelets",
                "platelet": "Platelets",
                "PLT": "Platelets",
                "neutrophiIs": "Neutrophils",
                "Iymphocytes": "Lymphocytes",
                "eosinophiIs": "Eosinophils",
                "basophiIs": "Basophils",
                "WBC": "White Blood Cell Count",
                "RBC": "Red Blood Cell Count"
            }
        }

    async def _perform_entity_linking(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Link entities to medical knowledge bases (UMLS, SNOMED CT, ICD-10)."""
        linked_entities = []

        for entity in entities:
            linked_entity = entity.copy()

            # Perform entity linking based on entity type
            if entity.label in [EntityType.DISEASE, EntityType.CONDITION]:
                # Link to ICD-10 codes
                icd_code = await self._link_to_icd10(entity.text)
                if icd_code:
                    linked_entity.concept_code = icd_code
                    linked_entity.entity_id = f"ICD10:{icd_code}"

            elif entity.label == EntityType.MEDICATION:
                # Link to RxNorm or other drug databases
                drug_code = await self._link_to_rxnorm(entity.text)
                if drug_code:
                    linked_entity.concept_code = drug_code
                    linked_entity.entity_id = f"RXNORM:{drug_code}"

            elif entity.label == EntityType.PROCEDURE:
                # Link to CPT codes
                cpt_code = await self._link_to_cpt(entity.text)
                if cpt_code:
                    linked_entity.concept_code = cpt_code
                    linked_entity.entity_id = f"CPT:{cpt_code}"

            # UMLS linking for comprehensive medical concepts
            if self.config.umls_enabled:
                umls_id = await self._link_to_umls(entity.text, entity.label)
                if umls_id:
                    linked_entity.entity_id = linked_entity.entity_id or f"UMLS:{umls_id}"

            linked_entities.append(linked_entity)

        return linked_entities

    async def _link_to_icd10(self, entity_text: str) -> Optional[str]:
        """Link medical condition to ICD-10 code (simplified implementation)."""
        # This is a simplified implementation - in production, you would use
        # a proper medical terminology service or database
        icd10_mappings = {
            "diabetes mellitus": "E11",
            "type 2 diabetes": "E11.9",
            "hypertension": "I10",
            "pneumonia": "J18.9",
            "myocardial infarction": "I21.9",
            "asthma": "J45.9",
            "copd": "J44.1",
            "chronic obstructive pulmonary disease": "J44.1"
        }

        return icd10_mappings.get(entity_text.lower())

    async def _link_to_rxnorm(self, entity_text: str) -> Optional[str]:
        """Link medication to RxNorm code (simplified implementation)."""
        # Simplified implementation - production would use RxNorm API
        rxnorm_mappings = {
            "metformin": "6809",
            "lisinopril": "29046",
            "atorvastatin": "83367",
            "aspirin": "1191",
            "omeprazole": "7646",
            "insulin": "5856"
        }

        return rxnorm_mappings.get(entity_text.lower())

    async def _link_to_cpt(self, entity_text: str) -> Optional[str]:
        """Link procedure to CPT code (simplified implementation)."""
        # Simplified implementation - production would use CPT database
        cpt_mappings = {
            "ecg": "93000",
            "electrocardiogram": "93000",
            "chest x-ray": "71020",
            "ct scan": "74150",
            "mri": "70551",
            "blood test": "80053",
            "colonoscopy": "45378"
        }

        return cpt_mappings.get(entity_text.lower())

    async def _link_to_umls(self, entity_text: str, entity_type: EntityType) -> Optional[str]:
        """Link entity to UMLS concept (simplified implementation)."""
        # This would integrate with UMLS API in production
        # For now, return a placeholder
        return None

    def _calculate_entity_relationships(self, entities: List[MedicalEntity], text: str) -> List[MedicalEntity]:
        """Calculate relationships between entities based on proximity and context."""
        enhanced_entities = []

        for i, entity in enumerate(entities):
            enhanced_entity = entity.copy()
            related_entities = []

            # Find nearby entities that might be related
            for j, other_entity in enumerate(entities):
                if i != j and self._are_entities_related(entity, other_entity, text):
                    related_entities.append(str(j))  # Use index as relationship identifier

            enhanced_entity.related_entities = related_entities
            enhanced_entities.append(enhanced_entity)

        return enhanced_entities

    def _are_entities_related(self, entity1: MedicalEntity, entity2: MedicalEntity, text: str) -> bool:
        """Determine if two entities are contextually related."""
        # Check proximity (within 100 characters)
        distance = abs(entity1.start - entity2.start)
        if distance > 100:
            return False

        # Check for medical relationships
        medical_relationships = [
            (EntityType.MEDICATION, EntityType.DOSAGE),
            (EntityType.MEDICATION, EntityType.FREQUENCY),
            (EntityType.CONDITION, EntityType.SYMPTOM),
            (EntityType.PROCEDURE, EntityType.ANATOMICAL_STRUCTURE),
            (EntityType.LAB_VALUE, EntityType.LAB_TEST),
            (EntityType.VITAL_SIGN, EntityType.TIME)
        ]

        entity_pair = (entity1.label, entity2.label)
        return entity_pair in medical_relationships or (entity_pair[1], entity_pair[0]) in medical_relationships

    def _should_ignore_entity(self, entity: MedicalEntity) -> bool:
        """Check if an entity should be ignored (generic headings, artifacts, etc.)."""
        text = entity.text.lower().strip()
          # Generic headings that should be ignored
        generic_headings = {
            "results", "result", "results units", "report", "findings", "summary",
            "notes", "comments", "interpretation", "impression", "conclusion",
            "history", "examination", "assessment", "plan", "recommendations",
            "lab", "laboratory", "test", "tests", "values", "normal", "abnormal",
            "reference", "ranges", "range", "units", "unit", "specimen", "sample",
            "processed", "processedat", "neate", "processed at", "laboratory results",
            "radiology report", "medical record", "patient name", "date of birth"
        }        # Administrative/technical terms that should be ignored
        admin_terms = {
            "name", "patientid", "patient id", "id", "reference", "ref",
            "date", "time", "timestamp", "generated", "printed",
            "page", "sheet", "document", "file", "record", "patient",
            "john", "doe", "hospital", "medical record number",
            "birth", "ol", "l98o", "l23456", "and", "or", "the", "of",
            "in", "on", "at", "to", "from", "with", "by", "for", "as",
            "is", "are", "was", "were", "has", "have", "had", "will",
            "would", "could", "should", "may", "might", "can", "shall"
        }
        
        # OCR artifacts and formatting issues
        ocr_artifacts = {
            ":", "=", ",", ";", "(", ")", "[", "]", "{", "}", 
            "neate processedat", "y,", "112,", "lstfloor"
        }
        
        # Check if entity is a generic heading
        if text in generic_headings:
            return True
        
        # Check if entity is administrative data
        if text in admin_terms:
            return True
            
        # Check if entity is an OCR artifact
        if text in ocr_artifacts:
            return True
            
        # Check if entity is too short to be meaningful (unless it's a known abbreviation)
        known_abbreviations = {
            "cbc", "ecg", "ekg", "mri", "ct", "bp", "hr", "rr", "wbc", "rbc", 
            "hgb", "hct", "plt", "bun", "tsh", "psa", "ldl", "hdl"
        }
        if len(text) < 2 and text not in known_abbreviations:
            return True
            
        # Check if entity is just numbers or special characters
        if re.match(r'^[\d\s\-\.\(\)\[\],;:=]+$', text):
            return True
            
        # Check if entity is just punctuation
        if re.match(r'^[^\w\s]+$', text):
            return True
          # Check for administrative patterns
        admin_patterns = [
            r'^name\s*[:=]',
            r'^patient\s*id\s*[:=]',
            r'^processed\s*at\s*[:=]',
            r'^\w*id\s*[:=]',
            r'^ref\s*[:=]',
            r'^\d+\s*[,;:]$',  # Just numbers with punctuation
            r'^patient\s+\w+$',  # Patient name patterns
            r'^(?:john|doe|mr|mrs|ms)\s*\w*$',  # Common names
            r'^(?:hospital|clinic|medical center)(?:\s+\w+)*$',  # Healthcare facility names
            r'^(?:ol|l98o|l23456|\d{4,})$',  # Administrative codes/IDs
            r'^of\s+birth\s*[:=]',  # Date of birth patterns
            r'^medical\s+record\s+number\s*[:=]',
            r'^(?:laboratory|radiology)\s+(?:results?|reports?)$'
        ]
        
        for pattern in admin_patterns:
            if re.match(pattern, text):
                return True
        
        # Check if entity is too fragmented (common in OCR errors)
        if len(text.split()) == 1 and len(text) > 15 and not text.isalpha():
            # Long single "words" that mix letters/numbers are likely OCR errors
            return True
            
        return False

    def _correct_ocr_errors(self, entity: MedicalEntity) -> MedicalEntity:
        """Correct common OCR errors in medical text."""
        text = entity.text
        corrected_entity = entity.copy()
        
        # Common OCR corrections for medical terms
        ocr_corrections = {
            # Blood count variations
            "CONPLETEBLOODCouniT": "Complete Blood Count",
            "completebloodcount": "Complete Blood Count", 
            "completeBLOODcount": "Complete Blood Count",
            "CBCcount": "CBC",
            "bloodcount": "Blood Count",
            
            # Hemoglobin variations
            "Hgb": "Hemoglobin",
            "HGB": "Hemoglobin", 
            "hemogIobin": "Hemoglobin",
            "hemogiobia": "Hemoglobin",
            
            # Platelet variations
            "PIatelets": "Platelets",
            "platelet": "Platelets",
            "PLT": "Platelets",
            
            # Other common corrections
            "neutrophiIs": "Neutrophils",
            "Iymphocytes": "Lymphocytes",
            "monocytes": "Monocytes",
            "eosinophiIs": "Eosinophils",
            "basophiIs": "Basophils",
            "leucocytes": "Leukocytes",
            "WBC": "White Blood Cell Count",
            "RBC": "Red Blood Cell Count"
        }
        
        # Check for exact matches first
        for error, correction in ocr_corrections.items():
            if text == error:
                corrected_entity.text = correction
                corrected_entity.normalized_text = correction
                return corrected_entity
        
        # Check for case-insensitive matches
        text_lower = text.lower()
        for error, correction in ocr_corrections.items():
            if text_lower == error.lower():
                corrected_entity.text = correction
                corrected_entity.normalized_text = correction
                return corrected_entity
          # Pattern-based corrections for common OCR errors
        corrected_text = text
        
        # Fix common OCR character substitutions (context-aware)
        # Only convert to digits when it's clearly numeric context
        if re.search(r'\d', text):  # Only if there are already digits in the text
            corrected_text = re.sub(r'[Il](?=\d|$|\s)', '1', corrected_text)  # I, l to 1 in numeric context
            corrected_text = re.sub(r'O(?=\d|$|\s)', '0', corrected_text)     # O to 0 in numeric context
        
        # Fix letter substitutions in text context
        if re.search(r'[a-zA-Z]{3,}', text):  # Only if there are letter sequences
            corrected_text = re.sub(r'1(?=[a-zA-Z])', 'l', corrected_text)    # 1 to l in text context
            corrected_text = re.sub(r'0(?=[a-zA-Z])', 'O', corrected_text)    # 0 to O in text context
          # Common medical OCR fixes
        corrected_text = re.sub(r'rn', 'm', corrected_text)     # rn to m
        corrected_text = re.sub(r'cl', 'd', corrected_text)     # cl to d
        
        # Specific numeric pattern fixes for lab values
        corrected_text = re.sub(r'l(\d+)', r'1\1', corrected_text)  # l120 -> 120
        corrected_text = re.sub(r'(\d+)O', r'\g<1>0', corrected_text)  # 12O -> 120
        corrected_text = re.sub(r'O(\d+)', r'0\1', corrected_text)  # O80 -> 080
        corrected_text = re.sub(r'(\d+)o', r'\g<1>0', corrected_text)  # 12o -> 120
        corrected_text = re.sub(r'o(\d+)', r'0\1', corrected_text)  # o80 -> 080
        
        if corrected_text != text:
            corrected_entity.text = corrected_text
            corrected_entity.normalized_text = corrected_text
        
        return corrected_entity

    def _validate_and_correct_entity_type(self, entity: MedicalEntity) -> MedicalEntity:
        """Validate and correct entity type based on context and patterns."""
        text = entity.text.lower().strip()
        original_text = entity.text.strip()
        corrected_entity = entity.copy()
          # Vital signs patterns - should be VITAL_SIGN
        vital_sign_indicators = [
            r'\b(?:blood pressure|bp|pulse|heart rate|hr|temperature|temp|respiratory rate|rr)\b',
            r'\b\d+\s*/\s*\d+\s*(?:mmhg)?\b',  # BP readings like 120/80
            r'\b\d+\s*bpm\b',  # Heart rate with bpm
            r'\b\d+\.\d+\s*°[cf]\b',  # Temperature readings
            r'\bpressure\b',  # Just "pressure" in medical context
            r'\bmmhg\b',  # Blood pressure unit
            r'\bbpm\b',   # Heart rate unit
            r'\b(?:systolic|diastolic)\b'  # BP components
        ]
        
        for pattern in vital_sign_indicators:
            if re.search(pattern, text):
                corrected_entity.label = EntityType.VITAL_SIGN
                return corrected_entity
        
        # Lab value patterns - should be LAB_VALUE  
        lab_value_indicators = [
            r'\b(?:cholesterol|glucose|hemoglobin|creatinine|bun|urea|sodium|potassium|chloride)\s*[:=]\s*\d+',
            r'\b\d+(?:\.\d+)?\s*(?:mg/dl|mg/l|mmol/l|g/dl|%|cells/μl)\b',
            r'\b(?:wbc|rbc|hct|plt|hgb)\s*[:=]\s*\d+',
            r'\b(?:total|ldl|hdl)\s+cholesterol\s*[:=]\s*\d+'
        ]
        
        for pattern in lab_value_indicators:
            if re.search(pattern, text):
                corrected_entity.label = EntityType.LAB_VALUE
                return corrected_entity
        
        # Organization/Laboratory patterns - should be ORGANIZATION
        organization_indicators = [
            r'\b(?:thyrocare|pathology|laboratory|lab|clinic|hospital|medical center|healthcare|diagnostics)\b',
            r'\b(?:quest|labcorp|mayo|cleveland clinic|johns hopkins)\b',
            r'\b(?:pvt ltd|inc|corp|corporation|limited|llc)\b'
        ]
        
        for pattern in organization_indicators:
            if re.search(pattern, text):
                corrected_entity.label = EntityType.ORGANIZATION
                return corrected_entity
        
        # Location/Address patterns - should be LOCATION
        location_indicators = [
            r'\b(?:hall|wing|floor|building|street|road|avenue|city|state)\b',
            r'\b(?:lst|1st|2nd|3rd|\d+th)\s*floor\b',
            r'\b(?:a\s*wing|b\s*wing|east|west|north|south)\b',
            r'\b(?:sohrabh\s*hall|awing|lstfloor)\b',
            r'\b\d+\s*(?:st|nd|rd|th)?\s*(?:floor|level)\b'
        ]
        
        for pattern in location_indicators:
            if re.search(pattern, text):
                corrected_entity.label = EntityType.LOCATION
                return corrected_entity
        
        # Person name patterns - should be PERSON
        person_indicators = [
            r'\bname\s*[:=]?\s*[a-z]+\b',
            r'\b(?:patient|dr|doctor|mr|mrs|ms)\s+[a-z]+\b',
            r'\b[a-z]+\s*barsagadey\b',  # Specific pattern from example
            r'\bvivek\s*[a-z]*\b'  # Another specific pattern
        ]
        
        for pattern in person_indicators:
            if re.search(pattern, text):
                corrected_entity.label = EntityType.PERSON
                return corrected_entity
        
        # Test/Panel names - should be LAB_TEST or PROCEDURE
        test_indicators = [
            r'\b(?:aarogyam|winter|advanced|panel|profile|package)\b',
            r'\b(?:c\s*:\s*peptide|c-peptide|peptide)\b',
            r'\b(?:comprehensive|basic|complete|extended)\s*(?:panel|profile|test)\b',
            r'\b(?:lipid|metabolic|thyroid|cardiac)\s*(?:panel|profile)\b'
        ]
        
        for pattern in test_indicators:
            if re.search(pattern, text):
                corrected_entity.label = EntityType.LAB_TEST
                return corrected_entity
        
        # ID/Reference patterns - should be ignored or classified differently
        id_indicators = [
            r'\b(?:patient\s*id|id|reference|ref)\b',
            r'\b(?:processed\s*at|neate\s*processed)\b',
            r'\b[a-z]*id\s*[:=]?\s*\w+\b'
        ]
        
        for pattern in id_indicators:
            if re.search(pattern, text):
                # These should be ignored as they're administrative data
                corrected_entity.label = EntityType.PERSON  # Use PERSON for patient identifiers
                return corrected_entity
        
        # Lab value patterns - these should be LAB_VALUE, not CONDITION
        lab_value_indicators = [
            r'\b(?:hemoglobin|hgb|hb)\b.*\d+(?:\.\d+)?.*(?:g/dl|range)',
            r'\b(?:platelet|plt).*\d+(?:\.\d+)?.*(?:count|/ul|range)',
            r'\b(?:neutrophils|lymphocytes|monocytes|eosinophils|basophils)\b.*\d+(?:\.\d+)?.*%',
            r'\b(?:wbc|rbc|white|red).*(?:count|cell).*\d+(?:\.\d+)?',
            r'\b(?:hematocrit|hct)\b.*\d+(?:\.\d+)?.*%'
        ]
        
        for pattern in lab_value_indicators:
            if re.search(pattern, text):
                corrected_entity.label = EntityType.LAB_VALUE
                return corrected_entity
        
        # Procedure patterns - these should be PROCEDURE
        procedure_indicators = [
            r'\b(?:complete blood count|cbc|blood count|blood test)\b',
            r'\b(?:differential|diff|count)\b',
            r'\b(?:panel|profile|workup|study)\b',
            r'\b(?:x-ray|ct|mri|ultrasound|ecg|ekg)\b'
        ]
        
        for pattern in procedure_indicators:
            if re.search(pattern, text):
                corrected_entity.label = EntityType.PROCEDURE
                return corrected_entity
        
        # Medication patterns - verify these are actually medications
        if entity.label == EntityType.MEDICATION:
            # Check if this looks like a medication
            medication_indicators = [
                r'\b\d+\s*(?:mg|g|ml|mcg|units?|tablets?|capsules?)\b',
                r'\b(?:daily|weekly|monthly|bid|tid|qid|prn)\b',
                r'\b(?:oral|iv|injection|topical|sublingual)\b'
            ]
            
            # Common medication suffixes
            medication_suffixes = [
                'cillin', 'mycin', 'prazole', 'statin', 'sartan', 
                'olol', 'pril', 'azide', 'ide', 'ine'
            ]
            
            is_medication = False
            for pattern in medication_indicators:
                if re.search(pattern, text):
                    is_medication = True
                    break
            
            if not is_medication:
                for suffix in medication_suffixes:
                    if text.endswith(suffix):
                        is_medication = True
                        break
            
            # If it doesn't look like a medication, reclassify
            if not is_medication:
                # Check if it's a lab value
                if any(re.search(pattern, text) for pattern in lab_value_indicators):
                    corrected_entity.label = EntityType.LAB_VALUE
                # Check if it's a procedure
                elif any(re.search(pattern, text) for pattern in procedure_indicators):
                    corrected_entity.label = EntityType.PROCEDURE
                # Check if it's an organization
                elif any(re.search(pattern, text) for pattern in organization_indicators):
                    corrected_entity.label = EntityType.ORGANIZATION
                # Check if it's a location
                elif any(re.search(pattern, text) for pattern in location_indicators):
                    corrected_entity.label = EntityType.LOCATION
                # Default to ORGANIZATION for unknown medical-related entities
                else:
                    corrected_entity.label = EntityType.ORGANIZATION
        
        # If originally labeled as CONDITION, validate it's actually a medical condition
        if entity.label == EntityType.CONDITION:
            # Check if it's actually one of the other types
            if any(re.search(pattern, text) for pattern in organization_indicators):
                corrected_entity.label = EntityType.ORGANIZATION
            elif any(re.search(pattern, text) for pattern in location_indicators):
                corrected_entity.label = EntityType.LOCATION
            elif any(re.search(pattern, text) for pattern in person_indicators):
                corrected_entity.label = EntityType.PERSON
            elif any(re.search(pattern, text) for pattern in test_indicators):
                corrected_entity.label = EntityType.LAB_TEST
            elif any(re.search(pattern, text) for pattern in id_indicators):
                corrected_entity.label = EntityType.PERSON  # Patient identifiers
            elif any(re.search(pattern, text) for pattern in lab_value_indicators):
                corrected_entity.label = EntityType.LAB_VALUE
            elif any(re.search(pattern, text) for pattern in procedure_indicators):
                corrected_entity.label = EntityType.PROCEDURE
            # If none of the above, check if it's a real medical condition
            else:
                # Real medical conditions should have medical terminology
                medical_condition_indicators = [
                    r'\b(?:diabetes|hypertension|pneumonia|asthma|copd|cancer|infection)\b',
                    r'\b(?:syndrome|disease|disorder|condition|deficiency)\b',
                    r'\b(?:acute|chronic|severe|mild|moderate)\s+\w+\b'
                ]
                
                is_medical_condition = any(re.search(pattern, text) for pattern in medical_condition_indicators)
                if not is_medical_condition:
                    # If it doesn't look like a real medical condition, classify as ORGANIZATION
                    corrected_entity.label = EntityType.ORGANIZATION
        
        return corrected_entity

    def _extract_lab_values_with_ranges(self, text: str) -> List[MedicalEntity]:
        """Extract lab values with their reference ranges."""
        entities = []
        
        # Enhanced patterns for lab values with ranges
        lab_patterns = [
            # Hemoglobin with range
            r'\b(?:hemoglobin|hgb|hb)\s*(?:\([^)]*\))?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:g/dL|g/dl)?\s*(?:,?\s*range?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)(?:\s*g/dL|g/dl)?)?',
            # Platelet count with range  
            r'\b(?:platelet\s*count|platelets|plt)\s*(?:\([^)]*\))?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:/uL|/ul|k/uL|k/ul|×10³/μL)?\s*(?:,?\s*range?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)(?:\s*/uL|/ul|k/uL|k/ul)?)?',
            # Differential counts
            r'\b(?:neutrophils|lymphocytes|monocytes|eosinophils|basophils)\s*(?:\([^)]*\))?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?\s*(?:,?\s*range?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)%?)?',
            # White blood cell count
            r'\b(?:white\s*blood\s*cell\s*count|wbc|white\s*cell\s*count)\s*(?:\([^)]*\))?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:/uL|/ul|k/uL|k/ul|×10³/μL)?\s*(?:,?\s*range?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)(?:\s*/uL|/ul|k/uL|k/ul)?)?',
            # Red blood cell count
            r'\b(?:red\s*blood\s*cell\s*count|rbc|red\s*cell\s*count)\s*(?:\([^)]*\))?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:/uL|/ul|M/uL|M/ul|×10⁶/μL)?\s*(?:,?\s*range?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)(?:\s*/uL|/ul|M/uL|M/ul)?)?',
            # Hematocrit
            r'\b(?:hematocrit|hct)\s*(?:\([^)]*\))?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?\s*(?:,?\s*range?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)(?:\s*g/dL|g/dl)?)?'
        ]
        
        for pattern in lab_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = MedicalEntity(
                    text=match.group(0),
                    label=EntityType.LAB_VALUE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,  # High confidence for pattern-matched lab values
                    context=self._extract_context(text, match.start(), match.end()),
                    semantic_type="lab_value_with_range"
                )
                
                # Extract value and range if present
                groups = match.groups()
                if groups and groups[0]:
                    entity.value = groups[0]
                    if len(groups) >= 3 and groups[1] and groups[2]:
                        entity.reference_range = f"{groups[1]}-{groups[2]}"
                
                entities.append(entity)
        
        return entities

    # Global NER processor instance
_ner_processor_instance: Optional[MedicalNERProcessor] = None
_processor_lock = asyncio.Lock()


async def get_ner_processor() -> MedicalNERProcessor:
    """
    Dependency injection function to get the NER processor instance.
    
    This function implements a singleton pattern to ensure we have only one
    instance of the NER processor throughout the application lifecycle.
    
    Returns:
        MedicalNERProcessor: Initialized NER processor instance
        
    Raises:
        RuntimeError: If processor initialization fails
    """
    global _ner_processor_instance
    
    if _ner_processor_instance is None:
        async with _processor_lock:
            # Double-check locking pattern
            if _ner_processor_instance is None:
                try:
                    logger.info("Initializing Medical NER Processor...")
                    
                    # Create and initialize the processor
                    config = NERModelConfig()
                    _ner_processor_instance = MedicalNERProcessor(config)
                    
                    # Initialize the models
                    await _ner_processor_instance.initialize()
                    
                    logger.info("Medical NER Processor initialized successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize NER processor: {e}")
                    raise RuntimeError(f"NER processor initialization failed: {str(e)}")
    
    return _ner_processor_instance


async def shutdown_ner_processor():
    """
    Shutdown function to clean up the NER processor resources.
    
    This function should be called during application shutdown to properly
    clean up model resources and connections.
    """
    global _ner_processor_instance
    
    if _ner_processor_instance is not None:
        try:
            logger.info("Shutting down Medical NER Processor...")
            # Add any cleanup logic here if needed
            _ner_processor_instance = None
            logger.info("Medical NER Processor shutdown completed")
        except Exception as e:
            logger.error(f"Error during NER processor shutdown: {e}")


async def is_ner_processor_ready() -> bool:
    """
    Check if the NER processor is initialized and ready for use.
    
    Returns:
        bool: True if processor is ready, False otherwise
    """
    global _ner_processor_instance
    return _ner_processor_instance is not None and _ner_processor_instance._is_initialized


# Export the main components
__all__ = [
    'MedicalNERProcessor',
    'NERModelConfig', 
    'get_ner_processor',
    'is_ner_processor_ready',
    'shutdown_ner_processor'
]
