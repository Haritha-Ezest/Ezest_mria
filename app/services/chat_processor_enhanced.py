"""
Enhanced Chat service for medical query processing with ChromaDB and Neo4j integration.

This module provides enhanced chat functionality specifically focused on retrieving
and processing data from ChromaDB vector store and Neo4j graph database.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import re
from transformers import pipeline
from app.schemas.graph import GraphQueryRequest
from app.schemas.chat import (
    ChatQueryRequest, ChatQueryResponse, ConversationContext,
    QueryIntent, QueryType, QueryComplexity, MessageRole,
    QueryAnalysis, MedicalEntity, QueryResult, ChatMessage,
    ConversationSession, ConversationStatus
)
from app.services.graph_client import Neo4jGraphClient
from app.services.insights_processor import InsightsProcessor
from app.services.ner_processor import MedicalNERProcessor
from app.services.vector_store import VectorStore
from app.services.ai_recommendations import get_ai_recommendation_generator
from app.common.utils import get_logger

logger = get_logger(__name__)


class EnhancedChatProcessor:
    """
    Enhanced chat processor with ChromaDB and Neo4j integration.
    
    This processor follows the specified workflow:
    1. Analyze user query and extract intent
    2. Retrieve relevant data from ChromaDB (semantic search)
    3. Query Neo4j graph database for structured data
    4. Combine and analyze data from both sources
    5. Generate comprehensive natural language response    """
    
    def __init__(
        self, 
        graph_client: Neo4jGraphClient,
        insights_processor: InsightsProcessor,
        ner_processor: MedicalNERProcessor,
        vector_store: VectorStore,
        enable_ai_enhancement: bool = True
    ):
        """Initialize enhanced chat processor with all dependencies."""
        self.graph_client = graph_client
        self.insights_processor = insights_processor
        self.ner_processor = ner_processor
        self.vector_store = vector_store
        self.enable_ai_enhancement = enable_ai_enhancement
        self.logger = logger
        
        # Initialize AI model only if enhancement is enabled
        if self.enable_ai_enhancement:
            try:
                self.ai_generator = pipeline(
                    "text-generation", 
                    model="distilgpt2",
                    device=-1,  # Use CPU to avoid GPU memory issues
                    framework="pt"
                )
                self.logger.info("AI model (DistilGPT-2) initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize AI model: {e}")
                self.ai_generator = None
                self.enable_ai_enhancement = False
        else:
            self.ai_generator = None

    
    
    async def process_enhanced_query(self, request: ChatQueryRequest) -> ChatQueryResponse:
        """
        Process query with enhanced data retrieval from ChromaDB and Neo4j.
        
        Workflow:
        1. Analyze query intent and extract medical entities
        2. Perform semantic search in ChromaDB
        3. Query structured data from Neo4j
        4. Combine and analyze results
        5. Generate comprehensive response
        """
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Analyze query intent
            query_analysis = await self._analyze_query_intent(request.query)
            
            # Step 2: Retrieve data from ChromaDB (semantic search)
            chromadb_results = await self._retrieve_from_chromadb(
                request.query, query_analysis, request.context
            )
            
            # Step 3: Query Neo4j graph database
            neo4j_results = await self._query_neo4j_graph(
                query_analysis, request.context
            )
            
            # Step 4: Combine and analyze results
            combined_results = await self._combine_and_analyze_results(
                chromadb_results, neo4j_results, query_analysis
            )
            
            # Step 5: Generate enhanced natural language response
            response_text = await self._generate_enhanced_response(
                combined_results, query_analysis, request.context
            )
            

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ChatQueryResponse(
                message="Enhanced query processed successfully",
                session_id=request.session_id or str(uuid.uuid4()),
                response_text=response_text,
                query_analysis=query_analysis,
                results=combined_results,
                medical_entities=await self._extract_medical_entities(combined_results),                suggested_followups=await self._generate_followups(query_analysis, combined_results),
                confidence=self._calculate_confidence(query_analysis, combined_results),
                processing_time=processing_time,
                sources_used=["chromadb", "neo4j", "medical_knowledge_graph"]
            )
            
        except Exception as e:
            self.logger.error(f"Error in enhanced query processing: {str(e)}")
            # Add specific debugging for slice errors
            if "unhashable type: 'slice'" in str(e):
                self.logger.error("SLICE ERROR DETECTED - investigating the call stack", exc_info=True)
                # Try to provide more diagnostic information
                try:
                    import traceback
                    self.logger.error(f"Full traceback: {traceback.format_exc()}")
                except Exception:
                    pass
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ChatQueryResponse(
                message="Error processing enhanced query",
                session_id=request.session_id or str(uuid.uuid4()),
                response_text=f"I apologize, but I encountered an error processing your query: {str(e)}. Please try rephrasing your question.",
                query_analysis=QueryAnalysis(
                    original_query=request.query,
                    cleaned_query=request.query,
                    detected_intent=QueryIntent(
                        primary_intent=QueryType.GENERAL_MEDICAL,
                        entities={},
                        confidence=0.0,
                        complexity=QueryComplexity.SIMPLE
                    ),
                    processing_approach="error_handling"
                ),
                results=[],                
                medical_entities=[],
                suggested_followups=["Could you please rephrase your question?"],
                confidence=0.0,
                processing_time=processing_time,
                sources_used=[]
            )

    async def _analyze_query_intent(self, query: str) -> QueryAnalysis:
        """Analyze query to understand intent and extract entities."""
        try:
            # Clean the query
            cleaned_query = re.sub(r'\s+', ' ', query.strip())
            
            # Use NER processor to extract medical entities with error handling
            medical_entities = {}
            try:
                medical_entities = await self.ner_processor.extract_entities(cleaned_query)
            except Exception as e:
                self.logger.warning(f"Entity extraction failed: {e}")
                # Provide fallback basic entities
                medical_entities = self._extract_basic_entities(cleaned_query)
            
            # Determine intent based on keywords and entities
            intent = self._determine_intent(cleaned_query)
            
            # Determine complexity
            complexity = self._determine_complexity(cleaned_query, medical_entities)
            
            return QueryAnalysis(
                original_query=query,
                cleaned_query=cleaned_query,
                detected_intent=QueryIntent(
                    primary_intent=intent,
                    entities=medical_entities,
                    confidence=0.8,  # Default confidence
                    complexity=complexity,
                    requires_graph_query=True,
                    requires_population_data="population" in cleaned_query.lower()
                ),
                processing_approach="enhanced_retrieval"
            )
        except Exception as e:
            self.logger.error(f"Error in query analysis: {e}")
            # Return basic analysis as fallback
            return QueryAnalysis(
                original_query=query,
                cleaned_query=query.strip(),
                detected_intent=QueryIntent(
                    primary_intent=QueryType.GENERAL_MEDICAL,
                    entities={},
                    confidence=0.3,
                    complexity=QueryComplexity.SIMPLE,
                    requires_graph_query=False,
                    requires_population_data=False
                ),                processing_approach="basic_retrieval"
            )

    async def _retrieve_from_chromadb(
        self, 
        query: str, 
        query_analysis: QueryAnalysis, 
        context: Optional[ConversationContext]
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant data from ChromaDB using semantic search with patient-specific focus."""
        chromadb_results = []
        
        try:
            # PRIORITY 1: If we have a patient context, get specific patient timeline first
            if context and context.patient_id:
                self.logger.info(f"Retrieving specific data for patient: {context.patient_id}")
                
                # Get patient timeline from ChromaDB
                patient_timeline = await self.vector_store.get_patient_timeline(context.patient_id)
                if patient_timeline:
                    chromadb_results.append({
                        "result_type": "chromadb_patient_timeline",
                        "content": patient_timeline.dict() if hasattr(patient_timeline, 'dict') else patient_timeline,
                        "relevance_score": 1.0,
                        "source_references": ["chromadb", "patient_timeline"],
                        "last_updated": datetime.utcnow().isoformat()
                    })
                    self.logger.info(f"Found patient timeline for {context.patient_id}")
                else:
                    self.logger.warning(f"No timeline found for patient {context.patient_id}")
                  # Search for patient-specific medical chunks
                patient_query = f"patient {context.patient_id} treatment medication therapy diagnosis"
                patient_chunks = await self.vector_store.search_similar_timelines(patient_query, limit=5)
                for chunk in patient_chunks:
                    chromadb_results.append({
                        "result_type": "chromadb_patient_chunk",
                        "content": chunk,
                        "relevance_score": chunk.get("similarity_score", 0.0),
                        "source_references": ["chromadb", "medical_chunk"],
                        "last_updated": datetime.utcnow().isoformat()
                    })
                
                # Also search with condition focus if provided
                if context.condition_focus:
                    condition_query = f"{context.patient_id} {context.condition_focus} treatment"
                    condition_results = await self.vector_store.search_similar_timelines(condition_query, limit=3)
                    for result in condition_results:
                        chromadb_results.append({
                            "result_type": "chromadb_condition_focused",
                            "content": result,
                            "relevance_score": result.get("similarity_score", 0.0),
                            "source_references": ["chromadb", "condition_focused"],
                            "last_updated": datetime.utcnow().isoformat()
                        })
            
            # PRIORITY 2: Perform semantic search for similar cases (lower priority)
            similar_timelines = await self.vector_store.search_similar_timelines(query, limit=3)
            for timeline in similar_timelines:
                chromadb_results.append({
                    "result_type": "chromadb_similar_timeline",
                    "content": timeline,
                    "relevance_score": timeline.get("similarity_score", 0.0),
                    "source_references": ["chromadb", "similar_timeline"],
                    "last_updated": datetime.utcnow().isoformat()
                })
              # PRIORITY 3: Search for relevant medical knowledge chunks
            if query_analysis.detected_intent.entities:
                entities_text = " ".join([
                    " ".join(entities) if isinstance(entities, list) else str(entities)
                    for entities in query_analysis.detected_intent.entities.values()
                    if entities
                ])
                if entities_text:
                    chunk_results = await self.vector_store.search_similar_timelines(entities_text, limit=2)
                    for chunk in chunk_results:
                        chromadb_results.append({
                            "result_type": "chromadb_medical_knowledge",
                            "content": chunk,
                            "relevance_score": chunk.get("similarity_score", 0.0),
                            "source_references": ["chromadb", "medical_chunk"],
                            "last_updated": datetime.utcnow().isoformat()
                        })
            
            self.logger.info(f"Retrieved {len(chromadb_results)} results from ChromaDB")
            return chromadb_results
            
        except Exception as e:
            self.logger.error(f"Error retrieving from ChromaDB: {str(e)}")
            return []

    async def _query_neo4j_graph(
        self, 
        query_analysis: QueryAnalysis, 
        context: Optional[ConversationContext]
    ) -> List[Dict[str, Any]]:
        """Query Neo4j graph database for structured medical data with patient-specific focus."""
        neo4j_results = []
        
        try:
            # PRIORITY 1: Patient-specific comprehensive queries
            if context and context.patient_id:
                self.logger.info(f"Querying Neo4j for specific patient: {context.patient_id}")
                  # Get comprehensive patient data including treatments
                patient_treatment_query = """
                MATCH (p:Patient {id: $patient_id})
                OPTIONAL MATCH (p)-[:HAS_VISIT]->(v:Visit)
                OPTIONAL MATCH (v)-[:DIAGNOSED_WITH]->(c:Condition)
                OPTIONAL MATCH (v)-[:PRESCRIBED]->(m:Medication)
                OPTIONAL MATCH (v)-[:PERFORMED]->(t:Test)
                OPTIONAL MATCH (v)-[:UNDERWENT]->(pr:Procedure)
                RETURN p.id as patient_id,
                       p.name as patient_name,                       collect(DISTINCT {
                           visit_id: v.id,
                           date: v.date,
                           visit_type: v.visit_type
                       }) as visits,
                       collect(DISTINCT c.condition_name) as all_conditions,
                       collect(DISTINCT m.medication_name) as all_medications,
                       collect(DISTINCT t.test_name) as all_tests,
                       collect(DISTINCT pr.procedure_name) as all_procedures
                LIMIT 1
                """
                
                patient_results = await self.graph_client.execute_query(
                    GraphQueryRequest(
                        query=patient_treatment_query,
                        parameters={"patient_id": context.patient_id}
                    )
                )
                
                if patient_results and patient_results.success and patient_results.results:
                    neo4j_results.append({
                        "result_type": "neo4j_patient_comprehensive",
                        "content": patient_results.results[0],
                        "relevance_score": 1.0,
                        "source_references": ["neo4j", "patient_graph"],
                        "last_updated": datetime.utcnow().isoformat()
                    })
                    self.logger.info(f"Found comprehensive patient data for {context.patient_id}")
                  # Get current treatments and medications for the patient
                current_treatments_query = """
                MATCH (p:Patient {id: $patient_id})-[:HAS_VISIT]->(v:Visit)-[:PRESCRIBED]->(m:Medication)
                OPTIONAL MATCH (v)-[:DIAGNOSED_WITH]->(c:Condition)
                RETURN m.medication_name as medication,
                       m.dosage as dosage,
                       m.start_date as start_date,
                       collect(DISTINCT c.condition_name) as treating_conditions,
                       v.date as prescribed_date
                ORDER BY v.date DESC
                LIMIT 10
                """
                
                treatments_results = await self.graph_client.execute_query(
                    GraphQueryRequest(
                        query=current_treatments_query,
                        parameters={"patient_id": context.patient_id}
                    )
                )
                
                if treatments_results and treatments_results.success and treatments_results.results:
                    neo4j_results.append({
                        "result_type": "neo4j_current_treatments",
                        "content": treatments_results.results,
                        "relevance_score": 1.0,
                        "source_references": ["neo4j", "current_treatments"],
                        "last_updated": datetime.utcnow().isoformat()
                    })
                  # If condition focus is specified, get condition-specific treatment data
                if context.condition_focus:
                    condition_treatment_query = """
                    MATCH (p:Patient {id: $patient_id})-[:HAS_VISIT]->(v:Visit)-[:DIAGNOSED_WITH]->(c:Condition)
                    WHERE c.condition_name CONTAINS $condition_name
                    OPTIONAL MATCH (v)-[:PRESCRIBED]->(m:Medication)
                    OPTIONAL MATCH (v)-[:PERFORMED]->(t:Test)
                    RETURN c.condition_name as condition,
                           v.date as visit_date,
                           collect(DISTINCT m.medication_name) as prescribed_medications,
                           collect(DISTINCT t.test_name) as related_tests
                    ORDER BY v.date DESC
                    LIMIT 10
                    """
                    
                    condition_results = await self.graph_client.execute_query(
                        GraphQueryRequest(
                            query=condition_treatment_query,
                            parameters={
                                "patient_id": context.patient_id,
                                "condition_name": context.condition_focus
                            }
                        )
                    )
                    
                    if condition_results and condition_results.success and condition_results.results:
                        neo4j_results.append({
                            "result_type": "neo4j_condition_treatments",
                            "content": condition_results.results,
                            "relevance_score": 1.0,
                            "source_references": ["neo4j", "condition_specific"],
                            "last_updated": datetime.utcnow().isoformat()
                        })
              # PRIORITY 2: Population-level queries for similar conditions (lower priority)
            entities = query_analysis.detected_intent.entities
            if entities.get("conditions"):
                conditions = entities["conditions"]
                population_query = """
                MATCH (p:Patient)-[:HAS_VISIT]->(v:Visit)-[:DIAGNOSED_WITH]->(c:Condition)
                WHERE c.condition_name IN $conditions
                WITH c, count(DISTINCT p) as patient_count, count(DISTINCT v) as visit_count
                OPTIONAL MATCH (p2:Patient)-[:HAS_VISIT]->(v2:Visit)-[:PRESCRIBED]->(m:Medication)
                WHERE (v2)-[:DIAGNOSED_WITH]->(c)
                RETURN c.condition_name as condition,
                       patient_count,
                       visit_count,
                       collect(DISTINCT m.medication_name)[0..5] as common_medications
                ORDER BY patient_count DESC LIMIT 5
                """
                
                population_results = await self.graph_client.execute_query(
                    GraphQueryRequest(
                        query=population_query,
                        parameters={"conditions": conditions}
                    )
                )
                
                if population_results and population_results.success and population_results.results:
                    neo4j_results.append({
                        "result_type": "neo4j_population_data",
                        "content": population_results.results,
                        "relevance_score": 0.8,
                        "source_references": ["neo4j", "population_graph"],
                        "last_updated": datetime.utcnow().isoformat()
                    })
            
            # PRIORITY 3: Medication interaction queries
            if entities.get("medications"):
                medications = entities["medications"]
                if len(medications) > 1:
                    interaction_query = """
                    MATCH (m1:Medication)-[r:INTERACTS_WITH]->(m2:Medication)
                    WHERE m1.medication_name IN $medications AND m2.medication_name IN $medications
                    RETURN m1.medication_name as med1, m2.medication_name as med2, 
                           collect(DISTINCT type(r)) as interaction_types
                    """
                    
                    interaction_results = await self.graph_client.execute_query(
                        GraphQueryRequest(
                            query=interaction_query,
                            parameters={"medications": medications}
                        )
                    )
                    
                    if interaction_results and interaction_results.success and interaction_results.results:
                        neo4j_results.append({
                            "result_type": "neo4j_drug_interactions",
                            "content": interaction_results.results,
                            "relevance_score": 0.9,
                            "source_references": ["neo4j", "medication_interactions"],
                            "last_updated": datetime.utcnow().isoformat()
                        })
            
            self.logger.info(f"Retrieved {len(neo4j_results)} results from Neo4j")
            return neo4j_results
            
        except Exception as e:
            self.logger.error(f"Error querying Neo4j: {str(e)}")
            return []
            return neo4j_results
            
        except Exception as e:
            self.logger.error(f"Error querying Neo4j: {str(e)}")
            return []

    async def _combine_and_analyze_results(
        self, 
        chromadb_results: List[Dict[str, Any]], 
        neo4j_results: List[Dict[str, Any]], 
        query_analysis: QueryAnalysis
    ) -> List[QueryResult]:
        """Combine and analyze results from both data sources."""

        if isinstance(chromadb_results, dict):
            self.logger.warning("chromadb_results was a dict, converting to list")
            chromadb_results = [chromadb_results]
        elif not isinstance(chromadb_results, list):
            chromadb_results = []

        if isinstance(neo4j_results, dict):
            self.logger.warning("neo4j_results was a dict, converting to list")
            neo4j_results = [neo4j_results]
        elif not isinstance(neo4j_results, list):
            neo4j_results = []
        combined_results = []        # Process ChromaDB results
        for result in chromadb_results:
            try:
                # Extract content properly
                content = result.get("content", {})
                if isinstance(content, str):
                    content = {"text": content}
                elif not isinstance(content, dict):
                    content = {"raw_data": str(content)}
                
                combined_results.append(QueryResult(
                    result_type=result.get("result_type", "chromadb_unknown"),
                    content=content,
                    relevance_score=max(0.0, min(1.0, float(result.get("relevance_score", 0.0)))),
                    source_references=result.get("source_references", ["chromadb", "unknown"]),
                    last_updated=datetime.utcnow()
                ))
            except Exception as e:
                self.logger.error(f"Error processing ChromaDB result: {e}")
                continue
        
        # Process Neo4j results
        for result in neo4j_results:
            try:
                # Extract content properly
                content = result.get("content", {})
                if isinstance(content, str):
                    content = {"text": content}
                elif not isinstance(content, dict):
                    content = {"raw_data": str(content)}
                
                combined_results.append(QueryResult(
                    result_type=result.get("result_type", "neo4j_unknown"),
                    content=content,
                    relevance_score=max(0.0, min(1.0, float(result.get("relevance_score", 0.0)))),
                    source_references=result.get("source_references", ["neo4j", "unknown"]),
                    last_updated=datetime.utcnow()
                ))
            except Exception as e:
                self.logger.error(f"Error processing Neo4j result: {e}")
                continue
          # Sort by relevance
        combined_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return combined_results

    async def _generate_enhanced_response(
        self, 
        results: List[QueryResult], 
        query_analysis: QueryAnalysis, 
        context: Optional[ConversationContext]
    ) -> str:
        """Generate enhanced natural language response from combined data with patient-specific focus."""
        if not results:
            return "I couldn't find specific information to answer your query. Could you please provide more details?"
        
        response_parts = []
        
        # Analyze results by source
        chromadb_data = [r for r in results if r.result_type.startswith("chromadb_")]
        neo4j_data = [r for r in results if r.result_type.startswith("neo4j_")]
        
        # PRIORITY: Patient-specific response generation
        if context and context.patient_id:
            response_parts.append(f"**Treatment Analysis for Patient {context.patient_id}**\n")
            
            # Process Neo4j comprehensive patient data
            patient_graph_data = [r for r in neo4j_data if r.result_type == "neo4j_patient_comprehensive"]
            current_treatments = [r for r in neo4j_data if r.result_type == "neo4j_current_treatments"]
            condition_treatments = [r for r in neo4j_data if r.result_type == "neo4j_condition_treatments"]
            
            if patient_graph_data:
                data = patient_graph_data[0].content
                response_parts.append(f"**Patient Overview:**")
                response_parts.append(f"- Patient ID: {data.get('patient_id', 'Unknown')}")
                response_parts.append(f"- Patient Name: {data.get('patient_name', 'Not specified')}")
                
                # Analyze visits and treatments
                visits = data.get('visits', [])
                if visits:
                    response_parts.append(f"- Total Medical Visits: {len(visits)}")
                    
                    # Extract latest visit information
                    latest_visit = visits[0] if visits else None
                    if latest_visit:
                        response_parts.append(f"- Latest Visit: {latest_visit.get('date', 'Unknown date')}")
                          # Current conditions
                        conditions = latest_visit.get('conditions', [])
                        if conditions:
                            # Safely handle conditions list
                            if isinstance(conditions, slice):
                                self.logger.warning("Found slice object in conditions data")
                                conditions = []
                            elif not isinstance(conditions, (list, tuple)):
                                try:
                                    conditions = list(conditions) if hasattr(conditions, '__iter__') and not isinstance(conditions, (str, dict)) else []
                                except Exception:
                                    conditions = []
                            
                            if conditions:
                                response_parts.append(f"- Current Conditions: {', '.join([c for c in conditions if c])}")
                        
                        # Current medications
                        medications = latest_visit.get('medications', [])
                        if medications:
                            # Safely handle medications list
                            if isinstance(medications, slice):
                                self.logger.warning("Found slice object in medications data")
                                medications = []
                            elif not isinstance(medications, (list, tuple)):
                                try:
                                    medications = list(medications) if hasattr(medications, '__iter__') and not isinstance(medications, (str, dict)) else []
                                except Exception:
                                    medications = []
                            
                            if medications:
                                response_parts.append(f"- Current Medications:")
                                med_list = medications[:5] if len(medications) > 5 else medications  # Show top 5
                                for med in med_list:
                                    if med.get('name'):
                                        med_info = f"  • {med['name']}"
                                        if med.get('dosage'):
                                            med_info += f" - {med['dosage']}"
                                        if med.get('frequency'):
                                            med_info += f" ({med['frequency']})"
                                        response_parts.append(med_info)
                          # Recent tests
                        tests = latest_visit.get('tests', [])
                        if tests:
                            # Safely handle tests list
                            if isinstance(tests, slice):
                                self.logger.warning("Found slice object in tests data")
                                tests = []
                            elif not isinstance(tests, (list, tuple)):
                                try:
                                    tests = list(tests) if hasattr(tests, '__iter__') and not isinstance(tests, (str, dict)) else []
                                except Exception:
                                    tests = []
                            
                            if tests:
                                response_parts.append(f"- Recent Lab Tests:")
                                test_list = tests[:3] if len(tests) > 3 else tests  # Show top 3
                                for test in test_list:
                                    if test.get('name'):
                                        test_info = f"  • {test['name']}"
                                        if test.get('value'):
                                            test_info += f": {test['value']}"
                                        if test.get('result'):
                                            test_info += f" ({test['result']})"
                                        response_parts.append(test_info)
                  # All medications across visits
                all_medications = data.get('all_medications', [])
                if all_medications:
                    # Safely handle all_medications list
                    if isinstance(all_medications, slice):
                        self.logger.warning("Found slice object in all_medications data")
                        all_medications = []
                    elif not isinstance(all_medications, (list, tuple)):
                        try:
                            all_medications = list(all_medications) if hasattr(all_medications, '__iter__') and not isinstance(all_medications, (str, dict)) else []
                        except Exception:
                            all_medications = []
                    
                    if all_medications:
                        response_parts.append(f"- All Prescribed Medications: {', '.join([m for m in all_medications if m])}")
              # Process current treatments
            if current_treatments:
                treatments = current_treatments[0].content
                if treatments:
                    # Handle case where treatments might be a slice object or other non-list type
                    if isinstance(treatments, slice):
                        self.logger.warning("Found slice object in treatments data, converting to empty list")
                        treatments = []
                    elif not isinstance(treatments, (list, tuple)):
                        self.logger.warning(f"Treatments data is not a list/tuple but {type(treatments)}, converting to list")
                        try:
                            # Try to convert to list if possible
                            treatments = list(treatments) if hasattr(treatments, '__iter__') and not isinstance(treatments, (str, dict)) else []
                        except Exception:
                            treatments = []
                    
                    if treatments:
                        response_parts.append(f"\n**Current Active Treatments:**")
                        # Safely slice the treatments list
                        treatment_list = treatments[:10] if len(treatments) > 10 else treatments
                        for treatment in treatment_list:  # Show top 10
                            med_name = treatment.get('medication', 'Unknown medication')
                            dosage = treatment.get('dosage', '')
                            frequency = treatment.get('frequency', '')
                            treating_conditions = treatment.get('treating_conditions', [])
                            
                            treatment_info = f"• {med_name}"
                            if dosage:
                                treatment_info += f" - {dosage}"
                            if frequency:
                                treatment_info += f" ({frequency})"
                            if treating_conditions:
                                treatment_info += f" [For: {', '.join(treating_conditions)}]"
                            
                            response_parts.append(treatment_info)
            
            # Process condition-specific treatments
            if condition_treatments and context.condition_focus:
                condition_data = condition_treatments[0].content
                
                # Safely handle condition_data list
                if isinstance(condition_data, slice):
                    self.logger.warning("Found slice object in condition_data")
                    condition_data = []
                elif not isinstance(condition_data, (list, tuple)):
                    try:
                        condition_data = list(condition_data) if hasattr(condition_data, '__iter__') and not isinstance(condition_data, (str, dict)) else []
                    except Exception:
                        condition_data = []
                
                if condition_data:
                    response_parts.append(f"\n**Treatment History for {context.condition_focus}:**")
                    record_list = condition_data[:5] if len(condition_data) > 5 else condition_data
                    for record in record_list:
                        visit_date = record.get('visit_date', 'Unknown date')
                        prescribed_medications = record.get('prescribed_medications', [])
                        
                        if prescribed_medications:
                            response_parts.append(f"• Visit on {visit_date}:")
                            for med in prescribed_medications:
                                if med.get('medication'):
                                    med_info = f"  - {med['medication']}"
                                    if med.get('dosage'):
                                        med_info += f" ({med['dosage']})"
                                    response_parts.append(med_info)
            
            # Process ChromaDB timeline data
            timeline_data = [r for r in chromadb_data if r.result_type == "chromadb_patient_timeline"]
            if timeline_data:
                timeline = timeline_data[0].content
                response_parts.append(f"\n**Clinical Timeline Summary:**")
                response_parts.append(f"- {timeline.get('summary', 'Timeline data available')}")
                
                # Timeline entries
                entries = timeline.get('timeline_entries', [])
                if entries:
                    response_parts.append(f"- Medical visits documented from {timeline.get('date_range', {}).get('start', 'Unknown')} to {timeline.get('date_range', {}).get('end', 'Unknown')}")
        
        else:
            # General response generation for non-patient specific queries
            intent = query_analysis.detected_intent.primary_intent
            
            if intent == QueryType.PATIENT_SUMMARY:
                response_parts.append(self._generate_patient_summary_from_data(chromadb_data, neo4j_data))
            elif intent == QueryType.POPULATION_HEALTH:
                response_parts.append(self._generate_population_response_from_data(chromadb_data, neo4j_data))
            elif intent == QueryType.TREATMENT_COMPARISON:
                response_parts.append(self._generate_treatment_comparison_from_data(chromadb_data, neo4j_data))
            elif intent == QueryType.MEDICATION_INTERACTION:
                response_parts.append(self._generate_medication_interaction_from_data(neo4j_data))
            else:
                response_parts.append(self._generate_general_response_from_data(results))
        
        # Add summary statistics
        total_data_points = len(results)
        response_parts.insert(1, f"\nI've analyzed {total_data_points} relevant data points from the medical knowledge base to answer your query.\n")
        
        # Add data source information
        sources_used = set()
        for result in results:
            if hasattr(result, 'source_references'):
                sources_used.update(result.source_references)
            else:
                # Fallback for older format
                sources_used.add("medical_database")        # Generate AI-enhanced response using the structured data
        structured_summary = await self._create_structured_medical_summary(results, context)
        
        # Use AI to enhance the response with medical insights if enabled
        if self.enable_ai_enhancement and self.ai_generator:
            ai_enhanced_response = await self._generate_ai_enhanced_medical_response(
                structured_data="\n".join(response_parts),
                query_context=query_analysis.original_query,
                patient_id=context.patient_id if context else None
            )
        else:
            # Fallback to basic structured response
            ai_enhanced_response = "\n".join(response_parts)
        
        # Add data source information
        sources_used_text = f"\n*This analysis is based on data from: {', '.join(sources_used)}*"
        
        # Return the AI-enhanced response with source information
        return f"{ai_enhanced_response}{sources_used_text}"
    
    def _generate_patient_summary_from_data(
        self, 
        chromadb_data: List[QueryResult], 
        neo4j_data: List[QueryResult]
    ) -> str:
        """Generate patient summary from combined data sources."""
        summary_parts = ["**Patient Summary**\n"]
        
        # Process timeline data from ChromaDB
        for result in chromadb_data:
            if result.result_type == "chromadb_patient_timeline":
                timeline_data = result.content
                summary_parts.append(f"**Clinical Timeline:**")
                summary_parts.append(f"- Total visits: {timeline_data.get('total_visits', 0)}")
                if timeline_data.get('summary'):
                    summary_parts.append(f"- Summary: {timeline_data['summary']}")
        
        # Process graph data from Neo4j
        for result in neo4j_data:
            if result.result_type == "neo4j_patient_graph":
                graph_data = result.content
                summary_parts.append(f"\n**Medical Records:**")                
                visits = graph_data.get('visits', [])
                if visits:
                    summary_parts.append(f"- Recent visits: {len(visits)}")
                
                conditions = graph_data.get('conditions', [])
                if conditions:
                    # Safely handle conditions list
                    if isinstance(conditions, slice):
                        self.logger.warning("Found slice object in conditions data")
                        conditions = []
                    elif not isinstance(conditions, (list, tuple)):
                        try:
                            conditions = list(conditions) if hasattr(conditions, '__iter__') and not isinstance(conditions, (str, dict)) else []
                        except Exception:
                            conditions = []
                    
                    if conditions:
                        condition_list = conditions[:5] if len(conditions) > 5 else conditions
                        condition_names = [c.get('name', 'Unknown') for c in condition_list]
                        summary_parts.append(f"- Active conditions: {', '.join(condition_names)}")                
                medications = graph_data.get('medications', [])
                if medications:
                    # Safely handle medications list
                    if isinstance(medications, slice):
                        self.logger.warning("Found slice object in medications data")
                        medications = []
                    elif not isinstance(medications, (list, tuple)):
                        try:
                            medications = list(medications) if hasattr(medications, '__iter__') and not isinstance(medications, (str, dict)) else []
                        except Exception:
                            medications = []
                    
                    if medications:
                        med_list = medications[:5] if len(medications) > 5 else medications
                        med_names = [m.get('name', 'Unknown') for m in med_list]
                        summary_parts.append(f"- Current medications: {', '.join(med_names)}")
        
        return "\n".join(summary_parts)
    
    def _generate_population_response_from_data(
        self, 
        chromadb_data: List[QueryResult], 
        neo4j_data: List[QueryResult]
    ) -> str:
        """Generate population health response from combined data."""
        response_parts = ["**Population Health Analysis**\n"]
          # Process similar cases from ChromaDB
        similar_cases = [r for r in chromadb_data if r.result_type == "chromadb_similar_timeline"]
        if similar_cases:
            # Safely handle similar_cases list
            if isinstance(similar_cases, slice):
                self.logger.warning("Found slice object in similar_cases data")
                similar_cases = []
            elif not isinstance(similar_cases, (list, tuple)):
                try:
                    similar_cases = list(similar_cases) if hasattr(similar_cases, '__iter__') and not isinstance(similar_cases, (str, dict)) else []
                except Exception:
                    similar_cases = []
            
            if similar_cases:
                response_parts.append(f"**Similar Patient Cases:** {len(similar_cases)} found")
                
                case_list = similar_cases[:3] if len(similar_cases) > 3 else similar_cases
                for i, case in enumerate(case_list, 1):
                    case_data = case.content
                    similarity = case_data.get('similarity_score', 0)
                    response_parts.append(f"{i}. Similarity: {similarity:.2f}")
          # Process population statistics from Neo4j
        for result in neo4j_data:
            if result.result_type == "neo4j_population_graph":
                pop_data = result.content
                
                # Safely handle pop_data list
                if isinstance(pop_data, slice):
                    self.logger.warning("Found slice object in pop_data")
                    pop_data = []
                elif not isinstance(pop_data, (list, tuple)):
                    try:
                        pop_data = list(pop_data) if hasattr(pop_data, '__iter__') and not isinstance(pop_data, (str, dict)) else []
                    except Exception:
                        pop_data = []
                
                if pop_data:
                    response_parts.append(f"\n**Population Statistics:**")
                    
                    pop_list = pop_data[:3] if len(pop_data) > 3 else pop_data  # Top 3 conditions
                    for condition_data in pop_list:
                        condition = condition_data.get('condition', 'Unknown')
                        patient_count = condition_data.get('patient_count', 0)
                        avg_visits = condition_data.get('avg_visits', 0)
                        
                        response_parts.append(f"- **{condition}:**")
                        response_parts.append(f"  - Patients: {patient_count}")
                        response_parts.append(f"  - Avg visits: {avg_visits:.1f}")
        
        return "\n".join(response_parts)
    
    def _generate_treatment_comparison_from_data(
        self, 
        chromadb_data: List[QueryResult], 
        neo4j_data: List[QueryResult]
    ) -> str:
        """Generate treatment comparison from data."""
        return "**Treatment Comparison Analysis**\n\nBased on the available data, I can provide insights into treatment patterns and outcomes from similar patient cases."
    
    def _generate_medication_interaction_from_data(self, neo4j_data: List[QueryResult]) -> str:
        """Generate medication interaction response from Neo4j data."""
        response_parts = ["**Medication Interaction Analysis**\n"]
        
        for result in neo4j_data:
            if result.result_type == "neo4j_medication_interactions":
                interactions = result.content
                if interactions:
                    response_parts.append("**Interactions Found:**")
                    for interaction in interactions:
                        med1 = interaction.get('med1', 'Unknown')
                        med2 = interaction.get('med2', 'Unknown')
                        response_parts.append(f"- {med1} ↔ {med2}")
                else:
                    response_parts.append("No known interactions found between the specified medications.")
        
        return "\n".join(response_parts)
    
    def _generate_general_response_from_data(self, results: List[QueryResult]) -> str:
        """Generate general response from available data."""
        return f"**Analysis Results**\n\nI've analyzed {len(results)} relevant data points from the medical knowledge base to answer your query."
    
    # Helper methods
    def _determine_intent(self, query: str) -> QueryType:
        """Determine query intent from text."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["summary", "summarize", "tell me about", "overview"]):
            return QueryType.PATIENT_SUMMARY
        elif any(word in query_lower for word in ["population", "patients", "cohort", "similar"]):
            return QueryType.POPULATION_HEALTH
        elif any(word in query_lower for word in ["compare", "comparison", "versus", "vs"]):
            return QueryType.TREATMENT_COMPARISON
        elif any(word in query_lower for word in ["interaction", "drug", "medication"]):
            return QueryType.MEDICATION_INTERACTION
        elif any(word in query_lower for word in ["timeline", "progression", "over time"]):
            return QueryType.TIMELINE_ANALYSIS
        else:
            return QueryType.GENERAL_MEDICAL
    
    def _determine_complexity(self, query: str, entities: Dict[str, List[str]]) -> QueryComplexity:
        """Determine query complexity."""
        entity_count = sum(len(entity_list) for entity_list in entities.values())
        
        if entity_count > 5 or "population" in query.lower():
            return QueryComplexity.ADVANCED
        elif entity_count > 2 or any(word in query.lower() for word in ["compare", "timeline", "progression"]):
            return QueryComplexity.COMPLEX
        elif entity_count > 0:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    async def _extract_medical_entities(self, results: List[QueryResult]) -> List[MedicalEntity]:
        """Extract medical entities from results."""
        entities = []
        
        for result in results:
            # Extract entities from result content
            if isinstance(result.content, dict):
                # Look for common medical entity fields
                for field in ["conditions", "medications", "procedures", "symptoms"]:
                    if field in result.content:
                        values = result.content[field]
                        if isinstance(values, list):
                            for value in values:
                                entities.append(MedicalEntity(
                                    entity_type=field[:-1],  # Remove 's' from plural
                                    entity_value=str(value),
                                    confidence=0.8
                                ))
        
        return entities
    
    async def _generate_followups(
        self, 
        query_analysis: QueryAnalysis, 
        results: List[QueryResult]
    ) -> List[str]:
        """Generate follow-up question suggestions."""
        followups = []
        
        intent = query_analysis.detected_intent.primary_intent
        
        if intent == QueryType.PATIENT_SUMMARY:
            followups.extend([
                "Would you like to see specific lab results?",
                "Should I compare this patient with similar cases?",
                "Do you want to analyze the treatment timeline?"
            ])
        elif intent == QueryType.POPULATION_HEALTH:
            followups.extend([
                "Would you like demographic breakdowns?",
                "Should I analyze treatment outcomes?",
                "Do you want to see trending patterns?"
            ])
        
        # Safely handle followups list
        if isinstance(followups, slice):
            self.logger.warning("Found slice object in followups data")
            followups = []
        elif not isinstance(followups, (list, tuple)):
            try:
                followups = list(followups) if hasattr(followups, '__iter__') and not isinstance(followups, (str, dict)) else []
            except Exception:
                followups = []
        
        return followups[:3] if len(followups) > 3 else followups
    
    def _calculate_confidence(
        self, 
        query_analysis: QueryAnalysis, 
        results: List[QueryResult]
    ) -> float:
        """Calculate response confidence."""
        if not results:
            return 0.0
        
        # Base confidence on query analysis
        base_confidence = query_analysis.detected_intent.confidence
        
        # Adjust based on result quality
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        
        return min((base_confidence * 0.4) + (avg_relevance * 0.6), 1.0)
    
    def _extract_basic_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract basic entities using keyword matching as fallback."""
        entities = {
            "patients": [],
            "conditions": [],
            "medications": [],
            "procedures": []
        }
        
        query_lower = query.lower()
        
        # Basic patient ID patterns
        patient_matches = re.findall(r'patient[_\s]*(\d+|[a-z]+\d+)', query_lower)
        entities["patients"].extend(patient_matches)
        
        # Common medical conditions
        conditions = ["diabetes", "hypertension", "asthma", "copd", "cancer", "pneumonia"]
        for condition in conditions:
            if condition in query_lower:
                entities["conditions"].append(condition)
        
        # Common medications
        medications = ["metformin", "lisinopril", "atorvastatin", "aspirin", "insulin"]
        for medication in medications:
            if medication in query_lower:
                entities["medications"].append(medication)
        
        return entities
    
    async def _generate_ai_enhanced_medical_response(
        self, 
        structured_data: str, 
        query_context: str, 
        patient_id: str = None
    ) -> str:
        """
        Generate AI-enhanced medical response using DistilGPT-2.
        
        Args:
            structured_data: The structured medical data summary
            query_context: The original query context
            patient_id: Patient identifier if available
        
        Returns:
            AI-generated comprehensive medical response
        """
        # Check if AI model is available
        if not self.ai_generator:
            self.logger.warning("AI model not available, returning structured data only")
            return structured_data
            
        try:
            # Create a more focused medical prompt
            if patient_id:
                prompt = f"""Clinical Summary for {patient_id}:
{structured_data}

Medical Interpretation: The clinical data indicates"""
            else:
                prompt = f"""Medical Data Analysis:
{structured_data}

Clinical Assessment: The analysis shows"""

            # Keep prompt concise to avoid repetition
            max_prompt_tokens = 300
            prompt_words = prompt.split()
            if len(prompt_words) > max_prompt_tokens:
                prompt = " ".join(prompt_words[:max_prompt_tokens])

            # Generate AI response with better parameters
            ai_response = self.ai_generator(
                prompt,
                max_new_tokens=100,  # Limit new token generation
                temperature=0.6,     # Less randomness for medical content
                do_sample=True,
                top_p=0.9,          # Nucleus sampling for better coherence
                repetition_penalty=1.2,  # Reduce repetition
                pad_token_id=self.ai_generator.tokenizer.eos_token_id,
                return_full_text=False,
                truncation=True
            )
            
            # Extract and clean the generated text
            if ai_response and len(ai_response) > 0:
                generated_text = ai_response[0]['generated_text'].strip()
                
                # Clean up the response
                ai_generated_part = self._clean_ai_medical_response(generated_text)
                
                # Only add AI insights if they're meaningful
                if ai_generated_part and len(ai_generated_part.strip()) > 15:
                    # Check if the AI output is not just repeating the input
                    if not self._is_repetitive_or_low_quality(ai_generated_part, structured_data):
                        return f"{structured_data}\n\n**AI Clinical Interpretation:**\n{ai_generated_part}"
                
            # Fallback to structured data if AI generation is poor
            return structured_data
                
        except Exception as e:
            self.logger.error(f"Error in AI response generation: {str(e)}")
            return structured_data
    
    def _is_repetitive_or_low_quality(self, ai_text: str, original_data: str) -> bool:
        """
        Check if AI-generated text is repetitive or low quality.
        
        Args:
            ai_text: AI-generated text
            original_data: Original structured data
            
        Returns:
            True if the text is repetitive or low quality
        """
        if not ai_text or len(ai_text.strip()) < 10:
            return True
        
        ai_words = ai_text.lower().split()
        
        # Check for excessive repetition
        unique_words = set(ai_words)
        if len(unique_words) / len(ai_words) < 0.5:  # Less than 50% unique words
            return True
        
        # Check if it's mostly copying the original data
        original_words = set(original_data.lower().split())
        ai_word_set = set(ai_words)
        overlap = len(ai_word_set & original_words)
        if overlap > len(ai_word_set) * 0.8:  # More than 80% overlap
            return True
        
        # Check for meaningless repeated phrases
        if "patient id" in ai_text.lower() and ai_text.lower().count("patient id") > 2:
            return True
            
        return False
    
    def _clean_ai_medical_response(self, ai_text: str) -> str:
        """
        Clean and validate AI-generated medical response.
        
        Args:
            ai_text: Raw AI-generated text
            
        Returns:
            Cleaned and validated medical response
        """
        if not ai_text:
            return ""
        
        # Remove potential harmful or inappropriate content
        cleaned_text = ai_text.strip()
        
        # Remove incomplete sentences at the end
        sentences = cleaned_text.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            cleaned_text = '.'.join(sentences[:-1]) + '.'
        
        # Ensure medical appropriateness
        medical_disclaimers = [
            "This analysis is for informational purposes only.",
            "Please consult with healthcare professionals for medical decisions.",
            "Individual patient factors may require different approaches."
        ]
        
        # Add medical disclaimer if the response is substantial
        if len(cleaned_text) > 50:
            cleaned_text += f"\n\n*Note: {medical_disclaimers[0]}*"
        
        return cleaned_text

    async def _create_structured_medical_summary(
        self, 
        results: List[QueryResult], 
        context: Optional[ConversationContext]
    ) -> str:
        """
        Create a structured medical summary from query results for AI processing.
        
        Args:
            results: Query results from ChromaDB and Neo4j
            context: Conversation context with patient information
            
        Returns:
            Structured medical data summary
        """
        summary_parts = []
        
        # Patient identification
        if context and context.patient_id:
            summary_parts.append(f"Patient ID: {context.patient_id}")
            if context.condition_focus:
                summary_parts.append(f"Primary Focus: {context.condition_focus}")
        
        # Process different types of medical data
        for result in results:
            if result.result_type == "chromadb_patient_timeline":
                timeline_data = result.content
                if isinstance(timeline_data, dict):
                    # Extract timeline information
                    timeline_entries = timeline_data.get('timeline_entries', [])
                    if timeline_entries:
                        summary_parts.append(f"Timeline: {len(timeline_entries)} medical visits")
                        
                        # Get most recent visit
                        latest_visit = timeline_entries[0] if timeline_entries else None
                        if latest_visit:
                            visit_date = latest_visit.get('date', 'Unknown')
                            summary_parts.append(f"Latest Visit: {visit_date}")
                            
                            # Extract key medical information
                            if latest_visit.get('tests'):
                                summary_parts.append(f"Recent Tests: {len(latest_visit['tests'])} performed")
                            if latest_visit.get('medications'):
                                summary_parts.append(f"Current Medications: {len(latest_visit['medications'])} prescribed")
                            if latest_visit.get('diagnoses'):
                                summary_parts.append(f"Diagnoses: {', '.join(latest_visit['diagnoses'])}")
            
            elif result.result_type == "neo4j_current_treatments":
                treatments = result.content
                if isinstance(treatments, list) and treatments:
                    summary_parts.append(f"Active Treatments: {len(treatments)} medications")
                    # Extract key medications
                    meds = [t.get('medication', 'Unknown') for t in treatments[:3]]
                    summary_parts.append(f"Key Medications: {', '.join(meds)}")
            
            elif result.result_type == "neo4j_patient_comprehensive":
                patient_data = result.content
                if isinstance(patient_data, dict):
                    visits = patient_data.get('visits', [])
                    conditions = patient_data.get('all_conditions', [])
                    medications = patient_data.get('all_medications', [])
                    
                    if visits:
                        summary_parts.append(f"Total Visits: {len(visits)}")
                    if conditions:
                        summary_parts.append(f"Conditions: {', '.join(conditions[:3])}")
                    if medications:
                        summary_parts.append(f"Medication History: {', '.join(medications[:3])}")
        
        return '\n'.join(summary_parts) if summary_parts else "Limited medical data available"

    # ...existing code...

# Dependency function
def get_enhanced_chat_processor() -> EnhancedChatProcessor:
    """Get enhanced chat processor with all dependencies."""
    from app.services.graph_client import Neo4jGraphClient
    from app.services.insights_processor import InsightsProcessor
    from app.services.ner_processor import MedicalNERProcessor
    from app.services.vector_store import VectorStore
    from app.config import get_settings
    
    settings = get_settings()
    
    # Initialize all dependencies
    graph_client = Neo4jGraphClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database
    )
    
    vector_store = VectorStore(persist_directory=settings.chroma_persist_directory)
    insights_processor = InsightsProcessor(graph_client)
    ner_processor = MedicalNERProcessor()
    
    return EnhancedChatProcessor(
        graph_client, 
        insights_processor, 
        ner_processor, 
        vector_store,
        enable_ai_enhancement=True  # Enable AI enhancement by default
    )
