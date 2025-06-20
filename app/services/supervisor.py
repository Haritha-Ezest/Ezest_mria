"""
LangChain-based supervisor service for orchestrating the medical document processing pipeline.

This module implements the core supervisor logic using LangChain/LangGraph for workflow
orchestration, job queue management, and agent coordination with Azure best practices.
"""

import asyncio
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from uuid import uuid4

import redis.asyncio as redis
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.schemas.supervisor import (
    JobRequest, JobResponse, JobStatus, WorkflowType, JobType, JobPriority,
    AgentType, QueueStatus, WorkflowDefinition
)
from app.schemas.ocr import OCRRequest, OCRResponse, OCRConfiguration, OCRStatus
from app.schemas.graph import PatientGraphRequest
from app.services.graph_client import Neo4jGraphClient
from app.common.utils import get_logger


# Configure logging
logger = get_logger(__name__)

# Global supervisor instance
supervisor_instance: Optional["LangChainSupervisor"] = None


class SupervisorState(dict):
    """State management for supervisor workflows."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize default state
        self.setdefault('job_id', str(uuid4()))
        self.setdefault('status', JobStatus.QUEUED)
        self.setdefault('current_agent', None)
        self.setdefault('agent_results', {})
        self.setdefault('errors', [])
        self.setdefault('retry_count', 0)
        self.setdefault('created_at', datetime.now())
        self.setdefault('context', {})


class SupervisorCallbackHandler(AsyncCallbackHandler):
    """Async callback handler for supervisor workflow monitoring."""
    
    def __init__(self, job_id: str, redis_client: redis.Redis):
        self.job_id = job_id
        self.redis_client = redis_client
        
    async def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Handle agent action events."""
        logger.info(f"Job {self.job_id}: Agent action - {action.tool}")
        await self._update_job_status(f"Executing: {action.tool}")
        
    async def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Handle agent completion events."""
        logger.info(f"Job {self.job_id}: Agent finished")
        await self._update_job_status("Agent completed")
        
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Handle tool execution start."""
        tool_name = serialized.get('name', 'unknown')
        logger.info(f"Job {self.job_id}: Starting tool - {tool_name}")
        
    async def on_tool_end(self, output: str, **kwargs) -> None:
        """Handle tool execution completion."""
        logger.info(f"Job {self.job_id}: Tool completed")        
    async def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Handle tool execution errors."""
        logger.error(f"Job {self.job_id}: Tool error - {error}")
        await self._update_job_error(str(error))
        
    async def _update_job_status(self, message: str):
        """Update job status in Redis."""
        try:
            key = f"job_status:{self.job_id}"
            status_data = {
                'current_stage': message,
                'updated_at': datetime.now().isoformat()
            }
            await self.redis_client.hset(key, mapping=status_data)
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
            
    async def _update_job_error(self, error_message: str):
        """Update job error in Redis."""
        try:
            key = f"job_errors:{self.job_id}"
            error_data = {
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            }
            await self.redis_client.lpush(key, json.dumps(error_data))
        except Exception as e:
            logger.error(f"Failed to update job error: {e}")


class LangChainSupervisor:
    """
    LangChain-based supervisor for orchestrating medical document processing workflows.
    
    Implements the linear workflow pattern specified in the graph LR:
    
    A[Supervisor Entry] --> B(OCR Agent)       --> C(Supervisor)
    C(Supervisor)       --> D(NER Agent)      --> E(Supervisor)  
    E(Supervisor)       --> F(Chunking Agent) --> G(Supervisor)
    G(Supervisor)       --> H(Graph Agent)    --> I(Supervisor)
    I(Supervisor)       --> J(Insight Agent)  --> K(Supervisor)
    K(Supervisor)       --> L(Chat Agent)     --> M[Supervisor End]
    
    Key Features:
    - Linear workflow progression with supervisor orchestration
    - Each agent returns to supervisor after processing
    - State persistence via LangGraph checkpointing
    - Redis backend for job queue management
    - Comprehensive error handling and retry logic
    - Real-time progress tracking and monitoring
    
    Workflow Types:
    - OCR_ONLY: Just OCR processing
    - OCR_TO_NER: OCR + NER processing
    - COMPLETE_PIPELINE: Full linear workflow (OCR → NER → Chunking → Graph → Insight → Chat)
    - DOCUMENT_TO_GRAPH: Document processing to knowledge graph with insights and chat
    - INSIGHT_GENERATION: Generate insights from existing data
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the supervisor with required components."""
        self.redis_client = None
        self.redis_url = redis_url
        self.workflow_graph = None
        self.checkpointer = MemorySaver()
        
        # Workflow definitions
        self.workflows = self._initialize_workflows()
        
        # Agent configurations
        self.agent_configs = self._initialize_agent_configs()
        
        # Metrics tracking
        self.metrics = {
            'jobs_processed': 0,
            'jobs_successful': 0,
            'jobs_failed': 0,
            'average_processing_time': 0.0
        }
        
    async def initialize(self):
        """Initialize async components."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Successfully connected to Redis")
            
            # Initialize workflow graph
            self._build_workflow_graph()
            
            logger.info("Supervisor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize supervisor: {e}")
            raise
            
    def _initialize_workflows(self) -> Dict[WorkflowType, WorkflowDefinition]:
        """Initialize predefined workflow definitions."""
        workflows = {}
        
        # OCR-only workflow
        workflows[WorkflowType.OCR_ONLY] = WorkflowDefinition(
            workflow_id="ocr_only",
            workflow_type=WorkflowType.OCR_ONLY,
            name="OCR Text Extraction",
            description="Extract text from documents using OCR",
            agent_sequence=[AgentType.OCR],
            estimated_duration_seconds=120
        )
        
        # OCR to NER workflow
        workflows[WorkflowType.OCR_TO_NER] = WorkflowDefinition(
            workflow_id="ocr_to_ner",
            workflow_type=WorkflowType.OCR_TO_NER,
            name="OCR and Medical Entity Recognition",
            description="Extract text and identify medical entities",
            agent_sequence=[AgentType.OCR, AgentType.NER],
            estimated_duration_seconds=180
        )
          # Complete pipeline workflow
        workflows[WorkflowType.COMPLETE_PIPELINE] = WorkflowDefinition(
            workflow_id="complete_pipeline",
            workflow_type=WorkflowType.COMPLETE_PIPELINE,
            name="Complete Medical Document Processing",
            description="Full pipeline from OCR to knowledge graph with insights and chat",
            agent_sequence=[
                AgentType.OCR, 
                AgentType.NER, 
                AgentType.CHUNKING, 
                AgentType.GRAPH,
                AgentType.INSIGHT,
                AgentType.CHAT
            ],
            estimated_duration_seconds=480
        )
          # OCR to Graph with Insights and Chat workflow
        workflows[WorkflowType.DOCUMENT_TO_GRAPH] = WorkflowDefinition(
            workflow_id="document_to_graph",
            workflow_type=WorkflowType.DOCUMENT_TO_GRAPH,
            name="Document to Knowledge Graph with Insights and Chat",
            description="Process documents into knowledge graph, generate insights, and prepare chat",
            agent_sequence=[
                AgentType.OCR, 
                AgentType.NER, 
                AgentType.CHUNKING, 
                AgentType.GRAPH,
                AgentType.INSIGHT,
                AgentType.CHAT
            ],
            estimated_duration_seconds=480
        )
        
        # Insight generation workflow
        workflows[WorkflowType.INSIGHT_GENERATION] = WorkflowDefinition(
            workflow_id="insight_generation",
            workflow_type=WorkflowType.INSIGHT_GENERATION,
            name="Patient Insight Generation",
            description="Generate insights from existing patient data",
            agent_sequence=[AgentType.INSIGHT],
            estimated_duration_seconds=90
        )

        return workflows
        
    def _initialize_agent_configs(self) -> Dict[AgentType, Dict[str, Any]]:
        """Initialize default configurations for each agent type."""
        agent_configs = {
            AgentType.OCR: {
                'timeout_seconds': 180,
                'retry_attempts': 3,
                'default_config': OCRConfiguration().dict()
            },
            AgentType.NER: {
                'timeout_seconds': 120,
                'retry_attempts': 2,
                'model_name': 'medical_ner_v1'
            },
            AgentType.CHUNKING: {
                'timeout_seconds': 90,
                'retry_attempts': 2,
                'chunk_size': 1000,
                'overlap': 200
            },
            AgentType.GRAPH: {
                'timeout_seconds': 150,
                'retry_attempts': 2,
                'batch_size': 50
            },
            AgentType.INSIGHT: {
                'timeout_seconds': 120,
                'retry_attempts': 2,
                'analysis_depth': 'comprehensive'            },
            AgentType.CHAT: {
                'timeout_seconds': 60,
                'retry_attempts': 2,
                'max_context_length': 4000,
                'response_format': 'medical'
            }
        }
        
        return agent_configs
        
    def _build_workflow_graph(self):
        """Build the LangGraph workflow for agent orchestration following linear supervisor flow."""
        
        def supervisor_node(state: SupervisorState) -> SupervisorState:
            """Supervisor decision node - manages linear workflow progression."""
            workflow_type = state.get('workflow_type', WorkflowType.COMPLETE_PIPELINE)
            workflow = self.workflows.get(workflow_type)
            
            if not workflow:
                state['status'] = JobStatus.FAILED
                state['errors'].append(f"Unknown workflow type: {workflow_type}")
                return state
                
            # Get current position in workflow
            current_agent_index = state.get('current_agent_index', 0)
            agent_sequence = workflow.agent_sequence
            
            # Check if workflow is complete
            if current_agent_index >= len(agent_sequence):
                state['status'] = JobStatus.COMPLETED
                state['current_agent'] = None
                state['next_action'] = 'END'
                logger.info(f"Job {state['job_id']}: Workflow completed successfully")
                return state
                
            # Set next agent in linear sequence
            next_agent = agent_sequence[current_agent_index]
            state['current_agent'] = next_agent
            state['next_action'] = next_agent.value            
            logger.info(f"Job {state['job_id']}: Supervisor directing to {next_agent.value} (step {current_agent_index + 1}/{len(agent_sequence)})")
            
            return state
            
        def ocr_agent_node(state: SupervisorState) -> SupervisorState:
            """OCR agent processing node - returns to supervisor after completion."""
            result = asyncio.create_task(self._execute_ocr_agent(state))
            return result
            
        def ner_agent_node(state: SupervisorState) -> SupervisorState:
            """NER agent processing node - returns to supervisor after completion."""
            result = asyncio.create_task(self._execute_ner_agent(state))
            return result
            
        def chunking_agent_node(state: SupervisorState) -> SupervisorState:
            """Chunking agent processing node - returns to supervisor after completion."""
            result = asyncio.create_task(self._execute_chunking_agent(state))
            return result
            
        def graph_agent_node(state: SupervisorState) -> SupervisorState:
            """Graph agent processing node - returns to supervisor after completion."""
            result = asyncio.create_task(self._execute_graph_agent(state))
            return result
            
        def insight_agent_node(state: SupervisorState) -> SupervisorState:
            """Insight agent processing node - returns to supervisor after completion."""
            result = asyncio.create_task(self._execute_insight_agent(state))
            return result
            
        def chat_agent_node(state: SupervisorState) -> SupervisorState:
            """Chat agent processing node - returns to supervisor after completion."""
            result = asyncio.create_task(self._execute_chat_agent(state))
            return result
            
        def should_continue(state: SupervisorState) -> str:
            """Determine next node based on current state - implements linear workflow."""
            if state.get('status') in [JobStatus.FAILED, JobStatus.COMPLETED]:
                return END
                
            next_action = state.get('next_action')
            if next_action == 'END':
                return END
            elif next_action == AgentType.OCR.value:
                return "ocr_agent"
            elif next_action == AgentType.NER.value:
                return "ner_agent"  
            elif next_action == AgentType.CHUNKING.value:
                return "chunking_agent"
            elif next_action == AgentType.GRAPH.value:
                return "graph_agent"
            elif next_action == AgentType.INSIGHT.value:
                return "insight_agent"
            elif next_action == AgentType.CHAT.value:
                return "chat_agent"
            else:
                return END                
        # Build the graph following your specified linear workflow:
        # A[Supervisor Entry] --> B(OCR Agent) --> C(Supervisor) --> D(NER Agent) --> etc.
        workflow = StateGraph(SupervisorState)
        
        # Add nodes
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("ocr_agent", ocr_agent_node)
        workflow.add_node("ner_agent", ner_agent_node)
        workflow.add_node("chunking_agent", chunking_agent_node)
        workflow.add_node("graph_agent", graph_agent_node)
        workflow.add_node("insight_agent", insight_agent_node)
        workflow.add_node("chat_agent", chat_agent_node)
        
        # Set entry point - A[Supervisor Entry]
        workflow.set_entry_point("supervisor")
        
        # Add conditional edges from supervisor to agents
        workflow.add_conditional_edges("supervisor", should_continue)
        
        # Linear flow: Each agent returns to supervisor for next decision
        # B(OCR Agent) --> C(Supervisor)
        # D(NER Agent) --> E(Supervisor)  
        # F(Chunking Agent) --> G(Supervisor)
        # H(Graph Agent) --> I(Supervisor)
        # J(Insight Agent) --> K(Supervisor)
        # L(Chat Agent) --> M[Supervisor End]
        workflow.add_edge("ocr_agent", "supervisor")
        workflow.add_edge("ner_agent", "supervisor")
        workflow.add_edge("chunking_agent", "supervisor")
        workflow.add_edge("graph_agent", "supervisor")
        workflow.add_edge("insight_agent", "supervisor")
        workflow.add_edge("chat_agent", "supervisor")
        
        # Compile the graph with checkpoint support for workflow persistence
        self.workflow_graph = workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("Workflow graph compiled successfully with linear supervisor flow")

    async def _execute_insight_agent(self, state: SupervisorState) -> SupervisorState:
        """Execute insight agent task."""
        job_id = state['job_id']
        logger.info(f"Job {job_id}: Executing insight agent")
        
        try:
            # Update status
            state['status'] = JobStatus.RUNNING
            state['current_stage'] = 'Insight Generation'
            
            # Get patient data from previous agents
            patient_id = state.get('patient_id')
            graph_results = state.get('agent_results', {}).get('graph', {})
            
            if not patient_id:
                logger.warning(f"Job {job_id}: No patient ID found for insight generation")
                state['agent_results']['insight'] = {
                    "success": False,
                    "message": "No patient ID provided for insight generation"
                }
            else:
                # Generate insights using the graph client
                graph_client = Neo4jGraphClient()
                await graph_client.initialize()
                
                try:
                    # Get comprehensive patient insights
                    patient_insights = await graph_client.get_patient_insights(patient_id)
                    
                    insights_result = {
                        "success": True,
                        "patient_id": patient_id,
                        "insights": patient_insights,
                        "generated_at": datetime.now().isoformat(),
                        "processing_time": 2.0,
                        "insight_categories": [
                            "patient_summary",
                            "timeline_analysis", 
                            "drug_interactions",
                            "care_gaps"
                        ]
                    }
                    
                    state['agent_results']['insight'] = insights_result
                    
                    logger.info(f"Job {job_id}: Insight agent completed successfully")
                    
                except Exception as insight_error:
                    logger.error(f"Job {job_id}: Failed to generate insights - {insight_error}")
                    state['agent_results']['insight'] = {
                        "success": False,
                        "error": str(insight_error),
                        "message": "Failed to generate patient insights"
                    }
                finally:
                    await graph_client.close()
            
            # Move to next agent
            state['current_agent_index'] = state.get('current_agent_index', 0) + 1
            
        except Exception as e:
            logger.error(f"Job {job_id}: Insight agent failed - {e}")
            state['status'] = JobStatus.FAILED
            state['errors'].append(f"Insight agent error: {str(e)}")
            
        return state
    
    async def _execute_chat_agent(self, state: SupervisorState) -> SupervisorState:
        """Execute chat agent task."""
        job_id = state['job_id']
        logger.info(f"Job {job_id}: Executing chat agent")
        
        try:
            # Update status
            state['status'] = JobStatus.RUNNING
            state['current_stage'] = 'Chat Processing'
            
            # Get context from previous agents
            patient_id = state.get('patient_id')
            extracted_text = state.get('extracted_text', "")
            ner_results = state.get('agent_results', {}).get('ner', {})
            insights = state.get('agent_results', {}).get('insight', {})
            
            # Prepare chat context
            chat_context = {
                "patient_id": patient_id,
                "document_text": extracted_text[:1000],  # Limit text for context
                "medical_entities": ner_results.get('entities', [])[:10],  # Limit entities
                "insights_summary": insights.get('insights', {}).get('patient_summary', {}),
                "available_data": {
                    "has_text": bool(extracted_text),
                    "has_entities": bool(ner_results.get('entities')),
                    "has_insights": bool(insights.get('success'))
                }
            }
            
            # Generate initial chat readiness response
            chat_result = {
                "success": True,
                "patient_id": patient_id,
                "chat_ready": True,
                "context_prepared": True,
                "context_summary": chat_context,
                "capabilities": [
                    "question_answering",
                    "medical_entity_explanation",
                    "patient_summary",
                    "document_analysis"
                ],
                "processing_time": 0.5,
                "message": "Chat agent ready for patient interaction"
            }
            
            state['agent_results']['chat'] = chat_result
            
            # Move to next agent
            state['current_agent_index'] = state.get('current_agent_index', 0) + 1
            
            logger.info(f"Job {job_id}: Chat agent completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id}: Chat agent failed - {e}")
            state['status'] = JobStatus.FAILED
            state['errors'].append(f"Chat agent error: {str(e)}")
            
        return state
        
    async def _execute_ocr_agent(self, state: SupervisorState) -> SupervisorState:
        """Execute OCR agent - placeholder implementation."""
        job_id = state['job_id']
        logger.info(f"Job {job_id}: Executing OCR agent")
        
        try:
            # Update status
            state['status'] = JobStatus.RUNNING
            state['current_stage'] = 'OCR Processing'
            
            # Get file paths from context
            file_paths = state.get('document_ids', [])
            
            # Placeholder OCR processing - in real implementation, would call OCR service
            extracted_text = "Sample extracted text from medical documents:\n"
            extracted_text += "Fasting Glucose: 165 mg/dL\n"
            extracted_text += "Diagnosis: Type 2 Diabetes Mellitus\n"
            extracted_text += "Patient history includes hypertension and hyperlipidemia."
            
            state['extracted_text'] = extracted_text
            state['agent_results']['ocr'] = {
                "success": True,
                "extracted_text": extracted_text,
                "processing_time": 2.5,
                "confidence_score": 0.95,
                "pages_processed": len(file_paths) if file_paths else 1,
                "document_ids": file_paths
            }
            
            # Move to next agent in workflow
            state['current_agent_index'] = state.get('current_agent_index', 0) + 1
            
            logger.info(f"Job {job_id}: OCR agent completed - returning to supervisor")
            
        except Exception as e:
            logger.error(f"Job {job_id}: OCR agent failed - {e}")
            state['status'] = JobStatus.FAILED
            state['errors'].append(f"OCR agent error: {str(e)}")
            
        return state
        
    async def _execute_ner_agent(self, state: SupervisorState) -> SupervisorState:
        """Execute NER agent - placeholder implementation."""
        job_id = state['job_id']
        logger.info(f"Job {job_id}: Executing NER agent")
        
        try:
            # Update status
            state['status'] = JobStatus.RUNNING
            state['current_stage'] = 'NER Processing'
            
            # Placeholder NER processing
            state['agent_results']['ner'] = {
                "success": True,
                "entities": [
                    {"text": "diabetes", "label": "CONDITION", "confidence": 0.98},
                    {"text": "metformin", "label": "MEDICATION", "confidence": 0.95}
                ],
                "processing_time": 1.8,
                "entity_count": 2
            }
            
            # Move to next agent in workflow
            state['current_agent_index'] = state.get('current_agent_index', 0) + 1
            
            logger.info(f"Job {job_id}: NER agent completed - returning to supervisor")
            
        except Exception as e:
            logger.error(f"Job {job_id}: NER agent failed - {e}")
            state['status'] = JobStatus.FAILED
            state['errors'].append(f"NER agent error: {str(e)}")
            
        return state
        
    async def _execute_chunking_agent(self, state: SupervisorState) -> SupervisorState:
        """Execute chunking agent - placeholder implementation."""
        job_id = state['job_id']
        logger.info(f"Job {job_id}: Executing Chunking agent")
        
        try:
            # Update status
            state['status'] = JobStatus.RUNNING
            state['current_stage'] = 'Document Chunking'
            
            # Placeholder chunking processing
            state['agent_results']['chunking'] = {
                "success": True,
                "chunks": [
                    {"chunk_id": "chunk_1", "text": "Patient medical history...", "metadata": {}},
                    {"chunk_id": "chunk_2", "text": "Current medications...", "metadata": {}}
                ],
                "chunk_count": 2,
                "processing_time": 1.2
            }
            
            # Move to next agent in workflow
            state['current_agent_index'] = state.get('current_agent_index', 0) + 1
            
            logger.info(f"Job {job_id}: Chunking agent completed - returning to supervisor")
            
        except Exception as e:
            logger.error(f"Job {job_id}: Chunking agent failed - {e}")
            state['status'] = JobStatus.FAILED
            state['errors'].append(f"Chunking agent error: {str(e)}")
            
        return state
        
    async def _execute_graph_agent(self, state: SupervisorState) -> SupervisorState:
        """Execute graph agent - placeholder implementation."""
        job_id = state['job_id']
        logger.info(f"Job {job_id}: Executing Graph agent")
        
        try:
            # Update status
            state['status'] = JobStatus.RUNNING
            state['current_stage'] = 'Knowledge Graph Update'
            
            # Placeholder graph processing
            state['agent_results']['graph'] = {
                "success": True,
                "nodes_created": 5,
                "relationships_created": 8,
                "processing_time": 3.2,
                "graph_updates": ["patient_node", "condition_nodes", "medication_nodes"]
            }
            
            # Move to next agent in workflow
            state['current_agent_index'] = state.get('current_agent_index', 0) + 1
            
            logger.info(f"Job {job_id}: Graph agent completed - returning to supervisor")
            
        except Exception as e:
            logger.error(f"Job {job_id}: Graph agent failed - {e}")
            state['status'] = JobStatus.FAILED
            state['errors'].append(f"Graph agent error: {str(e)}")
            
        return state

    async def enqueue_job(self, job_request: JobRequest) -> JobResponse:
        """Enqueue a new job for processing."""
        job_id = str(uuid4())
        created_at = datetime.now()
        
        # Get workflow definition
        workflow = self.workflows.get(job_request.workflow_type)
        if not workflow:
            raise ValueError(f"Unsupported workflow type: {job_request.workflow_type}")
        
        # Create job response
        job_response = JobResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            job_type=job_request.job_type,
            workflow_type=job_request.workflow_type,
            priority=job_request.priority,
            created_at=created_at,
            max_retries=job_request.max_retries,
            assigned_agents=workflow.agent_sequence,
            processing_time_seconds=None  # Will be calculated when job starts
        )
        
        # Store job in Redis
        try:
            status_key = f"job_status:{job_id}"
            job_data = {
                'job_id': job_id,
                'status': JobStatus.QUEUED.value,
                'job_type': job_request.job_type.value,
                'workflow_type': job_request.workflow_type.value,
                'priority': job_request.priority.value,
                'created_at': created_at.isoformat(),
                'max_retries': job_request.max_retries,
                'retry_count': 0,
                'assigned_agents': ','.join([agent.value for agent in workflow.agent_sequence]),
                'patient_id': job_request.patient_id or '',
                'document_ids': ','.join(job_request.document_ids),
                'file_paths': ','.join(job_request.file_paths),
                'processing_config': json.dumps(job_request.processing_config),
                'metadata': json.dumps(job_request.metadata),
                'timeout_seconds': job_request.timeout_seconds,
                'current_stage': 'Job queued for processing'
            }
            
            await self.redis_client.hset(status_key, mapping=job_data)
            
            # Add to processing queue
            queue_key = f"job_queue:{job_request.priority.value}"
            await self.redis_client.lpush(queue_key, job_id)
            
            logger.info(f"Job {job_id} enqueued successfully in Redis")
            
        except Exception as e:
            logger.error(f"Failed to store job {job_id} in Redis: {e}")
            raise
        
        return job_response

    async def get_job_status(self, job_id: str) -> Optional[JobResponse]:
        """Get current status of a job."""
        try:
            logger.info(f"Looking for job status: {job_id}")
            
            # Check Redis connection
            if not self.redis_client:
                logger.error("Redis client is not initialized")
                return None
            
            # Test Redis connection
            try:
                await self.redis_client.ping()
                logger.info("Redis connection is active")
            except Exception as ping_error:
                logger.error(f"Redis ping failed: {ping_error}")
                return None
            
            # Try to get job status from Redis
            status_key = f"job_status:{job_id}"
            logger.info(f"Checking Redis key: {status_key}")
            
            job_data = await self.redis_client.hgetall(status_key)
            logger.info(f"Retrieved job data: {job_data}")
            
            if not job_data:
                logger.warning(f"Job {job_id} not found in Redis")
                
                # Debug: Check all job keys
                all_job_keys = await self.redis_client.keys("job_status:*")
                logger.info(f"All job keys in Redis: {all_job_keys}")
                
                return None
              # Try to get job errors
            error_key = f"job_errors:{job_id}"
            error_list = await self.redis_client.lrange(error_key, 0, -1)
            errors = []
            for error_json in error_list:
                try:
                    error_data = json.loads(error_json)
                    errors.append(error_data.get('error', str(error_json)))
                except json.JSONDecodeError:
                    errors.append(str(error_json))
            
            # Create job response with processing time calculation
            started_at = datetime.fromisoformat(job_data['started_at']) if job_data.get('started_at') else None
            completed_at = datetime.fromisoformat(job_data['completed_at']) if job_data.get('completed_at') else None
            
            # Calculate processing time if job has started
            processing_time_seconds = None
            if started_at:
                end_time = completed_at if completed_at else datetime.now()
                processing_time_seconds = (end_time - started_at).total_seconds()
              # Parse enums safely
            try:
                job_status = JobStatus(job_data.get('status', JobStatus.QUEUED.value))
            except ValueError:
                logger.warning(f"Invalid job status: {job_data.get('status')}, defaulting to QUEUED")
                job_status = JobStatus.QUEUED
                
            try:
                workflow_type = WorkflowType(job_data.get('workflow_type', WorkflowType.COMPLETE_PIPELINE.value))
            except ValueError:
                logger.warning(f"Invalid workflow type: {job_data.get('workflow_type')}, defaulting to COMPLETE_PIPELINE")
                workflow_type = WorkflowType.COMPLETE_PIPELINE
              # Parse priority safely - handle both int and string values
            priority_value = job_data.get('priority', 'normal')
            if isinstance(priority_value, str):
                # Use the string value directly if it's valid
                priority = priority_value.lower()
                if priority not in ['low', 'normal', 'high', 'urgent']:
                    logger.warning(f"Invalid priority value: {priority_value}, defaulting to 'normal'")
                    priority = 'normal'
            else:
                # Convert integer to string
                priority_map = {
                    1: 'low',
                    2: 'normal',
                    3: 'high',
                    4: 'urgent'
                }
                priority = priority_map.get(priority_value, 'normal')
            
            return JobResponse(
                job_id=job_id,
                status=job_status,
                job_type=job_data.get('job_type', 'unknown'),
                workflow_type=workflow_type,
                priority=priority,
                created_at=datetime.fromisoformat(job_data.get('created_at', datetime.now().isoformat())),
                started_at=started_at,
                completed_at=completed_at,
                current_stage=job_data.get('current_stage'),
                errors=errors,
                max_retries=int(job_data.get('max_retries', 3)),
                assigned_agents=job_data.get('assigned_agents', '').split(',') if job_data.get('assigned_agents') else [],
                processing_time_seconds=processing_time_seconds
            )
            
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            return None

    async def cleanup(self):
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()

    async def execute_workflow(self, job_request: JobRequest) -> JobResponse:
        """
        Execute a complete workflow following the linear graph LR pattern:
        
        A[Supervisor Entry] --> B(OCR Agent) --> C(Supervisor) --> 
        D(NER Agent) --> E(Supervisor) --> F(Chunking Agent) --> 
        G(Supervisor) --> H(Graph Agent) --> I(Supervisor) --> 
        J(Insight Agent) --> K(Supervisor) --> L(Chat Agent) --> 
        M[Supervisor End]
        """
        job_id = str(uuid4())
        
        try:
            # Initialize workflow state
            initial_state = SupervisorState(
                job_id=job_id,
                workflow_type=job_request.workflow_type,
                patient_id=job_request.patient_id,
                document_ids=job_request.document_ids,
                priority=job_request.priority,
                current_agent_index=0,
                status=JobStatus.RUNNING
            )
            
            logger.info(f"Job {job_id}: Starting linear workflow execution - {job_request.workflow_type}")
            
            # Execute the workflow graph
            config = {"configurable": {"thread_id": job_id}}
            final_state = await self.workflow_graph.ainvoke(initial_state, config=config)
            
            # Create response based on final state
            job_response = JobResponse(
                job_id=job_id,
                status=final_state.get('status', JobStatus.COMPLETED),
                job_type=job_request.job_type,
                workflow_type=job_request.workflow_type,
                priority=job_request.priority,
                created_at=initial_state['created_at'],
                completed_at=datetime.now() if final_state.get('status') == JobStatus.COMPLETED else None,
                agent_results=final_state.get('agent_results', {}),
                errors=final_state.get('errors', []),
                max_retries=job_request.max_retries,
                assigned_agents=self.workflows[job_request.workflow_type].agent_sequence
            )
            
            logger.info(f"Job {job_id}: Workflow execution completed with status {job_response.status}")
            return job_response
            
        except Exception as e:
            logger.error(f"Job {job_id}: Workflow execution failed - {e}")
            
            # Return failed job response
            return JobResponse(
                job_id=job_id,
                status=JobStatus.FAILED,
                job_type=job_request.job_type,
                workflow_type=job_request.workflow_type,
                priority=job_request.priority,
                created_at=datetime.now(),
                errors=[f"Workflow execution error: {str(e)}"],
                max_retries=job_request.max_retries,
                assigned_agents=self.workflows.get(job_request.workflow_type, WorkflowDefinition()).agent_sequence
            )

    async def start_job(self, job_id: str) -> bool:
        """Mark a job as started and record start time."""
        try:
            status_key = f"job_status:{job_id}"
            
            # Check if job exists
            if not await self.redis_client.exists(status_key):
                logger.error(f"Cannot start job {job_id}: job not found")
                return False
            
            # Update job status to running with start time
            update_data = {
                'status': JobStatus.RUNNING.value,
                'started_at': datetime.now().isoformat(),
                'current_stage': 'Starting workflow execution'
            }
            
            await self.redis_client.hset(status_key, mapping=update_data)
            logger.info(f"Job {job_id} marked as started")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start job {job_id}: {e}")
            return False

    async def complete_job(self, job_id: str, results: Dict[str, Any] = None) -> bool:
        """Mark a job as completed and record completion time."""
        try:
            status_key = f"job_status:{job_id}"
            
            # Check if job exists
            if not await self.redis_client.exists(status_key):
                logger.error(f"Cannot complete job {job_id}: job not found")
                return False
            
            # Update job status to completed with completion time
            update_data = {
                'status': JobStatus.COMPLETED.value,
                'completed_at': datetime.now().isoformat(),
                'current_stage': 'Workflow completed successfully'
            }
            
            if results:
                update_data['results'] = json.dumps(results)
            
            await self.redis_client.hset(status_key, mapping=update_data)
            logger.info(f"Job {job_id} marked as completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete job {job_id}: {e}")
            return False

    async def fail_job(self, job_id: str, error_message: str) -> bool:
        """Mark a job as failed and record failure details."""
        try:
            status_key = f"job_status:{job_id}"
            
            # Check if job exists
            if not await self.redis_client.exists(status_key):
                logger.error(f"Cannot fail job {job_id}: job not found")
                return False
            
            # Update job status to failed
            update_data = {
                'status': JobStatus.FAILED.value,
                'completed_at': datetime.now().isoformat(),
                'current_stage': 'Workflow failed',
                'error_message': error_message
            }
            
            await self.redis_client.hset(status_key, mapping=update_data)
            
            # Also add to error log
            error_key = f"job_errors:{job_id}"
            error_data = {
                'error': error_message,
                'timestamp': datetime.now().isoformat(),
                'stage': 'workflow_execution'
            }
            await self.redis_client.lpush(error_key, json.dumps(error_data))
            
            logger.error(f"Job {job_id} marked as failed: {error_message}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark job {job_id} as failed: {e}")
            return False

    async def update_job_progress(self, job_id: str, progress: float, current_stage: str = None) -> bool:
        """Update job progress and current stage."""
        try:
            status_key = f"job_status:{job_id}"
            
            # Check if job exists
            if not await self.redis_client.exists(status_key):
                logger.error(f"Cannot update job {job_id}: job not found")
                return False
            
            update_data = {
                'progress': progress,
                'updated_at': datetime.now().isoformat()
            }
            
            if current_stage:
                update_data['current_stage'] = current_stage
            
            await self.redis_client.hset(status_key, mapping=update_data)
            logger.debug(f"Job {job_id} progress updated to {progress}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update job {job_id} progress: {e}")
            return False

    async def get_queue_status(self) -> QueueStatus:
        """Get current status of the job processing queue."""
        try:
            # Get counts for different job statuses
            total_jobs = 0
            queued_jobs = 0
            running_jobs = 0
            completed_jobs = 0
            failed_jobs = 0
            
            # Search for all job status keys
            status_keys = await self.redis_client.keys("job_status:*")
            
            for key in status_keys:
                job_data = await self.redis_client.hgetall(key)
                if job_data:
                    total_jobs += 1
                    status = job_data.get('status', '')
                    
                    if status == JobStatus.QUEUED.value:
                        queued_jobs += 1
                    elif status == JobStatus.RUNNING.value:
                        running_jobs += 1
                    elif status == JobStatus.COMPLETED.value:
                        completed_jobs += 1
                    elif status == JobStatus.FAILED.value:
                        failed_jobs += 1
            
            # Calculate average processing time from completed jobs
            total_processing_time = 0
            processing_time_count = 0
            
            for key in status_keys:
                job_data = await self.redis_client.hgetall(key)
                if job_data and job_data.get('status') == JobStatus.COMPLETED.value:
                    started_at = job_data.get('started_at')
                    completed_at = job_data.get('completed_at')
                    
                    if started_at and completed_at:
                        try:
                            start_time = datetime.fromisoformat(started_at)
                            end_time = datetime.fromisoformat(completed_at)
                            processing_time = (end_time - start_time).total_seconds()
                            total_processing_time += processing_time
                            processing_time_count += 1
                        except ValueError:
                            continue
            
            average_processing_time = (
                total_processing_time / processing_time_count 
                if processing_time_count > 0 else 0.0
            )
            
            # Estimate wait time based on queue length and average processing time
            estimated_wait_time = queued_jobs * average_processing_time if average_processing_time > 0 else 0.0
            
            # Mock system resource usage (in real implementation, get from system)
            import psutil
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            
            # Get active workflow types from running jobs
            active_workflows = []
            for key in status_keys:
                job_data = await self.redis_client.hgetall(key)
                if job_data and job_data.get('status') == JobStatus.RUNNING.value:
                    workflow_type = job_data.get('workflow_type')
                    if workflow_type and workflow_type not in active_workflows:
                        active_workflows.append(WorkflowType(workflow_type))
              # Determine queue health based on metrics
            if running_jobs > 10 or cpu_usage > 90 or memory_usage > 90:
                queue_health = "degraded"
            elif running_jobs > 5 or cpu_usage > 70 or memory_usage > 70:
                queue_health = "warning"
            else:
                queue_health = "healthy"
            
            return QueueStatus(
                total_jobs=total_jobs,
                queued_jobs=queued_jobs,
                running_jobs=running_jobs,
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                average_processing_time=round(average_processing_time, 2),
                queue_health=queue_health,
                estimated_wait_time=round(estimated_wait_time, 2),
                cpu_usage_percent=round(cpu_usage, 1),                memory_usage_percent=round(memory_usage, 1),
                active_workflows=active_workflows
            )
            
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            # Return default status on error
            return QueueStatus(
                total_jobs=0,
                queued_jobs=0,
                running_jobs=0,
                completed_jobs=0,
                failed_jobs=0,
                average_processing_time=0.0,
                queue_health="error",
                estimated_wait_time=0.0,
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                active_workflows=[]
            )

    async def retry_job(self, job_id: str) -> JobResponse:
        """Retry a failed job by creating a new job instance."""
        try:
            # Get original job data
            status_key = f"job_status:{job_id}"
            job_data = await self.redis_client.hgetall(status_key)
            
            if not job_data:
                raise ValueError(f"Original job {job_id} not found")
            
            # Create new job ID for retry
            new_job_id = str(uuid4())
            created_at = datetime.now()
            
            # Get retry count
            retry_count = int(job_data.get('retry_count', 0)) + 1
            max_retries = int(job_data.get('max_retries', 3))
            
            if retry_count > max_retries:
                raise ValueError(f"Job {job_id} has exceeded maximum retry attempts ({max_retries})")
            
            # Create new job data based on original
            new_job_data = {
                'job_id': new_job_id,
                'status': JobStatus.QUEUED.value,
                'job_type': job_data.get('job_type', 'document_processing'),
                'workflow_type': job_data.get('workflow_type', 'complete_pipeline'),
                'priority': job_data.get('priority', 'normal'),
                'created_at': created_at.isoformat(),
                'max_retries': max_retries,
                'retry_count': retry_count,
                'assigned_agents': job_data.get('assigned_agents', ''),
                'patient_id': job_data.get('patient_id', ''),
                'document_ids': job_data.get('document_ids', ''),
                'file_paths': job_data.get('file_paths', ''),
                'processing_config': job_data.get('processing_config', '{}'),
                'metadata': job_data.get('metadata', '{}'),
                'timeout_seconds': job_data.get('timeout_seconds', 300),
                'current_stage': f'Retry {retry_count} of {max_retries} - Job queued for processing',
                'original_job_id': job_id
            }
            
            # Store new job
            new_status_key = f"job_status:{new_job_id}"
            await self.redis_client.hset(new_status_key, mapping=new_job_data)
            
            # Add to processing queue
            priority = job_data.get('priority', 'normal')
            queue_key = f"job_queue:{priority}"
            await self.redis_client.lpush(queue_key, new_job_id)
            
            # Update original job to mark it as retried
            await self.redis_client.hset(status_key, mapping={
                'retry_job_id': new_job_id,
                'retried_at': datetime.now().isoformat()
            })
            
            # Create response
            job_response = JobResponse(
                job_id=new_job_id,
                status=JobStatus.QUEUED,
                job_type=JobType(job_data.get('job_type', 'document_processing')),
                workflow_type=WorkflowType(job_data.get('workflow_type', 'complete_pipeline')),
                priority=JobPriority(job_data.get('priority', 'normal')),
                created_at=created_at,
                max_retries=max_retries,
                retry_count=retry_count,
                assigned_agents=[AgentType(agent) for agent in job_data.get('assigned_agents', '').split(',') if agent],
                processing_time_seconds=None
            )
            
            logger.info(f"Job {job_id} retried as new job {new_job_id} (attempt {retry_count}/{max_retries})")
            return job_response
            
        except Exception as e:
            logger.error(f"Failed to retry job {job_id}: {e}")
            raise

    async def cancel_job(self, job_id: str) -> JobResponse:
        """Cancel a queued or running job."""
        try:
            # Get current job data
            status_key = f"job_status:{job_id}"
            job_data = await self.redis_client.hgetall(status_key)
            
            if not job_data:
                raise ValueError(f"Job {job_id} not found")
            
            current_status = job_data.get('status', '')
            
            if current_status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
                raise ValueError(f"Job {job_id} cannot be cancelled (current status: {current_status})")
            
            # Update job status to cancelled
            cancelled_at = datetime.now()
            update_data = {
                'status': JobStatus.CANCELLED.value,
                'completed_at': cancelled_at.isoformat(),
                'cancelled_at': cancelled_at.isoformat(),
                'current_stage': 'Job cancelled by user request'
            }
            
            await self.redis_client.hset(status_key, mapping=update_data)
            
            # Remove from processing queues if still queued
            if current_status == JobStatus.QUEUED.value:
                priority = job_data.get('priority', 'normal')
                queue_key = f"job_queue:{priority}"
                await self.redis_client.lrem(queue_key, 1, job_id)
              # Create response
            started_at = datetime.fromisoformat(job_data['started_at']) if job_data.get('started_at') else None
            processing_time = None
            if started_at:
                processing_time = (cancelled_at - started_at).total_seconds()
            
            job_response = JobResponse(
                job_id=job_id,
                status=JobStatus.CANCELLED,
                job_type=JobType(job_data.get('job_type', 'document_processing')),
                workflow_type=WorkflowType(job_data.get('workflow_type', 'complete_pipeline')),
                priority=JobPriority(job_data.get('priority', 'normal')),
                created_at=datetime.fromisoformat(job_data.get('created_at', datetime.now().isoformat())),
                started_at=started_at,
                completed_at=cancelled_at,
                cancelled_at=cancelled_at,
                max_retries=int(job_data.get('max_retries', 3)),
                retry_count=int(job_data.get('retry_count', 0)),
                assigned_agents=[AgentType(agent) for agent in job_data.get('assigned_agents', '').split(',') if agent],
                processing_time_seconds=processing_time
            )
            
            logger.info(f"Job {job_id} cancelled successfully")
            return job_response
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            raise


async def get_supervisor() -> LangChainSupervisor:
    """Get or create supervisor instance."""
    global supervisor_instance
    
    if supervisor_instance is None:
        logger.info("Creating new supervisor instance")
        supervisor_instance = LangChainSupervisor()
        await supervisor_instance.initialize()
        logger.info("Supervisor instance created and initialized")
    else:
        logger.info("Using existing supervisor instance")
        
        # Verify Redis connection is still active
        try:
            if supervisor_instance.redis_client:
                await supervisor_instance.redis_client.ping()
                logger.info("Existing supervisor Redis connection is active")
            else:
                logger.warning("Supervisor Redis client is None, reinitializing...")
                await supervisor_instance.initialize()
        except Exception as e:
            logger.error(f"Supervisor Redis connection failed: {e}, reinitializing...")
            supervisor_instance = LangChainSupervisor()
            await supervisor_instance.initialize()
    
    return supervisor_instance
