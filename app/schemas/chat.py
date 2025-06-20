"""
Chat schemas for medical query and natural language interface.

This module defines Pydantic models for chat interactions, conversation management,
and natural language query processing for medical professionals.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of medical queries supported by the chat agent."""
    PATIENT_SUMMARY = "patient_summary"
    TREATMENT_COMPARISON = "treatment_comparison"
    MEDICATION_INTERACTION = "medication_interaction"
    LAB_INTERPRETATION = "lab_interpretation"
    POPULATION_HEALTH = "population_health"
    CLINICAL_DECISION_SUPPORT = "clinical_decision_support"
    TIMELINE_ANALYSIS = "timeline_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    GENERAL_MEDICAL = "general_medical"


class MessageRole(str, Enum):
    """Roles in chat conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class QueryComplexity(str, Enum):
    """Complexity levels for query processing."""
    SIMPLE = "simple"      # Single entity lookup
    MODERATE = "moderate"  # Multi-entity analysis
    COMPLEX = "complex"    # Population-level analytics
    ADVANCED = "advanced"  # Multi-modal cross-referencing


class ConversationStatus(str, Enum):
    """Status of conversation sessions."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"


class ContextType(str, Enum):
    """Types of conversation context."""
    PATIENT_FOCUSED = "patient_focused"
    CONDITION_FOCUSED = "condition_focused"
    MEDICATION_FOCUSED = "medication_focused"
    POPULATION_FOCUSED = "population_focused"
    GENERAL_MEDICAL = "general_medical"


# Base Models
class ChatMessage(BaseModel):
    """Individual chat message in conversation."""
    id: str = Field(..., description="Unique message identifier")
    role: MessageRole = Field(..., description="Message sender role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional message metadata")


class ConversationContext(BaseModel):
    """Context information for conversation."""
    patient_id: Optional[str] = Field(default=None, description="Active patient context")
    condition_focus: Optional[str] = Field(default=None, description="Primary medical condition in focus")
    medication_focus: Optional[List[str]] = Field(default=None, description="Medications in focus")
    time_period: Optional[str] = Field(default=None, description="Time period for analysis")
    context_type: ContextType = Field(default=ContextType.GENERAL_MEDICAL, description="Type of conversation context")
    clinical_speciality: Optional[str] = Field(default=None, description="Medical specialty context")
    additional_context: Optional[Dict[str, Any]] = Field(default=None, description="Additional contextual information")


class QueryIntent(BaseModel):
    """Detected intent from natural language query."""
    primary_intent: QueryType = Field(..., description="Primary query intent")
    secondary_intents: Optional[List[QueryType]] = Field(default=None, description="Secondary query intents")
    entities: Dict[str, List[str]] = Field(default_factory=dict, description="Extracted medical entities")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent detection confidence")
    complexity: QueryComplexity = Field(..., description="Query complexity level")
    requires_graph_query: bool = Field(default=False, description="Whether query requires graph database access")
    requires_population_data: bool = Field(default=False, description="Whether query requires population-level data")


# Request Models
class ChatQueryRequest(BaseModel):
    """Request for natural language query processing."""
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query from user")
    session_id: Optional[str] = Field(default=None, description="Conversation session identifier")
    context: Optional[ConversationContext] = Field(default=None, description="Conversation context")
    user_id: Optional[str] = Field(default=None, description="User identifier for personalization")
    include_sources: bool = Field(default=True, description="Include source references in response")
    max_results: Optional[int] = Field(default=10, ge=1, le=100, description="Maximum number of results to return")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Summarize this patient's diabetes progression over the last year",
                "session_id": "session_12345",
                "context": {
                    "patient_id": "patient_789",
                    "condition_focus": "Type 2 Diabetes",
                    "context_type": "patient_focused"
                },
                "include_sources": True,
                "max_results": 10
            }
        }


class ConversationContextRequest(BaseModel):
    """Request to set or update conversation context."""
    session_id: str = Field(..., description="Conversation session identifier")
    context: ConversationContext = Field(..., description="New conversation context")
    merge_with_existing: bool = Field(default=True, description="Merge with existing context or replace")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_12345",
                "context": {
                    "patient_id": "patient_789",
                    "condition_focus": "Hypertension",
                    "context_type": "patient_focused",
                    "clinical_speciality": "Cardiology"
                },
                "merge_with_existing": True
            }
        }


class ChatFeedbackRequest(BaseModel):
    """Request to provide feedback on chat responses."""
    session_id: str = Field(..., description="Conversation session identifier")
    message_id: str = Field(..., description="Message identifier being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent)")
    feedback_text: Optional[str] = Field(default=None, max_length=1000, description="Optional feedback text")
    feedback_category: Optional[str] = Field(default=None, description="Category of feedback")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_12345",
                "message_id": "msg_456",
                "rating": 4,
                "feedback_text": "Good summary but could include more recent lab results",
                "feedback_category": "accuracy"
            }
        }


# Response Models
class QueryAnalysis(BaseModel):
    """Analysis of user query before processing."""
    original_query: str = Field(..., description="Original user query")
    cleaned_query: str = Field(..., description="Cleaned and normalized query")
    detected_intent: QueryIntent = Field(..., description="Detected query intent and entities")
    suggested_refinements: Optional[List[str]] = Field(default=None, description="Suggested query refinements")
    processing_approach: str = Field(..., description="Approach for processing this query")


class MedicalEntity(BaseModel):
    """Medical entity extracted from query or data."""
    entity_type: str = Field(..., description="Type of medical entity")
    entity_value: str = Field(..., description="Entity value/name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    source_span: Optional[str] = Field(default=None, description="Original text span")
    knowledge_base_id: Optional[str] = Field(default=None, description="External knowledge base identifier")
    synonyms: Optional[List[str]] = Field(default=None, description="Known synonyms")


class QueryResult(BaseModel):
    """Individual result item from query processing."""
    result_type: str = Field(..., description="Type of result")
    content: Dict[str, Any] = Field(..., description="Result content")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to query")
    source_references: Optional[List[str]] = Field(default=None, description="Source document references")
    last_updated: Optional[datetime] = Field(default=None, description="Last update timestamp")


class ChatQueryResponse(BaseModel):
    """Response to natural language query."""
    message: str = Field(..., description="Response status message")
    session_id: str = Field(..., description="Conversation session identifier")
    response_text: str = Field(..., description="Natural language response")
    query_analysis: QueryAnalysis = Field(..., description="Analysis of the user query")
    results: List[QueryResult] = Field(default_factory=list, description="Structured query results")
    medical_entities: List[MedicalEntity] = Field(default_factory=list, description="Relevant medical entities")
    suggested_followups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall response confidence")
    processing_time: float = Field(..., description="Query processing time in seconds")
    sources_used: List[str] = Field(default_factory=list, description="Data sources used for response")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Query processed successfully",
                "session_id": "session_12345",
                "response_text": "Based on the patient's records over the last year, their diabetes management shows significant improvement...",
                "confidence": 0.92,
                "processing_time": 2.34,
                "suggested_followups": [
                    "What medications contributed most to this improvement?",
                    "How does this compare to similar patients?"
                ]
            }
        }


class ConversationSession(BaseModel):
    """Complete conversation session information."""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE, description="Session status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    last_active: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    context: Optional[ConversationContext] = Field(default=None, description="Current conversation context")
    message_count: int = Field(default=0, ge=0, description="Number of messages in session")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Session metadata")


class ConversationHistoryResponse(BaseModel):
    """Response containing conversation history."""
    message: str = Field(..., description="Response status message")
    session: ConversationSession = Field(..., description="Session information")
    messages: List[ChatMessage] = Field(default_factory=list, description="Conversation messages")
    total_messages: int = Field(..., ge=0, description="Total number of messages")
    context_changes: Optional[List[Dict[str, Any]]] = Field(default=None, description="Context change history")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Conversation history retrieved successfully",
                "session": {
                    "session_id": "session_12345",
                    "status": "active",
                    "message_count": 8,
                    "context": {
                        "patient_id": "patient_789",
                        "context_type": "patient_focused"
                    }
                },
                "total_messages": 8
            }
        }


class ConversationContextResponse(BaseModel):
    """Response to context setting operations."""
    message: str = Field(..., description="Response status message")
    session_id: str = Field(..., description="Session identifier")
    context: ConversationContext = Field(..., description="Updated conversation context")
    context_applied: bool = Field(default=True, description="Whether context was successfully applied")
    previous_context: Optional[ConversationContext] = Field(default=None, description="Previous context if changed")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Conversation context updated successfully",
                "session_id": "session_12345",
                "context": {
                    "patient_id": "patient_789",
                    "condition_focus": "Type 2 Diabetes",
                    "context_type": "patient_focused"
                },
                "context_applied": True
            }
        }


class ChatFeedbackResponse(BaseModel):
    """Response to feedback submission."""
    message: str = Field(..., description="Response status message")
    session_id: str = Field(..., description="Session identifier")
    feedback_id: str = Field(..., description="Unique feedback identifier")
    feedback_recorded: bool = Field(default=True, description="Whether feedback was successfully recorded")
    thanks_message: str = Field(..., description="Thank you message for user")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Feedback recorded successfully",
                "session_id": "session_12345",
                "feedback_id": "feedback_789",
                "feedback_recorded": True,
                "thanks_message": "Thank you for your feedback! It helps us improve our responses."
            }
        }


# Utility Models
class ChatMetrics(BaseModel):
    """Metrics for chat system performance."""
    total_sessions: int = Field(..., ge=0, description="Total number of chat sessions")
    active_sessions: int = Field(..., ge=0, description="Currently active sessions")
    average_session_length: float = Field(..., ge=0.0, description="Average messages per session")
    average_response_time: float = Field(..., ge=0.0, description="Average response time in seconds")
    query_types_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution of query types")
    user_satisfaction: Optional[float] = Field(default=None, ge=0.0, le=5.0, description="Average user satisfaction rating")


class ChatSystemStatus(BaseModel):
    """Current status of chat system."""
    status: str = Field(..., description="Overall system status")
    active_sessions: int = Field(..., ge=0, description="Number of active sessions")
    processing_queue_size: int = Field(..., ge=0, description="Size of processing queue")
    average_response_time: float = Field(..., ge=0.0, description="Current average response time")
    last_health_check: datetime = Field(default_factory=datetime.utcnow, description="Last health check timestamp")
    dependencies_status: Dict[str, str] = Field(default_factory=dict, description="Status of system dependencies")
