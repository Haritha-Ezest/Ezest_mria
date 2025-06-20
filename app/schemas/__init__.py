"""
Schemas package for MRIA data models.

This package contains all the Pydantic models for API request/response:
- ingestion: File upload and document handling schemas
- supervisor: Workflow and job management schemas
- ocr: Text extraction schemas
- ner: Medical entity recognition schemas
- chunking: Document chunking schemas
- graph: Knowledge graph schemas
- insights: Medical insights and analysis schemas
"""

__all__ = [
    "ingestion",
    "supervisor",
    "ocr", 
    "ner",
    "chunking",
    "graph",
    "insights"
]
