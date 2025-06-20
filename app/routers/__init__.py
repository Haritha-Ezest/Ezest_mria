"""
Routers package for MRIA API endpoints.

This package contains all the FastAPI routers for different services:
- ingestion: Document upload and file handling
- supervisor: Workflow orchestration and job management
- ocr: Text extraction from images and PDFs
- ner: Medical entity recognition
- chunking: Document chunking and timeline structuring
- graph: Knowledge graph operations
- insights: Medical insights and pattern recognition
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
