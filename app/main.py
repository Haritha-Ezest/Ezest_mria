"""
MRIA (Multi-stage Retrieval and Intelligence Architecture) FastAPI Application

This module serves as the main entry point for the MRIA monolith application,
which handles document ingestion, processing, and graph-based knowledge extraction.

The application follows a multi-stage pipeline:
1. Document ingestion and file handling
2. OCR processing for image and PDF content
3.Named Entity Recognition (NER) for content analysis
4. Document chunking for vector storage
5. Graph database integration for knowledge representation
6. Supervisor coordination for workflow management

Author: MRIA Development Team
Version: 0.1.0
Date: June 2025
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.routers import (
    ingestion,
    supervisor,
    ocr,
    ner,
    chunking,
    graph,
    insights,
    chat
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Log request details and response status."""
        start_time = request.state.start_time = time.time()
        
        # Log incoming request
        logger.info(
            f"Incoming request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Time: {process_time:.4f}s"
            )
              # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as exc:
            # Log errors
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- Error: {str(exc)} - Time: {process_time:.4f}s"
            )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events.
    
    Handles startup and shutdown operations for the MRIA application,
    including database connections, cache initialization, and cleanup.
    """
    # Startup operations
    logger.info("Starting MRIA application...")
    
    # Initialize supervisor service
    try:
        from app.services.supervisor import get_supervisor
        supervisor = await get_supervisor()
        logger.info("Supervisor service initialized successfully")
    except Exception as exc:
        logger.error(f"Failed to initialize supervisor service: {exc}")
        # Don't fail startup, but log the error
    
    try:
        logger.info("Application startup completed successfully")
        yield
    except Exception as exc:
        logger.error(f"Application startup failed: {exc}")
        raise
    finally:
        # Shutdown operations
        logger.info("Shutting down MRIA application...")
        
        # Cleanup supervisor
        try:
            from app.services.supervisor import supervisor_instance
            if supervisor_instance:
                await supervisor_instance.cleanup()
                logger.info("Supervisor cleanup completed")
        except Exception as exc:
            logger.error(f"Supervisor cleanup failed: {exc}")
        
        logger.info("Application shutdown completed")


# Create FastAPI application instance
app = FastAPI(
    title="MRIA Monolith",
    version="0.1.0",
    description=(
        "Multi-stage Retrieval and Intelligence Architecture (MRIA) - "
        "A comprehensive document processing and knowledge extraction system. "
        "Handles document ingestion, OCR processing, NER analysis, content chunking, "
        "and graph-based knowledge representation in a unified pipeline."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
)

# Add custom logging middleware
app.add_middleware(LoggingMiddleware)


# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions with consistent error response format."""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail} for {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "status_code": exc.status_code,
                "message": exc.detail,
                "path": str(request.url.path),
                "method": request.method
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors with detailed error information."""
    logger.error(f"Validation error for {request.url}: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "status_code": 422,
                "message": "Request validation failed",
                "details": exc.errors(),
                "path": str(request.url.path),
                "method": request.method
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with generic error response."""
    logger.error(f"Unexpected error for {request.url}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_server_error",
                "status_code": 500,
                "message": "An unexpected error occurred",
                "path": str(request.url.path),
                "method": request.method
            }
        }
    )


# Health check endpoints
@app.get("/health", tags=["system"])
async def health_check() -> Dict[str, Any]:
    """
    Perform basic health check of the application.
    
    Returns:
        Dict containing health status and basic system information
    """
    return {
        "status": "healthy",
        "service": "MRIA Monolith",
        "version": "0.1.0",
        "timestamp": time.time()
    }


@app.get("/health/detailed", tags=["system"])
async def detailed_health_check() -> Dict[str, Any]:
    """
    Perform detailed health check including service dependencies.
    
    Returns:
        Dict containing detailed health status of all components    """
    # This would typically check database connections, external services, etc.
    return {
        "status": "healthy",
        "service": "MRIA Monolith",
        "version": "0.1.0",
        "timestamp": time.time(),
        "components": {
            "database": "healthy",  # Placeholder - implement actual checks
            "vector_store": "healthy",
            "ocr_service": "healthy",
            "ner_service": "healthy",
            "graph_service": "healthy"
        }
    }


# Root endpoint
@app.get("/", tags=["system"])
async def root() -> Dict[str, str]:
    """
    Root endpoint providing basic API information.
    
    Returns:
        Dict containing welcome message and API documentation links
    """
    return {
        "message": "Welcome to MRIA (Multi-stage Retrieval and Intelligence Architecture)",
        "version": "0.1.0",
        "documentation": "/docs",
        "health_check": "/health"
    }


# Register all routers with their respective prefixes and tags
app.include_router(
    ingestion.router, 
    prefix="/ingest", 
    tags=["ingestion"]
)

app.include_router(
    supervisor.router, 
    prefix="/supervisor", 
    tags=["supervisor"]
)

app.include_router(
    ocr.router, 
    prefix="/ocr", 
    tags=["ocr"]
)

app.include_router(
    ner.router, 
    prefix="/ner", 
    tags=["ner"]
)

app.include_router(
    chunking.router, 
    prefix="/chunk", 
    tags=["chunking"]
)

app.include_router(
    graph.router, 
    prefix="/graph", 
    tags=["graph"]
)

app.include_router(
    insights.router, 
    prefix="/insights", 
    tags=["insights"]
)

app.include_router(
    chat.router,
    prefix="/chat",
    tags=["chat"]
)


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "production") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
