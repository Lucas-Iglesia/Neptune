#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune API - Main Application
FastAPI application for aquatic surveillance with GPU-accelerated inference
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    API_VERSION, API_TITLE, API_DESCRIPTION,
    CORS_CONFIG, LOG_LEVEL, LOG_FORMAT
)
from routers import health, detection, video, stream
from services.detection_service import DetectionService

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Global services
detection_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    global detection_service
    
    # Startup
    logger.info("🚀 Starting Neptune API...")
    
    # Initialize detection service (loads models to GPU)
    detection_service = DetectionService()
    await detection_service.initialize()
    
    # Store in app state
    app.state.detection_service = detection_service
    
    logger.info("✅ Neptune API ready")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Neptune API...")
    if detection_service:
        await detection_service.cleanup()
    logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_CONFIG['allow_origins'],
    allow_credentials=CORS_CONFIG['allow_credentials'],
    allow_methods=CORS_CONFIG['allow_methods'],
    allow_headers=CORS_CONFIG['allow_headers'],
)

# Include routers
app.include_router(health.router, prefix=f"/api/{API_VERSION}", tags=["Health"])
app.include_router(detection.router, prefix=f"/api/{API_VERSION}", tags=["Detection"])
app.include_router(video.router, prefix=f"/api/{API_VERSION}", tags=["Video"])
app.include_router(stream.router, prefix=f"/api/{API_VERSION}", tags=["Stream"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Neptune Aquatic Surveillance API",
        "version": API_VERSION,
        "docs": f"/docs",
        "health": f"/api/{API_VERSION}/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )
