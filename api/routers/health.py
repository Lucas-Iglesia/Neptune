#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Health Router
Health check and system status endpoints
"""

from fastapi import APIRouter, Request
from datetime import datetime

from config import API_VERSION
from models.responses import HealthStatus

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
async def health_check(request: Request):
    """
    Health check endpoint
    Returns API status and model information
    """
    detection_service = request.app.state.detection_service
    model_info = detection_service.get_model_info()
    
    return HealthStatus(
        status="healthy" if model_info['initialized'] else "degraded",
        version=API_VERSION,
        timestamp=datetime.now(),
        gpu_available=model_info.get('device', 'cpu') == 'cuda',
        gpu_info=model_info.get('device_info'),
        models_loaded={
            "person_detection": model_info.get('person_model', False),
            "water_detection": model_info.get('water_model', False)
        }
    )


@router.get("/health/models")
async def models_status(request: Request):
    """
    Detailed model status endpoint
    """
    detection_service = request.app.state.detection_service
    return detection_service.get_model_info()
