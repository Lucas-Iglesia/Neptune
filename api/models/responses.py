#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune API Response Models
Pydantic models for API responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DetectionStatus(str, Enum):
    """Detection status enum"""
    SURFACE = "surface"
    UNDERWATER = "underwater"
    DANGER = "danger"


class PersonDetection(BaseModel):
    """Single person detection"""
    track_id: int = Field(description="Track ID")
    center_x: float = Field(description="Center X coordinate")
    center_y: float = Field(description="Center Y coordinate")
    width: float = Field(description="Bounding box width")
    height: float = Field(description="Bounding box height")
    confidence: float = Field(description="Detection confidence")
    status: DetectionStatus = Field(description="Person status")
    frames_underwater: int = Field(description="Frames underwater")
    underwater_duration: float = Field(description="Underwater duration in seconds")
    dangerosity_score: int = Field(description="Danger score (0-100)")
    distance_from_shore: float = Field(description="Distance from shore (0-1)")


class WaterZone(BaseModel):
    """Water zone information"""
    detected: bool = Field(description="Water zone detected")
    area: Optional[float] = Field(default=None, description="Water area in pixels")
    polygon: Optional[List[List[float]]] = Field(default=None, description="Water zone polygon")


class ImageDetectionResponse(BaseModel):
    """Response for image detection"""
    success: bool = Field(description="Detection success")
    timestamp: datetime = Field(description="Detection timestamp")
    detections: List[PersonDetection] = Field(description="List of person detections")
    water_zone: Optional[WaterZone] = Field(default=None, description="Water zone information")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    image_width: int = Field(description="Image width")
    image_height: int = Field(description="Image height")


class VideoSessionStatus(str, Enum):
    """Video session status enum"""
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoSessionInfo(BaseModel):
    """Video session information"""
    session_id: str = Field(description="Session ID")
    status: VideoSessionStatus = Field(description="Session status")
    created_at: datetime = Field(description="Creation timestamp")
    total_frames: int = Field(description="Total frames in video")
    processed_frames: int = Field(description="Frames processed")
    fps: float = Field(description="Video FPS")
    duration_seconds: float = Field(description="Video duration")


class VideoProcessingResult(BaseModel):
    """Video processing results"""
    session_id: str = Field(description="Session ID")
    status: VideoSessionStatus = Field(description="Processing status")
    total_frames: int = Field(description="Total frames")
    processed_frames: int = Field(description="Processed frames")
    detections_per_frame: Dict[int, List[PersonDetection]] = Field(description="Detections by frame")
    alerts: List[Dict[str, Any]] = Field(description="Drowning alerts")
    water_zone: Optional[WaterZone] = Field(default=None, description="Water zone")
    processing_time_seconds: float = Field(description="Total processing time")


class HealthStatus(BaseModel):
    """Health check response"""
    status: str = Field(description="API status")
    version: str = Field(description="API version")
    timestamp: datetime = Field(description="Check timestamp")
    gpu_available: bool = Field(description="GPU availability")
    gpu_info: Optional[Dict[str, Any]] = Field(default=None, description="GPU information")
    models_loaded: Dict[str, bool] = Field(description="Model loading status")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(description="Error type")
    detail: str = Field(description="Error detail")
    timestamp: datetime = Field(description="Error timestamp")
