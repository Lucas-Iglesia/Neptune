#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune API Request Models
Pydantic models for API requests
"""

from pydantic import BaseModel, Field
from typing import Optional


class DetectionConfig(BaseModel):
    """Configuration for detection"""
    conf_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold")
    underwater_threshold: int = Field(default=5, ge=1, description="Frames to consider underwater")
    danger_threshold: float = Field(default=5.0, ge=0.0, description="Danger threshold in seconds")


class VideoProcessingRequest(BaseModel):
    """Request for video processing"""
    conf_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence threshold")
    analyze_water: bool = Field(default=True, description="Analyze water zone")
    track_persons: bool = Field(default=True, description="Track persons")
    detect_drowning: bool = Field(default=True, description="Detect drowning events")


class ImageDetectionRequest(BaseModel):
    """Request for single image detection"""
    conf_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence threshold")
    analyze_water: bool = Field(default=True, description="Analyze water zone")
    return_annotated: bool = Field(default=True, description="Return annotated image")
