#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune API Models Package
"""

from .requests import DetectionConfig, VideoProcessingRequest, ImageDetectionRequest
from .responses import (
    DetectionStatus, PersonDetection, WaterZone, ImageDetectionResponse,
    VideoSessionStatus, VideoSessionInfo, VideoProcessingResult,
    HealthStatus, ErrorResponse
)

__all__ = [
    'DetectionConfig', 'VideoProcessingRequest', 'ImageDetectionRequest',
    'DetectionStatus', 'PersonDetection', 'WaterZone', 'ImageDetectionResponse',
    'VideoSessionStatus', 'VideoSessionInfo', 'VideoProcessingResult',
    'HealthStatus', 'ErrorResponse'
]
