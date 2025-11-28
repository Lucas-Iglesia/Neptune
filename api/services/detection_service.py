#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Detection Service
GPU-optimized detection service for images and video frames
"""

import logging
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import DETECTION, GPU_CONFIG, WATER_MODEL_PATH
from detection.models import ModelManager
from detection.water import WaterDetector
from core.tracker import UnderwaterPersonTracker
from models.responses import PersonDetection, WaterZone, DetectionStatus

logger = logging.getLogger(__name__)


class DetectionService:
    """
    Detection service with GPU acceleration
    Handles person detection, water detection, and tracking
    """
    
    def __init__(self):
        self.model_manager = None
        self.water_detector = WaterDetector()
        self.conf_threshold = DETECTION['conf_threshold']
        self._initialized = False
    
    async def initialize(self):
        """
        Initialize the detection service
        Loads models to GPU if available
        """
        if self._initialized:
            return
        
        logger.info("Initializing Detection Service...")
        
        # Initialize model manager with GPU settings
        self.model_manager = ModelManager(
            use_gpu=GPU_CONFIG['use_gpu'],
            use_mixed_precision=GPU_CONFIG['use_mixed_precision']
        )
        
        # Load models
        success = self.model_manager.load_models(water_model_path=str(WATER_MODEL_PATH))
        
        if not success:
            logger.error("Failed to load AI models")
            raise RuntimeError("Failed to initialize detection models")
        
        # Warm up models (run dummy inference)
        await self._warmup_models()
        
        self._initialized = True
        logger.info("✅ Detection Service initialized")
    
    async def _warmup_models(self):
        """Warm up models with dummy inference to optimize GPU"""
        try:
            logger.info("Warming up models...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model_manager.detect_persons(dummy_frame, self.conf_threshold)
            logger.info("✅ Models warmed up")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Detection Service...")
        self.model_manager = None
        self._initialized = False
    
    def detect_persons_in_image(
        self, 
        image: np.ndarray, 
        conf_threshold: Optional[float] = None
    ) -> Tuple[List[PersonDetection], float]:
        """
        Detect persons in a single image
        
        Args:
            image: Image in BGR format
            conf_threshold: Confidence threshold (optional)
        
        Returns:
            Tuple of (detections list, processing time in ms)
        """
        if not self._initialized:
            raise RuntimeError("Detection service not initialized")
        
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        start_time = time.time()
        
        # Detect persons
        raw_detections = self.model_manager.detect_persons(image, conf)
        
        # Convert to response format
        detections = []
        for idx, det in enumerate(raw_detections):
            cx = float(det.xywh[0][0])
            cy = float(det.xywh[0][1])
            w = float(det.xywh[0][2])
            h = float(det.xywh[0][3])
            confidence = float(det.conf[0])
            
            detections.append(PersonDetection(
                track_id=idx,  # Temporary ID for single image
                center_x=cx,
                center_y=cy,
                width=w,
                height=h,
                confidence=confidence,
                status=DetectionStatus.SURFACE,  # Default for single image
                frames_underwater=0,
                underwater_duration=0.0,
                dangerosity_score=0,
                distance_from_shore=0.0
            ))
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return detections, processing_time
    
    def detect_water_zone(self, image: np.ndarray) -> Tuple[Optional[WaterZone], float]:
        """
        Detect water zone in image
        
        Args:
            image: Image in BGR format
        
        Returns:
            Tuple of (water zone info, processing time in ms)
        """
        if not self._initialized:
            raise RuntimeError("Detection service not initialized")
        
        if not self.model_manager.has_water_model():
            return None, 0.0
        
        start_time = time.time()
        
        # Compute water zone and homography
        success = self.water_detector.compute_water_and_homography(
            image,
            self.model_manager.nwsd
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if not success:
            return WaterZone(detected=False), processing_time
        
        # Extract water polygon if available
        polygon = None
        area = None
        
        if self.water_detector.src_quad_global is not None:
            polygon = self.water_detector.src_quad_global.tolist()
        
        if self.water_detector.water_mask_global is not None:
            area = float(np.sum(self.water_detector.water_mask_global))
        
        water_zone = WaterZone(
            detected=True,
            area=area,
            polygon=polygon
        )
        
        return water_zone, processing_time
    
    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        if not self._initialized or not self.model_manager:
            return {
                "initialized": False,
                "person_model": False,
                "water_model": False
            }
        
        return {
            "initialized": True,
            "person_model": self.model_manager.has_person_model(),
            "water_model": self.model_manager.has_water_model(),
            "device": self.model_manager.device,
            "device_info": self.model_manager.get_device_info()
        }
