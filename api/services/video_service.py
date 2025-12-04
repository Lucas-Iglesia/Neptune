#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Video Processing Service
Processes video files with person tracking and drowning detection
"""

import logging
import time
import cv2
import asyncio
from typing import Dict, List, Optional
from pathlib import Path

from config import DETECTION
from services.detection_service import DetectionService
from core.tracker import UnderwaterPersonTracker
from models.responses import PersonDetection, DetectionStatus, VideoProcessingResult, VideoSessionStatus

logger = logging.getLogger(__name__)


class VideoProcessingService:
    """
    Video processing service
    Handles frame-by-frame video processing with tracking
    """
    
    def __init__(self, detection_service: DetectionService):
        self.detection_service = detection_service
    
    async def process_video(
        self,
        video_path: str,
        conf_threshold: Optional[float] = None,
        progress_callback=None
    ) -> VideoProcessingResult:
        """
        Process a video file with person detection and tracking
        
        Args:
            video_path: Path to video file
            conf_threshold: Detection confidence threshold
            progress_callback: Async callback for progress updates
        
        Returns:
            VideoProcessingResult with detections and alerts
        """
        logger.info(f"Processing video: {video_path}")
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration = total_frames / fps
            
            logger.info(f"Video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
            
            # Initialize tracker
            tracker = UnderwaterPersonTracker(
                max_distance=DETECTION['max_distance'],
                max_disappeared=DETECTION['max_disappeared']
            )
            
            # Process video
            detections_per_frame = {}
            alerts = []
            processed_frames = 0
            start_time = time.time()
            
            conf = conf_threshold if conf_threshold is not None else DETECTION['conf_threshold']
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx = processed_frames
                frame_ts = frame_idx / fps
                
                # Detect persons in frame
                raw_detections = self.detection_service.model_manager.detect_persons(frame, conf)
                
                # Update tracker
                assignments = tracker.update(raw_detections, frame_ts)
                
                # Convert tracks to response format
                frame_detections = []
                for track_id, track in tracker.tracks.items():
                    cx, cy = track['center']
                    
                    # Determine status
                    if track['dangerosity_score'] >= 80:
                        status = DetectionStatus.DANGER
                    elif track['status'] == 'underwater':
                        status = DetectionStatus.UNDERWATER
                    else:
                        status = DetectionStatus.SURFACE
                    
                    detection = PersonDetection(
                        track_id=track_id,
                        center_x=cx,
                        center_y=cy,
                        width=100,  # Approximate
                        height=180,
                        confidence=0.9,
                        status=status,
                        frames_underwater=track['frames_underwater'],
                        underwater_duration=track['underwater_duration'],
                        dangerosity_score=track['dangerosity_score'],
                        distance_from_shore=track['distance_from_shore']
                    )
                    frame_detections.append(detection)
                    
                    # Check for alerts
                    if track['danger_alert_sent'] and not any(a['track_id'] == track_id for a in alerts):
                        alerts.append({
                            'track_id': track_id,
                            'frame': frame_idx,
                            'timestamp': frame_ts,
                            'duration': track['underwater_duration'],
                            'message': f"Danger: Person {track_id} underwater for {track['underwater_duration']:.1f}s"
                        })
                
                detections_per_frame[frame_idx] = frame_detections
                processed_frames += 1
                
                # Progress callback
                if progress_callback and processed_frames % 30 == 0:
                    await progress_callback(processed_frames, total_frames)
                
                # Yield to event loop periodically
                if processed_frames % 10 == 0:
                    await asyncio.sleep(0)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Processed {processed_frames} frames in {processing_time:.2f}s ({processed_frames/processing_time:.1f} FPS)")
            logger.info(f"Found {len(alerts)} danger alerts")
            
            return VideoProcessingResult(
                session_id="",  # Will be set by caller
                status=VideoSessionStatus.COMPLETED,
                total_frames=total_frames,
                processed_frames=processed_frames,
                detections_per_frame=detections_per_frame,
                alerts=alerts,
                water_zone=None,  # Can be added if needed
                processing_time_seconds=processing_time
            )
            
        finally:
            cap.release()
    
    async def analyze_water_zone(self, video_path: str):
        """
        Analyze water zone from first frame of video
        
        Args:
            video_path: Path to video file
        
        Returns:
            WaterZone information
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        try:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Failed to read first frame")
            
            water_zone, _ = self.detection_service.detect_water_zone(frame)
            return water_zone
        finally:
            cap.release()
