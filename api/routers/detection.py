#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Detection Router
Image detection endpoints
"""

import io
import logging
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import Optional
import cv2
import numpy as np

from models.responses import ImageDetectionResponse
from models.requests import ImageDetectionRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect/image", response_model=ImageDetectionResponse)
async def detect_image(
    request: Request,
    image: UploadFile = File(..., description="Image file to analyze"),
    conf_threshold: Optional[float] = Form(None, description="Confidence threshold (0-1)"),
    analyze_water: bool = Form(True, description="Analyze water zone"),
    return_annotated: bool = Form(False, description="Return annotated image")
):
    """
    Detect persons in a single image
    
    Upload an image and get person detections with optional water zone analysis
    """
    detection_service = request.app.state.detection_service
    
    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    img_height, img_width = img.shape[:2]
    
    # Detect persons
    detections, person_time = detection_service.detect_persons_in_image(img, conf_threshold)
    
    # Detect water zone if requested
    water_zone = None
    water_time = 0.0
    if analyze_water:
        water_zone, water_time = detection_service.detect_water_zone(img)
    
    total_time = person_time + water_time
    
    logger.info(f"Detected {len(detections)} persons in image ({img_width}x{img_height}) in {total_time:.2f}ms")
    
    return ImageDetectionResponse(
        success=True,
        timestamp=datetime.now(),
        detections=detections,
        water_zone=water_zone,
        processing_time_ms=total_time,
        image_width=img_width,
        image_height=img_height
    )


@router.post("/detect/image/annotated")
async def detect_image_annotated(
    request: Request,
    image: UploadFile = File(..., description="Image file to analyze"),
    conf_threshold: Optional[float] = Form(None, description="Confidence threshold (0-1)"),
    analyze_water: bool = Form(True, description="Analyze water zone")
):
    """
    Detect persons and return annotated image
    
    Returns the image with bounding boxes drawn around detected persons
    """
    detection_service = request.app.state.detection_service
    
    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Detect persons
    detections, _ = detection_service.detect_persons_in_image(img, conf_threshold)
    
    # Draw detections
    for det in detections:
        x1 = int(det.center_x - det.width / 2)
        y1 = int(det.center_y - det.height / 2)
        x2 = int(det.center_x + det.width / 2)
        y2 = int(det.center_y + det.height / 2)
        
        # Color based on status
        color = (0, 255, 0)  # Green for surface
        if det.status == "underwater":
            color = (255, 165, 0)  # Orange
        elif det.status == "danger":
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, 
            f"ID:{det.track_id} {det.confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    # Detect and draw water zone
    if analyze_water:
        water_zone, _ = detection_service.detect_water_zone(img)
        if water_zone and water_zone.detected and water_zone.polygon:
            pts = np.array(water_zone.polygon, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)
    
    # Encode image to JPEG
    _, buffer = cv2.imencode('.jpg', img)
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=annotated_{image.filename}"}
    )
