#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Video Router
Video processing endpoints
"""

import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, BackgroundTasks, Form
from typing import Optional

from config import UPLOAD_CONFIG
from models.responses import VideoSessionInfo, VideoSessionStatus, VideoProcessingResult
from models.requests import VideoProcessingRequest
from utils.session_manager import SessionManager
from services.video_service import VideoProcessingService

logger = logging.getLogger(__name__)
router = APIRouter()

# Global session manager (will be initialized in main.py)
session_manager = SessionManager()


async def process_video_task(
    session_id: str,
    video_path: str,
    detection_service,
    conf_threshold: Optional[float]
):
    """Background task for video processing"""
    try:
        logger.info(f"Starting video processing for session {session_id}")
        
        await session_manager.update_session(session_id, status=VideoSessionStatus.PROCESSING)
        
        video_service = VideoProcessingService(detection_service)
        
        # Progress callback
        async def progress_callback(processed, total):
            await session_manager.update_session(
                session_id,
                processed_frames=processed
            )
        
        # Process video
        result = await video_service.process_video(
            video_path,
            conf_threshold=conf_threshold,
            progress_callback=progress_callback
        )
        
        result.session_id = session_id
        
        # Update session with results
        await session_manager.update_session(
            session_id,
            status=VideoSessionStatus.COMPLETED,
            processed_frames=result.processed_frames,
            results=result.dict()
        )
        
        logger.info(f"Video processing completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error processing video for session {session_id}: {e}", exc_info=True)
        await session_manager.update_session(session_id, status=VideoSessionStatus.FAILED)


@router.post("/video/upload", response_model=VideoSessionInfo)
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file to process"),
    conf_threshold: Optional[float] = Form(None, description="Confidence threshold (0-1)"),
    analyze_water: bool = Form(True, description="Analyze water zone"),
    track_persons: bool = Form(True, description="Track persons"),
    detect_drowning: bool = Form(True, description="Detect drowning events")
):
    """
    Upload a video for processing
    
    Returns a session ID that can be used to check status and retrieve results
    """
    # Check file extension
    file_ext = Path(video.filename).suffix.lower()
    if file_ext not in UPLOAD_CONFIG['allowed_extensions']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {UPLOAD_CONFIG['allowed_extensions']}"
        )
    
    # Create session
    session_id = await session_manager.create_session(video.filename)
    
    # Save uploaded file
    upload_path = UPLOAD_CONFIG['upload_dir'] / f"{session_id}{file_ext}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        await session_manager.remove_session(session_id)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")
    
    # Get video info
    import cv2
    cap = cv2.VideoCapture(str(upload_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps
    cap.release()
    
    # Update session info
    session = await session_manager.get_session(session_id)
    session.total_frames = total_frames
    session.fps = fps
    session.duration_seconds = duration
    session.video_path = str(upload_path)
    
    # Start background processing
    detection_service = request.app.state.detection_service
    background_tasks.add_task(
        process_video_task,
        session_id,
        str(upload_path),
        detection_service,
        conf_threshold
    )
    
    logger.info(f"Video uploaded: {video.filename} -> session {session_id}")
    
    return VideoSessionInfo(
        session_id=session_id,
        status=VideoSessionStatus.INITIALIZING,
        created_at=session.created_at,
        total_frames=total_frames,
        processed_frames=0,
        fps=fps,
        duration_seconds=duration
    )


@router.get("/video/session/{session_id}", response_model=VideoSessionInfo)
async def get_session_status(session_id: str):
    """
    Get video processing session status
    """
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return VideoSessionInfo(
        session_id=session.session_id,
        status=session.status,
        created_at=session.created_at,
        total_frames=session.total_frames,
        processed_frames=session.processed_frames,
        fps=session.fps,
        duration_seconds=session.duration_seconds
    )


@router.get("/video/session/{session_id}/results", response_model=VideoProcessingResult)
async def get_session_results(session_id: str):
    """
    Get video processing results
    
    Only available when processing is completed
    """
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != VideoSessionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Processing not completed. Current status: {session.status}"
        )
    
    return VideoProcessingResult(**session.results)


@router.delete("/video/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a video processing session and cleanup files
    """
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete video file
    try:
        Path(session.video_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to delete video file: {e}")
    
    # Remove session
    await session_manager.remove_session(session_id)
    
    return {"message": "Session deleted", "session_id": session_id}
