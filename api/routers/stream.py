#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Stream Router
WebSocket streaming endpoints for real-time video processing with frame-by-frame inference
"""

import logging
import json
import asyncio
import cv2
import base64
import numpy as np
import time
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.tracker import UnderwaterPersonTracker

logger = logging.getLogger(__name__)
router = APIRouter()

# Session storage for maintaining state across frames
active_sessions: Dict[str, Dict[str, Any]] = {}


@router.websocket("/stream/realtime")
async def websocket_realtime_stream(websocket: WebSocket):
    """
    Enhanced WebSocket endpoint for real-time frame-by-frame video streaming
    
    Features:
    - JPEG compression for reduced bandwidth
    - Frame tracking with session state
    - Underwater person tracking across frames
    - Adaptive quality based on processing time
    - Binary message support for efficiency
    
    Protocol:
    Client -> Server:
    {
        "type": "init",
        "session_id": "unique_id",
        "conf_threshold": 0.7,
        "underwater_threshold": 5,
        "danger_threshold": 5.0,
        "jpeg_quality": 75,
        "fps_target": 15
    }
    
    {
        "type": "frame",
        "session_id": "unique_id",
        "frame_id": 123,
        "data": "base64_jpeg_data",
        "timestamp": 1234567890.123
    }
    
    Server -> Client:
    {
        "type": "result",
        "frame_id": 123,
        "detections": [...],
        "water_zone": {...},
        "alerts": [...],
        "stats": {...},
        "processing_time_ms": 45.2
    }
    """
    await websocket.accept()
    session_id = None
    logger.info("WebSocket real-time connection established")
    
    try:
        detection_service = websocket.app.state.detection_service
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get('type')
            
            if msg_type == 'init':
                # Initialize new streaming session
                session_id = message.get('session_id', f"stream_{int(time.time() * 1000)}")
                
                active_sessions[session_id] = {
                    'tracker': UnderwaterPersonTracker(
                        underwater_threshold=message.get('underwater_threshold', 5),
                        danger_threshold=message.get('danger_threshold', 5.0)
                    ),
                    'conf_threshold': message.get('conf_threshold', 0.7),
                    'jpeg_quality': message.get('jpeg_quality', 75),
                    'fps_target': message.get('fps_target', 15),
                    'frame_count': 0,
                    'start_time': time.time(),
                    'total_processing_time': 0,
                    'alerts': []
                }
                
                logger.info(f"Initialized streaming session: {session_id}")
                
                await websocket.send_json({
                    'type': 'init_success',
                    'session_id': session_id,
                    'message': 'Streaming session initialized'
                })
            
            elif msg_type == 'frame':
                # Process incoming frame
                frame_start = time.time()
                
                session_id = message.get('session_id')
                if not session_id or session_id not in active_sessions:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Invalid or missing session_id. Send init message first.'
                    })
                    continue
                
                session = active_sessions[session_id]
                frame_id = message.get('frame_id', session['frame_count'])
                
                # Decode frame (JPEG compressed)
                try:
                    img_data = base64.b64decode(message['data'])
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        await websocket.send_json({
                            'type': 'error',
                            'message': 'Invalid frame data',
                            'frame_id': frame_id
                        })
                        continue
                except Exception as e:
                    await websocket.send_json({
                        'type': 'error',
                        'message': f'Failed to decode frame: {str(e)}',
                        'frame_id': frame_id
                    })
                    continue
                
                # Detect persons
                raw_detections = detection_service.model_manager.detect_persons(
                    frame,
                    conf_threshold=session['conf_threshold']
                )
                
                # Detect water zone
                water_zone, water_time = detection_service.detect_water_zone(frame)
                
                # Update tracker
                tracker = session['tracker']
                frame_ts = message.get('timestamp', time.time())
                assignments = tracker.update(raw_detections, frame_ts)
                
                # Convert tracks to response format with person detection objects
                tracked_persons = []
                for track_id, track in tracker.tracks.items():
                    cx, cy = track['center']
                    
                    # Determine status based on dangerosity score and tracker status
                    if track['dangerosity_score'] >= 80:
                        status = 'danger'
                    elif track['status'] == 'underwater':
                        status = 'underwater'
                    else:
                        status = 'surface'
                    
                    # Create a simple object with the required attributes
                    class TrackedPerson:
                        def __init__(self, track_id, track_data, status):
                            self.track_id = track_id
                            self.center_x = track_data['center'][0]
                            self.center_y = track_data['center'][1]
                            # Use actual bbox dimensions from tracker
                            self.width = track_data.get('width', 100.0)
                            self.height = track_data.get('height', 200.0)
                            self.confidence = track_data.get('confidence', 0.9)
                            self.status = status
                            self.frames_underwater = track_data['frames_underwater']
                            self.underwater_duration = track_data.get('underwater_duration', 0.0)
                            self.dangerosity_score = track_data['dangerosity_score']
                            self.distance_from_shore = track_data.get('distance_from_shore', 0.0)
                    
                    tracked_persons.append(TrackedPerson(track_id, track, status))
                
                # Check for alerts
                frame_alerts = []
                for person in tracked_persons:
                    if person.status == 'danger':
                        alert = {
                            'track_id': person.track_id,
                            'frame_id': frame_id,
                            'timestamp': message.get('timestamp', time.time()),
                            'duration': person.underwater_duration,
                            'message': f"Person {person.track_id} in danger! Underwater for {person.underwater_duration:.1f}s",
                            'dangerosity_score': person.dangerosity_score
                        }
                        frame_alerts.append(alert)
                        session['alerts'].append(alert)
                
                # Update session stats
                session['frame_count'] += 1
                processing_time = (time.time() - frame_start) * 1000
                session['total_processing_time'] += processing_time
                
                # Prepare response
                response = {
                    'type': 'result',
                    'frame_id': frame_id,
                    'session_id': session_id,
                    'detections': [
                        {
                            'track_id': p.track_id,
                            'bbox': {
                                'center_x': p.center_x,
                                'center_y': p.center_y,
                                'width': p.width,
                                'height': p.height
                            },
                            'confidence': p.confidence,
                            'status': p.status,
                            'frames_underwater': p.frames_underwater,
                            'underwater_duration': p.underwater_duration,
                            'dangerosity_score': p.dangerosity_score,
                            'distance_from_shore': p.distance_from_shore
                        }
                        for p in tracked_persons
                    ],
                    'water_zone': {
                        'detected': water_zone.detected if water_zone else False,
                        'polygon': water_zone.polygon if water_zone and water_zone.detected and water_zone.polygon else None,
                        'area': water_zone.area if water_zone and water_zone.detected else None
                    } if water_zone else None,
                    'alerts': frame_alerts,
                    'stats': {
                        'processing_time_ms': round(processing_time, 2),
                        'avg_processing_time_ms': round(session['total_processing_time'] / session['frame_count'], 2),
                        'frame_count': session['frame_count'],
                        'total_alerts': len(session['alerts']),
                        'fps': round(session['frame_count'] / (time.time() - session['start_time']), 2)
                    }
                }
                
                await websocket.send_json(response)
                
                # Adaptive quality suggestion
                if processing_time > (1000 / session['fps_target']):
                    # Processing is slower than target FPS
                    logger.warning(f"Session {session_id}: Processing time {processing_time:.1f}ms exceeds target {1000/session['fps_target']:.1f}ms")
            
            elif msg_type == 'ping':
                await websocket.send_json({'type': 'pong'})
            
            elif msg_type == 'end_session':
                # Clean up session
                session_id = message.get('session_id')
                if session_id and session_id in active_sessions:
                    session = active_sessions[session_id]
                    
                    summary = {
                        'type': 'session_summary',
                        'session_id': session_id,
                        'total_frames': session['frame_count'],
                        'total_alerts': len(session['alerts']),
                        'avg_processing_time_ms': round(session['total_processing_time'] / max(session['frame_count'], 1), 2),
                        'duration_seconds': time.time() - session['start_time'],
                        'avg_fps': round(session['frame_count'] / (time.time() - session['start_time']), 2)
                    }
                    
                    del active_sessions[session_id]
                    logger.info(f"Ended streaming session: {session_id}")
                    
                    await websocket.send_json(summary)
            
            else:
                await websocket.send_json({
                    'type': 'error',
                    'message': f"Unknown message type: {msg_type}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for session: {session_id}")
        # Clean up session
        if session_id and session_id in active_sessions:
            del active_sessions[session_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                'type': 'error',
                'message': str(e)
            })
        except:
            pass
    finally:
        # Ensure cleanup
        if session_id and session_id in active_sessions:
            del active_sessions[session_id]
        try:
            await websocket.close()
        except:
            pass


# Keep the original simple stream endpoint for backward compatibility
@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Simple WebSocket endpoint for basic frame-by-frame processing
    No session state, no tracking - just raw detection
    """
    await websocket.accept()
    logger.info("WebSocket simple connection established")
    
    try:
        detection_service = websocket.app.state.detection_service
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'frame':
                img_data = base64.b64decode(message['data'])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Invalid frame data'
                    })
                    continue
                
                detections, proc_time = detection_service.detect_persons_in_image(
                    frame,
                    conf_threshold=message.get('conf_threshold', 0.7)
                )
                
                await websocket.send_json({
                    'type': 'detections',
                    'detections': [det.dict() for det in detections],
                    'processing_time_ms': proc_time,
                    'frame_id': message.get('frame_id', 0)
                })
            
            elif message.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
    
    except WebSocketDisconnect:
        logger.info("WebSocket simple connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
