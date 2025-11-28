#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Session Manager
Manages video processing sessions
"""

import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, field

from config import SESSION_CONFIG
from models.responses import VideoSessionStatus

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Video processing session"""
    session_id: str
    status: VideoSessionStatus
    created_at: datetime
    video_path: str
    total_frames: int = 0
    processed_frames: int = 0
    fps: float = 30.0
    duration_seconds: float = 0.0
    results: Dict = field(default_factory=dict)
    last_accessed: datetime = field(default_factory=datetime.now)


class SessionManager:
    """
    Manager for video processing sessions
    Handles session lifecycle and cleanup
    """
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self._cleanup_task = None
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the session manager"""
        logger.info("Starting Session Manager...")
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop the session manager"""
        logger.info("Stopping Session Manager...")
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def create_session(self, video_path: str) -> str:
        """
        Create a new video processing session
        
        Args:
            video_path: Path to video file
        
        Returns:
            str: Session ID
        """
        async with self._lock:
            # Check if we've reached max sessions
            if len(self.sessions) >= SESSION_CONFIG['max_sessions']:
                # Remove oldest session
                oldest = min(self.sessions.values(), key=lambda s: s.last_accessed)
                await self.remove_session(oldest.session_id)
            
            session_id = str(uuid.uuid4())
            session = Session(
                session_id=session_id,
                status=VideoSessionStatus.INITIALIZING,
                created_at=datetime.now(),
                video_path=video_path
            )
            
            self.sessions[session_id] = session
            logger.info(f"Created session {session_id}")
            
            return session_id
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID
        
        Args:
            session_id: Session ID
        
        Returns:
            Session or None if not found
        """
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_accessed = datetime.now()
            return session
    
    async def update_session(
        self,
        session_id: str,
        status: Optional[VideoSessionStatus] = None,
        processed_frames: Optional[int] = None,
        results: Optional[Dict] = None
    ):
        """
        Update session information
        
        Args:
            session_id: Session ID
            status: New status (optional)
            processed_frames: Processed frame count (optional)
            results: Processing results (optional)
        """
        async with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            if status:
                session.status = status
            
            if processed_frames is not None:
                session.processed_frames = processed_frames
            
            if results:
                session.results = results
            
            session.last_accessed = datetime.now()
    
    async def remove_session(self, session_id: str):
        """
        Remove a session
        
        Args:
            session_id: Session ID
        """
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Removed session {session_id}")
    
    async def list_sessions(self) -> Dict[str, Session]:
        """List all sessions"""
        async with self._lock:
            return dict(self.sessions)
    
    async def _cleanup_loop(self):
        """Background task to cleanup expired sessions"""
        while True:
            try:
                await asyncio.sleep(SESSION_CONFIG['cleanup_interval'])
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        timeout = timedelta(seconds=SESSION_CONFIG['session_timeout'])
        
        async with self._lock:
            expired = [
                sid for sid, session in self.sessions.items()
                if now - session.last_accessed > timeout
            ]
            
            for sid in expired:
                del self.sessions[sid]
                logger.info(f"Cleaned up expired session {sid}")
