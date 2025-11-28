#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Streamlit Real-Time Frontend
Real-time frame-by-frame video streaming with WebSocket
"""

import streamlit as st
import cv2
import numpy as np
import time
import base64
import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import websockets
from threading import Thread
import queue

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
WS_BASE_URL = os.getenv("WS_BASE_URL", "ws://localhost:8000/api/v1")

# Page configuration
st.set_page_config(
    page_title="Neptune - Real-Time Surveillance",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #2b2b2b;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .danger-alert {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        animation: blink 1s infinite;
        margin: 10px 0;
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
    .underwater-alert {
        background-color: #ff9800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .surface-status {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    h1, h2, h3 {
        color: #00D4FF !important;
    }
    .stat-card {
        background-color: #3b3b3b;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #555;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def encode_frame_jpeg(frame, quality=75):
    """Encode frame as JPEG and return base64 string"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return base64.b64encode(buffer).decode('utf-8')


def get_color_by_dangerosity(score: int) -> tuple:
    """Calculate BGR color based on dangerosity score (0-100)"""
    if score <= 20:
        r = int(144 * (score / 20.0))
        g = int(100 + 138 * (score / 20.0))
        b = r
        return (b, g, r)
    
    if score <= 40:
        ratio = (score - 20) / 20.0
        return (int(144 * (1 - ratio)), int(238 + 17 * ratio), int(144 + 111 * ratio))
    
    if score <= 60:
        ratio = (score - 40) / 20.0
        return (0, int(255 - 90 * ratio), 255)
    
    if score <= 80:
        ratio = (score - 60) / 20.0
        return (0, int(165 * (1 - ratio)), 255)
    
    ratio = (score - 80) / 20.0
    return (0, 0, int(255 - 116 * ratio))


def draw_detections(frame, detections, water_zone=None):
    """Draw detections on frame (matching app renderer style)"""
    frame_copy = frame.copy()
    
    # Draw water zone
    if water_zone and water_zone.get('detected') and water_zone.get('polygon'):
        pts = np.array(water_zone['polygon'], dtype=np.int32)
        cv2.polylines(frame_copy, [pts], True, (0, 255, 0), 3)
        for i, p in enumerate(pts):
            cv2.circle(frame_copy, tuple(p), 8, (0, 255, 0), -1)
            cv2.putText(frame_copy, f"{i+1}", (p[0]+10, p[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw detections
    for det in detections:
        bbox = det['bbox']
        cx = bbox['center_x']
        cy = bbox['center_y']
        w = bbox['width']
        h = bbox['height']
        
        # Calculate box corners
        x0 = int(cx - w/2)
        y0 = int(cy - h/2)
        x1 = int(cx + w/2)
        y1 = int(cy + h/2)
        
        # Get color based on dangerosity score
        score = det['dangerosity_score']
        color = get_color_by_dangerosity(score)
        
        # Determine if danger status
        status = det['status']
        is_danger = (status == 'danger')
        
        # Draw bounding box (thicker for danger)
        thickness = 4 if is_danger else 2
        cv2.rectangle(frame_copy, (x0, y0), (x1, y1), color, thickness)
        
        # Prepare label
        track_id = det['track_id']
        if status == 'underwater':
            label = f"ID:{track_id} (UNDERWATER) - Score:{score} | {det['underwater_duration']:.1f}s"
        else:
            label = f"ID:{track_id} - Score:{score}"
        
        # Draw label background
        sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame_copy, (x0, y0-35), (x0+sz[0]+10, y0-5), (0, 0, 0), -1)
        
        # Draw label text
        cv2.putText(frame_copy, label, (x0+5, y0-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw confidence below box
        conf_text = f"Conf:{det['confidence']:.2f}"
        cv2.putText(frame_copy, conf_text, (x0, y1+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame_copy


class RealtimeStreamProcessor:
    """Handles real-time video streaming via WebSocket"""
    
    def __init__(self, ws_url, session_id, conf_threshold, underwater_threshold, danger_threshold, jpeg_quality, fps_target):
        self.ws_url = ws_url
        self.session_id = session_id
        self.conf_threshold = conf_threshold
        self.underwater_threshold = underwater_threshold
        self.danger_threshold = danger_threshold
        self.jpeg_quality = jpeg_quality
        self.fps_target = fps_target
        
        self.running = False
        self.websocket = None
        self.result_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=5)
        
    async def connect(self):
        """Connect to WebSocket and initialize session"""
        self.websocket = await websockets.connect(self.ws_url)
        
        # Send initialization message
        init_msg = {
            'type': 'init',
            'session_id': self.session_id,
            'conf_threshold': self.conf_threshold,
            'underwater_threshold': self.underwater_threshold,
            'danger_threshold': self.danger_threshold,
            'jpeg_quality': self.jpeg_quality,
            'fps_target': self.fps_target
        }
        await self.websocket.send(json.dumps(init_msg))
        
        # Wait for init success
        response = await self.websocket.recv()
        result = json.loads(response)
        
        if result.get('type') == 'init_success':
            return True
        return False
    
    async def send_frame(self, frame, frame_id, timestamp):
        """Send frame to server"""
        frame_b64 = encode_frame_jpeg(frame, self.jpeg_quality)
        
        message = {
            'type': 'frame',
            'session_id': self.session_id,
            'frame_id': frame_id,
            'data': frame_b64,
            'timestamp': timestamp
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def receive_result(self):
        """Receive detection result from server"""
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def end_session(self):
        """End streaming session"""
        message = {
            'type': 'end_session',
            'session_id': self.session_id
        }
        await self.websocket.send(json.dumps(message))
        
        # Get summary
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def close(self):
        """Close connection"""
        if self.websocket:
            await self.websocket.close()


def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"stream_{int(time.time() * 1000)}"
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'total_alerts' not in st.session_state:
        st.session_state.total_alerts = 0
    if 'recent_alerts' not in st.session_state:
        st.session_state.recent_alerts = []
    if 'current_detections' not in st.session_state:
        st.session_state.current_detections = []
    if 'stats' not in st.session_state:
        st.session_state.stats = {}
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = 0
    
    # Header
    st.title("🌊 Neptune - Real-Time Aquatic Surveillance")
    st.markdown("**Frame-by-frame real-time detection with WebSocket streaming**")
    
    # Sidebar - Control Panel
    with st.sidebar:
        st.header("🎮 Control Panel")
        
        # Video Source
        st.subheader("📹 Video Source")
        video_source_type = st.radio("Source Type:", ["Camera", "Video File"])
        
        if video_source_type == "Camera":
            st.session_state.camera_index = st.number_input(
                "Camera Index",
                min_value=0,
                max_value=10,
                value=0,
                help="Camera device index (usually 0 for default camera)"
            )
            source = st.session_state.camera_index
        else:
            video_path_input = st.text_input(
                "Video File Path:",
                value="/home/lucasiglesia/Epitech/EIP/G-EIP-700-REN-7-1-eip-adrien.picot/homography/test.mp4",
                help="Enter full path to video file"
            )
            if Path(video_path_input).exists():
                st.session_state.video_path = video_path_input
                source = video_path_input
            else:
                st.warning("File does not exist")
                source = None
        
        st.divider()
        
        # Configuration
        st.subheader("⚙️ Detection Settings")
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        underwater_threshold = st.number_input(
            "Underwater Frames",
            min_value=1,
            max_value=30,
            value=5,
            help="Frames to consider person underwater"
        )
        
        danger_threshold = st.number_input(
            "Danger Time (seconds)",
            min_value=1.0,
            max_value=30.0,
            value=5.0,
            step=0.5
        )
        
        st.divider()
        
        # Streaming Settings
        st.subheader("🌐 Streaming Settings")
        
        fps_target = st.slider(
            "Target FPS",
            min_value=5,
            max_value=30,
            value=15,
            help="Target frames per second for processing"
        )
        
        jpeg_quality = st.slider(
            "JPEG Quality",
            min_value=50,
            max_value=95,
            value=75,
            help="JPEG compression quality (lower = faster, less quality)"
        )
        
        skip_frames = st.number_input(
            "Skip Frames",
            min_value=1,
            max_value=10,
            value=2,
            help="Process every Nth frame (1 = all frames)"
        )
        
        st.divider()
        
        # Control Buttons
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button("▶️ Start", type="primary", disabled=st.session_state.streaming or source is None)
        
        with col2:
            stop_button = st.button("⏹️ Stop", type="secondary", disabled=not st.session_state.streaming)
        
        if start_button:
            st.session_state.streaming = True
            st.session_state.session_id = f"stream_{int(time.time() * 1000)}"
            st.session_state.frame_count = 0
            st.session_state.total_alerts = 0
            st.session_state.recent_alerts = []
            st.rerun()
        
        if stop_button:
            st.session_state.streaming = False
            st.rerun()
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Live Statistics")
        
        if st.session_state.stats:
            st.metric("Frames Processed", st.session_state.stats.get('frame_count', 0))
            st.metric("Current FPS", f"{st.session_state.stats.get('fps', 0):.1f}")
            st.metric("Avg Processing Time", f"{st.session_state.stats.get('avg_processing_time_ms', 0):.1f} ms")
            st.metric("Total Alerts", st.session_state.stats.get('total_alerts', 0))
        else:
            st.info("Start streaming to see stats")
    
    # Main Content Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("📹 Live Video Feed")
        
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if st.session_state.streaming:
            # Real-time streaming
            asyncio.run(stream_video(
                source,
                video_placeholder,
                status_placeholder,
                conf_threshold,
                underwater_threshold,
                danger_threshold,
                jpeg_quality,
                fps_target,
                skip_frames
            ))
        else:
            video_placeholder.info("👆 Click Start to begin real-time surveillance")
            video_placeholder.image(
                "https://via.placeholder.com/800x450/2b2b2b/00D4FF?text=Neptune+Real-Time+Feed",
                use_container_width=True
            )
    
    with col2:
        st.header("🚨 Live Alerts")
        
        alerts_placeholder = st.empty()
        
        if st.session_state.recent_alerts:
            with alerts_placeholder.container():
                for alert in st.session_state.recent_alerts[-10:]:
                    st.markdown(f"""
                    <div class="danger-alert">
                    🚨 DANGER ALERT<br>
                    Person ID: {alert['track_id']}<br>
                    Duration: {alert['duration']:.1f}s<br>
                    Score: {alert['dangerosity_score']}/100<br>
                    {alert['message']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            alerts_placeholder.success("✅ No alerts")
        
        st.divider()
        
        st.header("👥 Current Detections")
        
        detections_placeholder = st.empty()
        
        if st.session_state.current_detections:
            with detections_placeholder.container():
                for det in st.session_state.current_detections:
                    status = det['status']
                    status_emoji = {'surface': '🟢', 'underwater': '🟠', 'danger': '🔴'}.get(status, '⚪')
                    
                    status_class = 'surface-status' if status == 'surface' else 'underwater-alert' if status == 'underwater' else 'danger-alert'
                    
                    st.markdown(f"""
                    <div class="{status_class}">
                    {status_emoji} Person {det['track_id']}<br>
                    Status: {status.upper()}<br>
                    Danger Score: {det['dangerosity_score']}/100<br>
                    Underwater: {det['underwater_duration']:.1f}s
                    </div>
                    """, unsafe_allow_html=True)
        else:
            detections_placeholder.info("No persons detected")


async def stream_video(source, video_placeholder, status_placeholder, conf_threshold, 
                       underwater_threshold, danger_threshold, jpeg_quality, fps_target, skip_frames):
    """Process video stream in real-time"""
    
    # Open video source
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        status_placeholder.error("❌ Cannot open video source")
        st.session_state.streaming = False
        return
    
    # Connect to WebSocket
    ws_url = f"{WS_BASE_URL}/stream/realtime"
    processor = RealtimeStreamProcessor(
        ws_url,
        st.session_state.session_id,
        conf_threshold,
        underwater_threshold,
        danger_threshold,
        jpeg_quality,
        fps_target
    )
    
    try:
        # Initialize connection
        status_placeholder.info("Connecting to server...")
        connected = await processor.connect()
        
        if not connected:
            status_placeholder.error("❌ Failed to connect to server")
            st.session_state.streaming = False
            return
        
        status_placeholder.success("✅ Connected - Streaming...")
        
        frame_id = 0
        frame_counter = 0
        start_time = time.time()
        
        while st.session_state.streaming:
            ret, frame = cap.read()
            
            if not ret:
                # End of video or error
                if isinstance(source, str):
                    # Loop video file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            frame_counter += 1
            
            # Skip frames for performance
            if frame_counter % skip_frames != 0:
                continue
            
            # Send frame to server
            timestamp = time.time()
            await processor.send_frame(frame, frame_id, timestamp)
            
            # Receive result
            result = await processor.receive_result()
            
            if result.get('type') == 'result':
                # Update session state
                st.session_state.current_detections = result['detections']
                st.session_state.stats = result['stats']
                
                # Handle alerts
                if result['alerts']:
                    st.session_state.recent_alerts.extend(result['alerts'])
                    st.session_state.total_alerts = len(st.session_state.recent_alerts)
                
                # Draw detections on frame
                annotated_frame = draw_detections(
                    frame,
                    result['detections'],
                    result.get('water_zone')
                )
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display
                video_placeholder.image(frame_rgb, use_container_width=True)
                
                # Status update
                stats = result['stats']
                status_placeholder.success(
                    f"✅ Streaming | Frame: {frame_id} | "
                    f"FPS: {stats['fps']:.1f} | "
                    f"Processing: {stats['processing_time_ms']:.1f}ms | "
                    f"Alerts: {stats['total_alerts']}"
                )
            
            frame_id += 1
            st.session_state.frame_count = frame_id
            
            # Control frame rate
            elapsed = time.time() - timestamp
            target_delay = 1.0 / fps_target
            if elapsed < target_delay:
                await asyncio.sleep(target_delay - elapsed)
        
        # End session
        summary = await processor.end_session()
        status_placeholder.info(f"Session ended: {summary.get('total_frames', 0)} frames processed")
        
    except Exception as e:
        status_placeholder.error(f"❌ Streaming error: {str(e)}")
    finally:
        cap.release()
        await processor.close()
        st.session_state.streaming = False


if __name__ == "__main__":
    main()
