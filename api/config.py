#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune API Configuration
"""

import os
from pathlib import Path

# === API Configuration ===
API_VERSION = "v1"
API_TITLE = "Neptune Aquatic Surveillance API"
API_DESCRIPTION = """
API for real-time aquatic surveillance with person detection, 
tracking, and drowning alert system using AI models.

Features:
- Image detection with GPU acceleration
- Video processing with person tracking
- Real-time WebSocket streaming
- Water zone detection and homography
- Drowning alert system
"""

# === Paths ===
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
WATER_MODEL_PATH = MODEL_DIR / "nwd-v2.pt"

# === Detection Configuration ===
DETECTION = {
    'conf_threshold': 0.7,      # Confidence threshold for detection
    'max_distance': 100,        # Max distance for track association
    'max_disappeared': 300,     # Frames before removing a track
    'underwater_threshold': 5,  # Frames to consider person underwater
    'surface_threshold': 3,     # Frames to consider person on surface
}

# === Alert Configuration ===
ALERTS = {
    'danger_threshold': 5,      # Danger threshold (seconds underwater)
    'alert_duration': 8.0,      # Alert display duration (seconds)
}

# === GPU Configuration ===
GPU_CONFIG = {
    'use_gpu': True,                    # Try to use GPU if available
    'use_mixed_precision': True,        # Use FP16 for faster inference
    'batch_size': 1,                    # Batch size for inference
    'num_workers': 2,                   # Number of workers for data loading
}

# === Session Configuration ===
SESSION_CONFIG = {
    'max_sessions': 10,                 # Maximum concurrent sessions
    'session_timeout': 3600,            # Session timeout in seconds (1 hour)
    'cleanup_interval': 300,            # Cleanup interval in seconds (5 min)
}

# === Upload Configuration ===
UPLOAD_CONFIG = {
    'max_file_size': 500 * 1024 * 1024,  # 500 MB max file size
    'allowed_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.jpg', '.jpeg', '.png'],
    'upload_dir': BASE_DIR / 'uploads',
}

# === CORS Configuration ===
CORS_CONFIG = {
    'allow_origins': os.getenv('CORS_ORIGINS', '*').split(','),
    'allow_credentials': True,
    'allow_methods': ['*'],
    'allow_headers': ['*'],
}

# === Logging Configuration ===
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create necessary directories
UPLOAD_CONFIG['upload_dir'].mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
