#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Core Constants (API version)
"""

import numpy as np

# ===== Minimap constants =====
MAP_W_PX, MAP_H_PX = 400, 200
MIN_WATER_AREA_PX = 5_000

# ===== Destination rectangle for homography =====
DST_RECT = np.array([[0, 0], [MAP_W_PX, 0], [MAP_W_PX, MAP_H_PX], [0, MAP_H_PX]], dtype=np.float32)

# ===== Colors (BGR) =====
SURFACE_COLORS = [(0, 255, 0), (0, 255, 128), (128, 255, 0), (0, 255, 255), (128, 255, 128)]
UNDERWATER_COLORS = [(255, 0, 0), (255, 128, 0), (255, 0, 128), (128, 0, 255), (255, 64, 64)]
DANGER_COLOR = (0, 0, 255)  # red
