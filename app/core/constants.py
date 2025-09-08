#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Core Constants
- Constantes globales de l'application
"""

import numpy as np

# ===== Constantes de la minimap =====
MAP_W_PX, MAP_H_PX = 400, 200
MIN_WATER_AREA_PX = 5_000

# ===== Rectangle de destination pour l'homographie =====
DST_RECT = np.array([[0, 0], [MAP_W_PX, 0], [MAP_W_PX, MAP_H_PX], [0, MAP_H_PX]], dtype=np.float32)

# ===== Couleurs (BGR) =====
SURFACE_COLORS = [(0, 255, 0), (0, 255, 128), (128, 255, 0), (0, 255, 255), (128, 255, 128)]
UNDERWATER_COLORS = [(255, 0, 0), (255, 128, 0), (255, 0, 128), (128, 0, 255), (255, 64, 64)]
DANGER_COLOR = (0, 0, 255)  # rouge

# ===== Configuration PyQt =====
QT_ENV_CONFIG = {
    "QT_NO_XDG_DESKTOP_PORTAL": "1",
    "QT_STYLE_OVERRIDE": "Fusion",
    "QT_ICON_THEME": "hicolor"
}
