"""
Configuration pour l'application Neptune PyQt6
"""

# === Configuration des couleurs ===
COLORS = {
    'bg_dark': (45, 45, 45),
    'bg_light': (70, 70, 70),
    'primary': (0, 80, 243),
    'success': (0, 255, 0),
    'warning': (0, 255, 255),
    'danger': (0, 0, 255),
    'text_white': (255, 255, 255),
    'text_gray': (200, 200, 200),
    'border': (100, 100, 100),
    'water_zone': (0, 255, 0),
}

# === Configuration de la détection ===
DETECTION = {
    'conf_threshold': 0.55,
    'max_distance': 100,
    'max_disappeared': 300,
    'underwater_threshold': 15,
    'surface_threshold': 5,
}

# === Configuration des alertes ===
ALERTS = {
    'danger_threshold': 5,
    'alert_duration': 8.0,
    'popup_duration': 7.0,
}

# === Configuration audio ===
AUDIO = {
    'danger_message': "Alerte ! Baigneur en danger.",
    'test_message': "Test de l'alerte vocale. Système de surveillance aquatique opérationnel.",
    'language': 'fr',
    'slow_speech': True,
    'tld': 'fr',
}

# === Configuration de l'interface ===
UI = {
    'width': 1400,
    'height': 900,
    'control_panel_width': 350,
    'video_panel_min_width': 800,
    'video_panel_min_height': 600,
}
