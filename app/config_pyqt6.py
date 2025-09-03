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
    'conf_threshold': 0.7,      # Seuil de confiance pour la détection (MÊME QUE DEMO-5)
    'max_distance': 100,         # Distance maximale pour l'association des tracks
    'max_disappeared': 300,      # Frames avant suppression d'un track  
    'underwater_threshold': 5,   # Frames pour considérer une personne sous l'eau (RÉDUIT POUR TEST)
    'surface_threshold': 3,      # Frames pour considérer une personne en surface (RÉDUIT POUR TEST)
}

# === Configuration des alertes ===
ALERTS = {
    'danger_threshold': 5,       # Seuil de danger (secondes sous l'eau) - MÊME QUE DEMO-5
    'alert_duration': 8.0,       # Durée d'affichage des alertes (secondes)
    'popup_duration': 7.0,       # Durée du popup d'alerte
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
