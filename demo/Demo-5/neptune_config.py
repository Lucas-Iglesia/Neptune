# Configuration Neptune UI
# Ce fichier permet de personnaliser l'interface Neptune

# === Configuration du logo ===
# Chemin vers le logo Neptune (formats supportés: .png, .jpg, .jpeg)
# Laissez vide pour utiliser le logo par défaut
LOGO_PATH = "logo.png"

# === Configuration de l'interface ===
# Dimensions de l'interface utilisateur
UI_WIDTH = 1920
UI_HEIGHT = 1080

# === Configuration des couleurs ===
# Couleurs personnalisées (format BGR)
COLORS = {
    'bg_dark': (45, 45, 45),       # Arrière-plan sombre
    'bg_light': (70, 70, 70),      # Arrière-plan clair
    'primary': (0, 80, 243),      # Couleur principale (orange Neptune vif)
    'success': (0, 255, 0),        # Vert (succès)
    'warning': (0, 255, 255),      # Jaune (avertissement)
    'danger': (0, 0, 255),         # Rouge (danger)
    'text_white': (255, 255, 255), # Texte blanc
    'text_gray': (200, 200, 200),  # Texte gris
    'border': (100, 100, 100),     # Bordures
    'water_zone': (0, 255, 0),     # Zone d'eau
}

# === Configuration des dimensions ===
DIMENSIONS = {
    'header_height': 80,         # Hauteur du header
    'stats_panel_width': 400,    # Largeur du panneau de statistiques
    'minimap_width': 480,        # Largeur de la minimap
    'minimap_height': 240,       # Hauteur de la minimap
    'padding': 20,               # Espacement général
}

# === Configuration des alertes ===
ALERTS = {
    'danger_threshold': 5,       # Seuil de danger (secondes sous l'eau)
    'alert_duration': 8.0,       # Durée d'affichage des alertes (secondes)
    'popup_duration': 7.0,       # Durée du popup d'alerte
}

# === Configuration audio ===
AUDIO = {
    'danger_message': "Alerte ! Baigneur en danger.",
    'test_message': "Test de l'alerte vocale. Système de surveillance aquatique opérationnel.",
    'language': 'fr',            # Langue pour gTTS
    'slow_speech': True,         # Parole lente
    'tld': 'fr',                 # Top-level domain pour gTTS
}

# === Configuration de la détection ===
DETECTION = {
    'conf_threshold': 0.55,      # Seuil de confiance pour la détection
    'max_distance': 100,         # Distance maximale pour l'association des tracks
    'max_disappeared': 300,      # Frames avant suppression d'un track
    'underwater_threshold': 15,  # Frames pour considérer une personne sous l'eau
    'surface_threshold': 5,      # Frames pour considérer une personne en surface
}

# === Messages d'interface ===
MESSAGES = {
    'title': "Neptune",
    'active_label': "Active",
    'underwater_label': "Underwater", 
    'danger_label': "Danger",
    'max_score_label': "Max Score",
    'paused_text': "PAUSED",
    'legend_safe': "Safe",
    'legend_danger': "Danger",
}

# === Configuration des touches ===
KEYS = {
    'pause': 32,          # SPACE - Pause/Reprendre
    'water_toggle': 'w',  # W - Activer/Désactiver l'affichage de détection d'eau
    'test_alert': 't',    # T - Test d'alerte vocale
    'exit': 27,           # ESC - Quitter
}
