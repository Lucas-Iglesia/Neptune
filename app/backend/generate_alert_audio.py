#!/usr/bin/env python3
"""
Script pour générer le fichier audio d'alerte une seule fois
"""

import os
from pathlib import Path
from gtts import gTTS

# Message d'alerte en français
ALERT_MESSAGE = "Alerte ! Baigneur en danger."

# Chemin de sortie
OUTPUT_PATH = Path("assets/alert_audio.mp3")

def generate_alert_audio():
    """Génère le fichier audio d'alerte"""
    # Créer le dossier assets s'il n'existe pas
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    
    # Vérifier si le fichier existe déjà
    if OUTPUT_PATH.exists():
        print(f"Le fichier audio d'alerte existe déjà : {OUTPUT_PATH}")
        return
    
    try:
        # Générer l'audio avec gTTS
        tts = gTTS(text=ALERT_MESSAGE, lang='fr', slow=False)
        tts.save(str(OUTPUT_PATH))
        print(f"Fichier audio d'alerte généré : {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Erreur lors de la génération de l'audio : {e}")

if __name__ == "__main__":
    generate_alert_audio()
