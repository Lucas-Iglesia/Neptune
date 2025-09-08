#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Audio Utilities
- Génération et lecture d'alertes vocales
"""

import os
import time
import threading
from config_pyqt6 import AUDIO

# ===== Dépendances Audio (optionnelles) =====
try:
    from gtts import gTTS
    import pygame
    HAS_AUDIO = True
except Exception:
    print("Audio non disponible.")
    HAS_AUDIO = False


def generate_audio_files():
    """Génère les fichiers audio pour les alertes"""
    if not HAS_AUDIO:
        return
    
    os.makedirs("audio_alerts", exist_ok=True)
    files = {"danger": "alerte_danger.mp3", "test": "test_alerte.mp3"}
    texts = {"danger": AUDIO['danger_message'], "test": AUDIO['test_message']}
    
    for key, fname in files.items():
        path = os.path.join("audio_alerts", fname)
        if os.path.exists(path):
            continue
        try:
            gTTS(
                text=texts[key], 
                lang=AUDIO['language'], 
                slow=AUDIO['slow_speech'], 
                tld=AUDIO['tld']
            ).save(path)
        except Exception as e:
            print(f"[Audio] Erreur génération {fname}: {e}")


def speak_alert(kind="danger"):
    """Joue une alerte vocale de manière asynchrone"""
    if not HAS_AUDIO:
        print(f"[ALERTE VOCALE] {kind}")
        return
    
    def _play():
        try:
            files = {"danger": "alerte_danger.mp3", "test": "test_alerte.mp3"}
            path = os.path.join("audio_alerts", files.get(kind, "alerte_danger.mp3"))
            
            if not os.path.exists(path):
                print(f"[Audio] Manquant: {path}")
                return
            
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"[Audio] Lecture KO: {e}")
    
    threading.Thread(target=_play, daemon=True).start()


def initialize_audio():
    """Initialise le système audio"""
    if HAS_AUDIO:
        try:
            pygame.mixer.init()
            generate_audio_files()
            return True
        except Exception as e:
            print(f"[Audio] init KO: {e}")
            return False
    return False
