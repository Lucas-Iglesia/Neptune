#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune - Application de surveillance aquatique
Version refactorisée et organisée

Utilisation:
    python neptune_app.py

Raccourcis clavier:
    W - Basculer affichage détection d'eau
    T - Test alerte vocale
    R - Recalculer zone d'eau
"""

import os
import sys
from PyQt6.QtWidgets import QApplication

# Configuration de l'environnement Qt avant import
from core.constants import QT_ENV_CONFIG
for key, value in QT_ENV_CONFIG.items():
    os.environ.setdefault(key, value)

from config_pyqt6 import DETECTION, ALERTS
from ui.main_window import NeptuneMainWindow
from utils.audio import generate_audio_files, HAS_AUDIO


def main():
    """Point d'entrée principal de l'application"""
    # Génération des fichiers audio si disponible
    if HAS_AUDIO:
        generate_audio_files()
    
    # Configuration de l'application Qt
    app = QApplication(sys.argv)
    app.setApplicationName("Neptune")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Neptune Team")
    
    # Messages informatifs
    print("Neptune (factorisé) – Raccourcis: W eau / T test audio / R recalcul eau")
    print(f"Conf: conf={DETECTION['conf_threshold']}, "
          f"underwater={DETECTION['underwater_threshold']} frames, "
          f"danger={ALERTS['danger_threshold']} s")
    
    # Création et affichage de la fenêtre principale
    window = NeptuneMainWindow()
    window.show()
    
    # Lancement de la boucle d'événements
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
