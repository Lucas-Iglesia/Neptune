#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Main Window
- Interface utilisateur principale
"""

from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QGroupBox, QGridLayout, QDoubleSpinBox, 
    QSplitter, QLineEdit, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

from config_pyqt6 import DETECTION, ALERTS, UI
from core.video_processor import VideoProcessor
from utils.audio import speak_alert, initialize_audio


class NeptuneMainWindow(QMainWindow):
    """Fenêtre principale de l'application Neptune"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neptune")
        self.setGeometry(100, 100, UI['width'], UI['height'])
        
        # Style de l'interface
        self.setStyleSheet(self._get_stylesheet())
        
        # Composants principaux
        self.video_processor = VideoProcessor()
        self.is_playing = False
        
        # Construction de l'interface
        self._build_ui()
        self._connect_signals()
        
        # Initialisation audio
        initialize_audio()
    
    def _get_stylesheet(self):
        """Retourne le style CSS de l'interface"""
        return """
            QMainWindow { background:#2b2b2b; color:#fff; }
            QGroupBox { 
                font-weight:bold; 
                border:2px solid #555; 
                border-radius:8px; 
                margin-top:10px; 
                padding-top:10px; 
                background:#3b3b3b; 
            }
            QGroupBox::title { 
                left:10px; 
                padding:0 10px; 
                color:#00D4FF; 
            }
            QPushButton { 
                background:#4CAF50; 
                border:none; 
                color:white; 
                padding:10px; 
                border-radius:5px; 
                font-weight:bold; 
            }
            QPushButton:hover { background:#45a049; }
            QPushButton:pressed { background:#3d8b40; }
            QPushButton:disabled { background:#666; color:#999; }
            QLabel { color:#fff; }
            QTextEdit { 
                background:#1e1e1e; 
                border:1px solid #555; 
                color:#fff; 
                border-radius:5px; 
            }
        """
    
    def _build_ui(self):
        """Construit l'interface utilisateur"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panel gauche (contrôles)
        left_panel = self._create_control_panel()
        splitter.addWidget(left_panel)
        
        # Panel droit (vidéo)
        right_panel = self._create_video_panel()
        splitter.addWidget(right_panel)
        
        self.statusBar().showMessage("Prêt - Saisissez un chemin de vidéo")
    
    def _create_control_panel(self):
        """Crée le panel de contrôle gauche"""
        left = QWidget()
        left.setMaximumWidth(UI['control_panel_width'])
        left.setMinimumWidth(300)
        layout = QVBoxLayout(left)
        
        # Section fichier
        layout.addWidget(self._create_file_section())
        
        # Section lecture
        layout.addWidget(self._create_playback_section())
        
        # Section statistiques
        layout.addWidget(self._create_stats_section())
        
        # Section configuration
        layout.addWidget(self._create_config_section())
        
        # Section affichage
        layout.addWidget(self._create_display_section())
        
        # Section alertes
        layout.addWidget(self._create_alerts_section())
        
        layout.addStretch()
        return left
    
    def _create_file_section(self):
        """Crée la section de sélection de fichier"""
        group = QGroupBox("Fichier Vidéo")
        layout = QVBoxLayout(group)
        
        # Bouton désactivé (informatif)
        self.select_video_btn = QPushButton("Sélectionner une vidéo (DÉSACTIVÉ)")
        self.select_video_btn.setEnabled(False)
        self.select_video_btn.setToolTip("Utilisez le champ de texte ci-dessous")
        layout.addWidget(self.select_video_btn)
        
        # Label et champ de saisie
        label = QLabel("Chemin complet de la vidéo :")
        label.setStyleSheet("color:#FFD700; font-weight:bold;")
        layout.addWidget(label)
        
        # Ligne de saisie + bouton charger
        row = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setText("/home/achambaz/neptune/G-EIP-700-REN-7-1-eip-adrien.picot/app/video/rozel-15fps-fullhd.mp4")
        self.path_input.setPlaceholderText("Ex: /path/to/video.mp4")
        row.addWidget(self.path_input)
        
        btn_load = QPushButton("Charger")
        btn_load.clicked.connect(self.load_video_from_path)
        row.addWidget(btn_load)
        layout.addLayout(row)
        
        # Label du fichier sélectionné
        self.video_path_label = QLabel("Aucune vidéo sélectionnée")
        self.video_path_label.setWordWrap(True)
        self.video_path_label.setStyleSheet("color:#999;")
        layout.addWidget(self.video_path_label)
        
        return group
    
    def _create_playback_section(self):
        """Crée la section de contrôle de lecture"""
        group = QGroupBox("Lecture")
        layout = QVBoxLayout(group)
        
        row = QHBoxLayout()
        
        self.play_btn = QPushButton("Lecture")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        row.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("Arrêt")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_playback)
        row.addWidget(self.stop_btn)
        
        layout.addLayout(row)
        return group
    
    def _create_stats_section(self):
        """Crée la section des statistiques"""
        group = QGroupBox("Stats temps réel")
        layout = QGridLayout(group)
        
        self.stats_labels = {
            'active': QLabel("0"),
            'underwater': QLabel("0"),
            'danger': QLabel("0"),
            'max_score': QLabel("0")
        }
        
        labels_titles = {
            "active": "Actifs",
            "underwater": "Sous l'eau",
            "danger": "En danger",
            "max_score": "Score max"
        }
        
        for i, (key, label) in enumerate(self.stats_labels.items()):
            label.setStyleSheet("font-size:14px; font-weight:bold;")
            layout.addWidget(QLabel(labels_titles[key] + ":"), i, 0)
            layout.addWidget(label, i, 1)
        
        return group
    
    def _create_config_section(self):
        """Crée la section de configuration"""
        group = QGroupBox("Configuration")
        layout = QVBoxLayout(group)
        
        # Seuil de confiance
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Seuil confiance:"))
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setValue(DETECTION['conf_threshold'])
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.valueChanged.connect(self.update_confidence)
        row1.addWidget(self.conf_spin)
        layout.addLayout(row1)
        
        # Seuil de danger
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Seuil danger (s):"))
        
        self.danger_spin = QDoubleSpinBox()
        self.danger_spin.setRange(1.0, 30.0)
        self.danger_spin.setValue(ALERTS['danger_threshold'])
        self.danger_spin.valueChanged.connect(self.update_danger_threshold)
        row2.addWidget(self.danger_spin)
        layout.addLayout(row2)
        
        return group
    
    def _create_display_section(self):
        """Crée la section d'affichage"""
        group = QGroupBox("Affichage")
        layout = QVBoxLayout(group)
        
        # Toggle détection d'eau
        self.btn_toggle_water = QPushButton("Afficher Détection Eau")
        self.btn_toggle_water.setCheckable(True)
        self.btn_toggle_water.setChecked(False)
        self.btn_toggle_water.clicked.connect(self.toggle_water_detection)
        layout.addWidget(self.btn_toggle_water)
        
        # Recalcul zone d'eau
        self.btn_recalc_water = QPushButton("Recalculer Zone d'Eau")
        self.btn_recalc_water.setToolTip("Raccourci: R")
        self.btn_recalc_water.clicked.connect(self.recalculate_water_zone)
        layout.addWidget(self.btn_recalc_water)
        
        return group
    
    def _create_alerts_section(self):
        """Crée la section des alertes"""
        group = QGroupBox("Journal des Alertes")
        layout = QVBoxLayout(group)
        
        # Zone de texte des alertes
        self.alerts_text = QTextEdit()
        self.alerts_text.setMaximumHeight(150)
        self.alerts_text.setReadOnly(True)
        layout.addWidget(self.alerts_text)
        
        # Bouton test alerte
        self.test_alert_btn = QPushButton("Test Alerte Vocale")
        self.test_alert_btn.clicked.connect(self.test_voice_alert)
        layout.addWidget(self.test_alert_btn)
        
        return group
    
    def _create_video_panel(self):
        """Crée le panel d'affichage vidéo"""
        right = QWidget()
        layout = QVBoxLayout(right)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(
            UI['video_panel_min_width'], 
            UI['video_panel_min_height']
        )
        self.video_label.setStyleSheet(
            "QLabel { border:2px solid #555; border-radius:8px; "
            "background:#1e1e1e; color:#999; }"
        )
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText(
            "Aucune vidéo chargée\\n\\nSélectionnez une vidéo pour commencer"
        )
        self.video_label.setScaledContents(True)
        
        layout.addWidget(self.video_label)
        return right
    
    def _connect_signals(self):
        """Connecte les signaux du video processor"""
        self.video_processor.frameReady.connect(self.update_frame)
        self.video_processor.statsReady.connect(self.update_stats)
        self.video_processor.alertTriggered.connect(self.handle_alert)
        
        # Configuration initiale
        self.video_processor.conf_threshold = DETECTION['conf_threshold']
        self.video_processor.danger_threshold = ALERTS['danger_threshold']
    
    # === Slots d'interface ===
    
    def load_video_from_path(self):
        """Charge une vidéo depuis le chemin saisi"""
        path = self.path_input.text().strip()
        
        if not path:
            self.statusBar().showMessage("Veuillez saisir un chemin")
            return
        
        if not Path(path).exists():
            self.statusBar().showMessage("Fichier introuvable")
            return
        
        self.video_path_label.setText(Path(path).name)
        
        # Arrêt de l'ancien thread
        try:
            if self.video_processor.isRunning():
                self.video_processor.stop()
        except Exception:
            pass
        
        # Création d'un nouveau processor
        self.video_processor = VideoProcessor()
        self._connect_signals()
        
        # Chargement des modèles IA
        self.statusBar().showMessage("Chargement des modèles IA…")
        QApplication.processEvents()
        ok = self.video_processor.load_models()
        status_msg = "Modèles IA chargés" if ok else "Erreur chargement modèles"
        self.statusBar().showMessage(status_msg)
        
        # Chargement de la vidéo
        self.statusBar().showMessage("Chargement vidéo…")
        QApplication.processEvents()
        
        if self.video_processor.load_video(path):
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.statusBar().showMessage("Vidéo chargée – Lecture prête")
            self.video_label.setText("")
        else:
            self.statusBar().showMessage("Erreur chargement vidéo")
    
    def toggle_playback(self):
        """Bascule entre lecture et pause"""
        if not self.is_playing:
            if not self.video_processor.isRunning():
                self.video_processor.start()
            else:
                self.video_processor.is_paused = False
            
            self.is_playing = True
            self.play_btn.setText("Pause")
            self.statusBar().showMessage("Lecture…")
        else:
            self.video_processor.is_paused = True
            self.is_playing = False
            self.play_btn.setText("Lecture")
            self.statusBar().showMessage("En pause")
    
    def stop_playback(self):
        """Arrête la lecture"""
        try:
            if self.video_processor.isRunning():
                self.video_processor.stop()
        except Exception:
            pass
        
        self.is_playing = False
        self.play_btn.setText("Lecture")
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.video_label.setText(
            "Aucune vidéo chargée\\n\\nSélectionnez une vidéo pour commencer"
        )
        self.statusBar().showMessage("Arrêté")
    
    def update_frame(self, frame):
        """Met à jour l'affichage de la frame vidéo"""
        try:
            h, w, c = frame.shape
            qimg = QImage(frame.data, w, h, 3*w, QImage.Format.Format_RGB888).rgbSwapped()
            pix = QPixmap.fromImage(qimg).scaled(
                self.video_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(pix)
        except Exception as e:
            print(f"[UI] update_frame KO: {e}")
    
    def update_stats(self, stats):
        """Met à jour les statistiques affichées"""
        try:
            self.stats_labels['active'].setText(f"{stats['active']}")
            self.stats_labels['underwater'].setText(f"{stats['underwater']}")
            self.stats_labels['danger'].setText(f"{stats['danger']}")
            
            # Affichage du score max avec l'ID de la personne
            max_score = stats['max_score']
            max_score_id = stats.get('max_score_id')
            if max_score_id is not None and max_score > 0:
                self.stats_labels['max_score'].setText(f"{max_score:.2f} (ID:{max_score_id})")
            else:
                self.stats_labels['max_score'].setText(f"{max_score:.2f}")
            
            # Couleur spéciale pour les dangers
            danger_style = ("color:#ff4444; font-weight:bold;" 
                          if stats['danger'] > 0 
                          else "color:#ffffff; font-weight:bold;")
            self.stats_labels['danger'].setStyleSheet(danger_style)
            
            # Couleur spéciale pour le score max élevé
            if max_score >= 50:
                max_score_style = "color:#ff4444; font-weight:bold;"
            elif max_score >= 30:
                max_score_style = "color:#ffaa00; font-weight:bold;"
            else:
                max_score_style = "color:#ffffff; font-weight:bold;"
            self.stats_labels['max_score'].setStyleSheet(max_score_style)
            
        except Exception as e:
            print(f"[UI] update_stats KO: {e}")
    
    def handle_alert(self, message):
        """Gère l'affichage d'une nouvelle alerte"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.alerts_text.append(f"[{timestamp}] {message}")
        self.statusBar().showMessage(f"ALERTE: {message}")
    
    def test_voice_alert(self):
        """Teste l'alerte vocale"""
        speak_alert("test")
        self.handle_alert("Test d'alerte vocale déclenché")
    
    def toggle_water_detection(self, checked):
        """Bascule l'affichage de la détection d'eau"""
        self.video_processor.show_water_detection = checked
        text = "Masquer Détection Eau" if checked else "Afficher Détection Eau"
        self.btn_toggle_water.setText(text)
        print(f"[UI] Eau: {'ON' if checked else 'OFF'}")
    
    def recalculate_water_zone(self):
        """Recalcule la zone d'eau de manière thread-safe"""
        print("[UI] Recalcul zone d'eau…")
        
        # Désactiver le bouton pendant l'opération
        self.btn_recalc_water.setEnabled(False)
        self.btn_recalc_water.setText("Recalcul en cours...")
        
        # Forcer le traitement des événements UI
        QApplication.processEvents()
        
        try:
            if self.video_processor.recalculate_water_detection():
                self.btn_toggle_water.setChecked(True)
                self.toggle_water_detection(True)
                self.statusBar().showMessage("Zone d'eau recalculée !", 3000)
                self.handle_alert("Zone d'eau recalculée")
            else:
                self.statusBar().showMessage("Échec recalcul zone d'eau", 3000)
                self.handle_alert("Échec recalcul zone d'eau")
        
        except Exception as e:
            print(f"[UI] Erreur recalcul: {e}")
            self.statusBar().showMessage(f"Erreur: {e}", 3000)
        
        finally:
            # Réactiver le bouton
            self.btn_recalc_water.setEnabled(True)
            self.btn_recalc_water.setText("Recalculer Zone d'Eau")
    
    def update_confidence(self, value):
        """Met à jour le seuil de confiance"""
        self.video_processor.conf_threshold = float(value)
    
    def update_danger_threshold(self, value):
        """Met à jour le seuil de danger"""
        self.video_processor.danger_threshold = float(value)
    
    # === Gestion des événements ===
    
    def keyPressEvent(self, event):
        """Gère les raccourcis clavier"""
        if event.key() == Qt.Key.Key_W:
            self.btn_toggle_water.toggle()
            self.toggle_water_detection(self.btn_toggle_water.isChecked())
        elif event.key() == Qt.Key.Key_T:
            self.test_voice_alert()
        elif event.key() == Qt.Key.Key_R:
            self.recalculate_water_zone()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Gère la fermeture de l'application de manière sécurisée"""
        print("[UI] Fermeture de l'application...")
        
        try:
            if self.video_processor.isRunning():
                print("[UI] Arrêt du processeur vidéo...")
                self.video_processor.stop()
                print("[UI] Processeur vidéo arrêté")
        except Exception as e:
            print(f"[UI] Erreur lors de l'arrêt: {e}")
        
        print("[UI] Fermeture terminée")
        event.accept()
