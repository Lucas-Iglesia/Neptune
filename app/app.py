#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neptune PyQt6 Application
Application de détection de personnes sous l'eau avec interface PyQt6
Basé sur le système Demo-5 avec ajout de sélection vidéo
"""

import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
import time
import threading
from collections import defaultdict
import math
from datetime import datetime
import tempfile
import re

# --- Durcissement Qt AVANT QApplication (bypass portail GTK/icônes) ---
os.environ.setdefault("QT_NO_XDG_DESKTOP_PORTAL", "1")
os.environ.setdefault("QT_STYLE_OVERRIDE", "Fusion")
os.environ.setdefault("QT_ICON_THEME", "hicolor")

# Imports locaux
from config_pyqt6 import COLORS, DETECTION, ALERTS, AUDIO, UI
from underwater_tracker import UnderwaterPersonTracker

# AI/ML imports
try:
    from transformers import AutoImageProcessor, DFineForObjectDetection
    from ultralytics import YOLO
    HAS_AI_MODELS = True
except ImportError:
    print("⚠️ Modèles IA non disponibles. Fonctionnement en mode démo.")
    HAS_AI_MODELS = False

# Audio imports
try:
    from gtts import gTTS
    import pygame
    HAS_AUDIO = True
except ImportError:
    print("⚠️ Fonctionnalités audio non disponibles.")
    HAS_AUDIO = False

# PyQt6 imports avec gestion d'erreur
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, 
        QHBoxLayout, QPushButton, QLabel, QFileDialog,
        QSlider, QFrame, QGridLayout, QTextEdit, QProgressBar,
        QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
        QComboBox, QStatusBar, QSplitter, QTabWidget, QLineEdit
    )
    from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt, QSize
    from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor
    HAS_PYQT6 = True
except ImportError:
    print("❌ PyQt6 non installé. Veuillez installer PyQt6:")
    print("pip install PyQt6")
    sys.exit(1)


class VideoProcessor(QThread):
    """Thread pour traiter la vidéo sans bloquer l'interface"""
    frameReady = pyqtSignal(np.ndarray)
    statsReady = pyqtSignal(dict)
    alertTriggered = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.is_running = False
        self.is_paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        
        # Detection models
        self.nwsd = None
        self.dfine = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configuration des seuils
        self.conf_threshold = 0.3
        self.underwater_threshold = 15
        self.danger_threshold = 5.0
        
        # Tracking
        self.tracker = None
        self.H_latest = None
        self.water_mask_global = None
        
    def load_models(self):
        """Charger les modèles IA"""
        print("🔄 Début du chargement des modèles...")
        try:
            # Charger le modèle de segmentation d'eau
            model_path = Path("model/nwd-v2.pt")
            if not model_path.exists():
                # Essayer le chemin du Demo-5
                model_path = Path("demo/Demo-5/model/nwd-v2.pt")
            
            if model_path.exists():
                print(f"📁 Chargement du modèle d'eau depuis: {model_path}")
                self.nwsd = YOLO(str(model_path))
                print("✅ Modèle de détection d'eau chargé")
            else:
                print("ℹ️ Modèle d'eau introuvable, ignore la segmentation.")
                self.nwsd = None

            # Charger le modèle de détection de personnes
            model_id = "ustc-community/dfine-xlarge-obj2coco"
            print(f"🤖 Chargement de D-FINE depuis: {model_id}")
            print(f"📱 Device: {self.device}")
            
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            print("✅ Processor D-FINE chargé")
            
            self.dfine = DFineForObjectDetection.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device).eval()
            print("✅ Modèle D-FINE de détection de personnes chargé")
            
            # Vérifier que les modèles sont bien chargés
            print(f"🔍 État des modèles:")
            print(f"   - NWSD: {'✅' if self.nwsd is not None else '❌'}")
            print(f"   - Processor: {'✅' if self.processor is not None else '❌'}")
            print(f"   - D-FINE: {'✅' if self.dfine is not None else '❌'}")
            
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_video(self, video_path):
        """Charger une vidéo"""
        self.video_path = video_path
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                return False
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            print(f"✅ Vidéo chargée: {video_path}")
            print(f"   Frames total: {self.total_frames}")
            print(f"   FPS: {self.fps}")
            
            # Initialiser le tracker
            self.tracker = UnderwaterPersonTracker()
            
            # Analyser la première frame pour la détection d'eau
            self.analyze_water_detection()
            
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement de la vidéo: {e}")
            return False
    
    def analyze_water_detection(self):
        """Analyser la première frame pour détecter l'eau"""
        if not hasattr(self, "cap") or self.cap is None or self.nwsd is None:
            return
        
        try:
            # Lire la première frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return
            
            # Détecter l'eau
            seg_res = self.nwsd.predict(frame, imgsz=512, task="segment", conf=0.25, verbose=False)[0]
            if seg_res.masks is not None:
                mask = (seg_res.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8)
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                self.water_mask_global = mask.copy()
                
                # Calculer l'homographie
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 5000:
                        pts = cnt.reshape(-1, 2).astype(np.float32)
                        sums = pts.sum(axis=1)
                        diffs = np.diff(pts, axis=1).reshape(-1)
                        src_quad = np.array([
                            pts[np.argmin(sums)],
                            pts[np.argmin(diffs)],
                            pts[np.argmax(sums)],
                            pts[np.argmax(diffs)]
                        ], dtype=np.float32)
                        
                        dst_rect = np.array([[0, 0], [400, 0], [400, 200], [0, 200]], dtype=np.float32)
                        H, _ = cv2.findHomography(src_quad, dst_rect, cv2.RANSAC, 3.0)
                        if H is not None:
                            self.H_latest = H.copy()
                            print("✅ Homographie calculée")
            
            # Remettre la vidéo au début
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse de l'eau: {e}")
    
    def detect_persons(self, frame):
        """Détecter les personnes dans une frame"""
        if self.dfine is None or self.processor is None:
            if not hasattr(self, '_model_warning_shown'):
                print("⚠️ Modèles D-FINE non chargés - aucune détection possible")
                print(f"   - D-FINE: {'✅' if self.dfine is not None else '❌'}")
                print(f"   - Processor: {'✅' if self.processor is not None else '❌'}")
                self._model_warning_shown = True
            return []
        
        try:
            # Convertir BGR vers RGB pour le modèle
            rgb_frame = frame[:, :, ::-1]
            inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)
            if self.device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
            
            outputs = self.dfine(**inputs)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=[(frame.shape[0], frame.shape[1])], 
                threshold=self.conf_threshold
            )[0]
            
            persons = []
            for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
                if int(label.item()) == 0:  # classe personne
                    x0, y0, x1, y1 = box.tolist()
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    w, h = x1 - x0, y1 - y0
                    
                    # Créer un objet simple similaire à BoxStub
                    person = {
                        'xywh': [[cx, cy, w, h]],
                        'conf': [float(score.item())],
                        'cls': [0],
                        'bbox': [x0, y0, x1, y1]  # Ajouter aussi la bbox pour debug
                    }
                    persons.append(person)
            
            # Debug : afficher le nombre de détections (seulement parfois pour éviter le spam)
            if len(persons) > 0 and hasattr(self, '_debug_counter'):
                self._debug_counter = getattr(self, '_debug_counter', 0) + 1
                if self._debug_counter % 30 == 0:  # Afficher tous les 30 frames
                    print(f"Détections trouvées: {len(persons)} personnes avec conf >= {self.conf_threshold}")
                    for i, p in enumerate(persons):
                        print(f"  Personne {i+1}: conf={p['conf'][0]:.3f}, center=({p['xywh'][0][0]:.1f}, {p['xywh'][0][1]:.1f})")
            elif len(persons) == 0 and not hasattr(self, '_no_detection_warned'):
                print(f"⚠️ Aucune détection avec seuil {self.conf_threshold}")
                self._no_detection_warned = True
            
            return persons
        except Exception as e:
            print(f"❌ Erreur lors de la détection: {e}")
            return []
    
    def run(self):
        """Boucle principale de traitement vidéo"""
        if not self.video_path or not hasattr(self, "cap"):
            return
        
        self.is_running = True
        frame_time = 1.0 / max(self.fps, 1e-6)
        
        while self.is_running:
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    # Fin de vidéo, recommencer
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                start_time = time.time()
                
                # Détecter les personnes
                persons = self.detect_persons(frame)
                
                # Mettre à jour le tracker si disponible
                stats = {'active': 0, 'underwater': 0, 'danger': 0, 'max_score': 0.0}
                assignments = {}
                if self.tracker and persons:
                    frame_timestamp = time.time()
                    assignments = self.tracker.update(persons, frame_timestamp)
                    
                    # Calculer les statistiques
                    active_tracks = self.tracker.get_active_tracks()
                    underwater_tracks = self.tracker.get_underwater_tracks()
                    danger_tracks = self.tracker.get_danger_tracks()
                    
                    stats = {
                        'active': len(active_tracks),
                        'underwater': len(underwater_tracks),
                        'danger': len(danger_tracks),
                        'max_score': max([t.get('dangerosity_score', 0.0) for t in self.tracker.tracks.values()], default=0.0)
                    }
                    
                    # Vérifier les alertes de danger
                    for track_id, track in danger_tracks.items():
                        if not track.get('voice_alert_sent', False):
                            alert_msg = f"DANGER: Baigneur {track_id} en danger!"
                            self.alertTriggered.emit(alert_msg)
                            track['voice_alert_sent'] = True
                elif self.tracker:
                    # Pas de détections mais tracker existe - mettre à jour quand même
                    frame_timestamp = time.time()
                    assignments = self.tracker.update([], frame_timestamp)
                    
                    active_tracks = self.tracker.get_active_tracks()
                    underwater_tracks = self.tracker.get_underwater_tracks()
                    danger_tracks = self.tracker.get_danger_tracks()
                    
                    stats = {
                        'active': len(active_tracks),
                        'underwater': len(underwater_tracks),
                        'danger': len(danger_tracks),
                        'max_score': max([t.get('dangerosity_score', 0.0) for t in self.tracker.tracks.values()], default=0.0)
                    }
                
                # Dessiner les détections et informations du tracker sur la frame
                vis_frame = self.draw_detections(frame, persons)
                
                # Envoyer la frame et les stats
                self.frameReady.emit(vis_frame)
                self.statsReady.emit(stats)
                
                self.current_frame += 1
                
                # Contrôler la vitesse de lecture
                elapsed = time.time() - start_time
                sleep_time = max(0.0, frame_time - elapsed)
                if sleep_time > 0:
                    self.msleep(int(sleep_time * 1000))
            else:
                self.msleep(50)  # Pause
    
    def draw_detections(self, frame, persons):
        """Dessiner les détections sur la frame avec informations du tracker"""
        vis_frame = frame.copy()
        
        # Debug : afficher le nombre de détections reçues
        if len(persons) > 0:
            print(f"🎯 draw_detections: {len(persons)} détections à dessiner")
        
        # Obtenir les informations du tracker si disponible
        active_tracks = {}
        underwater_tracks = {}
        danger_tracks = {}
        assignments = {}
        
        if self.tracker:
            active_tracks = self.tracker.get_active_tracks()
            underwater_tracks = self.tracker.get_underwater_tracks()
            danger_tracks = self.tracker.get_danger_tracks()
        
        # Fonction utilitaire pour les couleurs basées sur la dangerosité
        def get_color_by_dangerosity(score):
            if score <= 20:
                ratio = score / 20.0
                b = int(144 * ratio)
                g = int(100 + (138 * ratio))
                r = int(144 * ratio)
                return (b, g, r)
            elif score <= 40:
                ratio = (score - 20) / 20.0
                b = int(144 * (1 - ratio))
                g = int(238 + (17 * ratio))
                r = int(144 + (111 * ratio))
                return (b, g, r)
            elif score <= 60:
                ratio = (score - 40) / 20.0
                b = 0
                g = int(255 - (90 * ratio))
                r = 255
                return (b, g, r)
            elif score <= 80:
                ratio = (score - 60) / 20.0
                b = 0
                g = int(165 * (1 - ratio))
                r = 255
                return (b, g, r)
            else:
                ratio = (score - 80) / 20.0
                b = 0
                g = 0
                r = int(255 - (116 * ratio))
                return (b, g, r)
        
        # Dessiner les détections avec les informations du tracker
        for det_idx, person in enumerate(persons):
            cx, cy, w, h = person['xywh'][0]
            conf = person['conf'][0]
            
            x0, y0 = int(cx - w/2), int(cy - h/2)
            x1, y1 = int(cx + w/2), int(cy + h/2)
            
            # TOUJOURS dessiner la bounding box d'abord (en vert par défaut)
            color = (0, 255, 0)  # Vert par défaut
            cv2.rectangle(vis_frame, (x0, y0), (x1, y1), color, 2)
            
            # Afficher la confiance
            conf_text = f"Conf: {conf:.2f}"
            cv2.putText(vis_frame, conf_text, (x0, y0 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Chercher si cette détection est assignée à une track
            track_id = None
            track = None
            if self.tracker:
                active_tracks = self.tracker.get_active_tracks()
                for tid, t in active_tracks.items():
                    if t.get('center'):
                        tcx, tcy = t['center']
                        # Vérifier si cette détection correspond à cette track (distance proche)
                        if abs(tcx - cx) < 50 and abs(tcy - cy) < 50:
                            track_id = tid
                            track = t
                            break
            
            if track_id is not None and track:
                # Détection trackée - utiliser les couleurs basées sur la dangerosité
                score = track.get('dangerosity_score', 0)
                color = get_color_by_dangerosity(score)
                danger_tracks = self.tracker.get_danger_tracks() if self.tracker else {}
                is_danger = track_id in danger_tracks
                
                # Redessiner avec la bonne couleur
                thickness = 4 if is_danger else 2
                cv2.rectangle(vis_frame, (x0, y0), (x1, y1), color, thickness)
                
                # Affichage du statut, ID et score
                if track['status'] == 'underwater':
                    status_text = f"ID:{track_id} (SOUS L'EAU) - Score:{score}"
                    if track.get('underwater_duration'):
                        status_text += f" | {track['underwater_duration']:.1f}s"
                else:
                    status_text = f"ID:{track_id} - Score:{score}"
                
                # Afficher le texte avec un fond pour la lisibilité
                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_frame, (x0, y0 - 30), (x0 + text_size[0], y0 - 5), (0, 0, 0), -1)
                cv2.putText(vis_frame, status_text, (x0, y0 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Redessiner la confiance avec la bonne couleur
                cv2.putText(vis_frame, conf_text, (x0, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Dessiner les croix de danger pour les personnes en danger
        for track_id, track in danger_tracks.items():
            if track.get('dive_point') is not None:
                dive_x, dive_y = int(track['dive_point'][0]), int(track['dive_point'][1])
                cv2.drawMarker(
                    vis_frame,
                    (dive_x, dive_y),
                    color=(0, 0, 255),  # Rouge vif
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=3
                )
        
        # Ajouter des informations générales sur l'image
        info_y = 30
        total_detections = len(persons)
        total_tracks = len(active_tracks)
        underwater_count = len(underwater_tracks)
        danger_count = len(danger_tracks)
        
        # Calculer le score max
        max_dangerosity = 0
        max_dangerosity_id = None
        if self.tracker and self.tracker.tracks:
            for tid, t in self.tracker.tracks.items():
                score = t.get('dangerosity_score', 0)
                if score > max_dangerosity:
                    max_dangerosity = score
                    max_dangerosity_id = tid
        
        # Fond semi-transparent pour le texte
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (10, 5), (500, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
        
        cv2.putText(vis_frame, f"Detections: {total_detections}", (20, info_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Tracks actives: {total_tracks}", (20, info_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Sous l'eau: {underwater_count}", (20, info_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(vis_frame, f"En danger: {danger_count}", (20, info_y + 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_frame, f"Score max: {max_dangerosity} (ID:{max_dangerosity_id})", (20, info_y + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return vis_frame
    
    def pause(self):
        """Mettre en pause/reprendre"""
        self.is_paused = not self.is_paused
    
    def stop(self):
        """Arrêter le traitement"""
        self.is_running = False
        if hasattr(self, 'cap'):
            try:
                self.cap.release()
            except Exception:
                pass
        self.wait()


class NeptuneMainWindow(QMainWindow):
    """Fenêtre principale de l'application Neptune"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neptune - Surveillance Aquatique PyQt6")
        self.setGeometry(100, 100, 1400, 900)
        
        # Thème sombre
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #3b3b3b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #00D4FF;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
            QLabel {
                color: #ffffff;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #555;
                color: #ffffff;
                border-radius: 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 8px;
                background: #1e1e1e;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00D4FF;
                border: 1px solid #555;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        
        # Variables
        self.video_processor = VideoProcessor()
        self.current_frame = None
        self.is_playing = False
        
        # Interface
        self.setup_ui()
        self.setup_connections()
        
        # Initialiser l'audio si disponible
        if HAS_AUDIO:
            try:
                pygame.mixer.init()
            except Exception as e:
                print(f"⚠️ Audio indisponible: {e}")
        
    def setup_ui(self):
        """Configurer l'interface utilisateur"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter pour diviser l'interface
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panel de gauche - Contrôles
        self.setup_control_panel(splitter)
        
        # Panel de droite - Affichage vidéo
        self.setup_video_panel(splitter)
        
        # Barre de statut
        self.statusBar().showMessage("Prêt - Sélectionnez une vidéo pour commencer")
        
    def setup_control_panel(self, parent):
        """Configurer le panel de contrôles"""
        control_widget = QWidget()
        control_widget.setMaximumWidth(350)
        control_widget.setMinimumWidth(300)
        control_layout = QVBoxLayout(control_widget)
        
        # === Sélection de fichier ===
        file_group = QGroupBox("Fichier Vidéo")
        file_layout = QVBoxLayout()
        
        self.select_video_btn = QPushButton("📁 Sélectionner une vidéo (DÉSACTIVÉ)")
        self.select_video_btn.setEnabled(False)  # Désactiver pour éviter le crash GTK
        self.select_video_btn.setToolTip("Utilisez le champ de texte ci-dessous pour éviter les problèmes GTK")
        # self.select_video_btn.clicked.connect(self.select_video)
        file_layout.addWidget(self.select_video_btn)
        
        # Champ de texte pour saisir le chemin directement
        path_label = QLabel("Saisissez le chemin complet de la vidéo :")
        path_label.setStyleSheet("color: #FFD700; font-weight: bold;")  # Jaune pour attirer l'attention
        file_layout.addWidget(path_label)
        
        path_input_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Ex: /home/achambaz/neptune/G-EIP-700-REN-7-1-eip-adrien.picot/app/video/rozel-15fps-fullhd.mp4")
        self.path_input.setText("/home/achambaz/neptune/G-EIP-700-REN-7-1-eip-adrien.picot/app/video/rozel-15fps-fullhd.mp4")  # Chemin par défaut
        path_input_layout.addWidget(self.path_input)
        
        load_btn = QPushButton("📁 Charger")
        load_btn.clicked.connect(self.load_video_from_path)
        path_input_layout.addWidget(load_btn)
        
        file_layout.addLayout(path_input_layout)
        
        self.video_path_label = QLabel("Aucune vidéo sélectionnée")
        self.video_path_label.setWordWrap(True)
        self.video_path_label.setStyleSheet("color: #999;")
        file_layout.addWidget(self.video_path_label)
        
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        # === Contrôles de lecture ===
        playback_group = QGroupBox("Contrôles de Lecture")
        playback_layout = QVBoxLayout()
        
        # Boutons de contrôle
        button_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶️ Lecture")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("⏹️ Arrêt")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)
        
        button_layout.addWidget(self.play_btn)
        button_layout.addWidget(self.stop_btn)
        playback_layout.addLayout(button_layout)
        
        playback_group.setLayout(playback_layout)
        control_layout.addWidget(playback_group)
        
        # === Statistiques de détection ===
        stats_group = QGroupBox("Statistiques en Temps Réel")
        stats_layout = QGridLayout()
        
        self.stats_labels = {
            'active': QLabel("Actifs: 0"),
            'underwater': QLabel("Sous l'eau: 0"),
            'danger': QLabel("En danger: 0"),
            'max_score': QLabel("Score max: 0")
        }
        
        for i, (key, label) in enumerate(self.stats_labels.items()):
            label.setStyleSheet("font-size: 14px; font-weight: bold;")
            stats_layout.addWidget(QLabel(key.title() + ":"), i, 0)
            stats_layout.addWidget(label, i, 1)
        
        stats_group.setLayout(stats_layout)
        control_layout.addWidget(stats_group)
        
        # === Configuration de détection ===
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        
        # Seuil de confiance
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Seuil confiance:"))
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.1, 1.0)
        self.conf_spinbox.setValue(0.55)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.valueChanged.connect(self.update_confidence)
        conf_layout.addWidget(self.conf_spinbox)
        config_layout.addLayout(conf_layout)
        
        # Seuil de danger
        danger_layout = QHBoxLayout()
        danger_layout.addWidget(QLabel("Seuil danger (s):"))
        self.danger_spinbox = QDoubleSpinBox()
        self.danger_spinbox.setRange(1.0, 30.0)
        self.danger_spinbox.setValue(5.0)
        self.danger_spinbox.valueChanged.connect(self.update_danger_threshold)
        danger_layout.addWidget(self.danger_spinbox)
        config_layout.addLayout(danger_layout)
        
        config_group.setLayout(config_layout)
        control_layout.addWidget(config_group)
        
        # === Journal des alertes ===
        alerts_group = QGroupBox("Journal des Alertes")
        alerts_layout = QVBoxLayout()
        
        self.alerts_text = QTextEdit()
        self.alerts_text.setMaximumHeight(150)
        self.alerts_text.setReadOnly(True)
        alerts_layout.addWidget(self.alerts_text)
        
        # Bouton de test d'alerte
        self.test_alert_btn = QPushButton("🔊 Test Alerte Vocale")
        self.test_alert_btn.clicked.connect(self.test_voice_alert)
        alerts_layout.addWidget(self.test_alert_btn)
        
        alerts_group.setLayout(alerts_layout)
        control_layout.addWidget(alerts_group)
        
        # Spacer
        control_layout.addStretch()
        
        parent.addWidget(control_widget)
    
    def setup_video_panel(self, parent):
        """Configurer le panel d'affichage vidéo"""
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        # Label pour afficher la vidéo
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                border-radius: 8px;
                background-color: #1e1e1e;
                color: #999;
            }
        """)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("Aucune vidéo chargée\n\nSélectionnez une vidéo pour commencer")
        self.video_label.setScaledContents(True)
        
        video_layout.addWidget(self.video_label)
        
        parent.addWidget(video_widget)
    
    def setup_connections(self):
        """Configurer les connexions de signaux"""
        self.video_processor.frameReady.connect(self.update_frame)
        self.video_processor.statsReady.connect(self.update_stats)
        self.video_processor.alertTriggered.connect(self.handle_alert)
    
    def select_video(self):
        """Sélectionner un fichier vidéo - Version sécurisée"""
        try:
            options = (
                QFileDialog.Option.DontUseNativeDialog |
                QFileDialog.Option.DontUseCustomDirectoryIcons
            )
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Sélectionner une vidéo",
                "",
                "Fichiers vidéo (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;Tous les fichiers (*)",
                options=options
            )
        except Exception as e:
            print(f"Erreur avec le dialog de fichiers: {e}")
            self.statusBar().showMessage("Erreur avec le sélecteur. Utilisez le champ de texte ci-dessous.")
            return
        
        if not file_path:
            return
            
        # Mettre le chemin dans le champ de texte et charger
        self.path_input.setText(file_path)
        self.load_video_from_path()
    
    def load_video_from_path(self):
        """Charger une vidéo depuis le chemin saisi"""
        file_path = self.path_input.text().strip()
        
        if not file_path:
            self.statusBar().showMessage("Veuillez saisir un chemin de vidéo")
            return
        
        if not Path(file_path).exists():
            self.statusBar().showMessage("Le fichier spécifié n'existe pas")
            return
        
        # Mettre à jour l'UI
        filename = Path(file_path).name
        self.video_path_label.setText(f"📹 {filename}")
        
        # Arrêter l'ancien thread si besoin
        try:
            if self.video_processor.isRunning():
                self.video_processor.stop()
        except Exception:
            pass

        # Recréer proprement le processor sans kwargs et reconnecter les signaux
        self.video_processor = VideoProcessor()
        self.setup_connections()
        
        # Charger les modèles si disponible
        print(f"🔍 HAS_AI_MODELS = {HAS_AI_MODELS}")
        if HAS_AI_MODELS:
            self.statusBar().showMessage("Chargement des modèles IA...")
            QApplication.processEvents()
            
            if self.video_processor.load_models():
                self.statusBar().showMessage("Modèles IA chargés")
            else:
                self.statusBar().showMessage("Erreur lors du chargement des modèles")
        else:
            self.statusBar().showMessage("Modèles IA non disponibles - Mode démo")
        
        # Charger la vidéo
        self.statusBar().showMessage("Chargement de la vidéo...")
        QApplication.processEvents()
        
        if self.video_processor.load_video(file_path):
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.statusBar().showMessage("Vidéo chargée - Prêt pour la lecture")
            # Effacer le placeholder vidéo
            self.video_label.setText("")
        else:
            self.statusBar().showMessage("Erreur lors du chargement de la vidéo")
    
    def toggle_playback(self):
        """Basculer entre lecture et pause"""
        if not self.is_playing:
            self.start_playback()
        else:
            self.pause_playback()
    
    def start_playback(self):
        """Démarrer la lecture"""
        if not self.video_processor.isRunning():
            self.video_processor.start()
        else:
            self.video_processor.is_paused = False
        
        self.is_playing = True
        self.play_btn.setText("⏸️ Pause")
        self.statusBar().showMessage("Lecture en cours...")
    
    def pause_playback(self):
        """Mettre en pause"""
        self.video_processor.is_paused = True
        self.is_playing = False
        self.play_btn.setText("▶️ Lecture")
        self.statusBar().showMessage("En pause")
    
    def stop_playback(self):
        """Arrêter la lecture"""
        try:
            if self.video_processor.isRunning():
                self.video_processor.stop()
        except Exception:
            pass
        self.is_playing = False
        self.play_btn.setText("▶️ Lecture")
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("Aucune vidéo chargée\n\nSélectionnez une vidéo pour commencer")
        self.statusBar().showMessage("Arrêté")
    
    def update_frame(self, frame):
        """Mettre à jour l'affichage de la frame"""
        try:
            # Convertir la frame OpenCV en QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            
            # Convertir en QPixmap et afficher
            pixmap = QPixmap.fromImage(q_image)
            
            # Redimensionner pour s'adapter au label
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour de la frame: {e}")
    
    def update_stats(self, stats):
        """Mettre à jour les statistiques"""
        try:
            self.stats_labels['active'].setText(f"{stats['active']}")
            self.stats_labels['underwater'].setText(f"{stats['underwater']}")
            self.stats_labels['danger'].setText(f"{stats['danger']}")
            self.stats_labels['max_score'].setText(f"{stats['max_score']:.2f}")
            
            # Changer la couleur en fonction du danger
            if stats['danger'] > 0:
                self.stats_labels['danger'].setStyleSheet("color: #ff4444; font-weight: bold;")
            else:
                self.stats_labels['danger'].setStyleSheet("color: #ffffff; font-weight: bold;")
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour des stats: {e}")
    
    def handle_alert(self, message):
        """Gérer les alertes"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert_text = f"[{timestamp}] {message}"
        
        # Ajouter au journal
        self.alerts_text.append(alert_text)
        
        # Jouer l'alerte vocale
        self.play_voice_alert(message)
        
        # Mettre à jour la barre de statut
        self.statusBar().showMessage(f"🚨 ALERTE: {message}")
    
    def play_voice_alert(self, message):
        """Jouer une alerte vocale"""
        if not HAS_AUDIO:
            return

        def _play_alert():
            try:
                # Créer un fichier audio temporaire
                tts = gTTS(text=message, lang='fr', slow=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tts.save(tmp_file.name)
                    
                    # Jouer l'audio
                    pygame.mixer.music.load(tmp_file.name)
                    pygame.mixer.music.play()
                    
                    # Attendre la fin de la lecture
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    # Nettoyer
                    os.unlink(tmp_file.name)
                    
            except Exception as e:
                print(f"Erreur lors de la lecture de l'alerte vocale: {e}")
        
        # Jouer dans un thread séparé
        threading.Thread(target=_play_alert, daemon=True).start()
    
    def test_voice_alert(self):
        """Tester l'alerte vocale"""
        test_message = "Test de l'alerte vocale. Système de surveillance aquatique opérationnel."
        self.handle_alert(test_message)
    
    def update_confidence(self, value):
        """Mettre à jour le seuil de confiance"""
        self.video_processor.conf_threshold = float(value)
    
    def update_danger_threshold(self, value):
        """Mettre à jour le seuil de danger"""
        self.video_processor.danger_threshold = float(value)
    
    def closeEvent(self, event):
        """Gérer la fermeture de l'application"""
        try:
            if self.video_processor.isRunning():
                self.video_processor.stop()
        except Exception:
            pass
        event.accept()


def main():
    """Fonction principale"""
    app = QApplication(sys.argv)
    
    # Configuration de l'application
    app.setApplicationName("Neptune")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Neptune Team")
    
    # Créer et afficher la fenêtre principale
    window = NeptuneMainWindow()
    window.show()
    
    # Démarrer la boucle d'événements
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
