#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Video Processor
- Thread de traitement vidéo principal
"""

import cv2
import time
import numpy as np
import threading
from PyQt6.QtCore import QThread, pyqtSignal

from config_pyqt6 import DETECTION, ALERTS
from core.constants import MAP_W_PX, MAP_H_PX
from core.tracker import UnderwaterPersonTracker
from detection.models import ModelManager
from detection.water import WaterDetector
from utils.alerts import AlertPopup
from utils.danger import calculate_distance_from_shore
from utils.audio import speak_alert


class VideoProcessor(QThread):
    """Thread principal de traitement vidéo"""
    
    frameReady = pyqtSignal(np.ndarray)
    statsReady = pyqtSignal(dict)
    alertTriggered = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # État de la vidéo
        self.video_path = None
        self.is_running = False
        self.is_paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        
        # Mutex pour la protection thread-safe
        self._lock = threading.Lock()
        
        # Configuration
        self.conf_threshold = DETECTION['conf_threshold']
        self.underwater_threshold = DETECTION['underwater_threshold']
        self.danger_threshold = ALERTS['danger_threshold']
        
        # Composants principaux
        self.model_manager = ModelManager()
        self.water_detector = WaterDetector()
        self.tracker = None
        self.alert_popup = AlertPopup(duration=7.0)
        
        # Interface
        self.show_water_detection = False
        
        # Temps
        self.frame_timestamp = None
        
    def load_models(self):
        """
        Charge les modèles IA
        
        Returns:
            bool: True si le chargement réussit
        """
        return self.model_manager.load_models()
    
    def load_video(self, path):
        """
        Charge une vidéo
        
        Args:
            path: Chemin vers la vidéo
        
        Returns:
            bool: True si le chargement réussit
        """
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        
        if not self.cap.isOpened():
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Initialisation du tracker
        self.tracker = UnderwaterPersonTracker(
            max_distance=DETECTION['max_distance'],
            max_disappeared=DETECTION['max_disappeared']
        )
        
        # Analyse de la première frame pour l'eau
        self._analyze_water_first_frame()
        
        return True
    
    def _analyze_water_first_frame(self):
        """Analyse la première frame pour détecter l'eau"""
        if not self.model_manager.has_water_model():
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if not ok:
            return
        
        self.water_detector.compute_water_and_homography(frame, self.model_manager.nwsd)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def recalculate_water_detection(self) -> bool:
        """
        Recalcule la détection d'eau sur la frame actuelle
        Sans interrompre la lecture vidéo
        
        Returns:
            bool: True si le recalcul réussit
        """
        if not hasattr(self, 'cap') or self.cap is None:
            print("Aucune vidéo chargée")
            return False
        
        # On utilise une nouvelle capture temporaire pour éviter les conflits
        try:
            temp_cap = cv2.VideoCapture(self.video_path)
            if not temp_cap.isOpened():
                print("[Water] Impossible d'ouvrir la vidéo temporaire")
                return False
            
            # Se positionner à la frame actuelle (ou proche)
            current_frame = getattr(self, 'current_frame', 0)
            temp_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 1))
            
            ok, frame = temp_cap.read()
            temp_cap.release()  # Libérer immédiatement
            
            if not ok:
                print("[Water] Impossible de lire la frame pour recalcul")
                return False
            
            # Recalculer l'homographie avec la frame obtenue
            ok2 = self.water_detector.compute_water_and_homography(frame, self.model_manager.nwsd)
            
            if ok2:
                self.show_water_detection = True
                print("[Water] Zone d'eau recalculée avec succès (sans interruption)")
            
            return ok2
            
        except Exception as e:
            print(f"[Water] Erreur lors du recalcul: {e}")
            return False
    
    def run(self):
        """Boucle principale de traitement vidéo"""
        if not self.video_path or not hasattr(self, "cap"):
            return
        
        self.is_running = True
        start_time = time.time()
        frame_idx = 0
        frame_time = 1.0 / max(self.fps, 1e-6)
        
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            ok, frame = self.cap.read()
            if not ok:
                # Retour au début de la vidéo
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_idx += 1
            self.current_frame = frame_idx
            self.frame_timestamp = start_time + (frame_idx / self.fps)
            
            # Si pas d'homographie, affichage basique
            if not self.water_detector.has_homography():
                self.frameReady.emit(frame)
                time.sleep(frame_time)
                continue
            
            # Détection et tracking
            persons = self.model_manager.detect_persons(frame, self.conf_threshold)
            assignments = self.tracker.update(persons, self.frame_timestamp)
            
            # Récupération des tracks
            active_tracks = self.tracker.get_active_tracks()
            underwater_tracks = self.tracker.get_underwater_tracks()
            danger_tracks = self.tracker.get_danger_tracks(now=self.frame_timestamp)
            
            # Gestion des alertes
            self._handle_danger_alerts(danger_tracks)
            
            # Création de la minimap
            map_canvas = self._create_minimap(active_tracks, danger_tracks, frame_idx)
            
            # Mise à jour des distances et scores
            self._update_tracks_positions(active_tracks)
            self._update_inactive_tracks_positions()
            
            # Force une mise à jour des scores avant calcul des stats
            self.tracker._update_dangerosity_scores(self.frame_timestamp)
            
            # Création de la frame finale
            vis = self._draw_overlays(frame, persons, assignments, active_tracks, 
                                    underwater_tracks, danger_tracks, map_canvas)
            
            # Statistiques
            all_tracks = self.tracker.get_all_tracks()
            max_score = 0
            max_score_id = None
            
            for tid, track in all_tracks.items():
                score = track.get('dangerosity_score', 0)
                if score > max_score:
                    max_score = score
                    max_score_id = tid
            
            stats = {
                'active': len(active_tracks),
                'underwater': len(underwater_tracks),
                'danger': len(danger_tracks),
                'max_score': max_score,
                'max_score_id': max_score_id,
            }
            
            # Émission des signaux
            self.frameReady.emit(vis)
            self.statsReady.emit(stats)
            
            # Contrôle du timing
            elapsed = time.time() - start_time - (frame_idx / self.fps)
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
    
    def _handle_danger_alerts(self, danger_tracks):
        """Gère les alertes de danger"""
        for tid, t in danger_tracks.items():
            if not t['voice_alert_sent']:
                msg = f"DANGER: Baigneur {tid} en danger!"
                speak_alert("danger")
                self.alert_popup.add_alert(msg, duration=8.0)
                t['voice_alert_sent'] = True
                self.alertTriggered.emit(msg)
    
    def _create_minimap(self, active_tracks, danger_tracks, frame_idx):
        """Crée la minimap avec les positions des baigneurs"""
        map_canvas = np.full((MAP_H_PX, MAP_W_PX, 3), 50, np.uint8)
        
        # Import nécessaire pour les couleurs
        from core.constants import DANGER_COLOR
        from utils.danger import get_color_by_dangerosity
        import math
        
        # Croix de plongée pour tous les tracks
        for tid, t in self.tracker.get_all_tracks().items():
            if t.get('dive_point') is None:
                continue
            
            dive_pos = self.water_detector.transform_point_to_minimap(t['dive_point'])
            if dive_pos is None:
                continue
            
            x, y = int(dive_pos[0]), int(dive_pos[1])
            if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                col = DANGER_COLOR if tid in danger_tracks else get_color_by_dangerosity(t.get('dangerosity_score', 0))
                cv2.drawMarker(map_canvas, (x, y), col, cv2.MARKER_CROSS, 10, 2)
        
        # Points actifs et traces
        for tid, t in active_tracks.items():
            if not t.get('history'):
                continue
            
            # Position actuelle
            current_pos = self.water_detector.transform_point_to_minimap(t['history'][-1])
            if current_pos is None:
                continue
            
            x, y = int(current_pos[0]), int(current_pos[1])
            if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                color = get_color_by_dangerosity(t['dangerosity_score'])
                
                if tid in danger_tracks and t['status'] == 'underwater':
                    cv2.circle(map_canvas, (x, y), 6, color, -1)
                    pulse = 8 + int(2 * math.sin(frame_idx * 0.3))
                    cv2.circle(map_canvas, (x, y), pulse, DANGER_COLOR, 2)
                else:
                    cv2.circle(map_canvas, (x, y), 4, color, -1)
                
                cv2.putText(map_canvas, f"{tid}({t['dangerosity_score']})",
                          (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Traces de mouvement
            if len(t['history']) > 1:
                color = get_color_by_dangerosity(t.get('dangerosity_score', 0))
                pts = []
                
                for hist_pos in t['history'][-15:]:
                    map_pos = self.water_detector.transform_point_to_minimap(hist_pos)
                    if map_pos is None:
                        continue
                    
                    x, y = int(map_pos[0]), int(map_pos[1])
                    if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                        pts.append((x, y))
                
                if len(pts) > 1:
                    cv2.polylines(map_canvas, [np.array(pts, np.int32)], False, color, 1)
        
        return map_canvas
    
    def _update_tracks_positions(self, active_tracks):
        """Met à jour les positions des tracks actifs dans la minimap"""
        for tid, t in active_tracks.items():
            if not t.get('history'):
                continue
            
            current_pos = self.water_detector.transform_point_to_minimap(t['history'][-1])
            if current_pos is None:
                continue
            
            x, y = current_pos
            if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                d = calculate_distance_from_shore(x, y, MAP_W_PX, MAP_H_PX)
                t['distance_from_shore'] = d
    
    def _update_inactive_tracks_positions(self):
        """Met à jour les positions des tracks inactifs"""
        active_tracks = self.tracker.get_active_tracks()
        
        for tid, t in self.tracker.get_all_tracks().items():
            if tid in active_tracks or not t.get('center'):
                continue
            
            pos = self.water_detector.transform_point_to_minimap(t['center'])
            if pos is None:
                continue
            
            x, y = pos
            if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                d = calculate_distance_from_shore(x, y, MAP_W_PX, MAP_H_PX)
                t['distance_from_shore'] = d
    
    def _draw_overlays(self, frame, persons, assignments, active_tracks, 
                      underwater_tracks, danger_tracks, map_canvas):
        """Dessine tous les overlays sur la frame"""
        # Import local pour éviter les dépendances circulaires
        import cv2
        import math
        from core.constants import DANGER_COLOR
        from utils.danger import get_color_by_dangerosity
        
        vis = frame.copy()
        
        # Overlay de détection d'eau
        if self.show_water_detection:
            water_mask = self.water_detector.get_water_mask()
            src_quad = self.water_detector.get_source_quad()
            
            if water_mask is not None:
                overlay = np.zeros_like(vis)
                overlay[water_mask > 0] = [255, 100, 0]
                vis = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)
            
            if src_quad is not None:
                q = src_quad.astype(np.int32)
                cv2.polylines(vis, [q], True, (0, 255, 0), 3)
                for i, p in enumerate(q):
                    cv2.circle(vis, tuple(p), 8, (0, 255, 0), -1)
                    cv2.putText(vis, f"{i+1}", (p[0]+10, p[1]-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Boîtes de détection des personnes
        for idx, p in enumerate(persons):
            cx, cy, w, h = p.xywh[0]
            conf = float(p.conf[0].item())
            x0, y0, x1, y1 = int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)
            
            tid = assignments.get(idx, -1)
            
            if tid != -1 and tid in active_tracks:
                t = active_tracks[tid]
                score = t.get('dangerosity_score', 0)
                color = get_color_by_dangerosity(score)
                is_danger = tid in danger_tracks
                
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, 4 if is_danger else 2)
                
                if t['status'] == 'underwater':
                    dur = self.frame_timestamp - (t.get('underwater_start_time') or self.frame_timestamp)
                    label = f"ID:{tid} (UNDERWATER) - Score:{score} | {dur:.1f}s"
                else:
                    label = f"ID:{tid} - Score:{score}"
                
                sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis, (x0, y0-35), (x0+sz[0]+10, y0-5), (0, 0, 0), -1)
                cv2.putText(vis, label, (x0+5, y0-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(vis, f"Conf:{conf:.2f}", (x0, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(vis, f"New:{conf:.2f}", (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Croix de danger aux points de plongée
        for tid in danger_tracks:
            t = self.tracker.tracks[tid]
            if t.get('dive_point') is None:
                continue
            dx, dy = map(int, t['dive_point'])
            cv2.drawMarker(vis, (dx, dy), DANGER_COLOR, cv2.MARKER_CROSS, 20, 3)
        
        # Minimap incrustée
        if map_canvas is not None:
            mh, mw = map_canvas.shape[:2]
            y0, x0 = 10, vis.shape[1] - mw - 10
            cv2.rectangle(vis, (x0-5, y0-5), (x0+mw+5, y0+mh+5), (0, 0, 0), -1)
            cv2.rectangle(vis, (x0-5, y0-5), (x0+mw+5, y0+mh+5), (255, 255, 255), 2)
            vis[y0:y0+mh, x0:x0+mw] = map_canvas
            cv2.putText(vis, "MINIMAP", (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alertes actives (HUD)
        self.alert_popup.update()
        alerts = self.alert_popup.get_active_alerts()
        if alerts:
            base_y = vis.shape[0] - 200
            height = min(len(alerts), 3) * 35 + 20
            cv2.rectangle(vis, (20, base_y-10), (600, base_y+height), (0, 0, 0), -1)
            cv2.rectangle(vis, (20, base_y-10), (600, base_y+height), (255, 0, 0), 2)
            cv2.putText(vis, "ALERTES ACTIVES:", (30, base_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, a in enumerate(alerts[-3:]):
                col = (0, 0, 255) if "DANGER" in a else (255, 165, 0)
                cv2.putText(vis, f"• {a}", (40, base_y+45 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        
        return vis
    
    # Contrôles du thread
    def set_video_path(self, path):
        """Définit le chemin de la vidéo"""
        self.video_path = path
    
    def pause(self):
        """Met en pause ou reprend la lecture"""
        self.is_paused = not self.is_paused
    
    def stop(self):
        """Arrête le traitement vidéo de manière sécurisée"""
        print("[VideoProcessor] Arrêt demandé...")
        self.is_running = False
        
        # Attendre que le thread se termine proprement
        if self.isRunning():
            self.wait(2000)  # Attendre max 2 secondes
            
        # Libérer les ressources vidéo
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                with self._lock:
                    self.cap.release()
                    self.cap = None
                print("[VideoProcessor] Ressources vidéo libérées")
            except Exception as e:
                print(f"[VideoProcessor] Erreur libération: {e}")
        
        print("[VideoProcessor] Arrêt terminé")
