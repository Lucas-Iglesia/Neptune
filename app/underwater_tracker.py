"""
Tracker pour la détection de personnes sous l'eau
Adapté pour PyQt6 depuis le Demo-5
"""

import time
import math
import numpy as np
from collections import defaultdict

class BoxStub:
    """Classe simple pour simuler les détections YOLO"""
    def __init__(self, cx, cy, w, h, conf):
        self.xywh = [[cx, cy, w, h]]
        self.conf = [conf]
        self.cls = [0]  # classe 0 = person

class UnderwaterPersonTracker:
    """Tracker amélioré pour la détection de personnes sous l'eau"""
    
    def __init__(self, max_distance=100, max_disappeared=300):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.frame_rate = 30
        
        # Seuils configurables
        self.underwater_threshold = 15  # frames pour considérer sous l'eau
        self.surface_threshold = 5      # frames pour considérer en surface
        self.danger_time_threshold = 5  # secondes sous l'eau avant danger

    def _init_track(self, center, timestamp):
        """Initialiser une nouvelle track"""
        return {
            'center': center,
            'disappeared': 0,
            'history': [center],
            'status': 'surface',
            'frames_underwater': 0,
            'frames_on_surface': 0,
            'last_seen_surface': timestamp,
            'underwater_start_time': None,
            'underwater_duration': 0,
            'submersion_events': [],
            'danger_alert_sent': False,
            'voice_alert_sent': False,
            'dangerosity_score': 0,
            'distance_from_shore': 0.0,
            'dive_point': None
        }

    def update(self, detections, frame_timestamp=None):
        """Mettre à jour le tracker avec les nouvelles détections"""
        if frame_timestamp is None:
            frame_timestamp = time.time()

        # Convertir les détections en format standard si nécessaire
        detection_centers = []
        if detections:
            for det in detections:
                if isinstance(det, dict):
                    cx, cy = det['xywh'][0][0], det['xywh'][0][1]
                else:
                    cx, cy = float(det.xywh[0][0]), float(det.xywh[0][1])
                detection_centers.append((cx, cy))

        # Si pas de détections, mettre à jour le statut sous-marin
        if not detection_centers:
            to_remove = []
            for track_id in self.tracks:
                track = self.tracks[track_id]
                track['disappeared'] += 1
                track['frames_underwater'] += 1
                track['frames_on_surface'] = 0

                # Vérifier si la personne est sous l'eau
                if track['frames_underwater'] >= self.underwater_threshold:
                    if track['status'] != 'underwater':
                        track['status'] = 'underwater'
                        track['underwater_start_time'] = frame_timestamp
                        track['dive_point'] = track['center']
                        track['danger_alert_sent'] = False
                        track['voice_alert_sent'] = False

                    # Calculer la durée sous l'eau
                    if track['underwater_start_time']:
                        track['underwater_duration'] = frame_timestamp - track['underwater_start_time']

                        # Vérifier le seuil de danger
                        if (track['underwater_duration'] > self.danger_time_threshold and 
                            not track['danger_alert_sent']):
                            track['danger_alert_sent'] = True

                # Supprimer les tracks perdues depuis trop longtemps
                if track['disappeared'] > self.max_disappeared:
                    if track['status'] == 'underwater' and track['underwater_start_time']:
                        duration = frame_timestamp - track['underwater_start_time']
                        track['submersion_events'].append((track['underwater_start_time'], duration))
                    to_remove.append(track_id)

            for track_id in to_remove:
                del self.tracks[track_id]

            return {}

        # Si pas de tracks existantes, en créer de nouvelles
        if not self.tracks:
            assignments = {}
            for i, center in enumerate(detection_centers):
                track_id = self.next_id
                self.tracks[track_id] = self._init_track(center, frame_timestamp)
                assignments[i] = track_id
                self.next_id += 1
            return assignments

        # Calcul des distances entre tracks existantes et nouvelles détections
        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[tid]['center'] for tid in track_ids]

        distances = np.zeros((len(track_centers), len(detection_centers)))
        for i, track_center in enumerate(track_centers):
            for j, det_center in enumerate(detection_centers):
                distances[i, j] = math.sqrt(
                    (track_center[0] - det_center[0])**2 +
                    (track_center[1] - det_center[1])**2
                )

        # Attribution glouton
        assignments = {}
        used_tracks = set()
        used_detections = set()

        distance_pairs = []
        for i in range(len(track_centers)):
            for j in range(len(detection_centers)):
                if distances[i, j] < self.max_distance:
                    distance_pairs.append((distances[i, j], i, j))

        distance_pairs.sort(key=lambda x: x[0])

        for dist, track_idx, det_idx in distance_pairs:
            if track_idx not in used_tracks and det_idx not in used_detections:
                track_id = track_ids[track_idx]
                track = self.tracks[track_id]

                # Mettre à jour la track existante
                track['center'] = detection_centers[det_idx]
                track['disappeared'] = 0
                track['history'].append(detection_centers[det_idx])
                track['last_seen_surface'] = frame_timestamp
                track['frames_on_surface'] += 1
                track['frames_underwater'] = 0

                # Vérifier si la personne a refait surface
                if (track['status'] == 'underwater' and 
                    track['frames_on_surface'] >= self.surface_threshold):
                    if track['underwater_start_time']:
                        duration = frame_timestamp - track['underwater_start_time']
                        track['submersion_events'].append((track['underwater_start_time'], duration))
                        track['underwater_duration'] = 0

                    track['status'] = 'surface'
                    track['underwater_start_time'] = None
                    track['danger_alert_sent'] = False
                    track['voice_alert_sent'] = False

                # Garder l'historique gérable
                if len(track['history']) > 50:
                    track['history'] = track['history'][-50:]

                assignments[det_idx] = track_id
                used_tracks.add(track_idx)
                used_detections.add(det_idx)

        # Créer de nouvelles tracks pour les détections non attribuées
        for j in range(len(detection_centers)):
            if j not in used_detections:
                track_id = self.next_id
                self.tracks[track_id] = self._init_track(detection_centers[j], frame_timestamp)
                assignments[j] = track_id
                self.next_id += 1

        # Marquer les tracks non attribuées comme disparues
        for i in range(len(track_centers)):
            if i not in used_tracks:
                track_id = track_ids[i]
                track = self.tracks[track_id]
                track['disappeared'] += 1
                track['frames_underwater'] += 1
                track['frames_on_surface'] = 0
                
                if track['frames_underwater'] >= self.underwater_threshold:
                    if track['status'] != 'underwater':
                        track['status'] = 'underwater'
                        track['underwater_start_time'] = frame_timestamp
                        track['dive_point'] = track['center']
                        track['danger_alert_sent'] = False
                        track['voice_alert_sent'] = False

                    if track['underwater_start_time']:
                        track['underwater_duration'] = frame_timestamp - track['underwater_start_time']

                        if (track['underwater_duration'] > self.danger_time_threshold and 
                            not track['danger_alert_sent']):
                            track['danger_alert_sent'] = True

        # Supprimer les tracks perdues depuis trop longtemps
        to_remove = []
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            if track['disappeared'] > self.max_disappeared:
                if track['status'] == 'underwater' and track['underwater_start_time']:
                    duration = frame_timestamp - track['underwater_start_time']
                    track['submersion_events'].append((track['underwater_start_time'], duration))
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

        # Mettre à jour les scores de dangerosité
        for track_id, track in self.tracks.items():
            track['dangerosity_score'] = self.calculate_dangerosity_score(track, frame_timestamp)

        return assignments

    def calculate_dangerosity_score(self, track, frame_timestamp):
        """Calculer le score de dangerosité (0-100)"""
        score = 0

        # Score de base pour la distance du rivage
        distance_from_shore = track.get('distance_from_shore', 0)
        score += int(distance_from_shore * 20)

        # Score pour la plongée/sous l'eau
        if track['frames_underwater'] > 0:
            diving_progress = min(track['frames_underwater'] / self.underwater_threshold, 1.0)
            score += int(10 + (diving_progress * 20))  # 10-30 pts
            
            if track['status'] == 'underwater':
                score += 20  # 20 pts supplémentaires pour être officiellement sous l'eau
                
                # Facteur temps sous l'eau (0-40 pts)
                if track['underwater_start_time']:
                    t = frame_timestamp - track['underwater_start_time']
                    if t > self.danger_time_threshold:
                        score += 40
                    else:
                        score += int((t / self.danger_time_threshold) * 40)
            
            # Facteur d'excès de frames sous l'eau (0-10 pts)
            if track['frames_underwater'] > self.underwater_threshold:
                excess = track['frames_underwater'] - self.underwater_threshold
                score += min(10, excess // 10)

        return min(100, score)

    def get_active_tracks(self):
        """Obtenir toutes les tracks actuellement actives"""
        return {tid: track for tid, track in self.tracks.items()
                if track['disappeared'] <= self.underwater_threshold}

    def get_underwater_tracks(self):
        """Obtenir les tracks des personnes actuellement sous l'eau"""
        return {tid: track for tid, track in self.tracks.items()
                if track['status'] == 'underwater'}

    def get_danger_tracks(self):
        """Obtenir les tracks des personnes en danger"""
        current_time = time.time()
        danger_tracks = {}
        for tid, track in self.tracks.items():
            if (track['status'] == 'underwater' and 
                track['underwater_start_time'] and
                (current_time - track['underwater_start_time']) > self.danger_time_threshold):
                danger_tracks[tid] = track
        return danger_tracks

def get_color_by_dangerosity(score):
    """Obtenir la couleur basée sur le score de dangerosité"""
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
