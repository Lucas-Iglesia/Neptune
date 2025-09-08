#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Person Tracker
- Suivi des personnes et détection de noyade
"""

import math
import time
import numpy as np
from config_pyqt6 import DETECTION, ALERTS
from utils.danger import calculate_dangerosity_score

# Récupération des seuils
UNDERWATER_THRESHOLD = DETECTION['underwater_threshold']
SURFACE_THRESHOLD = DETECTION['surface_threshold']
DANGER_TIME_THRESHOLD = ALERTS['danger_threshold']


class UnderwaterPersonTracker:
    """Tracker spécialisé pour la détection de noyade"""
    
    def __init__(self, max_distance=None, max_disappeared=None):
        """
        Initialise le tracker
        
        Args:
            max_distance: Distance maximale pour associer une détection à un track
            max_disappeared: Nombre max de frames avant suppression d'un track
        """
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance or DETECTION['max_distance']
        self.max_disappeared = max_disappeared or DETECTION['max_disappeared']
    
    def _init_track(self, center, ts):
        """
        Initialise un nouveau track
        
        Args:
            center: Position (x, y) du centre
            ts: Timestamp
        
        Returns:
            dict: Nouveau track initialisé
        """
        return {
            'center': center,
            'disappeared': 0,
            'history': [center],
            'status': 'surface',
            'frames_underwater': 0,
            'frames_on_surface': 0,
            'last_seen_surface': ts,
            'underwater_start_time': None,
            'underwater_duration': 0,
            'submersion_events': [],
            'danger_alert_sent': False,
            'voice_alert_sent': False,
            'dangerosity_score': 0,
            'distance_from_shore': 0.0,
            'dive_point': None
        }
    
    def update(self, detections, frame_ts=None):
        """
        Met à jour le tracker avec les nouvelles détections
        
        Args:
            detections: Liste des détections de personnes
            frame_ts: Timestamp de la frame
        
        Returns:
            dict: Mapping detection_index -> track_id
        """
        frame_ts = frame_ts or time.time()
        
        # Pas de détection: tout le monde potentiellement sous l'eau
        if not detections:
            return self._handle_no_detections(frame_ts)
        
        det_centers = [(float(d.xywh[0][0]), float(d.xywh[0][1])) for d in detections]
        
        # Premier cas: pas de tracks existants
        if not self.tracks:
            return self._create_initial_tracks(det_centers, frame_ts)
        
        # Association détections <-> tracks
        assignments = self._assign_detections_to_tracks(det_centers, frame_ts)
        
        # Création de nouveaux tracks pour détections non assignées
        self._create_new_tracks_for_unassigned(det_centers, assignments, frame_ts)
        
        # Gestion des tracks non assignés (disparus)
        self._handle_disappeared_tracks(frame_ts)
        
        # Suppression des tracks trop anciens
        self._remove_old_tracks(frame_ts)
        
        # Mise à jour des scores de dangerosité
        self._update_dangerosity_scores(frame_ts)
        
        return assignments
    
    def _handle_no_detections(self, frame_ts):
        """Gère le cas où aucune détection n'est trouvée"""
        to_remove = []
        
        for tid, t in self.tracks.items():
            t['disappeared'] += 1
            t['frames_underwater'] += 1
            t['frames_on_surface'] = 0
            
            if t['frames_underwater'] >= UNDERWATER_THRESHOLD:
                if t['status'] != 'underwater':
                    t['status'] = 'underwater'
                    t['underwater_start_time'] = frame_ts
                    t['dive_point'] = t['center']
                    t['danger_alert_sent'] = False
                    t['voice_alert_sent'] = False
                    print(f"Person {tid} UNDERWATER")
                
                if t['underwater_start_time']:
                    t['underwater_duration'] = frame_ts - t['underwater_start_time']
                    if t['underwater_duration'] > DANGER_TIME_THRESHOLD and not t['danger_alert_sent']:
                        print(f"DANGER ALERT: Person {tid} underwater {t['underwater_duration']:.1f}s")
                        t['danger_alert_sent'] = True
            
            if t['disappeared'] > self.max_disappeared:
                if t['status'] == 'underwater' and t['underwater_start_time']:
                    dur = frame_ts - t['underwater_start_time']
                    t['submersion_events'].append((t['underwater_start_time'], dur))
                to_remove.append(tid)
        
        for tid in to_remove:
            del self.tracks[tid]
        
        return {}
    
    def _create_initial_tracks(self, det_centers, frame_ts):
        """Crée les premiers tracks"""
        assignments = {}
        for i, c in enumerate(det_centers):
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = self._init_track(c, frame_ts)
            assignments[i] = tid
        return assignments
    
    def _assign_detections_to_tracks(self, det_centers, frame_ts):
        """Assigne les détections aux tracks existants"""
        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[tid]['center'] for tid in track_ids]
        
        # Calcul des distances
        distances = np.zeros((len(track_centers), len(det_centers)))
        for i, tc in enumerate(track_centers):
            for j, dc in enumerate(det_centers):
                distances[i, j] = math.hypot(tc[0] - dc[0], tc[1] - dc[1])
        
        # Création des paires possibles
        pairs = [(distances[i, j], i, j)
                 for i in range(len(track_centers))
                 for j in range(len(det_centers))
                 if distances[i, j] < self.max_distance]
        pairs.sort(key=lambda x: x[0])
        
        # Association greedy
        assignments, used_t, used_d = {}, set(), set()
        for _, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            
            tid = track_ids[i]
            t = self.tracks[tid]
            
            # Mise à jour du track
            t['center'] = det_centers[j]
            t['disappeared'] = 0
            t['history'].append(det_centers[j])
            t['last_seen_surface'] = frame_ts
            t['frames_on_surface'] += 1
            t['frames_underwater'] = 0
            
            # Retour en surface
            if t['status'] == 'underwater' and t['frames_on_surface'] >= SURFACE_THRESHOLD:
                if t['underwater_start_time']:
                    dur = frame_ts - t['underwater_start_time']
                    t['submersion_events'].append((t['underwater_start_time'], dur))
                t['status'] = 'surface'
                t['underwater_start_time'] = None
                t['danger_alert_sent'] = False
                t['voice_alert_sent'] = False
            
            # Limitation de l'historique
            if len(t['history']) > 50:
                t['history'] = t['history'][-50:]
            
            assignments[j] = tid
            used_t.add(i)
            used_d.add(j)
        
        return assignments
    
    def _create_new_tracks_for_unassigned(self, det_centers, assignments, frame_ts):
        """Crée de nouveaux tracks pour les détections non assignées"""
        for j in range(len(det_centers)):
            if j not in assignments:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = self._init_track(det_centers[j], frame_ts)
                assignments[j] = tid
    
    def _handle_disappeared_tracks(self, frame_ts):
        """Gère les tracks non assignés (disparus)"""
        assigned_track_ids = set()
        # On récupère les tracks qui ont été assignés dans cette frame
        # (cette logique est déjà gérée dans _assign_detections_to_tracks)
        
        for tid, t in self.tracks.items():
            if t['disappeared'] > 0:  # Track non assigné dans cette frame
                t['disappeared'] += 1
                t['frames_underwater'] += 1
                t['frames_on_surface'] = 0
                
                if t['frames_underwater'] >= UNDERWATER_THRESHOLD:
                    if t['status'] != 'underwater':
                        t['status'] = 'underwater'
                        t['underwater_start_time'] = frame_ts
                        t['dive_point'] = t['center']
                        t['danger_alert_sent'] = False
                        t['voice_alert_sent'] = False
                        print(f"Person {tid} UNDERWATER")
                    
                    if t['underwater_start_time']:
                        t['underwater_duration'] = frame_ts - t['underwater_start_time']
                        if t['underwater_duration'] > DANGER_TIME_THRESHOLD and not t['danger_alert_sent']:
                            print(f"DANGER ALERT: Person {tid} underwater {t['underwater_duration']:.1f}s")
                            t['danger_alert_sent'] = True
                
                # Mise à jour du score pour les tracks disparus aussi
                t['dangerosity_score'] = calculate_dangerosity_score(
                    t, frame_ts, t.get('distance_from_shore', 0.0)
                )
    
    def _remove_old_tracks(self, frame_ts):
        """Supprime les tracks trop anciens"""
        to_remove = []
        for tid, t in self.tracks.items():
            if t['disappeared'] > self.max_disappeared:
                if t['status'] == 'underwater' and t['underwater_start_time']:
                    dur = frame_ts - t['underwater_start_time']
                    t['submersion_events'].append((t['underwater_start_time'], dur))
                to_remove.append(tid)
        
        for tid in to_remove:
            del self.tracks[tid]
    
    def _update_dangerosity_scores(self, frame_ts):
        """Met à jour les scores de dangerosité de tous les tracks"""
        for tid, t in self.tracks.items():
            old_score = t.get('dangerosity_score', 0)
            t['dangerosity_score'] = calculate_dangerosity_score(
                t, frame_ts, t.get('distance_from_shore', 0.0)
            )
            # Debug pour voir l'évolution des scores
            if t['dangerosity_score'] != old_score:
                print(f"[DEBUG] Track {tid}: score {old_score} -> {t['dangerosity_score']} (status: {t['status']}, frames_underwater: {t['frames_underwater']})")
    
    def get_active_tracks(self):
        """
        Retourne les tracks actifs (récemment vus)
        
        Returns:
            dict: Tracks actifs {track_id: track_data}
        """
        return {tid: t for tid, t in self.tracks.items() if t['disappeared'] <= UNDERWATER_THRESHOLD}
    
    def get_underwater_tracks(self):
        """
        Retourne les tracks sous l'eau
        
        Returns:
            dict: Tracks sous l'eau {track_id: track_data}
        """
        return {tid: t for tid, t in self.tracks.items() if t['status'] == 'underwater'}
    
    def get_danger_tracks(self, now=None):
        """
        Retourne les tracks en danger (sous l'eau trop longtemps)
        
        Args:
            now: Timestamp actuel (optionnel)
        
        Returns:
            dict: Tracks en danger {track_id: track_data}
        """
        now = now or time.time()
        return {
            tid: t for tid, t in self.tracks.items()
            if (t['status'] == 'underwater' and t['underwater_start_time']
                and (now - t['underwater_start_time']) > DANGER_TIME_THRESHOLD)
        }
    
    def get_all_tracks(self):
        """
        Retourne tous les tracks
        
        Returns:
            dict: Tous les tracks {track_id: track_data}
        """
        return self.tracks.copy()
    
    def reset(self):
        """Remet à zéro le tracker"""
        self.tracks.clear()
        self.next_id = 1
