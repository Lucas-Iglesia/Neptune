#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Frame Renderer
- Rendu des overlays sur les frames vidéo
"""

import cv2
import numpy as np
from core.constants import MAP_W_PX, MAP_H_PX, DANGER_COLOR
from utils.danger import get_color_by_dangerosity


class FrameRenderer:
    """Gestionnaire du rendu des overlays vidéo"""
    
    def __init__(self, water_detector, alert_popup, show_water_detection=False):
        """
        Initialise le renderer
        
        Args:
            water_detector: Détecteur d'eau
            alert_popup: Gestionnaire d'alertes
            show_water_detection: Afficher la détection d'eau
        """
        self.water_detector = water_detector
        self.alert_popup = alert_popup
        self.show_water_detection = show_water_detection
    
    def render_frame(self, frame, persons, assignments, active_tracks, 
                    underwater_tracks, danger_tracks, map_canvas, frame_timestamp):
        """
        Rend une frame complète avec tous les overlays
        
        Args:
            frame: Frame vidéo de base
            persons: Liste des détections de personnes
            assignments: Mapping détection -> track
            active_tracks: Tracks actifs
            underwater_tracks: Tracks sous l'eau
            danger_tracks: Tracks en danger
            map_canvas: Canvas de la minimap
            frame_timestamp: Timestamp de la frame
        
        Returns:
            np.ndarray: Frame avec overlays
        """
        vis = frame.copy()
        
        # Overlay de détection d'eau
        if self.show_water_detection:
            vis = self._draw_water_overlay(vis)
        
        # Boîtes de détection des personnes
        vis = self._draw_person_boxes(vis, persons, assignments, active_tracks, 
                                    danger_tracks, frame_timestamp)
        
        # Croix de danger aux points de plongée
        vis = self._draw_danger_markers(vis, danger_tracks)
        
        # Minimap incrustée
        vis = self._draw_minimap(vis, map_canvas)
        
        # Alertes actives (HUD)
        vis = self._draw_alerts_hud(vis)
        
        return vis
    
    def _draw_water_overlay(self, vis):
        """Dessine l'overlay de détection d'eau"""
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
        
        return vis
    
    def _draw_person_boxes(self, vis, persons, assignments, active_tracks, 
                          danger_tracks, frame_timestamp):
        """Dessine les boîtes de détection des personnes"""
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
                
                # Rectangle principal
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, 4 if is_danger else 2)
                
                # Label avec informations
                if t['status'] == 'underwater':
                    dur = frame_timestamp - (t.get('underwater_start_time') or frame_timestamp)
                    label = f"ID:{tid} (UNDERWATER) - Score:{score} | {dur:.1f}s"
                else:
                    label = f"ID:{tid} - Score:{score}"
                
                # Fond du label
                sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis, (x0, y0-35), (x0+sz[0]+10, y0-5), (0, 0, 0), -1)
                
                # Texte du label
                cv2.putText(vis, label, (x0+5, y0-15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(vis, f"Conf:{conf:.2f}", (x0, y1+20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # Nouvelle détection
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(vis, f"New:{conf:.2f}", (x0, y0-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis
    
    def _draw_danger_markers(self, vis, danger_tracks):
        """Dessine les marqueurs de danger aux points de plongée"""
        for tid in danger_tracks:
            # Note: Nous devons accéder aux tracks depuis le tracker
            # Cette partie nécessiterait un refactoring pour accéder aux tracks
            pass
    
    def _draw_minimap(self, vis, map_canvas):
        """Incruste la minimap sur la frame"""
        if map_canvas is None:
            return vis
        
        mh, mw = map_canvas.shape[:2]
        y0, x0 = 10, vis.shape[1] - mw - 10
        
        # Bordure
        cv2.rectangle(vis, (x0-5, y0-5), (x0+mw+5, y0+mh+5), (0, 0, 0), -1)
        cv2.rectangle(vis, (x0-5, y0-5), (x0+mw+5, y0+mh+5), (255, 255, 255), 2)
        
        # Minimap
        vis[y0:y0+mh, x0:x0+mw] = map_canvas
        
        # Titre
        cv2.putText(vis, "MINIMAP", (x0, y0-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def _draw_alerts_hud(self, vis):
        """Dessine le HUD des alertes actives"""
        self.alert_popup.update()
        alerts = self.alert_popup.get_active_alerts()
        
        if not alerts:
            return vis
        
        base_y = vis.shape[0] - 200
        height = min(len(alerts), 3) * 35 + 20
        
        # Fond du HUD
        cv2.rectangle(vis, (20, base_y-10), (600, base_y+height), (0, 0, 0), -1)
        cv2.rectangle(vis, (20, base_y-10), (600, base_y+height), (255, 0, 0), 2)
        
        # Titre
        cv2.putText(vis, "ALERTES ACTIVES:", (30, base_y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Alertes (maximum 3)
        for i, a in enumerate(alerts[-3:]):
            col = (0, 0, 255) if "DANGER" in a else (255, 165, 0)
            cv2.putText(vis, f"• {a}", (40, base_y+45 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        
        return vis
