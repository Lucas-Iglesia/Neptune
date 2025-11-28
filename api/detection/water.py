#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Water Detection and Homography
- Détection de la zone d'eau
- Calcul de l'homographie pour la minimap
"""

import cv2
import numpy as np
from core.constants import MIN_WATER_AREA_PX, DST_RECT


class WaterDetector:
    """Gestionnaire de la détection d'eau et de l'homographie"""
    
    def __init__(self):
        self.water_mask_global = None
        self.src_quad_global = None
        self.H_latest = None  # Matrice d'homographie
    
    def compute_water_and_homography(self, frame, nwsd_model) -> bool:
        """
        Calcule la zone d'eau et l'homographie pour la minimap
        
        Args:
            frame: Frame à analyser
            nwsd_model: Modèle YOLO de détection d'eau
        
        Returns:
            bool: True si le calcul réussit
        """
        if nwsd_model is None:
            return False
        
        try:
            # Segmentation de l'eau
            seg = nwsd_model.predict(frame, imgsz=512, task="segment", conf=0.25, verbose=False)[0]
            if seg.masks is None:
                return False
            
            # Création du masque
            mask = (seg.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.water_mask_global = mask.copy()
            
            # Extraction des contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return False
            
            # Sélection du plus grand contour
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) <= MIN_WATER_AREA_PX:
                return False
            
            # Calcul du quadrilatère source
            pts = cnt.reshape(-1, 2).astype(np.float32)
            sums = pts.sum(axis=1)
            diffs = np.diff(pts, axis=1).reshape(-1)
            
            src_quad = np.array([
                pts[np.argmin(sums)],    # top-left
                pts[np.argmin(diffs)],   # top-right
                pts[np.argmax(sums)],    # bottom-right
                pts[np.argmax(diffs)]    # bottom-left
            ], dtype=np.float32)
            
            # Calcul de l'homographie
            H, _ = cv2.findHomography(src_quad, DST_RECT, cv2.RANSAC, 3.0)
            if H is None:
                return False
            
            self.H_latest = H.copy()
            self.src_quad_global = src_quad.copy()
            return True
            
        except Exception as e:
            print(f"[Water] Erreur homographie: {e}")
            return False
    
    def has_homography(self):
        """Vérifie si une homographie est disponible"""
        return self.H_latest is not None
    
    def transform_point_to_minimap(self, point):
        """
        Transforme un point de la frame vers la minimap
        
        Args:
            point: Point (x, y) dans la frame
        
        Returns:
            tuple: Point transformé (x, y) ou None si échec
        """
        if self.H_latest is None:
            return None
        
        try:
            pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.H_latest).reshape(-1, 2)[0]
            return tuple(transformed)
        except Exception:
            return None
    
    def get_water_mask(self):
        """Retourne le masque d'eau actuel"""
        return self.water_mask_global
    
    def get_source_quad(self):
        """Retourne le quadrilatère source pour l'homographie"""
        return self.src_quad_global
    
    def clear(self):
        """Remet à zéro la détection d'eau"""
        self.water_mask_global = None
        self.src_quad_global = None
        self.H_latest = None
