#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Color and Danger Utilities
- Calcul du score de danger
- Attribution des couleurs selon le niveau de danger
"""

from config_pyqt6 import DETECTION, ALERTS

# Récupération des seuils
UNDERWATER_THRESHOLD = DETECTION['underwater_threshold']
DANGER_TIME_THRESHOLD = ALERTS['danger_threshold']


def calculate_dangerosity_score(track, frame_ts, dist_from_shore=0.0):
    """
    Calcule le score de dangerosité d'un tracker
    
    Args:
        track: Dictionnaire contenant les informations du tracker
        frame_ts: Timestamp de la frame actuelle
        dist_from_shore: Distance par rapport au rivage (0.0 à 1.0)
    
    Returns:
        int: Score de danger de 0 à 100
    """
    score = int(dist_from_shore * 20)  # 0..20
    
    if track['frames_underwater'] > 0:
        prog = min(track['frames_underwater'] / UNDERWATER_THRESHOLD, 1.0)
        score += int(10 + 20 * prog)  # 10..30
        
        if track['status'] == 'underwater':
            score += 20
            
            if track['underwater_start_time']:
                t = frame_ts - track['underwater_start_time']
                if t > DANGER_TIME_THRESHOLD:
                    score += 40
                    if not track['danger_alert_sent']:
                        print(f"DANGER ALERT: underwater {t:.1f}s")
                        track['danger_alert_sent'] = True
                else:
                    score += int((t / DANGER_TIME_THRESHOLD) * 40)
        
        if track['frames_underwater'] > UNDERWATER_THRESHOLD:
            score += min(10, (track['frames_underwater'] - UNDERWATER_THRESHOLD) // 10)
    
    return min(100, score)


def get_color_by_dangerosity(score: int) -> tuple[int, int, int]:
    """
    Retourne une couleur BGR selon le score de dangerosité
    
    Args:
        score: Score de danger (0-100)
    
    Returns:
        tuple: Couleur BGR (B, G, R)
    """
    if score <= 20:
        r = int(144 * (score / 20.0))
        g = int(100 + 138 * (score / 20.0))
        b = r
        return (b, g, r)
    
    if score <= 40:
        ratio = (score - 20) / 20.0
        return (int(144 * (1 - ratio)), int(238 + 17 * ratio), int(144 + 111 * ratio))
    
    if score <= 60:
        ratio = (score - 40) / 20.0
        return (0, int(255 - 90 * ratio), 255)
    
    if score <= 80:
        ratio = (score - 60) / 20.0
        return (0, int(165 * (1 - ratio)), 255)
    
    ratio = (score - 80) / 20.0
    return (0, 0, int(255 - 116 * ratio))


def calculate_distance_from_shore(x, y, w, h) -> float:
    """
    Calcule la distance relative par rapport au rivage
    (rive considérée en bas de la minimap)
    
    Args:
        x, y: Position sur la minimap
        w, h: Dimensions de la minimap
    
    Returns:
        float: Distance normalisée (0.0 = rivage, 1.0 = large)
    """
    return max(0.0, min(1.0, (h - y) / h))
