#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Core Utilities
"""

from config import DETECTION, ALERTS

# Retrieve thresholds
UNDERWATER_THRESHOLD = DETECTION['underwater_threshold']
DANGER_TIME_THRESHOLD = ALERTS['danger_threshold']


def calculate_dangerosity_score(track, frame_ts, dist_from_shore=0.0):
    """
    Calculate the dangerosity score of a tracker
    
    Args:
        track: Dictionary containing tracker information
        frame_ts: Current frame timestamp
        dist_from_shore: Distance from shore (0.0 to 1.0)
    
    Returns:
        int: Danger score from 0 to 100
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
                        track['danger_alert_sent'] = True
                else:
                    score += int((t / DANGER_TIME_THRESHOLD) * 40)
        
        if track['frames_underwater'] > UNDERWATER_THRESHOLD:
            score += min(10, (track['frames_underwater'] - UNDERWATER_THRESHOLD) // 10)
    
    return min(100, score)


def get_color_by_dangerosity(score: int) -> tuple[int, int, int]:
    """
    Return a BGR color according to the dangerosity score
    
    Args:
        score: Danger score (0-100)
    
    Returns:
        tuple: BGR color (B, G, R)
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
    Calculate relative distance from shore
    (shore considered at the bottom of the minimap)
    
    Args:
        x, y: Position on minimap
        w, h: Minimap dimensions
    
    Returns:
        float: Normalized distance (0.0 = shore, 1.0 = far)
    """
    return max(0.0, min(1.0, (h - y) / h))
