#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Alert System
- Gestion des alertes popup
"""

import time


class AlertPopup:
    """Gestionnaire d'alertes avec durée d'affichage"""
    
    def __init__(self, duration=5.0):
        """
        Initialise le système d'alertes
        
        Args:
            duration: Durée par défaut d'affichage des alertes (secondes)
        """
        self.alerts = []  # (msg, ts, dur)
        self.default_duration = duration
    
    def add_alert(self, msg, duration=None):
        """
        Ajoute une nouvelle alerte
        
        Args:
            msg: Message de l'alerte
            duration: Durée d'affichage spécifique (optionnel)
        """
        self.alerts.append((msg, time.time(), duration or self.default_duration))
    
    def update(self):
        """Met à jour la liste des alertes (supprime les expirées)"""
        now = time.time()
        self.alerts = [(m, ts, d) for (m, ts, d) in self.alerts if now - ts < d]
    
    def get_active_alerts(self):
        """
        Retourne la liste des alertes actives
        
        Returns:
            list: Liste des messages d'alertes actives
        """
        return [m for m, _, _ in self.alerts]
    
    def clear_all(self):
        """Supprime toutes les alertes"""
        self.alerts.clear()
    
    def get_alert_count(self):
        """
        Retourne le nombre d'alertes actives
        
        Returns:
            int: Nombre d'alertes actives
        """
        return len(self.alerts)
