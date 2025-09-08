#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Detection Models
- Chargement et utilisation des modèles IA
- Détection de personnes et d'eau
"""

import torch
from pathlib import Path

# ===== Dépendances IA (optionnelles) =====
try:
    from transformers import AutoImageProcessor, DFineForObjectDetection
    from ultralytics import YOLO
    HAS_AI_MODELS = True
except Exception:
    print("Modèles IA non disponibles. Mode démo.")
    HAS_AI_MODELS = False


class BoxStub:
    """Stub pour les résultats de détection"""
    
    def __init__(self, cx, cy, w, h, conf):
        self.xywh = torch.tensor([[cx, cy, w, h]])
        self.conf = torch.tensor([conf])
        self.cls = torch.tensor([0])  # personne


@torch.inference_mode()
def detect_persons_dfine(frame_bgr, processor, dfine, device, conf_thres):
    """
    Détecte les personnes dans une frame avec D-FINE
    
    Args:
        frame_bgr: Frame en format BGR
        processor: Processeur d'images D-FINE
        dfine: Modèle D-FINE
        device: Device de calcul (cuda/cpu)
        conf_thres: Seuil de confiance
    
    Returns:
        list: Liste des détections (BoxStub)
    """
    inputs = processor(images=frame_bgr[:, :, ::-1], return_tensors="pt").to(device)
    
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    
    outputs = dfine(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])], threshold=conf_thres
    )[0]
    
    persons = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        if label.item() == 0:  # classe personne
            x0, y0, x1, y1 = box.tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            persons.append(BoxStub(cx, cy, x1 - x0, y1 - y0, score.item()))
    
    return persons


class ModelManager:
    """Gestionnaire des modèles IA"""
    
    def __init__(self):
        self.nwsd = None  # Modèle de détection d'eau
        self.dfine = None  # Modèle de détection de personnes
        self.processor = None  # Processeur D-FINE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_warning_shown = False
    
    def load_models(self):
        """
        Charge tous les modèles IA
        
        Returns:
            bool: True si le chargement réussit
        """
        if not HAS_AI_MODELS:
            return False
        
        try:
            # Modèle de détection d'eau
            self._load_water_detection_model()
            
            # Modèle de détection de personnes
            self._load_person_detection_model()
            
            print(f"[Models] NWSD:{'OK' if self.nwsd else 'NO'} D-FINE:OK Device:{self.device}")
            return True
        except Exception as e:
            print(f"[Models] Erreur: {e}")
            return False
    
    def _load_water_detection_model(self):
        """Charge le modèle de détection d'eau"""
        mp = Path("model/nwd-v2.pt")
        if not mp.exists():
            mp = Path("demo/Demo-5/model/nwd-v2.pt")
        
        if mp.exists():
            self.nwsd = YOLO(str(mp))
        else:
            print("[Models] Modèle de détection d'eau non trouvé")
    
    def _load_person_detection_model(self):
        """Charge le modèle de détection de personnes"""
        model_id = "ustc-community/dfine-xlarge-obj2coco"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.dfine = DFineForObjectDetection.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
    
    def detect_persons(self, frame, conf_threshold):
        """
        Détecte les personnes dans une frame
        
        Args:
            frame: Frame à analyser
            conf_threshold: Seuil de confiance
        
        Returns:
            list: Liste des détections
        """
        if self.dfine is None or self.processor is None:
            if not self._model_warning_shown:
                print("D-FINE non chargé")
                self._model_warning_shown = True
            return []
        
        return detect_persons_dfine(frame, self.processor, self.dfine, self.device, conf_threshold)
    
    def has_water_model(self):
        """Vérifie si le modèle d'eau est disponible"""
        return self.nwsd is not None
    
    def has_person_model(self):
        """Vérifie si le modèle de personnes est disponible"""
        return self.dfine is not None and self.processor is not None
