#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Detection Models (API version with GPU optimization)
"""

import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ===== AI Dependencies =====
try:
    from transformers import AutoImageProcessor, DFineForObjectDetection
    from ultralytics import YOLO
    HAS_AI_MODELS = True
except Exception:
    logger.warning("AI models not available. Running in demo mode.")
    HAS_AI_MODELS = False


class BoxStub:
    """Stub for detection results"""
    
    def __init__(self, cx, cy, w, h, conf):
        self.xywh = torch.tensor([[cx, cy, w, h]])
        self.conf = torch.tensor([conf])
        self.cls = torch.tensor([0])  # person


@torch.inference_mode()
def detect_persons_dfine(frame_bgr, processor, dfine, device, conf_thres):
    """
    Detect persons in a frame with D-FINE
    
    Args:
        frame_bgr: Frame in BGR format
        processor: D-FINE image processor
        dfine: D-FINE model
        device: Computing device (cuda/cpu)
        conf_thres: Confidence threshold
    
    Returns:
        list: List of detections (BoxStub)
    """
    # Convert BGR to RGB for processor
    inputs = processor(images=frame_bgr[:, :, ::-1], return_tensors="pt").to(device)
    
    # Use FP16 for GPU inference
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    
    outputs = dfine(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])], threshold=conf_thres
    )[0]
    
    persons = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        if label.item() == 0:  # person class
            x0, y0, x1, y1 = box.tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            persons.append(BoxStub(cx, cy, x1 - x0, y1 - y0, score.item()))
    
    return persons


class ModelManager:
    """
    AI Model Manager with GPU optimization
    """
    
    def __init__(self, use_gpu=True, use_mixed_precision=True):
        self.nwsd = None  # Water detection model
        self.dfine = None  # Person detection model
        self.processor = None  # D-FINE processor
        self.use_mixed_precision = use_mixed_precision
        
        # Determine device
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"🎮 GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA version: {torch.version.cuda}")
            logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.device = "cpu"
            logger.warning("⚠️  GPU not available, using CPU")
        
        self._model_warning_shown = False
    
    def load_models(self, water_model_path=None):
        """
        Load all AI models with GPU optimization
        
        Args:
            water_model_path: Path to water detection model
        
        Returns:
            bool: True if loading succeeds
        """
        if not HAS_AI_MODELS:
            logger.error("AI models dependencies not available")
            return False
        
        try:
            # Load water detection model
            self._load_water_detection_model(water_model_path)
            
            # Load person detection model
            self._load_person_detection_model()
            
            logger.info(f"✅ Models loaded - NWSD:{'OK' if self.nwsd else 'NO'} D-FINE:{'OK' if self.dfine else 'NO'} Device:{self.device}")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}", exc_info=True)
            return False
    
    def _load_water_detection_model(self, model_path=None):
        """
        Load water detection model (YOLO)
        
        Args:
            model_path: Path to model file
        """
        if model_path is None:
            # Search for model in common locations
            candidates = [
                Path("api/model/nwd-v2.pt"),
                Path("model/nwd-v2.pt"),
                Path("app/model/nwd-v2.pt"),
            ]
            
            for p in candidates:
                if p.exists():
                    model_path = p
                    break
        
        if model_path is None or not Path(model_path).exists():
            logger.warning(f"⚠️  Water detection model not found")
            return
        
        try:
            self.nwsd = YOLO(str(model_path))
            
            # Move to GPU if available
            if self.device == "cuda":
                self.nwsd.to(self.device)
            
            logger.info(f"✅ Water detection model loaded from {model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading water model: {e}")
            self.nwsd = None
    
    def _load_person_detection_model(self):
        """
        Load person detection model (D-FINE) with GPU optimization
        """
        model_id = "ustc-community/dfine-xlarge-obj2coco"
        
        try:
            logger.info(f"Loading D-FINE model: {model_id}")
            
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            
            # Use FP16 for GPU, FP32 for CPU
            dtype = torch.float16 if (self.device == "cuda" and self.use_mixed_precision) else torch.float32
            
            self.dfine = DFineForObjectDetection.from_pretrained(
                model_id,
                torch_dtype=dtype
            ).to(self.device).eval()
            
            # Optimize for inference
            if self.device == "cuda":
                self.dfine = torch.compile(self.dfine, mode="reduce-overhead")
            
            logger.info(f"✅ D-FINE model loaded with dtype={dtype}")
        except Exception as e:
            logger.error(f"❌ Error loading D-FINE model: {e}")
            self.dfine = None
            self.processor = None
    
    def detect_persons(self, frame, conf_threshold):
        """
        Detect persons in a frame
        
        Args:
            frame: Frame to analyze (BGR format)
            conf_threshold: Confidence threshold
        
        Returns:
            list: List of detections
        """
        if self.dfine is None or self.processor is None:
            if not self._model_warning_shown:
                logger.warning("D-FINE model not loaded")
                self._model_warning_shown = True
            return []
        
        return detect_persons_dfine(frame, self.processor, self.dfine, self.device, conf_threshold)
    
    def has_water_model(self):
        """Check if water model is available"""
        return self.nwsd is not None
    
    def has_person_model(self):
        """Check if person model is available"""
        return self.dfine is not None and self.processor is not None
    
    def get_device_info(self):
        """Get device information"""
        info = {
            "device": self.device,
            "mixed_precision": self.use_mixed_precision if self.device == "cuda" else False,
        }
        
        if self.device == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "allocated_memory_gb": torch.cuda.memory_allocated(0) / 1e9,
            })
        
        return info
