"""
src/mastitis_detector.py
========================
KsheerRaksha-AI (Phase 2 — Modular Udder Detection Interface)

Provides an extensible, pluggable interface for locating and cropping
cattle udder/teat regions from full RGB camera frames.

Pipeline Flow:
  Full RGB Video Frame -> Udder Detector -> Bounding Box -> Udder Crop Tensor
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

class BaseUdderDetector(ABC):
    """
    Abstract Base Class for Cattle Udder Localization.
    Any future YOLOv8/YOLOv11-Nano, Faster-RCNN, or keypoint detector must implement this interface.
    """
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects the udder region in a single BGR/RGB frame.
        
        Args:
            frame: numpy array of shape (H, W, 3), uint8.
            
        Returns:
            Dict containing:
              - 'detected': bool
              - 'bbox': [x1, y1, x2, y2] in pixel coordinates or None
              - 'confidence': float (0.0 to 1.0)
              - 'crop': numpy array of cropped region or None
              - 'detector_type': str
        """
        pass


class AnatomicalHeuristicUdderDetector(BaseUdderDetector):
    """
    Anatomical Prior / Motion-Region Udder Localizer.
    
    In controlled lane-view cameras (lateral or rear walkway), the cattle udder
    consistently occupies the lower-middle quadrant between the hind legs:
      - X range: 30% to 75% of frame width
      - Y range: 45% to 85% of frame height
      
    This baseline detector extracts the standardized anatomical region of interest (ROI)
    for downstream feature extraction when a trained YOLO weight checkpoint is not loaded.
    """
    
    def __init__(
        self,
        roi_x_range: Tuple[float, float] = (0.28, 0.72),
        roi_y_range: Tuple[float, float] = (0.42, 0.85),
        min_crop_size: Tuple[int, int] = (64, 64)
    ):
        self.roi_x_range = roi_x_range
        self.roi_y_range = roi_y_range
        self.min_crop_size = min_crop_size
        
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {
                "detected": False,
                "bbox": None,
                "confidence": 0.0,
                "crop": None,
                "detector_type": "AnatomicalHeuristic (Empty Frame)"
            }
            
        h, w = frame.shape[:2]
        x1 = max(0, int(w * self.roi_x_range[0]))
        x2 = min(w, int(w * self.roi_x_range[1]))
        y1 = max(0, int(h * self.roi_y_range[0]))
        y2 = min(h, int(h * self.roi_y_range[1]))
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        if crop_w < self.min_crop_size[0] or crop_h < self.min_crop_size[1]:
            return {
                "detected": False,
                "bbox": None,
                "confidence": 0.0,
                "crop": None,
                "detector_type": "AnatomicalHeuristic (Frame Too Small)"
            }
            
        crop = frame[y1:y2, x1:x2].copy()
        
        return {
            "detected": True,
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.85, # Prior confidence for standard lane geometry
            "crop": crop,
            "detector_type": "AnatomicalPriorLocalizer"
        }


class YOLOUdderDetectorStub(BaseUdderDetector):
    """
    Extensible Stub for future YOLOv8 / YOLOv11-Nano Udder Object Detection.
    When weights (.pt / ONNX) are trained on annotated cattle udder bounding boxes,
    this class loads ultralytics / ONNXRuntime directly without altering pipeline code.
    """
    
    def __init__(self, model_weights_path: Optional[str] = None, conf_threshold: float = 0.40):
        self.model_weights_path = model_weights_path
        self.conf_threshold = conf_threshold
        self.is_loaded = False
        self.model = None
        
        if model_weights_path and self._check_weights(model_weights_path):
            self._load_model()
            
    def _check_weights(self, path: str) -> bool:
        import os
        return os.path.exists(path)
        
    def _load_model(self):
        try:
            # Future ultralytics integration:
            # from ultralytics import YOLO
            # self.model = YOLO(self.model_weights_path)
            # self.is_loaded = True
            pass
        except Exception as e:
            self.is_loaded = False
            
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None:
            # Graceful fallback to anatomical heuristic
            fallback = AnatomicalHeuristicUdderDetector()
            res = fallback.detect(frame)
            res["detector_type"] = "YOLO_Stub (Fallback to Anatomical Prior - No .pt weights loaded)"
            return res
            
        # Future inference:
        # results = self.model(frame, verbose=False, conf=self.conf_threshold)
        # ... extract best udder box ...
        return {
            "detected": False,
            "bbox": None,
            "confidence": 0.0,
            "crop": None,
            "detector_type": "YOLO_Trained"
        }
