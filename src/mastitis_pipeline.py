"""
src/mastitis_pipeline.py
========================
KsheerRaksha-AI (Phase 2 — Integrated Visual Udder Screening Service)

Integrates the trained MobileNetV3-Small visual udder screening model into the
single-camera cattle video analysis pipeline.

PIPELINE FLOW:
  Single RGB Video Feed
            │
            ├── Lameness Locomotion Branch (ResNet-18 + BiGRU)
            │
            └── Visual Udder Screening Branch
                   │
                   ▼
              Quality-Filtered Candidate Keyframe Extraction
                   │
                   ▼
              MobileNetV3-Small Forward Inference (Eval Mode)
                   │
                   ▼
              Frame-Level Normal vs Abnormal Probabilities
                   │
                   ▼
              Spatiotemporal Aggregation (Median & Percentile Pooling)
                   │
                   ▼
              Cow-Level Visual Udder Screening Result

GOVERNANCE & VETERINARY INTEGRITY NOTICE:
- Objective: 'Normal Teat vs Visually Abnormal Teat' (Optical condition screening).
- NOT a clinical or subclinical mastitis diagnostic tool.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath("."))
import time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from src.mastitis_detector import BaseUdderDetector, AnatomicalHeuristicUdderDetector
from src.mastitis_frame_selector import UdderFrameSelector
from src.mastitis_features import VisualUdderFeatureExtractor

DEFAULT_UDDER_MODEL_PATH = "models/visual_udder_model.pth"

# Inference transform matching training eval
UDDER_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_visual_udder_model(
    model_path: str = DEFAULT_UDDER_MODEL_PATH,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, torch.device, Dict[str, Any]]:
    """
    Loads the trained MobileNetV3-Small visual udder screening checkpoint.
    
    Args:
        model_path: Path to visual_udder_model.pth checkpoint.
        device: Target device (CUDA if available, else CPU).
        
    Returns:
        (model, device, config)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Visual udder model checkpoint not found at: {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {
        "architecture": "MobileNetV3-Small",
        "num_classes": 2,
        "input_size": 224
    })
    
    # Instantiate architecture
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, config.get("num_classes", 2))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, device, config


class VisualUdderScreeningPipeline:
    """
    Integrated Visual Udder Screening Service.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        model_path: str = DEFAULT_UDDER_MODEL_PATH,
        detector: Optional[BaseUdderDetector] = None,
        frame_selector: Optional[UdderFrameSelector] = None,
        feature_extractor: Optional[VisualUdderFeatureExtractor] = None
    ):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model if weights exist
        if model is not None:
            self.model = model
            self.config = {"architecture": "MobileNetV3-Small", "num_classes": 2}
        elif os.path.exists(model_path):
            try:
                self.model, self.device, self.config = load_visual_udder_model(model_path=model_path, device=self.device)
            except Exception as e:
                print(f"[VisualUdderPipeline] Warning loading model: {e}")
                self.model = None
                self.config = {}
        else:
            self.model = None
            self.config = {}
            
        self.detector = detector or AnatomicalHeuristicUdderDetector()
        self.frame_selector = frame_selector or UdderFrameSelector(detector=self.detector)
        self.feature_extractor = feature_extractor or VisualUdderFeatureExtractor()
        self.transform = UDDER_INFERENCE_TRANSFORM

    def predict_crop_probability(self, crop: np.ndarray) -> Tuple[float, float]:
        """
        Runs model inference on a single cropped image/frame.
        
        Args:
            crop: BGR or RGB numpy array.
            
        Returns:
            (normal_probability, abnormal_probability)
        """
        if self.model is None:
            return 0.5, 0.5
            
        # Convert BGR crop to RGB PIL Image
        if len(crop.shape) == 3 and crop.shape[2] == 3:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = crop
            
        pil_img = Image.fromarray(crop_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
        p_normal = float(probs[0])
        p_abnormal = float(probs[1])
        return p_normal, p_abnormal

    def analyze_udder_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes a single video frame or still image.
        """
        start_time = time.time()
        
        if frame is None or frame.size == 0:
            return {
                "status": "ERROR",
                "error": "EMPTY_OR_INVALID_FRAME",
                "abnormal_probability": None,
                "normal_probability": None
            }
            
        # 1. Quality & Visibility Assessment
        is_quality_ok, rejection_reason, quality_metrics = self.frame_selector.evaluate_frame_quality(frame)
        if not is_quality_ok:
            return {
                "status": "INSUFFICIENT_VISUAL_DATA",
                "class": "UNKNOWN",
                "abnormal_probability": None,
                "normal_probability": None,
                "rejection_reason": rejection_reason,
                "quality_metrics": quality_metrics,
                "frames_analyzed": 0,
                "message": f"Frame rejected due to optical quality: {rejection_reason}",
                "model": "MobileNetV3-Small",
                "task": "Normal vs Visually Abnormal Teat",
                "disclaimer": "Visual screening benchmark; not a clinical mastitis diagnosis."
            }
            
        # 2. Udder Localization
        detection = self.detector.detect(frame)
        if not detection["detected"] or detection["crop"] is None:
            return {
                "status": "INSUFFICIENT_VISUAL_DATA",
                "class": "UNKNOWN",
                "abnormal_probability": None,
                "normal_probability": None,
                "frames_analyzed": 0,
                "detection": {
                    "detected": detection.get("detected", False),
                    "bbox": detection.get("bbox"),
                    "confidence": detection.get("confidence", 0.0)
                },
                "quality_metrics": quality_metrics,
                "message": "Udder region was not sufficiently localized in this frame.",
                "model": "MobileNetV3-Small",
                "task": "Normal vs Visually Abnormal Teat",
                "disclaimer": "Visual screening benchmark; not a clinical mastitis diagnosis."
            }
            
        crop = detection["crop"]
        p_norm, p_abn = self.predict_crop_probability(crop)
        indicators = self.feature_extractor.extract_all_indicators(crop)
        
        # Determine class label
        if p_abn < 0.35:
            class_label = "NORMAL"
            screening_result = "NORMAL VISUAL SCREENING"
            risk_level = "Low Visual Anomaly Risk"
        elif p_abn < 0.65:
            class_label = "VISUALLY_INCONCLUSIVE"
            screening_result = "WATCH / INCONCLUSIVE VISUAL SCREENING"
            risk_level = "Moderate Visual Anomaly Risk"
        else:
            class_label = "VISUALLY_ABNORMAL"
            screening_result = "VISUALLY ABNORMAL SCREENING"
            risk_level = "High Visual Anomaly Risk"
            
        elapsed_ms = round((time.time() - start_time) * 1000.0, 2)
        
        det_summary = {
            "detected": detection.get("detected", False),
            "bbox": detection.get("bbox"),
            "confidence": detection.get("confidence", 1.0),
            "detector_type": detection.get("detector_type", "AnatomicalHeuristicUdderDetector")
        }
        
        return {
            "status": "EXPERIMENTAL",
            "class": class_label,
            "screening_result": screening_result,
            "risk_level": risk_level,
            "abnormal_probability": round(p_abn, 4),
            "normal_probability": round(p_norm, 4),
            "confidence_percentage": round(max(p_norm, p_abn) * 100.0, 2),
            "frames_analyzed": 1,
            "processing_time_ms": elapsed_ms,
            "detection": det_summary,
            "optical_indicators": indicators,
            "model": "MobileNetV3-Small",
            "task": "Normal vs Visually Abnormal Teat",
            "disclaimer": "Visual screening benchmark; not a clinical mastitis diagnosis."
        }

    def screen_cow_udder_video(
        self,
        video_path: str,
        sample_stride: int = 4,
        max_candidate_frames: int = 8
    ) -> Dict[str, Any]:
        """
        Processes a full cow walking video feed:
          1. Selects unoccluded, high-clarity candidate udder frames.
          2. Runs MobileNetV3-Small visual abnormality inference per frame.
          3. Applies temporal median and percentile aggregation across keyframes.
          4. Returns structured cow-level visual udder screening output.
        """
        start_time = time.time()
        
        if not os.path.exists(video_path):
            return {
                "status": "ERROR",
                "error": f"Video file not found: {video_path}",
                "abnormal_probability": None,
                "normal_probability": None,
                "frames_analyzed": 0
            }
            
        # 1. Spatiotemporal Candidate Frame Selection
        selection_res = self.frame_selector.select_candidate_frames_from_video(
            video_path=video_path,
            sample_stride=sample_stride,
            max_usable_frames=max_candidate_frames
        )
        
        usable_frames = selection_res["usable_udder_frames"]
        rejected_frames = selection_res["rejected_frames"]
        summary = selection_res["summary"]
        
        # 2. Check if sufficient visual data was acquired
        if len(usable_frames) == 0:
            elapsed_ms = round((time.time() - start_time) * 1000.0, 1)
            return {
                "status": "INSUFFICIENT_VISUAL_DATA",
                "class": "UNKNOWN",
                "screening_result": "INSUFFICIENT_VISUAL_DATA",
                "abnormal_probability": None,
                "normal_probability": None,
                "frames_analyzed": 0,
                "rejected_frames_count": len(rejected_frames),
                "processing_time_ms": elapsed_ms,
                "message": "No unoccluded udder keyframes met the optical clarity and visibility thresholds.",
                "spatiotemporal_summary": summary,
                "model": "MobileNetV3-Small",
                "task": "Normal vs Visually Abnormal Teat",
                "disclaimer": "Visual screening benchmark; not a clinical mastitis diagnosis."
            }
            
        # 3. Frame-Level Inference & Optical Extraction
        frame_probabilities = []
        abnormal_probs = []
        normal_probs = []
        erythema_scores = []
        asymmetry_scores = []
        roughness_scores = []
        
        for cand in usable_frames:
            crop = cand["crop_frame"]
            p_norm, p_abn = self.predict_crop_probability(crop)
            
            abnormal_probs.append(p_abn)
            normal_probs.append(p_norm)
            
            # Optical indicator extraction
            ind = self.feature_extractor.extract_all_indicators(crop)
            if ind.get("valid"):
                erythema_scores.append(ind["erythema_indicators"]["erythema_proxy_index"])
                asymmetry_scores.append(ind["morphological_symmetry"]["asymmetry_index"])
                roughness_scores.append(ind["surface_texture"]["texture_roughness_index"])
                
            frame_probabilities.append({
                "frame_index": cand["frame_index"],
                "timestamp_sec": cand["timestamp_sec"],
                "sharpness_blur_var": cand["quality_metrics"]["blur_variance"],
                "normal_probability": round(p_norm, 4),
                "abnormal_probability": round(p_abn, 4),
                "predicted_class": "VISUALLY_ABNORMAL" if p_abn >= 0.50 else "NORMAL"
            })
            
        # 4. Spatiotemporal Aggregation Logic:
        # We calculate median abnormal probability as primary metric, cross-checked with 75th percentile
        # to ensure resilience against single-frame motion artifacts while capturing localized severe lesions.
        median_abnormal = float(np.median(abnormal_probs))
        p75_abnormal = float(np.percentile(abnormal_probs, 75))
        
        # Robust aggregated probability: 70% median + 30% 75th-percentile
        agg_abnormal = round(float(0.70 * median_abnormal + 0.30 * p75_abnormal), 4)
        agg_normal = round(float(1.0 - agg_abnormal), 4)
        
        # Tri-Band Screening Categorization:
        # NORMAL:   < 0.35
        # WATCH:    0.35 to 0.65
        # ABNORMAL: >= 0.65
        if agg_abnormal < 0.35:
            class_label = "NORMAL"
            screening_result = "NORMAL VISUAL SCREENING"
            risk_level = "Low Visual Anomaly Risk"
            explanation = "Extracted udder keyframes exhibit uniform coloration, smooth skin texture, and absence of prominent superficial lesions."
        elif agg_abnormal < 0.65:
            class_label = "VISUALLY_INCONCLUSIVE"
            screening_result = "WATCH / INCONCLUSIVE VISUAL SCREENING"
            risk_level = "Moderate Visual Anomaly Risk"
            explanation = "Extracted keyframes exhibit moderate skin roughness or mild pigmentation variation. Passive visual monitoring advised."
        else:
            class_label = "VISUALLY_ABNORMAL"
            screening_result = "VISUALLY ABNORMAL SCREENING"
            risk_level = "High Visual Anomaly Risk"
            explanation = "Extracted keyframes consistently exhibit visual indicators consistent with teat lesions, hyperkeratosis callosity, or skin chapping."
            
        elapsed_sec = round(time.time() - start_time, 4)
        elapsed_ms = round(elapsed_sec * 1000.0, 1)
        
        mean_erythema = round(float(np.mean(erythema_scores)), 4) if erythema_scores else 0.0
        mean_asymmetry = round(float(np.mean(asymmetry_scores)), 4) if asymmetry_scores else 0.0
        mean_roughness = round(float(np.mean(roughness_scores)), 4) if roughness_scores else 0.0
        
        return {
            "status": "EXPERIMENTAL",
            "class": class_label,
            "screening_result": screening_result,
            "risk_level": risk_level,
            "abnormal_probability": agg_abnormal,
            "normal_probability": agg_normal,
            "confidence_percentage": round(max(agg_normal, agg_abnormal) * 100.0, 2),
            "frames_analyzed": len(usable_frames),
            "rejected_frames_count": len(rejected_frames),
            "processing_time_ms": elapsed_ms,
            "device": str(self.device),
            "aggregation_method": "Temporal 70% median + 30% 75th-percentile pooling across quality-filtered candidate keyframes",
            "explanation": explanation,
            "optical_indicators": {
                "mean_erythema_index": mean_erythema,
                "mean_asymmetry_index": mean_asymmetry,
                "mean_texture_roughness": mean_roughness,
                "usable_frames_count": len(usable_frames)
            },
            "frame_probabilities": frame_probabilities,
            "spatiotemporal_summary": summary,
            "model": "MobileNetV3-Small",
            "task": "Normal vs Visually Abnormal Teat",
            "disclaimer": "Visual screening benchmark; not a clinical mastitis diagnosis."
        }


# Global Singleton Pipeline
_g_pipeline = None

def get_visual_udder_pipeline(model_path: str = DEFAULT_UDDER_MODEL_PATH) -> VisualUdderScreeningPipeline:
    global _g_pipeline
    if _g_pipeline is None:
        _g_pipeline = VisualUdderScreeningPipeline(model_path=model_path)
    return _g_pipeline

def analyze_udder_frame(frame: np.ndarray, model_path: str = DEFAULT_UDDER_MODEL_PATH) -> Dict[str, Any]:
    pipeline = get_visual_udder_pipeline(model_path=model_path)
    return pipeline.analyze_udder_frame(frame)

def screen_cow_udder_video(video_path: str, model_path: str = DEFAULT_UDDER_MODEL_PATH) -> Dict[str, Any]:
    pipeline = get_visual_udder_pipeline(model_path=model_path)
    return pipeline.screen_cow_udder_video(video_path)


if __name__ == "__main__":
    print("=" * 60)
    print("KsheerRaksha-AI — Visual Udder Screening Pipeline Self-Test")
    print("=" * 60)
    pipeline = get_visual_udder_pipeline()
    print(f"Loaded Model Device: {pipeline.device}")
    
    # Test on single dummy frame
    dummy_frame = np.full((480, 640, 3), 140, dtype=np.uint8)
    cv2.rectangle(dummy_frame, (180, 200), (460, 420), (60, 50, 200), -1)
    res = pipeline.analyze_udder_frame(dummy_frame)
    print("\nSingle Frame Result:")
    import json
    print(json.dumps(res, indent=2, default=str))
    print("=" * 60)
