"""
src/mastitis_frame_selector.py
==============================
KsheerRaksha-AI (Phase 2 — Candidate Udder Frame Selection)

Extracts and filters video frames to select high-quality candidate frames
where the cow's udder region is unobstructed, in focus, and adequately lit.

Does NOT assume all video frames are usable. Explicitly classifies frames into:
  - usable_udder_frames
  - rejected_frames (with specific rejection reasons)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.mastitis_detector import BaseUdderDetector, AnatomicalHeuristicUdderDetector

class UdderFrameSelector:
    """
    Spatiotemporal Frame Quality & Visibility Evaluator.
    
    Filters video frames based on:
      1. Motion / Blur score (Laplacian variance threshold)
      2. Lighting / Exposure balance (Luminance histogram limits)
      3. Udder ROI detectability & spatial extent
    """
    
    def __init__(
        self,
        min_blur_variance: float = 35.0,
        min_luminance: float = 30.0,
        max_luminance: float = 230.0,
        min_crop_area_ratio: float = 0.05,
        detector: Optional[BaseUdderDetector] = None
    ):
        self.min_blur_variance = min_blur_variance
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        self.min_crop_area_ratio = min_crop_area_ratio
        self.detector = detector or AnatomicalHeuristicUdderDetector()
        
    def evaluate_frame_quality(self, frame: np.ndarray) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """
        Assesses basic optical quality of a frame.
        
        Returns:
            (is_acceptable, rejection_reason, quality_metrics)
        """
        if frame is None or frame.size == 0:
            return False, "EMPTY_OR_CORRUPT_FRAME", {"blur_var": 0.0, "mean_lum": 0.0}
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # 1. Blur evaluation via Laplacian variance
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # 2. Illumination evaluation
        mean_lum = float(np.mean(gray))
        
        metrics = {
            "blur_variance": round(blur_var, 2),
            "mean_luminance": round(mean_lum, 2)
        }
        
        if blur_var < self.min_blur_variance:
            return False, f"MOTION_BLUR_EXCESSIVE (variance={blur_var:.1f} < {self.min_blur_variance})", metrics
            
        if mean_lum < self.min_luminance:
            return False, f"UNDEREXPOSED_TOO_DARK (lum={mean_lum:.1f} < {self.min_luminance})", metrics
            
        if mean_lum > self.max_luminance:
            return False, f"OVEREXPOSED_WASHOUT (lum={mean_lum:.1f} > {self.max_luminance})", metrics
            
        return True, None, metrics

    def select_candidate_frames_from_video(
        self,
        video_path: str,
        sample_stride: int = 4,
        max_usable_frames: int = 8
    ) -> Dict[str, Any]:
        """
        Processes an input cattle video and identifies usable candidate udder frames.
        
        Args:
            video_path: Path to cow walking video.
            sample_stride: Frame step interval during extraction.
            max_usable_frames: Maximum top quality frames to retain.
            
        Returns:
            Dict with 'usable_udder_frames', 'rejected_frames', and 'summary'.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video file at: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        usable_frames: List[Dict[str, Any]] = []
        rejected_frames: List[Dict[str, Any]] = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_stride == 0:
                timestamp_sec = round(frame_idx / fps, 3)
                
                # Step A: Evaluate frame quality (blur, light)
                is_quality_ok, quality_rejection_reason, metrics = self.evaluate_frame_quality(frame)
                
                if not is_quality_ok:
                    rejected_frames.append({
                        "frame_index": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "reason": quality_rejection_reason,
                        "metrics": metrics
                    })
                else:
                    # Step B: Evaluate Udder ROI detection & visibility
                    detection_res = self.detector.detect(frame)
                    
                    if not detection_res["detected"] or detection_res["crop"] is None:
                        rejected_frames.append({
                            "frame_index": frame_idx,
                            "timestamp_sec": timestamp_sec,
                            "reason": "UDDER_NOT_DETECTED_OR_OCCLUDED",
                            "metrics": metrics
                        })
                    else:
                        crop = detection_res["crop"]
                        h_f, w_f = frame.shape[:2]
                        h_c, w_c = crop.shape[:2]
                        area_ratio = (w_c * h_c) / (w_f * h_f)
                        
                        if area_ratio < self.min_crop_area_ratio:
                            rejected_frames.append({
                                "frame_index": frame_idx,
                                "timestamp_sec": timestamp_sec,
                                "reason": f"UDDER_ROI_TOO_SMALL (area_ratio={area_ratio:.3f} < {self.min_crop_area_ratio})",
                                "metrics": metrics
                            })
                        else:
                            usable_frames.append({
                                "frame_index": frame_idx,
                                "timestamp_sec": timestamp_sec,
                                "quality_metrics": metrics,
                                "bbox": detection_res["bbox"],
                                "detector_confidence": detection_res["confidence"],
                                "detector_type": detection_res["detector_type"],
                                "crop_shape": [h_c, w_c, 3],
                                "crop_frame": crop
                            })
                            
            frame_idx += 1
            
        cap.release()
        
        # Sort usable frames by blur variance (sharpness score) and select top-K
        usable_frames_sorted = sorted(
            usable_frames,
            key=lambda x: x["quality_metrics"]["blur_variance"],
            reverse=True
        )[:max_usable_frames]
        
        summary = {
            "total_video_frames": total_frames,
            "sampled_frames_count": (total_frames + sample_stride - 1) // sample_stride,
            "usable_frames_count": len(usable_frames_sorted),
            "rejected_frames_count": len(rejected_frames),
            "rejection_breakdown": {}
        }
        
        for r in rejected_frames:
            reason_category = r["reason"].split()[0]
            summary["rejection_breakdown"][reason_category] = summary["rejection_breakdown"].get(reason_category, 0) + 1
            
        return {
            "usable_udder_frames": usable_frames_sorted,
            "rejected_frames": rejected_frames,
            "summary": summary
        }
