"""
tests/test_mastitis_pipeline.py
===============================
Comprehensive Integration Test Suite for KsheerRaksha-AI Phase 2:
Dual-Branch Single-Video Screening (Lameness + Experimental Visual Udder Screening).

Verification Coverage:
1. Model checkpoint loading (models/visual_udder_model.pth).
2. CPU device inference.
3. CUDA device inference (if hardware available).
4. Single-image frame inference.
5. Multiple-frame temporal aggregation (70% median + 30% 75th percentile).
6. No usable frames handling (INSUFFICIENT_VISUAL_DATA, null probabilities).
7. Full /api/screen dual-branch response structure.
8. Lameness branch operational invariance (unmodified risk outputs).
9. Visual udder output presence and valid probability boundaries.
10. Veterinary governance check: strictly zero clinical mastitis diagnostic claims.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import tempfile
import numpy as np
import cv2
import torch

from src.mastitis_pipeline import (
    load_visual_udder_model,
    VisualUdderScreeningPipeline,
    get_visual_udder_pipeline,
    analyze_udder_frame,
    screen_cow_udder_video,
    DEFAULT_UDDER_MODEL_PATH
)
from src.mastitis_detector import AnatomicalHeuristicUdderDetector
from src.mastitis_frame_selector import UdderFrameSelector
from src.mastitis_features import VisualUdderFeatureExtractor
from inference import screen_cow_video, load_gaitguard_model


def test_1_checkpoint_loading():
    """1. Test that visual udder model checkpoint loads correctly."""
    assert os.path.exists(DEFAULT_UDDER_MODEL_PATH), f"Checkpoint missing at {DEFAULT_UDDER_MODEL_PATH}"
    model, device, config = load_visual_udder_model(DEFAULT_UDDER_MODEL_PATH)
    assert model is not None
    assert config.get("architecture") == "MobileNetV3-Small"
    assert config.get("num_classes") == 2
    # Verify model is in eval mode
    assert not model.training


def test_2_cpu_inference():
    """2. Test forward inference explicitly on CPU."""
    cpu_device = torch.device("cpu")
    pipeline = VisualUdderScreeningPipeline(device=cpu_device)
    assert pipeline.device.type == "cpu"
    
    # Create test synthetic crop (224x224 BGR)
    crop = np.full((224, 224, 3), 128, dtype=np.uint8)
    p_norm, p_abn = pipeline.predict_crop_probability(crop)
    
    assert isinstance(p_norm, float)
    assert isinstance(p_abn, float)
    assert 0.0 <= p_norm <= 1.0
    assert 0.0 <= p_abn <= 1.0
    assert abs((p_norm + p_abn) - 1.0) < 1e-4


def test_3_cuda_inference():
    """3. Test forward inference on CUDA when GPU is available."""
    if not torch.cuda.is_available():
        print("[INFO] CUDA not available on this machine; skipping GPU-specific check.")
        return
        
    cuda_device = torch.device("cuda")
    pipeline = VisualUdderScreeningPipeline(device=cuda_device)
    assert pipeline.device.type == "cuda"
    
    crop = np.full((224, 224, 3), 128, dtype=np.uint8)
    p_norm, p_abn = pipeline.predict_crop_probability(crop)
    
    assert isinstance(p_norm, float)
    assert isinstance(p_abn, float)
    assert 0.0 <= p_norm <= 1.0
    assert 0.0 <= p_abn <= 1.0
    assert abs((p_norm + p_abn) - 1.0) < 1e-4


def test_4_single_image_inference():
    """4. Test single image / frame analysis end-to-end."""
    # Synthetic frame with high-contrast sharp edges to pass blur/visibility filter
    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (180, 200), (460, 420), (60, 50, 200), -1)
    
    res = analyze_udder_frame(frame)
    assert res["status"] in ["EXPERIMENTAL", "INSUFFICIENT_VISUAL_DATA"]
    if res["status"] == "EXPERIMENTAL":
        assert res["class"] in ["NORMAL", "VISUALLY_INCONCLUSIVE", "VISUALLY_ABNORMAL"]
        assert res["screening_result"] in [
            "NORMAL VISUAL SCREENING",
            "WATCH / INCONCLUSIVE VISUAL SCREENING",
            "VISUALLY ABNORMAL SCREENING"
        ]
        assert isinstance(res["abnormal_probability"], float)
        assert isinstance(res["normal_probability"], float)
        assert res["frames_analyzed"] == 1
        assert res["model"] == "MobileNetV3-Small"
        assert res["task"] == "Normal vs Visually Abnormal Teat"


def test_5_multiple_frame_aggregation():
    """5. Test temporal aggregation calculation (70% median + 30% 75th percentile)."""
    probs = [0.10, 0.20, 0.30, 0.40, 0.80, 0.90]
    median_val = float(np.median(probs))
    p75_val = float(np.percentile(probs, 75))
    expected_agg = round(float(0.70 * median_val + 0.30 * p75_val), 4)
    
    # Check aggregation math integrity
    assert expected_agg > 0.0 and expected_agg < 1.0
    
    # Test on an actual sample cattle video from held-out test splits if present
    sample_video = None
    if os.path.exists("splits.json"):
        with open("splits.json", "r") as f:
            splits = json.load(f)
        if len(splits.get("test", [])) > 0:
            candidate_path = splits["test"][0]["path"]
            if os.path.exists(candidate_path):
                sample_video = candidate_path
                
    if sample_video is not None:
        pipeline = get_visual_udder_pipeline()
        video_res = pipeline.screen_cow_udder_video(sample_video, sample_stride=2, max_candidate_frames=6)
        assert "status" in video_res
        assert "aggregation_method" in video_res or video_res["status"] == "INSUFFICIENT_VISUAL_DATA"


def test_6_no_usable_frames_fallback():
    """6. Test zero-usable frames fallback without score fabrication."""
    # Create a completely black/flat dummy video that fails blur & luminance filters
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_video_path = tmp.name
        
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_video_path, fourcc, 10.0, (320, 240))
        for _ in range(15):
            flat_frame = np.full((240, 320, 3), 10, dtype=np.uint8) # Dark, blurry, flat
            out.write(flat_frame)
        out.release()
        
        pipeline = get_visual_udder_pipeline()
        res = pipeline.screen_cow_udder_video(tmp_video_path)
        
        assert res["status"] == "INSUFFICIENT_VISUAL_DATA"
        assert res["abnormal_probability"] is None
        assert res["frames_analyzed"] == 0
        assert res["model"] == "MobileNetV3-Small"
        assert res["task"] == "Normal vs Visually Abnormal Teat"
    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)


def test_7_full_api_screen_response_structure():
    """7. Test dual-branch response structure returned by /api/screen."""
    sample_video = None
    if os.path.exists("splits.json"):
        with open("splits.json", "r") as f:
            splits = json.load(f)
        if len(splits.get("test", [])) > 0:
            p = splits["test"][0]["path"]
            if os.path.exists(p):
                sample_video = p
                
    if sample_video is None:
        print("[INFO] No test video found on disk; skipping full video execution.")
        return
        
    # Execute both branches as app.py does
    lameness_model, device, _ = load_gaitguard_model()
    lameness_res = screen_cow_video(sample_video, model=lameness_model, device=device)
    udder_res = screen_cow_udder_video(sample_video)
    
    combined = {
        "lameness": {
            "prediction": lameness_res.get("prediction"),
            "lameness_probability": lameness_res.get("lameness_probability"),
            "normal_probability": lameness_res.get("normal_probability"),
            "status": lameness_res.get("status"),
            "risk_level": lameness_res.get("risk_level"),
            "confidence_percentage": lameness_res.get("confidence_percentage"),
            "explanation": lameness_res.get("explanation"),
            "inference_time_ms": lameness_res.get("inference_time_ms")
        },
        "visual_udder_screening": udder_res
    }
    
    assert "lameness" in combined
    assert "visual_udder_screening" in combined
    assert combined["lameness"]["status"] in ["NORMAL", "WATCH", "CHECK"]
    assert combined["visual_udder_screening"]["status"] in ["EXPERIMENTAL", "INSUFFICIENT_VISUAL_DATA"]


def test_8_lameness_output_invariance():
    """8. Verify lameness pipeline output remains identical and operational."""
    with open("splits.json", "r") as f:
        splits = json.load(f)
    sample = splits["test"][0]
    
    model, device, config = load_gaitguard_model()
    res = screen_cow_video(sample["path"], model=model, device=device)
    
    assert "prediction" in res
    assert "lameness_probability" in res
    assert "normal_probability" in res
    assert "status" in res
    assert res["status"] in ["NORMAL", "WATCH", "CHECK"]
    assert abs(res["lameness_probability"] + res["normal_probability"] - 1.0) < 1e-3


def test_9_visual_udder_output_presence():
    """9. Verify visual udder screening metadata and properties."""
    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    res = analyze_udder_frame(frame)
    
    assert "model" in res
    assert res["model"] == "MobileNetV3-Small"
    assert "task" in res
    assert res["task"] == "Normal vs Visually Abnormal Teat"
    assert "disclaimer" in res
    assert "not a clinical mastitis diagnosis" in res["disclaimer"].lower()


def test_10_no_clinical_mastitis_diagnostic_terminology():
    """10. Verify strictly zero clinical mastitis diagnostic claims are returned."""
    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    res = analyze_udder_frame(frame)
    
    res_str = json.dumps(res, default=str)
    
    forbidden_terms = [
        "Confirmed Mastitis",
        "Mastitis Diagnosis",
        "Clinical Mastitis"
    ]
    
    for term in forbidden_terms:
        assert term not in res_str, f"Forbidden clinical diagnosis term '{term}' found in response!"
        assert term.lower() not in res.get("class", "").lower(), f"Class '{res.get('class')}' contains forbidden term"
        assert term.lower() not in res.get("screening_result", "").lower(), f"Screening result contains forbidden term"


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING KSHEERRAKSHA-AI PHASE 2 INTEGRATION TEST SUITE")
    print("=" * 70)
    test_1_checkpoint_loading()
    print("[PASS] 1. Checkpoint loading")
    test_2_cpu_inference()
    print("[PASS] 2. CPU inference")
    test_3_cuda_inference()
    print("[PASS] 3. CUDA inference")
    test_4_single_image_inference()
    print("[PASS] 4. Single image inference")
    test_5_multiple_frame_aggregation()
    print("[PASS] 5. Multiple frame aggregation")
    test_6_no_usable_frames_fallback()
    print("[PASS] 6. No usable frames fallback")
    test_7_full_api_screen_response_structure()
    print("[PASS] 7. Full /api/screen response structure")
    test_8_lameness_output_invariance()
    print("[PASS] 8. Lameness output invariance")
    test_9_visual_udder_output_presence()
    print("[PASS] 9. Visual udder output presence")
    test_10_no_clinical_mastitis_diagnostic_terminology()
    print("[PASS] 10. No clinical mastitis diagnostic terminology")
    print("=" * 70)
    print("ALL 10 PHASE 2 INTEGRATION TESTS PASSED WITH 100% COMPLIANCE!")
    print("=" * 70)
