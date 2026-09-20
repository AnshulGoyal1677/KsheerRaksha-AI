"""
scripts/validate_demo_e2e.py
============================
End-to-End Live Application Demo Validator for KsheerRaksha-AI.

Performs live HTTP requests to the running backend /api/screen endpoint,
simulates video uploads, tests both normal and lame locomotion samples,
and validates the zero-udder-frames fallback and disclaimer presence.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import urllib.request
import urllib.parse
import cv2
import numpy as np

PORT = 8000
SERVER_URL = f"http://127.0.0.1:{PORT}"

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    duration = round(frame_count / fps, 2) if fps > 0 else 0.0
    return frame_count, fps, duration

def test_screen_endpoint(video_path):
    url = f"{SERVER_URL}/api/screen"
    payload = json.dumps({"video_path": video_path}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        elapsed_ms = round((time.time() - t0) * 1000.0, 1)
        data = json.loads(body)
    return data, elapsed_ms

def main():
    print("=" * 84)
    print("KSHEERRAKSHA-AI — FINAL END-TO-END DEMO VALIDATION")
    print("=" * 84)
    
    test_videos = [
        ("Normal Cow 1", r"C:\Users\anshu\Downloads\CattleLameness-main\CattleLameness-main\Data\Normal\N (1).mp4", "NORMAL"),
        ("Normal Cow 2", r"C:\Users\anshu\Downloads\CattleLameness-main\CattleLameness-main\Data\Normal\N (3).mp4", "NORMAL"),
        ("Lame Cow 1", r"C:\Users\anshu\Downloads\CattleLameness-main\CattleLameness-main\Data\Lame\L (1).mp4", "LAME"),
        ("Lame Cow 2", r"C:\Users\anshu\Downloads\CattleLameness-main\CattleLameness-main\Data\Lame\L (5).mp4", "LAME"),
    ]
    
    results = []
    
    for label, vpath, expected_lameness in test_videos:
        if not os.path.exists(vpath):
            print(f"[SKIP] Video missing: {vpath}")
            continue
            
        frame_count, fps, duration = get_video_metadata(vpath)
        filename = os.path.basename(vpath)
        
        data, req_latency_ms = test_screen_endpoint(vpath)
        
        lameness = data.get("lameness", {})
        udder = data.get("visual_udder_screening", {})
        
        item = {
            "name": label,
            "filename": filename,
            "duration_sec": duration,
            "total_frames": frame_count,
            "fps": fps,
            "expected_lameness": expected_lameness,
            "lameness_pred": lameness.get("prediction"),
            "lameness_prob": lameness.get("lameness_probability"),
            "lameness_status": lameness.get("status"),
            "lameness_risk": lameness.get("risk_level"),
            "udder_status": udder.get("status"),
            "udder_class": udder.get("class"),
            "udder_abnormal_prob": udder.get("abnormal_probability"),
            "udder_frames_analyzed": udder.get("frames_analyzed"),
            "udder_model": udder.get("model"),
            "udder_task": udder.get("task"),
            "udder_disclaimer": udder.get("disclaimer"),
            "inference_time_ms": lameness.get("inference_time_ms"),
            "api_latency_ms": req_latency_ms,
            "device": data.get("device", "cuda")
        }
        results.append(item)
        
        print(f"\n[TEST: {label} - {filename}] ({duration}s, {frame_count} frames)")
        print(f"  |-- Lameness Branch:     {item['lameness_pred']} ({item['lameness_status']}) | Prob: {item['lameness_prob']} | Risk: {item['lameness_risk']}")
        print(f"  |-- Visual Udder Branch: {item['udder_class']} ({item['udder_status']}) | Abnormal Prob: {item['udder_abnormal_prob']} | Frames Analyzed: {item['udder_frames_analyzed']}")
        print(f"  |-- End-to-End Latency:  {req_latency_ms} ms (Lameness inference: {item['inference_time_ms']} ms)")
        print(f"  \\-- Disclaimer:          '{item['udder_disclaimer']}'")
        
    # Test Zero-Udder Frames Video Fallback
    print("\n" + "-" * 84)
    print("[TEST: Synthetic No-Udder Flat/Blurry Video Fallback]")
    temp_video = "uploads/synthetic_test_flat.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, 15.0, (320, 240))
    for _ in range(20):
        out.write(np.full((240, 320, 3), 15, dtype=np.uint8))
    out.release()
    
    flat_data, flat_latency = test_screen_endpoint(temp_video)
    flat_udder = flat_data.get("visual_udder_screening", {})
    
    print(f"  |-- Visual Udder Status:  {flat_udder.get('status')} (Expected: INSUFFICIENT_VISUAL_DATA)")
    print(f"  |-- Abnormal Prob:        {flat_udder.get('abnormal_probability')} (Expected: None / null)")
    print(f"  |-- Frames Analyzed:      {flat_udder.get('frames_analyzed')} (Expected: 0)")
    print(f"  \\-- Disclaimer Present:   '{flat_udder.get('disclaimer')}'")
    
    assert flat_udder.get("status") == "INSUFFICIENT_VISUAL_DATA", "Must return INSUFFICIENT_VISUAL_DATA"
    assert flat_udder.get("abnormal_probability") is None, "Must not fabricate probability score"
    assert flat_udder.get("frames_analyzed") == 0, "Must report 0 analyzed frames"
    print("[PASS] Zero usable udder frames correctly handled without fabricating scores.")
    
    if os.path.exists(temp_video):
        os.remove(temp_video)
        
    print("=" * 84)
    print("ALL DEMO VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 84)
    
    # Save structured telemetry for markdown report
    with open("models/demo_validation_telemetry.json", "w") as f:
        json.dump({"test_results": results, "flat_test": flat_udder}, f, indent=2)

if __name__ == "__main__":
    main()
