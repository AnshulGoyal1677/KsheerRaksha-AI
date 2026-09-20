"""
scripts/diagnose_lameness.py
============================
Performs a rigorous, non-destructive audit of the lameness model
across the held-out test split, comparing results with models/metrics.json.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import numpy as np
import torch
import cv2

from inference import screen_cow_video, load_gaitguard_model
from src.dataset import GaitDataset

def main():
    print("=" * 80)
    print("GAITGUARD AI — FULL HELD-OUT TEST SET AUDIT & DIAGNOSTIC INSPECTION")
    print("=" * 80)
    
    # 1. Load splits
    with open("splits.json", "r") as f:
        splits = json.load(f)
        
    test_samples = splits["test"]
    print(f"Total Test Samples: {len(test_samples)}")
    
    # 2. Load model
    model, device, config = load_gaitguard_model()
    print(f"Loaded Checkpoint: models/best_model.pth on Device: {device}")
    print(f"Model Configuration: {config}")
    
    # 3. Load recorded historical metrics
    with open("models/metrics.json", "r") as f:
        historical_metrics = json.load(f)
        
    hist_test = historical_metrics.get("test_metrics", {})
    
    # 4. Run inference on EVERY test video
    results = []
    y_true = []
    y_pred = []
    probabilities = []
    
    print("\n" + "-" * 80)
    print(f"{'Filename':<14} | {'True Label':<10} | {'Pred Label':<10} | {'Prob(Lame)':<10} | {'Status':<8} | {'Time (ms)':<10} | {'Match'}")
    print("-" * 80)
    
    for s in test_samples:
        p = s["path"]
        t_label = s["label"]
        y_true.append(1 if t_label == "LAME" else 0)
        
        t0 = time.time()
        res = screen_cow_video(p, model=model, device=device)
        elapsed_ms = res["inference_time_ms"]
        
        p_lame = res["lameness_probability"]
        p_label = res["prediction"]
        pred_idx = 1 if p_label == "LAME" else 0
        y_pred.append(pred_idx)
        probabilities.append(p_lame)
        
        match_str = "[CORRECT]" if pred_idx == y_true[-1] else "[FALSE POSITIVE]" if pred_idx == 1 else "[FALSE NEGATIVE]"
        
        print(f"{s['filename']:<14} | {t_label:<10} | {p_label:<10} | {p_lame:<10.4f} | {res['status']:<8} | {elapsed_ms:<10.1f} | {match_str}")
        
        results.append({
            "filename": s["filename"],
            "true_label": t_label,
            "predicted_label": p_label,
            "lameness_probability": p_lame,
            "status": res["status"],
            "inference_time_ms": elapsed_ms,
            "explanation": res["explanation"]
        })
        
    print("-" * 80)
    
    # 5. Compute confusion matrix & metrics
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n--- TEST SET EVALUATION SUMMARY ---")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.1f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix (Rows: True [0=Normal, 1=Lame], Cols: Pred [0=Normal, 1=Lame]):")
    print(f"  [[TN={tn}, FP={fp}],")
    print(f"   [FN={fn}, TP={tp}]]")
    
    # 6. Compare with historical metrics.json
    print("\n--- COMPARISON WITH HISTORICAL models/metrics.json ---")
    print(f"Historical Accuracy:  {hist_test.get('accuracy'):.4f}  <-> Current Accuracy:  {accuracy:.4f} (Exact Match: {hist_test.get('accuracy') == accuracy})")
    print(f"Historical Precision: {hist_test.get('precision'):.4f} <-> Current Precision: {precision:.4f} (Exact Match: {hist_test.get('precision') == precision})")
    print(f"Historical Recall:    {hist_test.get('recall'):.4f}    <-> Current Recall:    {recall:.4f} (Exact Match: {hist_test.get('recall') == recall})")
    print(f"Historical F1:        {hist_test.get('f1'):.4f}        <-> Current F1:        {f1:.4f} (Exact Match: {hist_test.get('f1') == f1})")
    print(f"Historical Conf Mat:  {hist_test.get('confusion_matrix')} <-> Current Conf Mat: {[[tn, fp], [fn, tp]]}")
    
    # 7. Check specific sample N (9).mp4
    n9_sample = next(s for s in test_samples if s["filename"] == "N (9).mp4")
    print("\n--- SPECIFIC DEEP DIVE: N (9).mp4 ---")
    cap = cv2.VideoCapture(n9_sample["path"])
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"File Path:            {n9_sample['path']}")
    print(f"Dimensions:           {width}x{height}")
    print(f"Total Video Frames:   {frame_count}")
    print(f"Framerate (FPS):      {fps:.2f}")
    print(f"Duration:             {frame_count/fps:.2f}s")
    
    # Check probabilities in metrics.json for N (9).mp4
    hist_filenames = hist_test.get("filenames", [])
    if "N (9).mp4" in hist_filenames:
        idx = hist_filenames.index("N (9).mp4")
        hist_p = hist_test["probabilities"][idx]
        hist_pred = hist_test["predictions"][idx]
        print(f"Historical Prob in metrics.json: {hist_p:.4f} (Pred: {hist_pred})")
    
    res_n9 = screen_cow_video(n9_sample["path"], model=model, device=device)
    print(f"Current Prob via inference.py:   {res_n9['lameness_probability']:.4f} (Pred: {res_n9['prediction']}, Status: {res_n9['status']})")
    print("=" * 80)

if __name__ == "__main__":
    main()
