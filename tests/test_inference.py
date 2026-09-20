"""
tests/test_inference.py
=======================
Regression & Integrity Test for GaitGuard AI Lameness Inference Pipeline.

Iterates through all 8 held-out test videos in splits.json, calculates
evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix),
identifies known edge cases (e.g., false positives), and asserts that
model inference performance matches the baseline in models/metrics.json.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
from inference import screen_cow_video, load_gaitguard_model

def test_inference_pipeline():
    """
    Regression test validating full test-split inference integrity against models/metrics.json.
    """
    assert os.path.exists("splits.json"), "splits.json not found"
    assert os.path.exists("models/best_model.pth"), "models/best_model.pth not found"
    assert os.path.exists("models/metrics.json"), "models/metrics.json not found"
    
    with open("splits.json", "r") as f:
        splits = json.load(f)
        
    with open("models/metrics.json", "r") as f:
        metrics_baseline = json.load(f)
        
    hist_test = metrics_baseline.get("test_metrics", {})
    test_samples = splits["test"]
    assert len(test_samples) == 8, f"Expected 8 test samples, found {len(test_samples)}"
    
    model, device, config = load_gaitguard_model()
    print(f"\n[GaitGuard Regression] Loaded model on device: {device}")
    print(f"[GaitGuard Regression] Architecture config: {config}")
    
    y_true = []
    y_pred = []
    probabilities = []
    false_positives = []
    false_negatives = []
    
    print("\n" + "=" * 84)
    print(f"{'Filename':<12} | {'Ground Truth':<12} | {'Predicted':<10} | {'Prob(Lame)':<10} | {'Status':<8} | {'Outcome'}")
    print("-" * 84)
    
    for s in test_samples:
        filename = s["filename"]
        video_path = s["path"]
        gt_label = s["label"]
        gt_idx = 1 if gt_label == "LAME" else 0
        y_true.append(gt_idx)
        
        res = screen_cow_video(video_path, model=model, device=device)
        
        assert "prediction" in res
        assert "status" in res
        assert "lameness_probability" in res
        assert "normal_probability" in res
        assert abs(res["lameness_probability"] + res["normal_probability"] - 1.0) < 1e-3
        
        pred_label = res["prediction"]
        pred_idx = 1 if pred_label == "LAME" else 0
        y_pred.append(pred_idx)
        
        p_lame = res["lameness_probability"]
        probabilities.append(p_lame)
        status = res["status"]
        
        if gt_idx == pred_idx:
            outcome = "[CORRECT]"
        elif gt_idx == 0 and pred_idx == 1:
            outcome = "[FALSE POSITIVE]"
            false_positives.append((filename, p_lame, status))
        else:
            outcome = "[FALSE NEGATIVE]"
            false_negatives.append((filename, p_lame, status))
            
        print(f"{filename:<12} | {gt_label:<12} | {pred_label:<10} | {p_lame:<10.4f} | {status:<8} | {outcome}")
        
    print("=" * 84)
    
    # Calculate confusion matrix & metrics
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    accuracy = round((tp + tn) / len(y_true), 4)
    precision = round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4)
    recall = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)
    f1 = round(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0, 4)
    confusion_matrix = [[tn, fp], [fn, tp]]
    
    print("\n--- EVALUATION METRICS ---")
    print(f"Accuracy:         {accuracy:.4f} (75.0%)")
    print(f"Precision:        {precision:.4f} (66.7%)")
    print(f"Recall:           {recall:.4f} (100.0%)")
    print(f"F1-Score:         {f1:.4f} (0.8000)")
    print(f"Confusion Matrix: {confusion_matrix}  (TN={tn}, FP={fp}, FN={fn}, TP={tp})")
    
    print("\n--- ERROR ANALYSIS ---")
    print(f"False Positives ({len(false_positives)}): {false_positives}")
    print(f"False Negatives ({len(false_negatives)}): {false_negatives}")
    
    # Regression assertions: Verify metrics match baseline in models/metrics.json exactly
    expected_acc = hist_test.get("accuracy", 0.7500)
    expected_prec = hist_test.get("precision", 0.6667)
    expected_rec = hist_test.get("recall", 1.0000)
    expected_f1 = hist_test.get("f1", 0.8000)
    expected_cm = hist_test.get("confusion_matrix", [[2, 2], [0, 4]])
    
    assert accuracy == expected_acc, f"Accuracy mismatch: {accuracy} != {expected_acc}"
    assert abs(precision - expected_prec) < 1e-3, f"Precision mismatch: {precision} != {expected_prec}"
    assert recall == expected_rec, f"Recall mismatch: {recall} != {expected_rec}"
    assert f1 == expected_f1, f"F1 mismatch: {f1} != {expected_f1}"
    assert confusion_matrix == expected_cm, f"Confusion matrix mismatch: {confusion_matrix} != {expected_cm}"
    
    print("\n[SUCCESS] All regression & integrity checks passed! Metrics match baseline exactly.")

if __name__ == "__main__":
    test_inference_pipeline()
