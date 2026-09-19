import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
from inference import screen_cow_video, load_gaitguard_model

def test_inference_pipeline():
    with open("splits.json") as f:
        splits = json.load(f)
        
    model, device, config = load_gaitguard_model()
    print(f"Loaded model on device: {device}")
    
    # Test on one normal sample and one lame sample from test set
    test_samples = splits["test"]
    normal_sample = next(s for s in test_samples if s["label"] == "NORMAL")
    lame_sample = next(s for s in test_samples if s["label"] == "LAME")
    
    print(f"Testing Normal sample: {normal_sample['filename']}")
    res_normal = screen_cow_video(normal_sample["path"], model=model, device=device)
    print(f"  -> Prediction: {res_normal['prediction']}, Status: {res_normal['status']}, Time: {res_normal['inference_time_ms']}ms")
    assert "prediction" in res_normal
    assert "status" in res_normal
    assert "lameness_probability" in res_normal
    
    print(f"Testing Lame sample: {lame_sample['filename']}")
    res_lame = screen_cow_video(lame_sample["path"], model=model, device=device)
    print(f"  -> Prediction: {res_lame['prediction']}, Status: {res_lame['status']}, Time: {res_lame['inference_time_ms']}ms")
    assert "prediction" in res_lame
    assert "status" in res_lame
    assert "lameness_probability" in res_lame
    
    print("ALL INFERENCE PIPELINE TESTS PASSED!")

if __name__ == "__main__":
    test_inference_pipeline()
