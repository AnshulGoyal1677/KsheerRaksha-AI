import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import GaitDataset
from src.model import GaitGuardModel

DEFAULT_MODEL_PATH = "models/best_model.pth"

def load_gaitguard_model(model_path=DEFAULT_MODEL_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}. Please run training first.")
        
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {"num_classes": 2, "hidden_dim": 64, "num_frames": 16})
    
    model = GaitGuardModel(
        num_classes=config.get("num_classes", 2),
        hidden_dim=config.get("hidden_dim", 64),
        pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device, config

import time

def screen_cow_video(video_path, model=None, device=None, model_path=DEFAULT_MODEL_PATH):
    """
    Performs automated cattle lameness screening on an input video.
    
    Input:
      video_path: path to cow walking video (.mp4, .avi, etc.)
    
    Output:
      JSON-compatible dict containing AI screening results.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
        
    start_time = time.time()
    if model is None:
        model, device, config = load_gaitguard_model(model_path=model_path, device=device)
    else:
        config = {"num_frames": 16}
        if device is None:
            device = next(model.parameters()).device
            
    num_frames = config.get("num_frames", 16)
    
    # Process video through identical evaluation pipeline
    sample_entry = [{"filename": os.path.basename(video_path), "label": "NORMAL", "path": video_path}]
    dataset = GaitDataset(sample_entry, num_frames=num_frames, is_training=False)
    
    try:
        frames_tensor, _, _ = dataset.get_sample(0)
    except Exception as e:
        return {
            "error": f"Failed to extract video frames: {str(e)}",
            "video_path": video_path
        }
        
    # Shape: (1, 16, 3, 224, 224)
    input_tensor = torch.from_numpy(frames_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    inference_time_sec = round(time.time() - start_time, 4)
    inference_time_ms = round(inference_time_sec * 1000.0, 1)
    
    p_normal = float(probs[0])
    p_lame = float(probs[1])
    
    predicted_class = "LAME" if p_lame >= 0.50 else "NORMAL"
    
    # Tri-band screening categorization:
    # NORMAL: < 0.35
    # WATCH:  0.35 to 0.65
    # CHECK:  >= 0.65
    if p_lame < 0.35:
        status = "NORMAL"
        risk_level = "Low Risk"
        explanation = "Locomotion shows regular cadence, symmetric limb placement, and absence of characteristic arched-back or head-bobbing anomalies."
    elif p_lame < 0.65:
        status = "WATCH"
        risk_level = "Moderate Risk (Watch List)"
        explanation = "Movement pattern exhibits subtle irregularities or mild stride asymmetry. Flagged for passive monitoring over consecutive parlor passages."
    else:
        status = "CHECK"
        risk_level = "High Risk (Action Advised)"
        explanation = "Movement pattern is more consistent with the lame class (evident asymmetric weight-bearing, shortened stride, or altered gait posture)."

    result = {
        "prediction": predicted_class,
        "lameness_probability": round(p_lame, 4),
        "normal_probability": round(p_normal, 4),
        "status": status,
        "risk_level": risk_level,
        "confidence_percentage": round(max(p_lame, p_normal) * 100.0, 2),
        "inference_time_seconds": inference_time_sec,
        "inference_time_ms": inference_time_ms,
        "device": str(device),
        "explanation": explanation,
        "video_file": os.path.basename(video_path),
        "screening_nature": "AI screening result — not veterinary diagnosis",
        "screening_guidance": "AI screening result — veterinary inspection recommended.",
        "disclaimer": "This automated screening output provides early decision support for herd managers. Confirmatory clinical assessment by a licensed veterinarian or hoof-care specialist is recommended before clinical intervention."
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="GaitGuard AI Cattle Lameness Screening Tool")
    parser.add_argument("--video", type=str, required=True, help="Path to input cow video")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to model weights checkpoint")
    parser.add_argument("--json-only", action="store_true", help="Output only raw JSON")
    args = parser.parse_args()
    
    res = screen_cow_video(args.video, model_path=args.model)
    
    if args.json_only:
        print(json.dumps(res, indent=2))
    else:
        print("\n" + "="*60)
        print("GAITGUARD AI — CATTLE LOCOMOTION SCREENING RESULT")
        print("="*60)
        print(f"Video File:           {res.get('video_file')}")
        print(f"Prediction:           {res.get('prediction')}")
        print(f"Lameness Probability: {res.get('lameness_probability')} ({res.get('lameness_probability')*100:.1f}%)")
        print(f"Normal Probability:   {res.get('normal_probability')} ({res.get('normal_probability')*100:.1f}%)")
        print(f"Screening Status:     {res.get('status')}")
        print(f"Risk Level:           {res.get('risk_level')}")
        print(f"Confidence:           {res.get('confidence_percentage')}%")
        print(f"Inference Time:       {res.get('inference_time_seconds')}s ({res.get('inference_time_ms')} ms)")
        print(f"Compute Device:       {res.get('device')}")
        print("-"*60)
        print("Clinical Rationale:")
        print(f"  {res.get('explanation')}")
        print("-"*60)
        print("Display:")
        print(f"  {res.get('screening_guidance')}")
        print("="*60 + "\n")
        print("JSON Summary:")
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
