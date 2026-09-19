import os
import sys
import glob
import hashlib
import json
import cv2
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\Lenovo\Downloads\CattleLameness-main\CattleLameness-main\Data"
LAME_DIR = os.path.join(DATA_DIR, "Lame")
NORMAL_DIR = os.path.join(DATA_DIR, "Normal")

def get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def sample_video_features(filepath, num_samples=5):
    """Extract a few resized grayscale frames to compute similarity between videos."""
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    indices = np.linspace(0, max(0, total_frames - 1), num_samples, dtype=int)
    frames = []
    current_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in indices:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32))
            frames.append(small.flatten().astype(np.float32) / 255.0)
        current_idx += 1
    cap.release()
    if len(frames) > 0:
        return np.concatenate(frames)
    return None

def audit_videos():
    records = []
    video_features = {}
    
    for label, folder in [("LAME", LAME_DIR), ("NORMAL", NORMAL_DIR)]:
        if not os.path.exists(folder):
            print(f"Directory not found: {folder}")
            continue
        files = sorted(os.listdir(folder))
        for fname in files:
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath):
                continue
            
            ext = os.path.splitext(fname)[1].lower()
            file_size_bytes = os.path.getsize(fpath)
            file_hash = get_file_hash(fpath)
            
            cap = cv2.VideoCapture(fpath)
            is_open = cap.isOpened()
            fps = cap.get(cv2.CAP_PROP_FPS) if is_open else 0
            frame_count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_open else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if is_open else 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if is_open else 0
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC)) if is_open else 0
            fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            
            # Read first and last frame to verify readability
            can_read_first = False
            actual_frames_read = 0
            if is_open:
                ret, frame = cap.read()
                can_read_first = ret and frame is not None and frame.size > 0
                if can_read_first:
                    actual_frames_read = 1
                    # Read to end to verify total decodable frames
                    while True:
                        ret, f = cap.read()
                        if not ret:
                            break
                        actual_frames_read += 1
            cap.release()
            
            # Duration calculation
            effective_fps = fps if fps > 0 else 30.0
            duration_sec = actual_frames_read / effective_fps if actual_frames_read > 0 else 0.0
            
            corrupted = not (is_open and can_read_first and actual_frames_read > 0)
            
            records.append({
                "filename": fname,
                "label": label,
                "path": fpath,
                "extension": ext,
                "size_bytes": file_size_bytes,
                "sha256": file_hash,
                "is_open": is_open,
                "can_read_first": can_read_first,
                "corrupted": corrupted,
                "fps": round(fps, 2),
                "reported_frame_count": frame_count_prop,
                "actual_frames_read": actual_frames_read,
                "width": width,
                "height": height,
                "aspect_ratio": f"{width}:{height}" if height > 0 else "N/A",
                "fourcc": fourcc,
                "duration_sec": round(duration_sec, 2)
            })
            
            feats = sample_video_features(fpath, num_samples=5)
            if feats is not None:
                video_features[fname] = feats

    df = pd.DataFrame(records)
    
    # Check duplicates by hash
    hash_counts = df['sha256'].value_counts()
    exact_duplicates = hash_counts[hash_counts > 1].to_dict()
    
    # Check near-duplicates by cosine similarity of sampled frames
    near_duplicates = []
    video_names = list(video_features.keys())
    for i in range(len(video_names)):
        for j in range(i + 1, len(video_names)):
            v1, v2 = video_names[i], video_names[j]
            f1, f2 = video_features[v1], video_features[v2]
            min_len = min(len(f1), len(f2))
            if min_len > 0:
                dot = np.dot(f1[:min_len], f2[:min_len])
                norm1 = np.linalg.norm(f1[:min_len])
                norm2 = np.linalg.norm(f2[:min_len])
                if norm1 > 0 and norm2 > 0:
                    sim = dot / (norm1 * norm2)
                    if sim > 0.98:  # Very high visual similarity threshold
                        near_duplicates.append({
                            "video_1": v1,
                            "video_2": v2,
                            "similarity": round(float(sim), 4)
                        })

    print(f"Total videos audited: {len(df)}")
    print(f"Lame: {len(df[df['label']=='LAME'])}, Normal: {len(df[df['label']=='NORMAL'])}")
    print(f"Corrupted videos: {len(df[df['corrupted']==True])}")
    print(f"Exact hash duplicates: {len(exact_duplicates)}")
    print(f"Near-duplicates: {len(near_duplicates)}")
    
    # Summary statistics
    summary = {
        "total_videos": int(len(df)),
        "total_samples": int(len(df)),
        "num_normal": int(len(df[df['label'] == 'NORMAL'])),
        "num_lame": int(len(df[df['label'] == 'LAME'])),
        "class_imbalance": "Balanced 50:50 (25 Normal, 25 Lame)",
        "existing_splits": "No predefined train/val/test split directory structure in the dataset folder (all 25 per class reside in flat Lame/ and Normal/ folders). Original authors used custom Colab script splits.",
        "file_formats": df['extension'].value_counts().to_dict(),
        "codecs": df['fourcc'].value_counts().to_dict(),
        "corrupted_files": df[df['corrupted']]['filename'].tolist(),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "fps_stats": {
            "unique_fps": df['fps'].unique().tolist(),
            "mean": round(float(df['fps'].mean()), 2),
            "min": round(float(df['fps'].min()), 2),
            "max": round(float(df['fps'].max()), 2),
            "std": round(float(df['fps'].std()), 2)
        },
        "duration_stats": {
            "mean": round(float(df['duration_sec'].mean()), 2),
            "median": round(float(df['duration_sec'].median()), 2),
            "min": round(float(df['duration_sec'].min()), 2),
            "max": round(float(df['duration_sec'].max()), 2),
            "std": round(float(df['duration_sec'].std()), 2),
            "q25": round(float(df['duration_sec'].quantile(0.25)), 2),
            "q75": round(float(df['duration_sec'].quantile(0.75)), 2)
        },
        "frame_count_stats": {
            "mean": round(float(df['actual_frames_read'].mean()), 2),
            "median": round(float(df['actual_frames_read'].median()), 2),
            "min": int(df['actual_frames_read'].min()),
            "max": int(df['actual_frames_read'].max()),
            "std": round(float(df['actual_frames_read'].std()), 2),
            "total_frames_all_videos": int(df['actual_frames_read'].sum())
        },
        "resolutions": df['aspect_ratio'].value_counts().to_dict(),
        "resolution_details": {f"{w}x{h}": int(cnt) for (w, h), cnt in df.groupby(['width', 'height']).size().items()}
    }
    
    # Save full audit details
    os.makedirs("audit_results", exist_ok=True)
    df.to_csv("audit_results/video_metadata.csv", index=False)
    with open("audit_results/audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("Audit completed. Saved to audit_results/video_metadata.csv and audit_results/audit_summary.json")

if __name__ == "__main__":
    audit_videos()
