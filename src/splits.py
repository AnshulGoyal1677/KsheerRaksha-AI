import os
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\anshu\Downloads\CattleLameness-main\CattleLameness-main\Data"
SPLITS_FILE = "splits.json"

def create_stratified_splits(seed=42, test_size=0.16, val_size=0.16):
    """
    Creates strict video-level stratified splits to ensure zero frame leakage.
    50 videos total (25 Normal, 25 Lame).
    34 train (17 Normal, 17 Lame) -> 68%
    8 val (4 Normal, 4 Lame) -> 16%
    8 test (4 Normal, 4 Lame) -> 16%
    """
    metadata_csv = "audit_results/video_metadata.csv"
    if os.path.exists(metadata_csv):
        df = pd.read_csv(metadata_csv)
    else:
        # Fallback to direct directory scan
        rows = []
        for label, folder in [("LAME", os.path.join(DATA_DIR, "Lame")), ("NORMAL", os.path.join(DATA_DIR, "Normal"))]:
            for f in sorted(os.listdir(folder)):
                if f.endswith(".mp4"):
                    rows.append({"filename": f, "label": label, "path": os.path.join(folder, f)})
        df = pd.DataFrame(rows)

    train_val_df, test_df = train_test_split(
        df, test_size=8, random_state=seed, stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=8, random_state=seed, stratify=train_val_df['label']
    )

    splits = {
        "seed": seed,
        "train": train_df[["filename", "label", "path"]].to_dict(orient="records"),
        "val": val_df[["filename", "label", "path"]].to_dict(orient="records"),
        "test": test_df[["filename", "label", "path"]].to_dict(orient="records")
    }

    os.makedirs(os.path.dirname(SPLITS_FILE) if os.path.dirname(SPLITS_FILE) else ".", exist_ok=True)
    with open(SPLITS_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Stratified Splits Created (Seed: {seed}):")
    print(f"  Train: {len(train_df)} videos ({train_df['label'].value_counts().to_dict()})")
    print(f"  Val:   {len(val_df)} videos ({val_df['label'].value_counts().to_dict()})")
    print(f"  Test:  {len(test_df)} videos ({test_df['label'].value_counts().to_dict()})")
    return splits

if __name__ == "__main__":
    create_stratified_splits()
