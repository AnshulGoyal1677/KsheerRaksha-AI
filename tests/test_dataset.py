import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
from src.dataset import GaitDataset

def test_pipeline():
    with open('splits.json') as f:
        splits = json.load(f)

    ds_train = GaitDataset(splits['train'], num_frames=16, is_training=True)
    frames, label, info = ds_train.get_sample(0)
    print(f"Train sample loaded: shape={frames.shape}, dtype={frames.dtype}, label={label}, video={info['filename']}")
    assert frames.shape == (16, 3, 224, 224), f"Unexpected shape {frames.shape}"
    assert frames.dtype == 'float32', f"Unexpected dtype {frames.dtype}"

    ds_val = GaitDataset(splits['val'], num_frames=16, is_training=False)
    frames_val, label_val, info_val = ds_val.get_sample(0)
    print(f"Val sample loaded: shape={frames_val.shape}, dtype={frames_val.dtype}, label={label_val}, video={info_val['filename']}")
    assert frames_val.shape == (16, 3, 224, 224), f"Unexpected val shape {frames_val.shape}"

    print("ALL DATASET PIPELINE CHECKS PASSED!")

if __name__ == "__main__":
    test_pipeline()
