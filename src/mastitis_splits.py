"""
src/mastitis_splits.py
======================
KsheerRaksha-AI (Phase 2 — Visual Udder Screening Scaffold)

Case-Aware Dataset Partitioning & Manifest Generation
-----------------------------------------------------
AUDIT CONCLUSION & NON-TRAINING MANDATE:
- Total Images: 187 (170 mastitis folder, 10 normal teats, 7 loose).
- Class Imbalance: 17:1 ratio (94.4% positive / 5.6% negative).
- Content Reality: Scraped teat skin condition score photos (NMC Portfolio / Teat Club Int.),
  documenting hyperkeratosis, chemical burns, chaps, and viral pox lesions.
- NO SOMATIC CELL COUNT (SCC) OR BACTERIAL CULTURE GROUND TRUTH EXISTS.
- NO OFFICIAL COW IDENTIFIERS (RFID/Ear Tags) ARE AVAILABLE.

CRITICAL INSTRUCTION:
This script performs case-aware stem grouping to demonstrate proper veterinary
evaluation partitioning without data leakage, but NO CLASSIFIER SHOULD BE TRAINED
FROM THESE LABELS.
"""

import os
import re
import json
import random
from typing import Dict, List, Any, Tuple

DEFAULT_RAW_DIR = "data/mastitis_raw"
DEFAULT_OUTPUT_MANIFEST = "data/mastitis_splits.json"

def extract_case_group(filename: str, folder_name: str) -> str:
    """
    Extracts a grouping identifier based on filename stems to prevent
    multi-angle / paired shots of the same lesion/cow from leaking across splits.
    
    Examples:
      - 'Picture20-1-400x284.jpg' and 'Picture20-400x284.jpg' -> 'group_picture_20'
      - 'hyp1-e1515...jpg' through 'hyp25...jpg' -> 'group_hyp_series'
      - 'normal14-e1515...jpg' -> 'group_normal_14'
    """
    fn_lower = filename.lower()
    
    # 1. Picture series (e.g. Picture20, Picture20-1)
    pic_match = re.match(r"^picture(\d+)", fn_lower)
    if pic_match:
        return f"group_case_picture_{pic_match.group(1)}"
        
    # 2. Hyperkeratosis score series (hyp1..25)
    if fn_lower.startswith("hyp"):
        return "group_hyp_reference_series"
        
    # 3. Pox lesions series
    if "pox" in fn_lower:
        return "group_pox_lesions"
        
    # 4. Vascular congestion (red / blue series)
    if fn_lower.startswith("red"):
        return "group_vascular_red_congestion"
    if fn_lower.startswith("blue"):
        return "group_vascular_blue_congestion"
        
    # 5. Ringing series
    if "ring" in fn_lower:
        return "group_teat_end_ringing"
        
    # 6. Dry skin and cracks
    if "dry-skin" in fn_lower:
        return "group_dry_skin_chaps"
    if "chap" in fn_lower:
        return "group_chapped_skin"
        
    # 7. Chemical burns
    if "chem" in fn_lower:
        return "group_chemical_damage"
        
    # 8. Normal teats (each normal image is an individual macro photo)
    if folder_name == "normal_teats":
        num_match = re.search(r"normal(\d+)", fn_lower)
        if num_match:
            return f"group_normal_teat_{num_match.group(1)}"
        return f"group_normal_{fn_lower[:10]}"
        
    # 9. Loose whole-udder images
    if folder_name == "loose":
        # Handle near-duplicate pair: 001_30_IMG... and 26.jpeg
        if "001_30" in fn_lower or fn_lower == "26.jpeg":
            return "group_loose_udder_case_a"
        return f"group_loose_{fn_lower[:10]}"
        
    # 10. Fallback clean stem
    clean = re.sub(r"(-e\d+)?(-\d+x\d+)?(-\d+)?\.(jpg|jpeg|png)", "", fn_lower)
    stem = re.sub(r"\d+$", "", clean).strip("-_")
    return f"group_stem_{stem or 'misc'}"


def generate_case_aware_splits(
    raw_dir: str = DEFAULT_RAW_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generates case-aware group partitions ensuring paired images are never
    separated across train, validation, and test subsets.
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw dataset directory not found at: {raw_dir}")
        
    random.seed(seed)
    
    # Inventory all files
    items: List[Dict[str, Any]] = []
    folders = ["mastitis", "normal_teats", "loose"]
    
    for fldr in folders:
        fldr_path = os.path.join(raw_dir, fldr)
        if not os.path.exists(fldr_path):
            continue
        for fname in sorted(os.listdir(fldr_path)):
            fpath = os.path.join(fldr_path, fname)
            if not os.path.isfile(fpath):
                continue
                
            group_id = extract_case_group(fname, fldr)
            items.append({
                "filename": fname,
                "folder": fldr,
                "relative_path": os.path.join(fldr, fname).replace("\\", "/"),
                "full_path": os.path.abspath(fpath),
                "case_group_id": group_id,
                "raw_folder_label": "MASTITIS_FOLDER" if fldr == "mastitis" else ("NORMAL_FOLDER" if fldr == "normal_teats" else "LOOSE_UNLABELED")
            })
            
    # Group items by case_group_id
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["case_group_id"], []).append(item)
        
    # Separate normal groups from mastitis/loose groups to ensure stratified presence
    normal_group_keys = [k for k in groups.keys() if k.startswith("group_normal")]
    other_group_keys = [k for k in groups.keys() if not k.startswith("group_normal")]
    
    random.shuffle(normal_group_keys)
    random.shuffle(other_group_keys)
    
    # Allocate normal groups (10 total images across ~10 individual groups)
    # Ensure train has ~6-7, val has ~1-2, test has ~1-2
    n_norm = len(normal_group_keys)
    norm_val_count = max(1, int(round(n_norm * val_ratio)))
    norm_test_count = max(1, int(round(n_norm * test_ratio)))
    norm_train_count = n_norm - norm_val_count - norm_test_count
    
    norm_train = normal_group_keys[:norm_train_count]
    norm_val = normal_group_keys[norm_train_count:norm_train_count + norm_val_count]
    norm_test = normal_group_keys[norm_train_count + norm_val_count:]
    
    # Allocate other groups
    n_other = len(other_group_keys)
    other_val_count = max(1, int(round(n_other * val_ratio)))
    other_test_count = max(1, int(round(n_other * test_ratio)))
    other_train_count = n_other - other_val_count - other_test_count
    
    other_train = other_group_keys[:other_train_count]
    other_val = other_group_keys[other_train_count:other_train_count + other_val_count]
    other_test = other_group_keys[other_train_count + other_val_count:]
    
    split_group_keys = {
        "train": norm_train + other_train,
        "validation": norm_val + other_val,
        "test": norm_test + other_test
    }
    
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for split_name, gkeys in split_group_keys.items():
        for gk in gkeys:
            splits[split_name].extend(groups[gk])
            
    manifest = {
        "metadata": {
            "dataset_name": "NMC Teat Condition Portfolio (Web Scraped Subset)",
            "total_images": len(items),
            "total_case_groups": len(groups),
            "audit_date": "2026-09-20",
            "audit_mandate": "NO_TRAINING_AUTHORIZED",
            "clinical_ground_truth_status": "ABSENT (No Somatic Cell Count or Bacteriological Validation)",
            "cow_id_status": "Heuristic Filename Stem Clustering (Official Ear Tags/RFIDs Unavailable)",
            "target_ratio": f"{int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}",
            "split_counts": {
                "train": len(splits["train"]),
                "validation": len(splits["validation"]),
                "test": len(splits["test"])
            },
            "warnings": [
                "17:1 severe class imbalance prevents reliable statistical validation.",
                "Visual features show teat skin conditions / milking machine trauma, not clinical mastitis diagnosis.",
                "Confounding artifacts (gloves, green disinfectant, camera zoom) risk extreme shortcut learning.",
                "Subclinical mastitis cannot be detected from these RGB images."
            ]
        },
        "splits": splits
    }
    
    return manifest


def save_splits(manifest: Dict[str, Any], output_path: str = DEFAULT_OUTPUT_MANIFEST) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Case-aware split manifest saved to: {output_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("KsheerRaksha-AI — Case-Aware Mastitis Dataset Splitting Scaffold")
    print("=" * 70)
    manifest = generate_case_aware_splits()
    save_splits(manifest)
    
    meta = manifest["metadata"]
    print(f"\nTotal Images:       {meta['total_images']}")
    print(f"Total Case Groups:  {meta['total_case_groups']}")
    print(f"Train Split:        {meta['split_counts']['train']} images")
    print(f"Validation Split:   {meta['split_counts']['validation']} images")
    print(f"Test Split:         {meta['split_counts']['test']} images")
    print("\nLimitations & Warnings:")
    for w in meta["warnings"]:
        print(f"  * {w}")
    print("=" * 70)
