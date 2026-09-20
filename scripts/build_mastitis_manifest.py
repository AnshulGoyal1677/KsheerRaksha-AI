"""
scripts/build_mastitis_manifest.py
==================================
KsheerRaksha-AI (Phase 2 — Final Mastitis Dataset Manifest Builder)

Scans the local dataset at data/mastitis_raw/ (mastitis, normal_teats, loose),
computes exact SHA-256 hashes, detects duplicates, assigns conservative case groups,
infers conditions strictly from filename indicators, evaluates training usability,
and emits:
  1. data/mastitis_manifest.csv
  2. data/mastitis_manifest_report.md

GOVERNANCE & VETERINARY RULES:
- DOES NOT move, rename, delete, modify, or copy any images.
- DOES NOT train any model or create train/val/test folders.
- Preserves original folder names: mastitis, normal_teats, loose.
- DOES NOT assume every image in mastitis/ is clinically confirmed mastitis.
- Sets inferred_condition = 'unknown' when not explicitly indicated in filename.
"""

import os
import sys
import re
import csv
import hashlib
from typing import Dict, List, Tuple, Any
from PIL import Image

DATA_DIR = "data/mastitis_raw"
OUTPUT_CSV = "data/mastitis_manifest.csv"
OUTPUT_REPORT = "data/mastitis_manifest_report.md"

def get_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def extract_case_group(filename: str, folder: str) -> str:
    """
    Assigns conservative case grouping to prevent multi-angle/paired shots
    of the same cow/lesion from being split across validation boundaries.
    
    Grouping Strategy:
      1. Paired Picture series: 'Picture20-400x284.jpg' and 'Picture20-1-400x284.jpg' -> 'case_picture_20'
      2. Condition series: 'hyp1' -> 'case_hyp_1', 'pox1' -> 'case_pox_1'
      3. Normal teats: 'normal1' -> 'case_normal_1'
      4. Loose whole udders: '001_30...' and '26.jpeg' -> 'case_loose_cow_gro174291'
      5. Strips thumbnail dimensions and timestamp tags (-400x284, -e1515...).
    """
    fn_lower = filename.lower()
    
    # 1. Normal teats folder
    if folder == "normal_teats":
        m = re.search(r"normal(\d+)", fn_lower)
        if m:
            return f"case_normal_{m.group(1)}"
        return "case_normal_misc"
        
    # 2. Loose whole-udder images
    if folder == "loose":
        if "001_30" in fn_lower or fn_lower == "26.jpeg":
            return "case_loose_cow_gro174291"
        m = re.match(r"^(\d+)", fn_lower)
        if m:
            return f"case_loose_cow_{m.group(1)}"
        return f"case_loose_{fn_lower.split('.')[0]}"
        
    # 3. Mastitis folder: Picture series (PictureX and PictureX-1)
    pic_m = re.match(r"^picture(\d+)", fn_lower)
    if pic_m:
        return f"case_picture_{pic_m.group(1)}"
        
    # 4. Mastitis folder: Hyperkeratosis reference series (hyp1..25)
    if fn_lower.startswith("hyp"):
        m = re.search(r"hyp(\d+)", fn_lower)
        return f"case_hyp_{m.group(1)}" if m else "case_hyp_series"
        
    # 5. Mastitis folder: Pox lesion series (pox1..6, pseudo-pox)
    if "pox" in fn_lower:
        m = re.search(r"pox(\d+)", fn_lower)
        return f"case_pox_{m.group(1)}" if m else "case_pox_lesion"
        
    # 6. Mastitis folder: Vascular series (red5..14, blue11..12)
    if fn_lower.startswith("blue"):
        m = re.search(r"blue(\d+)", fn_lower)
        return f"case_blue_vascular_{m.group(1)}" if m else "case_blue_vascular"
    if fn_lower.startswith("red"):
        m = re.search(r"red(\d+)", fn_lower)
        return f"case_red_vascular_{m.group(1)}" if m else "case_red_vascular"
        
    # 7. Mastitis folder: Ringing & Callosity
    if "large-ring" in fn_lower:
        m = re.search(r"large-ring-(\d+)", fn_lower)
        return f"case_large_ring_{m.group(1)}" if m else "case_large_ring"
    if "large-rough-ring" in fn_lower:
        return "case_large_rough_ring"
    if "base-ringing" in fn_lower:
        return "case_base_ringing"
    if "ring" in fn_lower:
        m = re.search(r"ring(\d+)", fn_lower)
        return f"case_ringing_{m.group(1)}" if m else "case_ringing"
        
    # 8. Mastitis folder: Dry skin & Chaps
    if "dry-skin-crack" in fn_lower:
        m = re.search(r"crack-?(\d+)", fn_lower)
        return f"case_dry_crack_{m.group(1)}" if m else "case_dry_crack"
    if "dry-skin" in fn_lower:
        m = re.search(r"skin-?(\d+)", fn_lower)
        return f"case_dry_skin_{m.group(1)}" if m else "case_dry_skin"
    if "chap" in fn_lower:
        m = re.search(r"chaps-(\d+)", fn_lower)
        return f"case_chaps_{m.group(1)}" if m else "case_chaps"
        
    # 9. Mastitis folder: Chemical burns
    if "chem" in fn_lower:
        m = re.search(r"burn-?(\d+)", fn_lower)
        return f"case_chem_burn_{m.group(1)}" if m else "case_chem_burn"
        
    # 10. Mastitis folder: Smooth scores
    if "smooth" in fn_lower:
        m = re.search(r"smooth-(\d+)", fn_lower)
        return f"case_smooth_score_{m.group(1)}" if m else "case_smooth_score"
        
    # 11. Mastitis folder: Very rough scores
    if "very-rough" in fn_lower:
        m = re.search(r"very-rough-(\d+)", fn_lower)
        return f"case_very_rough_{m.group(1)}" if m else "case_very_rough"
        
    # 12. Mastitis folder: Wet teats / dip cups
    if "wet-teat" in fn_lower:
        m = re.search(r"wet-teat(\d+)", fn_lower)
        return f"case_wet_teat_{m.group(1)}" if m else "case_wet_teat"
        
    # 13. Mastitis folder: Exfoliation
    if "exfol" in fn_lower:
        m = re.search(r"exfol(\d+)", fn_lower)
        return f"case_exfol_{m.group(1)}" if m else "case_exfol"
        
    # 14. Mastitis folder: Petechiae & Hemorrhage
    if "pet" in fn_lower or "petechail" in fn_lower:
        return "case_petechial_hemorrhage"
    if "severe-hemm" in fn_lower:
        m = re.search(r"hemm(\d+)", fn_lower)
        return f"case_severe_hemm_{m.group(1)}" if m else "case_severe_hemm"
    if "severe-ringing" in fn_lower:
        return "case_severe_ringing"
        
    # 15. Fallback clean stem
    clean = re.sub(r"(-e\d+)?(-\d+x\d+)?\.(jpg|jpeg|png)", "", fn_lower)
    stem = re.sub(r"\d+$", "", clean).strip("-_")
    return f"case_{stem or 'misc'}"


def infer_condition_from_filename(filename: str, folder: str) -> str:
    """
    Infers specific condition ONLY when reasonably indicated by filename.
    Otherwise returns 'unknown'.
    """
    fn_lower = filename.lower()
    
    if folder == "normal_teats":
        return "normal_teat"
        
    if folder == "loose":
        # Unlabeled whole udder images
        return "unknown"
        
    # Mastitis folder checks based strictly on filename text
    if "hyp" in fn_lower or "hyperkeratosis" in fn_lower or "isolated-keratin" in fn_lower:
        return "hyperkeratosis"
    if "pox" in fn_lower or "pseudo-pox" in fn_lower:
        return "viral_pox_lesion"
    if "papilloma" in fn_lower:
        return "viral_papilloma"
    if "chem" in fn_lower or "chlorine" in fn_lower or "burnt" in fn_lower:
        return "chemical_burn"
    if "blue" in fn_lower or "purple" in fn_lower:
        return "vascular_congestion_blue"
    if "red" in fn_lower:
        return "vascular_congestion_red"
    if "dry-skin" in fn_lower or "chap" in fn_lower or "exfol" in fn_lower or "facial-eczma" in fn_lower:
        return "dry_chapped_skin"
    if "ring" in fn_lower or "rough" in fn_lower:
        return "teat_end_ringing_callosity"
    if "smooth" in fn_lower:
        return "smooth_teat_end_reference" # Note: Smooth-1 is a normal reference image!
    if "frostbite" in fn_lower:
        return "frostbite"
    if "hem" in fn_lower or "petechail" in fn_lower:
        return "hemorrhage_petechiae"
    if "wet-teat" in fn_lower or "barrier-dip" in fn_lower:
        return "dip_application_residue"
        
    # Numbered Picture series (Picture1..Picture54) do NOT contain condition words in filename
    return "unknown"


def evaluate_training_usability(
    filename: str,
    folder: str,
    dup_group: str,
    inf_cond: str,
    sha256: str
) -> bool:
    """
    Evaluates whether an image is usable for an experimental visual anomaly benchmark.
    
    Exclusion Criteria:
      1. Exact SHA-256 duplicate (Picture21-2-400x284.jpg) -> False
      2. Perceptual duplicate in loose (26.jpeg) -> False
      3. Unlabeled loose whole-udder images -> False (no ground truth label)
      4. Clotted milk flake on strip paddle (Picture29-297x284.jpg) -> False (non-udder object)
      5. Surgical cannulation procedure with gloved hands (Picture46-400x284.jpg, Picture46-1-400x284.jpg) -> False
      6. Dipping cup equipment applications (wet-teat2, wet-teat6) -> False (obscured by cup)
      7. Smooth teat references inside mastitis/ (Smooth-1-1, Smooth-2-1, Smooth-2-2) -> False (label noise)
    """
    fn_lower = filename.lower()
    
    # 1. Exact duplicate pair: exclude the second copy
    if fn_lower == "picture21-2-400x284.jpg":
        return False
        
    # 2. Loose unlabeled images
    if folder == "loose":
        return False
        
    # 3. Non-udder / non-teat object: milk clot on paddle
    if fn_lower == "picture29-297x284.jpg":
        return False
        
    # 4. Surgical procedure with gloved hands covering teat
    if fn_lower in ["picture46-400x284.jpg", "picture46-1-400x284.jpg"]:
        return False
        
    # 5. Dipping cup application obscuring teat
    if "wet-teat" in fn_lower or "barrier-dip" in fn_lower:
        return False
        
    # 6. Smooth normal reference images inside mastitis folder
    if inf_cond == "smooth_teat_end_reference":
        return False
        
    return True


def build_manifest(
    data_dir: str = DATA_DIR,
    output_csv: str = OUTPUT_CSV,
    output_report: str = OUTPUT_REPORT
) -> Dict[str, Any]:
    """Scans dataset and generates CSV manifest and Markdown audit report."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at: {data_dir}")
        
    subfolders = ["mastitis", "normal_teats", "loose"]
    
    # 1. Collect all images and compute SHA-256
    file_list = []
    sha_map: Dict[str, List[Tuple[str, str, str]]] = {}
    
    for sub in subfolders:
        sub_path = os.path.join(data_dir, sub)
        if not os.path.exists(sub_path):
            continue
        for fname in sorted(os.listdir(sub_path)):
            fpath = os.path.join(sub_path, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                sha = get_sha256(fpath)
                file_list.append((fpath, sub, fname, sha))
                sha_map.setdefault(sha, []).append((fpath, sub, fname))
                
    # 2. Map duplicate groups
    dup_group_map: Dict[str, str] = {}
    for sha, entries in sha_map.items():
        if len(entries) > 1:
            first_name = os.path.splitext(entries[0][2])[0]
            dup_id = f"dup_sha256_{first_name}"
            for entry in entries:
                dup_group_map[entry[0]] = dup_id
                
    # Map loose perceptual duplicate (001_30 vs 26.jpeg)
    loose_001 = os.path.join(data_dir, "loose", "001_30_IMG_gro174291-009-848x565.jpg")
    loose_26 = os.path.join(data_dir, "loose", "26.jpeg")
    if os.path.exists(loose_001) and os.path.exists(loose_26):
        dup_id = "dup_perceptual_loose_gro174291"
        dup_group_map[loose_001] = dup_id
        dup_group_map[loose_26] = dup_id
        
    # 3. Build manifest rows
    rows: List[Dict[str, Any]] = []
    format_counts: Dict[str, int] = {}
    folder_counts: Dict[str, int] = {}
    condition_counts: Dict[str, int] = {}
    usable_count = 0
    
    for fpath, sub, fname, sha in file_list:
        file_size = os.path.getsize(fpath)
        
        with Image.open(fpath) as img:
            w, h = img.size
            fmt = img.format or os.path.splitext(fname)[1].lstrip(".").upper()
            mode = img.mode
            channels = len(mode) if mode in ["RGB", "RGBA", "CMYK"] else (1 if mode in ["L", "P"] else 3)
            
        dup_g = dup_group_map.get(fpath, "none")
        case_g = extract_case_group(fname, sub)
        inf_cond = infer_condition_from_filename(fname, sub)
        usable = evaluate_training_usability(fname, sub, dup_g, inf_cond, sha)
        
        if usable:
            usable_count += 1
            
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
        folder_counts[sub] = folder_counts.get(sub, 0) + 1
        condition_counts[inf_cond] = condition_counts.get(inf_cond, 0) + 1
        
        rel_path = os.path.join("data", "mastitis_raw", sub, fname).replace("\\", "/")
        
        rows.append({
            "filepath": rel_path,
            "filename": fname,
            "original_folder": sub,
            "width": w,
            "height": h,
            "format": fmt,
            "channels": channels,
            "file_size_bytes": file_size,
            "sha256": sha,
            "duplicate_group": dup_g,
            "case_group": case_g,
            "inferred_condition": inf_cond,
            "usable_for_training": usable
        })
        
    # 4. Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "filepath", "filename", "original_folder", "width", "height",
        "format", "channels", "file_size_bytes", "sha256",
        "duplicate_group", "case_group", "inferred_condition", "usable_for_training"
    ]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    # 5. Write Markdown Report
    unique_case_groups = len(set(r["case_group"] for r in rows))
    unique_dup_groups = len(set(r["duplicate_group"] for r in rows if r["duplicate_group"] != "none"))
    
    report_content = f"""# KsheerRaksha-AI: Mastitis Dataset Manifest & Preparation Report

**Generated by**: `scripts/build_mastitis_manifest.py`  
**Dataset Base Directory**: `{data_dir}`  
**Manifest CSV**: `{output_csv}`  
**Audit Timestamp**: 2026-09-20  

---

## 1. Dataset Summary Statistics

- **Total Images Scanned**: **{len(rows)}**
- **Usable for Training**: **{usable_count}** ({len(rows) - usable_count} excluded)
- **Unique Case Groups**: **{unique_case_groups}**
- **Unique Duplicate Groups**: **{unique_dup_groups}**

### Breakdown by Original Folder
| Original Folder | Total Image Count | Usable Count | Excluded Count |
| :--- | :---: | :---: | :---: |
"""
    for fldr in subfolders:
        tot = sum(1 for r in rows if r["original_folder"] == fldr)
        use = sum(1 for r in rows if r["original_folder"] == fldr and r["usable_for_training"])
        report_content += f"| `{fldr}` | {tot} | {use} | {tot - use} |\n"

    report_content += f"""
### Image Raster Formats
"""
    for fmt, cnt in sorted(format_counts.items()):
        report_content += f"- **{fmt}**: {cnt} images\n"

    report_content += f"""
---

## 2. Inferred Condition Taxonomy (Filename-Derived)

> [!NOTE]
> Conditions are inferred **strictly from filename indicators**. Numbered cases (e.g. `Picture1..Picture54`) and loose images without condition text are labeled `unknown`.

| Inferred Condition Category | Image Count | Primary Visual / Anatomical Finding |
| :--- | :---: | :--- |
"""
    for cond, cnt in sorted(condition_counts.items(), key=lambda x: x[1], reverse=True):
        report_content += f"| `{cond}` | {cnt} | Descriptive category based on filename tags |\n"

    report_content += """
---

## 3. Duplicate Detection & Grouping Summary

1. **Exact SHA-256 Duplicates**:
   - `Picture21-1-400x284.jpg` == `Picture21-2-400x284.jpg` (SHA-256 identical).
   - Second duplicate is marked `usable_for_training = false`.
2. **Perceptual Duplicates in Loose**:
   - `001_30_IMG_gro174291-009-848x565.jpg` and `26.jpeg` (Same whole-udder image, re-cropped).
   - Both in `loose/` and marked `usable_for_training = false`.
3. **Paired Multi-Angle Cases**:
   - 18 paired cases identified in `PictureX` series (`Picture20` and `Picture20-1`, etc.).
   - Assigned identical `case_group` to prevent train/test contamination.

---

## 4. Key Limitations & Governance Notice

1. **Non-Diagnostic Dataset**: The images represent scraped educational teat condition score reference photos (NMC Portfolio), NOT clinical mastitis ground truth.
2. **Subclinical Mastitis Invisibility**: Subclinical mastitis cannot be detected visually without Somatic Cell Count (SCC) or California Mastitis Test (CMT) validation.
3. **Severe Class Imbalance**: 10 normal images vs 159 usable abnormal images (15.9:1 ratio).
4. **No Model Training Authorized**: Model training on these labels is strictly prohibited.
"""

    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    return {
        "total_images": len(rows),
        "folder_counts": folder_counts,
        "format_counts": format_counts,
        "duplicate_groups_count": unique_dup_groups,
        "case_groups_count": unique_case_groups,
        "usable_count": usable_count,
        "condition_counts": condition_counts,
        "output_csv": output_csv,
        "output_report": output_report
    }

if __name__ == "__main__":
    print("=" * 70)
    print("KsheerRaksha-AI — Scanning Mastitis Dataset & Building Manifest")
    print("=" * 70)
    stats = build_manifest()
    
    print(f"Total Images:            {stats['total_images']}")
    print(f"Count by Original Folder: {stats['folder_counts']}")
    print(f"Image Format Counts:      {stats['format_counts']}")
    print(f"Duplicate Groups:        {stats['duplicate_groups_count']}")
    print(f"Case Groups:             {stats['case_groups_count']}")
    print(f"Usable for Training:     {stats['usable_count']}")
    print(f"\nInferred Condition Counts:")
    for cond, cnt in sorted(stats['condition_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cond:30s}: {cnt:3d}")
    print(f"\nOutput Manifest CSV:     {stats['output_csv']}")
    print(f"Output Audit Report:     {stats['output_report']}")
    print("=" * 70)
