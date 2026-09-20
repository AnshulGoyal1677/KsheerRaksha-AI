"""
scripts/train_visual_udder_model.py
===================================
KsheerRaksha-AI — Train Experimental Visual Udder Screening Model

Benchmark Objective:
  Binary classification between:
    - Class 0: 'NORMAL TEAT' (Clean, smooth teat skin)
    - Class 1: 'VISUALLY ABNORMAL TEAT' (Teat lesions, hyperkeratosis, chaps, burns, trauma)

GOVERNANCE & VETERINARY COMPLIANCE:
  - This is strictly an experimental visual-condition benchmark on teat tissue.
  - DO NOT claim this is a clinical or subclinical mastitis diagnostic tool.
  - Case-aware split strictly isolates case_group clusters across partitions.
  - Existing lameness model (models/best_model.pth) is NOT modified or touched.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import random
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuration
MANIFEST_PATH = "data/mastitis_manifest.csv"
MODEL_SAVE_PATH = "models/visual_udder_model.pth"
METRICS_SAVE_PATH = "models/visual_udder_metrics.json"
HISTORY_SAVE_PATH = "models/visual_udder_training_history.json"
CONFUSION_MATRIX_PATH = "models/visual_udder_confusion_matrix.png"

SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 12

CLASS_NAMES = ["NORMAL TEAT", "VISUALLY ABNORMAL TEAT"]
CLASS_LABELS = {0: "NORMAL TEAT", 1: "VISUALLY ABNORMAL TEAT"}

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class VisualUdderDataset(Dataset):
    """PyTorch Dataset for Visual Udder Screening Benchmark."""
    def __init__(self, records: List[Dict[str, Any]], transform=None):
        self.records = records
        self.transform = transform
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        item = self.records[idx]
        filepath = item["filepath"]
        
        # Open image and convert to RGB
        try:
            with Image.open(filepath) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image at {filepath}: {e}")
            
        if self.transform is not None:
            tensor = self.transform(img_rgb)
        else:
            tensor = transforms.ToTensor()(img_rgb)
            
        label = item["target_label"]
        return tensor, label, item["filename"], item["case_group"]


def prepare_case_aware_partitions(
    manifest_path: str = MANIFEST_PATH,
    seed: int = SEED
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Partitions the dataset ensuring no case_group overlap between Train, Val, and Test.
    """
    df = pd.read_csv(manifest_path)
    
    # Filter only usable records
    # Handles boolean or string 'True'/'true'
    usable_mask = df["usable_for_training"].astype(str).str.lower().isin(["true", "1"])
    usable_df = df[usable_mask].copy()
    
    # Assign target binary labels
    # 0 = NORMAL TEAT, 1 = VISUALLY ABNORMAL TEAT
    usable_df["target_label"] = usable_df["original_folder"].apply(
        lambda f: 0 if f == "normal_teats" else 1
    )
    
    records = usable_df.to_dict("records")
    
    # Group by case_group
    case_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        case_groups.setdefault(r["case_group"], []).append(r)
        
    normal_cases = [cg for cg in case_groups if any(r["target_label"] == 0 for r in case_groups[cg])]
    abnormal_cases = [cg for cg in case_groups if cg not in normal_cases]
    
    random.seed(seed)
    random.shuffle(normal_cases)
    random.shuffle(abnormal_cases)
    
    # Normal splits (10 cases -> 6 Train, 2 Val, 2 Test)
    n_norm = len(normal_cases)
    norm_val_count = max(1, int(round(n_norm * 0.15)))
    norm_test_count = max(1, int(round(n_norm * 0.15)))
    norm_train_count = n_norm - norm_val_count - norm_test_count
    
    norm_train = normal_cases[:norm_train_count]
    norm_val = normal_cases[norm_train_count:norm_train_count + norm_val_count]
    norm_test = normal_cases[norm_train_count + norm_val_count:]
    
    # Abnormal splits (133 cases -> ~93 Train, 20 Val, 20 Test)
    n_abn = len(abnormal_cases)
    abn_val_count = max(1, int(round(n_abn * 0.15)))
    abn_test_count = max(1, int(round(n_abn * 0.15)))
    abn_train_count = n_abn - abn_val_count - abn_test_count
    
    abn_train = abnormal_cases[:abn_train_count]
    abn_val = abnormal_cases[abn_train_count:abn_train_count + abn_val_count]
    abn_test = abnormal_cases[abn_train_count + abn_val_count:]
    
    train_records = [r for cg in norm_train + abn_train for r in case_groups[cg]]
    val_records = [r for cg in norm_val + abn_val for r in case_groups[cg]]
    test_records = [r for cg in norm_test + abn_test for r in case_groups[cg]]
    
    # Verification of zero overlap
    train_cgs = set(norm_train + abn_train)
    val_cgs = set(norm_val + abn_val)
    test_cgs = set(norm_test + abn_test)
    
    assert train_cgs.isdisjoint(val_cgs), "Train and Val case groups overlap!"
    assert train_cgs.isdisjoint(test_cgs), "Train and Test case groups overlap!"
    assert val_cgs.isdisjoint(test_cgs), "Val and Test case groups overlap!"
    
    stats = {
        "total_usable_images": len(records),
        "total_case_groups": len(case_groups),
        "normal_case_groups": len(normal_cases),
        "abnormal_case_groups": len(abnormal_cases),
        "train": {
            "total_images": len(train_records),
            "normal_images": sum(1 for r in train_records if r["target_label"] == 0),
            "abnormal_images": sum(1 for r in train_records if r["target_label"] == 1),
            "case_groups": len(train_cgs)
        },
        "validation": {
            "total_images": len(val_records),
            "normal_images": sum(1 for r in val_records if r["target_label"] == 0),
            "abnormal_images": sum(1 for r in val_records if r["target_label"] == 1),
            "case_groups": len(val_cgs)
        },
        "test": {
            "total_images": len(test_records),
            "normal_images": sum(1 for r in test_records if r["target_label"] == 0),
            "abnormal_images": sum(1 for r in test_records if r["target_label"] == 1),
            "case_groups": len(test_cgs)
        }
    }
    
    return train_records, val_records, test_records, stats


def build_model(num_classes: int = 2) -> nn.Module:
    """Instantiates MobileNetV3-Small with pretrained ImageNet weights."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Replace final linear layer
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device, criterion=None) -> Dict[str, Any]:
    """Evaluates model across full metric suite."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets, filenames, cgs in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist()) # Probability of Abnormal
            
    avg_loss = total_loss / max(1, len(dataloader.dataset))
    
    # Calculate non-skewed balanced metrics
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    precision_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    
    # Per-class metrics
    p_per, r_per, f_per, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    
    # PR-AUC for Abnormal class
    try:
        pr_auc = average_precision_score(all_targets, all_probs)
    except Exception:
        pr_auc = 0.0
        
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1]).tolist()
    
    return {
        "loss": round(avg_loss, 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "pr_auc": round(float(pr_auc), 4),
        "per_class": {
            "normal_teat": {
                "precision": round(float(p_per[0]), 4) if len(p_per) > 0 else 0.0,
                "recall": round(float(r_per[0]), 4) if len(r_per) > 0 else 0.0,
                "f1": round(float(f_per[0]), 4) if len(f_per) > 0 else 0.0
            },
            "abnormal_teat": {
                "precision": round(float(p_per[1]), 4) if len(p_per) > 1 else 0.0,
                "recall": round(float(r_per[1]), 4) if len(r_per) > 1 else 0.0,
                "f1": round(float(f_per[1]), 4) if len(f_per) > 1 else 0.0
            }
        },
        "confusion_matrix": cm,
        "targets": all_targets,
        "predictions": all_preds,
        "probabilities_abnormal": all_probs
    }


def train_pipeline():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Utilizing compute device: {device} (CUDA={torch.cuda.is_available()})")
    
    # 1. Prepare Case-Aware Partitions
    print("\n" + "=" * 70)
    print("STEP 1: CASE-AWARE DATASET PARTITIONING & VERIFICATION")
    print("=" * 70)
    train_records, val_records, test_records, stats = prepare_case_aware_partitions()
    
    print(f"Total Usable Images:   {stats['total_usable_images']}")
    print(f"Total Case Groups:     {stats['total_case_groups']} ({stats['normal_case_groups']} normal, {stats['abnormal_case_groups']} abnormal)")
    print("-" * 70)
    print(f"TRAIN Split:           {stats['train']['total_images']} images (Normal: {stats['train']['normal_images']}, Abnormal: {stats['train']['abnormal_images']}, CaseGroups: {stats['train']['case_groups']})")
    print(f"VALIDATION Split:      {stats['validation']['total_images']} images (Normal: {stats['validation']['normal_images']}, Abnormal: {stats['validation']['abnormal_images']}, CaseGroups: {stats['validation']['case_groups']})")
    print(f"TEST Split:            {stats['test']['total_images']} images (Normal: {stats['test']['normal_images']}, Abnormal: {stats['test']['abnormal_images']}, CaseGroups: {stats['test']['case_groups']})")
    print("-" * 70)
    print("[Verification] ZERO case_group overlap confirmed between Train, Validation, and Test splits.")
    
    # 2. Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Dataloaders with Class-Balanced Sampling
    train_dataset = VisualUdderDataset(train_records, transform=train_transform)
    val_dataset = VisualUdderDataset(val_records, transform=eval_transform)
    test_dataset = VisualUdderDataset(test_records, transform=eval_transform)
    
    # Class counts in train
    n_train_norm = stats["train"]["normal_images"]
    n_train_abn = stats["train"]["abnormal_images"]
    
    # Weight per class for loss function
    weight_norm = len(train_records) / (2.0 * max(1, n_train_norm))
    weight_abn = len(train_records) / (2.0 * max(1, n_train_abn))
    class_weights_tensor = torch.tensor([weight_norm, weight_abn], dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Sample weights for WeightedRandomSampler to ensure balanced minibatches
    sample_weights = [weight_norm if r["target_label"] == 0 else weight_abn for r in train_records]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_records), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. Model, Optimizer, AMP
    model = build_model(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING EXPERIMENTAL VISUAL UDDER MODEL (MobileNetV3-Small)")
    print("=" * 70)
    
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    os.makedirs("models", exist_ok=True)
    
    start_train_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for inputs, targets, _, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item() * inputs.size(0)
            
        scheduler.step()
        avg_train_loss = train_loss / max(1, len(train_dataset))
        
        # Validation
        val_eval = evaluate_model(model, val_loader, device, criterion)
        
        history_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_eval["loss"],
            "val_balanced_accuracy": val_eval["balanced_accuracy"],
            "val_macro_f1": val_eval["macro_f1"],
            "val_pr_auc": val_eval["pr_auc"],
            "lr": round(optimizer.param_groups[0]["lr"], 6)
        }
        history.append(history_entry)
        
        print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_eval['loss']:.4f} | Val Bal-Acc: {val_eval['balanced_accuracy']:.3f} | Val Macro-F1: {val_eval['macro_f1']:.3f} | Val PR-AUC: {val_eval['pr_auc']:.3f}")
        
        # Checkpointing based on Validation Macro-F1
        if val_eval["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_eval["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_macro_f1,
                "config": {
                    "architecture": "MobileNetV3-Small",
                    "num_classes": 2,
                    "input_size": 224,
                    "class_labels": CLASS_LABELS,
                    "benchmark_target": "NORMAL TEAT vs VISUALLY ABNORMAL TEAT"
                },
                "split_stats": stats
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[Early Stopping] Triggered at epoch {epoch}. Best epoch was {best_epoch} (Val Macro-F1: {best_val_macro_f1:.4f}).")
                break
                
    total_training_sec = round(time.time() - start_train_time, 2)
    print(f"\nTraining completed in {total_training_sec}s. Best model from Epoch {best_epoch} saved to: {MODEL_SAVE_PATH}")
    
    # Save training history
    with open(HISTORY_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "total_training_sec": total_training_sec}, f, indent=2)
        
    # 5. Evaluate on Untouched Test Set
    print("\n" + "=" * 70)
    print("STEP 3: TEST SET EVALUATION (UNTOUCHED HELD-OUT CASES)")
    print("=" * 70)
    
    # Load best checkpoint
    best_checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    
    test_eval = evaluate_model(model, test_loader, device, criterion)
    
    print(f"Test Loss:             {test_eval['loss']:.4f}")
    print(f"Balanced Accuracy:     {test_eval['balanced_accuracy']:.4f} ({test_eval['balanced_accuracy']*100:.1f}%)")
    print(f"Macro F1-Score:        {test_eval['macro_f1']:.4f}")
    print(f"Macro Precision:       {test_eval['precision_macro']:.4f}")
    print(f"Macro Recall:          {test_eval['recall_macro']:.4f}")
    print(f"PR-AUC (Abnormal):     {test_eval['pr_auc']:.4f}")
    print("-" * 70)
    print("Per-Class Breakdown:")
    print(f"  Class 0 (NORMAL TEAT):           Precision={test_eval['per_class']['normal_teat']['precision']:.4f}, Recall={test_eval['per_class']['normal_teat']['recall']:.4f}, F1={test_eval['per_class']['normal_teat']['f1']:.4f}")
    print(f"  Class 1 (VISUALLY ABNORMAL TEAT): Precision={test_eval['per_class']['abnormal_teat']['precision']:.4f}, Recall={test_eval['per_class']['abnormal_teat']['recall']:.4f}, F1={test_eval['per_class']['abnormal_teat']['f1']:.4f}")
    print("-" * 70)
    print("Confusion Matrix:")
    print(f"  True Normal:    [TN={test_eval['confusion_matrix'][0][0]}, FP={test_eval['confusion_matrix'][0][1]}]")
    print(f"  True Abnormal:  [FN={test_eval['confusion_matrix'][1][0]}, TP={test_eval['confusion_matrix'][1][1]}]")
    print("-" * 70)
    print("Full Classification Report:")
    report_text = classification_report(
        test_eval["targets"],
        test_eval["predictions"],
        target_names=CLASS_NAMES,
        zero_division=0
    )
    print(report_text)
    
    # Save Metrics JSON
    metrics_summary = {
        "benchmark_target": "NORMAL TEAT vs VISUALLY ABNORMAL TEAT",
        "model_architecture": "MobileNetV3-Small",
        "best_epoch": best_epoch,
        "test_metrics": {
            "balanced_accuracy": test_eval["balanced_accuracy"],
            "macro_f1": test_eval["macro_f1"],
            "macro_precision": test_eval["precision_macro"],
            "macro_recall": test_eval["recall_macro"],
            "pr_auc": test_eval["pr_auc"],
            "loss": test_eval["loss"],
            "per_class": test_eval["per_class"],
            "confusion_matrix": {
                "TN": test_eval["confusion_matrix"][0][0],
                "FP": test_eval["confusion_matrix"][0][1],
                "FN": test_eval["confusion_matrix"][1][0],
                "TP": test_eval["confusion_matrix"][1][1]
            }
        },
        "split_statistics": stats,
        "governance_notice": "Experimental visual udder screening benchmark only. Non-diagnostic."
    }
    
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[OK] Metrics saved to: {METRICS_SAVE_PATH}")
    
    # 6. Generate & Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5), dpi=150)
    cm = np.array(test_eval["confusion_matrix"])
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Visual Udder Screening — Confusion Matrix (Test Set)", fontsize=11, fontweight="bold", pad=12)
    plt.colorbar()
    
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, ["Normal", "Abnormal"], fontsize=10)
    plt.yticks(tick_marks, ["Normal", "Abnormal"], fontsize=10)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight="bold", fontsize=14)
                     
    plt.ylabel("True Clinical Label", fontsize=10)
    plt.xlabel("Predicted Screening Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"[OK] Confusion matrix figure saved to: {CONFUSION_MATRIX_PATH}")
    
    # 7. Unseen Test Inference Demonstration
    print("\n" + "=" * 70)
    print("STEP 4: INFERENCE ON UNSEEN TEST SAMPLES")
    print("=" * 70)
    
    sample_indices = []
    # Pick both normal samples in test set
    norm_indices = [i for i, r in enumerate(test_records) if r["target_label"] == 0]
    abn_indices = [i for i, r in enumerate(test_records) if r["target_label"] == 1]
    
    sample_indices.extend(norm_indices)
    sample_indices.extend(abn_indices[:4]) # Pick 4 abnormal test samples
    
    for idx in sample_indices:
        item = test_records[idx]
        with Image.open(item["filepath"]) as img:
            t = eval_transform(img.convert("RGB")).unsqueeze(0).to(device)
            
        with torch.no_grad():
            out = model(t)
            prob = F.softmax(out, dim=1).squeeze(0).cpu().numpy()
            pred_class = int(np.argmax(prob))
            
        true_lbl = CLASS_LABELS[item["target_label"]]
        pred_lbl = CLASS_LABELS[pred_class]
        correct = (pred_class == item["target_label"])
        
        print(f"Sample: {item['filename']:35s} | CaseGroup: {item['case_group']:20s}")
        print(f"  True Label:  {true_lbl}")
        print(f"  Prediction:  {pred_lbl} ({'[MATCH]' if correct else '[MISMATCH]'})")
        print(f"  Confidence:  Normal: {prob[0]*100:.1f}% | Abnormal: {prob[1]*100:.1f}%")
        print("-" * 70)
        
    # 8. Checkpoint Load Verification
    print("\n" + "=" * 70)
    print("STEP 5: VERIFY MODEL CHECKPOINT RELOAD & INTEGRITY")
    print("=" * 70)
    reloaded_checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    assert "model_state_dict" in reloaded_checkpoint
    assert "config" in reloaded_checkpoint
    
    reloaded_model = build_model(num_classes=2).to(device)
    reloaded_model.load_state_dict(reloaded_checkpoint["model_state_dict"])
    reloaded_model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        dummy_out = reloaded_model(dummy_input)
        assert dummy_out.shape == (1, 2)
        
    print(f"[OK] Checkpoint '{MODEL_SAVE_PATH}' reloaded and executed dummy forward pass successfully.")
    print("=" * 70)
    print("VISUAL UDDER MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    train_pipeline()
