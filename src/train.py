import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.dataset import GaitDataset
from src.model import GaitGuardModel

class PyTorchGaitDataset(Dataset):
    """Wrapper connecting GaitDataset to PyTorch DataLoader."""
    def __init__(self, samples, num_frames=16, is_training=False):
        self.dataset = GaitDataset(samples, num_frames=num_frames, is_training=is_training)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frames, label, info = self.dataset.get_sample(idx)
        return torch.from_numpy(frames), torch.tensor(label, dtype=torch.long), info["filename"]

def check_gpu_readiness():
    """
    Enforces strict GPU readiness check. Aborts if CUDA is unavailable.
    """
    print("\n" + "="*60)
    print("GPU READINESS & HARDWARE VERIFICATION CHECK")
    print("="*60)
    print(f"1. torch.__version__:       {torch.__version__}")
    print(f"2. torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"3. torch.cuda.device_count: {torch.cuda.device_count()}")
    
    if not torch.cuda.is_available():
        error_msg = (
            "FATAL ERROR: CUDA is NOT available in the current PyTorch environment!\n"
            "Hardware check indicates NVIDIA GPU is present, but PyTorch was built without CUDA support.\n"
            "ABORTING TRAINING: Silent fallback to CPU is strictly prohibited."
        )
        print(error_msg)
        print("="*60 + "\n")
        raise RuntimeError(error_msg)
        
    gpu_name = torch.cuda.get_device_name(0)
    print(f"4. torch.cuda.get_device_name(0): {gpu_name}")
    print(f"5. torch.version.cuda:            {torch.version.cuda}")
    
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    alloc_mem = torch.cuda.memory_allocated(0)
    res_mem = torch.cuda.memory_reserved(0)
    print(f"6. GPU Memory Information:")
    print(f"   Total VRAM:     {total_mem / (1024**2):.1f} MB ({total_mem / (1024**3):.2f} GB)")
    print(f"   Free VRAM:      {free_mem / (1024**2):.1f} MB ({free_mem / (1024**3):.2f} GB)")
    print(f"   Allocated VRAM: {alloc_mem / (1024**2):.2f} MB")
    print(f"   Reserved VRAM:  {res_mem / (1024**2):.2f} MB")
    print("="*60 + "\n")
    return True

def run_smoke_test(device):
    """
    Smoke test to confirm:
    GaitGuardModel is on CUDA -> tensors on CUDA -> labels on CUDA -> loss on CUDA -> backprop on CUDA -> validation on CUDA
    using CUDA Automatic Mixed Precision (AMP).
    """
    print("\n" + "="*60)
    print("STARTING CUDA SINGLE-BATCH SMOKE TEST WITH AMP")
    print("="*60)
    
    if device.type != "cuda":
        raise RuntimeError("Smoke test must run on CUDA device! CPU fallback prohibited.")
        
    with open("splits.json") as f:
        splits = json.load(f)
        
    smoke_train_samples = splits["train"][:2]
    smoke_val_samples = splits["val"][:2]
    
    train_loader = DataLoader(PyTorchGaitDataset(smoke_train_samples, num_frames=16, is_training=True), batch_size=2, shuffle=True)
    val_loader = DataLoader(PyTorchGaitDataset(smoke_val_samples, num_frames=16, is_training=False), batch_size=2, shuffle=False)
    
    print(f"[1/6] Moving GaitGuardModel to CUDA ({torch.cuda.get_device_name(0)})...")
    model = GaitGuardModel(num_classes=2, hidden_dim=64, pretrained=True).to(device)
    assert next(model.parameters()).is_cuda, "Model parameters must reside on CUDA!"
    print(f"      Model parameter device: {next(model.parameters()).device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    print("[2/6] Loading single batch onto CUDA...")
    model.train()
    for batch_x, batch_y, filenames in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        assert batch_x.is_cuda, "Input video tensor must be on CUDA!"
        assert batch_y.is_cuda, "Labels must be on CUDA!"
        print(f"      Video tensor device: {batch_x.device}, shape: {batch_x.shape}")
        print(f"      Label tensor device: {batch_y.device}, labels: {batch_y.tolist()}")
        
        print("[3/6] Running CUDA forward pass with AMP (autocast float16)...")
        with torch.amp.autocast('cuda'):
            outputs = model(batch_x)
            assert outputs.is_cuda, "Output logits must be on CUDA!"
            loss = criterion(outputs, batch_y)
            assert loss.is_cuda, "Loss computation must be on CUDA!"
            
        print(f"      Output logits shape: {outputs.shape}")
        print(f"      Calculated loss (CUDA): {loss.item():.4f}")
        
        print("[4/6] Running CUDA backward pass with GradScaler...")
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print("      Backpropagation and optimizer update on CUDA successful.")
        
        mem_alloc = torch.cuda.memory_allocated(0) / (1024**2)
        mem_res = torch.cuda.memory_reserved(0) / (1024**2)
        print(f"      VRAM Post-Backward: Allocated={mem_alloc:.2f} MB, Reserved={mem_res:.2f} MB")
        break
        
    print("[5/6] Testing CUDA validation evaluation pass...")
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, filenames in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(batch_x)
                val_loss = criterion(outputs, batch_y)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            print(f"      Val Loss (CUDA): {val_loss.item():.4f}, Preds: {preds.tolist()}")
            break
            
    print("[6/6] CUDA SMOKE TEST COMPLETED SUCCESSFULLY! All components verified on GPU.")
    print("="*60 + "\n")
    return True

def evaluate(model, data_loader, criterion, device):
    """Evaluates model on given data loader and computes all classification metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    all_filenames = []
    
    with torch.no_grad():
        for batch_x, batch_y, filenames in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_filenames.extend(filenames)
            
    n_samples = len(all_targets)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()
    
    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "predictions": all_preds,
        "targets": all_targets,
        "probabilities": [round(p, 4) for p in all_probs],
        "filenames": all_filenames
    }

def train_model(epochs=25, batch_size=4, lr=1e-4, num_frames=16, patience=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    os.makedirs("models", exist_ok=True)
    with open("splits.json") as f:
        splits = json.load(f)
        
    train_dataset = PyTorchGaitDataset(splits["train"], num_frames=num_frames, is_training=True)
    val_dataset = PyTorchGaitDataset(splits["val"], num_frames=num_frames, is_training=False)
    test_dataset = PyTorchGaitDataset(splits["test"], num_frames=num_frames, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"DataLoaders prepared: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test.")
    
    model = GaitGuardModel(num_classes=2, hidden_dim=64, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Differential learning rates: smaller for backbone, higher for temporal head
    backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.gru.parameters()) + list(model.classifier.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.2},
        {"params": head_params, "lr": lr}
    ], weight_decay=1e-3)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = "models/best_model.pth"
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "learning_rates": []
    }
    
    start_time = time.time()
    print("\n" + "="*75)
    print("STARTING FULL CUDA ACCELERATED TRAINING RUN")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Base LR: {lr} | Frames: {num_frames} | AMP: {device.type == 'cuda'}")
    print("="*75)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            train_targets.extend(batch_y.cpu().numpy().tolist())
            
        current_lr = optimizer.param_groups[1]["lr"]
        scheduler.step()
        
        train_avg_loss = train_loss / len(train_dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(round(train_avg_loss, 4))
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["learning_rates"].append(round(current_lr, 6))
        
        gpu_info_str = ""
        if device.type == "cuda":
            alloc_mb = torch.cuda.memory_allocated(0) / (1024**2)
            res_mb = torch.cuda.memory_reserved(0) / (1024**2)
            gpu_info_str = f" | GPU Mem: {alloc_mb:.1f}MB/{res_mb:.1f}MB"
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] LR: {current_lr:.6f}{gpu_info_str} || "
              f"Train Loss: {train_avg_loss:.4f} | Train Acc: {train_acc:.2%} || "
              f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2%} | "
              f"Val Prec: {val_metrics['precision']:.4f} | Val Rec: {val_metrics['recall']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")
        
        # Checkpointing condition: primary is F1 score, secondary is loss
        is_best = False
        if val_metrics["f1"] > best_val_f1 or (val_metrics["f1"] == best_val_f1 and val_metrics["loss"] < best_val_loss):
            best_val_f1 = val_metrics["f1"]
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            is_best = True
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {
                    "num_classes": 2,
                    "hidden_dim": 64,
                    "num_frames": num_frames
                }
            }, best_model_path)
            print(f"  --> [*] Saved new best checkpoint to {best_model_path} (Val F1: {best_val_f1:.4f}, Val Acc: {val_metrics['accuracy']:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered] No improvement for {patience} consecutive epochs.")
                break
                
    total_training_time = round(time.time() - start_time, 2)
    print(f"\nTraining completed in {total_training_time} seconds ({total_training_time/60:.2f} minutes).")
    
    # Load best checkpoint for final evaluation on Test Set
    print("\n" + "="*60)
    print("PHASE 5 — FINAL EVALUATION ON INDEPENDENT TEST SET")
    print("="*60)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    val_final_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    test_preds = test_metrics["predictions"]
    test_targets = test_metrics["targets"]
    correct_count = sum([1 for p, t in zip(test_preds, test_targets) if p == t])
    total_test = len(test_targets)
    incorrect_count = total_test - correct_count
    
    print(f"Best Model Selected from Validation Epoch: {checkpoint['epoch']}")
    print(f"Best Validation Performance: Acc={val_final_metrics['accuracy']:.2%}, F1={val_final_metrics['f1']:.4f}, Prec={val_final_metrics['precision']:.4f}, Rec={val_final_metrics['recall']:.4f}, Loss={val_final_metrics['loss']:.4f}")
    print("\n--- Held-Out Test Set Performance (8 Videos) ---")
    print(f"Test Accuracy:      {test_metrics['accuracy']:.2%} ({test_metrics['accuracy']})")
    print(f"Test Precision:     {test_metrics['precision']:.4f}")
    print(f"Test Recall:        {test_metrics['recall']:.4f}")
    print(f"Test F1-Score:      {test_metrics['f1']:.4f}")
    print(f"Correct Videos:     {correct_count} / {total_test} ({correct_count/total_test:.1%})")
    print(f"Incorrect Videos:   {incorrect_count} / {total_test} ({incorrect_count/total_test:.1%})")
    print(f"Test Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")
    print("\nNote: 'held-out test performance on our hackathon dataset.'")
    print("      No deterministic cow-level split can be guaranteed due to lack of animal identity metadata.")
    print("="*60)
    
    # Save training history
    with open("models/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    # Save comprehensive metrics
    comprehensive_metrics = {
        "model_architecture": "Pretrained ResNet-18 + Bi-directional GRU",
        "training_time_seconds": total_training_time,
        "best_epoch": checkpoint["epoch"],
        "validation_metrics": val_final_metrics,
        "test_metrics": {
            **test_metrics,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "total_test_videos": total_test,
            "evaluation_note": "held-out test performance on our hackathon dataset",
            "leakage_disclaimer": "No deterministic cow-level split can be guaranteed."
        },
        "class_mapping": {"0": "NORMAL", "1": "LAME"}
    }
    with open("models/metrics.json", "w") as f:
        json.dump(comprehensive_metrics, f, indent=2)
        
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(test_metrics["confusion_matrix"])
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["NORMAL", "LAME"], yticklabels=["NORMAL", "LAME"],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title("Test Set Confusion Matrix (GaitGuard AI)", fontsize=13, fontweight="bold")
    plt.ylabel("Actual True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=200)
    plt.close()
    
    # Plot Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(history["val_loss"], label="Val Loss", color="#3498db", linewidth=2)
    ax1.set_title("Cross-Entropy Loss vs Epoch", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    ax2.plot(history["train_acc"], label="Train Acc", color="#2ecc71", linewidth=2)
    ax2.plot(history["val_acc"], label="Val Acc", color="#9b59b6", linewidth=2)
    ax2.plot(history["val_f1"], label="Val F1", color="#f39c12", linestyle="--", linewidth=2)
    ax2.set_title("Accuracy & F1 Score vs Epoch", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("models/loss_accuracy_curve.png", dpi=200)
    plt.close()
    
    print("\nSaved artifacts to models/:")
    print("  - models/best_model.pth")
    print("  - models/metrics.json")
    print("  - models/training_history.json")
    print("  - models/confusion_matrix.png")
    print("  - models/loss_accuracy_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test only")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    
    # Phase 1: Hardware & CUDA Verification
    check_gpu_readiness()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        run_smoke_test(device)
    else:
        # Phase 2: Smoke test verification before long run
        smoke_ok = run_smoke_test(device)
        if smoke_ok:
            # Phase 3 & 4: Full training run with validation & checkpointing
            train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)
