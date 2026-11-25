import csv
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import ProjectConfig
from src.cnn.dataset import Galaxy10Dataset
from src.cnn.model   import build_resnet18, freeze_backbone, unfreeze_all


# ----------------------------------------------------------------------
# Single-epoch runner
# ----------------------------------------------------------------------
def _run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss    += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total         += images.size(0)

    return total_loss / total, total_correct / total


# ----------------------------------------------------------------------
# Main trainer
# ----------------------------------------------------------------------
def train_cnn(train_idx, val_idx,
              h5_path=ProjectConfig.DATA_PATH,
              verbose=True):
    """
    Two-phase fine-tuning of ResNet18 on Galaxy10.

    Returns dict with keys: model, best_val_accuracy, history.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\n[CNN] device = {device}")
        if device.type == "cpu":
            print("[CNN] WARNING: training on CPU is slow (hours). "
                  "Enable a GPU for ~20-minute training.")

    # ---- datasets ----
    train_ds = Galaxy10Dataset(
        h5_path, train_idx, ProjectConfig.CNN_IMAGE_SIZE, augment=True
    )
    val_ds = Galaxy10Dataset(
        h5_path, val_idx, ProjectConfig.CNN_IMAGE_SIZE, augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=ProjectConfig.CNN_BATCH_SIZE,
        shuffle=True,
        num_workers=ProjectConfig.CNN_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(ProjectConfig.CNN_NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ProjectConfig.CNN_BATCH_SIZE,
        shuffle=False,
        num_workers=ProjectConfig.CNN_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(ProjectConfig.CNN_NUM_WORKERS > 0),
    )

    # ---- model ----
    model = build_resnet18().to(device)
    ckpt = ProjectConfig.CNN_MODEL_PATH.parent / "checkpoint_latest.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[CNN] resumed from {ckpt}")
        
    criterion = nn.CrossEntropyLoss(
        label_smoothing=ProjectConfig.CNN_LABEL_SMOOTHING
    )

    best_acc   = 0.0
    best_state = copy.deepcopy(model.state_dict())
    history    = []

    # ---- Phase 1: head only ----
    if verbose:
        print(f"\n[CNN] Phase 1 — frozen backbone, "
              f"{ProjectConfig.CNN_EPOCHS_HEAD} epochs")

    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ProjectConfig.CNN_LR_HEAD,
        weight_decay=ProjectConfig.CNN_WEIGHT_DECAY,
    )

    for epoch in range(ProjectConfig.CNN_EPOCHS_HEAD):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion,
                                     optimizer, device, train=True)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion,
                                     optimizer, device, train=False)

        history.append({
            "phase": "head", "epoch": epoch + 1,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss":   va_loss, "val_acc":   va_acc,
            "seconds": round(time.time() - t0, 1),
        })

        if verbose:
            print(f"  epoch {epoch + 1:>2}/{ProjectConfig.CNN_EPOCHS_HEAD}  "
                  f"train {tr_loss:.3f}/{tr_acc:.3f}  "
                  f"val {va_loss:.3f}/{va_acc:.3f}  "
                  f"({time.time() - t0:.0f}s)")

        if va_acc > best_acc:
            best_acc   = va_acc
            best_state = copy.deepcopy(model.state_dict())
            if verbose:
                print(f"    ↳ new best: {best_acc:.4f}")


    # Safety checkpoint — survives interruption
    torch.save(
        model.state_dict(),
        ProjectConfig.CNN_MODEL_PATH.parent / "checkpoint_latest.pt",
    )

    # ---- Phase 2: fine-tune all ----
    if verbose:
        print(f"\n[CNN] Phase 2 — unfrozen, "
              f"{ProjectConfig.CNN_EPOCHS_FINETUNE} epochs")

    unfreeze_all(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ProjectConfig.CNN_LR_FINETUNE,
        weight_decay=ProjectConfig.CNN_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ProjectConfig.CNN_EPOCHS_FINETUNE
    )

    for epoch in range(ProjectConfig.CNN_EPOCHS_FINETUNE):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion,
                                     optimizer, device, train=True)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion,
                                     optimizer, device, train=False)
        scheduler.step()

        history.append({
            "phase": "finetune", "epoch": epoch + 1,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss":   va_loss, "val_acc":   va_acc,
            "seconds": round(time.time() - t0, 1),
        })

        if verbose:
            print(f"  epoch {epoch + 1:>2}/{ProjectConfig.CNN_EPOCHS_FINETUNE}  "
                  f"train {tr_loss:.3f}/{tr_acc:.3f}  "
                  f"val {va_loss:.3f}/{va_acc:.3f}  "
                  f"({time.time() - t0:.0f}s)")

        if va_acc > best_acc:
            best_acc   = va_acc
            best_state = copy.deepcopy(model.state_dict())
            if verbose:
                print(f"    ↳ new best: {best_acc:.4f}")

    # Safety checkpoint — survives interruption
    torch.save(
        model.state_dict(),
        ProjectConfig.CNN_MODEL_PATH.parent / "checkpoint_latest.pt",
    )

    # ---- persist ----
    model.load_state_dict(best_state)
    model.to("cpu")

    ProjectConfig.CNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # FIX: verify the live model is trained, then save it 
    live_std = model.fc.weight.std().item()
    print(f"[SAVE] live fc.std = {live_std:.4f}")
    if live_std < 0.05:
        raise RuntimeError(
            f"fc.std={live_std:.4f} ≈ random init — model not trained, refusing to save. "
            f"Check optimizer construction in train_cnn()."
        )
    torch.save(model.state_dict(), ProjectConfig.CNN_MODEL_PATH)
    print(f"[SAVE] ✓ saved trained model (fc.std={live_std:.4f})")
    torch.save(best_state, ProjectConfig.CNN_MODEL_PATH)

    with open(ProjectConfig.CNN_HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    if verbose:
        print(f"\n[CNN] best val accuracy : {best_acc:.4f}")
        print(f"[CNN] model saved       : {ProjectConfig.CNN_MODEL_PATH}")
        print(f"[CNN] history saved     : {ProjectConfig.CNN_HISTORY_PATH}")

    return {
        "model":             model,
        "best_val_accuracy": float(best_acc),
        "history":           history,
    }


# ----------------------------------------------------------------------
# Training-curve plotter
# ----------------------------------------------------------------------
def plot_training_history(history, out_path=None):
    """
    Plot loss and accuracy curves (train vs val) from the training history.
    Splits visually at the head → finetune phase boundary.
    """
    epochs       = [h["epoch"]      for h in history]
    train_loss   = [h["train_loss"] for h in history]
    val_loss     = [h["val_loss"]   for h in history]
    train_acc    = [h["train_acc"]  for h in history]
    val_acc      = [h["val_acc"]    for h in history]
    phase_switch = sum(1 for h in history if h["phase"] == "head")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "o-", label="Train")
    ax1.plot(epochs, val_loss,   "s-", label="Validation")
    ax1.axvline(phase_switch + 0.5, color="grey", ls="--",
                label="head → finetune")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("CNN Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, "o-", label="Train")
    ax2.plot(epochs, val_acc,   "s-", label="Validation")
    ax2.axvline(phase_switch + 0.5, color="grey", ls="--",
                label="head → finetune")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("CNN Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if out_path is None:
        out_path = ProjectConfig.FIGURES_PATH / "cnn_training_curves.png"
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
