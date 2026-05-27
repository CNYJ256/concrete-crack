"""Training loop for concrete crack detection.

Run as: python -m src.train [--smoke]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from src.config import (
    TRAIN_IMG, TRAIN_LAB,
    BATCH_SIZE, NUM_EPOCHS, PHASE1_EPOCHS,
    LR, WEIGHT_DECAY, NUM_WORKERS, SEED,
    VAL_SPLIT, PIN_MEMORY, GRAD_CLIP,
    BCE_WEIGHT, DICE_WEIGHT, POS_WEIGHT,
    MODEL_NAMES, CKPT_DIR, LOG_DIR, RESULT_DIR,
)
from src.utils import set_seed, ensure_dirs
from src.augment import get_transforms
from src.dataset import DeepCrackDataset
from src.models import create_model
from src.losses import CombinedLoss
from src.evaluate import evaluate_model


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _split_train_val(model_name: str):
    """Split TRAIN_IMG filenames into train (80%) and val (20%) with seed=42.

    Returns (train_dataset, val_dataset) as Subset wrappers.
    """
    # Sorted list matches DeepCrackDataset's internal ordering
    all_names = sorted([p.name for p in TRAIN_IMG.glob("*.jpg")])

    # Shuffle with fixed seed
    rng = np.random.RandomState(SEED)
    indices = np.arange(len(all_names))
    rng.shuffle(indices)

    split = int(len(all_names) * (1.0 - VAL_SPLIT))
    train_idx = sorted(indices[:split].tolist())
    val_idx = sorted(indices[split:].tolist())

    # Create full dataset then slice with Subset
    train_full = DeepCrackDataset(
        img_dir=TRAIN_IMG,
        lab_dir=TRAIN_LAB,
        transform=get_transforms(train=True),
        model_name=model_name,
    )
    val_full = DeepCrackDataset(
        img_dir=TRAIN_IMG,
        lab_dir=TRAIN_LAB,
        transform=get_transforms(train=False),
        model_name=model_name,
    )

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    print(f"  Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")
    return train_ds, val_ds


def _build_loaders(train_ds, val_ds):
    """Create DataLoaders for training and validation."""
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,  # avoid batch-norm issues on small last batch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(model, val_loader, device):
    """Run evaluate_model and return the global IoU scalar."""
    model.eval()
    results = evaluate_model(model, val_loader, device)
    model.train()
    return results["global_iou"][0]  # (mean, std) → take mean


# ---------------------------------------------------------------------------
# Training one model
# ---------------------------------------------------------------------------

def train_one_model(model_name: str, smoke: bool = False):
    """Full training pipeline for a single model architecture.

    Args:
        model_name: One of "unet", "deeplabv3plus", "fcn8s".
        smoke: If True, run only 2 epochs to verify pipeline integrity.

    Returns:
        dict with keys: model_name, best_val_iou, best_epoch, history.
    """
    total_epochs = 2 if smoke else NUM_EPOCHS
    phase1_end = min(PHASE1_EPOCHS, total_epochs)
    val_interval = max(1, total_epochs // 20)  # validate ~20 times across training
    if smoke:
        val_interval = 1

    print(f"\n{'=' * 60}")
    print(f"Training: {model_name}")
    print(f"  Epochs: {total_epochs}  |  Phase 1: 1-{phase1_end} (BCE warmup)")
    if total_epochs > phase1_end:
        print(f"  Phase 2: {phase1_end + 1}-{total_epochs} (BCE+Dice + CosineLR + GradClip)")
    print(f"  Validation every {val_interval} epoch(s)")
    print(f"{'=' * 60}")

    # ---- Setup ----
    set_seed(SEED)
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    model = create_model(model_name, num_classes=1).to(device)
    train_ds, val_ds = _split_train_val(model_name)
    train_loader, val_loader = _build_loaders(train_ds, val_ds)

    # ---- Phase 1: BCE warmup ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT], device=device)
    )
    scaler = GradScaler("cuda")

    # ---- Phase 2 state (initialized later) ----
    combined_loss = None
    scheduler = None
    in_phase2 = False

    # ---- Bookkeeping ----
    history = {"train_loss": [], "val_iou": [], "val_epochs": []}
    best_val_iou = 0.0
    best_epoch = 0

    # ---- Training loop ----
    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        # --- Phase transition ---
        if epoch == phase1_end + 1 and not in_phase2:
            in_phase2 = True
            combined_loss = CombinedLoss(
                bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT
            ).to(device)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - phase1_end,
                eta_min=1e-6,
            )
            print(f"\n  >>> Switching to Phase 2: CombinedLoss + CosineAnnealingLR <<<\n")

        phase_label = "Phase 1" if epoch <= phase1_end else "Phase 2"
        model.train()

        epoch_losses = []
        for batch_idx, batch in enumerate(train_loader):
            images, masks = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = model(images)
                loss = bce_loss(logits, masks) if not in_phase2 else combined_loss(logits, masks)

            scaler.scale(loss).backward()

            if in_phase2:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [{phase_label}] Epoch {epoch:3d}/{total_epochs}  "
                    f"Batch {batch_idx + 1:3d}/{len(train_loader)}  "
                    f"loss={loss.item():.4f}  lr={current_lr:.2e}"
                )

        avg_loss = float(np.mean(epoch_losses))
        elapsed = time.time() - epoch_start

        # Step scheduler in Phase 2
        if in_phase2 and scheduler is not None:
            scheduler.step()

        # --- Validation ---
        val_iou = None
        if epoch % val_interval == 0 or epoch == total_epochs:
            val_iou = run_validation(model, val_loader, device)
            phase_str = "P1" if epoch <= phase1_end else "P2"
            print(
                f"  [{phase_str}] Epoch {epoch:3d}/{total_epochs}  "
                f"Train Loss: {avg_loss:.4f}  "
                f"Val IoU: {val_iou:.4f}  "
                f"Time: {elapsed:.1f}s"
            )
        else:
            phase_str = "P1" if epoch <= phase1_end else "P2"
            print(
                f"  [{phase_str}] Epoch {epoch:3d}/{total_epochs}  "
                f"Train Loss: {avg_loss:.4f}  "
                f"Time: {elapsed:.1f}s"
            )

        # Record history
        history["train_loss"].append(avg_loss)
        if val_iou is not None:
            history["val_iou"].append(val_iou)
            history["val_epochs"].append(epoch)
        else:
            history["val_iou"].append(np.nan)
            history["val_epochs"].append(epoch)

        # --- Checkpointing ---
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "model_name": model_name,
            "phase": "phase2" if in_phase2 else "phase1",
        }
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        if val_iou is not None:
            ckpt["val_iou"] = val_iou

        # Save latest
        latest_path = CKPT_DIR / f"{model_name}_latest.pth"
        torch.save(ckpt, latest_path)

        # Save best (by val IoU)
        if val_iou is not None and val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            best_path = CKPT_DIR / f"{model_name}_best.pth"
            torch.save(ckpt, best_path)
            print(f"  >>> New best val IoU: {best_val_iou:.4f} @ epoch {best_epoch}")

    # Save training history
    hist_path = CKPT_DIR / f"{model_name}_history.npz"
    np.savez(hist_path, **history)
    print(f"  History saved to {hist_path}")

    return {
        "model_name": model_name,
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list):
    """Print and save summary of all trained models."""
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")
    header = f"{'Model':<20} {'Best Val IoU':>14} {'Best Epoch':>12}"
    print(header)
    print("-" * len(header))

    csv_lines = ["model,best_val_iou,best_epoch"]
    for r in results:
        line = f"{r['model_name']:<20} {r['best_val_iou']:>14.4f} {r['best_epoch']:>12}"
        print(line)
        csv_lines.append(f"{r['model_name']},{r['best_val_iou']:.6f},{r['best_epoch']}")

    # Save CSV
    csv_path = RESULT_DIR / "training_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(csv_lines) + "\n")
    print(f"\nSummary saved to {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train crack detection models")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only 2 epochs per model to verify the pipeline.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Concrete Crack Detection — Training Pipeline")
    print(f"Models: {MODEL_NAMES}")
    print(f"Smoke mode: {'ON (2 epochs)' if args.smoke else 'OFF'}")
    print("=" * 60)

    all_results = []
    for model_name in MODEL_NAMES:
        try:
            result = train_one_model(model_name, smoke=args.smoke)
            all_results.append(result)
        except KeyboardInterrupt:
            print(f"\n  KeyboardInterrupt during {model_name}. Saving latest checkpoint...")
            # Checkpoint was already saved at end of last completed epoch
            continue
        except Exception as e:
            print(f"\n  ERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        print_summary(all_results)
    else:
        print("\n  No models were trained successfully.")


if __name__ == "__main__":
    main()
