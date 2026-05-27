"""Evaluation metrics for crack detection.

Per-image Precision/Recall/F1/IoU with mean±std, plus global pixel-wise aggregation.
Handles all-background images gracefully.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_metrics(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Compute per-image segmentation metrics.

    Args:
        pred_binary: Binary prediction (H, W) or (1, H, W), dtype bool or uint8/float.
        gt_binary: Binary ground truth (H, W) or (1, H, W), dtype bool or uint8/float.
        eps: Small constant to avoid division by zero.

    Returns:
        Dict with keys "precision", "recall", "f1", "iou".
    """
    pred = np.asarray(pred_binary, dtype=bool).squeeze()
    gt = np.asarray(gt_binary, dtype=bool).squeeze()

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()

    # Handle all-background images
    if tp + fn == 0:
        # No cracks in GT
        if tp + fp == 0:
            # No cracks predicted either — technically perfect on empty scene
            precision = 1.0
        else:
            precision = 0.0
        recall = 0.0
    else:
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    conf_threshold: float = 0.5,
) -> Dict[str, Tuple[float, float]]:
    """Evaluate a model on a dataloader, computing per-image and global metrics.

    Args:
        model: PyTorch model outputting logits (B, 1, H, W).
        dataloader: DataLoader yielding (image, mask) batches.
        device: torch.device to run inference on.
        conf_threshold: Threshold on sigmoid probability for binarization.

    Returns:
        Dict mapping metric name to (mean, std) over all images.
        Also includes "global_*" keys for pixel-wise aggregated metrics.
    """
    model.eval()
    per_image_metrics: Dict[str, list] = {
        "precision": [],
        "recall": [],
        "f1": [],
        "iou": [],
    }

    # Global accumulators (pixel-wise across all images)
    global_tp = 0
    global_fp = 0
    global_fn = 0

    with torch.no_grad():
        for batch in dataloader:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > conf_threshold).float()

            # Per-image metrics
            for i in range(preds.size(0)):
                pred_np = preds[i].cpu().numpy()
                mask_np = masks[i].cpu().numpy()
                metrics = compute_metrics(pred_np, mask_np)
                for k, v in metrics.items():
                    per_image_metrics[k].append(v)

                # Global accumulators
                pred_bool = pred_np.astype(bool).squeeze()
                mask_bool = mask_np.astype(bool).squeeze()
                global_tp += (pred_bool & mask_bool).sum()
                global_fp += (pred_bool & ~mask_bool).sum()
                global_fn += (~pred_bool & mask_bool).sum()

    # Compute per-image mean ± std
    eps = 1e-7
    results: Dict[str, Tuple[float, float]] = {}
    for metric_name, values in per_image_metrics.items():
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        results[metric_name] = (mean_val, std_val)

    # Compute global (pixel-wise aggregated) metrics
    global_precision = global_tp / (global_tp + global_fp + eps)
    global_recall = global_tp / (global_tp + global_fn + eps)
    global_f1 = 2.0 * global_precision * global_recall / (global_precision + global_recall + eps)
    global_iou = global_tp / (global_tp + global_fp + global_fn + eps)

    results["global_precision"] = (float(global_precision), 0.0)
    results["global_recall"] = (float(global_recall), 0.0)
    results["global_f1"] = (float(global_f1), 0.0)
    results["global_iou"] = (float(global_iou), 0.0)

    return results


# ---------------------------------------------------------------------------
# Inline tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing evaluate.py ===")

    eps = 1e-7
    H, W = 8, 8

    # --- Test 1: Perfect prediction ---
    pred_perfect = np.ones((H, W), dtype=np.uint8)
    gt_perfect = np.ones((H, W), dtype=np.uint8)
    m = compute_metrics(pred_perfect, gt_perfect)
    print(f"Perfect: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, IoU={m['iou']:.4f}")
    assert abs(m["precision"] - 1.0) < 1e-6, f"Precision should be 1, got {m['precision']}"
    assert abs(m["recall"] - 1.0) < 1e-6, f"Recall should be 1, got {m['recall']}"
    assert abs(m["f1"] - 1.0) < 1e-6, f"F1 should be 1, got {m['f1']}"
    assert abs(m["iou"] - 1.0) < 1e-6, f"IoU should be 1, got {m['iou']}"

    # --- Test 2: All false positives (no overlap) ---
    pred_fp = np.ones((H, W), dtype=np.uint8)
    gt_fp = np.zeros((H, W), dtype=np.uint8)
    m = compute_metrics(pred_fp, gt_fp)
    print(f"All FP: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, IoU={m['iou']:.4f}")
    assert abs(m["precision"] - 0.0) < 1e-6, f"Precision should be 0, got {m['precision']}"
    assert abs(m["recall"] - 0.0) < 1e-6, f"Recall should be 0, got {m['recall']}"
    assert abs(m["f1"] - 0.0) < 1e-6, f"F1 should be 0, got {m['f1']}"
    assert abs(m["iou"] - 0.0) < 1e-6, f"IoU should be 0, got {m['iou']}"

    # --- Test 3: All background GT (no cracks) with FP ---
    pred_bg_fp = np.ones((H, W), dtype=np.uint8)  # predicted cracks everywhere
    gt_bg = np.zeros((H, W), dtype=np.uint8)        # GT is all background
    m = compute_metrics(pred_bg_fp, gt_bg)
    print(f"All-bg GT + all-FP pred: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, IoU={m['iou']:.4f}")
    # TP=0, FP=H*W, FN=0, TP+FN=0 → precision=0 (FP>0), recall=0
    assert m["precision"] == 0.0, f"Precision should be 0, got {m['precision']}"
    assert m["recall"] == 0.0, f"Recall should be 0, got {m['recall']}"

    # --- Test 4: All background GT (no cracks) with perfect empty prediction ---
    pred_bg_correct = np.zeros((H, W), dtype=np.uint8)
    m = compute_metrics(pred_bg_correct, gt_bg)
    print(f"All-bg GT + correct empty pred: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, IoU={m['iou']:.4f}")
    # TP=0, FP=0, FN=0, TP+FN=0 → precision=1.0, recall=0.0
    assert m["precision"] == 1.0, f"Precision should be 1.0 (no FP), got {m['precision']}"
    assert m["recall"] == 0.0, f"Recall should be 0.0, got {m['recall']}"

    # --- Test 5: Mixed prediction ---
    # Create a simple case: GT has 4 crack pixels, pred has 3/4 correct + 2 FP
    gt_mixed = np.zeros((H, W), dtype=np.uint8)
    gt_mixed[0, 0] = 1
    gt_mixed[0, 1] = 1
    gt_mixed[1, 0] = 1
    gt_mixed[1, 1] = 1  # 4 crack pixels

    pred_mixed = np.zeros((H, W), dtype=np.uint8)
    pred_mixed[0, 0] = 1  # TP
    pred_mixed[0, 1] = 1  # TP
    pred_mixed[1, 0] = 1  # TP
    # missing gt_mixed[1,1] → FN
    pred_mixed[2, 0] = 1  # FP
    pred_mixed[2, 1] = 1  # FP

    m = compute_metrics(pred_mixed, gt_mixed)
    print(f"Mixed: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, IoU={m['iou']:.4f}")
    # TP=3, FP=2, FN=1
    # P = 3/(3+2) = 0.6
    # R = 3/(3+1) = 0.75
    # F1 = 2*0.6*0.75/(0.6+0.75) = 0.9/1.35 = 0.6667
    # IoU = 3/(3+2+1) = 0.5
    assert abs(m["precision"] - 0.6) < 1e-6, f"Precision should be 0.6, got {m['precision']}"
    assert abs(m["recall"] - 0.75) < 1e-6, f"Recall should be 0.75, got {m['recall']}"
    assert abs(m["f1"] - 0.6666667) < 1e-4, f"F1 should be ~0.6667, got {m['f1']}"
    assert abs(m["iou"] - 0.5) < 1e-6, f"IoU should be 0.5, got {m['iou']}"

    # --- Test 6: 3D input (with channel dim) ---
    pred_3d = np.ones((1, H, W), dtype=np.uint8)
    gt_3d = np.ones((1, H, W), dtype=np.uint8)
    m = compute_metrics(pred_3d, gt_3d)
    assert abs(m["iou"] - 1.0) < 1e-6, "3D input should work correctly"

    print("All evaluate.py tests passed!")
