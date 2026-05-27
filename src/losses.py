"""Loss functions for concrete crack detection.

BCEWithLogitsLoss(pos_weight=33.0) | DiceLoss(smooth=1e-5) | CombinedLoss(0.5*BCE + 0.5*Dice)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import POS_WEIGHT, BCE_WEIGHT, DICE_WEIGHT


class BCEWithLogitsLoss(nn.Module):
    """Binary cross-entropy with logits, weighted for crack/non-crack imbalance."""

    def __init__(self, pos_weight: float = None):
        super().__init__()
        if pos_weight is None:
            pos_weight = float(POS_WEIGHT)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred_logits, target)


class DiceLoss(nn.Module):
    """Soft Dice loss operating on sigmoid probabilities.

    Formula: 1 - (2*intersection + smooth) / (|pred| + |target| + smooth)
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            pred_logits: Raw logits (B, C, H, W)
            target: Ground truth (B, C, H, W) in [0, 1]
        """
        pred = torch.sigmoid(pred_logits)
        intersection = (pred * target).sum()
        cardinality = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Equal-weighted combination of BCE (on logits) and Dice (on probabilities)."""

    def __init__(self, bce_weight: float = None, dice_weight: float = None):
        super().__init__()
        if bce_weight is None:
            bce_weight = BCE_WEIGHT
        if dice_weight is None:
            dice_weight = DICE_WEIGHT
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(pred_logits, target)
        loss_dice = self.dice(pred_logits, target)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


# ---------------------------------------------------------------------------
# Inline tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing losses.py ===")

    # --- DiceLoss: perfect match ---
    dice_loss = DiceLoss(smooth=1e-5)
    pred_perfect = torch.tensor([[[[10.0]]]])  # large logit → sigmoid ≈ 1
    target_ones = torch.ones(1, 1, 1, 1)
    loss_perfect = dice_loss(pred_perfect, target_ones)
    print(f"Perfect match Dice loss: {loss_perfect.item():.6f}")
    assert loss_perfect.item() < 0.001, f"Expected ~0, got {loss_perfect.item()}"

    # --- DiceLoss: no overlap ---
    pred_ones = torch.tensor([[[[-10.0]]]])  # negative logit → sigmoid ≈ 0... wait, this should be sigmoid≈0 for "no overlap with target=all-ones"
    # Actually for "no overlap": pred=all-1 but target=all-0
    # Let's test both interpretations from the spec:
    # "No overlap: all-ones pred vs all-zeros target → Dice ≈ 1"
    pred_all_ones = torch.tensor([[[[100.0]]]])  # sigmoid ≈ 1
    target_zeros = torch.zeros(1, 1, 1, 1)
    loss_no_overlap = dice_loss(pred_all_ones, target_zeros)
    print(f"No-overlap Dice loss (all-1 pred vs all-0 target): {loss_no_overlap.item():.6f}")
    assert loss_no_overlap.item() > 0.99, f"Expected ~1, got {loss_no_overlap.item()}"

    # --- BCE gradient exists ---
    bce_loss = BCEWithLogitsLoss()
    pred_var = torch.randn(2, 1, 8, 8, requires_grad=True)
    target_var = torch.randint(0, 2, (2, 1, 8, 8)).float()
    loss_bce = bce_loss(pred_var, target_var)
    loss_bce.backward()
    assert pred_var.grad is not None, "BCE gradient should exist"
    assert pred_var.grad.abs().sum() > 0, "BCE gradient should be non-zero"
    print(f"BCE gradient norm: {pred_var.grad.norm().item():.4f}")

    # --- CombinedLoss forward ---
    combined = CombinedLoss()
    loss_combined = combined(pred_var.detach(), target_var)
    print(f"Combined loss: {loss_combined.item():.4f}")
    assert loss_combined.item() > 0, "Combined loss should be positive"

    # --- Dice: pred exactly equals target (both all-zeros) ---
    pred_zeros = torch.zeros(1, 1, 1, 1)  # sigmoid(0) = 0.5, not zero...
    # Better: use very negative logits
    pred_neg = torch.tensor([[[[-100.0]]]])  # sigmoid ≈ 0
    loss_both_zero = dice_loss(pred_neg, target_zeros)
    print(f"Both-zero Dice loss (pred≈0, target=0): {loss_both_zero.item():.6f}")
    # When both are all zeros: intersection=0, |pred|≈0, |target|=0 → dice = smooth/smooth = 1 → loss = 0
    assert loss_both_zero.item() < 0.001, f"Expected ~0, got {loss_both_zero.item()}"

    print("All losses.py tests passed!")
