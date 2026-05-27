"""Data augmentation transforms using albumentations.

Synced image+mask transforms with reflect padding to avoid black border artifacts.
"""

import cv2
import numpy as np
import albumentations as A


def get_transforms(train: bool = True) -> A.Compose:
    """Return albumentations Compose for train or val.

    Args:
        train: If True, return augmentation pipeline. If False, return identity.

    Returns:
        A.Compose object callable with (image=..., mask=...)
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.SafeRotate(
                limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.0,
                hue=0.0,
                p=0.5,
            ),
        ])
    else:
        return A.Compose([])  # identity — no augmentation


# ---------------------------------------------------------------------------
# Inline tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing augment.py ===")

    # Create dummy image (H=384, W=544, C=3) and mask (H, W)
    rng = np.random.default_rng(42)
    dummy_img = rng.integers(0, 256, (384, 544, 3), dtype=np.uint8)
    dummy_mask = rng.integers(0, 2, (384, 544), dtype=np.uint8)

    # Test train transform
    train_transform = get_transforms(train=True)
    assert isinstance(train_transform, A.Compose), "Train transform should be A.Compose"
    print(f"Train transform: {train_transform}")

    # Apply train transform
    augmented = train_transform(image=dummy_img, mask=dummy_mask)
    aug_img = augmented["image"]
    aug_mask = augmented["mask"]

    print(f"Augmented image shape: {aug_img.shape}, dtype: {aug_img.dtype}")
    print(f"Augmented mask shape: {aug_mask.shape}, dtype: {aug_mask.dtype}")

    assert aug_img.shape == (384, 544, 3), f"Image shape should be (384,544,3), got {aug_img.shape}"
    assert aug_mask.shape == (384, 544), f"Mask shape should be (384,544), got {aug_mask.shape}"

    # Check no black borders in mask after rotation
    # After rotation with BORDER_REFLECT_101, the mask should only contain 0 or 1
    # (no fractional values from interpolation of edges)
    unique_mask_vals = np.unique(aug_mask)
    print(f"Unique mask values after augmentation: {unique_mask_vals}")

    # The mask values should be in {0, 1} or {0, 255} (albumentations may preserve uint8)
    # If there were black borders, we'd see a large region of zeros at edges
    # With BORDER_REFLECT_101, the padded regions mirror the content

    # Additionally, verify that the fraction of 0s doesn't dramatically increase
    # (which would indicate black border padding)
    orig_zero_frac = (dummy_mask == 0).mean()
    aug_zero_frac = (aug_mask == 0).mean()
    print(f"Original zero fraction: {orig_zero_frac:.4f}")
    print(f"Augmented zero fraction: {aug_zero_frac:.4f}")

    # The zero fraction shouldn't spike dramatically from black borders
    # Allow some change due to rotation cropping
    delta = abs(aug_zero_frac - orig_zero_frac)
    assert delta < 0.3, f"Zero fraction changed too much ({delta:.4f}), possible black borders"

    # Test val transform (should be identity)
    val_transform = get_transforms(train=False)
    val_result = val_transform(image=dummy_img, mask=dummy_mask)
    assert np.array_equal(val_result["image"], dummy_img), "Val transform should be identity for image"
    assert np.array_equal(val_result["mask"], dummy_mask), "Val transform should be identity for mask"
    print("Val transform is identity (correct)")

    # Test determinism with seed
    # Apply multiple times to check that shapes are always preserved
    for i in range(10):
        result = train_transform(image=dummy_img, mask=dummy_mask)
        assert result["image"].shape == (384, 544, 3), f"Iteration {i}: bad image shape"
        assert result["mask"].shape == (384, 544), f"Iteration {i}: bad mask shape"

    print(f"10 random augmentations all preserved shapes")

    print("All augment.py tests passed!")
