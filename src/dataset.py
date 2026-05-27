"""DeepCrackDataset for concrete crack segmentation.

Loads RGB images + binary label masks, applies transforms, normalizes per model.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import MODEL_NORM, IMG_H, IMG_W


class DeepCrackDataset(Dataset):
    """Dataset for DeepCrack crack detection.

    Args:
        img_dir: Path to directory containing JPEG images.
        lab_dir: Path to directory containing PNG labels. If None, dataset is in test mode.
        transform: Albumentations Compose object (from src.augment.get_transforms).
        model_name: One of "unet", "deeplabv3plus", "fcn8s". Determines normalization.
    """

    def __init__(
        self,
        img_dir: Union[str, Path],
        lab_dir: Optional[Union[str, Path]],
        transform,
        model_name: str,
    ):
        self.img_dir = Path(img_dir)
        self.lab_dir = Path(lab_dir) if lab_dir is not None else None
        self.transform = transform
        self.model_name = model_name

        if model_name not in MODEL_NORM:
            raise ValueError(
                f"model_name must be one of {list(MODEL_NORM.keys())}, got {model_name}"
            )

        self.norm = MODEL_NORM[model_name]
        self.mean = np.array(self.norm["mean"], dtype=np.float32)
        self.std = np.array(self.norm["std"], dtype=np.float32)

        self.img_paths = sorted(self.img_dir.glob("*.jpg"))
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No .jpg images found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, str]]:
        img_path = self.img_paths[idx]

        # Read image (BGR → RGB, uint8)
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        filename = img_path.name

        if self.lab_dir is not None:
            # Train/val mode: load label
            lab_path = self.lab_dir / (img_path.stem + ".png")
            label = cv2.imread(str(lab_path), cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise FileNotFoundError(f"Could not read label: {lab_path}")

            # Apply albumentations transform (synced on image + mask)
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

            # Convert label to float32 binary mask in [0, 1]
            label = label.astype(np.float32) / 255.0
            label = (label > 0.5).astype(np.float32)

            # Normalize image
            image = image.astype(np.float32) / 255.0
            image = (image - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)

            # Convert to CHW tensors
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()        # (3, H, W)
            mask_tensor = torch.from_numpy(label).unsqueeze(0).float()              # (1, H, W)

            return image_tensor, mask_tensor
        else:
            # Test mode: no labels, return filename for saving results
            # Still apply transform (identity for val) for consistency
            augmented = self.transform(image=image)
            image = augmented["image"]

            # Normalize image
            image = image.astype(np.float32) / 255.0
            image = (image - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)

            # Convert to CHW tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()        # (3, H, W)

            return image_tensor, filename


# ---------------------------------------------------------------------------
# Inline tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing dataset.py ===")

    from src.augment import get_transforms
    from src.config import TRAIN_IMG, TRAIN_LAB

    # Test 1: Instantiate with train data, verify basic properties
    train_transform = get_transforms(train=True)
    dataset = DeepCrackDataset(
        img_dir=TRAIN_IMG,
        lab_dir=TRAIN_LAB,
        transform=train_transform,
        model_name="unet",
    )

    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 300, f"Expected 300 training images, got {len(dataset)}"

    # Test 2: Check shapes and value ranges
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")

    assert img.shape == (3, IMG_H, IMG_W), f"Expected (3, 384, 544), got {img.shape}"
    assert mask.shape == (1, IMG_H, IMG_W), f"Expected (1, 384, 544), got {mask.shape}"
    assert mask.dtype == torch.float32

    mask_np = mask.numpy()
    assert mask_np.min() >= 0.0 and mask_np.max() <= 1.0, \
        f"Mask values should be in [0,1], got [{mask_np.min()}, {mask_np.max()}]"
    unique_vals = np.unique(mask_np)
    print(f"Unique mask values: {unique_vals}")
    assert set(unique_vals).issubset({0.0, 1.0}), f"Mask should be binary, got {unique_vals}"

    # Test 3: U-Net normalization — values should be in roughly [0, 1]
    img_np = img.numpy()
    print(f"U-Net image value range: [{img_np.min():.4f}, {img_np.max():.4f}]")
    assert img_np.min() >= -0.1 and img_np.max() <= 1.1, \
        f"U-Net image should be in [0,1] range, got [{img_np.min()}, {img_np.max()}]"

    # Test 4: Pretrained normalization (ImageNet stats) — values should be centered around 0
    pretrained_dataset = DeepCrackDataset(
        img_dir=TRAIN_IMG,
        lab_dir=TRAIN_LAB,
        transform=get_transforms(train=False),  # deterministic for this test
        model_name="deeplabv3plus",
    )
    img_pt, mask_pt = pretrained_dataset[0]
    img_pt_np = img_pt.numpy()
    print(f"ImageNet-norm image value range: [{img_pt_np.min():.4f}, {img_pt_np.max():.4f}]")
    # After ImageNet normalization, values should have mean ~0 (roughly)
    channel_means = img_pt_np.mean(axis=(1, 2))
    print(f"Channel means: {channel_means}")
    assert abs(channel_means[0]) < 3.0, f"Channel mean should be near 0 after ImageNet norm"

    # Test 5: Test mode (no labels)
    test_dataset = DeepCrackDataset(
        img_dir=TRAIN_IMG,
        lab_dir=None,
        transform=get_transforms(train=False),
        model_name="unet",
    )
    result = test_dataset[0]
    assert isinstance(result, tuple) and len(result) == 2
    img_test, fname = result
    assert isinstance(fname, str), f"Second element should be filename string, got {type(fname)}"
    print(f"Test mode — image shape: {img_test.shape}, filename: {fname}")

    # Test 6: Invalid model_name
    try:
        DeepCrackDataset(TRAIN_IMG, TRAIN_LAB, get_transforms(train=False), "invalid_model")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("Invalid model_name correctly raises ValueError")

    print("All dataset.py tests passed!")
