"""Inference module for concrete crack detection.

Run as: python -m src.predict --model unet --checkpoint <path> --input <path>
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.config import (
    MODEL_NORM, CONF_THRESHOLD,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
)
from src.models import create_model


# ---------------------------------------------------------------------------
# CLAHE preprocessing
# ---------------------------------------------------------------------------

def apply_clahe(image_np: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel of an RGB image in LAB space.

    Args:
        image_np: RGB uint8 image of shape (H, W, 3).

    Returns:
        RGB uint8 image of same shape with CLAHE applied.
    """
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE,
    )
    l_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def predict_single(
    model: torch.nn.Module,
    image_path: str | Path,
    device: torch.device,
    use_clahe: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a single image.

    Args:
        model: Trained segmentation model.
        image_path: Path to input JPEG/PNG image.
        device: torch.device to run on.
        use_clahe: Whether to apply CLAHE preprocessing.

    Returns:
        (binary_mask, original_image)
            binary_mask: (H, W) uint8 array, values 0 or 255.
            original_image: (H, W, 3) uint8 RGB array (before CLAHE).
    """
    model.eval()

    # Load image
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Keep original for overlay
    original_image = image.copy()

    # Optional CLAHE
    if use_clahe:
        image = apply_clahe(image)

    # Normalize: determine model type from class name
    model_class = model.__class__.__name__.lower()
    # Map class names to config keys
    class_to_key = {
        "unet": "unet",
        "deeplabv3plus": "deeplabv3plus",
        "fcn8s": "fcn8s",
        "fcn": "fcn8s",
    }
    model_key = None
    for pattern, key in class_to_key.items():
        if pattern in model_class:
            model_key = key
            break
    if model_key is None:
        raise ValueError(f"Cannot determine normalization for model class: {model_class}")

    norm = MODEL_NORM[model_key]
    mean = np.array(norm["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(norm["std"], dtype=np.float32).reshape(1, 1, 3)

    # Normalize
    image_f = image.astype(np.float32) / 255.0
    image_f = (image_f - mean) / std

    # To tensor: (C, H, W)
    tensor = torch.from_numpy(image_f).permute(2, 0, 1).unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        pred = (probs > CONF_THRESHOLD).float()

    # To numpy binary mask
    binary_mask = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)

    return binary_mask, original_image


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay a semi-transparent colored mask on the original image.

    Args:
        image: RGB uint8 array (H, W, 3).
        mask: Binary uint8 array (H, W), values 0 or 255.
        color: RGB tuple for the overlay color.
        alpha: Opacity of the overlay (0 = invisible, 1 = opaque).

    Returns:
        PIL Image with the mask overlaid.
    """
    overlay = image.copy()
    mask_bool = mask > 127

    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            (image[:, :, c] * (1.0 - alpha) + color[c] * alpha).astype(np.uint8),
            image[:, :, c],
        )

    return Image.fromarray(overlay)


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def predict_batch(
    model: torch.nn.Module,
    img_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
    use_clahe: bool = True,
):
    """Run inference on all JPEG/PNG images in a directory.

    Saves two files per input image:
        {output_dir}/{name}_pred.png    — binary mask
        {output_dir}/{name}_overlay.png  — red overlay on original

    Args:
        model: Trained segmentation model.
        img_dir: Directory containing input images.
        output_dir: Directory to save results.
        device: torch.device to run on.
        use_clahe: Whether to apply CLAHE preprocessing.
    """
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    img_paths = []
    for ext in exts:
        img_paths.extend(sorted(img_dir.glob(ext)))

    if not img_paths:
        print(f"  No images found in {img_dir}")
        return

    print(f"  Processing {len(img_paths)} images from {img_dir}")

    model.eval()
    with torch.no_grad():
        for i, img_path in enumerate(img_paths):
            try:
                binary_mask, original = predict_single(
                    model, img_path, device, use_clahe=use_clahe
                )

                stem = img_path.stem

                # Save binary mask
                mask_img = Image.fromarray(binary_mask)
                mask_img.save(output_dir / f"{stem}_pred.png")

                # Save overlay
                overlay = overlay_mask(original, binary_mask)
                overlay.save(output_dir / f"{stem}_overlay.png")

                if (i + 1) % 10 == 0:
                    print(f"    [{i + 1}/{len(img_paths)}] done")
            except Exception as e:
                print(f"    [{i + 1}/{len(img_paths)}] ERROR: {img_path.name} — {e}")

    print(f"  Done. Results saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crack detection inference")
    parser.add_argument(
        "--model",
        required=True,
        choices=["unet", "deeplabv3plus", "fcn8s"],
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to .pth checkpoint file.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input image file or directory of images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results"),
        help="Output directory for results (default: outputs/results).",
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE preprocessing.",
    )
    args = parser.parse_args()

    # ---- Validate ----
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        raise SystemExit(1)

    if not args.input.exists():
        print(f"ERROR: Input path not found: {args.input}")
        raise SystemExit(1)

    # ---- Setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = create_model(args.model, num_classes=1).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    if "epoch" in ckpt:
        print(f"  Epoch: {ckpt['epoch']}  |  Val IoU: {ckpt.get('val_iou', 'N/A')}")

    use_clahe = not args.no_clahe

    # ---- Infer ----
    if args.input.is_file():
        print(f"Processing single image: {args.input}")
        args.output.mkdir(parents=True, exist_ok=True)

        binary_mask, original = predict_single(
            model, args.input, device, use_clahe=use_clahe
        )

        stem = args.input.stem
        mask_path = args.output / f"{stem}_pred.png"
        overlay_path = args.output / f"{stem}_overlay.png"

        Image.fromarray(binary_mask).save(mask_path)
        overlay_mask(original, binary_mask).save(overlay_path)

        print(f"  Mask saved to:    {mask_path}")
        print(f"  Overlay saved to: {overlay_path}")
    else:
        print(f"Processing directory: {args.input}")
        predict_batch(model, args.input, args.output, device, use_clahe=use_clahe)
