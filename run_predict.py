"""Standalone prediction runner. Run with: python run_predict.py --model unet --checkpoint <path> --input <path>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import torch
from src.predict import predict_single, predict_batch, overlay_mask
from src.models import create_model
from src.config import CKPT_DIR, RESULT_DIR


def main():
    parser = argparse.ArgumentParser(description="Crack detection inference")
    parser.add_argument("--model", required=True, choices=["unet", "deeplabv3plus", "fcn8s"])
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", default=str(RESULT_DIR))
    parser.add_argument("--no-clahe", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = create_model(args.model, num_classes=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        mask, original = predict_single(model, str(input_path), device, use_clahe=not args.no_clahe)
        name = input_path.stem
        overlay = overlay_mask(original, mask)
        overlay.save(output_path / f"{name}_overlay.png")
        from PIL import Image
        Image.fromarray(mask).save(output_path / f"{name}_pred.png")
        print(f"Saved to {output_path}/{name}_overlay.png")
    elif input_path.is_dir():
        predict_batch(model, str(input_path), str(output_path), device, use_clahe=not args.no_clahe)
        print(f"Batch prediction complete: {output_path}")
    else:
        print(f"Input not found: {args.input}")


if __name__ == "__main__":
    main()
