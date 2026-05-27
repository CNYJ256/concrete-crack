"""Standalone training runner. Run with: python run_train.py [--smoke]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from src.train import train_one_model, print_summary
from src.config import MODEL_NAMES

if __name__ == "__main__":
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    smoke = "--smoke" in sys.argv
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Smoke mode: {smoke}")
    print(f"Models: {MODEL_NAMES}")

    results = []
    for name in MODEL_NAMES:
        r = train_one_model(name, smoke=smoke)
        results.append(r)

    print_summary(results)
    print("Done!")
