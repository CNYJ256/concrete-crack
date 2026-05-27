"""Central configuration for concrete crack detection."""
from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "deepcrack"
TRAIN_IMG = DATA_ROOT / "train_img"
TRAIN_LAB = DATA_ROOT / "train_lab"
TEST_IMG = DATA_ROOT / "test_img"
TEST_LAB = DATA_ROOT / "test_lab"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
RESULT_DIR = OUTPUT_DIR / "results"
FIGURE_DIR = RESULT_DIR / "figures"
PERSONAL_VAL_DIR = ROOT / "personal_val"
LOG_DIR = OUTPUT_DIR / "logs"

# ---- Data ----
IMG_H = 384
IMG_W = 544
IN_CHANNELS = 3
N_CLASSES = 1
CRACK_PIXEL_RATIO = 0.0291  # measured from train set
POS_WEIGHT = (1.0 - CRACK_PIXEL_RATIO) / CRACK_PIXEL_RATIO  # ≈ 33.3

# ---- Training ----
BATCH_SIZE = 8
NUM_EPOCHS = 100
PHASE1_EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42
VAL_SPLIT = 0.2
PIN_MEMORY = True
GRAD_CLIP = 1.0

# ---- Loss ----
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# ---- Normalization ----
# U-Net: simple [0,1] rescaling
UNET_MEAN = (0.0, 0.0, 0.0)
UNET_STD = (1.0, 1.0, 1.0)

# Pretrained models (VGG16, ResNet50): ImageNet stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Per-model normalization lookup
MODEL_NORM = {
    "unet": {"mean": UNET_MEAN, "std": UNET_STD},
    "deeplabv3plus": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "fcn8s": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
}

# ---- Inference ----
CONF_THRESHOLD = 0.5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# ---- Models ----
MODEL_NAMES = ["unet", "deeplabv3plus", "fcn8s"]
