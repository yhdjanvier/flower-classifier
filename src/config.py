"""
Central configuration: every path and every hyper-parameter lives here.

Why one file?  Training, evaluation and the Streamlit app must agree on the image
size, class list and thresholds.  Keeping them in one place removes "magic numbers"
and prevents the classic bug where training and inference use different settings.

No absolute paths: everything is derived from this file's location, so the same code
runs in PyCharm on Windows and on Streamlit Cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # dataset is extracted here (never committed)
PROCESSED_DIR = DATA_DIR / "processed"     # generated split table (never committed)
SPLITS_CSV = PROCESSED_DIR / "splits.csv"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"   # ignored by Git
MODEL_FILE = MODELS_DIR / "flower_classifier.keras"
CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"
CONFIG_FILE = MODELS_DIR / "config.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVAL_DIR = REPORTS_DIR / "evaluation"

# ----------------------------------------------------------------------------- dataset
KAGGLE_DATASET = "alxmamaev/flowers-recognition"
KAGGLE_URL = "https://www.kaggle.com/datasets/alxmamaev/flowers-recognition"
# Formats TensorFlow's decoder handles reliably (used for training). WEBP is accepted
# only at inference time (Pillow converts it), never in the training set.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MIN_DATASET_IMAGE_SIDE = 32   # drop absurdly small files while cleaning


@dataclass(frozen=True)
class Config:
    """Hyper-parameters. `frozen=True` so nothing changes them by accident mid-run."""

    # --- reproducibility
    seed: int = 42

    # --- input pipeline
    img_height: int = 224          # MobileNetV2 was pre-trained on 224x224 images
    img_width: int = 224
    batch_size: int = 32           # small enough for a laptop CPU, large enough for stable gradients
    val_fraction: float = 0.15     # of ALL images
    test_fraction: float = 0.15    # of ALL images  -> train = 70 %

    # --- augmentation (only "identity-preserving" changes; a flower stays the same flower)
    rotation_factor: float = 0.05      # fraction of a full turn: +/- 0.05 * 360 = +/- 18 degrees
    zoom_factor: float = 0.15
    translation_factor: float = 0.10
    brightness_factor: float = 0.15
    contrast_factor: float = 0.15

    # --- model head
    dropout: float = 0.3

    # --- phase 1: train the new classification head, backbone frozen
    phase1_epochs: int = 15            # upper bound; EarlyStopping usually stops earlier
    phase1_lr: float = 1e-3

    # --- phase 2: fine-tune the upper part of the backbone, very small learning rate
    phase2_epochs: int = 10
    phase2_lr: float = 1e-5
    fine_tune_at: int = 100            # MobileNetV2 has 154 layers: layers < 100 stay frozen

    # --- callbacks
    early_stopping_patience: int = 4
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5

    # --- class imbalance: apply class weights only if largest/smallest class > this ratio
    class_weight_ratio_threshold: float = 1.5

    # --- application behaviour
    confidence_threshold: float = 0.60   # below this the UI says "uncertain"
    top_k: int = 3

    @property
    def img_size(self) -> tuple[int, int]:
        return (self.img_height, self.img_width)
