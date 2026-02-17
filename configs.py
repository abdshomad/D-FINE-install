"""
Global configuration settings for D-FINE training on chicken detection dataset.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Dataset configuration
DATASET_CONFIG = {
    # Dataset paths (relative to project root)
    "train_ann_file": "chicken-detection-labelme-format/coco-format/train/_annotations.coco.json",
    "val_ann_file": "chicken-detection-labelme-format/coco-format/valid/_annotations.coco.json",

    # Image folders (relative to project root)
    "train_img_folder": "chicken-detection-labelme-format/coco-format/train",
    "val_img_folder": "chicken-detection-labelme-format/coco-format/valid",

    # Dataset settings
    "num_classes": 2,  # chicken, not-chicken
    "remap_mscoco_category": False,

    # Class names
    "class_names": ["chicken", "not-chicken"],
}

# D-FINE paths
DFINE_PATH = PROJECT_ROOT / "D-FINE"
DFINE_TRAIN_SCRIPT = DFINE_PATH / "train.py"

# Config file paths (relative to project root) — five variants
CONFIG_PATHS = {
    "n": "configs/dfine_chicken/dfine_hgnetv2_n_chicken.yml",
    "s": "configs/dfine_chicken/dfine_hgnetv2_s_chicken.yml",
    "m": "configs/dfine_chicken/dfine_hgnetv2_m_chicken.yml",
    "l": "configs/dfine_chicken/dfine_hgnetv2_l_chicken.yml",
    "x": "configs/dfine_chicken/dfine_hgnetv2_x_chicken.yml",
}

# Default training settings
DEFAULT_TRAINING_CONFIG = {
    "use_amp": False,  # Set to True to enable Automatic Mixed Precision
    "seed": 0,
    "num_workers": 4,
}

# Output directories (relative to project root) — five variants
OUTPUT_DIRS = {
    "n": "chicken-detection-models/dfine-n",
    "s": "chicken-detection-models/dfine-s",
    "m": "chicken-detection-models/dfine-m",
    "l": "chicken-detection-models/dfine-l",
    "x": "chicken-detection-models/dfine-x",
}

# Ordered list of all variants (smallest to largest) for all-variants script
VARIANTS_ORDER = ["n", "s", "m", "l", "x"]
