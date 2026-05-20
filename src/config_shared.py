"""
config_shared.py — Central Hyperparameter Registry
YOLOv8m Road Hazard Detection Ablation Study
BITS Pilani Hyderabad Campus, 2026

HOW TO USE:
  1. Set DATA_YAML to the absolute path of your data.yaml file.
  2. Set DEVICE to your available GPU indices, e.g. [0] or [0, 1].
  3. All experiment scripts import this file — change once, applies everywhere.
  4. Override any value via environment variable without editing this file:
       export DATA_YAML=/path/to/data.yaml
       export DEVICE=0
"""

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
#  PATHS  —  Edit these before running any experiment
# ═══════════════════════════════════════════════════════════════════

DATA_YAML = os.environ.get("DATA_YAML", "configs/data.yaml")
RUNS_DIR  = Path(os.environ.get("RUNS_DIR", "runs"))

# ═══════════════════════════════════════════════════════════════════
#  HARDWARE
# ═══════════════════════════════════════════════════════════════════

_device_env = os.environ.get("DEVICE", None)
DEVICE  = [int(x) for x in _device_env.split(",")] if _device_env else [0, 1]
WORKERS = int(os.environ.get("WORKERS", 8))

# ═══════════════════════════════════════════════════════════════════
#  TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════

EPOCHS     = int(os.environ.get("EPOCHS",     200))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
IMG_SIZE   = int(os.environ.get("IMG_SIZE",   640))
CACHE      = os.environ.get("CACHE", "true").lower() == "true"
PATIENCE   = 50

OPTIMIZER    = "SGD"
LR0          = 0.01
LRF          = 0.01
MOMENTUM     = 0.937
WEIGHT_DECAY = 0.0005

SAVE_PERIOD = 20
SAVE_BEST   = True

# ═══════════════════════════════════════════════════════════════════
#  AUGMENTATION PROFILES
# ═══════════════════════════════════════════════════════════════════

DEFAULT_AUG = {
    "mosaic":     1.0,
    "mixup":      0.0,
    "copy_paste": 0.0,
    "degrees":    0.0,
    "translate":  0.1,
    "scale":      0.5,
    "fliplr":     0.5,
    "flipud":     0.0,   # Keep off — road geometry requires sky-up
    "hsv_h":      0.015,
    "hsv_s":      0.7,
    "hsv_v":      0.4,
}

COPY_PASTE_AUG = {
    **DEFAULT_AUG,
    "copy_paste": 0.5,
    "mixup":      0.15,
    "scale":      0.9,
}

LOSS_WEIGHTS = {"box": 7.5, "cls": 0.5, "dfl": 1.5}

# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT PRIORITY ORDER
# ═══════════════════════════════════════════════════════════════════

PRIORITY_ORDER = [
    "baseline", "p2_head", "copy_paste",
    "wiou", "combined", "cbam",
    "mobilevit", "swinv2", "siou",
]
