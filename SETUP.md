# Setup Guide

A step-by-step guide to getting the repository running from scratch.

---

## 1. Prerequisites

| Requirement | Minimum Version | Check command |
|:------------|:---------------:|:-------------|
| Python | 3.10 | `python --version` |
| CUDA | 11.8 (12.1 recommended) | `nvidia-smi` |
| Git | 2.x | `git --version` |
| RAM | 32 GB | — |
| VRAM | 8 GB (16+ recommended) | `nvidia-smi` |

---

## 2. Clone with Submodule

```bash
# Clone the repo AND the modified ultralytics submodule in one step
git clone --recurse-submodules https://github.com/YOUR_USERNAME/yolov8m-road-hazard-detection.git
cd yolov8m-road-hazard-detection
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

---

## 3. Python Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / Mac
# .venv\Scripts\activate           # Windows PowerShell

# Install PyTorch — match your CUDA version:
# CUDA 12.1 (H100, RTX 40xx):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (RTX 30xx, older):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 4. Install Modified Ultralytics

> **Important:** Do NOT `pip install ultralytics`. Use the vendored submodule
> which contains the WIoU, SIoU, and P2-head modifications.

```bash
cd ultralytics
pip install -e .
cd ..

# Verify
python -c "from ultralytics import YOLO; print('OK')"
```

---

## 5. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Configure Your Dataset

```bash
# Edit configs/data.yaml — set the `path` field to your dataset root
nano configs/data.yaml

# Or set via environment variable (no file edit needed):
export DATA_YAML=/absolute/path/to/your/dataset/data.yaml
```

Get the dataset from Roboflow:
1. Contact the authors or use your own road hazard dataset.
2. Export in **YOLOv8 format**.
3. Point `DATA_YAML` at the exported `data.yaml`.

---

## 7. Run Your First Experiment

```bash
# Quickest validation — baseline (640px, CIoU, ~1.7h on 2×H100)
python experiments/01_baseline_ciou.py

# Best single modification — P2 head
python experiments/05_p2_head.py

# Check results
python src/utils/analyse_results.py
```

---

## 8. Single GPU? Use this override

```bash
# Override DEVICE without editing config_shared.py
export DEVICE=0
export BATCH_SIZE=8    # halve batch if single GPU
python experiments/05_p2_head.py
```

---

## HPC / SLURM

See [`scripts/run_experiments.sh`](scripts/run_experiments.sh) and the
**HPC / SLURM Deployment** section in [README.md](README.md).
