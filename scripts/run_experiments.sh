#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=yolo_h100
#SBATCH --output=logs/exp_%j.log
#SBATCH --error=logs/exp_%j.err
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --mail-user=your@email.com
#SBATCH --mail-type=ALL

# ===========================================
# INITIALIZATION - SIMPLIFIED
# ===========================================

cd $SLURM_SUBMIT_DIR || exit 1
mkdir -p logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# ===========================================
# LOAD PYTHON MODULE - DIRECT PATH
# ===========================================

# The compute nodes and login nodes have different Spack installations.
# Using the actual path that exists on the compute nodes (hash: wikzev7qkzbzsl4fk64dkfzrbqsa6jvp)

PYTHON_PATH="/apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/python-3.10.8-wikzev7qkzbzsl4fk64dkfzrbqsa6jvp/bin"

echo "Loading Python from: $PYTHON_PATH"

if [ -f "$PYTHON_PATH/python3" ]; then
  export PATH="$PYTHON_PATH:$PATH"
  PYTHON_CMD="python3"
  echo "✓ Found python3 at: $PYTHON_PATH/python3"
else
  echo "❌ ERROR: Python not found at expected path: $PYTHON_PATH"
  echo "Listing directory..."
  ls -la $(dirname "$PYTHON_PATH")
  exit 1
fi

# Verify Python
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "✓ Using: $PYTHON_VERSION"

# ===========================================
# GPU CHECK
# ===========================================

echo "=========================================="
echo "GPU Information:"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv
else
  echo "⚠️  nvidia-smi not available"
fi
echo "=========================================="

# ===========================================
# VIRTUAL ENVIRONMENT SETUP
# ===========================================

VENV_DIR="$(pwd)/env"

# Clean up old environment
if [ -d "$VENV_DIR" ]; then
  echo "Removing old virtual environment..."
  rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment with $PYTHON_CMD..."
$PYTHON_CMD -m venv "$VENV_DIR" || {
  echo "❌ Failed to create virtual environment"
  exit 1
}

source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"
echo "✓ Using Python: $(which python)"

# ===========================================
# INSTALL DEPENDENCIES
# ===========================================

echo "Installing dependencies..."

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch for H100 (CUDA 12.1)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ===========================================
# INSTALL YOUR MODIFIED ULTRALYTICS
# ===========================================

echo "=========================================="
echo "Installing your modified ultralytics..."

if [ ! -d "ultralytics" ]; then
  echo "❌ ERROR: ultralytics directory not found!"
  echo "Current directory: $(pwd)"
  ls -la
  exit 1
fi

echo "Found ultralytics at: $(pwd)/ultralytics"

# Install ultralytics (not editable mode to avoid path issues between local and HPC)
cd ultralytics || exit 1
echo "Installing ultralytics..."
pip install . || {
  echo "❌ Failed to install ultralytics"
  cd ..
  exit 1
}
cd ..

echo "✓ Ultralytics installed"

# Verify installation
echo "Verifying YOLO import..."
python -c "from ultralytics import YOLO; print('✓ YOLO import successful')" || {
  echo "❌ YOLO import failed!"
  exit 1
}

# ===========================================
# INSTALL ADDITIONAL DEPENDENCIES
# ===========================================

echo "Installing additional packages..."
pip install opencv-python matplotlib seaborn pandas tqdm
pip install transformers timm 2>/dev/null || echo "⚠️  transformers/timm not installed (optional)"

echo "✓ All dependencies installed"

# ===========================================
# FINAL VERIFICATION
# ===========================================

echo "=========================================="
echo "Final verification:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')

import ultralytics
print(f'Ultralytics: {ultralytics.__version__}')
"
echo "=========================================="

# ===========================================
# RUN EXPERIMENTS
# ===========================================

EXP_TYPE="${1:-test}"

run_experiment() {
  local script=$1
  local name=$2

  echo "──────────────────────────────────────────"
  echo "Starting: $name"
  echo "Time: $(date)"
  echo "──────────────────────────────────────────"

  if [ -f "$script" ]; then
    python "$script"
    local code=$?
    if [ $code -eq 0 ]; then
      echo "✓ Completed: $name"
    else
      echo "❌ Failed: $name (code: $code)"
    fi
    return $code
  else
    echo "❌ Script not found: $script"
    return 1
  fi
}

case $EXP_TYPE in
1) run_experiment "01_baseline_ciou.py" "Baseline" ;;
2) run_experiment "02_wiou_loss.py" "WIoU" ;;
3) run_experiment "03_siou_loss.py" "SIoU" ;;
4) run_experiment "04_copy_paste_aug.py" "Copy-Paste" ;;
5) run_experiment "05_p2_head.py" "P2 Head" ;;
6) run_experiment "06_cbam_attention.py" "CBAM" ;;
7) run_experiment "07_mobilevit_backbone.py" "MobileViT" ;;
8) run_experiment "08_swinv2_backbone.py" "SwinV2" ;;
9) run_experiment "09_combined_best.py" "Combined" ;;

priority)
  echo "Running PRIORITY experiments..."
  run_experiment "01_baseline_ciou.py" "Baseline"
  run_experiment "05_p2_head.py" "P2 Head"
  run_experiment "02_wiou_loss.py" "WIoU"
  run_experiment "04_copy_paste_aug.py" "Copy-Paste"
  ;;

test)
  echo "TEST MODE: Environment ready!"
  echo "To run experiments:"
  echo "  sbatch $0 priority"
  EXIT_CODE=0
  ;;

*)
  echo "Usage: sbatch $0 <option>"
  echo "Options: 1-9, priority, test"
  exit 1
  ;;
esac

EXIT_CODE=$?

deactivate
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
