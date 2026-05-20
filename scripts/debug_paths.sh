#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=debug_path
#SBATCH --output=logs/debug_%j.log
#SBATCH --error=logs/debug_%j.err
#SBATCH --time=00:10:00

echo "==========================================="
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "PWD: $(pwd)"
echo "==========================================="

echo ""
echo "--- What architectures are in /apps/spack/opt/spack? ---"
ls -la /apps/spack/opt/spack 2>&1

echo ""
echo "--- Looking for Python in each architecture ---"
for arch in /apps/spack/opt/spack/*; do
    if [ -d "$arch" ]; then
        echo ""
        echo "Architecture: $arch"
        # Look for python directories
        find "$arch" -maxdepth 3 -type d -name "python-3.*" 2>/dev/null | head -5
    fi
done

echo ""
echo "--- Checking /apps/spack-stable (alternative location) ---"
ls -ld /apps/spack-stable 2>&1
if [ -d /apps/spack-stable/opt ]; then
    ls /apps/spack-stable/opt 2>&1
fi

echo ""
echo "--- Current PATH ---"
echo $PATH

echo ""
echo "--- What python3 is being used ---"
which python3 2>&1
python3 --version 2>&1
