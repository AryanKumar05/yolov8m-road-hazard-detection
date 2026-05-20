#!/bin/bash
# setup_git_submodule.sh
# Run this ONCE after creating your GitHub repo to wire up the
# ultralytics fork as a proper git submodule.
#
# Prerequisites:
#   1. You have pushed your modified ultralytics fork to GitHub at:
#      https://github.com/YOUR_USERNAME/ultralytics-road-hazard
#   2. You are inside the yolov8m-road-hazard-detection root directory.

set -e

FORK_URL="${1:-https://github.com/YOUR_USERNAME/ultralytics-road-hazard.git}"

echo "=================================================="
echo " Setting up ultralytics as a git submodule"
echo " Fork URL: $FORK_URL"
echo "=================================================="

# Remove the existing ultralytics directory (not yet a submodule)
if [ -d "ultralytics" ] && [ ! -f ".gitmodules" ]; then
  echo "Removing plain ultralytics/ directory..."
  rm -rf ultralytics
fi

# Add as submodule
git submodule add "$FORK_URL" ultralytics
git submodule update --init --recursive

echo ""
echo "✓ Submodule added. Now commit the change:"
echo ""
echo "  git add .gitmodules ultralytics"
echo "  git commit -m 'chore: add ultralytics fork as git submodule'"
echo "  git push"
echo ""
echo "Anyone cloning will get the submodule with:"
echo "  git clone --recurse-submodules <your-repo-url>"
