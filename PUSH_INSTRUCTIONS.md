# How to Push to GitHub
## One repo. ~15 minutes.

---

### Step 1 — Create the GitHub repository

1. Go to https://github.com/new
2. Name it: `yolov8m-road-hazard-detection`
3. Set to **Public**
4. **Do NOT** tick "Add a README", ".gitignore", or "license" — we have our own
5. Click **Create repository**

---

### Step 2 — Initialise git locally

Open a terminal inside this folder (the unzipped `final_repo/`):

```bash
git init
git branch -M main
```

---

### Step 3 — Commit everything

```bash
git add .
git status          # sanity check — no .pt files, no images, no datasets

git commit -m "feat: initial release — YOLOv8m road hazard detection ablation study

- 9 architectural experiments: loss functions, P2 head, CBAM, ViT backbones
- Clean config system with environment variable overrides
- SLURM-ready HPC scripts with MLflow tracking
- Comprehensive README with full ablation leaderboard and architecture diagram"
```

---

### Step 4 — Push

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/yolov8m-road-hazard-detection.git
git push -u origin main
```

GitHub will prompt for your username + a **Personal Access Token** (not your password).
Generate one at: https://github.com/settings/tokens → New token → tick `repo` scope.

---

### Step 5 — Upload model weights as a Release (not git push)

Never commit `.pt` files to git. Use GitHub Releases instead:

1. Go to your repo → **Releases** → **Create a new release**
2. Tag: `v1.0.0`
3. Title: `Best weights — YOLOv8m-P2 mAP@50 0.7105`
4. Drag and drop `best_p2_head.pt` into the assets area
5. Click **Publish release**

Anyone can then download it with:
```bash
wget https://github.com/YOUR_USERNAME/yolov8m-road-hazard-detection/releases/download/v1.0.0/best_p2_head.pt
```

---

### Troubleshooting

**"large file" error on push**
```bash
# You accidentally staged a weight or dataset file
git rm --cached path/to/the/big/file
git commit --amend --no-edit
git push
```

**Authentication failed**
Use a Personal Access Token, not your GitHub password.
Generate at: https://github.com/settings/tokens

**Push rejected (non-fast-forward)**
```bash
git pull --rebase origin main
git push
```
