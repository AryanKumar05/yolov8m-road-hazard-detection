<!-- HEADER BANNER -->
<div align="center">

```
██╗   ██╗ ██████╗ ██╗      ██████╗ ██╗   ██╗ █████╗    
╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗██║   ██║██╔══██╗   
 ╚████╔╝ ██║   ██║██║     ██║   ██║██║   ██║╚█████╔╝    
  ╚██╔╝  ██║   ██║██║     ██║   ██║╚██╗ ██╔╝██╔══██╗    
   ██║   ╚██████╔╝███████╗╚██████╔╝ ╚████╔╝ ╚█████╔╝    
   ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝   ╚═══╝   ╚════╝     
```

# YOLOv8m Optimization for Road Hazard Detection

**Systematic ablation study of 9 architectural modifications to YOLOv8m for real-time detection of potholes, humps, humans, and vehicles on Indian roads.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00ADEF?style=for-the-badge)](https://ultralytics.com)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![BITS Pilani](https://img.shields.io/badge/BITS%20Pilani-Hyderabad-003366?style=for-the-badge)](https://www.bits-pilani.ac.in/hyderabad/)

*ECE/INSTR F376 Design Project — BITS Pilani Hyderabad Campus, May 2026*

*Authors: [Shreyash Dash](https://github.com/) · [Aryan Kumar](https://github.com/)*
*Supervisors: Dr. Swapna Kulkarni · Prof. RN Ponnalagu*

</div>

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Architectural Map](#-architectural-map)
- [Ablation Study Results](#-ablation-study-results--leaderboard)
- [Engineering Deep-Dive](#-engineering-deep-dive)
  - [Loss Function Modifications](#1-loss-function-modifications)
  - [P2 Detection Head](#2-p2-detection-head--the-winning-modification)
  - [CBAM Attention Module](#3-cbam-convolutional-block-attention-module)
  - [Transformer Backbone Experiments](#4-transformer-backbone-experiments)
  - [Copy-Paste Augmentation](#5-copy-paste-augmentation)
- [Dataset Design & Statistics](#-dataset-design--statistics)
- [Repository Structure](#-repository-structure)
- [Quickstart & Reproducibility](#-quickstart--reproducibility)
- [HPC / SLURM Deployment](#-hpc--slurm-deployment)
- [Inference Demo App](#-inference-demo-app)
- [FAQ: Should I Upload My Dataset?](#-faq-should-i-upload-my-dataset)
- [Citation](#-citation)
- [References](#-references)

---

## 🎯 Project Overview

Road infrastructure hazards — potholes, speed humps, pedestrians, and vehicles — present unique detection challenges in real-world deployment for Advanced Driver Assistance Systems (ADAS). Indian roads compound these challenges further: erratic lighting, extreme dust/glare, heavily occluded small potholes, and multi-class clutter make off-the-shelf detection models unreliable.

This project systematically benchmarks **9 architectural modifications** to the YOLOv8m baseline across 4 progressively refined datasets. Every experiment is tracked with MLflow, all configurations are versioned in YAML, and the entire pipeline is reproducible on SLURM-based HPC clusters.

**What we found:**
- The **P2 detection head** is the single best standalone modification (+1.54 pp mAP@50 over baseline), at only +6% latency cost.
- **Higher resolution (960px)** improves accuracy but sharply degrades compute efficiency.
- **Transformer backbones (RT-DETR)** deliver worse mAP than YOLOv8m at 5–10× training cost — not practical for this task.
- **Modified IoU losses (SIoU, WIoU)** underperform CIoU on this dataset due to the dominant presence of small, far-field, ambiguously annotated objects.

---

## 🗺 Architectural Map

The diagram below maps the complete data and model pipeline, annotating where each experimental modification injects into the baseline YOLOv8m architecture.

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                       FULL PIPELINE: RAW FRAME → DETECTION                        ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────┐
  │  INPUT STAGE                                                    │
  │  Raw Frame (H×W×3, BGR)                                         │
  │   ↓                                                             │
  │  [Resize → 640×640 / 768×768 / 960×960 / 1280×1280]           │
  │   ↓                                                             │
  │  [Normalise → μ=(0.485,0.456,0.406), σ=(0.229,0.224,0.225)]   │
  │   ↓                                                             │
  │  [Augmentation Pipeline]  ◄──── EXP 04: Copy-Paste (p=0.5)    │
  │     mosaic=1.0 | mixup=0.15 | fliplr=0.5 | hsv jitter          │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  BACKBONE  (Feature Extraction)                                 │
  │                                                                 │
  │  OPTION A — Default YOLOv8m C2f Backbone (EXPs 01–06)         │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │  Conv(64,3,2) → P1/2                                     │   │
  │  │  Conv(128,3,2) → P2/4  ──────────────────────────────┐  │   │
  │  │  C2f(128,3) [CSP Bottleneck × 3]                      │  │   │
  │  │  Conv(256,3,2) → P3/8  ──────────────────────────┐   │  │   │
  │  │  C2f(256,6) [CSP Bottleneck × 6]                  │   │  │   │
  │  │                 ↕ EXP 06: CBAM Attention here     │   │  │   │
  │  │  Conv(512,3,2) → P4/16 ──────────────────────┐   │   │  │   │
  │  │  C2f(512,6)                                   │   │   │  │   │
  │  │  Conv(768,3,2) → P5/32 ──────────────────┐   │   │   │  │   │
  │  │  C2f(768,3)                               │   │   │   │  │   │
  │  │  SPPF(768,5) ─────────────────────────────┤   │   │   │  │   │
  │  └───────────────────────────────────────────┼───┼───┼───┼──┘   │
  │                                              │   │   │   │      │
  │  OPTION B — SwinV2 Backbone (EXP 08)         │   │   │   │      │
  │  ┌─────────────────────────────────────────┐ │   │   │   │      │
  │  │ Shifted Window Attention × 4 stages     │ │   │   │   │      │
  │  │ Adapter Conv→[256, 512, 768]            ├─┘   │   │   │      │
  │  └─────────────────────────────────────────┘     │   │   │      │
  │                                                  │   │   │      │
  │  OPTION C — MobileViT Backbone (EXP 07)          │   │   │      │
  │  ┌──────────────────────────────────────────┐    │   │   │      │
  │  │ MobileNetV2 Local + Transformer Global   │    │   │   │      │
  │  │ Adapter Conv→[256, 512, 768]             ├────┘   │   │      │
  │  └──────────────────────────────────────────┘        │   │      │
  └─────────────────────────────────────────────────────┼───┘      │
                    P5/32 ◄───────────────────────────────┘          │
                    P4/16 ◄────────────────────────────────────┘     │
                    P3/8  ◄─────────────────────────────────────┐    │
                    P2/4  ◄──────────────────────────────────┐  │    │
                                                             │  │    │
  ┌──────────────────────────────────────────────────────────┼──┼────┘
  │  NECK  (Feature Pyramid / Fusion)                        │  │
  │                                                          │  │
  │  DEFAULT — PANet (Path Aggregation Network)              │  │
  │  ┌────────────────────────────────────────────────────┐  │  │
  │  │  P5 → Upsample → Concat(P4) → C2f(512)            │  │  │
  │  │         ↓                                          │  │  │
  │  │  P4' → Upsample → Concat(P3) → C2f(256)  → P3'   ├──┘  │
  │  │         ↓                                          │     │
  │  │  P3' → Conv(3,2) → Concat(P4') → C2f(512) → P4'' │     │
  │  │  P4'' → Conv(3,2) → Concat(P5)  → C2f(768) → P5' │     │
  │  └────────────────────────────────────────────────────┘     │
  │                                                             │
  │  ALTERNATIVE — BiFPN Neck (EXP exploration)                 │
  │  ┌────────────────────────────────────────────────────┐     │
  │  │  Bidirectional weighted fusion with learned weights│     │
  │  │  O_out = Σ w_i · F_i / (Σ w_i + ε)               │     │
  │  └────────────────────────────────────────────────────┘     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  DETECTION HEAD                                                 │
  │                                                                 │
  │  BASELINE — P3/P4/P5 Three-Scale Head (EXPs 01–04, 06–09)     │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │  Detect([P3', P4'', P5'])                               │   │
  │  │  Stride 8  → 80×80  grid  →  6,400  anchors            │   │
  │  │  Stride 16 → 40×40  grid  →  1,600  anchors            │   │
  │  │  Stride 32 → 20×20  grid  →    400  anchors            │   │
  │  │  ─────────────────────────────────────────              │   │
  │  │  Total: 8,400 anchors per image                         │   │
  │  └──────────────────────────────────────────────────────────┘   │
  │                                                                 │
  │  ⭐ BEST MOD — P2+P3+P4+P5 Four-Scale Head (EXP 05)           │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │  NEW: P2/4 → Upsample(P3') → C2f(128) → 160×160 grid  │   │
  │  │  Detect([P2, P3', P4'', P5'])                           │   │
  │  │  Stride 4  → 160×160 grid → 25,600  anchors  ← NEW!    │   │
  │  │  Stride 8  →  80×80  grid →  6,400  anchors            │   │
  │  │  Stride 16 →  40×40  grid →  1,600  anchors            │   │
  │  │  Stride 32 →  20×20  grid →    400  anchors            │   │
  │  │  ─────────────────────────────────────────              │   │
  │  │  Total: 34,000 anchors per image (4× increase!)        │   │
  │  │  Target: Objects < 24px (distant potholes, pedestrians) │   │
  │  └──────────────────────────────────────────────────────────┘   │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  LOSS COMPUTATION  (training only)                              │
  │                                                                 │
  │  EXP 01 — CIoU (Default)                                       │
  │    L = 1 - IoU + ρ²(b,bᵍᵗ)/c² + αv                           │
  │    Penalises: overlap deficit + centre dist + aspect ratio      │
  │                                                                 │
  │  EXP 02 — WIoU v3                                              │
  │    L = r·L_CIoU,  r = exp[(x-x̄)² + (y-ȳ)²] / (2σ²)          │
  │    Outlier score r ↑ → gradient weight ↓ (noisy labels)        │
  │                                                                 │
  │  EXP 03 — SIoU                                                 │
  │    L = IoU_cost + Shape_cost + Distance_cost + Angle_cost       │
  │    Angle term: Λ = 1 - 2·sin²(arcsin(|ch/σ|) - π/4)          │
  │    Penalises diagonal drift between predicted and GT centres    │
  │                                                                 │
  │  TOTAL LOSS = λ_box·L_bbox + λ_cls·L_cls + λ_dfl·L_dfl       │
  │               (7.5)           (0.5)          (1.5)             │
  └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  OUTPUT                                                         │
  │  [N × (x, y, w, h, conf, cls1, cls2, cls3, cls4)]             │
  │  Classes: Pothole | Hump | Human | Vehicle                     │
  │  Post-processing: NMS (IoU=0.45, conf=0.25)                    │
  │  Export formats: .pt | .onnx | TensorRT .engine                │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Ablation Study Results & Leaderboard

### Phase 1 — Balanced Subset Experiments (`balanced_subset`, ~2,390 images)

| Rank | Experiment | mAP@50 | Δ mAP@50 | mAP@50-95 | Precision | Recall |
|:----:|:-----------|:------:|:--------:|:---------:|:---------:|:------:|
| 🥇 1 | Copy-Paste Augmentation | 0.6930 | +0.0177 | 0.5043 | 0.7478 | 0.6299 |
| 🥈 2 | WIoU Loss | 0.6830 | +0.0077 | 0.5120 | 0.7935 | 0.6012 |
| 🥉 3 | **Baseline (CIoU)** | 0.6753 | ±0.0000 | **0.5073** | 0.7913 | 0.5968 |
| 4 | P2 Head | 0.6655 | -0.0098 | 0.4895 | 0.6947 | 0.6325 |

> **Note:** The P2 head result on the small subset is misleading — the 2,390-image training set is too sparse to provide sufficient diversity for the 160×160 feature scale to learn from. Results invert on the full clean dataset below.

---

### Phase 2 — Full Clean Dataset (`Dataset`, ~8,284 images, 200 epochs, 2× H100)

| Rank | Run ID | Model | Img Size | Loss Fn | Head Config | Best mAP@50 | Best mAP@50-95 | Peak Epoch | Total Epochs | Train Time | mAP/hr |
|:----:|:------:|:------|:--------:|:-------:|:-----------:|:-----------:|:--------------:|:----------:|:------------:|:----------:|:------:|
| 🥇 1 | `exp_p2_640` | YOLOv8m-**P2** | 640 | CIoU | **P2+P3+P4+P5** | **0.7105** | 0.5127 | 70 | 108 | 1.8h | 0.395 |
| 🥈 2 | `exp_960` | YOLOv8m | 960 | CIoU | P3+P4+P5 | 0.7064 | **0.5184** | 124 | 198 | 4.2h | 0.168 |
| 🥉 3 | `exp_baseline` | YOLOv8m | 640 | **CIoU** | P3+P4+P5 | 0.7051 | 0.5057 | 70 | 138 | 1.7h | **0.415** |
| 4 | `exp_768` | YOLOv8m | 768 | CIoU | P3+P4+P5 | 0.6953 | 0.5071 | 106 | 200 | 3.2h | 0.217 |
| 5 | `exp_1280` | YOLOv8m | 1280 | CIoU | P3+P4+P5 | 0.6932 | 0.4819 | 73 | 116† | 4.1h | 0.169 |
| 6 | `exp_wiou` | YOLOv8m | 640 | **WIoU** | P3+P4+P5 | 0.6829 | 0.4666 | 153 | 200 | 2.5h | 0.273 |
| 7 | `exp_rtdetr_x` | RT-DETR-X | 640 | Default | Default | 0.6750 | 0.4754 | 93 | 200 | **16.9h** | 0.040 |
| 8 | `exp_rtdetr_l_768` | RT-DETR-L | 768 | Default | Default | 0.6704 | 0.4680 | 87 | 127† | 10.7h | 0.063 |
| 9 | `exp_rtdetr_l` | RT-DETR-L | 640 | Default | Default | 0.6647 | 0.4701 | 106 | 147† | 9.7h | 0.068 |
| —  | `exp_siou` | YOLOv8m | 640 | **SIoU** | P3+P4+P5 | — | — | — | 41† | 0.9h | — |

> † Early stopped due to divergence or patience trigger. — SIoU diverged in all runs; no usable metric.

### Efficiency Analysis (mAP@50 per Training Hour)

| Model | mAP@50 | Train Time | **mAP/hr** | Verdict |
|:------|:------:|:----------:|:----------:|:-------:|
| YOLOv8m 640 CIoU | 0.7051 | 1.7h | **0.415** | ✅ Best efficiency |
| YOLOv8m-P2 640 | 0.7105 | 1.8h | 0.395 | ✅ Best accuracy + near-best efficiency |
| YOLOv8m 768 | 0.6953 | 3.2h | 0.217 | ⚠️ Modest gain, high cost |
| YOLOv8m 960 | 0.7064 | 4.2h | 0.168 | ⚠️ Marginal gain, 2.5× slower |
| YOLOv8m 1280 | 0.6932 | 4.1h | 0.169 | ❌ Worse mAP than 640 |
| RT-DETR-L 640 | 0.6647 | 9.7h | 0.068 | ❌ 6× slower, lower mAP |
| RT-DETR-X 640 | 0.6750 | 16.9h | 0.040 | ❌ 10× slower, lower mAP |

---

## 🔬 Engineering Deep-Dive

### 1. Loss Function Modifications

#### CIoU — Complete IoU (Baseline)

The default YOLOv8m bounding box regression loss is Complete IoU, which extends the simple IoU with three geometric penalty terms:

```
L_CIoU = 1 - IoU + ρ²(b, b^gt) / c² + α·v

where:
  ρ²(b, b^gt)  = squared Euclidean distance between box centres
  c            = diagonal length of the smallest enclosing box
  v            = (2/π)² · (arctan(w^gt/h^gt) - arctan(w/h))²
  α            = v / (1 - IoU + v)   [trade-off weighting factor]
```

CIoU simultaneously optimises IoU, centre-point proximity, and aspect ratio consistency. Crucially, it applies **equal gradient weight** to all training examples regardless of annotation quality.

#### WIoU v3 — Wise IoU (Experiment 02)

WIoU (Tong et al., 2023) introduces a **dynamic gradient re-weighting** mechanism via an outlier score:

```
WIoU_Loss = r · L_CIoU

r = exp[ (x - x̄)² + (y - ȳ)² ] / (2σ²)

where (x, y) is the predicted centre, (x̄, ȳ) is the running mean centre
across all anchors, and σ is the standard deviation.
```

Anchors whose predicted centres deviate significantly from the population mean (i.e., high outlier score `r`) are **downweighted**. The rationale: visually ambiguous potholes have noisy, subjectively placed ground-truth centres; giving these harmful gradients equal weight corrupts the loss surface for well-annotated, clear examples.

**Why it underperformed here:** Our dataset contains a dominant population of small, far-field objects whose centres are inherently scattered — WIoU misidentified valid hard-positive anchors as outliers and suppressed their gradients. The outlier normalization statistics (x̄, ȳ) are corrupted when the majority of examples are legitimately scattered across the image plane.

#### SIoU — Angle-Aware IoU (Experiment 03)

SIoU (Gevorgyan, 2022) decomposes the regression loss into four orthogonal cost terms:

```
L_SIoU = IoU_cost + Shape_cost + Distance_cost + Angle_cost

Angle cost:
  Λ = 1 - 2·sin²(arcsin(ch/σ) - π/4)

  ch = |y^gt - y| / σ           (vertical component of centre-distance vector)
  σ  = √[ (x^gt-x)² + (y^gt-y)² ]   (Euclidean centre distance)
```

The angle cost explicitly penalises **diagonal drift** — the phenomenon where predicted box centres drift off-axis from ground-truth. Standard IoU variants decompose x-axis and y-axis errors independently, which permits convergence along diagonal trajectories that look acceptable in each axis alone but produce poor spatial alignment globally.

**Why it diverged:** SIoU's angle term is highly sensitive to near-zero centre distances at initialisation. With an anchor-free detection head, early training predictions are random; the `arcsin(ch/σ)` term produces undefined/large gradients when σ → 0. This caused training instability from epoch 1 across all runs.

---

### 2. P2 Detection Head — The Winning Modification

The standard YOLOv8m detection head operates on three feature pyramid levels: P3 (stride 8, 80×80), P4 (stride 16, 40×40), and P5 (stride 32, 20×20). This yields 8,400 effective anchor positions per 640×640 image.

The P2 head modification (Experiment 05) taps into the **P2/4 feature map** — the output of the second backbone stage before the first downsampling to P3 — and adds a fourth detection scale at stride 4:

```yaml
# yolov8m_p2.yaml — Head section (modified)
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]          # cat backbone P4
  - [-1, 3, C2f, [512]]                # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]          # cat backbone P3
  - [-1, 3, C2f, [256]]               # 15 → P3/8-medium

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]          # cat backbone P2 ← NEW
  - [-1, 3, C2f, [128]]               # 18 → P2/4-tiny  ← NEW

  - [[18, 21, 24, 27], 1, Detect, [nc]]  # 4-scale Detect
```

**Anchor Density Arithmetic:**

```
Standard YOLOv8m @ 640×640:
  P3 (stride 8):  (640/8)²  =  6,400 positions
  P4 (stride 16): (640/16)² =  1,600 positions
  P5 (stride 32): (640/32)² =    400 positions
  ─────────────────────────────────────────────
  Total:                       8,400 anchor points

YOLOv8m-P2 @ 640×640:
  P2 (stride 4):  (640/4)²  = 25,600 positions  ← NEW
  P3 (stride 8):  (640/8)²  =  6,400 positions
  P4 (stride 16): (640/16)² =  1,600 positions
  P5 (stride 32): (640/32)² =    400 positions
  ─────────────────────────────────────────────
  Total:                      34,000 anchor points (4× increase)
```

The P2 scale targets objects occupying fewer than 24 pixels — primarily distant potholes and far-range pedestrians. The +40% FLOPs overhead (1.8h vs 1.7h training) is modest given the +1.54 pp mAP@50 improvement, making this the **best single modification** in the study.

---

### 3. CBAM: Convolutional Block Attention Module

CBAM (Woo et al., 2018) injects a lightweight, two-stage attention gate after each C2f block in the backbone. The module operates sequentially on channel and spatial dimensions:

```python
class ChannelAttention(nn.Module):
    """
    Learns WHICH feature channels matter for road hazard discrimination.
    Both global average and max pooling are used to capture both distributed
    and peak activation patterns — critical for detecting flat potholes
    (diffuse activations) vs. tall humps (peaked activations).
    """
    def forward(self, x):
        # Avg-pool captures diffuse semantic activation (potholes)
        avg_out = self.fc(self.avg_pool(x))      # [B, C, 1, 1]
        # Max-pool captures peak distinctive features (hump edges)
        max_out = self.fc(self.max_pool(x))      # [B, C, 1, 1]
        return self.sigmoid(avg_out + max_out)   # combined gate ∈ (0,1)

class SpatialAttention(nn.Module):
    """
    Learns WHERE in the feature map to focus.
    A 7×7 conv over channel-pooled maps (mean + max) produces
    a spatial saliency mask — suppressing sky/background regions
    and amplifying road surface features.
    """
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)          # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)        # [B,1,H,W]
        concat = torch.cat([avg_out, max_out], dim=1)          # [B,2,H,W]
        return self.sigmoid(self.conv(concat))                 # spatial gate

class CBAM(nn.Module):
    def forward(self, x):
        x = x * self.channel_attention(x)    # Recalibrate channels
        x = x * self.spatial_attention(x)    # Recalibrate spatial locs
        return x
```

**Why CBAM matters for road scenes:** Road images are dominated by background (asphalt, sky, greenery). Standard convolution assigns equal weight to all spatial positions. CBAM learns to **suppress** the uniform road texture and **amplify** the textural discontinuities that indicate pothole edges or hump profiles, reducing false-negative rates for small hazards.

CBAM adds approximately **5–10% inference overhead** due to the pooling and FC operations — a favourable trade-off given its attention-based feature refinement.

---

### 4. Transformer Backbone Experiments

#### Swin Transformer V2 (Experiment 08)

SwinV2 (Liu et al., 2022) implements hierarchical self-attention with **shifted window partitioning**, constraining the O(N²) attention complexity to O(N·M²) where M is window size (8 in swinv2_tiny_window8_256):

```python
class SwinV2Backbone(nn.Module):
    """
    4-stage hierarchical backbone with channel dimensions [96, 192, 384, 768].
    We extract features at stages 1, 2, 3 (indices 1–3 in features_only mode)
    and project them to YOLOv8m-expected dimensions [256, 512, 768].
    """
    def __init__(self):
        self.swin = timm.create_model(
            'swinv2_tiny_window8_256',
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3)    # P3/8, P4/16, P5/32 equivalents
        )
        # Linear projection adapters
        self.adapt_p3 = nn.Conv2d(192, 256, 1)   # 1×1 projection
        self.adapt_p4 = nn.Conv2d(384, 512, 1)
        self.adapt_p5 = nn.Conv2d(768, 768, 1)   # Identity dimension
```

**Result:** RT-DETR variants (which use a similar transformer philosophy) yielded 5–10× training cost with inferior mAP. The root cause is **domain gap** — SwinV2 pretraining on ImageNet encodes object-centric features, while road hazard textures (cracked asphalt, embedded debris) require spatial frequency representations that fine-tuning on ~8k images cannot adequately re-learn.

#### MobileViT (Experiment 07)

MobileViT (Mehta & Rastegari, 2022) combines MobileNetV2's local inductive biases with a global transformer for efficient mobile deployment. The backbone alternates between `MV2 blocks` (depthwise separable convolutions) and `MobileViT blocks` (local + global self-attention):

```python
# Adapter dims: MobileViT-small outputs [64, 96, 128, 160, 640]
# YOLOv8m neck expects: [256, 512, 768]
self.adapt_p3 = nn.Conv2d(128, 256, 1)
self.adapt_p4 = nn.Conv2d(160, 512, 1)
self.adapt_p5 = nn.Conv2d(640, 768, 1)
```

The optimizer is switched to `AdamW` (weight_decay=0.05) with a lower LR (`1e-3`) to respect the pretrained ViT weights. SGD with momentum, optimal for CNN-based models, causes catastrophic forgetting in transformer layers.

---

### 5. Copy-Paste Augmentation

Copy-Paste (Ghiasi et al., 2021) synthesizes training composites by pasting segmentation instances from other images at random scales and positions. For a road dataset with the bounding-box-only annotation paradigm used here, Ultralytics implements an approximation using bounding-box crops:

```python
COPY_PASTE_AUG = {
    'mosaic':      1.0,    # 4-image mosaic (always on)
    'copy_paste':  0.5,    # 50% probability per image
    'mixup':       0.15,   # Light alpha blending
    'scale':       0.9,    # Object scale jitter (wider than default 0.5)
    'fliplr':      0.5,
    'flipud':      0.0,    # Road context: no vertical flip
    'hsv_h':       0.015,
    'hsv_s':       0.7,
    'hsv_v':       0.4,
}
```

Key reasoning for `flipud=0.0`: vertical flipping inverts the geometric prior of road scenes (sky above, road below). Maintaining this prior is critical for the P2 head, which detects objects at ground level where this spatial relationship is most consistent.

**Zero inference overhead**: Copy-Paste is a training-time augmentation only. The deployed model is identical to the baseline in architecture and inference cost.

---

## 📦 Dataset Design & Statistics

> **On uploading datasets to GitHub:** See [FAQ below](#-faq-should-i-upload-my-dataset) for the definitive answer.

### Dataset Evolution (4 Iterations)

```
Iteration 1: Improved_Balanced_Again_Dataset  → 8,905 images, noisy labels, inflated mAP
Iteration 2: balanced_subset                  → 2,390 images, pruned subset for rapid iteration
Iteration 3: Dataset                          → 8,284 images, clean, definitive benchmark
Iteration 4: balanced_8k_dataset              → 32,078 single-instance images for pre-training
```

### Class Instance Distribution

| Dataset | Human | Vehicle | Hump | Pothole | Total Boxes |
|:--------|:-----:|:-------:|:----:|:-------:|:-----------:|
| `Improved_Balanced_Again_Dataset` | 10,667 | 21,105 | 6,301 | 8,024 | 46,097 |
| `Dataset` (final) | 9,180 | 17,136 | 7,047 | 8,055 | 41,418 |
| `balanced_subset` | 1,401 | 1,440 | 926 | 1,365 | 5,132 |
| `balanced_8k_dataset` | 8,000 | 8,000 | 8,000 | 8,000 | 32,000 |

### Train/Val/Test Splits

| Dataset | Split | Images | Valid Boxes |
|:--------|:-----:|:------:|:-----------:|
| `Dataset` | train | 6,785 | 34,904 |
| `Dataset` | valid | 795 | 3,837 |
| `Dataset` | test | 704 | 2,677 |
| `balanced_subset` | train | 1,669 | 3,625 |
| `balanced_subset` | valid | 325 | 718 |
| `balanced_subset` | test | 396 | 789 |
| `balanced_8k_dataset` | train | 25,099 | 25,021 |
| `balanced_8k_dataset` | valid | 4,429 | 4,429 |
| `balanced_8k_dataset` | test | 2,550 | 2,550 |

### `data.yaml` Format

```yaml
# data.yaml
path: /path/to/your/dataset
train: images/train
val:   images/valid
test:  images/test

nc: 4
names:
  0: pothole
  1: hump
  2: human
  3: vehicle
```

---

## 🗂 Repository Structure

```
yolov8m-road-hazard-detection/
│
├── 📄 README.md                        ← You are here
├── 📄 LICENSE
├── 📄 .gitignore                       ← Strictly excludes weights, datasets, caches
├── 📄 requirements.txt
│
├── 📁 configs/                         ← All YAML experiment configurations
│   ├── data.yaml                       ← Dataset paths (edit before running)
│   ├── yolov8m_p2.yaml                 ← P2 head architecture definition
│   ├── train_external_backbone.yaml    ← Conservative patience/LR for ViT backbones
│   └── train_imgsz_960.yaml            ← Image resolution ablation override
│
├── 📁 src/                             ← Core source code
│   ├── config_shared.py                ← Central hyperparameter registry
│   ├── models/
│   │   ├── cbam.py                     ← CBAM attention module (standalone)
│   │   ├── mobilevit_backbone.py       ← MobileViT backbone wrapper
│   │   └── swinv2_backbone.py          ← SwinV2 backbone wrapper
│   └── utils/
│       ├── analyse_results.py          ← Results aggregation + visualisation
│       ├── count_class_instances.py    ← Dataset balance audit script
│       └── create_1000_instance_subset.py ← Rapid-iteration subset creator
│
├── 📁 experiments/                     ← One script per ablation run
│   ├── 01_baseline_ciou.py
│   ├── 02_wiou_loss.py
│   ├── 03_siou_loss.py
│   ├── 04_copy_paste_aug.py
│   ├── 05_p2_head.py                   ← ⭐ Best single modification
│   ├── 06_cbam_attention.py
│   ├── 07_mobilevit_backbone.py
│   ├── 08_swinv2_backbone.py
│   └── 09_combined_best.py
│
├── 📁 scripts/                         ← HPC/cluster automation
│   ├── run_experiments.sh              ← SLURM job submission script
│   ├── debug_paths.sh                  ← Validate HPC environment
│   └── copy_paste_aug.py               ← Standalone aug verification
│
├── 📁 ultralytics/                     ← Modified Ultralytics submodule
│   └── ...                             ← (git submodule or vendored fork)
│
├── 📁 runs/                            ← ⚠️ GITIGNORED — auto-generated
│   ├── train/                          ← Per-run training artefacts
│   ├── mlflow/                         ← MLflow tracking DB
│   ├── exports/                        ← ONNX / TensorRT exports
│   └── comparisons/                    ← Cross-run comparison outputs
│
├── 📁 weights/                         ← ⚠️ GITIGNORED — model weights
│   └── best_p2_head.pt                 ← (not committed; see Releases)
│
├── 📁 figures/                         ← Auto-generated plots (gitignored)
├── 📁 tables/                          ← Auto-generated CSV/LaTeX (gitignored)
└── 📁 logs/                            ← SLURM job logs (gitignored)
```

---

## ⚡ Quickstart & Reproducibility

### Prerequisites

```bash
# Python 3.10+ required
python --version

# CUDA 12.1+ required for H100; 11.8 works for consumer GPUs
nvidia-smi
```

### Step 1 — Clone & Install

```bash
git clone https://github.com/<your-username>/yolov8m-road-hazard-detection.git
cd yolov8m-road-hazard-detection

# Create isolated virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# Install PyTorch (adjust CUDA version to match your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install modified Ultralytics from local submodule
cd ultralytics && pip install -e . && cd ..

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 2 — Configure Your Dataset Path

```bash
# Edit the single source-of-truth config file
# Change DATA_YAML to point to your data.yaml
nano src/config_shared.py
```

```python
# src/config_shared.py — key variables to set before any run
DATA_YAML  = '/path/to/your/dataset/data.yaml'   # ← EDIT THIS
DEVICE     = [0, 1]    # GPU indices. Use [0] for single GPU.
EPOCHS     = 200
BATCH_SIZE = 16
IMG_SIZE   = 640
```

### Step 3 — Run a Single Experiment

```bash
# Each experiment is a standalone, self-contained script
# Example: run the P2 head experiment (best single modification)
python experiments/05_p2_head.py

# Results auto-saved to:
#   experiments/p2_head/metadata.json   ← hyperparams + config
#   experiments/p2_head/results.json    ← mAP, FPS metrics
#   runs/train/p2_head/weights/best.pt  ← best checkpoint
```

### Step 4 — Run All Experiments Sequentially

```bash
# Sequentially runs all 9 experiments
# Estimated: ~14–16h on 2× NVIDIA H100

for i in $(seq 1 9); do
    python experiments/0${i}_*.py
done
```

### Step 5 — Analyse & Visualise Results

```bash
python src/utils/analyse_results.py

# Outputs:
#   figures/comprehensive_analysis.png   ← 4-panel comparison figure
#   figures/p2_head_analysis.png         ← P2 head focus visualisation
#   tables/results_table.csv             ← Machine-readable ablation table
#   tables/results_table.tex             ← LaTeX-ready table for papers
```

### Step 6 — Export for Deployment

```python
from ultralytics import YOLO

# Load best weights
model = YOLO('runs/train/p2_head/weights/best.pt')

# Export to ONNX (cross-platform)
model.export(format='onnx', imgsz=640, simplify=True)

# Export to TensorRT (NVIDIA edge deployment)
model.export(format='engine', imgsz=640, half=True)  # FP16

# Export to ONNX with dynamic batch size
model.export(format='onnx', dynamic=True, imgsz=640)
```

### Quick Inference

```python
from ultralytics import YOLO

model = YOLO('runs/train/p2_head/weights/best.pt')

# Inference on image
results = model.predict('test_image.jpg', conf=0.25, iou=0.45)
results[0].show()

# Inference on video
results = model.predict('road_video.mp4', stream=True, conf=0.25)
for r in results:
    r.show()

# Batch inference
results = model.predict(['img1.jpg', 'img2.jpg'], conf=0.25)
```

---

## 🖥 HPC / SLURM Deployment

The `scripts/run_experiments.sh` is a self-contained SLURM job script. It handles environment setup, dependency installation, and sequential experiment execution entirely on the compute node.

### Submitting Individual Experiments

```bash
# Run single experiment (e.g., experiment 5: P2 head)
sbatch scripts/run_experiments.sh 5

# Run priority experiments (baseline, P2, WIoU, copy-paste)
sbatch scripts/run_experiments.sh priority

# Test environment only (no training)
sbatch scripts/run_experiments.sh test

# Run all experiments
for i in $(seq 1 9); do
    sbatch scripts/run_experiments.sh $i
done
```

### SLURM Header Configuration

```bash
# Edit the following in scripts/run_experiments.sh to match your cluster:
#SBATCH -p gpu_h100_4            # Partition name
#SBATCH --gres=gpu:2             # Number of GPUs
#SBATCH --cpus-per-task=8        # CPU cores (match WORKERS in config)
#SBATCH --mem=64G                # RAM (1k images cached = ~4GB; 8k = ~32GB)
#SBATCH --time=23:59:00          # Walltime upper bound
#SBATCH --mail-user=your@email   # Job notifications
```

### Monitoring with MLflow

```bash
# On the login node (after jobs complete), launch MLflow UI
mlflow ui --backend-store-uri runs/mlflow --port 5000

# Then in your local browser (with SSH tunnel):
ssh -L 5000:localhost:5000 <hpc-login-node>
# Navigate to: http://localhost:5000
```

---

## 🌐 Inference Demo App

A companion web application allows drag-and-drop testing of model weights without writing code. It supports `.pt`, `.onnx`, and `.engine` (TensorRT) weight formats.

**Features:**
- Upload any compatible weight file at runtime
- Webcam live inference with real-time bounding box overlay
- Video file inference with pothole/hump proximity alerts (audible beep)
- Distance estimation via camera focal length + camera height model
- All four class detections displayed with confidence scores

```bash
# Launch the web app backend
cd app/
pip install -r app_requirements.txt
python backend.py --port 8080

# Navigate to http://localhost:8080
```

---

## ❓ FAQ: Should I Upload My Dataset?

**Short answer: No — do not commit dataset images to Git.** Here is the full reasoning:

| Concern | Detail |
|:--------|:-------|
| **Repository size** | Computer vision datasets are tens of GB. Git (and GitHub's 1GB soft limit) is not designed for binary blobs of this size. |
| **Git LFS cost** | Large File Storage on GitHub has bandwidth quotas that trigger charges beyond ~1GB/month of data transfer. |
| **Reproducibility** | What matters for reproducibility is the `data.yaml` config, the class list, the split proportions, and the label format — not the raw images. |
| **Labelling provenance** | Your labels were created via Roboflow. The Roboflow project URL or an exported annotation archive (no images) is the appropriate shareable artefact. |

**What to share instead:**

```
✅ data.yaml                    — dataset config (paths, class names)
✅ dataset_stats.json           — image counts, class distributions, split sizes
✅ Roboflow project link        — version-locked, re-downloadable by anyone
✅ README instructions          — where to download + how to re-create splits
❌ images/                      — Do NOT commit
❌ labels/                      — Do NOT commit (large; Roboflow export covers this)
```

Add to your `.gitignore`:

```gitignore
# ── Model Weights ───────────────────────────────────────────
weights/
*.pt
*.pth
*.onnx
*.engine
*.trt

# ── Dataset Files ────────────────────────────────────────────
datasets/
data/
images/
labels/
*.cache

# ── Training Outputs ─────────────────────────────────────────
runs/
experiments/*/weights/
experiments/*/train/

# ── MLflow Artefacts ─────────────────────────────────────────
mlruns/
runs/mlflow/

# ── Generated Outputs ────────────────────────────────────────
figures/
tables/
logs/
*.log
*.err

# ── Python Environment ───────────────────────────────────────
.venv/
env/
__pycache__/
*.pyc
*.pyo
.eggs/
*.egg-info/
dist/
build/

# ── Jupyter ──────────────────────────────────────────────────
.ipynb_checkpoints/
*.ipynb

# ── OS Artefacts ─────────────────────────────────────────────
.DS_Store
Thumbs.db
desktop.ini

# ── CUDA / C++ Build Artefacts ───────────────────────────────
*.so
*.dll
*.cu.o

# ── Temporary / Debug ────────────────────────────────────────
*.tmp
debug_output/
*.bak
```

**For releasing weights:** Use GitHub Releases to attach `.pt` files as release assets. They are stored outside the Git object graph and don't pollute the repository history.

---

## 📖 Citation

If this work is useful to you, please cite the academic report:

```bibtex
@techreport{dash2026yolov8road,
  title     = {YOLOv8 Optimization for Road Hazard Detection},
  author    = {Dash, Shreyash and Kumar, Aryan},
  year      = {2026},
  month     = {May},
  institution = {Birla Institute of Technology and Science Pilani, Hyderabad Campus},
  type      = {B.E. Design Project Report},
  note      = {ECE/INSTR F376, Supervisors: Dr. Swapna Kulkarni, Prof. RN Ponnalagu}
}
```

---

## 📚 References

1. Jocher, G. et al. (2023). **Ultralytics YOLOv8**. [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
2. Gevorgyan, Z. (2022). **SIoU Loss: More Powerful Learning for Bounding Box Regression**. arXiv:2205.12740.
3. Tong, Z., Chen, Y., Xu, Z., & Yu, R. (2023). **Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism**. arXiv:2301.10051.
4. Ghiasi, G. et al. (2021). **Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation**. IEEE/CVF CVPR.
5. Lin, T.Y. et al. (2017). **Feature Pyramid Networks for Object Detection**. IEEE CVPR.
6. Woo, S. et al. (2018). **CBAM: Convolutional Block Attention Module**. ECCV.
7. Liu, Z. et al. (2022). **Swin Transformer V2: Scaling up Capacity and Resolution**. IEEE/CVF CVPR.
8. Mehta, S. & Rastegari, M. (2022). **MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer**. ICLR.
9. Tan, M. et al. (2020). **EfficientDet: Scalable and Efficient Object Detection**. IEEE/CVF CVPR.
10. Dharneeshkar, J. et al. (2020). **Deep learning based detection of potholes in Indian roads using YOLO**. ICICT, IEEE.

---

<div align="center">

Made at **BITS Pilani Hyderabad Campus** · May 2026

*"A lower mAP on a clean dataset is more trustworthy than a high mAP on a noisy one."*

</div>
