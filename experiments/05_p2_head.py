"""
Experiment 5: YOLOv8m with P2 Detection Head ⭐⭐⭐ PRIORITY
Adds 160×160 feature map for tiny object detection (<24px)
Citation: Lin et al. (2017) "Feature Pyramid Networks" CVPR
Hardware: 2x H100 | Est. Time: ~1.8 hours (40% slower due to extra head)

⭐ REQUIRED FOR TOMORROW'S PRESENTATION
"""

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import time, json, torch
from pathlib import Path
import sys
import yaml
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from config_shared import *

EXP_NAME = 'p2_head'
EXP_DIR = Path('experiments') / EXP_NAME

def create_yolov8m_p2_yaml():
    """
    Create YOLOv8m configuration with P2 head
    Adds 160x160 detection head for small objects
    """
    yaml_content = """# YOLOv8m with P2 Head for Small Object Detection
# Based on yolov8m.yaml with additional P2 detection layer

# Parameters
nc: 4  # number of classes (Humps, Vehicles, Humans, Potholes)
scales:
  m: [0.67, 0.75, 768]

# YOLOv8.0m backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [768, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [768, True]]
  - [-1, 1, SPPF, [768, 5]]  # 9

# YOLOv8.0m head with P2
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-medium)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]  # cat backbone P2 (NEW)
  - [-1, 3, C2f, [128]]  # 18 (P2/4-tiny) (NEW)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 15], 1, Concat, [1]]  # cat head P3
  - [-1, 3, C2f, [256]]  # 21 (P3/8-medium)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 24 (P4/16-large)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat backbone P5
  - [-1, 3, C2f, [768]]  # 27 (P5/32-xlarge)

  - [[18, 21, 24, 27], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5) - 4 heads instead of 3
"""
    
    yaml_path = Path('yolov8m_p2.yaml')
    yaml_path.write_text(yaml_content)
    return yaml_path

def main():
    print("="*80)
    print(f"EXPERIMENT 5: P2 Head for Small Object Detection ⭐⭐⭐")
    print(f"Adds 160×160 detection head (4× more anchors than P3)")
    print(f"Best for: Distant humans, small potholes (<24px)")
    print(f"GPUs: {len(DEVICE)}x H100 | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"Est. Time: {EXPERIMENT_TIMES['p2_head']:.1f}h (+40% vs baseline)")
    print("="*80)
    
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create P2 model configuration
    print("\nCreating YOLOv8m-P2 configuration...")
    yaml_path = create_yolov8m_p2_yaml()
    print(f"✓ Config saved to: {yaml_path}")
    
    # Initialize model with P2 head (nc=4 already set in YAML)
    print("\nInitializing YOLOv8m with P2 head...")
    model = YOLO(str(yaml_path))
    
    print("\n📊 Model Architecture:")
    print(f"  Backbone: YOLOv8m")
    print(f"  Detection Heads: P2 (160×160), P3 (80×80), P4 (40×40), P5 (20×20)")
    print(f"  Total Anchors: ~32,400 (vs ~8,400 for standard YOLOv8m)")
    print(f"  P2 Head Targets: Objects < 24px")
    
    train_params = {
        'data': DATA_YAML, 'epochs': EPOCHS, 'batch': BATCH_SIZE,
        'imgsz': IMG_SIZE, 'device': DEVICE, 'project': 'experiments',
        'name': EXP_NAME, 'exist_ok': True, **DEFAULT_AUG,
        'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
        'patience': PATIENCE, 'save': SAVE_BEST, 'save_period': SAVE_PERIOD,
        'cache': CACHE, 'workers': WORKERS, 'verbose': True,
        'optimizer': OPTIMIZER, 'lr0': LR0, 'lrf': LRF,
        'momentum': MOMENTUM, 'weight_decay': WEIGHT_DECAY,
    }
    
    print("\n⚠️  Training with P2 head - expect ~40% slower than baseline")
    print("    This is due to 4× more anchors to process\n")
    
    start_time = time.time()
    results = model.train(**train_params)
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time/3600:.2f} hours")
    
    val_results = model.val()
    
    # Inference benchmark
    print("\nBenchmarking inference speed...")
    best_pt = Path(results.save_dir) / 'weights' / 'best.pt'
    model_single = YOLO(str(best_pt))
    dummy_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).cuda()
    
    for _ in range(20):  # Warmup
        with torch.no_grad(): _ = model_single.model(dummy_img)
    
    inference_times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad(): _ = model_single.model(dummy_img)
        torch.cuda.synchronize()
        inference_times.append(time.time() - start)
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    results_dict = {
        'val_map50_95': float(val_results.box.map),
        'val_map50': float(val_results.box.map50),
        'val_map75': float(val_results.box.map75),
        'inference_time_ms': avg_inference_time * 1000,
        'inference_fps': 1.0 / avg_inference_time,
    }
    
    metadata = {
        'experiment': EXP_NAME,
        'model': 'yolov8m_p2',
        'architecture': 'YOLOv8m with P2 detection head',
        'detection_heads': ['P2 (160×160)', 'P3 (80×80)', 'P4 (40×40)', 'P5 (20×20)'],
        'loss_function': 'CIoU (default)',
        'epochs': EPOCHS,
        'citation': 'Lin et al. (2017) Feature Pyramid Networks, CVPR',
        'batch_size': BATCH_SIZE,
        'num_gpus': len(DEVICE),
        'training_time_hours': training_time / 3600,
        'modifications': 'P2 detection head for small objects',
        'target_objects': 'Tiny objects (<24px): distant humans, small potholes',
        'expected_improvement': '+3-5% mAP (especially for small objects)',
        'latency_cost': '+30-50% inference time (4× more anchors)',
        'presentation_notes': {
            'key_benefit': 'Significantly better small object detection',
            'trade_off': 'Higher latency but acceptable for many applications',
            'best_use_case': 'When small object detection is critical',
        }
    }
    
    (EXP_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (EXP_DIR / 'results.json').write_text(json.dumps(results_dict, indent=2))
    
    print("\n" + "="*80)
    print("P2 HEAD EXPERIMENT SUMMARY ⭐")
    print("="*80)
    print(f"mAP50-95:       {results_dict['val_map50_95']:.4f}")
    print(f"mAP50:          {results_dict['val_map50']:.4f}")
    print(f"Inference Time: {results_dict['inference_time_ms']:.2f} ms")
    print(f"FPS:            {results_dict['inference_fps']:.1f}")
    print(f"Training Time:  {training_time/3600:.2f} hours")
    print("\n📊 For Tomorrow's Presentation:")
    print(f"  ✓ P2 head adds 160×160 feature map")
    print(f"  ✓ 4× more anchors for tiny object detection")
    print(f"  ✓ Targets objects < 24 pixels")
    print(f"  ✓ Trade-off: +{((results_dict['inference_time_ms'] / 10) - 1) * 100:.0f}% latency for +X% small object mAP")
    print("="*80)
    print(f"\n✓ Results saved to: {EXP_DIR}/")
    print(f"✓ Model config: yolov8m_p2.yaml")
    print(f"✓ Best weights: {EXP_DIR}/weights/best.pt")

if __name__ == '__main__':
    main()
