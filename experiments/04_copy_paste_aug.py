"""
Experiment 4: YOLOv8m with Copy-Paste Augmentation
Zero-cost augmentation for improved small object detection
Citation: Ghiasi et al. (2021) CVPR
Hardware: 2x H100 | Est. Time: ~1.3 hours
"""

from ultralytics import YOLO
import time, json, torch
from pathlib import Path
import sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from config_shared import *

EXP_NAME = 'copy_paste'
EXP_DIR = Path('experiments') / EXP_NAME

def main():
    print("="*80)
    print(f"EXPERIMENT 4: Copy-Paste Augmentation")
    print(f"Improves small object detection (humans at distance, small potholes)")
    print(f"GPUs: {len(DEVICE)}x H100 | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"Est. Time: {EXPERIMENT_TIMES['copy_paste']:.1f}h")
    print("="*80)
    
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO('yolov8m.pt')
    
    train_params = {
        'data': DATA_YAML, 'epochs': EPOCHS, 'batch': BATCH_SIZE,
        'imgsz': IMG_SIZE, 'device': DEVICE, 'project': 'experiments',
        'name': EXP_NAME, 'exist_ok': True, 
        **COPY_PASTE_AUG,  # Use copy-paste augmentation config
        'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
        'patience': PATIENCE, 'save': SAVE_BEST, 'save_period': SAVE_PERIOD,
        'cache': CACHE, 'workers': WORKERS, 'verbose': True,
        'optimizer': OPTIMIZER, 'lr0': LR0, 'lrf': LRF,
        'momentum': MOMENTUM, 'weight_decay': WEIGHT_DECAY,
    }
    
    print("\n⭐ Copy-Paste Augmentation ENABLED (p=0.5)")
    print("   Also using mixup=0.15, scale=0.9\n")
    
    start_time = time.time()
    results = model.train(**train_params)
    training_time = time.time() - start_time
    
    val_results = model.val()
    
    best_pt = Path(results.save_dir) / 'weights' / 'best.pt'
    model_single = YOLO(str(best_pt))
    dummy_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).cuda()
    for _ in range(20): 
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
        'experiment': EXP_NAME, 'model': 'yolov8m',
        'loss_function': 'CIoU (default)', 'epochs': EPOCHS,
        'citation': 'Ghiasi et al. (2021) CVPR',
        'batch_size': BATCH_SIZE, 'num_gpus': len(DEVICE),
        'training_time_hours': training_time / 3600,
        'modifications': 'Copy-Paste augmentation (p=0.5)',
        'augmentations': COPY_PASTE_AUG,
        'expected_improvement': '+1-3% mAP vs baseline',
        'latency_cost': '0% (augmentation only affects training)',
    }
    
    (EXP_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (EXP_DIR / 'results.json').write_text(json.dumps(results_dict, indent=2))
    
    print(f"\n✓ Completed in {training_time/3600:.2f}h | mAP: {results_dict['val_map50_95']:.4f} | FPS: {results_dict['inference_fps']:.1f}")

if __name__ == '__main__':
    main()
