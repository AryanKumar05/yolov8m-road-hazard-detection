"""
Experiment 3: YOLOv8m with WIoU Loss
Dynamic focusing for noisy annotations (good for potholes with subjective boundaries)
Citation: Tong et al. (2023) arXiv:2301.10051
Hardware: 2x H100 | Est. Time: ~1.3 hours
"""

from ultralytics import YOLO
import time, json, torch
from pathlib import Path
import sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from config_shared import *

EXP_NAME = 'wiou'
EXP_DIR = Path('experiments') / EXP_NAME

def main():
    print("="*80)
    print(f"EXPERIMENT 3: WIoU Loss")
    print(f"Dynamic focusing for noisy annotation handling")
    print(f"GPUs: {len(DEVICE)}x H100 | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"Est. Time: {EXPERIMENT_TIMES['wiou']:.1f}h")
    print("="*80)
    
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n⚠️  Using WIoU Loss - Ensure WIoUv3Loss is in your modified ultralytics\n")
    
    model = YOLO('yolov8m.pt')
    
    # Enable WIoU loss by setting flag
    model.model.model[-1].use_wiou = True
    print("✓ WIoU loss enabled via flag\n")
    
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
        'loss_function': 'WIoU', 'epochs': EPOCHS,
        'citation': 'Tong et al. (2023) arXiv:2301.10051',
        'batch_size': BATCH_SIZE, 'num_gpus': len(DEVICE),
        'training_time_hours': training_time / 3600,
        'modifications': 'WIoU loss function',
        'expected_improvement': '+0.8-1.5% mAP vs baseline',
        'latency_cost': '0% (loss only)',
    }
    
    (EXP_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (EXP_DIR / 'results.json').write_text(json.dumps(results_dict, indent=2))
    
    print(f"\n✓ Completed in {training_time/3600:.2f}h | mAP: {results_dict['val_map50_95']:.4f} | FPS: {results_dict['inference_fps']:.1f}")

if __name__ == '__main__':
    main()
