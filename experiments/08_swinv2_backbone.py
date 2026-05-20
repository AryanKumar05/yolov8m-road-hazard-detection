"""
Experiment 8: YOLOv8m with Swin Transformer V2 Backbone
Hierarchical Vision Transformer for maximum accuracy
Citation: Liu et al. (2022) CVPR
Hardware: 2x H100 | Est. Time: ~2.3 hours (80% slower - heavy backbone)
"""

from ultralytics import YOLO
import time, json, torch, torch.nn as nn
from pathlib import Path
import sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from config_shared import *

# Check for timm library
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("⚠️  timm library not found. Install with: pip install timm")

EXP_NAME = 'swinv2'
EXP_DIR = Path('experiments') / EXP_NAME

class SwinV2Backbone(nn.Module):
    """Swin Transformer V2 backbone for YOLOv8"""
    def __init__(self, model_name='swinv2_tiny_window8_256'):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm library required: pip install timm")
        
        # Load pretrained Swin V2
        self.swin = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3)  # Get P3, P4, P5 equivalent features
        )
        
        # Get feature dimensions from Swin
        # swinv2_tiny: [96, 192, 384, 768]
        # We'll use indices 1,2,3 which give us [192, 384, 768]
        swin_channels = self.swin.feature_info.channels()
        
        # Adapt to YOLOv8m expected channels [256, 512, 768]
        self.adapt_p3 = nn.Conv2d(swin_channels[0], 256, 1)  # 192 -> 256
        self.adapt_p4 = nn.Conv2d(swin_channels[1], 512, 1)  # 384 -> 512  
        self.adapt_p5 = nn.Conv2d(swin_channels[2], 768, 1)  # 768 -> 768
        
        print(f"SwinV2 channels: {swin_channels}")
        print(f"Adapted to YOLOv8: [256, 512, 768]")
    
    def forward(self, x):
        # Get multi-scale features from Swin
        features = self.swin(x)
        
        # Adapt channels to match YOLOv8 neck
        p3 = self.adapt_p3(features[0])  # P3/8
        p4 = self.adapt_p4(features[1])  # P4/16
        p5 = self.adapt_p5(features[2])  # P5/32
        
        return [p3, p4, p5]

def replace_backbone_with_swinv2(model):
    """Replace YOLOv8 backbone with SwinV2"""
    print("Replacing backbone with Swin Transformer V2...")
    
    if not HAS_TIMM:
        print("❌ Cannot proceed without timm library")
        return None
    
    try:
        swinv2_backbone = SwinV2Backbone()
        
        # Replace backbone (layers 0-9 in YOLOv8)
        original_neck_head = model.model.model[10:]
        
        # Create new model list
        new_model = nn.ModuleList([swinv2_backbone] + list(original_neck_head))
        model.model.model = new_model
        
        print("✓ SwinV2 backbone installed")
        return model
    except Exception as e:
        print(f"❌ Error replacing backbone: {e}")
        return None

def main():
    print("="*80)
    print(f"EXPERIMENT 8: Swin Transformer V2 Backbone")
    print(f"Hierarchical ViT for maximum accuracy")
    print(f"GPUs: {len(DEVICE)}x H100 | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"Est. Time: {EXPERIMENT_TIMES['swinv2']:.1f}h (+80% vs baseline)")
    print("="*80)
    
    if not HAS_TIMM:
        print("\n❌ SKIPPING: timm library not installed")
        print("   Install with: pip install timm")
        return
    
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    model = YOLO('yolov8m.pt')
    model = replace_backbone_with_swinv2(model)
    
    if model is None:
        print("❌ Failed to create SwinV2 model")
        return
    
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"\nModel with SwinV2: {total_params:,} parameters")
    
    train_params = {
        'data': DATA_YAML, 'epochs': EPOCHS, 'batch': BATCH_SIZE,
        'imgsz': IMG_SIZE, 'device': DEVICE, 'project': 'experiments',
        'name': EXP_NAME, 'exist_ok': True, **DEFAULT_AUG,
        'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
        'patience': PATIENCE, 'save': SAVE_BEST, 'save_period': SAVE_PERIOD,
        'cache': CACHE, 'workers': WORKERS, 'verbose': True,
        'optimizer': 'AdamW',  # Better for transformers
        'lr0': 0.0001,  # Much lower LR for pretrained Swin
        'lrf': 0.01, 'weight_decay': 0.05,
    }
    
    print("\n⚠️  Training with SwinV2 - expect ~80% slower than baseline")
    print("    This is due to transformer self-attention complexity\n")
    
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
        'experiment': EXP_NAME, 'model': 'yolov8m_swinv2',
        'backbone': 'Swin Transformer V2 Tiny', 'epochs': EPOCHS,
        'citation': 'Liu et al. (2022) Swin V2, CVPR',
        'batch_size': BATCH_SIZE, 'num_gpus': len(DEVICE),
        'training_time_hours': training_time / 3600,
        'modifications': 'SwinV2 Transformer backbone',
        'parameters': total_params,
        'expected_improvement': '+2-4% mAP (highest accuracy)',
        'latency_cost': '+100-200% inference time (much slower)',
        'best_use_case': 'Offline processing where accuracy is critical',
    }
    
    (EXP_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (EXP_DIR / 'results.json').write_text(json.dumps(results_dict, indent=2))
    
    print(f"\n✓ Completed in {training_time/3600:.2f}h | mAP: {results_dict['val_map50_95']:.4f} | FPS: {results_dict['inference_fps']:.1f}")

if __name__ == '__main__':
    main()
