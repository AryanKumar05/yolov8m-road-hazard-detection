"""
Experiment 6: YOLOv8m with CBAM Attention
Channel + Spatial attention for improved feature representation
Citation: Woo et al. (2018) ECCV
Hardware: 2x H100 | Est. Time: ~1.5 hours
"""

from ultralytics import YOLO
import time, json, torch, torch.nn as nn
from pathlib import Path
import sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from config_shared import *

EXP_NAME = 'cbam'
EXP_DIR = Path('experiments') / EXP_NAME

# CBAM Implementation
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

def add_cbam_to_model(model):
    """Add CBAM modules to YOLOv8 backbone"""
    print("Adding CBAM attention modules...")
    
    # Add CBAM after C2f blocks in backbone
    backbone = model.model.model[:10]
    cbam_count = 0
    
    for idx, module in enumerate(backbone):
        if hasattr(module, 'cv2') and hasattr(module.cv2, 'conv'):
            out_channels = module.cv2.conv.out_channels
            cbam = CBAM(out_channels).to(module.cv2.conv.weight.device)
            
            original_forward = module.forward
            def make_forward(orig_fwd, attn):
                def forward(x):
                    x = orig_fwd(x)
                    return attn(x)
                return forward
            
            module.forward = make_forward(original_forward, cbam)
            cbam_count += 1
            print(f"  Added CBAM at layer {idx} (channels: {out_channels})")
    
    print(f"✓ Added {cbam_count} CBAM modules")
    return model

def main():
    print("="*80)
    print(f"EXPERIMENT 6: CBAM Attention")
    print(f"Channel + Spatial attention for better features")
    print(f"GPUs: {len(DEVICE)}x H100 | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"Est. Time: {EXPERIMENT_TIMES['cbam']:.1f}h")
    print("="*80)
    
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    model = YOLO('yolov8m.pt')
    model = add_cbam_to_model(model)
    
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"\nModel with CBAM: {total_params:,} parameters")
    
    train_params = {
        'data': DATA_YAML, 'epochs': EPOCHS, 'batch': BATCH_SIZE,
        'imgsz': IMG_SIZE, 'device': DEVICE, 'project': 'experiments',
        'name': EXP_NAME, 'exist_ok': True, **DEFAULT_AUG,
        'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
        'patience': PATIENCE, 'save': SAVE_BEST, 'save_period': SAVE_PERIOD,
        'cache': CACHE, 'workers': WORKERS, 'verbose': True,
        'optimizer': OPTIMIZER, 'lr0': 0.009,  # Slightly lower for attention
        'lrf': LRF, 'momentum': MOMENTUM, 'weight_decay': WEIGHT_DECAY,
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
        'experiment': EXP_NAME, 'model': 'yolov8m_cbam',
        'attention_module': 'CBAM', 'epochs': EPOCHS,
        'citation': 'Woo et al. (2018) CBAM, ECCV',
        'batch_size': BATCH_SIZE, 'num_gpus': len(DEVICE),
        'training_time_hours': training_time / 3600,
        'modifications': 'CBAM attention in backbone',
        'parameters': total_params,
        'expected_improvement': '+1-2% mAP',
        'latency_cost': '+5-10% inference time',
    }
    
    (EXP_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (EXP_DIR / 'results.json').write_text(json.dumps(results_dict, indent=2))
    
    print(f"\n✓ Completed in {training_time/3600:.2f}h | mAP: {results_dict['val_map50_95']:.4f} | FPS: {results_dict['inference_fps']:.1f}")

if __name__ == '__main__':
    main()
