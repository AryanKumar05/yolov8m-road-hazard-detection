"""
Experiment 7: YOLOv8m with MobileViT Backbone
Hybrid CNN-Transformer for efficient mobile deployment
Citation: Mehta & Rastegari (2022) ICLR
Hardware: 2x H100 | Est. Time: ~1.7 hours
"""

from ultralytics import YOLO
import time, json, torch, torch.nn as nn
from pathlib import Path
import sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from config_shared import *

# Check if transformers library is available
try:
    from transformers import MobileViTModel, MobileViTConfig
    HAS_MOBILEVIT = True
except ImportError:
    HAS_MOBILEVIT = False
    print("⚠️  transformers library not found. Install with: pip install transformers")

EXP_NAME = 'mobilevit'
EXP_DIR = Path('experiments') / EXP_NAME

class MobileViTBackbone(nn.Module):
    """MobileViT backbone for YOLOv8"""
    def __init__(self):
        super().__init__()
        if not HAS_MOBILEVIT:
            raise ImportError("transformers library required: pip install transformers")
        
        # Use MobileViT-small
        config = MobileViTConfig.from_pretrained("apple/mobilevit-small")
        self.mobilevit = MobileViTModel.from_pretrained("apple/mobilevit-small")
        
        # Adapters to match YOLOv8 neck input dimensions
        # MobileViT-small outputs: 64, 96, 128, 160, 640
        # YOLOv8m expects: 256, 512, 768
        self.adapt_p3 = nn.Conv2d(128, 256, 1)
        self.adapt_p4 = nn.Conv2d(160, 512, 1)
        self.adapt_p5 = nn.Conv2d(640, 768, 1)
    
    def forward(self, x):
        # Get MobileViT features
        outputs = self.mobilevit(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Extract multi-scale features
        # MobileViT returns features at different scales
        p3 = self.adapt_p3(hidden_states[3])  # P3/8
        p4 = self.adapt_p4(hidden_states[4])  # P4/16
        p5 = self.adapt_p5(hidden_states[5])  # P5/32
        
        return [p3, p4, p5]

def replace_backbone_with_mobilevit(model):
    """Replace YOLOv8 backbone with MobileViT"""
    print("Replacing backbone with MobileViT...")
    
    if not HAS_MOBILEVIT:
        print("❌ Cannot proceed without transformers library")
        return None
    
    try:
        mobilevit_backbone = MobileViTBackbone()
        
        # Replace backbone (layers 0-9 in YOLOv8)
        # Keep neck and head (layers 10+)
        original_neck_head = model.model.model[10:]
        
        # Create new model list
        new_model = nn.ModuleList([mobilevit_backbone] + list(original_neck_head))
        model.model.model = new_model
        
        print("✓ MobileViT backbone installed")
        return model
    except Exception as e:
        print(f"❌ Error replacing backbone: {e}")
        return None

def main():
    print("="*80)
    print(f"EXPERIMENT 7: MobileViT Backbone")
    print(f"Hybrid CNN-Transformer for efficiency")
    print(f"GPUs: {len(DEVICE)}x H100 | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"Est. Time: {EXPERIMENT_TIMES['mobilevit']:.1f}h")
    print("="*80)
    
    if not HAS_MOBILEVIT:
        print("\n❌ SKIPPING: transformers library not installed")
        print("   Install with: pip install transformers")
        return
    
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    model = YOLO('yolov8m.pt')
    model = replace_backbone_with_mobilevit(model)
    
    if model is None:
        print("❌ Failed to create MobileViT model")
        return
    
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"\nModel with MobileViT: {total_params:,} parameters")
    
    train_params = {
        'data': DATA_YAML, 'epochs': EPOCHS, 'batch': BATCH_SIZE,
        'imgsz': IMG_SIZE, 'device': DEVICE, 'project': 'experiments',
        'name': EXP_NAME, 'exist_ok': True, **DEFAULT_AUG,
        'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
        'patience': PATIENCE, 'save': SAVE_BEST, 'save_period': SAVE_PERIOD,
        'cache': CACHE, 'workers': WORKERS, 'verbose': True,
        'optimizer': 'AdamW',  # Better for transformers
        'lr0': 0.001,  # Lower LR for pretrained ViT
        'lrf': 0.01, 'weight_decay': 0.05,
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
        'experiment': EXP_NAME, 'model': 'yolov8m_mobilevit',
        'backbone': 'MobileViT-small', 'epochs': EPOCHS,
        'citation': 'Mehta & Rastegari (2022) ICLR',
        'batch_size': BATCH_SIZE, 'num_gpus': len(DEVICE),
        'training_time_hours': training_time / 3600,
        'modifications': 'MobileViT backbone (hybrid CNN-Transformer)',
        'parameters': total_params,
        'expected_improvement': '-1 to +1% mAP (trade accuracy for speed)',
        'latency_cost': '-20 to -30% inference time (faster)',
    }
    
    (EXP_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (EXP_DIR / 'results.json').write_text(json.dumps(results_dict, indent=2))
    
    print(f"\n✓ Completed in {training_time/3600:.2f}h | mAP: {results_dict['val_map50_95']:.4f} | FPS: {results_dict['inference_fps']:.1f}")

if __name__ == '__main__':
    main()
