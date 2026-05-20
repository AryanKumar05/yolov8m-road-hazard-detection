"""
Results Analysis and Visualization - Updated for all 9 experiments
Generates plots and tables for paper
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

EXPERIMENTS = {
    'baseline': 'Baseline (CIoU)',
    'siou': 'SIoU Loss',
    'wiou': 'WIoU Loss',
    'copy_paste': 'Copy-Paste Aug',
    'cbam': 'CBAM',
    'p2_head': 'P2 Head',
    'mobilevit': 'MobileViT',
    'swinv2': 'SwinV2',
    'combined_best': 'Combined Best',
}

def load_all_results():
    results = {}
    for exp_key, exp_name in EXPERIMENTS.items():
        metadata_path = Path('experiments') / exp_key / 'metadata.json'
        results_path = Path('experiments') / exp_key / 'results.json'
        
        if metadata_path.exists() and results_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            with open(results_path) as f:
                exp_results = json.load(f)
            
            results[exp_key] = {**metadata, **exp_results, 'display_name': exp_name}
        else:
            print(f"⚠️  Results not found for {exp_key}")
    
    return results

def create_comparison_table(results):
    baseline = results.get('baseline', {})
    baseline_map = baseline.get('val_map50_95', 0)
    baseline_fps = baseline.get('inference_fps', 0)
    
    print("\n" + "="*120)
    print("COMPREHENSIVE ABLATION STUDY RESULTS")
    print("="*120)
    print(f"{'Experiment':<20} {'mAP50-95':<12} {'Δ mAP':<10} {'FPS':<10} {'Δ FPS%':<10} {'Time(h)':<10} {'Category':<15}")
    print("-"*120)
    
    table_data = []
    
    categories = {
        'baseline': 'Baseline',
        'siou': 'Loss Function',
        'wiou': 'Loss Function',
        'copy_paste': 'Augmentation',
        'cbam': 'Attention',
        'p2_head': 'Architecture',
        'mobilevit': 'Backbone',
        'swinv2': 'Backbone',
        'combined_best': 'Combined',
    }
    
    for exp_key in EXPERIMENTS.keys():
        if exp_key not in results:
            continue
            
        exp_data = results[exp_key]
        map_val = exp_data.get('val_map50_95', 0)
        fps = exp_data.get('inference_fps', 0)
        train_time = exp_data.get('training_time_hours', 0)
        
        map_delta = map_val - baseline_map
        fps_delta = ((fps - baseline_fps) / baseline_fps * 100) if baseline_fps > 0 else 0
        
        print(f"{exp_data['display_name']:<20} "
              f"{map_val:<12.4f} "
              f"{map_delta:+<10.4f} "
              f"{fps:<10.1f} "
              f"{fps_delta:+<10.1f} "
              f"{train_time:<10.2f} "
              f"{categories[exp_key]:<15}")
        
        table_data.append({
            'Experiment': exp_data['display_name'],
            'Category': categories[exp_key],
            'mAP50-95': map_val,
            'Δ mAP': map_delta,
            'FPS': fps,
            'Δ FPS%': fps_delta,
            'Training Hours': train_time,
        })
    
    print("="*120)
    return pd.DataFrame(table_data)

def plot_comprehensive_comparison(results):
    """Create comprehensive multi-panel figure"""
    Path('figures').mkdir(exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    exp_names = []
    maps = []
    fps_vals = []
    categories = []
    colors_map = {
        'Baseline': '#888888',
        'Loss Function': '#2E86AB',
        'Augmentation': '#A23B72',
        'Attention': '#F18F01',
        'Architecture': '#C73E1D',
        'Backbone': '#6A4C93',
        'Combined': '#1B998B',
    }
    
    category_assign = {
        'baseline': 'Baseline', 'siou': 'Loss Function', 'wiou': 'Loss Function',
        'copy_paste': 'Augmentation', 'cbam': 'Attention', 'p2_head': 'Architecture',
        'mobilevit': 'Backbone', 'swinv2': 'Backbone', 'combined_best': 'Combined',
    }
    
    baseline_map = results['baseline']['val_map50_95']
    baseline_fps = results['baseline']['inference_fps']
    
    for exp_key in EXPERIMENTS.keys():
        if exp_key in results:
            exp_names.append(EXPERIMENTS[exp_key])
            maps.append(results[exp_key]['val_map50_95'])
            fps_vals.append(results[exp_key]['inference_fps'])
            categories.append(category_assign[exp_key])
    
    colors = [colors_map[cat] for cat in categories]
    
    # Plot 1: mAP Comparison
    bars = ax1.barh(exp_names, maps, color=colors, alpha=0.8, edgecolor='black')
    ax1.axvline(x=baseline_map, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax1.set_xlabel('mAP50-95', fontweight='bold')
    ax1.set_title('mAP Comparison', fontweight='bold', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Speed Comparison
    bars = ax2.barh(exp_names, fps_vals, color=colors, alpha=0.8, edgecolor='black')
    ax2.axvline(x=baseline_fps, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax2.set_xlabel('FPS', fontweight='bold')
    ax2.set_title('Inference Speed Comparison', fontweight='bold', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Speed-Accuracy Trade-off
    for i, (map_val, fps, cat) in enumerate(zip(maps, fps_vals, categories)):
        ax3.scatter(fps, map_val, s=200, c=colors_map[cat], alpha=0.7, 
                   edgecolors='black', linewidth=1.5, label=cat if cat not in ax3.get_legend_handles_labels()[1] else '')
        ax3.annotate(exp_names[i], (fps, map_val), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('FPS', fontweight='bold')
    ax3.set_ylabel('mAP50-95', fontweight='bold')
    ax3.set_title('Speed-Accuracy Trade-off', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=8)
    
    # Plot 4: Improvement Breakdown
    improvements = [(map_val - baseline_map) / baseline_map * 100 for map_val in maps[1:]]  # Skip baseline
    exp_names_no_baseline = exp_names[1:]
    colors_no_baseline = colors[1:]
    
    bars = ax4.barh(exp_names_no_baseline, improvements, color=colors_no_baseline, alpha=0.8, edgecolor='black')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_xlabel('mAP Improvement vs Baseline (%)', fontweight='bold')
    ax4.set_title('Relative Improvement Analysis', fontweight='bold', fontsize=14)
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comprehensive analysis to figures/comprehensive_analysis.png")
    plt.close()

def plot_p2_head_focus(results):
    """Create special visualization for P2 head (for tomorrow's presentation)"""
    if 'p2_head' not in results or 'baseline' not in results:
        print("⚠️  P2 head or baseline results not available")
        return
    
    Path('figures').mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    baseline = results['baseline']
    p2 = results['p2_head']
    
    # Comparison bars
    metrics = ['mAP50-95', 'mAP50', 'mAP75']
    baseline_vals = [baseline['val_map50_95'], baseline['val_map50'], baseline['val_map75']]
    p2_vals = [p2['val_map50_95'], p2['val_map50'], p2['val_map75']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color='#888888', alpha=0.8)
    ax1.bar(x + width/2, p2_vals, width, label='P2 Head', color='#C73E1D', alpha=0.8)
    ax1.set_ylabel('mAP Score', fontweight='bold')
    ax1.set_title('P2 Head vs Baseline: Accuracy Comparison', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (b_val, p_val) in enumerate(zip(baseline_vals, p2_vals)):
        improvement = ((p_val - b_val) / b_val * 100)
        ax1.text(i, max(b_val, p_val) + 0.01, f'+{improvement:.1f}%', 
                ha='center', fontsize=10, fontweight='bold', color='green')
    
    # Architecture diagram (simplified)
    detection_heads = ['P2\n(160×160)', 'P3\n(80×80)', 'P4\n(40×40)', 'P5\n(20×20)']
    baseline_heads = [0, 1, 1, 1]  # Baseline has P3, P4, P5
    p2_heads = [1, 1, 1, 1]  # P2 has all four
    
    x2 = np.arange(len(detection_heads))
    ax2.bar(x2, baseline_heads, 0.35, label='Baseline', color='#888888', alpha=0.8)
    ax2.bar(x2, p2_heads, 0.35, label='P2 Head', color='#C73E1D', alpha=0.3)
    ax2.set_ylabel('Detection Head Present', fontweight='bold')
    ax2.set_title('Architecture: Detection Head Comparison', fontweight='bold', fontsize=14)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(detection_heads)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No', 'Yes'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax2.text(0, 0.5, '← NEW', fontsize=12, fontweight='bold', color='red', ha='center')
    
    plt.tight_layout()
    plt.savefig('figures/p2_head_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved P2 head analysis to figures/p2_head_analysis.png")
    plt.close()

def generate_latex_table(df):
    Path('tables').mkdir(exist_ok=True)
    
    latex_table = df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Comprehensive ablation study results for YOLOv8m optimization (1000 images, 200 epochs, 2x H100 GPUs).",
        label="tab:comprehensive_results",
        column_format="l" + "c" * (len(df.columns) - 1)
    )
    
    with open('tables/results_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("✓ Saved LaTeX table to tables/results_table.tex")

def main():
    print("="*80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    
    results = load_all_results()
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"✓ Loaded {len(results)} experiment results\n")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_comprehensive_comparison(results)
    plot_p2_head_focus(results)  # Special for tomorrow
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(df)
    
    # Save CSV
    df.to_csv('tables/results_table.csv', index=False)
    print("✓ Saved CSV to tables/results_table.csv")
    
    # Summary
    if 'baseline' in results and 'combined_best' in results:
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        baseline_map = results['baseline']['val_map50_95']
        combined_map = results['combined_best']['val_map50_95']
        improvement = ((combined_map - baseline_map) / baseline_map * 100)
        
        print(f"✓ Combined Best: +{improvement:.2f}% mAP improvement")
        
        if 'p2_head' in results:
            p2_map = results['p2_head']['val_map50_95']
            p2_improvement = ((p2_map - baseline_map) / baseline_map * 100)
            print(f"✓ P2 Head: +{p2_improvement:.2f}% mAP (for small objects)")
        
        print("="*80)
    
    print("\n✓ Analysis complete! Check figures/ and tables/ directories")

if __name__ == '__main__':
    main()
