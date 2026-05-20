import os
import glob
from pathlib import Path

# Paths
BASE_DIR = '/home/shred/Desktop/Programming/Work/Yolov8m/yolov8m/balanced_subset'
SPLITS = ['train', 'valid', 'test']
CLASSES = ['human', 'hump', 'vehicle', 'pothole']

def count_labels(split_name):
    label_dir = os.path.join(BASE_DIR, split_name, 'labels')
    if not os.path.exists(label_dir):
        print(f"Warning: {label_dir} does not exist.")
        return {}
    
    counts = {cls: 0 for cls in CLASSES}
    
    files = glob.glob(os.path.join(label_dir, '*.txt'))
    print(f"Processing {len(files)} files in {split_name}...")
    
    for f in files:
        with open(f, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    cls_idx = int(parts[0])
                    if 0 <= cls_idx < len(CLASSES):
                        cls_name = CLASSES[cls_idx]
                        counts[cls_name] += 1
                    else:
                        print(f"Warning: Class index {cls_idx} out of range in {f}")
                        
    return counts

print(f"{'Split':<10} {'Human':<10} {'Hump':<10} {'Vehicle':<10} {'Pothole':<10} {'Total':<10}")
print("-" * 65)

total_counts = {cls: 0 for cls in CLASSES}

for split in SPLITS:
    counts = count_labels(split)
    row_total = sum(counts.values())
    print(f"{split:<10} {counts['human']:<10} {counts['hump']:<10} {counts['vehicle']:<10} {counts['pothole']:<10} {row_total:<10}")
    
    for cls in CLASSES:
        total_counts[cls] += counts[cls]

print("-" * 65)
row_total = sum(total_counts.values())
print(f"{'TOTAL':<10} {total_counts['human']:<10} {total_counts['hump']:<10} {total_counts['vehicle']:<10} {total_counts['pothole']:<10} {row_total:<10}")
