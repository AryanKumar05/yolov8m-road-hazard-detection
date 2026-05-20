#!/usr/bin/env python3
"""
Create a balanced subset with:
- 1000 instances per class for train
- 200 instances per class for valid
- 200 instances per class for test

Strategy: Select images randomly until we hit target instances per class.
Prioritize images with fewer total objects to reduce class overlap issues.
"""

import os
import shutil
import random
from collections import defaultdict
from pathlib import Path
import yaml

# Configuration
BASE_PATH = Path("/home/shred/Desktop/Programming/Work/Yolov8m/yolov8m/Improved_Balanced_Again_Dataset")
OUTPUT_PATH = Path("/home/shred/Desktop/Programming/Work/Yolov8m/yolov8m/balanced_subset")
CLASS_NAMES = {0: 'human', 1: 'hump', 2: 'vehicle', 3: 'pothole'}

# Target instances per class per split
TARGETS = {
    'train': 1000,
    'valid': 200,
    'test': 200
}

SEED = 42  # For reproducibility
random.seed(SEED)


def parse_label_file(label_path):
    """Parse YOLO label file and return list of (class_id, bbox) tuples."""
    annotations = []
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    annotations.append((class_id, bbox, line))
    return annotations


def get_image_path(label_path, images_dir):
    """Find corresponding image file for a label file."""
    stem = label_path.stem
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        img_path = images_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None


def build_image_index(split_path):
    """Build index of all images and their class counts."""
    labels_dir = split_path / 'labels'
    images_dir = split_path / 'images'
    
    image_data = []
    
    if not labels_dir.exists():
        print(f"Warning: Labels directory not found: {labels_dir}")
        return []

    for label_file in labels_dir.glob('*.txt'):
        annotations = parse_label_file(label_file)
        if not annotations:
            continue
            
        img_path = get_image_path(label_file, images_dir)
        if img_path is None:
            continue
        
        # Count instances per class in this image
        class_counts = defaultdict(int)
        for class_id, bbox, line in annotations:
            class_counts[class_id] += 1
        
        image_data.append({
            'label_path': label_file,
            'image_path': img_path,
            'annotations': annotations,
            'class_counts': dict(class_counts),
            'total_objects': len(annotations)
        })
    
    return image_data


def select_balanced_images(image_data, target_per_class):
    """
    Select images to achieve balanced class distribution.
    Uses a greedy approach that prioritizes classes with fewer instances.
    """
    # Shuffle for randomness
    random.shuffle(image_data)
    
    # Sort by total objects (prefer simpler images first for cleaner sampling)
    image_data.sort(key=lambda x: x['total_objects'])
    
    selected = []
    current_counts = {c: 0 for c in CLASS_NAMES.keys()}
    used_images = set()
    
    # Keep selecting until all classes reach target
    iterations = 0
    max_iterations = len(image_data) * 2
    
    while not all(current_counts[c] >= target_per_class for c in CLASS_NAMES.keys()):
        iterations += 1
        if iterations > max_iterations:
            print(f"Warning: Could not reach target for all classes")
            break
        
        # Find the class with minimum instances
        min_class = min(CLASS_NAMES.keys(), key=lambda c: current_counts[c])
        
        if current_counts[min_class] >= target_per_class:
            break
        
        # Find best image to add (contains min_class, preferably not over-represented classes)
        best_image = None
        best_score = float('inf')
        
        for img in image_data:
            if id(img) in used_images:
                continue
            
            if min_class not in img['class_counts']:
                continue
            
            # Score: prefer images that help underrepresented classes
            # without adding too many of already-full classes
            score = 0
            will_overflow = False
            
            for c, count in img['class_counts'].items():
                if current_counts[c] + count > target_per_class * 1.5:  # Allow 50% overflow to ensure we hit targets
                    will_overflow = True
                    break
                # Penalty for adding to already-full classes
                if current_counts[c] >= target_per_class:
                    score += count * 10
                else:
                    score += count
            
            if not will_overflow and score < best_score:
                best_score = score
                best_image = img
        
        if best_image is None:
            # Fallback 1: Relax overflow
            for img in image_data:
                if id(img) in used_images:
                    continue
                if min_class in img['class_counts']:
                     # Just check if checks min_class
                    best_image = img
                    break
        
        if best_image is None:
            print(f"Warning: No more images available for class {CLASS_NAMES[min_class]}")
            # Remove this class from consideration to avoid infinite loop
            # But practically we just break if we can't find anything
            break
        
        selected.append(best_image)
        used_images.add(id(best_image))
        
        for c, count in best_image['class_counts'].items():
            current_counts[c] += count
    
    return selected, current_counts


def copy_selected_images(selected_images, output_split_dir):
    """Copy selected images and labels to output directory."""
    images_out = output_split_dir / 'images'
    labels_out = output_split_dir / 'labels'
    
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    for img_data in selected_images:
        # Copy image
        src_img = img_data['image_path']
        dst_img = images_out / src_img.name
        shutil.copy2(src_img, dst_img)
        
        # Copy label
        src_label = img_data['label_path']
        dst_label = labels_out / src_label.name
        shutil.copy2(src_label, dst_label)


def create_data_yaml(output_path):
    """Create data.yaml for the balanced subset."""
    data = {
        'path': str(output_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(CLASS_NAMES),
        'names': list(CLASS_NAMES.values())
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def main():
    print("=" * 70)
    print("CREATING BALANCED SUBSET (1000 instances/class)")
    print("=" * 70)
    
    # Clean output directory
    if OUTPUT_PATH.exists():
        print(f"\nRemoving existing output directory: {OUTPUT_PATH}")
        shutil.rmtree(OUTPUT_PATH)
    
    OUTPUT_PATH.mkdir(parents=True)
    
    final_stats = {}
    
    for split, target in TARGETS.items():
        print(f"\n{'='*50}")
        print(f"Processing {split.upper()} split (target: {target} per class)")
        print(f"{'='*50}")
        
        split_path = BASE_PATH / split
        if not split_path.exists():
            print(f"Warning: {split} split not found, skipping")
            continue
        
        # Build index
        print("Building image index...")
        image_data = build_image_index(split_path)
        print(f"Found {len(image_data)} images with labels")
        
        if not image_data:
             print("No images found, skipping split.")
             continue

        # Select balanced images
        print("Selecting balanced images...")
        selected, counts = select_balanced_images(image_data, target)
        
        print(f"\nSelected {len(selected)} images:")
        for c in sorted(CLASS_NAMES.keys()):
            name = CLASS_NAMES[c]
            count = counts[c]
            status = "✓" if count >= target else f"⚠ (only {count})"
            print(f"  Class {c} ({name:>8}): {count:>4} instances {status}")
        
        # Copy files
        output_split = OUTPUT_PATH / split
        print(f"\nCopying to {output_split}...")
        copy_selected_images(selected, output_split)
        
        final_stats[split] = {
            'images': len(selected),
            'class_counts': counts
        }
    
    # Create data.yaml
    print("\n" + "=" * 70)
    yaml_path = create_data_yaml(OUTPUT_PATH)
    print(f"Created {yaml_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    total_images = 0
    for split, stats in final_stats.items():
        print(f"\n{split.upper()}: {stats['images']} images")
        for c in sorted(CLASS_NAMES.keys()):
            print(f"  {CLASS_NAMES[c]:>8}: {stats['class_counts'].get(c, 0):>4} instances")
        total_images += stats['images']
    
    print(f"\nTotal images in subset: {total_images}")
    print(f"\nBalanced subset created at: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
