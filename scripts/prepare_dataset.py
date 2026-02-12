"""
Dataset Preparation and Validation Script
Checks your Roboflow dataset structure and creates data.yaml
"""

import os
import yaml
from pathlib import Path
import shutil

def check_dataset_structure(base_path):
    """Verify Roboflow dataset structure"""
    
    print("="*60)
    print("DATASET STRUCTURE CHECK")
    print("="*60)
    
    base_path = Path(base_path)
    
    # Check required folders
    required_folders = [
        'train/images', 'train/labels',
        'valid/images', 'valid/labels',
        'test/images', 'test/labels'
    ]
    
    all_exist = True
    for folder in required_folders:
        folder_path = base_path / folder
        exists = folder_path.exists()
        status = "✓" if exists else "✗"
        
        if exists:
            file_count = len(list(folder_path.glob('*')))
            print(f"{status} {folder}: {file_count} files")
        else:
            print(f"{status} {folder}: NOT FOUND")
            all_exist = False
    
    return all_exist

def detect_classes(base_path):
    """Auto-detect classes from label files"""
    
    print("\n" + "="*60)
    print("CLASS DETECTION")
    print("="*60)
    
    base_path = Path(base_path)
    label_path = base_path / 'train' / 'labels'
    
    if not label_path.exists():
        print("✗ Label directory not found!")
        return None
    
    # Check for classes.txt (Roboflow format)
    classes_file = base_path / 'classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"✓ Found classes.txt with {len(classes)} classes:")
        for i, cls in enumerate(classes):
            print(f"  {i}: {cls}")
        return classes
    
    # Alternatively, scan label files
    print("classes.txt not found, scanning label files...")
    class_ids = set()
    
    for label_file in label_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
    
    class_ids = sorted(list(class_ids))
    print(f"✓ Found {len(class_ids)} unique class IDs: {class_ids}")
    print("\nPlease manually specify class names in data.yaml")
    
    return None

def count_instances(base_path):
    """Count total instances per class"""
    
    print("\n" + "="*60)
    print("INSTANCE COUNT")
    print("="*60)
    
    base_path = Path(base_path)
    
    for split in ['train', 'valid', 'test']:
        label_path = base_path / split / 'labels'
        
        if not label_path.exists():
            continue
        
        class_counts = {}
        total_instances = 0
        
        for label_file in label_path.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_instances += 1
        
        print(f"\n{split.upper()} SET:")
        print(f"  Total instances: {total_instances}")
        print(f"  Images: {len(list((base_path / split / 'images').glob('*')))}")
        
        if class_counts:
            print(f"  Class distribution:")
            for class_id in sorted(class_counts.keys()):
                count = class_counts[class_id]
                percentage = (count / total_instances) * 100
                print(f"    Class {class_id}: {count} ({percentage:.1f}%)")

def create_data_yaml(base_path, classes=None):
    """Create data.yaml configuration file"""
    
    print("\n" + "="*60)
    print("CREATING data.yaml")
    print("="*60)
    
    base_path = Path(base_path)
    
    # If classes not provided, use default Philippine vehicle classes
    if classes is None:
        print("Using default Philippine vehicle classes...")
        classes = [
            'motorcycle',
            'tricycle', 
            'car',
            'jeepney',
            'bus',
            'truck'
        ]
    
    # Create data.yaml content
    data_yaml = {
        'path': str(base_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': classes
    }
    
    # Save data.yaml
    yaml_path = base_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created: {yaml_path}")
    print(f"\nContents:")
    print("-" * 60)
    print(yaml.dump(data_yaml, default_flow_style=False, sort_keys=False))
    print("-" * 60)

def main():
    # Path to your Roboflow dataset
    # UPDATE THIS PATH if different
    dataset_path = 'datasets/roboflow_ph_vehicles'
    
    print("\nDataset path:", os.path.abspath(dataset_path))
    
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"\n✗ ERROR: Dataset path not found: {dataset_path}")
        print("\nPlease:")
        print("1. Download your Roboflow dataset")
        print("2. Extract it to: datasets/roboflow_ph_vehicles/")
        print("3. Make sure it contains train/, valid/, and test/ folders")
        return
    
    # Check structure
    if not check_dataset_structure(dataset_path):
        print("\n✗ ERROR: Dataset structure is incorrect")
        print("\nExpected structure:")
        print("datasets/roboflow_ph_vehicles/")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  ├── valid/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  └── test/")
        print("      ├── images/")
        print("      └── labels/")
        return
    
    # Detect classes
    classes = detect_classes(dataset_path)
    
    # Count instances
    count_instances(dataset_path)
    
    # Create data.yaml
    create_data_yaml(dataset_path, classes)
    
    print("\n" + "="*60)
    print("✓ DATASET PREPARATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review data.yaml and update class names if needed")
    print("2. Run: python scripts/train_yolov8_full.py")

if __name__ == '__main__':
    main()