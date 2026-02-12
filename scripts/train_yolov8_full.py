"""
YOLOv8 Training Script - Full Dataset
Trains YOLOv8 on complete Roboflow dataset
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import os

def verify_environment():
    """Verify GPU and dataset before training"""
    
    print("="*60)
    print("PRE-TRAINING VERIFICATION")
    print("="*60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("✗ WARNING: CUDA not available - training will be VERY slow")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            exit()
    else:
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA: {torch.version.cuda}")
    
    # Check dataset
    data_yaml = 'datasets/roboflow_ph_vehicles/data.yaml'
    if not os.path.exists(data_yaml):
        print(f"\n✗ ERROR: {data_yaml} not found")
        print("Run: python scripts/prepare_dataset.py first")
        exit()
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n✓ Dataset config loaded")
    print(f"  Classes: {data_config['nc']}")
    print(f"  Names: {data_config['names']}")
    
    print("\n" + "="*60)

def train_yolov8():
    """Main training function"""
    
    # Verify everything is ready
    verify_environment()
    
    print("\nStarting training...")
    print("This will take 2-3 hours on RTX 3070")
    print("Press Ctrl+C to stop training\n")
    
    # Load pretrained YOLOv8 model
    # Options: yolov8n.pt (fastest), yolov8s.pt (balanced), yolov8m.pt (accurate)
    model = YOLO('yolov8s.pt')
    
    # Training parameters
    results = model.train(
        # Data
        data='datasets/roboflow_ph_vehicles/data.yaml',
        
        # Training duration
        epochs=100,
        patience=20,  # Early stopping
        
        # Image size and batch
        imgsz=640,
        batch=16,  # Reduce to 8 if out of memory
        
        # Hardware
        device=0,  # GPU 0, use 'cpu' if no GPU
        workers=4,  # Adjust based on CPU cores
        
        # Optimization
        amp=True,  # Automatic Mixed Precision - faster training
        close_mosaic=10,  # Disable mosaic augmentation last 10 epochs
        
        # Checkpointing
        save=True,
        save_period=10,  # Save every 10 epochs
        
        # Output
        project='models/yolov8_full',
        name='run1',
        exist_ok=True,
        
        # Resume from checkpoint (if needed)
        # resume=True,  # Uncomment to resume interrupted training
        
        # Augmentation (Roboflow already did augmentation)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    # Validate the trained model
    print("\nRunning final validation...")
    metrics = model.val()
    
    # Print results
    print("\n" + "="*60)
    print("FINAL METRICS")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # Per-class metrics
    print("\nPer-Class AP50:")
    for i, name in model.names.items():
        print(f"  {name}: {metrics.box.maps[i]:.4f}")
    
    best_weights = Path(model.trainer.best)
    print(f"\n✓ Best model saved: {best_weights}")
    print(f"✓ Training results: models/yolov8_full/run1/")
    
    print("\nNext steps:")
    print("1. Check training curves: models/yolov8_full/run1/results.png")
    print("2. Run evaluation: python scripts/evaluate_model.py")

if __name__ == '__main__':
    try:
        train_yolov8()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("To resume, uncomment 'resume=True' in train_yolov8_full.py")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check data.yaml path is correct")
        print("2. Verify GPU memory (reduce batch size if OOM)")
        print("3. Ensure dataset images and labels exist")