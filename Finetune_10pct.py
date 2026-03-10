"""
Domain Bridge Fine-Tuning — 10% Real Data Version
===================================================
Gemini's Point 1: Using 100% of real data to fine-tune destroys the
efficiency claim. This script randomly samples 10% (~100 frames) of
the real training set so the panel can't ask "why not just train on
all 1000 frames?"

Defense argument:
  "Our proposed pipeline achieves competitive performance using only
   100 distilled synthetic canvases plus 100 real frames (10% of the
   full dataset), compared to the baseline which required all 800
   real training frames."
"""

import os
import random
import shutil
import yaml
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DISTILLED_MODEL     = "runs/detect/train_distilled_v3/weights/best.pt"
REAL_TRAIN_IMAGES   = "datasets/datasetsfinal_1000_frames/train/images"
REAL_TRAIN_LABELS   = "datasets/datasetsfinal_1000_frames/train/labels"
REAL_VAL_IMAGES     = "datasets/datasetsfinal_1000_frames/valid/images"
REAL_TEST_IMAGES    = "datasets/datasetsfinal_1000_frames/test/images"

SUBSET_IMAGES_DIR   = "datasets/finetune_10pct/images"
SUBSET_LABELS_DIR   = "datasets/finetune_10pct/labels"
FINETUNE_YAML       = "finetune_10pct.yaml"

SAMPLE_FRACTION     = 0.10   # 10% of real training frames
RANDOM_SEED         = 42     # fixed seed → reproducible subset for thesis


def create_10pct_subset():
    """Randomly sample 10% of real training frames and copy to subset folder."""
    os.makedirs(SUBSET_IMAGES_DIR, exist_ok=True)
    os.makedirs(SUBSET_LABELS_DIR, exist_ok=True)

    all_images = [f for f in os.listdir(REAL_TRAIN_IMAGES) if f.endswith('.jpg')]
    random.seed(RANDOM_SEED)
    subset = random.sample(all_images, max(1, int(len(all_images) * SAMPLE_FRACTION)))

    copied = 0
    for img_file in subset:
        lbl_file = img_file.replace('.jpg', '.txt')
        src_img  = os.path.join(REAL_TRAIN_IMAGES, img_file)
        src_lbl  = os.path.join(REAL_TRAIN_LABELS, lbl_file)
        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.copy(src_img, os.path.join(SUBSET_IMAGES_DIR, img_file))
            shutil.copy(src_lbl, os.path.join(SUBSET_LABELS_DIR, lbl_file))
            copied += 1

    print(f"  Sampled {copied}/{len(all_images)} frames "
          f"({copied/len(all_images)*100:.1f}%) → {SUBSET_IMAGES_DIR}")
    return copied


def create_finetune_yaml():
    config = {
        "train": SUBSET_IMAGES_DIR,   # only 10% of real frames
        "val":   REAL_VAL_IMAGES,
        "test":  REAL_TEST_IMAGES,
        "nc":    7,
        "names": ['bus', 'cars', 'e-jeepney', 'jeepney', 'motorcycle', 'trike', 'trucks']
    }
    with open(FINETUNE_YAML, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Created: {FINETUNE_YAML}")


def run_finetuning():
    print("="*65)
    print("  DOMAIN BRIDGE FINE-TUNING  —  10% Real Data")
    print("="*65)

    if not os.path.exists(DISTILLED_MODEL):
        print(f"\n  ERROR: Distilled model not found: {DISTILLED_MODEL}")
        print("  Train on distilled v3 canvases first:")
        print("  yolo train data=distilled_v3.yaml model=yolov8s.pt epochs=150 name=train_distilled_v3")
        return

    print(f"\n[1/3] Creating 10% real-frame subset (seed={RANDOM_SEED})...")
    n_frames = create_10pct_subset()

    print("[2/3] Creating fine-tune YAML...")
    create_finetune_yaml()

    print(f"[3/3] Fine-tuning on {n_frames} real frames...")
    print("      freeze=10  → backbone locked (preserves distilled features)")
    print("      lr0=0.001  → low LR (avoids catastrophic forgetting)")
    print("      epochs=50  → short run, just bridging domain gap\n")

    model   = YOLO(DISTILLED_MODEL)
    results = model.train(
        data      = FINETUNE_YAML,
        epochs    = 50,
        freeze    = 10,
        lr0       = 0.001,
        lrf       = 0.01,
        batch     = 16,
        imgsz     = 640,
        patience  = 20,
        name      = "train_finetuned_10pct",
        mosaic    = 0.5,
        mixup     = 0.0,
        erasing   = 0.2,
    )

    print("\n" + "="*65)
    print("  FINE-TUNING COMPLETE")
    print("  Model: runs/detect/train_finetuned_10pct/weights/best.pt")
    print(f"\n  Training data used:")
    print(f"    Distilled canvases : 100 synthetic images")
    print(f"    Real frames        : {n_frames} ({SAMPLE_FRACTION*100:.0f}% of dataset)")
    print(f"    Total              : {100 + n_frames} images")
    print(f"    vs Baseline        : ~800 real images")
    print(f"    Data reduction     : ~{(1 - (100+n_frames)/800)*100:.0f}%")
    print("="*65)


if __name__ == '__main__':
    run_finetuning()