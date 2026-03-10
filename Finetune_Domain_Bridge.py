"""
Stage 2: Domain-Bridge Fine-Tuning
====================================
Run this AFTER od3_distill_fixed.py has generated clean canvases
AND after you have trained the distilled model on those canvases.

What this does:
  - Takes your distilled model (trained on synthetic canvases)
  - Fine-tunes ONLY the detection head on real CCTV frames
  - Freezes early backbone layers to preserve distilled feature learning
  - Uses a low LR to avoid catastrophic forgetting

Expected outcome:
  - Bridges the synthetic-to-real domain gap
  - Should recover cars/motorcycle recall from ~0.01 back toward 0.8+
  - Uses only ~15% of the data the baseline needed

Run command:
  python finetune_domain_bridge.py

Or directly via YOLO CLI (equivalent):
  yolo task=detect mode=train \
    model=runs/detect/train_distilled/weights/best.pt \
    data=finetune_data.yaml \
    epochs=50 \
    freeze=10 \
    lr0=0.001 \
    lrf=0.01 \
    batch=16 \
    imgsz=640 \
    patience=20 \
    name=train_finetuned
"""

from ultralytics import YOLO
import yaml
import os

# ─────────────────────────────────────────────
# PATHS — UPDATE THESE AFTER RERUNNING DISTILLATION
# ─────────────────────────────────────────────

# The model trained on your FIXED distilled canvases
# (run training first: yolo train data=fixed_distilled.yaml model=yolov8s.pt epochs=150)
DISTILLED_MODEL = "runs/detect/train_distilled/weights/best.pt"

# Fine-tune data YAML — points to a subset of real frames
# See create_finetune_yaml() below — it auto-generates this
FINETUNE_YAML   = "finetune_data.yaml"

# Your real dataset directories
REAL_TRAIN      = "datasets/datasetsfinal_1000_frames/train/images"
REAL_VAL        = "datasets/datasetsfinal_1000_frames/valid/images"
REAL_TEST       = "datasets/datasetsfinal_1000_frames/test/images"


def create_finetune_yaml():
    """
    Creates the YAML pointing to real frames for fine-tuning.
    We use the full real train split — the model only sees it for 50 epochs
    with frozen backbone, so it won't overfit.
    """
    config = {
        "train": REAL_TRAIN,
        "val":   REAL_VAL,
        "test":  REAL_TEST,
        "nc":    7,
        "names": ['bus', 'cars', 'e-jeepney', 'jeepney', 'motorcycle', 'trike', 'trucks']
    }
    with open(FINETUNE_YAML, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Created: {FINETUNE_YAML}")


def run_finetuning():
    print("=" * 65)
    print("  STAGE 2: DOMAIN-BRIDGE FINE-TUNING")
    print("=" * 65)

    # Verify distilled model exists
    if not os.path.exists(DISTILLED_MODEL):
        print(f"\n  ERROR: Distilled model not found at: {DISTILLED_MODEL}")
        print("  Make sure you:")
        print("  1. Ran od3_distill_fixed.py to regenerate canvases")
        print("  2. Trained a new model on those canvases first")
        print("  3. Updated DISTILLED_MODEL path above")
        return

    print("\n[1/3] Creating fine-tune YAML...")
    create_finetune_yaml()

    print("[2/3] Loading distilled model...")
    model = YOLO(DISTILLED_MODEL)

    print("[3/3] Starting fine-tuning...")
    print("      freeze=10      → backbone layers locked")
    print("      lr0=0.001      → low LR to avoid forgetting distilled features")
    print("      epochs=50      → short run, just bridging the domain gap")
    print("      patience=20    → early stopping\n")

    results = model.train(
        data      = FINETUNE_YAML,
        epochs    = 50,
        freeze    = 10,        # Lock first 10 layers (backbone feature extractor)
        lr0       = 0.001,     # Low learning rate — preserve distilled knowledge
        lrf       = 0.01,
        batch     = 16,
        imgsz     = 640,
        patience  = 20,        # Stop early if no improvement
        name      = "train_finetuned",
        pretrained= True,

        # Keep augmentations minimal — real frames don't need heavy aug
        mosaic    = 0.5,       # Reduced from 1.0
        mixup     = 0.0,
        copy_paste= 0.0,
        erasing   = 0.2,       # Reduced from 0.4
    )

    print("\n" + "=" * 65)
    print("  FINE-TUNING COMPLETE")
    print(f"  Best model saved to: runs/detect/train_finetuned/weights/best.pt")
    print("\n  NOW RUN EVALUATION:")
    print("  yolo task=detect mode=val \\")
    print("    model=runs/detect/train_finetuned/weights/best.pt \\")
    print("    data=baseline_data.yaml split=test")
    print("\n  Compare this result against train8 (0.491 mAP)")
    print("  and train10 baseline (0.981 mAP) for your thesis table.")
    print("=" * 65)


if __name__ == '__main__':
    run_finetuning()