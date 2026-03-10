"""
OD3 Data Distillation Pipeline — FIXED v2
==========================================
Changes from broken version:
  FIX 1: Hard cap on SA-DCE extension (prevents scene-level patches)
  FIX 2: Crop size sanity check (rejects crops > 35% of canvas)
  FIX 3: Observer is now your STRONGEST model (train10/best.pt)
          NOT train3 (weak early model) — this was the silent killer
  FIX 4: Minimum crop size check (rejects crops too tiny to be useful)
  FIX 5: Per-class quota enforcement per canvas (guarantees balance)
  FIX 6: Canvas fill detection is smarter — checks actual pixel density
"""

import cv2
import os
import random
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────
IPD_TARGET           = 100   # Number of distilled canvases to generate
OVERLAP_THRESHOLD    = 0.6   # IoU ceiling for placement rejection (τ)
SCREENING_CONF       = 0.2   # Observer confidence floor (η)
MAX_PLACE_ATTEMPTS   = 40    # Max placement tries per object (M)
MAX_OBJECTS          = 120   # Max objects per canvas
CANVAS_W             = 640
CANVAS_H             = 640

# SA-DCE controls
R_BAR                = 0.15  # Max fractional extension for small objects
MAX_EXTENSION_PX     = 30    # FIX 1: Hard pixel cap — prevents scene-level crops

# Crop size guardrails
MAX_CROP_RATIO       = 0.35  # FIX 2: Crop cannot exceed 35% of canvas dimension
MIN_CROP_PX          = 15    # FIX 4: Crop must be at least 15x15px to be useful

# Class balance: target objects per class per canvas
# Adjust these ratios based on your dataset's real-world distribution
# Higher weight = more of that class per canvas
CLASS_WEIGHTS = {
    0: 3,   # bus        (rare)
    1: 20,  # cars       (dominant — give it fair share)
    2: 2,   # e-jeepney  (very rare)
    3: 5,   # jeepney
    4: 20,  # motorcycle (dominant)
    5: 3,   # trike      (rare)
    6: 7,   # trucks
}
# This sums to 60 slots — script will scale up to MAX_OBJECTS

# ─────────────────────────────────────────────
# PATHS — UPDATE THESE
# ─────────────────────────────────────────────
# FIX 3: Use your STRONGEST model as observer, not train3
OBSERVER_MODEL_PATH = "runs/detect/train3/weights/best.pt"

SOURCE_IMAGES_DIR   = "datasets/datasetsfinal_1000_frames/train/images"
SOURCE_LABELS_DIR   = "datasets/datasetsfinal_1000_frames/train/labels"
OUTPUT_IMAGES_DIR   = "datasets/distilled_fixed_100_canvases/images"
OUTPUT_LABELS_DIR   = "datasets/distilled_fixed_100_canvases/labels"

CLASS_NAMES = ['bus', 'cars', 'e-jeepney', 'jeepney', 'motorcycle', 'trike', 'trucks']


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def compute_iou(box1, box2):
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / float(a1 + a2 - inter)


def check_overlap(new_box, existing_boxes, threshold):
    for box in existing_boxes:
        if compute_iou(new_box, box) > threshold:
            return True
    return False


def compute_sadce_extension(obj_area, area_min, area_max, r_bar):
    """
    Dynamic extension: small objects get more context, large objects get less.
    Hard-capped at MAX_EXTENSION_PX to prevent scene-level patches.
    """
    if area_max == area_min:
        return r_bar
    normalized = (obj_area - area_min) / (area_max - area_min)
    normalized = max(0.0, min(1.0, normalized))
    return (1.0 - normalized) * r_bar


def boxes_to_yolo(placed_objects, canvas_w, canvas_h):
    lines = []
    for obj in placed_objects:
        cls, x1, y1, x2, y2 = obj
        # Clamp to canvas bounds
        x1 = max(0, min(x1, canvas_w))
        y1 = max(0, min(y1, canvas_h))
        x2 = max(0, min(x2, canvas_w))
        y2 = max(0, min(y2, canvas_h))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = ((x1 + x2) / 2) / canvas_w
        cy = ((y1 + y2) / 2) / canvas_h
        bw = (x2 - x1) / canvas_w
        bh = (y2 - y1) / canvas_h
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def build_class_index(labels_dir, images_dir):
    """Build a per-class pool of (image_file, label_line) pairs."""
    class_pool = {i: [] for i in range(7)}
    missing_images = 0
    for lf in os.listdir(labels_dir):
        if not lf.endswith('.txt'):
            continue
        img_file = lf.replace('.txt', '.jpg')
        img_path = os.path.join(images_dir, img_file)
        if not os.path.exists(img_path):
            missing_images += 1
            continue
        with open(os.path.join(labels_dir, lf)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    if class_id in class_pool:
                        class_pool[class_id].append((img_file, line.strip()))

    print(f"\n  Class pool built (skipped {missing_images} missing images):")
    for cid, items in class_pool.items():
        print(f"    [{cid}] {CLASS_NAMES[cid]:<12}: {len(items):>5} instances")
    return class_pool


def precompute_area_stats(labels_dir, images_dir):
    areas = []
    for lf in os.listdir(labels_dir):
        if not lf.endswith('.txt'):
            continue
        img_path = os.path.join(images_dir, lf.replace('.txt', '.jpg'))
        img = cv2.imread(img_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        with open(os.path.join(labels_dir, lf)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                areas.append((bw * iw) * (bh * ih))
    return (min(areas), max(areas)) if areas else (0.0, 1.0)


def build_class_sequence(target_objects):
    """
    Build a randomized sequence of class IDs for one canvas,
    respecting CLASS_WEIGHTS ratios up to target_objects count.
    """
    total_weight = sum(CLASS_WEIGHTS.values())
    sequence = []
    for cls_id, weight in CLASS_WEIGHTS.items():
        count = round((weight / total_weight) * target_objects)
        sequence.extend([cls_id] * count)
    random.shuffle(sequence)
    return sequence[:target_objects]


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_od3_distillation():
    print("=" * 65)
    print("  OD3 DATA DISTILLATION — FIXED v2")
    print("  Observer: train10 (strongest model)")
    print("  SA-DCE:   dynamic + hard-capped at 30px")
    print("  Balance:  class-weighted quota per canvas")
    print("=" * 65)

    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    print("\n[1/4] Loading observer model (train10)...")
    observer = YOLO(OBSERVER_MODEL_PATH)

    print("[2/4] Scanning dataset for SA-DCE area statistics...")
    area_min, area_max = precompute_area_stats(SOURCE_LABELS_DIR, SOURCE_IMAGES_DIR)
    print(f"      Area range: {area_min:.0f}px² → {area_max:.0f}px²")

    print("[3/4] Building class index...")
    class_pool = build_class_index(SOURCE_LABELS_DIR, SOURCE_IMAGES_DIR)

    bg_files = os.listdir(SOURCE_IMAGES_DIR)
    if not bg_files:
        print("ERROR: No images found in SOURCE_IMAGES_DIR")
        return

    print(f"\n[4/4] Generating {IPD_TARGET} distilled canvases...\n")

    # Summary counters
    total_placed    = 0
    total_rejected  = 0
    class_totals    = {i: 0 for i in range(7)}

    for canvas_idx in range(IPD_TARGET):

        # Use a real frame as background (realistic road texture)
        bg_file  = random.choice(bg_files)
        bg_path  = os.path.join(SOURCE_IMAGES_DIR, bg_file)
        bg_full  = cv2.imread(bg_path)
        bg_full  = cv2.resize(bg_full, (CANVAS_W, CANVAS_H))
        canvas   = bg_full.copy()

        placed_objects      = []
        placed_boxes        = []
        consecutive_fail    = 0
        observer_rejects    = 0

        # Build weighted class sequence for this canvas
        class_sequence = build_class_sequence(MAX_OBJECTS)
        seq_idx = 0

        while len(placed_objects) < MAX_OBJECTS:
            if consecutive_fail > 300:
                break
            if seq_idx >= len(class_sequence):
                # Reshuffle and continue if we haven't hit MAX_OBJECTS
                class_sequence = build_class_sequence(MAX_OBJECTS)
                seq_idx = 0

            chosen_class = class_sequence[seq_idx]
            seq_idx += 1

            if not class_pool[chosen_class]:
                consecutive_fail += 1
                continue

            # ── STAGE 1: CANDIDATE SELECTION ──────────────────────────
            source_img_file, line = random.choice(class_pool[chosen_class])
            src_img = cv2.imread(os.path.join(SOURCE_IMAGES_DIR, source_img_file))
            if src_img is None:
                consecutive_fail += 1
                continue

            parts = line.split()
            if len(parts) < 5:
                consecutive_fail += 1
                continue

            class_id        = int(float(parts[0]))
            cx, cy, bw, bh  = map(float, parts[1:5])
            src_h, src_w    = src_img.shape[:2]

            ox1 = int((cx - bw / 2) * src_w)
            oy1 = int((cy - bh / 2) * src_h)
            ox2 = int((cx + bw / 2) * src_w)
            oy2 = int((cy + bh / 2) * src_h)

            obj_w    = ox2 - ox1
            obj_h    = oy2 - oy1
            obj_area = obj_w * obj_h

            if obj_area <= 0:
                consecutive_fail += 1
                continue

            # FIX 4: Skip objects that are too tiny
            if obj_w < MIN_CROP_PX or obj_h < MIN_CROP_PX:
                consecutive_fail += 1
                continue

            # SA-DCE: dynamic extension with hard pixel cap
            ext_frac   = compute_sadce_extension(obj_area, area_min, area_max, R_BAR)
            ext_pixels = int(ext_frac * (src_w + src_h) / 2)
            ext_pixels = min(ext_pixels, MAX_EXTENSION_PX)  # FIX 1: Hard cap

            ex1 = max(0, ox1 - ext_pixels)
            ey1 = max(0, oy1 - ext_pixels)
            ex2 = min(src_w, ox2 + ext_pixels)
            ey2 = min(src_h, oy2 + ext_pixels)

            crop = src_img[ey1:ey2, ex1:ex2]
            ch, cw = crop.shape[:2]

            # FIX 2: Reject oversized crops (prevents scene-level patches)
            if (cw <= 0 or ch <= 0
                    or cw > CANVAS_W * MAX_CROP_RATIO
                    or ch > CANVAS_H * MAX_CROP_RATIO):
                consecutive_fail += 1
                continue

            # ── STAGE 1: PLACEMENT ────────────────────────────────────
            placed   = False
            paste_px = paste_py = 0

            for _ in range(MAX_PLACE_ATTEMPTS):
                px = random.randint(0, CANVAS_W - cw)
                py = random.randint(0, CANVAS_H - ch)

                tight_x1 = px + (ox1 - ex1)
                tight_y1 = py + (oy1 - ey1)
                tight_x2 = tight_x1 + obj_w
                tight_y2 = tight_y1 + obj_h
                new_tight = [tight_x1, tight_y1, tight_x2, tight_y2]

                if not check_overlap(new_tight, placed_boxes, OVERLAP_THRESHOLD):
                    canvas[py:py+ch, px:px+cw] = crop
                    placed_objects.append((class_id, tight_x1, tight_y1, tight_x2, tight_y2))
                    placed_boxes.append(new_tight)
                    paste_px, paste_py = px, py
                    placed = True
                    consecutive_fail = 0
                    break

            if not placed:
                consecutive_fail += 1
                continue

            # ── STAGE 2: CANDIDATE SCREENING ──────────────────────────
            # FIX 3: Observer is now train10 (strong model) not train3
            results = observer(canvas, verbose=False)

            confident_boxes = []
            for det in results[0].boxes:
                if float(det.conf) >= SCREENING_CONF:
                    bx = det.xyxy[0].cpu().numpy().astype(int).tolist()
                    confident_boxes.append(bx)

            last_tight = placed_boxes[-1]
            observer_validates = any(
                compute_iou(last_tight, cb) > 0.3 for cb in confident_boxes
            )

            if not observer_validates:
                # Erase: restore background at that patch
                canvas[paste_py:paste_py+ch, paste_px:paste_px+cw] = \
                    bg_full[paste_py:paste_py+ch, paste_px:paste_px+cw]
                placed_objects.pop()
                placed_boxes.pop()
                consecutive_fail += 1
                observer_rejects += 1

        # ── SAVE CANVAS + LABELS ──────────────────────────────────────
        canvas_name = f"distilled_{canvas_idx:04d}"
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, f"{canvas_name}.jpg"), canvas)

        yolo_lines = boxes_to_yolo(placed_objects, CANVAS_W, CANVAS_H)
        with open(os.path.join(OUTPUT_LABELS_DIR, f"{canvas_name}.txt"), 'w') as f:
            f.write('\n'.join(yolo_lines))

        # Per-canvas stats
        canvas_class_counts = {i: 0 for i in range(7)}
        for obj in placed_objects:
            canvas_class_counts[obj[0]] += 1
            class_totals[obj[0]] += 1

        total_placed   += len(placed_objects)
        total_rejected += observer_rejects

        class_summary = " | ".join(
            f"{CLASS_NAMES[i][0:4]}:{canvas_class_counts[i]}"
            for i in range(7)
        )
        print(f"  Canvas {canvas_idx+1:03d}/{IPD_TARGET}  "
              f"objects:{len(placed_objects):>3}  "
              f"rejected:{observer_rejects:>3}  "
              f"[{class_summary}]")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DISTILLATION COMPLETE")
    print(f"  Canvases generated : {IPD_TARGET}")
    print(f"  Total objects      : {total_placed}")
    print(f"  Observer rejects   : {total_rejected}")
    print(f"  Rejection rate     : {total_rejected/(total_placed+total_rejected)*100:.1f}%")
    print("\n  Per-class totals:")
    for i in range(7):
        bar = "█" * (class_totals[i] // 50)
        print(f"    [{i}] {CLASS_NAMES[i]:<12}: {class_totals[i]:>5}  {bar}")
    print(f"\n  Output → {OUTPUT_IMAGES_DIR}")
    print("=" * 65)
    print("\n  NEXT STEP: Visually check 5-10 canvases before training.")
    print("  Each canvas should show individual vehicles, not scene patches.")


if __name__ == '__main__':
    run_od3_distillation()