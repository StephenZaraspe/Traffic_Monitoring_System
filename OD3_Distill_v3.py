"""
OD3 Data Distillation Pipeline — FIXED v3
==========================================
New in this version:
  FIX 5: Hard minimum quotas for minority classes per canvas
          Every canvas MUST contain at least N of each rare class
          before filling remaining slots with cars/motorcycles.
          This guarantees the observer learns minority class features.

Quota logic:
  Phase 1 — Fill mandatory minority slots first (bus, e-jeepney, jeepney, trike)
  Phase 2 — Fill remaining slots with weighted random sampling
"""

import cv2
import os
import random
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────
IPD_TARGET           = 100
OVERLAP_THRESHOLD    = 0.6
SCREENING_CONF       = 0.2
MAX_PLACE_ATTEMPTS   = 40
MAX_OBJECTS          = 120
CANVAS_W             = 640
CANVAS_H             = 640

R_BAR                = 0.15
MAX_EXTENSION_PX     = 30
MAX_CROP_RATIO       = 0.35
MIN_CROP_PX          = 15

# ── FIX 5: Hard minimum quotas per canvas ─────────────────────────────────
# These run in Phase 1 BEFORE any random filling.
# Script will retry aggressively to meet these quotas.
HARD_MINIMUMS = {
    0: 3,   # bus        — must place at least 3 per canvas
    2: 3,   # e-jeepney  — must place at least 3 per canvas
    3: 4,   # jeepney    — must place at least 4 per canvas
    5: 3,   # trike      — must place at least 3 per canvas
}
# Total mandatory slots = 13. Remaining 107 slots filled by weighted random.

# Phase 2 weights for random fill (after quotas are met)
CLASS_WEIGHTS = {
    0: 2,   # bus
    1: 25,  # cars
    2: 2,   # e-jeepney
    3: 4,   # jeepney
    4: 25,  # motorcycle
    5: 2,   # trike
    6: 8,   # trucks
}

CLASS_NAMES = ['bus', 'cars', 'e-jeepney', 'jeepney', 'motorcycle', 'trike', 'trucks']

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
OBSERVER_MODEL_PATH = "runs/detect/train3/weights/best.pt"   # best manually-annotated model
SOURCE_IMAGES_DIR   = "datasets/datasetsfinal_1000_frames/train/images"
SOURCE_LABELS_DIR   = "datasets/datasetsfinal_1000_frames/train/labels"
OUTPUT_IMAGES_DIR   = "datasets/distilled_v3_100_canvases/images"
OUTPUT_LABELS_DIR   = "datasets/distilled_v3_100_canvases/labels"


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def compute_iou(box1, box2):
    ix1 = max(box1[0], box2[0]); iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2]); iy2 = min(box1[3], box2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / float(a1+a2-inter)

def check_overlap(new_box, existing_boxes, threshold):
    return any(compute_iou(new_box, b) > threshold for b in existing_boxes)

def compute_sadce_extension(obj_area, area_min, area_max, r_bar):
    if area_max == area_min: return r_bar
    normalized = max(0.0, min(1.0, (obj_area-area_min)/(area_max-area_min)))
    return (1.0 - normalized) * r_bar

def boxes_to_yolo(placed_objects, canvas_w, canvas_h):
    lines = []
    for cls, x1, y1, x2, y2 in placed_objects:
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(x2,canvas_w),min(y2,canvas_h)
        if x2<=x1 or y2<=y1: continue
        cx = ((x1+x2)/2)/canvas_w; cy = ((y1+y2)/2)/canvas_h
        bw = (x2-x1)/canvas_w;     bh = (y2-y1)/canvas_h
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines

def build_class_index(labels_dir, images_dir):
    class_pool = {i: [] for i in range(7)}
    for lf in os.listdir(labels_dir):
        if not lf.endswith('.txt'): continue
        img_path = os.path.join(images_dir, lf.replace('.txt', '.jpg'))
        if not os.path.exists(img_path): continue
        with open(os.path.join(labels_dir, lf)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(float(parts[0]))
                    if cid in class_pool:
                        class_pool[cid].append((lf.replace('.txt','.jpg'), line.strip()))
    print("\n  Class pool:")
    for cid, items in class_pool.items():
        quota = f"  [quota≥{HARD_MINIMUMS[cid]}]" if cid in HARD_MINIMUMS else ""
        print(f"    [{cid}] {CLASS_NAMES[cid]:<12}: {len(items):>5} instances{quota}")
    return class_pool

def precompute_area_stats(labels_dir, images_dir):
    areas = []
    for lf in os.listdir(labels_dir):
        if not lf.endswith('.txt'): continue
        img = cv2.imread(os.path.join(images_dir, lf.replace('.txt','.jpg')))
        if img is None: continue
        ih, iw = img.shape[:2]
        with open(os.path.join(labels_dir, lf)) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    areas.append((float(p[3])*iw)*(float(p[4])*ih))
    return (min(areas), max(areas)) if areas else (0.0, 1.0)

def try_place_object(canvas, bg_full, class_id, class_pool,
                     placed_objects, placed_boxes,
                     area_min, area_max, observer, source_images_dir):
    """
    Attempt to place one object of class_id onto the canvas.
    Returns True if placed and observer-validated, False otherwise.
    """
    if not class_pool[class_id]:
        return False

    source_img_file, line = random.choice(class_pool[class_id])
    src_img = cv2.imread(os.path.join(source_images_dir, source_img_file))
    if src_img is None: return False

    parts = line.split()
    if len(parts) < 5: return False

    cls_id          = int(float(parts[0]))
    cx,cy,bw,bh     = map(float, parts[1:5])
    src_h,src_w     = src_img.shape[:2]
    ox1 = int((cx-bw/2)*src_w); oy1 = int((cy-bh/2)*src_h)
    ox2 = int((cx+bw/2)*src_w); oy2 = int((cy+bh/2)*src_h)
    obj_w, obj_h    = ox2-ox1, oy2-oy1
    obj_area        = obj_w * obj_h

    if obj_area <= 0 or obj_w < MIN_CROP_PX or obj_h < MIN_CROP_PX:
        return False

    ext_frac   = compute_sadce_extension(obj_area, area_min, area_max, R_BAR)
    ext_pixels = min(int(ext_frac*(src_w+src_h)/2), MAX_EXTENSION_PX)
    ex1=max(0,ox1-ext_pixels); ey1=max(0,oy1-ext_pixels)
    ex2=min(src_w,ox2+ext_pixels); ey2=min(src_h,oy2+ext_pixels)

    crop = src_img[ey1:ey2, ex1:ex2]
    ch, cw = crop.shape[:2]
    if (ch<=0 or cw<=0
            or cw > CANVAS_W*MAX_CROP_RATIO
            or ch > CANVAS_H*MAX_CROP_RATIO):
        return False

    paste_px = paste_py = 0
    placed = False
    for _ in range(MAX_PLACE_ATTEMPTS):
        px = random.randint(0, CANVAS_W-cw)
        py = random.randint(0, CANVAS_H-ch)
        t_x1 = px+(ox1-ex1); t_y1 = py+(oy1-ey1)
        t_x2 = t_x1+obj_w;   t_y2 = t_y1+obj_h
        new_tight = [t_x1, t_y1, t_x2, t_y2]
        if not check_overlap(new_tight, placed_boxes, OVERLAP_THRESHOLD):
            canvas[py:py+ch, px:px+cw] = crop
            placed_objects.append((cls_id, t_x1, t_y1, t_x2, t_y2))
            placed_boxes.append(new_tight)
            paste_px, paste_py = px, py
            placed = True
            break

    if not placed:
        return False

    # Observer screening
    results = observer(canvas, verbose=False)
    confident_boxes = [
        det.xyxy[0].cpu().numpy().astype(int).tolist()
        for det in results[0].boxes
        if float(det.conf) >= SCREENING_CONF
    ]
    last_tight = placed_boxes[-1]
    validated  = any(compute_iou(last_tight, cb) > 0.3 for cb in confident_boxes)

    if not validated:
        canvas[paste_py:paste_py+ch, paste_px:paste_px+cw] = \
            bg_full[paste_py:paste_py+ch, paste_px:paste_px+cw]
        placed_objects.pop()
        placed_boxes.pop()
        return False

    return True

def build_weighted_sequence(target_count):
    total_w = sum(CLASS_WEIGHTS.values())
    seq = []
    for cls_id, w in CLASS_WEIGHTS.items():
        seq.extend([cls_id] * round((w/total_w)*target_count))
    random.shuffle(seq)
    return seq[:target_count]


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_od3_distillation():
    print("="*65)
    print("  OD3 DATA DISTILLATION — FIXED v3")
    print("  Hard minority class quotas enforced per canvas")
    print("="*65)

    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    print("\n[1/4] Loading observer model (train3 — manually annotated)...")
    observer = YOLO(OBSERVER_MODEL_PATH)

    print("[2/4] Scanning area statistics for SA-DCE...")
    area_min, area_max = precompute_area_stats(SOURCE_LABELS_DIR, SOURCE_IMAGES_DIR)
    print(f"      Area range: {area_min:.0f} → {area_max:.0f} px²")

    print("[3/4] Building class index...")
    class_pool = build_class_index(SOURCE_LABELS_DIR, SOURCE_IMAGES_DIR)

    bg_files = [f for f in os.listdir(SOURCE_IMAGES_DIR) if f.endswith('.jpg')]
    print(f"\n[4/4] Generating {IPD_TARGET} canvases...\n")

    total_placed   = 0
    total_rejected = 0
    quota_failures = {cid: 0 for cid in HARD_MINIMUMS}
    class_totals   = {i: 0 for i in range(7)}

    for canvas_idx in range(IPD_TARGET):
        bg_file  = random.choice(bg_files)
        bg_full  = cv2.imread(os.path.join(SOURCE_IMAGES_DIR, bg_file))
        bg_full  = cv2.resize(bg_full, (CANVAS_W, CANVAS_H))
        canvas   = bg_full.copy()

        placed_objects = []
        placed_boxes   = []
        canvas_rejects = 0

        # ── PHASE 1: Hard quota enforcement ───────────────────────────────
        # For each minority class, keep trying until quota is met
        # (up to 5× the quota in attempts to avoid infinite loop)
        quota_met = {cid: 0 for cid in HARD_MINIMUMS}

        for cid, required in HARD_MINIMUMS.items():
            max_attempts = required * 5
            attempts     = 0
            while quota_met[cid] < required and attempts < max_attempts:
                attempts += 1
                success = try_place_object(
                    canvas, bg_full, cid, class_pool,
                    placed_objects, placed_boxes,
                    area_min, area_max, observer, SOURCE_IMAGES_DIR
                )
                if success:
                    quota_met[cid] += 1
                else:
                    canvas_rejects += 1

            if quota_met[cid] < required:
                quota_failures[cid] += 1

        # ── PHASE 2: Fill remaining slots with weighted random ─────────────
        remaining_slots = MAX_OBJECTS - len(placed_objects)
        fill_sequence   = build_weighted_sequence(remaining_slots * 2)
        consecutive_fail = 0

        for cid in fill_sequence:
            if len(placed_objects) >= MAX_OBJECTS:
                break
            if consecutive_fail > 200:
                break
            success = try_place_object(
                canvas, bg_full, cid, class_pool,
                placed_objects, placed_boxes,
                area_min, area_max, observer, SOURCE_IMAGES_DIR
            )
            if success:
                consecutive_fail = 0
            else:
                canvas_rejects  += 1
                consecutive_fail += 1

        # ── SAVE ──────────────────────────────────────────────────────────
        canvas_name = f"distilled_{canvas_idx:04d}"
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, f"{canvas_name}.jpg"), canvas)
        yolo_lines = boxes_to_yolo(placed_objects, CANVAS_W, CANVAS_H)
        with open(os.path.join(OUTPUT_LABELS_DIR, f"{canvas_name}.txt"), 'w') as f:
            f.write('\n'.join(yolo_lines))

        canvas_class_counts = {i: 0 for i in range(7)}
        for obj in placed_objects:
            canvas_class_counts[obj[0]] += 1
            class_totals[obj[0]] += 1

        total_placed   += len(placed_objects)
        total_rejected += canvas_rejects

        quota_str = " ".join(f"{CLASS_NAMES[cid][0:3]}:{quota_met[cid]}/{req}"
                             for cid, req in HARD_MINIMUMS.items())
        print(f"  Canvas {canvas_idx+1:03d}/{IPD_TARGET}  "
              f"obj:{len(placed_objects):>3}  "
              f"rej:{canvas_rejects:>3}  "
              f"quotas[{quota_str}]")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  DISTILLATION v3 COMPLETE")
    print(f"  Total objects    : {total_placed}")
    print(f"  Total rejected   : {total_rejected}")
    print(f"  Rejection rate   : {total_rejected/(total_placed+total_rejected)*100:.1f}%")
    print("\n  Per-class totals:")
    for i in range(7):
        bar  = "█" * (class_totals[i] // 30)
        note = f"  ← quota was {HARD_MINIMUMS[i]}/canvas" if i in HARD_MINIMUMS else ""
        print(f"    [{i}] {CLASS_NAMES[i]:<12}: {class_totals[i]:>5}  {bar}{note}")
    print("\n  Quota failures (canvases where minimum wasn't met):")
    for cid, fails in quota_failures.items():
        status = "✓ OK" if fails == 0 else f"⚠ {fails} canvases fell short"
        print(f"    [{cid}] {CLASS_NAMES[cid]:<12}: {status}")
    print(f"\n  Output → {OUTPUT_IMAGES_DIR}")
    print("="*65)

if __name__ == '__main__':
    run_od3_distillation()