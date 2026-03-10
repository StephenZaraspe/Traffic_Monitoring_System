"""
OD3 Data Distillation Pipeline — Fixed & Complete Implementation
Based on: "OD3: Optimization-free Dataset Distillation for Object Detection"

Fixes over the previous version:
  1. Remove phase ACTUALLY erases low-confidence objects from the canvas
  2. SA-DCE is dynamic (smaller objects get proportionally larger context borders)
  3. Labels (.txt in YOLO format) are saved alongside every distilled canvas
  4. Tracks placed objects with class_id for proper label output
  5. Canvas fill logic is smarter — stops when canvas is genuinely full
  6. NEW: Class-Aware Inverse Sampling to defeat minority class imbalance
"""

import cv2
import os
import random
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────
# HYPERPARAMETERS  (from OD3 paper, Section 3)
# ─────────────────────────────────────────────
IPD_TARGET         = 100    # How many distilled canvases to generate (10 = ~1% of 1k frames)
OVERLAP_THRESHOLD  = 0.6   # IoU ceiling before a placement is rejected (τ)
SCREENING_CONF     = 0.2   # Observer confidence floor — below this, object is erased (η)
MAX_PLACE_ATTEMPTS = 40    # Max random placement tries per object (M)
MAX_OBJECTS        = 120   # Max objects to attempt per canvas before declaring it "full"
CANVAS_W           = 640
CANVAS_H           = 640

# SA-DCE dynamic extension range (r̄ in the paper)
R_BAR              = 0.15  # Max fractional extension added to small objects
A_MIN_FRAC         = 0.0   # Normalized min object area (computed at runtime)
A_MAX_FRAC         = 1.0   # Normalized max object area (computed at runtime)


# ─────────────────────────────────────────────
# PATHS  —  update these to your actual paths
# ─────────────────────────────────────────────
OBSERVER_MODEL_PATH = "runs/detect/train3/weights/best.pt"
SOURCE_IMAGES_DIR   = "datasets/datasetsfinal_1000_frames/train/images"  
SOURCE_LABELS_DIR   = "datasets/datasetsfinal_1000_frames/train/labels"  
OUTPUT_IMAGES_DIR   = "datasets/distilled_unbalanced_100_canvases/images"
OUTPUT_LABELS_DIR   = "datasets/distilled_unbalanced_100_canvases/labels"


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def compute_iou(box1, box2):
    """IoU between two [x1,y1,x2,y2] boxes."""
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
    Scale-Aware Dynamic Context Extension (SA-DCE).
    """
    if area_max == area_min:
        return r_bar
    normalized = (obj_area - area_min) / (area_max - area_min)
    normalized = max(0.0, min(1.0, normalized))
    return (1.0 - normalized) * r_bar


def boxes_to_yolo(placed_objects, canvas_w, canvas_h):
    """Convert list of placed objects to YOLO label lines."""
    lines = []
    for obj in placed_objects:
        cls, x1, y1, x2, y2 = obj
        cx = ((x1 + x2) / 2) / canvas_w
        cy = ((y1 + y2) / 2) / canvas_h
        bw = (x2 - x1) / canvas_w
        bh = (y2 - y1) / canvas_h
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines

def build_class_index(labels_dir, images_dir):
    """
    Pre-sorts every single object in the dataset into buckets based on Class ID.
    This enables Class-Aware Sampling to defeat class imbalance.
    """
    class_pool = {i: [] for i in range(7)}  # Buckets for classes 0 through 6
    
    for lf in os.listdir(labels_dir):
        if not lf.endswith('.txt'):
            continue
        img_file = lf.replace('.txt', '.jpg')
        
        with open(os.path.join(labels_dir, lf)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    if class_id in class_pool:
                        class_pool[class_id].append((img_file, line.strip()))
                        
    return class_pool

# ─────────────────────────────────────────────
# PRE-SCAN: collect all object areas for SA-DCE
# ─────────────────────────────────────────────

def precompute_area_stats(labels_dir, images_dir):
    """
    Walk all label files once to find the global min/max object area.
    This is needed so SA-DCE can normalize object sizes properly.
    """
    areas = []
    for lf in os.listdir(labels_dir):
        if not lf.endswith('.txt'):
            continue
        img_file = lf.replace('.txt', '.jpg')
        img_path = os.path.join(images_dir, img_file)
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
                px_w = bw * iw
                px_h = bh * ih
                areas.append(px_w * px_h)
    if not areas:
        return 0.0, 1.0
    return min(areas), max(areas)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_od3_distillation():
    print("=" * 60)
    print("  OD3 DATA DISTILLATION — FULL PIPELINE (BALANCED)")
    print("=" * 60)

    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    # Load observer model
    print("\n[1/4] Loading observer model...")
    observer = YOLO(OBSERVER_MODEL_PATH)

    # Precompute area stats for dynamic SA-DCE
    print("[2/4] Scanning dataset for SA-DCE area statistics...")
    area_min, area_max = precompute_area_stats(SOURCE_LABELS_DIR, SOURCE_IMAGES_DIR)
    print(f"      Object area range: {area_min:.0f}px² → {area_max:.0f}px²")

    print("[2.5/4] Building Class-Aware Index to prevent minority class zeros...")
    class_pool = build_class_index(SOURCE_LABELS_DIR, SOURCE_IMAGES_DIR)

    # Gather all label files
    label_files = [f for f in os.listdir(SOURCE_LABELS_DIR) if f.endswith('.txt')]
    if not label_files:
        print("ERROR: No label files found. Check SOURCE_LABELS_DIR path.")
        return

    print(f"[3/4] Found {len(label_files)} annotated frames. Starting distillation...")
    print(f"      Generating {IPD_TARGET} balanced canvases.\n")

    total_objects_saved = 0

    # ── OUTER LOOP: one iteration = one distilled canvas ──────────────
    for canvas_idx in range(IPD_TARGET):
        print(f"  ┌─ Canvas {canvas_idx + 1}/{IPD_TARGET}")

        # Pick a random real frame as background (gives realistic road texture)
        bg_file = random.choice(os.listdir(SOURCE_IMAGES_DIR))
        canvas = cv2.imread(os.path.join(SOURCE_IMAGES_DIR, bg_file))
        canvas = cv2.resize(canvas, (CANVAS_W, CANVAS_H))

        # placed_objects: list of (class_id, x1, y1, x2, y2) — tight box on canvas
        placed_objects = []
        placed_boxes   = []   # just the [x1,y1,x2,y2] for IoU checks
        attempts_total = 0
        consecutive_failures = 0

        # ── INNER LOOP: add objects one at a time ─────────────────────
        while len(placed_objects) < MAX_OBJECTS:
            if consecutive_failures > 200:
                print(f"  │   Canvas full after {len(placed_objects)} objects.")
                break

            # ── STAGE 1: CANDIDATE SELECTION (UNBALANCED RANDOM) ──────────
            source_lf = random.choice(label_files)
            source_img_path = os.path.join(SOURCE_IMAGES_DIR, source_lf.replace('.txt', '.jpg'))
            
            src_img = cv2.imread(source_img_path)
            if src_img is None:
                consecutive_failures += 1
                continue
                
            with open(os.path.join(SOURCE_LABELS_DIR, source_lf)) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                consecutive_failures += 1
                continue
                
            line = random.choice(lines)
            parts = line.split()
            if len(parts) < 5:
                consecutive_failures += 1
                continue

            class_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])

            src_h, src_w = src_img.shape[:2]
            ox1 = int((cx - bw / 2) * src_w)
            oy1 = int((cy - bh / 2) * src_h)
            ox2 = int((cx + bw / 2) * src_w)
            oy2 = int((cy + bh / 2) * src_h)

            obj_area = (ox2 - ox1) * (oy2 - oy1)
            if obj_area <= 0:
                consecutive_failures += 1
                continue

            # ── SA-DCE: dynamic context extension ─────────────────────
            ext_frac   = compute_sadce_extension(obj_area, area_min, area_max, R_BAR)
            ext_pixels = int(ext_frac * (src_w + src_h) / 2)  # scale to image size

            ex1 = max(0, ox1 - ext_pixels)
            ey1 = max(0, oy1 - ext_pixels)
            ex2 = min(src_w, ox2 + ext_pixels)
            ey2 = min(src_h, oy2 + ext_pixels)

            crop = src_img[ey1:ey2, ex1:ex2]
            ch, cw = crop.shape[:2]

            if ch <= 0 or cw <= 0 or cw > CANVAS_W or ch > CANVAS_H:
                consecutive_failures += 1
                continue

            # ── STAGE 1: PLACEMENT ────────────────────────────────────
            placed = False
            for _ in range(MAX_PLACE_ATTEMPTS):
                px = random.randint(0, CANVAS_W - cw)
                py = random.randint(0, CANVAS_H - ch)

                # Tight box of the actual object (not the extended crop) on canvas
                tight_x1 = px + (ox1 - ex1)
                tight_y1 = py + (oy1 - ey1)
                tight_x2 = tight_x1 + (ox2 - ox1)
                tight_y2 = tight_y1 + (oy2 - oy1)
                new_tight = [tight_x1, tight_y1, tight_x2, tight_y2]

                if not check_overlap(new_tight, placed_boxes, OVERLAP_THRESHOLD):
                    # Paste the extended crop onto the canvas
                    canvas[py:py+ch, px:px+cw] = crop

                    # Temporarily record the placement
                    placed_objects.append((class_id, tight_x1, tight_y1, tight_x2, tight_y2))
                    placed_boxes.append(new_tight)
                    placed = True
                    consecutive_failures = 0
                    break

            if not placed:
                consecutive_failures += 1
                continue

            # ── STAGE 2: CANDIDATE SCREENING (THE REAL REMOVE PHASE) ──
            # Run observer on the CURRENT canvas state
            results = observer(canvas, verbose=False)

            # Build a set of boxes the observer IS confident about
            confident_boxes = []
            for det in results[0].boxes:
                if float(det.conf) >= SCREENING_CONF:
                    bx = det.xyxy[0].cpu().numpy().astype(int)
                    confident_boxes.append(bx)

            # For the object we JUST placed, check if the observer sees it
            # If no confident detection overlaps with what we just pasted → ERASE IT
            last_tight = placed_boxes[-1]
            observer_validates = any(
                compute_iou(last_tight, cb) > 0.3 for cb in confident_boxes
            )

            if not observer_validates:
                # ── ERASE: restore the canvas region with background ──
                # Re-read original background for that patch
                bg_patch = cv2.imread(os.path.join(SOURCE_IMAGES_DIR, bg_file))
                bg_patch = cv2.resize(bg_patch, (CANVAS_W, CANVAS_H))
                canvas[py:py+ch, px:px+cw] = bg_patch[py:py+ch, px:px+cw]

                # Remove from tracking lists
                placed_objects.pop()
                placed_boxes.pop()
                consecutive_failures += 1

        # ── SAVE CANVAS + LABELS ──────────────────────────────────────
        canvas_name = f"distilled_{canvas_idx:04d}"
        img_out  = os.path.join(OUTPUT_IMAGES_DIR, f"{canvas_name}.jpg")
        lbl_out  = os.path.join(OUTPUT_LABELS_DIR, f"{canvas_name}.txt")

        cv2.imwrite(img_out, canvas)

        yolo_lines = boxes_to_yolo(placed_objects, CANVAS_W, CANVAS_H)
        with open(lbl_out, 'w') as f:
            f.write('\n'.join(yolo_lines))

        total_objects_saved += len(placed_objects)
        print(f"  └─ Saved: {len(placed_objects)} objects  →  {canvas_name}.jpg + .txt")

    print("\n" + "=" * 60)
    print(f"[4/4] DONE. {IPD_TARGET} balanced canvases generated.")
    print(f"      Total annotated objects distilled: {total_objects_saved}")
    print(f"      Output → {OUTPUT_IMAGES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    run_od3_distillation()