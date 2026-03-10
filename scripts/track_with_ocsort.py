"""
Traffic Tracking System with YOLOv8 + OC-SORT
FIXED VERSION — 4 bugs corrected + Anti-Fragmentation + Choke Point Tripwire
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import cv2
import numpy as np
from ocsort import OCSort
import time
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# TUNING KNOBS
# ─────────────────────────────────────────────────────────────────────────────
TRIPWIRE_Y_FRACTION  = 0.38  # Moved up to 45% (Line 1 Choke Point to stop FOV leakage)
MIN_AGE_TO_COUNT     = 5
DEDUP_RADIUS         = 80
DEDUP_MEMORY_FRAMES  = 30

# FIX 4: Correct class order matching YAML:
# ['bus','cars','e-jeepney','jeepney','motorcycle','trike','trucks']
CLASS_COLORS = {
    0: (255, 0,   0  ),   # bus        – blue
    1: (0,   255, 0  ),   # cars       – green
    2: (255, 165, 0  ),   # e-jeepney  – orange
    3: (0,   255, 255),   # jeepney    – yellow
    4: (128, 0,   128),   # motorcycle – purple
    5: (0,   165, 255),   # trike      – orange-red
    6: (255, 255, 0  ),   # trucks     – cyan
}

# FIX 1 & 3: Must exactly match model.names from your YAML
ALL_CLASS_NAMES = ['bus', 'cars', 'e-jeepney', 'jeepney', 'motorcycle', 'trike', 'trucks']

# ─────────────────────────────────────────────────────────────────────────────

class TrafficTracker:
    def __init__(self, model_path, video_path, output_path, conf_threshold=0.25):
        print("=" * 60)
        print("TRAFFIC TRACKER  —  YOLOv8 + OC-SORT  (FIXED)")
        print("=" * 60)

        print(f"\n✓ Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # Print class mapping so you can verify at runtime
        print("✓ Model class mapping:")
        for i, name in self.model.names.items():
            print(f"    [{i}] {name}")

        print("✓ Initializing OC-SORT tracker")
        self.tracker = OCSort(
            det_thresh    = 0.15,
            max_age       = 150,   # TWEAK 1: Massive patience for occlusion
            min_hits      = 3,     # TWEAK 2: Quick spawning (3 frames)
            iou_threshold = 0.3,   # TWEAK 3: Stickier bounding boxes to prevent ID switches
            delta_t       = 3,
            inertia       = 0.3,
        )

        self.video_path  = video_path
        self.output_path = output_path

        self.vehicle_counts_in  = defaultdict(int)
        self.vehicle_counts_out = defaultdict(int)
        self.counted_ids        = set()
        self.track_age          = defaultdict(int)
        self.track_history      = defaultdict(list)
        self._last_positions    = {}
        self._lost_track_memo   = {}
        self.counting_line_y    = None
        self.total_frames       = 0

        print("✓ Initialization complete\n")

    # ──────────────────────────────────────────────────────────────────────────
    def process_video(self, max_frames=None, display=False):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"✗ ERROR: Could not open video: {self.video_path}")
            return

        fps     = int(cap.get(cv2.CAP_PROP_FPS))
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_f = min(total_f, max_frames)

        self.counting_line_y = int(height * TRIPWIRE_Y_FRACTION)

        print(f"Video Properties:")
        print(f"  Resolution : {width}×{height}")
        print(f"  FPS        : {fps}")
        print(f"  Frames     : {total_f}")
        print(f"  Tripwire Y : {self.counting_line_y}  ({TRIPWIRE_Y_FRACTION*100:.0f}% of height)")
        print()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_count     = 0
        fps_list        = []
        active_ids_prev = set()

        print("Processing video...")
        print("-" * 60)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break

            t0 = time.time()

            results    = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            detections = self._prepare_detections(results)
            tracks     = self.tracker.update(detections)

            active_ids = self._update_track_metadata(tracks, frame_count)

            # Expire old lost-track memories
            expired = [tid for tid, info in self._lost_track_memo.items()
                       if frame_count - info['frame'] > DEDUP_MEMORY_FRAMES]
            for tid in expired:
                del self._lost_track_memo[tid]

            for tid in active_ids_prev - active_ids:
                if tid in self._last_positions:
                    self._lost_track_memo[tid] = {
                        **self._last_positions[tid],
                        'frame': frame_count,
                    }
            active_ids_prev = active_ids

            self._check_tripwire(tracks)

            frame = self._draw_tracks(frame, tracks)
            self._draw_tripwire(frame, width)
            frame = self._draw_stats(frame, len(tracks), frame_count, fps_list)

            elapsed = time.time() - t0
            fps_list.append(1 / elapsed if elapsed > 0 else 0)

            out.write(frame)

            if display:
                cv2.imshow('Traffic Tracker (Fixed)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                avg_fps  = np.mean(fps_list[-100:])
                progress = frame_count / total_f * 100
                total_counted = sum(self.vehicle_counts_in.values())
                print(f"Frame {frame_count}/{total_f} ({progress:.1f}%) | "
                      f"Active: {len(tracks)} | Counted: {total_counted} | "
                      f"FPS: {avg_fps:.2f}")

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        self.total_frames = frame_count
        self._print_final_stats(fps_list)

    # ──────────────────────────────────────────────────────────────────────────
    def _prepare_detections(self, results):
        detections = []
        if len(results.boxes) > 0:
            boxes   = results.boxes.xyxy.cpu().numpy()
            scores  = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            for box, score, cls in zip(boxes, scores, classes):
                detections.append([*box, score, cls])
        return np.array(detections) if detections else np.empty((0, 6))

    # ──────────────────────────────────────────────────────────────────────────
    def _update_track_metadata(self, tracks, frame_count):
        active_ids = set()
        for track in tracks:
            x1, y1, x2, y2, track_id, cls, *_ = track
            track_id = int(track_id)
            cls      = int(cls)
            cx, cy   = (x1 + x2) / 2, (y1 + y2) / 2

            active_ids.add(track_id)
            self.track_age[track_id] += 1
            self._last_positions[track_id] = {'pos': (cx, cy), 'cls': cls}

            self.track_history[track_id].append((cx, cy))
            if len(self.track_history[track_id]) > 5:
                self.track_history[track_id].pop(0)

        return active_ids

    # ──────────────────────────────────────────────────────────────────────────
    def _check_tripwire(self, tracks):
        for track in tracks:
            x1, y1, x2, y2, track_id, cls, *_ = track
            track_id = int(track_id)
            cls      = int(cls)

            if self.track_age[track_id] < MIN_AGE_TO_COUNT:
                continue
            if track_id in self.counted_ids:
                continue

            history = self.track_history[track_id]
            if len(history) < 2:
                continue

            prev_y = history[-2][1]
            curr_y = history[-1][1]
            line_y = self.counting_line_y

            crossed_down = prev_y < line_y <= curr_y
            crossed_up   = prev_y > line_y >= curr_y

            if not (crossed_down or crossed_up):
                continue

            cx, cy      = history[-1]
            is_fragment = False
            for lost_id, info in list(self._lost_track_memo.items()):
                if info['cls'] != cls:
                    continue
                lx, ly = info['pos']
                if ((cx - lx)**2 + (cy - ly)**2) ** 0.5 < DEDUP_RADIUS:
                    is_fragment = True
                    del self._lost_track_memo[lost_id]
                    break

            self.counted_ids.add(track_id)

            if not is_fragment:
                class_name = self.model.names[cls]
                if crossed_down:
                    self.vehicle_counts_in[class_name] += 1
                else:
                    self.vehicle_counts_out[class_name] += 1

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_tripwire(self, frame, width):
        y = self.counting_line_y
        cv2.line(frame, (0, y), (width, y), (0, 0, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (20, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_tracks(self, frame, tracks):
        for track in tracks:
            x1, y1, x2, y2, track_id, cls, *_ = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            cls      = int(cls)

            graduated   = self.track_age[track_id] >= MIN_AGE_TO_COUNT
            color       = CLASS_COLORS.get(cls, (255, 255, 255))
            box_color   = color if graduated else (80, 80, 80)
            was_counted = track_id in self.counted_ids

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            dot_color = (0, 255, 0) if was_counted else (0, 0, 255)
            cv2.circle(frame, (cx, cy), 4, dot_color, -1)

            class_name = self.model.names[cls]
            label      = f"ID:{track_id} {class_name}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw, y1), box_color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_stats(self, frame, active_tracks, frame_count, fps_list):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (460, 340), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 35
        def txt(msg, color=(255, 255, 255), scale=0.6):
            nonlocal y
            cv2.putText(frame, msg, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            y += 28

        txt(f"Frame: {frame_count}")
        txt(f"Active Tracks: {active_tracks}")

        avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
        fps_col = (0,255,0) if avg_fps>=20 else (0,165,255) if avg_fps>=15 else (0,0,255)
        txt(f"FPS: {avg_fps:.1f}", fps_col)

        y += 8
        txt("Tripwire Counts (IN / OUT):", (0, 255, 200), 0.55)

        for cls_name in ALL_CLASS_NAMES:
            n_in  = self.vehicle_counts_in.get(cls_name, 0)
            n_out = self.vehicle_counts_out.get(cls_name, 0)
            if n_in or n_out:
                txt(f"  {cls_name}: {n_in} in / {n_out} out", scale=0.5)

        total_in  = sum(self.vehicle_counts_in.values())
        total_out = sum(self.vehicle_counts_out.values())
        y += 4
        txt(f"TOTAL: {total_in} in / {total_out} out", (0, 255, 0), 0.55)

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    def _print_final_stats(self, fps_list):
        print("\n" + "="*60)
        print("TRACKING COMPLETED")
        print("="*60)
        print(f"\n✓ Output saved : {self.output_path}")
        print(f"  Frames processed : {self.total_frames}")
        print(f"  IDs issued       : {max(self.counted_ids | {0})}")
        print(f"  Vehicles counted : {len(self.counted_ids)}")

        print(f"\n{'Class':<14} {'IN':>6} {'OUT':>6} {'TOTAL':>8}")
        print("-" * 38)
        for cls in ALL_CLASS_NAMES:
            n_in  = self.vehicle_counts_in.get(cls, 0)
            n_out = self.vehicle_counts_out.get(cls, 0)
            if n_in or n_out:
                print(f"  {cls:<12} {n_in:>6} {n_out:>6} {n_in+n_out:>8}")

        total_in  = sum(self.vehicle_counts_in.values())
        total_out = sum(self.vehicle_counts_out.values())
        print("-" * 38)
        print(f"  {'TOTAL':<12} {total_in:>6} {total_out:>6} {total_in+total_out:>8}")

        avg_fps = np.mean(fps_list)
        print(f"\n=== PERFORMANCE ===")
        print(f"Avg FPS: {avg_fps:.2f}  Min: {min(fps_list):.2f}  Max: {max(fps_list):.2f}")
        status = ("✓ Real-time"       if avg_fps >= 20 else
                  "⚠ Near real-time"  if avg_fps >= 15 else
                  "✗ Below real-time")
        print(status)
        print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    MODEL_PATH  = 'runs/detect/train_finetuned_10pct/weights/best.pt'
    VIDEO_PATH  = 'datasets/mmda_footage4/MMDA_Footage4.mp4'
    OUTPUT_PATH = 'outputs/tracking/tracked_output_final_10pct.mp4'

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model not found: {MODEL_PATH}"); return
    if not os.path.exists(VIDEO_PATH):
        print(f"✗ Video not found: {VIDEO_PATH}"); return

    tracker = TrafficTracker(
        model_path      = MODEL_PATH,
        video_path      = VIDEO_PATH,
        output_path     = OUTPUT_PATH,
        conf_threshold  = 0.25,
    )

    print("\n🎬 Processing Thesis Demo (3600 frames)")
    tracker.process_video(max_frames=3600, display=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()