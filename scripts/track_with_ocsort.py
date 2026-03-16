"""
Traffic Tracking System — YOLOv8 + OC-SORT
Pre-process mode: runs inference fully offline, saves annotated .mp4.
Reports progress via progress_sink callback so the dashboard can show a bar.
"""

import threading
import queue
import requests
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import cv2
import numpy as np
from ocsort import OCSort
import time
from collections import defaultdict

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_URL                   = "http://localhost:8080"
TRIPWIRE_Y_FRACTION       = 0.59
MIN_AGE_TO_COUNT          = 5
DEDUP_RADIUS              = 80
DEDUP_MEMORY_FRAMES       = 30

CLASS_COLORS = {
    0: (255, 0,   0  ),   # bus
    1: (0,   255, 0  ),   # cars
    2: (255, 165, 0  ),   # e-jeepney
    3: (0,   255, 255),   # jeepney
    4: (128, 0,   128),   # motorcycle
    5: (0,   165, 255),   # trike
    6: (255, 255, 0  ),   # trucks
}
ALL_CLASS_NAMES = ['bus', 'cars', 'e-jeepney', 'jeepney', 'motorcycle', 'trike', 'trucks']


class TrafficTracker:

    def __init__(
        self,
        model_path,
        video_path,
        output_path,
        conf_threshold  = 0.25,
        event_sink      = None,   # queue.Queue for detection events
        stop_event      = None,   # threading.Event for graceful stop
        progress_sink   = None,   # callable(pct: int) — called every 30 frames
        frame_sink      = None,   # callable(jpeg_bytes) — for live webcam streaming
    ):
        print(f"\n{'='*60}")
        print("TRAFFIC TRACKER  [PRE-PROCESS]  —  YOLOv8 + OC-SORT")
        print(f"{'='*60}")

        self.event_sink    = event_sink
        self.stop_event    = stop_event or threading.Event()
        self.progress_sink = progress_sink
        self.frame_sink    = frame_sink

        print(f"✓ Loading model : {model_path}")
        self.model          = YOLO(model_path)
        self.model.to("cuda")   # RTX 3070 — ~60fps inference
        self.conf_threshold = conf_threshold

        print("✓ Initialising OC-SORT")
        self.tracker = OCSort(
            det_thresh=0.15, max_age=120, min_hits=2,
            iou_threshold=0.10, delta_t=3, inertia=0.4,
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

        print("✓ Ready\n")

    # ─────────────────────────────────────────────────────────────────────────
    def _push_event(self, class_name, direction, confidence):
        payload = {
            "vehicleType": class_name,
            "direction":   direction,
            "timestamp":   datetime.now().isoformat(),
            "confidence":  float(confidence),
        }
        if self.event_sink is not None:
            self.event_sink.put(payload)

    # ─────────────────────────────────────────────────────────────────────────
    def _push_frame(self, frame):
        """Send annotated frame to shared memory for MJPEG streaming (webcam mode)."""
        if not self.frame_sink:
            return
        is_webcam = (str(self.video_path) == "0" or self.video_path == 0)
        # Webcam: full 854x480, video file: skip every other frame
        small = cv2.resize(frame, (854, 480))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        self.frame_sink(buf.tobytes())

    # ─────────────────────────────────────────────────────────────────────────
    def process_video(self, max_frames=None, display=False):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"✗ Cannot open: {self.video_path}")

        # Force 16:9 resolution for webcam
        is_webcam = (str(self.video_path) == "0" or self.video_path == 0)
        if is_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print("◉ Webcam set to 1280x720 @ 30fps")              
            

        fps     = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Skip first minute for file videos; never skip for webcam (source=0)
        is_webcam = (str(self.video_path) == "0" or self.video_path == 0)
        if is_webcam:
            start_frame = 0
            print("◉ Webcam mode — starting from frame 0 (no skip)")
        else:
            start_frame = int(1 * 60 * fps)
            if start_frame >= total_f - (10 * fps):
                start_frame = 0
                print("▶ Video too short to skip — starting from frame 0")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                print(f"⏩ Skipping to 1-min mark (frame {start_frame})")

        frames_to_process = min(total_f - start_frame, max_frames) if max_frames else (total_f - start_frame)
        frames_to_process = max(frames_to_process, 1)

        self.counting_line_y = int(height * TRIPWIRE_Y_FRACTION)
        print(f"Video : {width}x{height} @ {fps}fps  |  Frames to process: {frames_to_process}")
        print(f"Tripwire Y : {self.counting_line_y}  ({TRIPWIRE_Y_FRACTION*100:.0f}% of height)")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # saved raw; api.py re-encodes to H264 via ffmpeg
        out    = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_count     = 0
        fps_list        = []
        active_ids_prev = set()
        last_pct        = -1

        print("Processing...\n" + "-"*50)

        while cap.isOpened():
            if self.stop_event.is_set():
                print("⏹  Tracker stopped")
                break

            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break

            t0 = time.time()

            results    = self.model(frame, conf=self.conf_threshold, verbose=False, device=0)[0]
            detections = self._prepare_detections(results)
            tracks     = self.tracker.update(detections)

            active_ids = self._update_track_metadata(tracks, frame_count)

            expired = [tid for tid, info in self._lost_track_memo.items()
                       if frame_count - info['frame'] > DEDUP_MEMORY_FRAMES]
            for tid in expired:
                del self._lost_track_memo[tid]

            for tid in active_ids_prev - active_ids:
                if tid in self._last_positions:
                    self._lost_track_memo[tid] = {**self._last_positions[tid], 'frame': frame_count}
            active_ids_prev = active_ids

            self._check_tripwire(tracks)

            frame = self._draw_tracks(frame, tracks)
            self._draw_tripwire(frame, width)
            frame = self._draw_stats(frame, len(tracks), frame_count, fps_list)

            elapsed = time.time() - t0
            fps_list.append(1 / elapsed if elapsed > 0 else 0)

            # Only write to file for pre-recorded videos, not webcam
            is_webcam = (str(self.video_path) == "0" or self.video_path == 0)
            if not is_webcam:
                out.write(frame)

            # Stream annotated frame for live webcam mode
            if is_webcam and self.frame_sink:
                self._push_frame(frame)

            frame_count += 1

            # Report progress every 30 frames (video files only; webcam runs forever)
            is_webcam = (str(self.video_path) == "0" or self.video_path == 0)
            if self.progress_sink and not is_webcam and frame_count % 30 == 0:
                pct = min(99, int((frame_count / frames_to_process) * 100))
                if pct != last_pct:
                    self.progress_sink(pct)
                    last_pct = pct

            if frame_count % 150 == 0:
                avg_fps = np.mean(fps_list[-150:])
                pct     = int((frame_count / frames_to_process) * 100)
                total_c = sum(self.vehicle_counts_in.values())
                print(f"[{pct:3d}%] Frame {frame_count}/{frames_to_process} | "
                      f"Tracks: {len(tracks)} | Counted: {total_c} | FPS: {avg_fps:.1f}")

            if display:
                cv2.imshow('Traffic Tracker', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        # Signal 100% complete
        if self.progress_sink:
            self.progress_sink(100)

        self.total_frames = frame_count
        self._print_final_stats(fps_list)

    # ─────────────────────────────────────────────────────────────────────────
    def _prepare_detections(self, results):
        dets = []
        if len(results.boxes) > 0:
            for box, score, cls in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy(),
            ):
                dets.append([*box, score, cls])
        return np.array(dets) if dets else np.empty((0, 6))

    def _update_track_metadata(self, tracks, frame_count):
        active_ids = set()
        for track in tracks:
            x1, y1, x2, y2, track_id, cls, *_ = track
            track_id = int(track_id); cls = int(cls)
            cx, cy   = (x1+x2)/2, (y1+y2)/2
            active_ids.add(track_id)
            self.track_age[track_id] += 1
            self._last_positions[track_id] = {'pos': (cx, cy), 'cls': cls}
            self.track_history[track_id].append((cx, cy))
            if len(self.track_history[track_id]) > 5:
                self.track_history[track_id].pop(0)
        return active_ids

    def _check_tripwire(self, tracks):
        for track in tracks:
            x1, y1, x2, y2, track_id, cls, score = track[:7]
            track_id = int(track_id); cls = int(cls)
            if self.track_age[track_id] < MIN_AGE_TO_COUNT: continue
            if track_id in self.counted_ids: continue
            history = self.track_history[track_id]
            if len(history) < 2: continue
            prev_y, curr_y = history[-2][1], history[-1][1]
            line_y         = self.counting_line_y
            crossed_down   = prev_y < line_y <= curr_y
            crossed_up     = prev_y > line_y >= curr_y
            if not (crossed_down or crossed_up): continue
            cx, cy      = history[-1]
            is_fragment = False
            for lost_id, info in list(self._lost_track_memo.items()):
                if info['cls'] != cls: continue
                lx, ly = info['pos']
                if ((cx-lx)**2 + (cy-ly)**2)**0.5 < DEDUP_RADIUS:
                    is_fragment = True
                    del self._lost_track_memo[lost_id]; break
            self.counted_ids.add(track_id)
            if not is_fragment:
                class_name = self.model.names[cls]
                direction  = "IN" if crossed_down else "OUT"
                if crossed_down: self.vehicle_counts_in[class_name]  += 1
                else:            self.vehicle_counts_out[class_name] += 1
                self._push_event(class_name, direction, score)

    def _draw_tripwire(self, frame, width):
        y = self.counting_line_y
        cv2.line(frame, (0, y), (width, y), (0, 0, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (20, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _draw_tracks(self, frame, tracks):
        for track in tracks:
            x1, y1, x2, y2, track_id, cls, *_ = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id); cls = int(cls)
            color   = CLASS_COLORS.get(cls, (255, 255, 255))
            counted = track_id in self.counted_ids
            aged    = self.track_age[track_id] >= MIN_AGE_TO_COUNT
            cx, cy  = int((x1+x2)/2), int((y1+y2)/2)

            if counted:
                # Full box + compact label for counted vehicles
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                label     = f"{self.model.names[cls]} #{track_id}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1-lh-8), (x1+lw+4, y1), color, -1)
                cv2.putText(frame, label, (x1+2, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            elif aged:
                # Thin dim box only — no label
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              tuple(int(c*0.4) for c in color), 1)
            # Too young: draw nothing
        return frame

    def _draw_stats(self, frame, active_tracks, frame_count, fps_list):
        is_webcam = (str(self.video_path) == "0" or self.video_path == 0)
        h, w = frame.shape[:2]
        # Scale overlay to frame size
        box_w = min(380, int(w * 0.45))
        box_h = min(260, int(h * 0.55))

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (box_w, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        scale = 0.45 if is_webcam else 0.55
        gap   = 20  if is_webcam else 24

        y = 28
        def txt(msg, color=(255, 255, 255)):
            nonlocal y
            cv2.putText(frame, msg, (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
            y += gap

        avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
        txt(f"Frame: {frame_count}  |  Tracks: {active_tracks}")
        txt(f"FPS: {avg_fps:.1f}",
            (0,255,0) if avg_fps>=20 else (0,165,255) if avg_fps>=15 else (0,0,255))
        y += 4
        txt("Tripwire Counts (IN / OUT):", (0, 255, 200))
        for cn in ALL_CLASS_NAMES:
            ni = self.vehicle_counts_in.get(cn, 0)
            no = self.vehicle_counts_out.get(cn, 0)
            if ni or no:
                txt(f"  {cn}: {ni} in / {no} out")
        y += 4
        ti = sum(self.vehicle_counts_in.values())
        to = sum(self.vehicle_counts_out.values())
        txt(f"TOTAL: {ti} in / {to} out", (0, 255, 0))
        return frame

    def _print_final_stats(self, fps_list):
        print("\n" + "="*60 + "\nTRACKING COMPLETED\n" + "="*60)
        print(f"Frames: {self.total_frames}  |  Vehicles: {len(self.counted_ids)}")
        print(f"\n{'Class':<14}{'IN':>6}{'OUT':>6}{'TOTAL':>8}")
        print("-"*38)
        for cn in ALL_CLASS_NAMES:
            ni, no = self.vehicle_counts_in.get(cn,0), self.vehicle_counts_out.get(cn,0)
            if ni or no: print(f"  {cn:<12}{ni:>6}{no:>6}{ni+no:>8}")
        ti = sum(self.vehicle_counts_in.values())
        to = sum(self.vehicle_counts_out.values())
        print("-"*38)
        print(f"  {'TOTAL':<12}{ti:>6}{to:>6}{ti+to:>8}")
        if fps_list:
            avg = np.mean(fps_list)
            print(f"\nAvg FPS: {avg:.2f}")
        print("="*60)


# ─── STANDALONE ───────────────────────────────────────────────────────────────
def main():
    MODEL_PATH  = 'runs/detect/train_finetuned_10pct/weights/best.pt'
    VIDEO_PATH  = 'datasets/mmda_footage/raw_videos/EastWoods_Daytime2.mp4'
    OUTPUT_PATH = 'outputs/tracking/eastwoods_demo.mp4'
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH): print(f"✗ Model not found: {MODEL_PATH}"); return
    if not os.path.exists(VIDEO_PATH): print(f"✗ Video not found: {VIDEO_PATH}"); return
    TrafficTracker(
        model_path=MODEL_PATH, video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH, conf_threshold=0.25,
    ).process_video(display=False)

if __name__ == '__main__':
    try: main()
    except KeyboardInterrupt: print("\nInterrupted")
    except Exception as e:
        import traceback; traceback.print_exc()