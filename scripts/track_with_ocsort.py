"""
Traffic Tracking System with YOLOv8 + OC-SORT
Real-time vehicle detection and tracking
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

class TrafficTracker:
    def __init__(self, model_path, video_path, output_path, conf_threshold=0.25):
        """
        Initialize traffic tracking system
        
        Args:
            model_path: Path to trained YOLOv8 model
            video_path: Path to input video
            output_path: Path to save output video
            conf_threshold: Confidence threshold for detections
        """
        print("="*60)
        print("TRAFFIC TRACKING SYSTEM - YOLOv8 + OC-SORT")
        print("="*60)
        
        # Load YOLOv8 model
        print(f"\n✓ Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Initialize OC-SORT tracker
        print("✓ Initializing OC-SORT tracker")
        self.tracker = OCSort(
            det_thresh=0.3,
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            delta_t=3,
            inertia=0.2
        )
        
        # Video paths
        self.video_path = video_path
        self.output_path = output_path
        
        # Statistics
        self.vehicle_counts = defaultdict(int)
        self.tracked_ids = set()
        self.total_frames = 0
        
        print("✓ Initialization complete\n")
    
    def process_video(self, max_frames=None, display=False):
        """
        Process video with detection and tracking
        
        Args:
            max_frames: Maximum frames to process (None = all)
            display: Show video while processing
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"✗ ERROR: Could not open video: {self.video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Video Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print()
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        # Processing loop
        frame_count = 0
        fps_list = []
        
        print("Processing video...")
        print("-" * 60)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            start_time = time.time()
            
            # YOLOv8 detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            
            # Prepare detections for OC-SORT
            detections = self._prepare_detections(results)
            
            # OC-SORT tracking
            tracks = self.tracker.update(detections)
            
            # Draw results
            frame = self._draw_tracks(frame, tracks)
            
            # Update statistics
            self._update_stats(tracks)
            
            # Calculate FPS
            process_time = time.time() - start_time
            current_fps = 1 / process_time if process_time > 0 else 0
            fps_list.append(current_fps)
            
            # Draw statistics overlay
            frame = self._draw_stats(frame, len(tracks), frame_count, fps_list)
            
            # Write frame
            out.write(frame)
            
            # Display (optional)
            if display:
                cv2.imshow('Traffic Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            frame_count += 1
            if frame_count % 100 == 0:
                avg_fps = np.mean(fps_list[-100:])
                progress = (frame_count / total_frames) * 100
                print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) | FPS: {avg_fps:.2f}")
        
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()
        
        self.total_frames = frame_count
        
        # Print final statistics
        self._print_final_stats(fps_list)
    
    def _prepare_detections(self, results):
        """Convert YOLOv8 results to OC-SORT format"""
        detections = []
        
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                detections.append([*box, score, cls])
        
        return np.array(detections) if len(detections) > 0 else np.empty((0, 6))
    
    def _update_stats(self, tracks):
        """Update vehicle counting statistics"""
        for track in tracks:
            track_id = int(track[4])
            cls = int(track[5])
            
            if track_id not in self.tracked_ids:
                self.tracked_ids.add(track_id)
                class_name = self.model.names[cls]
                self.vehicle_counts[class_name] += 1
    
    def _draw_tracks(self, frame, tracks):
        """Draw bounding boxes and track IDs"""
        for track in tracks:
            x1, y1, x2, y2, track_id, cls, conf = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            cls = int(cls)
            
            # Generate consistent color for track ID
            color = self._get_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            class_name = self.model.names[cls]
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            
            # Label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def _draw_stats(self, frame, active_tracks, frame_count, fps_list):
        """Draw statistics overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Statistics text
        y_offset = 35
        cv2.putText(frame, f"Frame: {frame_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Active Tracks: {active_tracks}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        avg_fps = np.mean(fps_list[-30:]) if len(fps_list) > 0 else 0
        fps_color = (0, 255, 0) if avg_fps >= 20 else (0, 165, 255) if avg_fps >= 15 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        y_offset += 35
        cv2.putText(frame, "Vehicle Counts:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for class_name, count in sorted(self.vehicle_counts.items()):
            y_offset += 25
            cv2.putText(frame, f"  {class_name}: {count}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def _get_color(self, track_id):
        """Generate consistent color for each track ID"""
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def _print_final_stats(self, fps_list):
        """Print final statistics"""
        print("\n" + "="*60)
        print("TRACKING COMPLETED")
        print("="*60)
        
        print(f"\n✓ Output saved: {self.output_path}")
        print(f"\nTotal Frames Processed: {self.total_frames}")
        print(f"Total Unique Vehicles: {len(self.tracked_ids)}")
        
        print("\nVehicle Breakdown:")
        for class_name, count in sorted(self.vehicle_counts.items()):
            print(f"  {class_name}: {count}")
        
        avg_fps = np.mean(fps_list)
        min_fps = min(fps_list)
        max_fps = max(fps_list)
        
        print(f"\n=== PERFORMANCE METRICS ===")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Min FPS: {min_fps:.2f}")
        print(f"Max FPS: {max_fps:.2f}")
        
        if avg_fps >= 20:
            print("✓ Real-time capable (>= 20 FPS)")
        elif avg_fps >= 15:
            print("⚠ Near real-time (15-20 FPS)")
        else:
            print("✗ Below real-time (< 15 FPS)")
        
        print("\n" + "="*60)


def main():
    # Configuration
    MODEL_PATH = 'runs/detect/models/yolov8_baseline/run13/weights/best.pt'
    VIDEO_PATH = 'datasets/mmda_footage/raw_videos/MMDA_Footage.mp4'
    OUTPUT_PATH = 'outputs/tracking/tracked_output.mp4'
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"✗ ERROR: Model not found: {MODEL_PATH}")
        print("Please train the model first: python scripts/train_yolov8_full.py")
        return
    
    if not os.path.exists(VIDEO_PATH):
        print(f"✗ ERROR: Video not found: {VIDEO_PATH}")
        print("Please place your MMDA video in: datasets/mmda_footage/raw_videos/")
        return
    
    # Initialize tracker
    tracker = TrafficTracker(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        conf_threshold=0.25
    )
    
    # Process video
    # Set max_frames=1000 for testing, None for full video
    tracker.process_video(max_frames=None, display=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTracking interrupted by user")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()