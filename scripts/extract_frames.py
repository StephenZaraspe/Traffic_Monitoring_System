import cv2
import os

def extract_evenly_spaced_frames():
    print("="*60)
    print("DATA DISTILLATION: SEED FRAME EXTRACTOR")
    print("="*60)

    # 1. Update this to your actual MMDA video filename
    VIDEO_PATH = 'datasets/mmda_footage/raw_videos/MMDA_Footage.mp4' 
    OUTPUT_DIR = 'datasets/mmda_footage/extracted_frames'
    TARGET_FRAMES = 200

    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"✗ ERROR: Video not found at {VIDEO_PATH}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\n✓ Video loaded successfully.")
    print(f"  Total Frames: {total_frames}")
    print(f"  FPS: {fps}")
    
    # Calculate the exact mathematical jump to get exactly 200 frames
    step_size = max(1, total_frames // TARGET_FRAMES)
    print(f"\n✓ Extracting 1 frame every {step_size} frames to hit target.")

    frame_count = 0
    saved_count = 0

    while cap.isOpened() and saved_count < TARGET_FRAMES:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Only save the frame if it hits our calculated step size
        if frame_count % step_size == 0:
            # Format filename with padding (e.g., frame_001.jpg)
            filename = os.path.join(OUTPUT_DIR, f"frame_{saved_count:03d}.jpg")
            
            # Save as high-quality JPG
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            saved_count += 1
            
            if saved_count % 50 == 0:
                print(f"  Progress: Saved {saved_count}/{TARGET_FRAMES} frames...")
                
        frame_count += 1

    cap.release()
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"✓ Successfully saved {saved_count} frames to: {OUTPUT_DIR}")

if __name__ == '__main__':
    extract_evenly_spaced_frames()