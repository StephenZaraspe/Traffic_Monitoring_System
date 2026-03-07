import cv2
import os

def extract_window_frames():
    print("="*60)
    print("DATA DISTILLATION: TIME-WINDOW EXTRACTOR (MINUTE 20-22)")
    print("="*60)

    # 1. Update this to your exact video filename path!
    VIDEO_PATH = 'datasets\mmda_footage4\MMDA_Footage4.mp4' 
    OUTPUT_DIR = 'datasets/mmda_footage/extracted_frames_roxas_day_20m'
    
    # 2. Set your specific time window and target frames
    START_MINUTE = 20
    END_MINUTE = 22
    TARGET_FRAMES = 50

    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"✗ ERROR: Video not found at {VIDEO_PATH}")
        print("  Fix: Right-click MMDA_Footage4.mp4 in VS Code and select 'Copy Relative Path'")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 3. Calculate exact frames for the time window
    start_frame = START_MINUTE * 60 * fps
    end_frame = END_MINUTE * 60 * fps
    frames_in_window = end_frame - start_frame
    
    # Safety check to ensure the video is actually long enough
    if start_frame >= total_frames:
        print("✗ ERROR: Video is shorter than your start minute!")
        return
        
    print(f"\n✓ Video loaded successfully.")
    print(f"  FPS: {fps}")
    print(f"  Extracting exactly {TARGET_FRAMES} frames from Minute {START_MINUTE} to Minute {END_MINUTE}")
    
    # 4. Fast-forward directly to the 20-minute mark
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Calculate the mathematical jump for the 2-minute window
    step_size = max(1, frames_in_window // TARGET_FRAMES)
    print(f"\n✓ Extracting 1 frame every {step_size} frames within the window.")

    current_frame = start_frame
    saved_count = 0

    # Loop strictly within your start and end frames
    while cap.isOpened() and saved_count < TARGET_FRAMES and current_frame <= end_frame:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Only save the frame if it hits our calculated step size
        if (current_frame - start_frame) % step_size == 0:
            # Filenames will look like: roxas_day_20m_001.jpg
            filename = os.path.join(OUTPUT_DIR, f"roxas_day_20m_{saved_count:03d}.jpg")
            
            # Save as high-quality JPG
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"  Progress: Saved {saved_count}/{TARGET_FRAMES} frames...")
                
        current_frame += 1

    cap.release()
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"✓ Successfully saved {saved_count} frames to: {OUTPUT_DIR}")

if __name__ == '__main__':
    extract_window_frames()