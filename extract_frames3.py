import cv2
import os

def extract_window_frames():
    print("="*60)
    print("DATA DISTILLATION: ROXAS DAYTIME (200 FRAMES)")
    print("="*60)

    # 1. Updated perfectly to match your VS Code file explorer screenshot
    VIDEO_PATH = 'datasets/mmda_footage/raw_videos/EastWoods_Daytime2.mp4' 
    OUTPUT_DIR = 'datasets/mmda_footage/extracted_frames_Eastwoods_Daytime_batch2'
    
    # 2. Extracting exactly 200 frames spread across a 10-minute window
    # (Change END_MINUTE if your video is shorter than 10 minutes!)
    START_MINUTE = 0
    END_MINUTE = 10
    TARGET_FRAMES = 50

    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"✗ ERROR: Video not found at {VIDEO_PATH}")
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
    
    # Safety check
    if start_frame >= total_frames:
        print("✗ ERROR: Video is shorter than your start minute!")
        return
        
    # Prevent the end_frame from exceeding the actual video length
    if end_frame > total_frames:
        end_frame = total_frames
        frames_in_window = end_frame - start_frame
        print("⚠ WARNING: Video is shorter than END_MINUTE. Adjusting to end of video.")

    print(f"\n✓ Video loaded successfully.")
    print(f"  FPS: {fps}")
    print(f"  Extracting exactly {TARGET_FRAMES} frames from Minute {START_MINUTE} to Minute {END_MINUTE}")
    
    # 4. Fast-forward to the start time
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Calculate the mathematical jump
    step_size = max(1, frames_in_window // TARGET_FRAMES)
    print(f"\n✓ Extracting 1 frame every {step_size} frames.")

    current_frame = start_frame
    saved_count = 0

    # Loop strictly within your start and end frames
    while cap.isOpened() and saved_count < TARGET_FRAMES and current_frame <= end_frame:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Only save the frame if it hits our calculated step size
        if (current_frame - start_frame) % step_size == 0:
            filename = os.path.join(OUTPUT_DIR, f"roxas_day_batch2_{saved_count:03d}.jpg")
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            saved_count += 1
            
            if saved_count % 25 == 0:
                print(f"  Progress: Saved {saved_count}/{TARGET_FRAMES} frames...")
                
        current_frame += 1

    cap.release()
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"✓ Successfully saved {saved_count} frames to: {OUTPUT_DIR}")

if __name__ == '__main__':
    extract_window_frames()