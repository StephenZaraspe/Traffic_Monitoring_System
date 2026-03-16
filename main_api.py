"""
Traffic Monitoring API — FastAPI Backend (Pre-Process Edition)
- POST /api/upload/video  — upload video, tracker runs fully offline
- GET  /api/status        — progress % + state (idle/processing/done/error)
- GET  /api/video/output  — serve the finished annotated .mp4
- GET  /api/traffic/summary, /api/traffic/log — counts from DB
- WS   /ws/traffic        — live detection events during processing

Run:
    uvicorn main_api:app --host 0.0.0.0 --port 8080
"""

import asyncio
import queue
import threading
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ─── PATHS ────────────────────────────────────────────────────────────────────
MODEL_PATH = "runs/detect/train_finetuned_10pct/weights/best.pt"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs/tracking"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── DATABASE ─────────────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///./traffic.db"
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase): pass

class VehicleEvent(Base):
    __tablename__ = "vehicle_events"
    id           = Column(Integer, primary_key=True, index=True)
    vehicle_type = Column(String)
    direction    = Column(String)
    confidence   = Column(Float)
    timestamp    = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ─── SHARED STATE ─────────────────────────────────────────────────────────────
_event_queue: queue.Queue = queue.Queue()

# ─── LIVE FRAME BUFFER (webcam MJPEG) ────────────────────────────────────────
_frame_lock  = threading.Lock()
_frame_data: bytes = b""

def _set_frame(jpeg_bytes: bytes):
    global _frame_data
    with _frame_lock:
        _frame_data = jpeg_bytes

_tracker_status = {
    "state":       "idle",     # idle | processing | reencoding | done | error | webcam
    "video":       None,       # original filename
    "output":      None,       # path to finished .mp4
    "progress":    0,          # 0-100
    "message":     "No video uploaded yet",
}
_webcam_active = False
_tracker_lock      = threading.Lock()
_tracker_stop_event = threading.Event()
_tracker_thread: Optional[threading.Thread] = None


def _reencode_h264(input_path: str, output_path: str) -> bool:
    """Re-encode mp4v to H.264 using moviepy (bundles its own ffmpeg — no system install needed)."""
    # Try moviepy first (pip install moviepy)
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        clip = VideoFileClip(input_path)
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio=False,
            logger=None,
            fps=clip.fps,
        )
        clip.close()
        os.replace(output_path, input_path)
        print(f"✓ Re-encoded to H.264 via moviepy: {input_path}")
        return True
    except ImportError:
        pass  # moviepy not installed, try system ffmpeg
    except Exception as e:
        print(f"⚠ moviepy error: {e}")

    # Fallback: system ffmpeg
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-vcodec", "libx264", "-crf", "23",
             "-preset", "fast", "-an", output_path],
            capture_output=True, timeout=300
        )
        if result.returncode == 0:
            os.replace(output_path, input_path)
            print(f"✓ Re-encoded to H.264 via ffmpeg: {input_path}")
            return True
        else:
            print(f"⚠ ffmpeg failed: {result.stderr.decode()[:200]}")
    except FileNotFoundError:
        print("⚠ Neither moviepy nor ffmpeg found.")
        print("  Fix: pip install moviepy")
    except Exception as e:
        print(f"⚠ ffmpeg error: {e}")
    return False


def _set_progress(pct: int):
    """Called by tracker every 30 frames to update progress."""
    with _tracker_lock:
        if pct == 100:
            output = _tracker_status.get("output")
            _tracker_status["state"]    = "reencoding"
            _tracker_status["progress"] = 100
            _tracker_status["message"]  = "Re-encoding to H.264 for browser playback..."

            # Re-encode in a thread so we don't block
            def _do_reencode():
                if output and os.path.exists(output):
                    tmp = output.replace(".mp4", "_h264.mp4")
                    _reencode_h264(output, tmp)
                with _tracker_lock:
                    _tracker_status["state"]   = "done"
                    _tracker_status["message"] = "Processing complete — video ready"
                print("✓ Video ready to play")
            threading.Thread(target=_do_reencode, daemon=True).start()

        elif pct == -1:
            _tracker_status["state"]   = "error"
            _tracker_status["message"] = "Tracker error — check terminal"
        else:
            _tracker_status["progress"] = pct
            _tracker_status["message"]  = f"Processing... {pct}%"


def _start_tracker(video_path: str, output_path: str):
    global _tracker_stop_event, _tracker_thread

    with _tracker_lock:
        # Stop any running tracker
        _tracker_stop_event.set()
        if _tracker_thread and _tracker_thread.is_alive():
            _tracker_thread.join(timeout=5)

        _tracker_stop_event = threading.Event()
        stop_ev = _tracker_stop_event

        _tracker_status["state"]    = "processing"
        _tracker_status["progress"] = 0
        _tracker_status["video"]    = os.path.basename(video_path)
        _tracker_status["output"]   = output_path
        _tracker_status["message"]  = "Starting tracker..."

        # Clear DB for fresh session
        db = SessionLocal()
        try:
            db.query(VehicleEvent).delete()
            db.commit()
        finally:
            db.close()

    import sys
    sys.path.append("scripts")
    try:
        from tracker import TrafficTracker
    except ModuleNotFoundError:
        from scripts.track_with_ocsort import TrafficTracker

    tracker = TrafficTracker(
        model_path     = MODEL_PATH,
        video_path     = video_path,
        output_path    = output_path,
        conf_threshold = 0.25,
        event_sink     = _event_queue,
        stop_event     = stop_ev,
        progress_sink  = _set_progress,
    )

    def _run():
        try:
            tracker.process_video(display=False)
        except Exception as e:
            print(f"✗ Tracker error: {e}")
            _set_progress(-1)

    _tracker_thread = threading.Thread(target=_run, daemon=True, name="TrackerThread")
    _tracker_thread.start()
    print(f"✓ Tracker started: {video_path}")


# ─── WEBSOCKET MANAGER ────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self): self.active: list[WebSocket] = []
    async def connect(self, ws):    await ws.accept(); self.active.append(ws)
    def  disconnect(self, ws):      self.active.remove(ws) if ws in self.active else None
    async def broadcast(self, msg):
        data, dead = json.dumps(msg), []
        for ws in self.active:
            try:    await ws.send_text(data)
            except: dead.append(ws)
        for ws in dead: self.active.remove(ws)

manager = ConnectionManager()

# ─── LIFESPAN ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    async def drain_events():
        while True:
            try:
                for _ in range(20):
                    if _event_queue.empty(): break
                    event = _event_queue.get_nowait()
                    db = SessionLocal()
                    try:
                        db.add(VehicleEvent(
                            vehicle_type = event["vehicleType"],
                            direction    = event["direction"],
                            confidence   = event.get("confidence", 0.0),
                            timestamp    = datetime.fromisoformat(event["timestamp"]),
                        ))
                        db.commit()
                    finally:
                        db.close()
                    await manager.broadcast(event)
            except Exception as e:
                print(f"[drain_events] {e}")
            await asyncio.sleep(0.05)

    drain_task = asyncio.create_task(drain_events())
    yield
    drain_task.cancel()
    _tracker_stop_event.set()

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Traffic Monitor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"status": "Traffic Monitor API running", "docs": "/docs"}

# ─── MJPEG LIVE FEED (webcam mode) ───────────────────────────────────────────
async def _gen_frames():
    last_frame = None
    while True:
        with _frame_lock:
            frame = _frame_data
        if frame and frame != last_frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            last_frame = frame
        await asyncio.sleep(0.01)
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

# ─── WEBCAM ENDPOINTS ─────────────────────────────────────────────────────────
@app.post("/api/webcam/start")
async def webcam_start(camera_index: int = 0):
    """Start live webcam tracking via OpenCV on the backend."""
    global _webcam_active

    if not os.path.exists(MODEL_PATH):
        return {"error": f"Model not found at {MODEL_PATH}"}

    import sys
    sys.path.append("scripts")
    try:
        from tracker import TrafficTracker
    except ModuleNotFoundError:
        from scripts.track_with_ocsort import TrafficTracker

    # Reset counts
    db = SessionLocal()
    try:
        db.query(VehicleEvent).delete()
        db.commit()
    finally:
        db.close()

    # Clear frame buffer
    global _frame_data
    with _frame_lock:
        _frame_data = b""

    _webcam_active = True
    with _tracker_lock:
        _tracker_status["state"]   = "webcam"
        _tracker_status["video"]   = f"webcam:{camera_index}"
        _tracker_status["message"] = "Live webcam feed active"

    stop_ev = threading.Event()

    # Store stop event so /webcam/stop can signal it
    global _tracker_stop_event, _tracker_thread
    _tracker_stop_event.set()  # stop any existing tracker
    if _tracker_thread and _tracker_thread.is_alive():
        _tracker_thread.join(timeout=3)
    _tracker_stop_event = stop_ev

    tracker = TrafficTracker(
        model_path     = MODEL_PATH,
        video_path     = camera_index,   # 0 = default webcam
        output_path    = os.path.join(OUTPUT_DIR, "webcam_tracked.mp4"),
        conf_threshold = 0.25,
        event_sink     = _event_queue,
        stop_event     = stop_ev,
        frame_sink     = _set_frame,     # push annotated frames to MJPEG buffer
    )

    def _run():
        global _webcam_active
        try:
            tracker.process_video(display=True)   # opens native OpenCV window
        finally:
            _webcam_active = False
            with _tracker_lock:
                _tracker_status["state"]   = "idle"
                _tracker_status["message"] = "Webcam stopped"
            print("◉ Webcam tracker stopped")

    _tracker_thread = threading.Thread(target=_run, daemon=True, name="WebcamTracker")
    _tracker_thread.start()

    return {"status": "started", "camera": camera_index, "feed": "/video_feed"}

@app.post("/api/webcam/stop")
async def webcam_stop():
    """Stop the live webcam tracker."""
    global _webcam_active
    _tracker_stop_event.set()
    _webcam_active = False
    with _tracker_lock:
        _tracker_status["state"]   = "idle"
        _tracker_status["message"] = "Webcam stopped"
    return {"status": "stopped"}

@app.get("/api/status")
def get_status():
    with _tracker_lock:
        return dict(_tracker_status)

@app.get("/api/video/output")
def get_output_video():
    """Serve the finished annotated .mp4 to the browser."""
    with _tracker_lock:
        path = _tracker_status.get("output")
    if not path or not os.path.exists(path):
        return {"error": "No output video available yet"}
    return FileResponse(
        path,
        media_type   = "video/mp4",
        filename     = os.path.basename(path),
        headers      = {"Accept-Ranges": "bytes"},
    )

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {"error": f"Unsupported format: {ext}"}

    save_path  = os.path.join(UPLOAD_DIR, file.filename)
    chunk_size = 1024 * 1024
    written    = 0
    with open(save_path, "wb") as f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk: break
            f.write(chunk)
            written += len(chunk)
            await asyncio.sleep(0)

    file_size_mb = written / (1024 * 1024)
    print(f"✓ Uploaded: {file.filename} ({file_size_mb:.1f} MB)")

    if not os.path.exists(MODEL_PATH):
        return {"error": f"Model not found at {MODEL_PATH}"}

    output_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(file.filename)[0] + "_tracked.mp4",
    )

    threading.Thread(
        target=_start_tracker,
        args=(save_path, output_path),
        daemon=True,
    ).start()

    return {
        "status":   "processing",
        "filename": file.filename,
        "size_mb":  round(file_size_mb, 1),
    }

@app.get("/api/traffic/summary")
def get_summary():
    db = SessionLocal()
    try:
        events = db.query(VehicleEvent).all()
        counts = {}; in_c = out_c = 0
        for e in events:
            counts[e.vehicle_type] = counts.get(e.vehicle_type, 0) + 1
            if e.direction == "IN":  in_c  += 1
            if e.direction == "OUT": out_c += 1
        return {"total": len(events), "inbound": in_c, "outbound": out_c, "counts": counts}
    finally:
        db.close()

@app.get("/api/traffic/log")
def get_log(limit: int = 16):
    db = SessionLocal()
    try:
        events = db.query(VehicleEvent).order_by(VehicleEvent.id.desc()).limit(limit).all()
        return [{"vehicleType": e.vehicle_type, "direction": e.direction,
                 "confidence": e.confidence, "timestamp": e.timestamp.isoformat()}
                for e in events]
    finally:
        db.close()

@app.websocket("/ws/traffic")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await asyncio.sleep(30)
            await ws.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        manager.disconnect(ws)