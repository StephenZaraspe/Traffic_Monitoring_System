cat > README.md << 'EOF'
# MMDA Traffic Intelligence System
**YOLOv8s + OC-SORT real-time vehicle counter — TIP Manila BS Computer Science Thesis 2025**

Counts 7 Philippine vehicle classes (cars, motorcycle, jeepney, e-jeepney, trike, bus, trucks)
from CCTV footage or live webcam using a custom-trained YOLOv8s model and OC-SORT tracking.

---

## Features
- Upload CCTV video → GPU processes → annotated playback in browser
- Live webcam mode with native OpenCV window + browser MJPEG stream
- FastAPI backend with WebSocket live detection events
- React dashboard with per-class counts, detection log, AP ablation panel

---

## Requirements
- Python 3.10+
- CUDA-capable NVIDIA GPU (tested on RTX 3070 / 3060)
- Node.js 18+ (for dashboard dev server)
- Windows or Linux

---

## Setup

### 1. Clone
```bash
git clone https://github.com/StephenZaraspe/Traffic_Monitoring_System.git
cd Traffic_Monitoring_System
```

### 2. Python environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install PyTorch with CUDA 12.1 (adjust if your CUDA version differs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install everything else
pip install -r requirements.txt
```

### 3. Model weights
The trained weights are **not included** in this repo (file size).
Download `best.pt` from the link below and place it at:
```
runs/detect/train_finetuned_10pct/weights/best.pt
```
**Download:** [Google Drive link — add yours here]

### 4. Run the backend
```bash
uvicorn main_api:app --host 0.0.0.0 --port 8080
```
Then open `http://localhost:8080/docs` to verify the API is running.

### 5. Open the dashboard

**Option A — Production build (recommended)**
```bash
cd traffic-dashboard
npm install
npm run build
# Then serve the build folder, or just open http://localhost:8080 if backend serves it
```

**Option B — Dev server**
```bash
cd traffic-dashboard
npm install
npm run dev
# Opens at http://localhost:3000
```

---

## Usage

### Video Upload mode
1. Open the dashboard
2. Enter your backend URL (`http://localhost:8080`)
3. Drag and drop a `.mp4` / `.avi` / `.mov` video
4. Click **Upload and Process** — GPU runs inference, progress bar updates live
5. Annotated video plays automatically when done

### Webcam mode
1. Click **Use Webcam** in the dashboard
2. Backend opens your camera via OpenCV
3. Native OpenCV window shows raw annotated feed
4. Browser MJPEG stream shows the same feed at `/video_feed`

---

## Project Structure
```
Traffic_Monitoring_System/
├── main_api.py                  # FastAPI backend
├── scripts/
│   └── track_with_ocsort.py     # YOLOv8s + OC-SORT tracker
├── traffic-dashboard/           # React frontend
├── requirements.txt
└── runs/detect/.../best.pt      # Model weights (not in repo — download separately)
```

---

## Model Performance (Finetuned — Distilled + 10% Domain Bridge)

| Class | AP |
|---|---|
| Cars | 98.6% |
| Motorcycle | 97.0% |
| E-Jeepney | 100% |
| Jeepney | 91.2% |
| Trike | 84.3% |
| Bus | 93.5% |
| Trucks | 95.1% |
| **Overall CER** | **9.69%** |

---

## Tech Stack
- [YOLOv8s](https://github.com/ultralytics/ultralytics) — object detection
- [OC-SORT](https://github.com/noahcao/OC_SORT) — multi-object tracking
- [FastAPI](https://fastapi.tiangolo.com/) — backend API
- [React](https://react.dev/) — dashboard frontend
- OD3 Dataset Distillation + Minority Quota Synthesis — custom training pipeline

---

## Citation
If you use this project, please cite:
```
TIP Manila — BS Computer Science Thesis 2025
MMDA Traffic Intelligence System
Authors: [Your full name + thesis group]
```
EOF