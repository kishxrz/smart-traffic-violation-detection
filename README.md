# Smart Traffic Intelligence & Violation Detection System

A production-quality Computer Vision + Deep Learning system that analyses traffic camera footage to detect vehicles, track their movement, and identify traffic violations.

> **Portfolio project demonstrating:** Python · OpenCV · NumPy · PyTorch · YOLOv8 · Object Tracking · Computer Vision Geometry · FastAPI · Next.js · Docker

---

## Problem Statement

Traffic law enforcement faces two fundamental challenges:
1. **Scale** — manual monitoring of thousands of cameras is not feasible.
2. **Evidence** — violations must be captured with timestamped, annotated evidence.

This system addresses both by building an automated pipeline from raw camera input to structured violation records with evidence images.

---

## Architecture

```
Traffic Video / Image
        ↓
OpenCV Video Processing (video_processor.py)
        ↓
Frame Preprocessing — CLAHE, Gaussian blur, resize (preprocessing.py)
        ↓
YOLO Object Detection — cars, motorcycles, persons, traffic lights (yolo_detector.py)
        ↓
IoU + Hungarian Object Tracking — persistent vehicle IDs (object_tracker.py)
        ↓
Traffic Scene Analysis — ROI, perspective, stop-line (roi.py, perspective.py)
        ↓
Violation Engine — rule-based per violation type (engine.py)
        ↓
Evidence Generation — annotated frames (evidence_generator.py)
        ↓
FastAPI REST API (main.py)
        ↓
Next.js Dashboard (app/page.tsx)
```

---

## Features

### Computer Vision Pipeline
- OpenCV video/image I/O with graceful error handling
- CLAHE contrast enhancement, Gaussian blur, Canny edge detection
- Configurable Region of Interest (polygon + rectangle)
- Perspective transformation to bird's-eye view
- Frame skip for configurable inference FPS

### Object Detection
- YOLOv8 (Ultralytics) with PyTorch backend
- Traffic-relevant COCO class filtering (car, motorcycle, bus, truck, person, bicycle, traffic light)
- Configurable confidence and IoU thresholds
- GPU/CPU/MPS device selection

### Object Tracking
- IoU-based multi-object tracker
- Hungarian algorithm for optimal detection-to-track assignment
- Persistent track IDs with trajectory recording (last 50 positions)
- Stale track removal (configurable max_age)

### Violation Detection
| Violation | Implementation | Model Required |
|-----------|---------------|----------------|
| **No Helmet** | Motorcycle + person association + helmet classifier | Custom helmet model (stub provided) |
| **Red Light** | Stop-line crossing while light is RED | COCO YOLO (traffic light bbox) + color analysis |
| **Wrong Way** | Movement vector vs configured expected direction | COCO YOLO |
| **Lane Violation** | Vehicle center leaves lane polygon | COCO YOLO |

### Severity Engine
Rule-based, transparent severity scoring:
- **LOW** / **MEDIUM** / **HIGH** / **CRITICAL**
- Considers: violation type, detection confidence, repeat offenders
- Not pretended to be an ML model — explicitly rule-based

### Evidence Generation
- Annotated JPEG evidence images per violation
- Cooldown deduplication (no duplicate frames for same event)
- Filename format: `violation_YYYYMMDD_HHMMSS_{type}_vehicle_{id}.jpg`

### FastAPI Backend
- `GET /api/health` — health check
- `POST /api/detection/image` — single image analysis
- `POST /api/detection/video` — full video pipeline
- `GET /api/violations` — violation log with filtering
- `GET /api/analytics/summary` — aggregated statistics
- `GET /api/analytics/violations` — chart data
- `GET /api/config` — current configuration
- Auto-generated OpenAPI docs at `/api/docs`

### Next.js Dashboard
- **Overview**: 6 stat cards + violation type chart + vehicle distribution pie chart
- **Analysis**: Drag-and-drop image/video upload
- **Violations**: Sortable table with severity badges
- **Analytics**: Traffic volume time series + severity breakdown

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV 4.10 |
| Array Processing | NumPy |
| Tracking | Custom IoU + Hungarian (scipy) |
| Backend | FastAPI + Uvicorn |
| Config | Pydantic Settings |
| Frontend | Next.js 15, TypeScript, Tailwind CSS |
| Charts | Recharts |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Container | Docker, Docker Compose |
| Testing | pytest, httpx |
| Python | 3.13 |
| Node | 24 |

---

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp ../.env.example ../.env
# Edit .env as needed
```

### Frontend

```bash
cd frontend
npm install
```

---

## Dataset Setup

> **No dataset is required to run the system.**
> YOLOv8 pretrained COCO weights detect all primary traffic objects out of the box.

For custom model training (e.g., helmet detection):
1. See `datasets/README.md` for the expected dataset structure.
2. Annotate images using [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/heartexlabs/labelImg).
3. Use `scripts/train.py` (documented in that file).

---

## Model Setup

### Option 1: Auto-download (recommended for first run)
The system automatically downloads YOLOv8n from Ultralytics on first run.

### Option 2: Manual download
```bash
python scripts/download_models.py --size n
# Saves to models/yolov8n.pt
```

Available sizes: `n` (nano) · `s` (small) · `m` (medium) · `l` (large) · `x` (xlarge)

For a custom helmet model, set `HELMET_MODEL_PATH` in `.env`.

---

## Running Locally

### Backend
```bash
cd backend
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/api/docs
```

### Frontend
```bash
cd frontend
npm run dev
# Dashboard: http://localhost:3000
```

### Both (Docker Compose)
```bash
# Copy and configure .env
cp .env.example .env

docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

---

## Running Tests

```bash
cd backend
python -m pytest -v
```

**41 tests** covering:
- Geometric utilities (line crossing, direction, IoU, point-in-polygon)
- Object tracker (ID persistence, trajectory, stale removal)
- Violation detectors (red light, wrong way, severity engine)
- API endpoints (health, config, analytics, violations)

All tests run **without GPU** — mock detections and frames are used.

---

## API Documentation

Interactive API documentation is automatically generated:
- **Swagger UI**: `http://localhost:8000/api/docs`
- **ReDoc**: `http://localhost:8000/api/redoc`
- **OpenAPI JSON**: `http://localhost:8000/api/openapi.json`

---

## Model Performance

> **Important:** No custom training has been performed.
> The following are typical reported metrics for the pretrained COCO models.

### YOLOv8n (COCO pretrained)

| Metric | Value |
|--------|-------|
| mAP@50 | 52.9% |
| mAP@50:95 | 37.3% |
| Inference (CPU, 640px) | ~80ms |
| Parameters | 3.2M |

These metrics are for the **80-class COCO benchmark** — not for traffic-specific performance.

**Traffic-specific performance** requires evaluation on a traffic dataset.
See `docs/deep-learning.md` for evaluation methodology.

### Helmet Detection

The current helmet detection uses a stub classifier.
**No helmet detection accuracy claim is made** because no custom model is loaded.
Plug in a trained model via `HelmetClassifier` in `app/violations/helmet.py`.

---

## Limitations

1. **Helmet detection** requires a custom-trained model not included here.
2. **Traffic light state** detection uses HSV heuristics — works in daylight, degrades at night.
3. **Tracking** uses simple IoU — ID switches can occur when vehicles overlap.
4. **Lane detection** requires manual calibration of lane polygon coordinates.
5. **Perspective calibration** uses estimated defaults — proper calibration needs known road markings.
6. **Video processing** is synchronous — large files should use a task queue (Celery/ARQ).

---

## Future Improvements

1. **DeepSORT / ByteTrack** for robust re-ID tracking.
2. **Custom helmet model** training on a labeled dataset.
3. **Speed estimation** from bird's-eye trajectory + calibrated pixel-to-meter mapping.
4. **License plate recognition** integration.
5. **WebSocket streaming** for real-time camera feeds.
6. **PostgreSQL** persistence with proper session management.
7. **Celery** background task queue for video processing.
8. **Model quantization** (INT8) for edge deployment.

---

## Project Structure

```
smart-traffic-ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Typed configuration
│   │   ├── logging_config.py    # Structured logging
│   │   ├── api/routes/          # HTTP endpoints
│   │   ├── cv/                  # OpenCV pipeline
│   │   ├── detection/           # YOLO abstraction
│   │   ├── tracking/            # Object tracker
│   │   ├── violations/          # Rule engine
│   │   ├── analytics/           # Metrics
│   │   ├── evidence/            # Evidence images
│   │   └── schemas/             # Pydantic models
│   ├── tests/                   # 41 unit tests
│   └── requirements.txt
├── frontend/
│   ├── app/page.tsx             # Dashboard
│   └── lib/api.ts               # Typed API client
├── docs/
│   ├── computer-vision.md
│   ├── interview-preparation.md
│   └── violation-logic.md
├── scripts/
│   └── download_models.py
├── models/                      # Model weights (gitignored)
├── datasets/                    # Training data (gitignored)
├── docker-compose.yml
└── .env.example
```

---

## Interview Preparation

See [`docs/interview-preparation.md`](docs/interview-preparation.md) for detailed explanations of:
- OpenCV operations and why each is used
- YOLO architecture and training
- Object tracking algorithms
- Traffic rule logic implementation
- FastAPI architecture
- Docker deployment
