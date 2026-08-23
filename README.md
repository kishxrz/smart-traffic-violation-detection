# Smart Traffic Intelligence & Violation Detection System

A production-grade, AI-powered computer vision system designed for real-time urban traffic monitoring, multi-class vehicle tracking, and automated traffic violation detection.

---

## 1. Project Overview
The **Smart Traffic Intelligence & Violation Detection System** provides end-to-end video analytics for traffic monitoring stations. By integrating custom object detection, multi-object tracking, spatial rider-motorcycle association, and temporal state machines, the system automatically detects violations and generates legal-grade evidence packages with severity scoring.

---

## 2. Architecture
```
[ Input Stream / Video / Image ]
               │
               ▼
   [ YOLOv8 Vehicle Detector ] ──(Detections)──► [ ByteTRACK / Object Tracker ]
                                                        │
               ┌────────────────────────────────────────┴────────────────────────────────────────┐
               ▼                                         ▼                                       ▼
  [ Spatial Rider Associator ]               [ Stop-Line Geometry Engine ]             [ Lane ROI & Direction Engine ]
               │                                         │                                       │
     (Rider/Head Crops)                         (Line Crossing Check)                    (Vector Direction / Crossing)
               ▼                                         ▼                                       ▼
  [ Helmet Classifier (v3) ]                  [ Red Light Detector ]                   [ Wrong-Way & Lane Detectors ]
               │                                         │                                       │
               └────────────────────────────────────────┬────────────────────────────────────────┘
                                                        │
                                                        ▼
                                           [ Temporal State Aggregator ]
                                                        │
                                                        ▼
                                            [ Severity Calculation Engine ]
                                                        │
                                                        ▼
                                          [ H.264 Video & Crop Evidence ]
                                                        │
                                                        ▼
                                          [ SQLite Database & REST API ]
                                                        │
                                                        ▼
                                           [ Next.js Admin Dashboard ]
```

---

## 3. Features
- **Multi-Class Object Detection:** Detects cars, motorcycles, buses, trucks, and pedestrians.
- **Persistent Vehicle Tracking:** Assigns unique IDs and tracks trajectories over time.
- **Rider-Motorcycle Spatial Association:** Maps riders to motorcycles with crop fallback.
- **No-Helmet Violation Detection:** Classifies rider head crops using YOLOv8n-cls (`models/helmet_v3.pt`).
- **Red-Light Violation Detection:** Detects stop-line crossing during red light signals.
- **Wrong-Way Violation Detection:** Monitors vehicle displacement against legal flow vectors.
- **Lane Boundary Violation Detection:** Detects illegal lane changes across solid lane markings.
- **Explainable Severity Scoring:** Calculates `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` severity with numeric scores (`0–100`) and human-readable explanations.
- **Browser-Playable H.264 MP4 Evidence:** Transcodes annotated videos into browser-compatible H.264 / `yuv420p` MP4 streams.
- **Full Analytics Dashboard:** Live metrics, violation tables, evidence modals, and interactive Recharts visualizations.

---

## 4. Tech Stack
- **Core Logic & Vision:** Python 3.13, OpenCV (`cv2`), PyTorch, Ultralytics YOLOv8.
- **Backend API & Async Server:** FastAPI, Uvicorn, Pydantic v2.
- **Database & Storage:** SQLite (`sqlite+aiosqlite`), SQLAlchemy async.
- **Video Processing:** `imageio-ffmpeg` static binary pipeline for H.264 / `yuv420p` transcoding.
- **Frontend Framework:** Next.js 16 (App Router), TypeScript, React 19, Recharts, Lucide Icons, Vanilla CSS / Tailwind tokens.
- **Containerization & Deployment:** Docker, Docker Compose, Multi-stage builds.

---

## 5. Computer Vision Pipeline
1. **Frame Ingestion:** Ingests video files or static images, scaling to uniform frame dimensions.
2. **Quality Head Crop Extraction:** Computes upper 35% rider ROI, clamps frame boundaries, checks aspect ratio, and resizes to $224 \times 224$ via `cv2.INTER_LANCZOS4`.
3. **Temporal Aggregation:** Requires `min_frames_tracked >= 10`, `min_observations >= 5`, and `no_helmet_ratio >= 0.70` before confirming violations.

---

## 6. Deep Learning Models
- **Primary Object Detector:** YOLOv8n (`models/yolov8n.pt`) trained on COCO dataset for multi-class vehicle detection.
- **Production Helmet Classifier:** Custom YOLOv8n-cls model (`models/helmet_v3.pt`) operating at $224 \times 224$ resolution.

---

## 7. Violation Detection Logic
- **Helmet Safeguard:** Requires temporal consistency over multiple frames; `UNKNOWN` states never trigger violations.
- **Wrong-Way Safeguard:** Requires minimum displacement (`min_displacement_px >= 30`) and sustained direction vector opposition to eliminate stationary box jitter.
- **Red-Light Safeguard:** Only triggers on active line crossing during `RED` signal state.
- **Lane Safeguard:** Evaluates vehicle centroid transition between restricted polygon ROIs.

---

## 8. Backend
Structure located in `backend/`:
- `app/api/`: FastAPI route handlers (`health`, `detection`, `violations`, `analytics`).
- `app/cv/`: Video processor, ROI geometry, frame processor, H.264 MP4 video encoder.
- `app/detection/`: YOLO detector and custom Helmet detector wrappers.
- `app/violations/`: Violation engine, temporal state accumulators, severity calculator.
- `app/database/`: SQLite database models and async session handlers.

---

## 9. Frontend
Structure located in `frontend/`:
- `app/page.tsx`: Single-page Next.js dashboard with stats, uploads, player, table, evidence modal, and analytics.
- `lib/api.ts`: Centralized API client utilizing `process.env.NEXT_PUBLIC_API_URL`.

---

## 10. Installation
```bash
# Clone repository
git clone https://github.com/example/smart-traffic-ai.git
cd "smart traffic int and violation detection sys"

# Backend setup
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
cd ..

# Frontend setup
cd frontend
npm install
cd ..
```

---

## 11. Environment Variables
Copy `.env.example` to `.env`:
```env
APP_NAME="Smart Traffic AI"
APP_ENV=production
NEXT_PUBLIC_API_URL=http://localhost:8000
DATABASE_URL=sqlite+aiosqlite:///./traffic.db
YOLO_MODEL_PATH=models/yolov8n.pt
HELMET_MODEL_PATH=models/helmet_v3.pt
EVIDENCE_DIR=evidence
ALLOWED_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000","http://localhost:8000"]
```

---

## 12. Running Locally
```bash
# Terminal 1: Backend
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

---

## 13. Docker Instructions
```bash
# Build and run containers with Docker Compose
docker compose up --build

# Access Frontend at http://localhost:3000
# Access Backend API at http://localhost:8000/api/health
# Access Swagger API Docs at http://localhost:8000/docs
```

---

## 14. API Endpoints
- `GET /api/health`: Health status & system environment metadata.
- `POST /api/detection/image`: Single image detection & violation analysis.
- `POST /api/detection/video`: Asynchronous video processing & H.264 annotated video generation.
- `GET /api/violations`: List recorded violations with pagination & filtering.
- `GET /api/analytics/summary`: Aggregate metrics, processing FPS, and counts.
- `GET /api/analytics/violations`: Violation breakdown by type and severity.

---

## 15. Model Limitations
- **Helmet Classifier Generalization:** The current helmet classifier (`models/helmet_v3.pt`) performs reliably on helmet-positive detection but has limited `NO_HELMET` generalization under extreme shadows, open-face helmets with skin exposure, or tinted visors. Conservative temporal confirmation (`min_observations >= 5`, `no_helmet_ratio >= 0.70`) is enforced to eliminate false violation reports.
- **Serverless Storage:** SQLite database and local `evidence/` directory require persistent filesystem storage volumes when containerized.

---

## 16. Testing Results
- **Held-Out Test Set Accuracy (V3):** `97.4%` (710 / 729 crops)
- **HELMET Precision / Recall (V3):** `98.4%` / `98.8%`
- **Real-World Helmeted Video False Positive Rate:** `0.0%` (0 false violations across 329 frames)
- **Automated Backend Test Suite:** `101/101 PASSED` (0 failures)
- **Frontend Build Status:** `0 errors` (`next build` compiled successfully)

---

## 17. Future Improvements
1. **Multi-Stage Head Detection:** Integrate a dedicated facial/head landmark detector to separate open-face helmet visors from skin tones.
2. **Cloud Object Storage:** Transition evidence image/video storage from local disk to S3/Cloud Storage.
3. **PostgreSQL Migration:** Replace SQLite with async PostgreSQL for distributed multi-camera deployments.
