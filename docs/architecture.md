# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Traffic Camera                        │
│              (file / RTSP stream / webcam)               │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              OpenCV Video Processor                      │
│         (frame extraction, frame skip, FPS)              │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              Frame Preprocessor                          │
│    (CLAHE, Gaussian blur, resize, optional Canny)        │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              YOLO Detector (YOLOv8)                      │
│    car / motorcycle / person / bus / truck / traffic     │
│                   light / bicycle                        │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│            IoU + Hungarian Object Tracker               │
│     Persistent IDs, trajectories, movement vectors      │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│               Violation Engine                           │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐  │
│  │ Helmet Det.  │ │ Red Light    │ │ Wrong Way       │  │
│  │ (stub: needs │ │ (line cross  │ │ (direction      │  │
│  │ custom model)│ │ + light RED) │ │ vector check)   │  │
│  └──────────────┘ └──────────────┘ └─────────────────┘  │
│  ┌──────────────┐                                         │
│  │ Lane Viol.   │   Severity Engine (rule-based)          │
│  │ (polygon ROI)│                                         │
│  └──────────────┘                                         │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│             Evidence Generator                           │
│    Annotated JPEG, cooldown deduplication               │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│            Analytics Accumulator                         │
│  vehicles, violations, confidence, FPS, class dist.     │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                FastAPI REST API                          │
│   /api/health  /api/detection/*  /api/violations        │
│   /api/analytics/*  /api/config                         │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              Next.js Dashboard                           │
│  Overview | Analysis | Violations | Analytics            │
└─────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | File | Responsibility |
|-----------|------|---------------|
| VideoProcessor | `cv/video_processor.py` | Frame I/O, frame skip |
| FramePreprocessor | `cv/preprocessing.py` | CLAHE, blur, resize |
| YOLODetector | `detection/yolo_detector.py` | YOLO inference |
| ObjectTracker | `tracking/object_tracker.py` | ID assignment, trajectories |
| ViolationEngine | `violations/engine.py` | Orchestrate detectors |
| SeverityEngine | `violations/severity.py` | Rule-based severity |
| EvidenceGenerator | `evidence/evidence_generator.py` | Annotated images |
| AnalyticsAccumulator | `analytics/metrics.py` | Session statistics |
| FastAPI app | `main.py` | HTTP server |
| Dashboard | `frontend/app/page.tsx` | UI |

## Data Flow — Key Types

```
np.ndarray (raw frame)
    → Detection[]  (YOLO output, single frame)
    → TrackedObject[] (tracker output, with persistent IDs + trajectory)
    → Violation[] (violation engine output)
    → AnalyticsSummary (accumulated session stats)
    → JSON (API response)
```

## Dependency Architecture

```
main.py
  → api/routes/detection.py
    → api/dependencies.py          (singleton factory)
      → detection/models.py        (creates YOLODetector)
      → tracking/object_tracker.py (creates tracker)
      → violations/engine.py       (orchestrates detectors)
  → cv/frame_processor.py          (per-frame pipeline)
    → cv/preprocessing.py
    → detection/detector.py        (abstract)
      → detection/yolo_detector.py (concrete)
    → tracking/tracker.py          (abstract)
      → tracking/object_tracker.py (concrete)
```

## Extension Points

To add a new violation type:
1. Create `violations/my_violation.py` implementing `ViolationDetector`.
2. Register it in `api/dependencies.py` with `engine.register(MyViolationDetector(...))`.
3. No other changes required.

To swap the tracker:
1. Create a new class implementing `BaseTracker`.
2. Return it from `get_tracker()` in `api/dependencies.py`.

To add a custom detector (e.g., RT-DETR):
1. Create a class implementing `BaseDetector`.
2. Return it from `create_detector()` in `detection/models.py`.
