"""
scripts/run_inference.py
─────────────────────────
Run the full detection + tracking + violation pipeline on a single
video file or image, and print/save results.

Usage:
    python scripts/run_inference.py --source traffic.mp4 --output results/
    python scripts/run_inference.py --source image.jpg
    python scripts/run_inference.py --source 0  # webcam

This script demonstrates the full pipeline without the HTTP API.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_on_video(source: str, output_dir: Path) -> None:
    from app.config import get_settings
    from app.cv.frame_processor import FrameProcessor
    from app.cv.preprocessing import PreprocessingConfig
    from app.cv.video_processor import VideoProcessor
    from app.cv.visualization import draw_tracked_object, draw_hud
    from app.detection.models import create_detector
    from app.evidence.evidence_generator import EvidenceGenerator
    from app.schemas.violation import SceneState, TrafficLightState
    from app.tracking.object_tracker import ObjectTracker
    from app.violations.engine import ViolationEngine
    from app.violations.severity import SeverityEngine
    from app.violations.red_light import RedLightViolationDetector
    from app.violations.wrong_way import WrongWayViolationDetector
    import cv2

    settings = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading detector...")
    detector = create_detector(settings)
    detector.warmup()

    tracker = ObjectTracker()
    severity = SeverityEngine()
    engine = ViolationEngine()
    engine.register(RedLightViolationDetector(severity, stop_line_y=400))
    engine.register(WrongWayViolationDetector(severity))
    evidence = EvidenceGenerator(output_dir / "evidence")

    preprocessor_cfg = PreprocessingConfig(clahe=True)
    frame_proc = FrameProcessor(detector, tracker, preprocessor_cfg)

    all_violations = []

    try:
        source_val = int(source) if source.isdigit() else source
    except Exception:
        source_val = source

    with VideoProcessor(source_val, frame_skip=settings.frame_skip) as vp:
        metadata = vp.metadata
        logger.info("Processing: %s", metadata)

        writer = None
        for frame_num, ts, frame in vp.frames():
            result = frame_proc.process(frame, frame_num, ts)

            scene = SceneState(
                frame_number=frame_num,
                timestamp=ts,
                traffic_light_state=TrafficLightState.UNKNOWN,
            )
            violations = engine.evaluate(frame, result.tracked_objects, scene)
            all_violations.extend(violations)

            tracked_by_id = {t.track_id: t for t in result.tracked_objects}
            evidence.save_batch(frame, violations, tracked_by_id)

            # Annotate frame
            annotated = frame.copy()
            for obj in result.tracked_objects:
                viol_types = [v.violation_type for v in violations if v.vehicle_id == obj.track_id]
                annotated = draw_tracked_object(annotated, obj, viol_types)
            annotated = draw_hud(
                annotated,
                fps=metadata.fps / settings.frame_skip,
                frame_number=frame_num,
                vehicle_count=len(result.tracked_objects),
                violation_count=len(all_violations),
            )

            if writer is None and isinstance(source_val, str):
                out_path = output_dir / "annotated_output.mp4"
                writer = VideoProcessor.create_writer(
                    out_path, metadata.fps / settings.frame_skip,
                    metadata.width, metadata.height,
                )
            if writer:
                writer.write(annotated)

        if writer:
            writer.release()

    results = {
        "source": str(source),
        "total_violations": len(all_violations),
        "violations": [v.model_dump(mode="json") for v in all_violations],
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Results saved: %s", results_path)
    logger.info("Total violations: %d", len(all_violations))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traffic AI inference.")
    parser.add_argument("--source", required=True, help="Video path, image path, or camera index")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    run_on_video(args.source, Path(args.output))


if __name__ == "__main__":
    main()
