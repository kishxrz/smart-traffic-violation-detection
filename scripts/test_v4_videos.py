"""
scripts/test_v4_videos.py
──────────────────────────
Runs helmet_v4.pt on:
 1. NO_HELMET control video (evidence/videos/no_helmet_control_test.mp4)
 2. Helmeted real-world video (evidence/helmet_validation/VID_20250930_161454_debug_annotated.mp4)

Reports:
 HELMET count
 NO_HELMET count
 UNKNOWN count
 Confirmed violations
 Pipeline FPS
"""
import sys
import time
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.cv.video_processor import VideoProcessor
from app.detection.yolo_detector import YOLODetector
from app.tracking.object_tracker import ObjectTracker
from app.violations.rider_association import RiderAssociator
from app.cv.head_roi import extract_head_crop_with_quality
from app.detection.helmet_detector import HelmetDetector
from app.violations.helmet import HelmetViolationDetector
from app.violations.severity import SeverityEngine
from app.schemas.violation import SceneState


def test_video(video_path: str, model_path: str = "models/helmet_v4.pt", name: str = "Test Video"):
    print(f"\n=======================================================")
    print(f" TESTING MODEL {model_path} ON: {name}")
    print(f" Path: {video_path}")
    print(f"=======================================================")

    yolo = YOLODetector()
    tracker = ObjectTracker()
    associator = RiderAssociator(allow_motorcycle_crop_fallback=True)
    helmet_detector = HelmetDetector(model_path=Path(model_path))
    severity = SeverityEngine()
    violation_detector = HelmetViolationDetector(
        severity_engine=severity,
        helmet_detector=helmet_detector,
        min_frames_tracked=0, # for test video compatibility
    )

    frames = 0
    helmet_cnt = 0
    no_helmet_cnt = 0
    unknown_cnt = 0
    total_violations = 0

    t0 = time.time()
    with VideoProcessor(video_path, frame_skip=1) as vp:
        for frame_num, ts, frame in vp.frames():
            frames += 1
            dets = yolo.detect(frame, frame_number=frame_num, timestamp=ts)
            tracked = tracker.update(dets, frame_number=frame_num, timestamp=ts)

            # Per-frame predictions telemetry
            associations = associator.associate(tracked)
            for assoc in associations:
                rider_obj = next((o for o in tracked if o.track_id == assoc.rider_id), None)
                if rider_obj is None:
                    rider_obj = next((o for o in tracked if o.track_id == assoc.motorcycle_id), None)
                if rider_obj is None:
                    continue
                crop_res = extract_head_crop_with_quality(frame, rider_obj.bbox, target_size=(224, 224))
                if not crop_res.is_accepted:
                    continue

                pred = helmet_detector.predict_crop(crop_res.crop, assoc.motorcycle_id, assoc.rider_id, frame_num)
                if pred.status == "HELMET":
                    helmet_cnt += 1
                elif pred.status == "NO_HELMET":
                    no_helmet_cnt += 1
                else:
                    unknown_cnt += 1

            scene_state = SceneState(frame_number=frame_num, timestamp=ts, active_tracked_objects=len(tracked))
            v_list = violation_detector.evaluate(frame, tracked, scene_state, original_frame=frame)
            total_violations += len(v_list)

    elapsed = max(0.001, time.time() - t0)
    fps = frames / elapsed

    print(f" Frames Processed  : {frames}")
    print(f" HELMET status     : {helmet_cnt}")
    print(f" NO_HELMET status  : {no_helmet_cnt}")
    print(f" UNKNOWN status    : {unknown_cnt}")
    print(f" Confirmed Violations: {total_violations}")
    print(f" Pipeline FPS      : {fps:.2f}")

    return {
        "frames": frames,
        "helmet": helmet_cnt,
        "no_helmet": no_helmet_cnt,
        "unknown": unknown_cnt,
        "violations": total_violations,
        "fps": fps,
    }


if __name__ == "__main__":
    v_control = "evidence/videos/no_helmet_control_test.mp4"
    v_helmeted = "evidence/helmet_validation/VID_20250930_161454_debug_annotated.mp4"
    model = "models/helmet_v4.pt"

    print("Running V4 Video Telemetry Evaluation...")
    res_ctrl = test_video(v_control, model, "NO_HELMET Control Video")
    res_helm = test_video(v_helmeted, model, "Helmeted Real-World Video")
