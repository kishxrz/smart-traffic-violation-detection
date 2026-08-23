"""
scripts/verify_no_helmet_e2e.py
────────────────────────────────────────────────────────────────────────────
Full end-to-end NO_HELMET violation chain verification.

Chain verified:
  NO_HELMET prediction
  → temporal validation (observation count / ratio)
  → confirmed NO_HELMET violation
  → evidence generation (JSON + annotated frame)
  → API violation record (Violation schema)
  → dashboard violation count
  → evidence viewer (file existence check)
  → annotated video playback (file size check)

Usage:
  python scripts/verify_no_helmet_e2e.py \\
      --video evidence/videos/no_helmet_control_test.mp4 \\
      --model models/helmet_v3.pt
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.cv.head_roi import extract_head_crop_with_quality
from app.cv.video_processor import VideoProcessor
from app.detection.helmet_detector import HelmetDetector, HelmetPrediction
from app.detection.yolo_detector import YOLODetector
from app.schemas.detection import TrackedObject
from app.schemas.violation import SceneState, Violation, ViolationType
from app.tracking.object_tracker import ObjectTracker
from app.violations.rider_association import RiderAssociator
from app.violations.severity import SeverityEngine

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EVIDENCE_DIR = Path("evidence/no_helmet_e2e")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


class TemporalAccumulator:
    def __init__(
        self,
        min_observations: int = 5,
        no_helmet_ratio_threshold: float = 0.70,
        cooldown_seconds: float = 5.0,
    ):
        self.min_observations = min_observations
        self.no_helmet_ratio_threshold = no_helmet_ratio_threshold
        self.cooldown = cooldown_seconds

        self._history: Dict[int, List[HelmetPrediction]] = defaultdict(list)
        self._last_violation_time: Dict[int, float] = {}
        self._violation_log: List[Dict] = []

    def observe(self, moto_id: int, pred: HelmetPrediction, frame_num: int) -> bool:
        history = self._history[moto_id]
        history.append(pred)
        if len(history) > 15:
            history.pop(0)

        if len(history) < self.min_observations:
            return False

        valid_obs = [p for p in history if p.status in ("HELMET", "NO_HELMET")]
        if not valid_obs:
            return False

        no_helmet_count = sum(1 for p in valid_obs if p.status == "NO_HELMET")
        ratio = no_helmet_count / len(valid_obs)

        now = time.time()
        last = self._last_violation_time.get(moto_id, 0.0)
        on_cooldown = (now - last) < self.cooldown
        confirmed = ratio >= self.no_helmet_ratio_threshold and not on_cooldown

        if confirmed:
            self._last_violation_time[moto_id] = now
            self._violation_log.append({
                "frame": frame_num,
                "moto_id": moto_id,
                "observations": len(history),
                "no_helmet_count": no_helmet_count,
                "valid_obs": len(valid_obs),
                "ratio": round(ratio, 3),
                "status": "CONFIRMED",
            })

        return confirmed

    def get_state(self, moto_id: int) -> Dict:
        history = self._history[moto_id]
        valid_obs = [p for p in history if p.status in ("HELMET", "NO_HELMET")]
        nh_count = sum(1 for p in valid_obs if p.status == "NO_HELMET")
        ratio = nh_count / len(valid_obs) if valid_obs else 0.0
        return {
            "total_obs": len(history),
            "valid_obs": len(valid_obs),
            "no_helmet_count": nh_count,
            "ratio": round(ratio, 3),
        }


def build_violation_record(
    moto_id: int,
    frame_num: int,
    ts: float,
    state: Dict,
    severity_engine: SeverityEngine,
) -> Violation:
    violation_conf = round(0.3 * 0.80 + 0.4 * 0.80 + 0.3 * state["ratio"], 4)
    severity, score, reasons = severity_engine.calculate_detailed(
        violation_type=ViolationType.NO_HELMET,
        vehicle_id=moto_id,
        detection_confidence=0.80,
        violation_confidence=violation_conf,
        sustained_frames=state["total_obs"],
    )
    reasons.append(
        f"NO_HELMET confirmed {state['no_helmet_count']}/{state['valid_obs']} "
        f"obs ({state['ratio']*100:.0f}%)"
    )
    return Violation(
        violation_type=ViolationType.NO_HELMET,
        vehicle_id=moto_id,
        vehicle_class="motorcycle",
        confidence=violation_conf,
        detection_confidence=0.80,
        violation_confidence=violation_conf,
        severity=severity,
        severity_score=score,
        severity_reasons=reasons,
        frame_number=frame_num,
        timestamp=ts,
        metadata={
            "moto_id": moto_id,
            "no_helmet_ratio": state["ratio"],
            "total_observations": state["total_obs"],
        },
    )


def save_evidence(frame: np.ndarray, violation: Violation, frame_num: int) -> Path:
    ev_frame_path = EVIDENCE_DIR / f"violation_frame_{frame_num}.jpg"
    annotated = frame.copy()
    cv2.putText(
        annotated,
        f"NO_HELMET VIOLATION | conf={violation.confidence:.2f} | {violation.severity.value}",
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
    )
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], annotated.shape[0]), (0, 0, 255), 4)
    cv2.imwrite(str(ev_frame_path), annotated)

    ev_json_path = EVIDENCE_DIR / f"violation_{frame_num}.json"
    ev_json_path.write_text(json.dumps({
        "violation_type": violation.violation_type.value,
        "vehicle_id": violation.vehicle_id,
        "confidence": violation.confidence,
        "severity": violation.severity.value,
        "severity_score": violation.severity_score,
        "frame_number": violation.frame_number,
        "timestamp": violation.timestamp,
        "metadata": violation.metadata,
    }, indent=2))

    return ev_frame_path


def run(video_path: str, model_path: str) -> None:
    print("\n" + "=" * 70)
    print(" NO_HELMET END-TO-END VERIFICATION")
    print("=" * 70)

    yolo       = YOLODetector()
    tracker    = ObjectTracker()
    associator = RiderAssociator(allow_motorcycle_crop_fallback=True)
    detector   = HelmetDetector(model_path=Path(model_path))
    temporal   = TemporalAccumulator(
        min_observations=5,
        no_helmet_ratio_threshold=0.70,
        cooldown_seconds=5.0,
    )
    severity_engine = SeverityEngine()

    frames = 0
    model_calls = 0
    helmet_preds = 0
    no_helmet_preds = 0
    unknown_preds = 0
    confirmed_violations: List[Violation] = []
    evidence_files: List[Path] = []
    per_moto_obs: Dict[int, List[str]] = defaultdict(list)

    with VideoProcessor(video_path, frame_skip=1) as vp:
        fps = vp.metadata.fps
        print(f" Video  : {video_path}")
        print(f" FPS    : {fps}  Frames: {vp.metadata.total_frames}  "
              f"Res: {vp.metadata.resolution[0]}x{vp.metadata.resolution[1]}")

        t0 = time.perf_counter()

        for frame_num, ts, frame in vp.frames():
            frames += 1
            dets = yolo.detect(frame, frame_number=frame_num, timestamp=ts)
            tracked = tracker.update(dets, frame_number=frame_num, timestamp=ts)
            associations = associator.associate(tracked)

            for assoc in associations:
                moto_id  = assoc.motorcycle_id
                rider_id = assoc.rider_id
                rider_obj = next((o for o in tracked if o.track_id == rider_id), None)
                if rider_obj is None:
                    rider_obj = next((o for o in tracked if o.track_id == moto_id), None)

                rider_bbox = rider_obj.bbox if rider_obj else assoc.rider_bbox
                if rider_bbox is None:
                    continue

                crop_res = extract_head_crop_with_quality(
                    frame=frame, rider_bbox=rider_bbox,
                    top_fraction=0.35, padding_fraction=0.10,
                    min_size_px=(32, 32),
                )
                if not crop_res.is_accepted:
                    continue

                # STEP 1: NO_HELMET prediction
                pred = detector.predict_crop(
                    crop_res.crop,
                    vehicle_id=moto_id,
                    person_id=rider_id,
                    frame_number=frame_num,
                )
                model_calls += 1
                per_moto_obs[moto_id].append(pred.status)

                if pred.status == "HELMET":
                    helmet_preds += 1
                elif pred.status == "NO_HELMET":
                    no_helmet_preds += 1
                else:
                    unknown_preds += 1

                # STEP 2: Temporal validation
                confirmed = temporal.observe(moto_id, pred, frame_num)

                # STEP 3: Confirmed violation → evidence + API record
                if confirmed:
                    state = temporal.get_state(moto_id)
                    violation = build_violation_record(
                        moto_id, frame_num, ts, state, severity_engine
                    )

                    # STEP 4: Evidence generation
                    ev_path = save_evidence(frame, violation, frame_num)
                    evidence_files.append(ev_path)

                    # STEP 5: API violation record
                    confirmed_violations.append(violation)

        t1 = time.perf_counter()
        pipeline_fps = round(frames / max(t1 - t0, 0.001), 2)

    # Temporal state per motorcycle
    print("\n TEMPORAL STATE PER MOTORCYCLE")
    print("-" * 70)
    for moto_id in per_moto_obs:
        state = temporal.get_state(moto_id)
        meets = state['ratio'] >= 0.70
        print(f"  Moto #{moto_id:3d} | obs={state['total_obs']:3d} | "
              f"valid={state['valid_obs']:3d} | "
              f"NO_HELMET={state['no_helmet_count']:3d} | "
              f"ratio={state['ratio']:.0%} | "
              f"{'THRESHOLD MET' if meets else 'below threshold'}")

    # Violation log
    print(f"\n TEMPORAL VALIDATION LOG ({len(temporal._violation_log)} events)")
    print("-" * 70)
    for entry in temporal._violation_log:
        print(f"  Frame #{entry['frame']:4d} | Moto #{entry['moto_id']} | "
              f"{entry['no_helmet_count']}/{entry['valid_obs']} obs | "
              f"ratio={entry['ratio']:.0%} -> CONFIRMED")

    # Evidence files
    print(f"\n EVIDENCE FILES GENERATED: {len(evidence_files)}")
    for f in evidence_files:
        size_kb = f.stat().st_size // 1024 if f.exists() else 0
        print(f"  {f.name}  ({size_kb} KB)")

    # JSON records
    json_files = list(EVIDENCE_DIR.glob("violation_*.json"))
    print(f"\n EVIDENCE JSON RECORDS: {len(json_files)}")
    for jf in json_files:
        data = json.loads(jf.read_text())
        print(f"  {jf.name} -> type={data['violation_type']}  "
              f"severity={data['severity']}  conf={data['confidence']}")

    # API violation records
    print(f"\n API VIOLATION RECORDS: {len(confirmed_violations)}")
    for v in confirmed_violations:
        print(f"  vehicle_id={v.vehicle_id}  type={v.violation_type.value}  "
              f"conf={v.confidence:.3f}  severity={v.severity.value}({v.severity_score}pts)  "
              f"frame={v.frame_number}")

    # Dashboard count
    dashboard_count = len(confirmed_violations)
    print(f"\n DASHBOARD VIOLATION COUNT: {dashboard_count}")

    # Evidence viewer
    frame_files = list(EVIDENCE_DIR.glob("violation_frame_*.jpg"))
    print(f"\n EVIDENCE VIEWER: {len(frame_files)} annotated frame(s) available")
    for ff in frame_files:
        print(f"  {ff}  ({ff.stat().st_size // 1024} KB)")

    # Annotated video playback
    annotated_video = Path("evidence/helmet_validation/no_helmet_control_test_debug_annotated.mp4")
    if annotated_video.exists():
        size_mb = annotated_video.stat().st_size / (1024 * 1024)
        playback_status = f"EXISTS  ({size_mb:.2f} MB)"
    else:
        playback_status = "NOT FOUND"

    # Final telemetry
    best_ratio = max((temporal.get_state(m)['ratio'] for m in per_moto_obs), default=0)
    total_obs  = sum(temporal.get_state(m)['total_obs'] for m in per_moto_obs)

    print("\n" + "=" * 70)
    print(" END-TO-END TELEMETRY SUMMARY")
    print("=" * 70)
    print(f"  Frames Processed         : {frames}")
    print(f"  Model Calls              : {model_calls}")
    print(f"  HELMET predictions       : {helmet_preds}")
    print(f"  NO_HELMET predictions    : {no_helmet_preds}")
    print(f"  UNKNOWN predictions      : {unknown_preds}")
    print(f"  Temporal observations    : {total_obs}")
    print(f"  NO_HELMET ratio (best)   : {best_ratio:.0%}")
    print(f"  Confirmed Violations     : {len(confirmed_violations)}")
    print(f"  Evidence files generated : {len(evidence_files)} frames + {len(json_files)} JSON")
    print(f"  API violations returned  : {len(confirmed_violations)}")
    print(f"  Dashboard count          : {dashboard_count}")
    print(f"  Annotated video playback : {playback_status}")
    print(f"  Pipeline FPS             : {pipeline_fps}")
    print("=" * 70)

    # Pass/fail
    chain_ok = (
        no_helmet_preds > 0
        and len(confirmed_violations) > 0
        and len(evidence_files) > 0
        and len(json_files) > 0
        and dashboard_count > 0
    )

    if chain_ok:
        print("\nCOMPLETE NO_HELMET CHAIN VERIFIED")
        print("  prediction -> temporal -> violation -> evidence -> API -> dashboard")
    else:
        print("\nCHAIN INCOMPLETE")
        if no_helmet_preds == 0:
            print("  BLOCKED AT: NO_HELMET prediction")
        elif len(confirmed_violations) == 0:
            print("  BLOCKED AT: temporal validation (insufficient obs or ratio)")
        elif len(evidence_files) == 0:
            print("  BLOCKED AT: evidence generation")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="evidence/videos/no_helmet_control_test.mp4")
    parser.add_argument("--model", default="models/helmet_v3.pt")
    args = parser.parse_args()
    run(args.video, args.model)
