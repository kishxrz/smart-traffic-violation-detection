"""
scripts/diagnose_no_helmet_domain.py
──────────────────────────────────────────────────────────────────────────────
CONTROLLED EXPERIMENT: Isolate cause of NO_HELMET detection failure.

For each of 20 NO_HELMET source images, tests THREE conditions:

  Condition 1 (DIRECT JPG):
      Load original JPG crop → run helmet_v3.pt directly

  Condition 2 (MP4-DECODED FULL FRAME → RE-EXTRACTED CROP):
      Encode source image as mp4v → transcode to H.264 → decode frame
      → extract crop at same bbox → run helmet_v3.pt

  Condition 3 (RESIZED ONLY, no encoding):
      Resize to 640×640 (same as video pipeline) → run helmet_v3.pt

Compares predictions to isolate:
  A. H.264 encoding/decoding causes the class flip
  B. Original JPG also fails (model generalization issue)
  C. Crop extraction / resize causes the flip
  D. Another cause
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import os
from pathlib import Path
import random
import glob

import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ultralytics import YOLO
import imageio_ffmpeg


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/helmet_v3.pt"
SOURCE_DIR   = "datasets/helmet_binary"
CROPS_DIR    = Path("datasets/helmet_crops/images/test")
CROPS_LBL    = Path("datasets/helmet_crops/labels/test")
N_SAMPLES    = 20
RANDOM_SEED  = 42
VIDEO_FPS    = 15.0
VIDEO_W, VIDEO_H = 640, 640


def load_model():
    model = YOLO(MODEL_PATH)
    print(f"Model task  : {model.task}")
    print(f"Class names : {model.names}")
    assert model.names[0] == "helmet" and model.names[1] == "no_helmet", \
        "UNEXPECTED CLASS ORDER — abort"
    return model


def predict(model, img: np.ndarray) -> dict:
    """Run classification and return probabilities + predicted class."""
    res = model(img, verbose=False)
    probs = res[0].probs
    if probs is None:
        return {"top1": -1, "top1_name": "error", "helmet_p": 0.0, "no_helmet_p": 0.0}
    top1 = int(probs.top1)
    data = [float(x) for x in probs.data.tolist()]
    return {
        "top1": top1,
        "top1_name": model.names[top1],
        "helmet_p":    round(data[0], 4),
        "no_helmet_p": round(data[1], 4),
    }


def encode_decode_frame(source_img: np.ndarray) -> np.ndarray | None:
    """
    Replicate the EXACT encoding pipeline used in create_no_helmet_control_video.py:
      1. Resize to 640×640
      2. Write with OpenCV mp4v VideoWriter
      3. Transcode to H.264 via imageio-ffmpeg (libx264, yuv420p, fast preset)
      4. Decode with OpenCV VideoCapture
      5. Return first frame
    """
    tmp_dir = Path(tempfile.mkdtemp())
    raw_mp4  = tmp_dir / "raw.mp4"
    h264_mp4 = tmp_dir / "h264.mp4"

    try:
        resized = cv2.resize(source_img, (VIDEO_W, VIDEO_H))

        # Step 1: Write with mp4v (same as control video creation)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_mp4), fourcc, VIDEO_FPS, (VIDEO_W, VIDEO_H))
        # Write 3 copies to ensure a decodable file
        for _ in range(3):
            writer.write(resized)
        writer.release()

        # Step 2: Transcode to H.264 (same as encode_to_browser_mp4)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", str(raw_mp4),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-movflags", "+faststart",
            str(h264_mp4),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0 or not h264_mp4.exists():
            return None

        # Step 3: Decode first frame with OpenCV VideoCapture
        cap = cv2.VideoCapture(str(h264_mp4))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None

        return frame

    finally:
        for f in [raw_mp4, h264_mp4]:
            try: f.unlink(missing_ok=True)
            except: pass
        try: tmp_dir.rmdir()
        except: pass


def collect_no_helmet_crops(n: int) -> list[Path]:
    """Collect N NO_HELMET crops from the held-out test set."""
    random.seed(RANDOM_SEED)
    candidates = []
    for lbl_file in CROPS_LBL.glob("*.txt"):
        try:
            cls = int(lbl_file.read_text().strip().splitlines()[0].split()[0])
        except Exception:
            continue
        if cls != 1:
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            img_f = CROPS_DIR / (lbl_file.stem + ext)
            if img_f.exists():
                candidates.append(img_f)
                break
    random.shuffle(candidates)
    return candidates[:n]


def main():
    print("\n" + "=" * 80)
    print(" CONTROLLED DOMAIN COMPARISON EXPERIMENT")
    print(" Models: helmet_v3.pt   Samples: 20 NO_HELMET test crops")
    print("=" * 80)

    model = load_model()

    # Collect 20 NO_HELMET crops from held-out test set
    crop_paths = collect_no_helmet_crops(N_SAMPLES)
    if not crop_paths:
        print("ERROR: No NO_HELMET test crops found.")
        sys.exit(1)
    print(f"\nCollected {len(crop_paths)} NO_HELMET crops from datasets/helmet_crops/test/\n")

    # ── Table header ─────────────────────────────────────────────────────────
    header = (
        f"{'#':>3}  {'Condition':<22}  {'Predicted':<10}  "
        f"{'helmet_p':>9}  {'no_helmet_p':>11}  {'Correct?'}"
    )
    print(header)
    print("-" * 80)

    results = []

    for i, crop_path in enumerate(crop_paths, 1):
        orig_jpg = cv2.imread(str(crop_path))
        if orig_jpg is None:
            print(f"{i:>3}  SKIP (unreadable): {crop_path.name}")
            continue

        # ── Condition 1: Direct JPG ───────────────────────────────────────────
        p1 = predict(model, orig_jpg)
        correct1 = "YES" if p1["top1_name"] == "no_helmet" else "NO ❌"
        print(f"{i:>3}  {'1-DIRECT JPG':<22}  {p1['top1_name']:<10}  "
              f"{p1['helmet_p']:>9.4f}  {p1['no_helmet_p']:>11.4f}  {correct1}")

        # ── Condition 2: H.264 encode → decode → predict ──────────────────────
        decoded_frame = encode_decode_frame(orig_jpg)
        if decoded_frame is not None:
            # The decoded frame is 640×640 (full scene). Since source crop is
            # already a head crop, we pass the entire decoded frame as the crop.
            p2 = predict(model, decoded_frame)
            correct2 = "YES" if p2["top1_name"] == "no_helmet" else "NO ❌"
        else:
            p2 = {"top1_name": "ERROR", "helmet_p": 0.0, "no_helmet_p": 0.0}
            correct2 = "ERR"
        print(f"     {'2-H264 DECODED':<22}  {p2['top1_name']:<10}  "
              f"{p2['helmet_p']:>9.4f}  {p2['no_helmet_p']:>11.4f}  {correct2}")

        # ── Condition 3: Resize only (no encoding) ────────────────────────────
        resized = cv2.resize(orig_jpg, (VIDEO_W, VIDEO_H))
        p3 = predict(model, resized)
        correct3 = "YES" if p3["top1_name"] == "no_helmet" else "NO ❌"
        print(f"     {'3-RESIZE ONLY':<22}  {p3['top1_name']:<10}  "
              f"{p3['helmet_p']:>9.4f}  {p3['no_helmet_p']:>11.4f}  {correct3}")
        print()

        results.append({
            "sample": i,
            "jpg_correct":     p1["top1_name"] == "no_helmet",
            "h264_correct":    p2["top1_name"] == "no_helmet",
            "resize_correct":  p3["top1_name"] == "no_helmet",
            "jpg_no_helmet_p":    p1["no_helmet_p"],
            "h264_no_helmet_p":   p2["no_helmet_p"],
            "resize_no_helmet_p": p3["no_helmet_p"],
        })

    # ── Aggregate summary ─────────────────────────────────────────────────────
    n = len(results)
    jpg_correct    = sum(r["jpg_correct"]    for r in results)
    h264_correct   = sum(r["h264_correct"]   for r in results)
    resize_correct = sum(r["resize_correct"] for r in results)

    avg_jpg_nh    = sum(r["jpg_no_helmet_p"]    for r in results) / n
    avg_h264_nh   = sum(r["h264_no_helmet_p"]   for r in results) / n
    avg_resize_nh = sum(r["resize_no_helmet_p"] for r in results) / n

    print("=" * 80)
    print(" AGGREGATE RESULTS")
    print("=" * 80)
    print(f"  Condition              | Correct/Total | Accuracy | Avg no_helmet_p")
    print(f"  -----------------------|---------------|----------|-----------------")
    print(f"  1. Direct JPG          | {jpg_correct:>5}/{n}      | {jpg_correct/n:>7.1%}  | {avg_jpg_nh:.4f}")
    print(f"  2. H.264 Encode/Decode | {h264_correct:>5}/{n}      | {h264_correct/n:>7.1%}  | {avg_h264_nh:.4f}")
    print(f"  3. Resize Only         | {resize_correct:>5}/{n}      | {resize_correct/n:>7.1%}  | {avg_resize_nh:.4f}")
    print("=" * 80)

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    print("\n DIAGNOSIS")
    print("-" * 80)

    if jpg_correct < n * 0.5:
        # Even the original JPG fails
        print("  FINDING: Original JPG crops ALSO fail (< 50% correct).")
        print("  DIAGNOSIS: B — Dataset/model generalization problem.")
        print("  The model has not learned the NO_HELMET class from these crops.")
        print("  Possible cause: training augmentation or class-folder naming mismatch.")

    elif resize_correct < n * 0.5 and jpg_correct >= n * 0.5:
        print("  FINDING: Direct JPG works but resize breaks classification.")
        print("  DIAGNOSIS: C — Crop resize/preprocessing is the cause.")
        print("  The model is sensitive to the 640x640 resize of small head crops.")

    elif h264_correct < resize_correct - 2:
        print("  FINDING: Resize is fine but H.264 encode/decode breaks classification.")
        print("  DIAGNOSIS: A — H.264/domain preprocessing is the cause.")
        print("  YUV420p color space conversion and lossy compression degrade the signal.")

    else:
        print("  FINDING: All three conditions show similar accuracy.")
        print("  DIAGNOSIS: D — The failure is consistent across all preprocessing.")
        print("  The model generalizes poorly from training crops to inference crops,")
        print("  regardless of encoding. Dataset or model architecture change needed.")

    # ── Class mapping confirmation ─────────────────────────────────────────────
    print("\n CLASS MAPPING CONFIRMATION")
    print(f"  model.names = {model.names}")
    print(f"  0 = helmet     ← correct")
    print(f"  1 = no_helmet  ← correct")


if __name__ == "__main__":
    main()
