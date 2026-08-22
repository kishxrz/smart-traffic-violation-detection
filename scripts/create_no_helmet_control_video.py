"""
scripts/create_no_helmet_control_video.py
──────────────────────────────────────────
Generates a real-world NO_HELMET control test video from verified Roboflow NO_HELMET traffic scene images.
Encodes to H.264 MP4 format: evidence/videos/no_helmet_control_test.mp4.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.cv.video_encoder import encode_to_browser_mp4
from app.detection.yolo_detector import YOLODetector


def create_control_video(
    source_dir: str = "datasets/helmet_binary",
    output_path: str = "evidence/videos/no_helmet_control_test.mp4",
    num_frames: int = 60,
    fps: float = 15.0,
) -> Path:
    yolo = YOLODetector(confidence_threshold=0.30)
    val_labels = glob.glob(os.path.join(source_dir, "labels/val/*.txt"))
    active_no_helmet_imgs = []

    for lpath in val_labels:
        with open(lpath) as f:
            lines = f.readlines()
        has_no_helmet = any(line.split() and line.split()[0] == "1" for line in lines)
        if has_no_helmet:
            stem = os.path.splitext(os.path.basename(lpath))[0]
            img_p = os.path.join(source_dir, "images/val", stem + ".jpg")
            if os.path.exists(img_p):
                mat = cv2.imread(img_p)
                if mat is not None:
                    dets = yolo.detect(mat, frame_number=1, timestamp=0.1)
                    if any(d.class_name == "motorcycle" for d in dets):
                        active_no_helmet_imgs.append(img_p)

    if not active_no_helmet_imgs:
        raise ValueError("No active NO_HELMET motorcycle images found in source directory!")

    print(f"Selected {len(active_no_helmet_imgs)} verified NO_HELMET motorcycle traffic images.")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_raw = out_file.parent / f"temp_{out_file.name}"

    # Target resolution: 640x640
    target_w, target_h = 640, 640

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_raw), fourcc, fps, (target_w, target_h))

    img_idx = 0
    for f_idx in range(num_frames):
        src_p = active_no_helmet_imgs[img_idx % len(active_no_helmet_imgs)]
        mat = cv2.imread(src_p)
        if mat is None:
            continue
        resized_mat = cv2.resize(mat, (target_w, target_h))
        writer.write(resized_mat)

        # Hold each scene image for 15 frames (~1 second) to simulate continuous tracking
        if (f_idx + 1) % 15 == 0:
            img_idx += 1

    writer.release()

    encode_to_browser_mp4(str(temp_raw), str(out_file))
    if temp_raw.exists():
        temp_raw.unlink()

    print(f"Successfully generated NO_HELMET control test video: {out_file} ({out_file.stat().st_size} bytes)")
    return out_file


if __name__ == "__main__":
    create_control_video()
