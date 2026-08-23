"""
scripts/train_helmet_v3_fast.py
────────────────────────────────
Ultra-fast Helmet V3 training:
  - Subsamples 500 HELMET + 500 NO_HELMET from helmet_crops_v3/train
  - Trains YOLOv8n for 5 epochs at imgsz=160 on CPU
  - Copies best weights to models/helmet_v3.pt
Estimated time: ~3-5 minutes on CPU.
"""
from __future__ import annotations

import os
import random
import shutil
import sys
import textwrap
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC_V3  = ROOT / "datasets" / "helmet_crops_v3"
MINI_DS = ROOT / "datasets" / "helmet_crops_v3_mini"
MODELS  = ROOT / "models"

SAMPLES_PER_CLASS = 500
EPOCHS  = 5
IMGSZ   = 160
BATCH   = 32

def _copy_subset(src_img_dir: Path, src_lbl_dir: Path,
                 dst_img_dir: Path, dst_lbl_dir: Path,
                 class_id: int, n: int) -> int:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for lbl_file in src_lbl_dir.glob("*.txt"):
        try:
            first_line = lbl_file.read_text().strip().splitlines()[0]
            if int(first_line.split()[0]) == class_id:
                img_stem = lbl_file.stem
                for ext in (".jpg", ".jpeg", ".png"):
                    img_file = src_img_dir / (img_stem + ext)
                    if img_file.exists():
                        candidates.append((img_file, lbl_file))
                        break
        except Exception:
            continue

    chosen = random.sample(candidates, min(n, len(candidates)))
    for img_f, lbl_f in chosen:
        shutil.copy(img_f, dst_img_dir / img_f.name)
        shutil.copy(lbl_f, dst_lbl_dir / lbl_f.name)
    return len(chosen)


def build_mini_dataset() -> Path:
    print("Building mini dataset …")
    if MINI_DS.exists():
        shutil.rmtree(MINI_DS)

    src_train_img = SRC_V3 / "images" / "train"
    src_train_lbl = SRC_V3 / "labels" / "train"

    dst_train_img = MINI_DS / "images" / "train"
    dst_train_lbl = MINI_DS / "labels" / "train"

    n_h  = _copy_subset(src_train_img, src_train_lbl, dst_train_img, dst_train_lbl, 0, SAMPLES_PER_CLASS)
    n_nh = _copy_subset(src_train_img, src_train_lbl, dst_train_img, dst_train_lbl, 1, SAMPLES_PER_CLASS)
    print(f"  train: {n_h} HELMET, {n_nh} NO_HELMET")

    # Symlink/copy val & test from original (use first 200 each to keep it small)
    for split in ("val", "test"):
        src_si = SRC_V3 / "images" / split
        src_sl = SRC_V3 / "labels" / split
        dst_si = MINI_DS / "images" / split
        dst_sl = MINI_DS / "labels" / split
        dst_si.mkdir(parents=True, exist_ok=True)
        dst_sl.mkdir(parents=True, exist_ok=True)
        imgs = list(src_si.glob("*.jpg")) + list(src_si.glob("*.jpeg")) + list(src_si.glob("*.png"))
        for img_f in imgs[:200]:
            shutil.copy(img_f, dst_si / img_f.name)
            lbl_f = src_sl / (img_f.stem + ".txt")
            if lbl_f.exists():
                shutil.copy(lbl_f, dst_sl / lbl_f.name)
        print(f"  {split}: {min(len(imgs),200)} images")

    yaml_content = textwrap.dedent(f"""\
        path: {MINI_DS.as_posix()}
        train: images/train
        val: images/val
        test: images/test
        names:
          0: helmet
          1: no_helmet
    """)
    yaml_path = MINI_DS / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  YAML: {yaml_path}")
    return yaml_path


def train(yaml_path: Path) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining V3 on {device}  epochs={EPOCHS}  imgsz={IMGSZ}  batch={BATCH}")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        project=str(ROOT / "runs" / "helmet"),
        name="v3",
        exist_ok=True,
        patience=3,
        verbose=True,
        workers=0,
    )

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = save_dir / "weights" / "last.pt"

    MODELS.mkdir(parents=True, exist_ok=True)
    target = MODELS / "helmet_v3.pt"
    shutil.copy(best_weights, target)
    print(f"\n✅ helmet_v3.pt saved → {target}")
    return target


if __name__ == "__main__":
    random.seed(42)
    yaml_path = build_mini_dataset()
    train(yaml_path)
    print("\nDone. Run calibrate_helmet_threshold.py next.")
