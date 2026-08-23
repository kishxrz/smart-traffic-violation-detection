"""
scripts/train_helmet_v3_cls.py
───────────────────────────────
Train YOLOv8n-cls (image classification) on the balanced helmet_crops_v3 dataset.

WHY classification instead of detection:
  - Crops are already extracted head ROIs — no bbox needed
  - Classification is ~5x faster on CPU (no bbox regression head)
  - softmax score directly gives HELMET vs NO_HELMET confidence
  - Better class separation on small crops

Dataset layout required (ImageNet-style):
  datasets/helmet_cls/
    train/helmet/   *.jpg
    train/no_helmet/ *.jpg
    val/helmet/     *.jpg
    val/no_helmet/  *.jpg

Output: models/helmet_v3.pt  (replaces detection V3)
"""
from __future__ import annotations

import shutil
import sys
import random
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC_V3 = ROOT / "datasets" / "helmet_crops_v3"
CLS_DS = ROOT / "datasets" / "helmet_cls"
MODELS = ROOT / "models"

EPOCHS = 10
IMGSZ  = 224      # standard classification size, works well on CPU
BATCH  = 64
PATIENCE = 5

# ── Build ImageNet-style classification dataset ──────────────────────────────

def _build_split(split: str, src_img_dir: Path, src_lbl_dir: Path,
                 dst_root: Path, class_map: dict[int, str],
                 max_per_class: int | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cls_id, cls_name in class_map.items():
        out_dir = dst_root / split / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # collect candidates
        candidates = []
        for lbl_file in src_lbl_dir.glob("*.txt"):
            try:
                first_cls = int(lbl_file.read_text().strip().splitlines()[0].split()[0])
            except Exception:
                continue
            if first_cls != cls_id:
                continue
            for ext in (".jpg", ".jpeg", ".png"):
                img = src_img_dir / (lbl_file.stem + ext)
                if img.exists():
                    candidates.append(img)
                    break

        random.shuffle(candidates)
        if max_per_class:
            candidates = candidates[:max_per_class]

        for img in candidates:
            shutil.copy(img, out_dir / img.name)
        counts[cls_name] = len(candidates)
    return counts


def build_cls_dataset() -> Path:
    print("Building ImageNet-style classification dataset …")
    if CLS_DS.exists():
        shutil.rmtree(CLS_DS)

    class_map = {0: "helmet", 1: "no_helmet"}

    # train — use balanced V3 dataset (3920:3920), cap at 1500/class for speed
    counts = _build_split(
        "train",
        SRC_V3 / "images" / "train",
        SRC_V3 / "labels" / "train",
        CLS_DS, class_map,
        max_per_class=1500,
    )
    print(f"  train: {counts}")

    # val — use original crops (no augmentation)
    src_orig = ROOT / "datasets" / "helmet_crops"
    counts_val = _build_split(
        "val",
        src_orig / "images" / "val",
        src_orig / "labels" / "val",
        CLS_DS, class_map,
    )
    print(f"  val:   {counts_val}")

    return CLS_DS


def train(cls_ds: Path) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining YOLOv8n-cls on {device}  epochs={EPOCHS}  imgsz={IMGSZ}  batch={BATCH}")

    model = YOLO("yolov8n-cls.pt")          # classification nano model
    results = model.train(
        data=str(cls_ds),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        project=str(ROOT / "runs" / "helmet_cls"),
        name="v3",
        exist_ok=True,
        patience=PATIENCE,
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
    print(f"\n✅ helmet_v3.pt (classification) saved → {target}")
    return target


if __name__ == "__main__":
    random.seed(42)
    cls_ds = build_cls_dataset()
    train(cls_ds)
    print("\nDone. Now update HelmetDetector to use classification mode.")
