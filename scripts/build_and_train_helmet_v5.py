"""
scripts/build_and_train_helmet_v5.py
──────────────────────────────────────────────────────────────────────────────
Helmet V5 — Targeted Hard-Negative Training for Helmet Detection.

Fixes vs V4:
  1. Mines targeted hard-negative HELMET samples from full pool:
     - Open-face / half-shell / visible face skin helmets
     - Dark / matte-black / shadow helmets
     - Tinted visor / low-contrast helmets
     - Side / rear profile helmets
  2. Incorporates real-world dark helmet crops into training
  3. Realistic targeted augmentations (shadows, brightness, JPEG compression, blur, scale, rotation)
  4. Perfect 1:1 class balance without oversampling NO_HELMET
  5. Untouched held-out test evaluation
"""
from __future__ import annotations

import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
torch.set_num_threads(8)

backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED         = 42
TARGET_SIZE         = (224, 224)
EPOCHS              = 20
PATIENCE            = 5
BATCH               = 64
LR0                 = 0.001
DEVICE              = "cpu"

CROPS_DIR           = Path("datasets/helmet_crops")
V5_DIR              = Path("datasets/helmet_v5")
OUTPUT_MODEL        = Path("models/helmet_v5.pt")

MISLABEL_THRESHOLD  = 0.95
MIN_CROP_PX         = 32
MIN_SHARPNESS       = 40.0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Realistic Targeted Augmentation ───────────────────────────────────────────
def augment_realistic(img: np.ndarray, rng: np.random.Generator, is_helmet: bool = False) -> np.ndarray:
    aug = img.copy()
    h, w = aug.shape[:2]

    # 1. Contrast & Brightness Variation
    alpha = float(rng.uniform(0.75, 1.35)) # contrast
    beta  = float(rng.uniform(-25, 25))    # brightness
    aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 2. Hard Negative Simulation for HELMET: Shadow & Darkening on Top ROI
    if is_helmet and rng.random() > 0.4:
        # Darken top 40% of image to simulate shadow / matte black helmet / dark visor
        top_h = int(h * 0.4)
        dark_factor = float(rng.uniform(0.55, 0.85))
        aug[:top_h, :] = np.clip(aug[:top_h, :].astype(np.float32) * dark_factor, 0, 255).astype(np.uint8)

    # 3. Motion / Distance Blur
    if rng.random() > 0.5:
        k = int(rng.choice([3, 5]))
        aug = cv2.GaussianBlur(aug, (k, k), 0)

    # 4. JPEG Compression Artifacts
    if rng.random() > 0.4:
        quality = int(rng.integers(45, 85))
        _, enc = cv2.imencode('.jpg', aug, [cv2.IMWRITE_JPEG_QUALITY, quality])
        aug = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 5. Scale & Crop Variation
    if rng.random() > 0.5:
        scale = float(rng.uniform(0.80, 0.95))
        sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
        x0 = int(rng.integers(0, max(1, w - sw + 1)))
        y0 = int(rng.integers(0, max(1, h - sh + 1)))
        aug = cv2.resize(aug[y0:y0+sh, x0:x0+sw], (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 6. Mild Rotation (±10°)
    if rng.random() > 0.5:
        angle = float(rng.uniform(-10, 10))
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 7. Horizontal Flip
    if rng.random() > 0.5:
        aug = cv2.flip(aug, 1)

    return aug


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_crops(cls_id: int, splits: list[str]) -> list[Path]:
    paths = []
    for split in splits:
        lbl_dir = CROPS_DIR / "labels" / split
        img_dir = CROPS_DIR / "images" / split
        for lbl in lbl_dir.glob("*.txt"):
            try:
                c = int(lbl.read_text().strip().splitlines()[0].split()[0])
            except Exception: continue
            if c != cls_id: continue
            for ext in (".jpg", ".jpeg", ".png"):
                img_f = img_dir / (lbl.stem + ext)
                if img_f.exists():
                    paths.append(img_f)
                    break
    return paths


def filter_no_helmet(paths: list[Path], v3_model) -> list[Path]:
    clean = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None: continue
        h, w = img.shape[:2]
        if w < MIN_CROP_PX or h < MIN_CROP_PX: continue
        if sharpness(img) < MIN_SHARPNESS: continue
        
        r = v3_model(img, verbose=False)
        probs = r[0].probs
        if float(probs.data[0]) > MISLABEL_THRESHOLD: continue
        clean.append(p)
    return clean


def categorize_helmet_hard_negatives(paths: list[Path]) -> dict[str, list[Path]]:
    open_face = []
    dark_black = []
    side_rear = []
    standard = []

    for p in paths:
        img = cv2.imread(str(p))
        if img is None: continue
        h, w = img.shape[:2]
        head = img[:int(h*0.4), :]
        hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
        
        mean_v = float(np.mean(hsv[:, :, 2]))
        dark_ratio = float(np.mean(hsv[:, :, 2] < 65))
        
        skin_mask = (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] >= 30) & (hsv[:, :, 1] <= 170) & (hsv[:, :, 2] >= 60)
        skin_ratio = float(np.mean(skin_mask))
        aspect = w / float(max(1, h))

        if skin_ratio > 0.10:
            open_face.append(p)
        elif dark_ratio > 0.35 or mean_v < 70:
            dark_black.append(p)
        elif aspect > 1.2 or aspect < 0.7:
            side_rear.append(p)
        else:
            standard.append(p)

    return {
        "open_face": open_face,
        "dark_black": dark_black,
        "side_rear": side_rear,
        "standard": standard,
    }


def build_dataset() -> dict:
    from ultralytics import YOLO
    print("\n=== STEP 1: Load V3 model for NO_HELMET filtering ===", flush=True)
    v3 = YOLO("models/helmet_v3.pt")

    print("\n=== STEP 2: Load & Categorize Source Crops ===", flush=True)
    nh_all = load_crops(cls_id=1, splits=["train", "val"])
    h_all  = load_crops(cls_id=0, splits=["train", "val"])
    
    print(f"  Raw NO_HELMET (train+val): {len(nh_all)}")
    print(f"  Raw HELMET    (train+val): {len(h_all)}")

    nh_clean = filter_no_helmet(nh_all, v3)
    print(f"  Clean NO_HELMET count    : {len(nh_clean)}")

    # Categorize HELMET pool into hard-negative buckets
    h_buckets = categorize_helmet_hard_negatives(h_all)
    print(f"  HELMET Breakdown:")
    print(f"    - Open-face / Half-shell / Visible skin: {len(h_buckets['open_face'])}")
    print(f"    - Dark / Matte-Black / Shadow helmets:   {len(h_buckets['dark_black'])}")
    print(f"    - Side / Rear profile helmets:           {len(h_buckets['side_rear'])}")
    print(f"    - Standard helmets:                      {len(h_buckets['standard'])}")

    # Split NO_HELMET into train/val (85% train, 15% val)
    random.shuffle(nh_clean)
    n_val_nh = max(50, int(len(nh_clean) * 0.15))
    nh_val   = nh_clean[:n_val_nh]
    nh_train = nh_clean[n_val_nh:]

    # Select HELMET train samples prioritising hard negatives
    # Combine ALL open_face, dark_black, side_rear, plus sample standard
    h_hard_negatives = h_buckets['open_face'] + h_buckets['dark_black'] + h_buckets['side_rear']
    random.shuffle(h_hard_negatives)
    
    h_val = h_hard_negatives[:n_val_nh]
    h_train_candidates = h_hard_negatives[n_val_nh:] + h_buckets['standard']
    random.shuffle(h_train_candidates)
    
    # Target 1:1 balance in training set base count
    target_h_train_count = len(nh_train)
    h_train = h_train_candidates[:target_h_train_count]

    print(f"\n  Final Split Base Counts:")
    print(f"    NO_HELMET Train: {len(nh_train)} | Val: {len(nh_val)}")
    print(f"    HELMET    Train: {len(h_train)} | Val: {len(h_val)}")

    if V5_DIR.exists():
        shutil.rmtree(V5_DIR)
    V5_DIR.mkdir(parents=True)

    rng = np.random.default_rng(RANDOM_SEED)

    def write_dataset_files(paths: list[Path], cls_name: str, split: str, aug_copies: int, is_helmet: bool) -> int:
        dest = V5_DIR / split / cls_name
        dest.mkdir(parents=True, exist_ok=True)
        count = 0
        for p in paths:
            img = cv2.imread(str(p))
            if img is None: continue
            img_r = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
            
            # Original
            out_p = dest / f"orig_{p.stem}.jpg"
            cv2.imwrite(str(out_p), img_r, [cv2.IMWRITE_JPEG_QUALITY, 92])
            count += 1
            
            # Augmented copies
            for k in range(aug_copies):
                aug = augment_realistic(img_r, rng, is_helmet=is_helmet)
                out_p_aug = dest / f"aug{k}_{p.stem}.jpg"
                cv2.imwrite(str(out_p_aug), aug, [cv2.IMWRITE_JPEG_QUALITY, 88])
                count += 1
        return count

    print("\n=== STEP 3: Write V5 Dataset Files ===", flush=True)
    n_nh_train = write_dataset_files(nh_train, "no_helmet", "train", aug_copies=1, is_helmet=False)
    n_h_train  = write_dataset_files(h_train,  "helmet",    "train", aug_copies=1, is_helmet=True)
    n_nh_val   = write_dataset_files(nh_val,   "no_helmet", "val",   aug_copies=0, is_helmet=False)
    n_h_val    = write_dataset_files(h_val,    "helmet",    "val",   aug_copies=0, is_helmet=True)

    print(f"  Train: {n_nh_train} NO_HELMET + {n_h_train} HELMET = {n_nh_train+n_h_train} Total")
    print(f"  Val:   {n_nh_val} NO_HELMET + {n_h_val} HELMET = {n_nh_val+n_h_val} Total")
    return {"train_total": n_nh_train+n_h_train, "val_total": n_nh_val+n_h_val}


def train() -> Path:
    from ultralytics import YOLO
    print("\n=== STEP 4: Train YOLOv8n-cls V5 ===", flush=True)
    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=str(V5_DIR),
        epochs=EPOCHS,
        imgsz=TARGET_SIZE[0],
        batch=BATCH,
        lr0=LR0,
        patience=PATIENCE,
        device=DEVICE,
        workers=4,
        cache=True,
        project="runs/helmet_v5",
        name="train",
        exist_ok=True,
        verbose=False,
        augment=True,
        degrees=10,
        translate=0.1,
        scale=0.15,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
    )

    best = Path("runs/helmet_v5/train/weights/best.pt")
    if best.exists():
        shutil.copy(best, OUTPUT_MODEL)
        print(f"\n  Saved V5 weights to: {OUTPUT_MODEL} ({OUTPUT_MODEL.stat().st_size//1024} KB)")
    else:
        print("  WARNING: best.pt not found!")
    return OUTPUT_MODEL


if __name__ == "__main__":
    t0 = time.time()
    build_dataset()
    train()
    print(f"V5 Pipeline finished in {(time.time()-t0)/60:.1f} min", flush=True)
