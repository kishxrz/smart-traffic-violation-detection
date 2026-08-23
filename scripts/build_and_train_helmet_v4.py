"""
scripts/build_and_train_helmet_v4.py
──────────────────────────────────────────────────────────────────────────────
Helmet V4 — Clean NO_HELMET dataset + balanced augmented training.

Strategy vs V3:
  V3 problems identified:
    1. Val set: 1500 HELMET vs 77 NO_HELMET (15:1 imbalance → biased val signal)
    2. Training set: all 1310 NO_HELMET crops used — includes ambiguous/low-quality ones
    3. Augmentation: basic only (YOLO defaults)
    4. No sample filtering before training

  V4 fixes:
    1. Filter low-quality NO_HELMET crops (tiny, blurry, or already
       misclassified by V3 with high confidence → likely mislabeled).
    2. Balanced val: equal HELMET and NO_HELMET counts.
    3. Stronger synthetic augmentation applied OFFLINE before training
       (blur, brightness, JPEG compression, scale, rotation).
    4. Use ALL available NO_HELMET crops that pass quality filter.
    5. Match HELMET count to NO_HELMET count for perfect balance.

Dataset layout (ImageNet-style):
    datasets/helmet_v4/
        train/
            helmet/
            no_helmet/
        val/
            helmet/
            no_helmet/

Test set (UNTOUCHED):
    datasets/helmet_crops/images/test/    ← same held-out set as V3

Usage:
    python scripts/build_and_train_helmet_v4.py
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

backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED         = 42
TARGET_SIZE         = (224, 224)
EPOCHS              = 15
PATIENCE            = 5              # early stopping
BATCH               = 32
LR0                 = 0.001
DEVICE              = "cpu"

# Source datasets
CROPS_DIR           = Path("datasets/helmet_crops")
V4_DIR              = Path("datasets/helmet_v4")

# Output model
OUTPUT_MODEL        = Path("models/helmet_v4.pt")

# Quality filter: skip NO_HELMET crops where V3 is extremely confident HELMET
# (helmet_p > MISLABEL_THRESHOLD → likely mislabeled or genuinely ambiguous)
MISLABEL_THRESHOLD  = 0.95          # very high confidence → suspect mislabel
MIN_CROP_PX         = 40            # minimum dimension in pixels
MIN_SHARPNESS       = 50.0          # Laplacian variance → blur filter

# Augmentation per original image
AUG_COPIES_TRAIN    = 1             # 1 offline copy + online YOLO augmentation
AUG_COPIES_VAL      = 0             # no augmentation for validation

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Augmentation ──────────────────────────────────────────────────────────────
def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random combination of realistic augmentations."""
    aug = img.copy()

    # 1. Brightness/contrast
    alpha = float(rng.uniform(0.7, 1.4))   # contrast
    beta  = float(rng.uniform(-30, 30))    # brightness
    aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 2. Gaussian blur (simulate motion / distance)
    if rng.random() > 0.5:
        k = int(rng.choice([3, 5]))
        aug = cv2.GaussianBlur(aug, (k, k), 0)

    # 3. JPEG compression (simulate video codec artifacts)
    if rng.random() > 0.4:
        quality = int(rng.integers(40, 85))
        _, enc = cv2.imencode('.jpg', aug, [cv2.IMWRITE_JPEG_QUALITY, quality])
        aug = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 4. Scale variation: random crop then resize back
    if rng.random() > 0.5:
        h, w = aug.shape[:2]
        scale = float(rng.uniform(0.75, 0.95))
        sw, sh = int(w * scale), int(h * scale)
        x0 = int(rng.integers(0, w - sw + 1))
        y0 = int(rng.integers(0, h - sh + 1))
        aug = cv2.resize(aug[y0:y0+sh, x0:x0+sw], (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 5. Mild rotation (±12°)
    if rng.random() > 0.5:
        angle = float(rng.uniform(-12, 12))
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 6. Horizontal flip
    if rng.random() > 0.5:
        aug = cv2.flip(aug, 1)

    # 7. Noise
    if rng.random() > 0.6:
        noise = rng.integers(-15, 15, aug.shape, dtype=np.int16)
        aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return aug


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Load + filter ─────────────────────────────────────────────────────────────
def load_crops(cls_id: int, splits: list[str]) -> list[Path]:
    """Load all crop paths for a given class from specified splits."""
    paths = []
    for split in splits:
        lbl_dir = CROPS_DIR / "labels" / split
        img_dir = CROPS_DIR / "images" / split
        for lbl in lbl_dir.glob("*.txt"):
            try:
                c = int(lbl.read_text().strip().splitlines()[0].split()[0])
            except Exception:
                continue
            if c != cls_id:
                continue
            for ext in (".jpg", ".jpeg", ".png"):
                img_f = img_dir / (lbl.stem + ext)
                if img_f.exists():
                    paths.append(img_f)
                    break
    return paths


def filter_no_helmet(paths: list[Path], model) -> tuple[list[Path], dict]:
    """
    Filter NO_HELMET crops removing:
      - Images too small
      - Images too blurry
      - Images V3 predicts HELMET with very high confidence (likely mislabeled)
    Returns (clean_paths, stats).
    """
    clean, rejected_small, rejected_blur, rejected_mislabel = [], 0, 0, 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Size filter
        if w < MIN_CROP_PX or h < MIN_CROP_PX:
            rejected_small += 1
            continue

        # Blur filter
        if sharpness(img) < MIN_SHARPNESS:
            rejected_blur += 1
            continue

        # Mislabel filter: if V3 is extremely confident HELMET → skip
        r = model(img, verbose=False)
        probs = r[0].probs
        helmet_p = float(probs.data[0])
        if helmet_p > MISLABEL_THRESHOLD:
            rejected_mislabel += 1
            continue

        clean.append(p)

    stats = {
        "total": len(paths),
        "clean": len(clean),
        "rejected_small": rejected_small,
        "rejected_blur": rejected_blur,
        "rejected_mislabel": rejected_mislabel,
    }
    return clean, stats


# ── Dataset builder ───────────────────────────────────────────────────────────
def write_split(
    paths: list[Path],
    cls_name: str,
    out_dir: Path,
    augment_copies: int,
    rng: np.random.Generator,
) -> int:
    """Write original + augmented images to out_dir/cls_name/. Returns count."""
    dest = out_dir / cls_name
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_r = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

        # Original
        out_path = dest / f"orig_{p.stem}.jpg"
        cv2.imwrite(str(out_path), img_r, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1

        # Augmented copies
        for k in range(augment_copies):
            aug = augment(img_r, rng)
            out_path = dest / f"aug{k}_{p.stem}.jpg"
            cv2.imwrite(str(out_path), aug, [cv2.IMWRITE_JPEG_QUALITY, 88])
            n += 1
    return n


def build_dataset() -> dict:
    from ultralytics import YOLO
    print("\n=== STEP 1: Load V3 model for quality filtering ===")
    v3 = YOLO("models/helmet_v3.pt")

    print("\n=== STEP 2: Load source crops ===")
    # Use train+val for training (keeping test untouched)
    nh_all = load_crops(cls_id=1, splits=["train", "val"])
    h_all  = load_crops(cls_id=0, splits=["train", "val"])
    print(f"  Raw NO_HELMET (train+val): {len(nh_all)}")
    print(f"  Raw HELMET    (train+val): {len(h_all)}")

    print("\n=== STEP 3: Filter NO_HELMET for quality ===")
    nh_clean, nh_stats = filter_no_helmet(nh_all, v3)
    print(f"  Total:             {nh_stats['total']}")
    print(f"  Clean:             {nh_stats['clean']}")
    print(f"  Rejected (small):  {nh_stats['rejected_small']}")
    print(f"  Rejected (blur):   {nh_stats['rejected_blur']}")
    print(f"  Rejected (mislabel confidence > {MISLABEL_THRESHOLD:.0%}): {nh_stats['rejected_mislabel']}")

    # Shuffle and split clean NO_HELMET: 85% train, 15% val
    random.shuffle(nh_clean)
    n_val_nh = max(40, int(len(nh_clean) * 0.15))
    nh_val   = nh_clean[:n_val_nh]
    nh_train = nh_clean[n_val_nh:]
    print(f"\n  NO_HELMET train: {len(nh_train)}  val: {len(nh_val)}")

    # Balance HELMET: sample to match NO_HELMET counts (with augmentation budget)
    # After augmentation, NO_HELMET train = len(nh_train) * (1 + AUG_COPIES_TRAIN)
    # We want HELMET train ≈ same total → sample proportionally
    target_h_train = len(nh_train)  # same base count; augmentation applied to both
    target_h_val   = len(nh_val)

    random.shuffle(h_all)
    h_train = h_all[:target_h_train]
    h_val   = h_all[target_h_train:target_h_train + target_h_val]
    if len(h_val) < target_h_val:
        # wrap-around if needed
        h_val = h_all[-target_h_val:]
    print(f"  HELMET    train: {len(h_train)}  val: {len(h_val)}")

    print("\n=== STEP 4: Build dataset directory ===")
    if V4_DIR.exists():
        shutil.rmtree(V4_DIR)
    V4_DIR.mkdir(parents=True)

    rng = np.random.default_rng(RANDOM_SEED)

    print("  Writing train/no_helmet ...")
    n_nh_train = write_split(nh_train, "no_helmet", V4_DIR/"train", AUG_COPIES_TRAIN, rng)
    print("  Writing train/helmet ...")
    n_h_train  = write_split(h_train,  "helmet",    V4_DIR/"train", AUG_COPIES_TRAIN, rng)
    print("  Writing val/no_helmet ...")
    n_nh_val   = write_split(nh_val,   "no_helmet", V4_DIR/"val",   AUG_COPIES_VAL,  rng)
    print("  Writing val/helmet ...")
    n_h_val    = write_split(h_val,    "helmet",    V4_DIR/"val",   AUG_COPIES_VAL,  rng)

    print(f"\n  Train: {n_nh_train} NO_HELMET  +  {n_h_train} HELMET  = {n_nh_train+n_h_train} total")
    print(f"  Val:   {n_nh_val} NO_HELMET    +  {n_h_val} HELMET    = {n_nh_val+n_h_val} total")
    print(f"  Balance ratio train: {n_nh_train/max(1,n_h_train):.2f}")
    print(f"  Balance ratio val:   {n_nh_val/max(1,n_h_val):.2f}")

    return {
        "nh_train": n_nh_train, "h_train": n_h_train,
        "nh_val": n_nh_val,     "h_val":   n_h_val,
    }


# ── Training ──────────────────────────────────────────────────────────────────
def train() -> Path:
    from ultralytics import YOLO
    print("\n=== STEP 5: Train YOLOv8n-cls V4 ===")
    print(f"  Dataset : {V4_DIR}")
    print(f"  imgsz   : {TARGET_SIZE[0]}")
    print(f"  epochs  : {EPOCHS} (patience={PATIENCE})")
    print(f"  batch   : {BATCH}")
    print(f"  device  : {DEVICE}")

    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=str(V4_DIR),
        epochs=EPOCHS,
        imgsz=TARGET_SIZE[0],
        batch=BATCH,
        lr0=LR0,
        patience=PATIENCE,
        device=DEVICE,
        project="runs/helmet_v4",
        name="train",
        exist_ok=True,
        verbose=False,
        # Augmentation (YOLO cls defaults + ours applied offline)
        augment=True,
        degrees=10,
        translate=0.1,
        scale=0.2,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
    )

    # Copy best weights
    best = Path("runs/helmet_v4/train/weights/best.pt")
    if best.exists():
        shutil.copy(best, OUTPUT_MODEL)
        print(f"\n  Saved: {OUTPUT_MODEL}  ({OUTPUT_MODEL.stat().st_size//1024} KB)")
    else:
        print("  WARNING: best.pt not found!")
    return OUTPUT_MODEL


# ── Evaluation on held-out test set ──────────────────────────────────────────
def evaluate(model_path: Path) -> dict:
    from ultralytics import YOLO
    print(f"\n=== STEP 6: Evaluate {model_path.name} on held-out test set ===")
    model = YOLO(str(model_path))

    lbl_dir = CROPS_DIR / "labels" / "test"
    img_dir = CROPS_DIR / "images" / "test"

    tp_nh, fp_nh, fn_nh, tn_nh = 0, 0, 0, 0  # NO_HELMET metrics
    tp_h,  fp_h,  fn_h,  tn_h  = 0, 0, 0, 0  # HELMET metrics

    for lbl in lbl_dir.glob("*.txt"):
        try:
            true_cls = int(lbl.read_text().strip().splitlines()[0].split()[0])
        except Exception:
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            img_f = img_dir / (lbl.stem + ext)
            if img_f.exists():
                break
        else:
            continue
        img = cv2.imread(str(img_f))
        if img is None:
            continue

        r = model(img, verbose=False)
        pred_cls = int(r[0].probs.top1)

        if true_cls == 1:   # ground truth NO_HELMET
            if pred_cls == 1: tp_nh += 1
            else:             fn_nh += 1
        else:               # ground truth HELMET
            if pred_cls == 1: fp_nh += 1
            else:             tp_h  += 1

    total = tp_nh + fn_nh + fp_nh + tp_h
    correct = tp_nh + tp_h
    acc = correct / max(1, total)

    prec_nh = tp_nh / max(1, tp_nh + fp_nh)
    rec_nh  = tp_nh / max(1, tp_nh + fn_nh)
    f1_nh   = 2 * prec_nh * rec_nh / max(1e-9, prec_nh + rec_nh)

    total_h = tp_h + fp_nh   # all predicted HELMET
    prec_h  = tp_h / max(1, tp_h + fn_nh)   # helmet prec = correct helmet / all helmet GT
    rec_h   = tp_h / max(1, tp_h + fn_nh)   # same denominator for this binary case

    print(f"\n  Test set total: {total}  ({tp_nh+fn_nh} NO_HELMET  +  {tp_h+fp_nh} HELMET GT)")
    print(f"\n  Overall Accuracy : {acc:.3f}  ({correct}/{total})")
    print(f"\n  NO_HELMET:")
    print(f"    Precision : {prec_nh:.3f}  ({tp_nh} TP / {tp_nh+fp_nh} predicted)")
    print(f"    Recall    : {rec_nh:.3f}  ({tp_nh} TP / {tp_nh+fn_nh} actual)")
    print(f"    F1        : {f1_nh:.3f}")
    print(f"\n  HELMET:")
    print(f"    Precision : {tp_h}/{tp_h+fn_nh} = {tp_h/max(1,tp_h+fn_nh):.3f}")
    print(f"    Recall    : {tp_h}/{tp_h+fp_nh} = {tp_h/max(1,tp_h+fp_nh):.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"                    Pred HELMET    Pred NO_HELMET")
    print(f"    True HELMET  :  {tp_h:>5}            {fn_nh:>5}")
    print(f"    True NO_HELMET: {fp_nh:>5}            {tp_nh:>5}")

    return {
        "accuracy": acc, "prec_nh": prec_nh, "rec_nh": rec_nh, "f1_nh": f1_nh,
        "tp_nh": tp_nh, "fp_nh": fp_nh, "fn_nh": fn_nh, "tp_h": tp_h,
    }


# ── V3 baseline evaluation ────────────────────────────────────────────────────
def evaluate_v3() -> dict:
    print("\n=== BASELINE: V3 evaluation on test set ===")
    return evaluate(Path("models/helmet_v3.pt"))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    # 1. Evaluate V3 first (baseline)
    v3_metrics = evaluate_v3()

    # 2. Build V4 dataset
    ds_stats = build_dataset()

    # 3. Train V4
    output = train()

    # 4. Evaluate V4
    v4_metrics = evaluate(output)

    # 5. Compare
    print("\n" + "=" * 70)
    print(" V3 vs V4 COMPARISON (held-out test set)")
    print("=" * 70)
    print(f"  {'Metric':<20} {'V3':>8}  {'V4':>8}  {'Delta':>8}")
    print(f"  {'-'*48}")
    for key, label in [
        ("accuracy", "Overall Accuracy"),
        ("prec_nh",  "NO_HELMET Prec"),
        ("rec_nh",   "NO_HELMET Recall"),
        ("f1_nh",    "NO_HELMET F1"),
    ]:
        v3v = v3_metrics[key]
        v4v = v4_metrics[key]
        delta = v4v - v3v
        symbol = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
        print(f"  {label:<20} {v3v:>8.3f}  {v4v:>8.3f}  {symbol}{abs(delta):>6.3f}")

    print(f"\n  NO_HELMET test: V3={v3_metrics['tp_nh']}/{v3_metrics['tp_nh']+v3_metrics['fn_nh']}  "
          f"V4={v4_metrics['tp_nh']}/{v4_metrics['tp_nh']+v4_metrics['fn_nh']}")
    print(f"  HELMET false positives (false NO_HELMET): V3={v3_metrics['fp_nh']}  V4={v4_metrics['fp_nh']}")
    print(f"\n  Training time: {(time.time()-t0)/60:.1f} min")
    print("=" * 70)

    # 6. Verdict
    nh_improved = v4_metrics["rec_nh"] > v3_metrics["rec_nh"] + 0.05
    fp_acceptable = v4_metrics["fp_nh"] <= v3_metrics["fp_nh"] + 10  # at most 10 more FP
    if nh_improved and fp_acceptable:
        print("\n  → V4 NO_HELMET recall improved without excessive false positives.")
    elif nh_improved:
        print("\n  → V4 NO_HELMET recall improved BUT false positive count increased.")
    else:
        print("\n  → V4 did NOT significantly improve NO_HELMET recall.")
    print(f"\n  Model saved: {output}")
