# models/

This directory stores YOLO model weights and any custom-trained models.

## Contents

| File | Description | Size |
|------|-------------|------|
| `yolov8n.pt` | YOLOv8 Nano (COCO pretrained) | ~6 MB |
| `yolov8s.pt` | YOLOv8 Small | ~22 MB |
| `yolov8m.pt` | YOLOv8 Medium | ~52 MB |
| `helmet_v1.pt` | Custom helmet detector *(not yet trained)* | — |

## Downloading Pretrained Models

```bash
python scripts/download_models.py --size n
```

This uses Ultralytics' auto-download to fetch the COCO-pretrained weights.

## Custom Model Requirements

### Helmet Detection Model

Classes required:
```
0: helmet
1: no_helmet
```

Dataset: annotate images from traffic camera footage showing motorcycle
riders, with and without helmets. Roboflow datasets are a good starting point.

Training:
```bash
python scripts/train.py --task helmet --data datasets/helmet/data.yaml --epochs 100
```

Expected minimum dataset size: 5,000 images (2,500 per class).

## Git: Model Weights Are Gitignored

All `.pt`, `.onnx`, `.pth` files are excluded from version control.
Models should be stored in cloud storage (S3, GCS) and downloaded on deployment.
