# YOLO26 Face 👤

Reference repo for training **YOLO26** on face detection with **5 facial keypoints** using a WIDER Face style dataset converted to Ultralytics YOLO pose labels.

> Lightweight reference project for training, benchmarking, and exporting YOLO26 face models.

## ✨ Overview

- 🚀 train YOLO26 face models with `train.py`
- 🔎 run inference with `predict.py`
- 🧩 convert WIDER-style labels with `dataset/train2yolo.py` and `dataset/val2yolo.py`
- 📦 export trained checkpoints to ONNX with `export_onnx.py`

## 🔗 Model Links

- 🤗 Hugging Face: [`ammirosoh/yolo26n-face`](https://huggingface.co/ammirosoh/yolo26n-face)
- 🤗 Hugging Face: [`ammirosoh/yolo26s-face`](https://huggingface.co/ammirosoh/yolo26s-face)

## 🗂️ Dataset Layout

```text
dataset/
  data.yaml
  train/
    image_1.jpg
    image_1.txt
    ...
  val/
    image_1.jpg
    image_1.txt
    ...
```

Use [`dataset/data.yaml.example`](dataset/data.yaml.example) as the starting point for your local config.

Expected label format:

```text
class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v kp3_x kp3_y kp3_v kp4_x kp4_y kp4_v kp5_x kp5_y kp5_v
```

## ⚙️ Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 🔄 Data Conversion

Train split:

```bash
python dataset/train2yolo.py D:\Datasets\WIDERFACE\train dataset\train
```

Validation split:

```bash
python dataset/val2yolo.py D:\Datasets\WIDERFACE dataset\val
```

Note: the current validation converter writes face boxes but fills keypoints with `0 0 0`, so pose validation metrics are not meaningful unless your validation set has real landmark labels.

## 🏋️ Training

`yolo26n`

```bash
python train.py --data dataset/data.yaml --weights yolo26n.pt --epochs 100 --imgsz 640 --batch 16 --cache --project runs/pose/face --name yolo26n --exist-ok
```

`yolo26s`

```bash
python train.py --data dataset/data.yaml --weights yolo26s.pt --epochs 100 --imgsz 640 --batch 16 --cache --project runs/pose/face --name yolo26s --exist-ok
```

If you pass a detect checkpoint such as `yolo26n.pt` or `yolo26s.pt`, `train.py` now detects the correct scale and upgrades it to the matching pose model automatically.

## 🎯 Inference

```bash
python predict.py --weights runs/pose/face/yolo26s/weights/best.pt --source path/to/image.jpg
```

## 📊 Benchmarks

Current validation box metrics from the committed runs:

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `yolo26n-face` | 0.84635 | 0.60386 | 0.69276 | 0.37801 |
| `yolo26s-face` | 0.86803 | 0.66325 | 0.73280 | 0.40367 |
| `yolo26m-face` | pending | pending | pending | pending |

Quick read:

- 🥈 `yolo26s-face` is currently ahead of `yolo26n-face` on all committed box metrics
- 🧪 `yolo26m-face` remains pending until its final run results are added
- 🖼️ `yolo26s` result images are intentionally not embedded in this README

### Snapshot

| Variant | Status | Notes |
| --- | --- | --- |
| `n` | released | baseline public checkpoint and plots |
| `s` | released | stronger benchmark numbers than `n` |
| `m` | pending | benchmark row reserved for final results |

## 📦 Export To ONNX

Export a trained checkpoint:

```bash
python export_onnx.py --weights runs/pose/face/yolo26s/weights/best.pt --imgsz 640 --opset 12 --simplify
```

Dynamic-shape export:

```bash
python export_onnx.py --weights runs/pose/face/yolo26s/weights/best.pt --imgsz 640 --dynamic
```

## 📝 Notes

- `train.py` validates dataset structure before training starts.
- If result plotting fails because Ultralytics cannot download a font on a remote machine, training now continues and still saves weights and `results.csv`.
- Large local artifacts and local dataset config are ignored by default via [`.gitignore`](.gitignore).
