from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO face checkpoint to ONNX.")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to a trained checkpoint, for example runs/pose/face/yolo26s/weights/best.pt.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Export device, usually cpu or 0.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic input shapes.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16. Use only when exporting on compatible GPU hardware.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the exported ONNX graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Missing weights file: {weights}")

    model = YOLO(str(weights))
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        device=args.device,
        opset=args.opset,
        dynamic=args.dynamic,
        half=args.half,
        simplify=args.simplify,
    )

    print(f"[onnx] Exported model to: {Path(exported_path).resolve()}")


if __name__ == "__main__":
    main()
