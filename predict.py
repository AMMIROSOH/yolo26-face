from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO26 face + keypoint inference on images, videos, folders, or webcam streams."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/pose/face/yolo26n/weights/best.pt"),
        help="Path to a trained YOLO pose checkpoint.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image path, directory, video path, URL, or webcam index such as 0.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", type=str, default="", help="Device, for example 0 or cpu.")
    parser.add_argument("--project", type=Path, default=Path("runs/pose/tests"))
    parser.add_argument("--name", type=str, default="predict")
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--show", action="store_true", help="Show live predictions in a window.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO text predictions.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence values to text output.")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def coerce_source(raw_source: str) -> str | int:
    if raw_source.isdigit():
        return int(raw_source)
    return raw_source


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights.resolve()))

    results = model.predict(
        source=coerce_source(args.source),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        line_width=args.line_width,
        show=args.show,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        exist_ok=args.exist_ok,
    )

    save_dir = Path(results[0].save_dir).resolve() if results else args.project.resolve() / args.name
    print(f"[predict] Saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
