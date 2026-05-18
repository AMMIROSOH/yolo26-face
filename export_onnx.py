from __future__ import annotations

import argparse
from pathlib import Path

import onnx
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
        "--batch",
        type=int,
        default=1,
        help="Export batch size. Use this together with --dynamic-batch to define the max traced batch shape.",
    )
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
        help="Export with fully dynamic ONNX shapes, including height and width.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with dynamic batch only, while keeping width and height fixed at --imgsz.",
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


def patch_batch_dynamic(onnx_path: Path) -> None:
    model = onnx.load(str(onnx_path))

    def make_batch_dynamic(value_info) -> None:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape") or not tensor_type.shape.dim:
            return
        batch_dim = tensor_type.shape.dim[0]
        batch_dim.ClearField("dim_value")
        batch_dim.dim_param = "batch"

    for value_info in model.graph.input:
        make_batch_dynamic(value_info)
    for value_info in model.graph.output:
        make_batch_dynamic(value_info)

    onnx.save(model, str(onnx_path))


def main() -> None:
    args = parse_args()
    if args.dynamic and args.dynamic_batch:
        raise ValueError("Use either --dynamic or --dynamic-batch, not both.")

    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Missing weights file: {weights}")

    model = YOLO(str(weights))
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        opset=args.opset,
        dynamic=args.dynamic,
        half=args.half,
        simplify=args.simplify,
    )
    exported_path = Path(exported_path).resolve()

    if args.dynamic_batch:
        patch_batch_dynamic(exported_path)
        print(
            f"[onnx] Patched batch dimension to dynamic while keeping spatial size fixed at {args.imgsz}x{args.imgsz}"
        )

    print(f"[onnx] Exported model to: {exported_path}")


if __name__ == "__main__":
    main()
