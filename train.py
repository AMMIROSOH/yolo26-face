from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results


EXPECTED_KPT_SHAPE = [5, 3]
EXPECTED_LABEL_VALUES = 5 + EXPECTED_KPT_SHAPE[0] * EXPECTED_KPT_SHAPE[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune YOLO26 for face detection + 5 facial keypoints."
    )
    parser.add_argument("--data", type=Path, help="path of dataset .yml file.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolo26n.pt"),
        help="Detect or pose checkpoint. Detect checkpoints are upgraded to pose automatically.",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default="yolo26-pose.yaml",
        help="Pose model YAML used when --weights is a detect checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/pose/face"))
    parser.add_argument("--name", type=str, default="yolo26_face_kps")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/face"),
        help="Export final public plots here using the same filenames as akanametov/yolo-face.",
    )
    # 5e-4 for nano models
    parser.add_argument("--lr0", type=float, default=0.00038)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--pose-loss", type=float, default=12.0)
    parser.add_argument("--kobj-loss", type=float, default=2.0)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--final-val",
        action="store_true",
        help="Run one extra validation pass on best.pt after training. This is slower and usually unnecessary.",
    )
    parser.add_argument(
        "--label-check-files",
        type=int,
        default=8,
        help="How many label files to sample from train/val for format checks.",
    )
    return parser.parse_args()


def load_data_yaml(data_yaml: Path) -> dict:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Missing data YAML: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data


def resolve_split_dir(split_value: str | Path, data_yaml: Path) -> Path:
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return (data_yaml.parent / split_path).resolve()


def iter_label_files(images_dir: Path):
    for suffix in ("*.txt",):
        yield from sorted(images_dir.glob(suffix))


def validate_dataset_layout(data_yaml: Path, label_check_files: int) -> None:
    data = load_data_yaml(data_yaml)
    if data.get("nc") != 1:
        raise ValueError(f"Expected nc=1 for a single face class, got: {data.get('nc')}")
    if list(data.get("kpt_shape", [])) != EXPECTED_KPT_SHAPE:
        raise ValueError(
            f"Expected kpt_shape={EXPECTED_KPT_SHAPE}, got: {data.get('kpt_shape')}"
        )

    for split_name in ("train", "val"):
        split_dir = resolve_split_dir(data[split_name], data_yaml)
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing {split_name} directory: {split_dir}")

        checked = 0
        for label_path in iter_label_files(split_dir):
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                continue

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != EXPECTED_LABEL_VALUES:
                    raise ValueError(
                        f"{label_path} line {line_number} has {len(parts)} values, "
                        f"expected {EXPECTED_LABEL_VALUES} for 5 keypoints."
                    )
                if parts[0] != "0":
                    raise ValueError(
                        f"{label_path} line {line_number} uses class {parts[0]}, expected class 0."
                    )

            checked += 1
            if checked >= label_check_files:
                break

        if checked == 0:
            raise FileNotFoundError(f"No non-empty label files found in {split_dir}")


def build_model(weights: Path, pose_model: str) -> YOLO:
    pretrained = YOLO(str(weights))
    if pretrained.task == "pose":
        print(f"[model] Loaded pose checkpoint directly: {weights}")
        return pretrained

    if pretrained.task != "detect":
        raise ValueError(
            f"Unsupported checkpoint task '{pretrained.task}'. Provide a detect or pose checkpoint."
        )

    model = YOLO(pose_model).load(str(weights))
    print(
        f"[model] Built pose model from {pose_model} and transferred weights from detect checkpoint {weights}"
    )
    return model


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def clean_results_csv(results_csv: Path) -> None:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return
    rows[0] = [column.strip() for column in rows[0]]
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def resolve_public_file(primary_dir: Path, train_dir: Path, candidates: list[str]) -> Path | None:
    for directory in (primary_dir, train_dir):
        for name in candidates:
            path = directory / name
            if path.is_file():
                return path
    return None


def export_artifacts(
    train_dir: Path,
    public_results_dir: Path,
    val_dir: Path | None = None,
) -> dict[str, Path]:
    public_results_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    primary_dir = val_dir if val_dir is not None else train_dir

    results_csv = train_dir / "results.csv"
    if results_csv.is_file():
        clean_results_csv(results_csv)
        plot_results(file=str(results_csv))

    export_map = {
        "P_curve.png": ["P_curve.png", "BoxP_curve.png"],
        "PR_curve.png": ["PR_curve.png", "BoxPR_curve.png"],
        "R_curve.png": ["R_curve.png", "BoxR_curve.png"],
        "results.png": ["results.png"],
        "confusion_matrix.png": ["confusion_matrix.png", "confusion_matrix_normalized.png"],
    }

    for public_name, candidates in export_map.items():
        source = resolve_public_file(primary_dir, train_dir, candidates)
        if source is None:
            continue
        destination = public_results_dir / public_name
        shutil.copy2(source, destination)
        outputs[public_name] = destination

    return outputs


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.weights = args.weights.resolve()
    args.project = args.project.resolve()
    args.results_dir = args.results_dir.resolve()

    validate_dataset_layout(args.data, args.label_check_files)

    model = build_model(args.weights, args.pose_model)
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        seed=args.seed,
        lr0=args.lr0,
        cache=args.cache,
        fraction=args.fraction,
        patience=args.patience,
        plots=True,
        pretrained=True,
        close_mosaic=args.close_mosaic,
        pose=args.pose_loss,
        kobj=args.kobj_loss,
    )

    print(f"[train] Starting training with args: {train_kwargs}")
    model.train(**train_kwargs)
    train_dir = Path(model.trainer.save_dir).resolve()

    best_weights = first_existing(
        [train_dir / "weights" / "best.pt", train_dir / "weights" / "last.pt"]
    )
    if best_weights is None:
        raise FileNotFoundError(f"No trained weights were saved under {train_dir / 'weights'}")

    val_dir: Path | None = None
    if args.final_val:
        print(f"[val] Running validation from: {best_weights}")
        best_model = YOLO(str(best_weights))
        best_model.val(
            data=str(args.data),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=True,
            project=str(args.project),
            name=f"{args.name}_val",
            exist_ok=True,
        )
        val_dir = Path(best_model.validator.save_dir).resolve()

    artifacts = export_artifacts(train_dir, args.results_dir, val_dir)

    print("\nTraining finished.")
    print(f"train_dir: {train_dir}")
    if val_dir is not None:
        print(f"val_dir:   {val_dir}")
    print(f"best_pt:   {best_weights}")
    print(f"results:   {args.results_dir}")
    for label, path in artifacts.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
