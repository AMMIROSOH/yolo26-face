from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local WIDER-YOLO dataset folder to the Hugging Face Hub."
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="Local dataset directory to upload, for example /runpod-volume/WIDER-yolo.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repo id in the form username/repo-name.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token. If omitted, use HF_TOKEN from the environment or local login state.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional branch or revision to upload to.",
    )
    parser.add_argument(
        "--strategy",
        choices=("upload_large_folder", "upload_folder"),
        default="upload_large_folder",
        help="Use upload_large_folder by default for better resumability on large datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = args.local_dir.resolve()

    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local dataset directory not found: {local_dir}")

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    if args.strategy == "upload_large_folder":
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(local_dir),
            revision=args.revision,
        )
    else:
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(local_dir),
            revision=args.revision,
        )

    print(f"[hf] Uploaded dataset folder: {local_dir}")
    print(f"[hf] Repo: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
