#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cd /app
  exec python train.py --help
fi

detect_data_root() {
  if [[ -n "${YOLO_FACE_DATA_ROOT:-}" ]]; then
    printf '%s\n' "${YOLO_FACE_DATA_ROOT}"
    return 0
  fi

  local candidates=(
    "/runpod-volume/WIDER-yolo"
    "/runpod-volume/datasets/WIDER-yolo"
    "/workspace/WIDER-yolo"
    "/workspace/datasets/WIDER-yolo"
    "/data/WIDER-yolo"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}/train" && -d "${candidate}/val" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

make_data_yaml() {
  local output_path="$1"
  local data_root="$2"

  cat > "${output_path}" <<EOF
path: ${data_root}
train: train
val: val

kpt_shape: [5, 3]
flip_idx: [1, 0, 2, 4, 3]

nc: 1
names:
  0: face
EOF
}

DATA_YAML="${YOLO_FACE_DATA_YAML:-}"
if [[ -z "${DATA_YAML}" ]]; then
  DATA_ROOT="$(detect_data_root || true)"
  if [[ -z "${DATA_ROOT}" ]]; then
    echo "[docker] Could not find a dataset root." >&2
    echo "[docker] Set YOLO_FACE_DATA_ROOT or YOLO_FACE_DATA_YAML explicitly." >&2
    echo "[docker] Expected a WIDER-YOLO layout under one of:" >&2
    echo "  /runpod-volume/WIDER-yolo" >&2
    echo "  /workspace/WIDER-yolo" >&2
    exit 1
  fi

  DATA_YAML="/tmp/yolo26-face-data.yaml"
  make_data_yaml "${DATA_YAML}" "${DATA_ROOT}"
  echo "[docker] Using detected dataset root: ${DATA_ROOT}"
else
  echo "[docker] Using dataset YAML from YOLO_FACE_DATA_YAML: ${DATA_YAML}"
fi

WEIGHTS_PATH="${YOLO_FACE_WEIGHTS:-/app/yolo26n.pt}"
PROJECT_DIR="${YOLO_FACE_PROJECT:-/workspace/runs/pose/face}"
RESULTS_DIR="${YOLO_FACE_RESULTS_DIR:-/workspace/results/face}"

mkdir -p "${PROJECT_DIR}" "${RESULTS_DIR}"

if [[ ! -f "${WEIGHTS_PATH}" ]]; then
  echo "[docker] Missing weights file: ${WEIGHTS_PATH}" >&2
  echo "[docker] Mount a checkpoint and set YOLO_FACE_WEIGHTS, or rebuild with a different bundled checkpoint." >&2
  exit 1
fi

cd /app
exec python train.py \
  --data "${DATA_YAML}" \
  --weights "${WEIGHTS_PATH}" \
  --project "${PROJECT_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  "$@"
