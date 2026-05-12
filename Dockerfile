FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ULTRALYTICS_SETTINGS_DIR=/tmp/ultralytics

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY train.py predict.py README.md yolo26n.pt ./
COPY dataset ./dataset
COPY docker ./docker
COPY scripts ./scripts

RUN chmod +x /app/docker/train-runpod.sh

ENTRYPOINT ["bash", "/app/docker/train-runpod.sh"]
CMD ["--help"]
