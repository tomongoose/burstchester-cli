FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    WORKSPACE=/workspace/burstchester-training

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/python /app/src/python
COPY remote-trainer/requirements.txt /app/remote-trainer/requirements.txt
COPY remote-trainer/entrypoint.py /app/remote-trainer/entrypoint.py

RUN python3 -m pip install --break-system-packages --no-cache-dir -r /app/remote-trainer/requirements.txt

ENTRYPOINT ["python3", "/app/remote-trainer/entrypoint.py"]
