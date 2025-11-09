# MIDOG 2021 Research - Reproducible Docker Environment
# Base: Python 3.11.10 on Debian Bookworm

FROM python:3.11.10-slim-bookworm

LABEL maintainer="ksy"
LABEL project="MIDOG-Generalization-Project"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTHONUNBUFFERED=1

# System dependencies (OpenCV 의존성 추가)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenslide0 \
    libopenjp2-7 \
    libvips42 \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir pip==24.3.1 setuptools==75.3.0 wheel==0.45.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import tiatoolbox; print(f'TIAToolbox: {tiatoolbox.__version__}')" && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

CMD ["/bin/bash"]