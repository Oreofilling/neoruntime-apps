# Face Cascade Inference Application
# Base image with Python and AIPC SDK

FROM python:3.10-slim

LABEL maintainer="CamThink Open Source <opensource@camthink.ai>"
LABEL description="Face Detection + Landmarks Cascade Inference"
LABEL version="1.0.0"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SDK from PyPI, pinned via --build-arg SDK_VERSION by build_app.sh; the
# default only matters for a bare `docker build` without the helper.
ARG SDK_VERSION=0.5.0
RUN pip install --no-cache-dir "neoruntime-ipc-sdk==${SDK_VERSION}"

# Copy application
COPY main.py .

# Bundled models (fetched by scripts/fetch_models.sh into models/, sha256-pinned)
COPY models/ /opt/aipc/models/

# Create logs directory
RUN mkdir -p /app/logs

# Environment defaults
ENV STREAM_ID=cam0_main
ENV DETECTION_MODEL=person_vehicle_v1
ENV LANDMARKS_MODEL=face_landmarks_v1
ENV FPS=10
ENV DETECTION_THRESHOLD=0.5
ENV PUBLISH_TOPIC=inference/face_cascade/cam0
ENV LOG_LEVEL=info

# Run
CMD ["python", "-u", "main.py"]
