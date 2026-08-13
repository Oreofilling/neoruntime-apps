FROM python:3.11-slim-bookworm

LABEL version="0.1.0"
LABEL description="GymOps Pose Coach - gym pose analysis, counting, zone occupancy, safety alerts"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl procps libglib2.0-0 libsm6 libxext6 libxrender1 \
    fonts-wqy-zenhei ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY .aipc-build/hailo_ipc_sdk-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/hailo_ipc_sdk-*.whl && rm -f /tmp/hailo_ipc_sdk-*.whl

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Patch simple-websocket to disable permessage-deflate (client compat, same as model-showcase)
RUN sed -i 's/extensions=\[PerMessageDeflate()\]/extensions=[]/' \
    $(python3 -c "import simple_websocket.ws; print(simple_websocket.ws.__file__)")

COPY config.py pose.py counter.py zones.py alerts.py overlay.py \
     video_source.py tracker.py app.py \
     exercise_classifier.py equipment_stats.py \
     face_crop.py face_db.py \
     audio_prompt.py access_control.py /app/
COPY templates/ /app/templates/
COPY static/ /app/static/

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
RUN mkdir -p /app/uploads && chown appuser:appuser /app/uploads

ENV APP_ID=gym-ops
ENV PYTHONUNBUFFERED=1
ENV WEB_PORT=8890
ENV LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${WEB_PORT}/api/health || exit 1

USER appuser
CMD ["python3", "/app/app.py"]
