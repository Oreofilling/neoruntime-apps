# 07-platform-overlay — rung 7 of the examples ladder.
# Long-running process without a web server: the healthcheck probes the
# app process itself (pgrep); the model bundling matches rungs 03-06.

FROM python:3.11-slim-bookworm

LABEL maintainer="opensource@camthink.ai"
LABEL version="1.0.0"
LABEL description="NeoRuntime examples ladder 07 - results pushed to the platform overlay"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# SDK from PyPI, pinned via --build-arg SDK_VERSION by build_app.sh; the
# pinned source is sdk.lock at the repo root.
ARG SDK_VERSION=0.8.0
RUN pip install --no-cache-dir "neoruntime-ipc-sdk==${SDK_VERSION}"

COPY app.py /app/app.py

# Bundled model (fetched by scripts/fetch_models.sh into models/,
# sha256-pinned via models.manifest).
COPY models/ /opt/aipc/models/

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# No web server here: liveness = the app process itself is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD pgrep -f "python3 /app/app.py" || exit 1

CMD ["python3", "/app/app.py"]
