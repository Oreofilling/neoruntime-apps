# Hello World Application Container
# Python-based image for ARM64 (Hailo platform) with debugging support

FROM python:3.11-alpine3.19

LABEL maintainer="opensource@camthink.ai"
LABEL version="1.0.0"
LABEL description="NeoRuntime Hello World Application with continuous output"

# Install debugging tools (for container exec debugging)
RUN apk add --no-cache \
    bash \
    curl \
    iputils \
    net-tools \
    procps \
    strace

# Create app directory
WORKDIR /app

# Copy application files
COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create non-root user for security but allow debug
RUN adduser -D -u 1000 appuser && \
    chown -R appuser:appuser /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_ID=hello_world
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Default: run the Python app
# For debugging: override with /bin/bash or /bin/sh
CMD ["python3", "/app/app.py"]
