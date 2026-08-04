# Multi-stage build for RTSP plugin container
# Builds a minimal container with the RTSP server binary

# ---- Build stage ----
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy HAL headers + common
COPY hal/include/ /build/hal/include/
COPY hal/common/ /build/hal/common/

# Copy camera-daemon headers + rtsp_server source
COPY platform/camera-daemon/include/ /build/platform/camera-daemon/include/
COPY platform/camera-daemon/src/rtsp_server.cpp /build/platform/camera-daemon/src/rtsp_server.cpp

# Copy plugin source
COPY apps/plugin-rtsp/ /build/apps/plugin-rtsp/

WORKDIR /build/apps/plugin-rtsp
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc)

# ---- Runtime stage ----
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/apps/plugin-rtsp/build/rtsp-plugin /usr/local/bin/rtsp-plugin

# Default environment
ENV RTSP_PORT=8554
ENV ENCODED_SOCK=/run/aipc/encoded/main.sock
ENV STREAM_NAME=main
ENV CODEC=h264
ENV WIDTH=1920
ENV HEIGHT=1080
ENV FPS=30

EXPOSE 8554

ENTRYPOINT ["/usr/local/bin/rtsp-plugin"]
