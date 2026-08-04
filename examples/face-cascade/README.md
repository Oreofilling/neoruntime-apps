# Face Cascade Inference Application

Cascade inference sample application for face detection and landmark detection.

## Features

1. Subscribes to camera video frames.
2. Runs a face detection model (`yolov8_face`).
3. Crops each detected face and runs a landmark model (`face_landmarks`).
4. Merges the results and publishes them to the Event Bus.

## Build

```bash
# Build with Docker (recommended)
./build.sh

# Build without Docker
./build.sh --no-docker
```

## Install

```bash
# Install with aipc-cli
aipc-cli app install -f dist/face-cascade-1.0.0-bundle.tar.gz

# Or provide the manifest and image separately
aipc-cli app install --manifest app.yaml --image dist/face-cascade-1.0.0-rootfs.tar.gz
```

## Start and Stop

```bash
# Start the app
aipc-cli app start face_cascade

# Stop the app
aipc-cli app stop face_cascade

# View status
aipc-cli app info face_cascade

# View logs
aipc-cli app logs face_cascade
```

## Configuration

Configure the app with environment variables in `app.yaml`.

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `STREAM_ID` | `cam0_main` | Camera stream ID |
| `DETECTION_MODEL` | `yolov8_face` | Face detection model ID |
| `LANDMARKS_MODEL` | `face_landmarks` | Landmark model ID |
| `FPS` | `10` | Target frame rate |
| `DETECTION_THRESHOLD` | `0.5` | Detection confidence threshold |
| `PUBLISH_TOPIC` | `inference/face_cascade/cam0` | Event publish topic |
| `LOG_LEVEL` | `info` | Log level |

## Output Format

Results are published to the Event Bus as JSON:

```json
{
  "frame_sequence": 12345,
  "timestamp_ns": 1709456789000000000,
  "stream_id": "cam0_main",
  "num_faces": 2,
  "faces": [
    {
      "face_id": 0,
      "bbox": {"x": 0.2, "y": 0.1, "w": 0.15, "h": 0.2},
      "confidence": 0.95,
      "num_landmarks": 468,
      "landmarks": [
        {"x": 0.3, "y": 0.2, "confidence": 0.99}
      ]
    }
  ],
  "detection_time_us": 15000,
  "landmarks_time_us": 8000,
  "total_time_us": 25000
}
```

## Required Models

Register these models before running the app:

```bash
# Register face detection model
aipc-cli model register /opt/aipc/models/yolov8_face.hef --id yolov8_face

# Register landmark model
aipc-cli model register /opt/aipc/models/face_landmarks_lite.hef --id face_landmarks
```

## Subscribe to Results

```bash
aipc-cli event subscribe "inference/face_cascade/*"
```

## File Layout

```text
face-cascade/
|-- app.yaml          # App manifest
|-- main.py           # Application code
|-- Dockerfile        # Docker build file
|-- requirements.txt  # Python dependencies
|-- build.sh          # Build script
|-- README.md         # This document
`-- dist/             # Build output
    |-- face-cascade-1.0.0-bundle.tar.gz
    `-- face-cascade-1.0.0-rootfs.tar.gz
```
