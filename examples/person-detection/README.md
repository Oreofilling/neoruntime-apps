# Person Detection Application

Complete NeoRuntime sample application for real-time person detection.

## Features

- Runs AI inference through the AIPC SDK.
- Subscribes to video-frame inference results.
- Detects people and publishes events.
- Optionally controls device lights.
- Demonstrates a complete app permission manifest.

## Layout

```text
person-detection/
|-- app.yaml          # App manifest and permission configuration
|-- app.py            # Application code
|-- Dockerfile        # Container image definition
|-- requirements.txt  # Python dependencies
|-- build.sh          # Build script
`-- README.md         # This document
```

## Quick Start

### 1. Build the app

```bash
cd apps/person-detection
chmod +x build.sh
./build.sh arm64
```

### 2. Install the app

Web console:

1. Open `http://192.0.2.72:8080`.
2. Go to app management.
3. Upload `person-detection.aipc`.

API:

```bash
curl -X POST http://192.0.2.72:8080/api/v1/apps \
  -H "Authorization: Bearer <your-token-key>" \
  -F "app=@person-detection.aipc"
```

### 3. Start the app

```bash
curl -X POST http://192.0.2.72:8080/api/v1/apps/person-detection/start \
  -H "Authorization: Bearer <your-token-key>"
```

### 4. View logs

```bash
curl http://192.0.2.72:8080/api/v1/apps/person-detection/logs \
  -H "Authorization: Bearer <your-token-key>"
```

## Permissions

```yaml
permissions:
  # Video stream access
  video:
    - cam0_main.raw      # Raw video stream through SHM

  # AI inference permissions
  inference:
    models:
      - person_v1        # Person detection model
    max_qps: 30

  # Event bus permissions
  events:
    publish:
      - app/person-detection/*
      - alerts/detection
    subscribe:
      - system/*
```

## SDK Examples

### AI inference

```python
from hailo_ipc_sdk import InferenceClient

inf = InferenceClient()

for frame_seq, result in inf.subscribe(
    stream="cam0_main",
    model="person_v1",
    fps=10,
):
    person_count = result.count_by_label("person")
    print(f"Detected {person_count} persons")
```

### Event publishing

```python
import time
from hailo_ipc_sdk import EventClient

events = EventClient()

events.publish("app/person-detection/detection", {
    "person_count": 2,
    "timestamp": time.time(),
})
```

### Device control

```python
from hailo_ipc_sdk import DeviceClient

device = DeviceClient()

device.set_white_light(80)  # 80% brightness
device.set_ir_led(True)
```

## Environment Variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `APP_ID` | `person-detection` | App ID |
| `DETECTION_THRESHOLD` | `0.7` | Detection confidence threshold |
| `ALERT_COOLDOWN_SECONDS` | `5` | Alert cooldown period |
| `LOG_LEVEL` | `INFO` | Log level |

## Test Flow

### Web console

1. Upload and install the `.aipc` package from app management.
2. Start the app and confirm that its state becomes running.
3. Open the app details page and view live logs.
4. Subscribe to `app/person-detection/*` in the event view.
5. Open a container terminal and verify `/run/aipc/` socket access.

### API

```bash
# 1. Install
curl -X POST http://192.0.2.72:8080/api/v1/apps \
  -H "Authorization: Bearer <your-token-key>" \
  -F "app=@person-detection.aipc"

# 2. Start
curl -X POST http://192.0.2.72:8080/api/v1/apps/person-detection/start \
  -H "Authorization: Bearer <your-token-key>"

# 3. Get status
curl http://192.0.2.72:8080/api/v1/apps/person-detection \
  -H "Authorization: Bearer <your-token-key>"

# 4. Get logs
curl http://192.0.2.72:8080/api/v1/apps/person-detection/logs \
  -H "Authorization: Bearer <your-token-key>"

# 5. List event topics
curl http://192.0.2.72:8080/api/v1/events/topics \
  -H "Authorization: Bearer <your-token-key>"
```
