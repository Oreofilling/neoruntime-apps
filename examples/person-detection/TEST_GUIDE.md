# Person Detection Web Console Test Guide

## 1. Preparation

### Confirm that platform services are running

Open `http://192.0.2.72:8080`.

Use the administrator credentials configured for your deployment. Do not write
real credentials into documentation or source control.

Confirm that the dashboard reports these services as healthy:

- AI Runtime
- Event Bus
- App Manager
- Device Control

### Confirm that models are registered

Open the AI model view and confirm that these models are available:

- `person_v1` - person detection model
- `person_vehicle_v1` - person and vehicle detection model

If a model is missing, upload the corresponding `.hef` file through the model
management page.

## 2. Build the App Image

### Option 1: Build with the project script

```bash
cd <person-detection-app-dir>
./build.sh arm64
```

The build creates `person-detection.aipc`.

### Option 2: Build manually

```bash
# 1. Copy the SDK
cp -r ../../sdk/python/hailo_ipc_sdk ./
cp ../../sdk/python/setup.py ./

# 2. Build the Docker image
docker buildx build --platform linux/arm64 -t aipc/person-detection:1.0.0 .

# 3. Export the image
docker save aipc/person-detection:1.0.0 -o image.tar

# 4. Package the app
zip person-detection.aipc app.yaml image.tar
```

## 3. Install from the Web Console

1. Open app management.
2. Click the install button.
3. Choose `person-detection.aipc`.
4. Confirm the parsed manifest values.
5. Complete the installation and wait for the app to appear in the list.

Expected manifest summary:

```text
App ID: person-detection
Name: Person Detection
Version: 1.0.0
Image: aipc/person-detection:1.0.0
CPU: 50%
Memory: 256Mi
Video stream: cam0_main.raw
AI models: person_v1, person_vehicle_v1
Event topics: app/person-detection/*, alerts/detection
Device control: light, IR cut
```

## 4. Start the App

### Web console

1. Find `person-detection` in the app list.
2. Start the app.
3. Confirm that the app state becomes running.

### API

```bash
curl -X POST http://192.0.2.72:8080/api/v1/apps/person-detection/start \
  -H "Authorization: Bearer <your-token-key>"
```

## 5. Verify Behavior

### App logs

Open the app details page and view live logs.

Expected log lines:

```text
[2026-03-26 20:10:00] [INFO] Person Detection Application v1.0.0
[2026-03-26 20:10:00] [INFO] App ID: person-detection
[2026-03-26 20:10:00] [INFO] Initializing AI Inference client...
[2026-03-26 20:10:00] [INFO] Available models: ['person_v1', 'person_vehicle_v1']
[2026-03-26 20:10:00] [INFO] Starting person detection loop...
[2026-03-26 20:10:01] [INFO] [Frame 1] Detected 2 person(s)
[2026-03-26 20:10:02] [INFO] [Frame 10] Detected 1 person(s)
```

### AI inference

Open the AI model view, select `person_v1`, and confirm that inference count,
average latency, and QPS are changing.

### Events

Subscribe to `app/person-detection/*` from the event view and confirm that
detection events are published:

```json
{
  "topic": "app/person-detection/detection",
  "payload": {
    "app_id": "person-detection",
    "frame_sequence": 100,
    "person_count": 2,
    "objects": [
      {
        "label": "person",
        "confidence": 0.95,
        "bbox": {"x": 0.5, "y": 0.3, "width": 0.2, "height": 0.4}
      }
    ]
  }
}
```

### Resource usage

Open container management and confirm that CPU and memory usage are within the
expected range.

## 6. Debug in the Container

Open a terminal for the running container.

```bash
# Check environment variables
env | grep APP
# APP_ID=person-detection
# LOG_LEVEL=INFO

# Check SDK sockets
ls -la /run/aipc/
# ai-runtime.sock
# event-bus.sock
# device-control.sock

# Import the SDK
python3 -c "from hailo_ipc_sdk import InferenceClient; print('SDK OK')"

# Inspect app code and logs
cat /app/app.py | head -50
ls -la /app/logs/
```

Run a single inference test:

```bash
python3 << 'EOF'
from hailo_ipc_sdk import InferenceClient
import numpy as np

inf = InferenceClient()
image = np.zeros((1080, 1920, 3), dtype=np.uint8)
result = inf.infer(image, model_id="person_v1")
print(f"Detected {len(result.objects)} objects")
inf.close()
EOF
```

Publish a test event:

```bash
python3 << 'EOF'
from hailo_ipc_sdk import EventClient
import time

events = EventClient()
events.publish("app/person-detection/test", {
    "message": "Hello from terminal",
    "timestamp": time.time()
})
print("Event published")
events.close()
EOF
```

## 7. Stop, Restart, and Uninstall

Use the web console to stop, restart, or uninstall the app. Stop the app before
uninstalling it.

## 8. Troubleshooting

### App start failure

Check container state, container logs, and image architecture.

```bash
docker buildx build --platform linux/arm64 -t aipc/person-detection:1.0.0 .
```

### SDK sockets are unavailable

```bash
ls /run/aipc/
ls -la /run/aipc/*.sock
```

Check `app.yaml` permissions and video configuration.

### AI inference has no results

1. Confirm that the model is loaded.
2. Check `permissions.inference.models`.
3. Review app logs for inference errors.

```yaml
permissions:
  inference:
    models:
      - person_v1
```

### Events are not published

1. Confirm the configured event topic.
2. Review app logs for event publish errors.

```yaml
permissions:
  events:
    publish:
      - app/person-detection/*
```

## 9. Test Checklist

| Check | Status | Notes |
| ----- | ------ | ----- |
| Services | TODO | All services are running |
| Model registration | TODO | `person_v1` is available |
| Image build | TODO | ARM64 image was built |
| App install | TODO | Manifest loaded correctly |
| App start | TODO | App state is running |
| AI inference | TODO | Logs show detection results |
| Event publishing | TODO | Event stream has data |
| Resource usage | TODO | CPU and memory look normal |
| Container terminal | TODO | SDK sockets are accessible |
| Logs | TODO | No error messages |

## 10. API Quick Reference

```bash
TOKEN="Bearer <your-token-key>"
BASE="http://192.0.2.72:8080/api/v1"

# App management
curl -H "Authorization: $TOKEN" $BASE/apps
curl -X POST -H "Authorization: $TOKEN" -F "app=@app.aipc" $BASE/apps
curl -H "Authorization: $TOKEN" $BASE/apps/person-detection
curl -X POST -H "Authorization: $TOKEN" $BASE/apps/person-detection/start
curl -X POST -H "Authorization: $TOKEN" $BASE/apps/person-detection/stop
curl -H "Authorization: $TOKEN" $BASE/apps/person-detection/logs
curl -X DELETE -H "Authorization: $TOKEN" $BASE/apps/person-detection

# AI models
curl -H "Authorization: $TOKEN" $BASE/ai/models
curl -H "Authorization: $TOKEN" $BASE/ai/models/person_v1

# Event bus
curl -H "Authorization: $TOKEN" $BASE/events/topics

# Containers
curl -H "Authorization: $TOKEN" $BASE/containers
curl -H "Authorization: $TOKEN" $BASE/containers/person-detection/stats
```
