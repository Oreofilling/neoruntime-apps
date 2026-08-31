# NeoRuntime Application Template

A template for creating NeoRuntime applications.

## Quick Start

### 1. Customize the Application

Edit `app.py` to implement your logic:

```python
def process_frame(self, frame, result):
    # Your custom logic here
    pass
```

### 2. Update Manifest

Edit `app.yaml` to configure your application:

```yaml
metadata:
  id: my_custom_app
  name: My Custom Application
  version: 1.0.0

spec:
  permissions:
    inference:
      models: [person_v1]  # Models you need
    device:
      light: true          # Permissions you need
```

### 3. Build Container

```bash
# Build Docker image
docker build -t my-app:1.0.0 .

# Export as tar
docker save my-app:1.0.0 -o my-app.tar
```

### 4. Deploy to Device

```bash
# Copy files to device
scp app.yaml my-app.tar root@<device-ip>:/tmp/

# SSH to device and install
ssh root@<device-ip>
cd /tmp
aipc-cli app install app.yaml my-app.tar

# Start the app
aipc-cli app start my_custom_app

# View logs
aipc-cli app logs my_custom_app
```

## Development

### Local Testing (without container)

```bash
# Install SDK
pip install hailo-ipc-sdk

# Run directly
python3 app.py
```

### With Mocked Services

```bash
# Start mock services
cd ../../tests/mocks
./start_mock_services.sh

# Run your app
cd ../../apps/your-app
python3 app.py
```

## Application Structure

```
my-app/
├── app.py           # Main application code
├── app.yaml         # Application manifest
├── Dockerfile       # Container definition
├── requirements.txt # Python dependencies (optional)
└── README.md        # Application documentation
```

## Manifest Reference

### Metadata

```yaml
metadata:
  id: unique_app_id      # Required: Unique identifier
  name: Display Name     # Required: Human-readable name
  version: 1.0.0         # Required: Semantic version
  description: "..."     # Optional
  author: Your Name      # Optional
  email: you@example.com  # Optional
```

### Resources

```yaml
resources:
  cpu: "50%"            # CPU limit (percentage or cores)
  memory: "256Mi"       # Memory limit (Mi/Gi)
  shm: true             # Access to shared memory
```

### Permissions

```yaml
permissions:
  # Video streams
  video:
    - cam0_main.raw
    - cam0_sub.raw
  
  # AI inference
  inference:
    models: [person_v1, vehicle_v1]
    max_qps: 30
    max_concurrent: 2
  
  # Events
  events:
    publish: [app/myapp/*]
    subscribe: [model/*/detections, system/*]
  
  # Device control
  device:
    light: true
    ir_cut: true
    ptz: false
    lens: false
    gpio:
      read: [12]
      write: [21]
  
  # Network
  network:
    outbound:
      - "https://api.example.com"
      - "mqtt://broker.example.com:8883"
```

### Advanced Options

```yaml
  # Environment variables
  env:
    - name: MY_CONFIG
      value: "production"
  
  # Volume mounts
  volumes:
    - host: /opt/aipc/data/myapp
      container: /app/data
      readonly: false
  
  # Startup
  autostart: true
  restart_policy: on-failure
  restart_max_retries: 3
  
  # Health check
  healthcheck:
    enabled: true
    interval: 30s
    timeout: 5s
    retries: 3
```

## SDK Usage Examples

### AI Inference

```python
from neoruntime_ipc_sdk import InferenceClient

inf = InferenceClient()

# Stream inference
for frame, result in inf.subscribe("cam0_main", "person_v1", fps=10):
    for obj in result.objects:
        print(f"{obj.label}: {obj.score:.2f}")
```

### Device Control

```python
from neoruntime_ipc_sdk import DeviceClient, IrCutMode

dev = DeviceClient()

# Light control
dev.set_white_light(80)
dev.set_ir_led(True)
dev.set_ircut(IrCutMode.NIGHT)

# PTZ control
dev.call_preset(3)
dev.pan_left(speed=50)
```

### Event Publishing

```python
from neoruntime_ipc_sdk import EventClient

events = EventClient()

# Publish
events.publish("app/myapp/alert", {
    "type": "person_detected",
    "confidence": 0.95
})

# Subscribe
for event in events.subscribe("model/*/detections"):
    print(f"Event: {event.topic}")
```

## Best Practices

1. **Error Handling**
   - Always use try-except blocks
   - Handle gRPC connection errors
   - Implement graceful shutdown

2. **Resource Management**
   - Close connections properly
   - Use context managers (`with` statements)
   - Don't leak file descriptors

3. **Performance**
   - Process frames efficiently
   - Avoid blocking operations
   - Use appropriate FPS limits

4. **Logging**
   - Use structured logging
   - Log important events
   - Don't spam logs

5. **Security**
   - Only request permissions you need
   - Validate all inputs
   - Don't store sensitive data in logs

## Common Patterns

### Zone-Based Detection

```python
from neoruntime_ipc_sdk import Zone

zone_a = Zone("entrance", [(0.1, 0.5), (0.9, 0.5), (0.9, 1.0), (0.1, 1.0)])

for frame, result in inf.subscribe("cam0_main", "person_v1"):
    for obj in result.objects:
        if zone_a.contains_bbox(obj.bbox):
            events.publish("app/myapp/zone_intrusion", {
                "zone": zone_a.name,
                "object": obj.label
            })
```

### Time-Based Actions

```python
import time

last_action_time = 0
cooldown_sec = 60

for frame, result in inf.subscribe("cam0_main", "person_v1"):
    current_time = time.time()
    
    if result.has_person() and (current_time - last_action_time) > cooldown_sec:
        # Perform action
        dev.set_white_light(100)
        last_action_time = current_time
```

### Statistics Tracking

```python
from collections import deque

class Statistics:
    def __init__(self, window_size=100):
        self.history = deque(maxlen=window_size)
    
    def update(self, value):
        self.history.append(value)
    
    def get_average(self):
        return sum(self.history) / len(self.history) if self.history else 0

stats = Statistics()

for frame, result in inf.subscribe("cam0_main", "person_v1"):
    count = result.count_by_label("person")
    stats.update(count)
    
    if stats.get_average() > threshold:
        # Alert
        pass
```

## Troubleshooting

### App Won't Start

```bash
# Check logs
aipc-cli app logs myapp

# Check permissions in manifest
aipc-cli app inspect myapp

# Check container status
ssh root@device "ctr -n aipc containers list"
```

### No Inference Results

- Check if model is loaded: `aipc-cli model list`
- Check permissions in manifest
- Check if camera is running: `aipc-cli stream list`
- View AI runtime logs: `journalctl -u aipc-ai-runtime`

### Device Control Not Working

- Check permissions in manifest
- Check device capabilities: `aipc-cli device status`
- View device-control logs: `journalctl -u aipc-device-control`

## More Examples

See the `apps/` directory for more complete examples:
- `people-counting/` - People counting with alerts
- `smart-lighting/` - AI-driven lighting control
- `object-detection/` - Object detection and tracking

## Support

- Documentation: https://github.com/camthink-ai/neoruntime-sdks
- Issues: https://github.com/camthink-ai/neoruntime-apps/issues
- Email: opensource@camthink.ai
