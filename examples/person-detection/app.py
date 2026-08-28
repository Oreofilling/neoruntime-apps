#!/usr/bin/env python3
"""
Person Detection Application for AIPC Platform

Features:
- Subscribe to video stream inference results
- Detect persons using AI model
- Publish detection events to event bus
- Control device (light) on detection
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
from typing import Optional

# AIPC SDK
from neoruntime_ipc_sdk import (
    InferenceClient,
    EventClient,
    DeviceClient,
    FdMediaClient as MediaClient,
    Config,
    InferenceResult,
    DetectedObject,
)

# Stream id from env (app.yaml STREAM_ID; devices expose main/sub, dev rigs
# used third)
STREAM_ID = os.environ.get("STREAM_ID", "third")

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')),
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PersonDetectionApp:
    """Person Detection Application"""

    def __init__(self):
        self.running = True
        self.app_id = Config.get_app_id()
        self.debug = Config.is_debug()

        # Configuration from environment
        self.detection_threshold = float(os.environ.get('DETECTION_THRESHOLD', '0.2'))
        self.alert_cooldown = int(os.environ.get('ALERT_COOLDOWN_SECONDS', '5'))

        # SDK Clients
        self.inference: Optional[InferenceClient] = None
        self.events: Optional[EventClient] = None
        self.device: Optional[DeviceClient] = None
        self.media: Optional[MediaClient] = None

        # State tracking
        self.frame_count = 0
        self.total_detections = 0
        self.last_alert_time = 0
        self.person_count_history = []

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("=" * 60)
        logger.info(f"  Person Detection Application v1.0.0")
        logger.info(f"  App ID: {self.app_id}")
        logger.info(f"  Platform: {os.uname().machine}")
        logger.info(f"  Detection Threshold: {self.detection_threshold}")
        logger.info("=" * 60)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def initialize(self) -> bool:
        """Initialize SDK clients"""
        try:
            # Initialize Inference Client
            logger.info("Initializing AI Inference client...")
            self.inference = InferenceClient()

            # List available models
            models = self.inference.list_models()
            logger.info(f"Available models: {[m.model_id for m in models]}")

            # Check if required model is available
            # (resolved id from spec.models, falls back to the bundled id)
            required_model = os.environ.get("AIPC_MODEL_detector", "person-detection")
            model_available = any(m.model_id == required_model for m in models)
            if model_available:
                logger.info(f"[OK] Model '{required_model}' is available for inference")
            else:
                logger.warning(f"[WARN] Model '{required_model}' NOT found - inference may fail")

            # Initialize Event Bus Client
            logger.info("Initializing Event Bus client...")
            self.events = EventClient()

            # Initialize Device Control Client (optional)
            try:
                logger.info("Initializing Device Control client...")
                self.device = DeviceClient()
            except Exception as e:
                logger.warning(f"Device control not available: {e}")
                self.device = None

            # Initialize Media Client (for raw video stream access)
            try:
                logger.info("Initializing Media client...")
                self.media = MediaClient()

                # List available video streams
                available_streams = self.media.list_streams()
                logger.info(f"Available video streams: {available_streams}")

                # Check if required stream is available
                required_stream = STREAM_ID
                stream_info = self.media.get_stream_info(required_stream)
                if stream_info:
                    logger.info(f"[OK] Video stream '{required_stream}' is available: "
                               f"{stream_info.width}x{stream_info.height} @ {stream_info.fps}fps, format={stream_info.format}")
                else:
                    shm_path = f"/run/aipc/shm/{required_stream}.raw"
                    if os.path.exists(shm_path):
                        logger.info(f"[OK] Video stream '{required_stream}' SHM file exists at {shm_path}")
                    else:
                        logger.warning(f"[WARN] Video stream '{required_stream}' NOT found - will use simulation mode")
            except Exception as e:
                logger.warning(f"Media client not available: {e}")
                self.media = None

            logger.info("All clients initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False

    def run(self):
        """Main application loop"""
        if not self.initialize():
            logger.error("Initialization failed, exiting")
            return 1

        logger.info("Starting person detection loop...")
        logger.info(f"Subscribing to stream '{STREAM_ID}' with model 'person-detection'")
        logger.info("Waiting for inference results... (this may take a moment if stream is initializing)")

        first_frame_received = False

        try:
            # Subscribe to video stream inference results
            # The platform will run inference on each frame and send results
            for frame_seq, result in self.inference.subscribe(
                stream=STREAM_ID,
                model=os.environ.get("AIPC_MODEL_detector", "person-detection"),
                fps=10  # Process at 10 FPS
            ):
                if not self.running:
                    break

                # Log when first frame is received
                if not first_frame_received:
                    first_frame_received = True
                    logger.info(f"[OK] Received first inference result - stream and model are working!")
                    logger.info(f"  Frame sequence: {frame_seq}, timestamp: {result.timestamp_ns}")
                    # Check if running in simulation mode (no actual inference)
                    if result.status_message == "simulation":
                        logger.warning("  WARNING: Running in SIMULATION mode - no actual inference!")
                        logger.warning("  This means FdReceiver cannot subscribe to video stream.")
                        logger.warning("  Check: 1) camera-daemon is running, 2) /run/aipc/camera.sock exists")

                self._process_frame(frame_seq, result)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return 1
        finally:
            if not first_frame_received:
                logger.warning(f"No inference results received - check if video stream '{STREAM_ID}' is active and model 'person-detection' is loaded")
            self._cleanup()

        return 0

    def _process_frame(self, frame_seq: int, result: InferenceResult):
        """Process a single frame's inference result"""
        self.frame_count += 1

        # Count persons with confidence above threshold
        persons = [
            obj for obj in result.objects
            if obj.label == "person" and obj.score >= self.detection_threshold
        ]
        person_count = len(persons)

        # Track history for analytics
        self.person_count_history.append(person_count)
        if len(self.person_count_history) > 100:
            self.person_count_history.pop(0)

        # Log detection
        if person_count > 0:
            self.total_detections += 1
            logger.info(f"[Frame {frame_seq}] Detected {person_count} person(s)")

            for i, obj in enumerate(persons):
                bbox = obj.bbox
                logger.debug(
                    f"  Person {i+1}: confidence={obj.score:.2f}, "
                    f"position=({bbox.x:.2f}, {bbox.y:.2f}), "
                    f"size=({bbox.width:.2f}x{bbox.height:.2f})"
                )

        # Publish detection event
        self._publish_detection_event(frame_seq, result, persons)

        # Device control: turn on light when person detected
        if person_count > 0 and self.device:
            self._trigger_light()

        # Print statistics every 100 frames
        if self.frame_count % 100 == 0:
            self._print_statistics()

    def _publish_detection_event(self, frame_seq: int, result: InferenceResult, persons: list):
        """Publish detection event to event bus"""
        try:
            event_data = {
                "app_id": self.app_id,
                "frame_sequence": frame_seq,
                "timestamp_ns": result.timestamp_ns,
                "timestamp_iso": datetime.now().isoformat(),
                "person_count": len(persons),
                "total_frames_processed": self.frame_count,
                "total_detections": self.total_detections,
                "objects": [
                    {
                        "label": obj.label,
                        "confidence": round(obj.score, 3),
                        "bbox": {
                            "x": round(obj.bbox.x, 3),
                            "y": round(obj.bbox.y, 3),
                            "width": round(obj.bbox.width, 3),
                            "height": round(obj.bbox.height, 3)
                        }
                    }
                    for obj in persons
                ]
            }

            # Publish to app-specific topic
            self.events.publish(f"app/{self.app_id}/detection", event_data)

            # Publish alert if cooldown expired
            current_time = time.time()
            if len(persons) > 0 and (current_time - self.last_alert_time) >= self.alert_cooldown:
                self.events.publish("alerts/detection", {
                    "type": "person_detected",
                    "app_id": self.app_id,
                    "person_count": len(persons),
                    "timestamp": datetime.now().isoformat()
                })
                self.last_alert_time = current_time
                logger.debug("Alert event published")

        except Exception as e:
            logger.error(f"Failed to publish event: {e}")

    def _trigger_light(self):
        """Trigger white light when person detected"""
        try:
            # Set white light to 50% brightness
            self.device.set_white_light(50)
            logger.debug("Light triggered")

            # Schedule light off (in a real app, use a timer thread)
            # For simplicity, we just toggle it
        except Exception as e:
            logger.debug(f"Light control failed: {e}")

    def _print_statistics(self):
        """Print processing statistics"""
        avg_persons = sum(self.person_count_history) / len(self.person_count_history) if self.person_count_history else 0
        logger.info(
            f"Statistics: frames={self.frame_count}, "
            f"detections={self.total_detections}, "
            f"avg_persons={avg_persons:.2f}"
        )

    def _cleanup(self):
        """Cleanup resources before exit"""
        logger.info("Cleaning up resources...")

        # Print final statistics
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Total detections: {self.total_detections}")

        # Close SDK clients
        if self.inference:
            self.inference.close()
        if self.events:
            self.events.close()
        if self.device:
            self.device.close()
        if self.media:
            self.media.close()

        logger.info("Cleanup complete. Goodbye!")


def main():
    """Main entry point"""
    app = PersonDetectionApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()