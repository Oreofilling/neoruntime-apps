#!/usr/bin/env python3
"""
Object Detection Application

Detects and tracks objects in video stream.
Publishes detection events and optionally controls devices.
"""

import os
import time
import signal
from collections import defaultdict
from neoruntime_ipc_sdk import InferenceClient, EventClient, DeviceClient


class ObjectTracker:
    """Simple object tracker"""
    
    def __init__(self, timeout_sec=2.0):
        self.tracks = {}  # track_id -> last_seen_time
        self.timeout = timeout_sec
    
    def update(self, objects):
        """Update tracks with new detections"""
        current_time = time.time()
        
        # Update existing tracks
        for obj in objects:
            if obj.track_id:
                self.tracks[obj.track_id] = current_time
        
        # Remove stale tracks
        stale_tracks = [
            tid for tid, last_seen in self.tracks.items()
            if current_time - last_seen > self.timeout
        ]
        for tid in stale_tracks:
            del self.tracks[tid]
    
    def get_active_count(self):
        """Get count of active tracks"""
        return len(self.tracks)


class ObjectDetectionApp:
    """Object detection application"""
    
    def __init__(self):
        self.running = True
        
        # Initialize clients
        self.inference = InferenceClient()
        self.events = EventClient()
        self.device = DeviceClient()
        
        # Statistics
        self.frame_count = 0
        self.detection_count = defaultdict(int)
        self.tracker = ObjectTracker()
        
        # Configuration
        self.target_labels = ["person", "car", "truck", "bicycle"]
        self.confidence_threshold = 0.7
        
        print("[ObjectDetection] Application started")
        
        # Signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown"""
        print(f"\n[ObjectDetection] Shutting down...")
        self.running = False
    
    def run(self):
        """Main loop"""
        print("[ObjectDetection] Starting detection loop")
        
        try:
            for frame, result in self.inference.subscribe(
                # Stream id from env (app.yaml STREAM_ID; devices expose
                # main/sub, dev rigs used cam0_main)
                stream=os.environ.get("STREAM_ID", "cam0_main"),
                # Resolved id from spec.models (falls back to the bundled id)
                model=os.environ.get("AIPC_MODEL_detector", "person_vehicle_v1"),
                fps=15
            ):
                if not self.running:
                    break
                
                self.process_frame(frame, result)
                
        except Exception as e:
            print(f"[ObjectDetection] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def process_frame(self, frame, result):
        """Process detection results"""
        self.frame_count += 1
        
        # Filter by confidence
        filtered_objects = [
            obj for obj in result.objects
            if obj.score >= self.confidence_threshold 
            and obj.label in self.target_labels
        ]
        
        if not filtered_objects:
            return
        
        # Update tracker
        self.tracker.update(filtered_objects)
        
        # Update statistics
        for obj in filtered_objects:
            self.detection_count[obj.label] += 1
        
        # Log every 100 frames
        if self.frame_count % 100 == 0:
            self.print_statistics()
        
        # Publish detection event
        self.publish_detection_event(frame, result, filtered_objects)
        
        # Handle special cases
        self.handle_detections(filtered_objects)
    
    def publish_detection_event(self, frame, result, objects):
        """Publish detection event"""
        event_data = {
            "frame_sequence": frame,
            "timestamp": result.timestamp_ns,
            "objects": [
                {
                    "id": obj.track_id,
                    "label": obj.label,
                    "score": obj.score,
                    "bbox": {
                        "x": obj.bbox.x,
                        "y": obj.bbox.y,
                        "width": obj.bbox.width,
                        "height": obj.bbox.height
                    }
                }
                for obj in objects
            ],
            "count": len(objects),
            "active_tracks": self.tracker.get_active_count()
        }
        
        self.events.publish("app/object_detection/detections", event_data)
    
    def handle_detections(self, objects):
        """Handle specific detection scenarios"""
        # Example: Alert on multiple people
        person_count = sum(1 for obj in objects if obj.label == "person")
        
        if person_count >= 3:
            self.events.publish("app/object_detection/alert", {
                "type": "crowd_detected",
                "count": person_count
            })
            
            # Optional: Turn on light
            # self.device.set_white_light(100)
        
        # Example: Alert on vehicle
        vehicle_count = sum(
            1 for obj in objects 
            if obj.label in ["car", "truck"]
        )
        
        if vehicle_count > 0:
            self.events.publish("app/object_detection/vehicle", {
                "count": vehicle_count
            })
    
    def print_statistics(self):
        """Print statistics"""
        print(f"\n[ObjectDetection] Statistics after {self.frame_count} frames:")
        print(f"  Active tracks: {self.tracker.get_active_count()}")
        
        if self.detection_count:
            print("  Total detections:")
            for label, count in sorted(self.detection_count.items()):
                print(f"    {label}: {count}")
        
        print()
    
    def cleanup(self):
        """Cleanup"""
        print(f"[ObjectDetection] Cleaning up...")
        
        # Print final statistics
        self.print_statistics()
        
        # Close connections
        self.inference.close()
        self.events.close()
        self.device.close()
        
        print("[ObjectDetection] Shutdown complete")


def main():
    app = ObjectDetectionApp()
    app.run()


if __name__ == "__main__":
    main()

