#!/usr/bin/env python3
"""
NE503 Application Template

This is a template for creating NE503 applications.
Modify this file to implement your custom logic.
"""

import time
import signal
import sys
from neoruntime_ipc_sdk import InferenceClient, EventClient, DeviceClient, Config


class TemplateApp:
    """Template application"""
    
    def __init__(self):
        """Initialize the application"""
        self.running = True
        
        # Initialize clients
        self.inference = InferenceClient()
        self.events = EventClient()
        self.device = DeviceClient()
        
        # Get app configuration
        self.app_id = Config.get_app_id()
        self.debug = Config.is_debug()
        
        print(f"[{self.app_id}] Application initialized")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[{self.app_id}] Received signal {signum}, shutting down...")
        self.running = False
    
    def run(self):
        """Main application loop"""
        print(f"[{self.app_id}] Starting main loop")
        
        try:
            # Subscribe to AI inference results
            for frame_seq, result in self.inference.subscribe(
                stream="cam0_main",
                model="person_v1",
                fps=10
            ):
                if not self.running:
                    break

                # Process the inference result
                self.process_frame(frame_seq, result)
                
        except KeyboardInterrupt:
            print(f"\n[{self.app_id}] Interrupted by user")
        except Exception as e:
            print(f"[{self.app_id}] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def process_frame(self, frame_seq, result):
        """
        Process each frame's inference result

        Args:
            frame_seq: Frame sequence number (int) — subscribe() yields
                (sequence, InferenceResult) tuples, not frame objects
            result: InferenceResult object
        """
        # Example: Print detection results
        if len(result.objects) > 0:
            print(f"[Frame {frame_seq}] Detected {len(result.objects)} objects:")

            for obj in result.objects:
                print(f"  - {obj.label}: {obj.score:.2f} at {obj.bbox}")

                # Example: React to person detection
                if obj.label == "person" and obj.score > 0.8:
                    self.on_person_detected(frame_seq, obj)

        # Publish custom events
        if result.has_person():
            self.events.publish(f"app/{self.app_id}/person_detected", {
                "frame": frame_seq,
                "timestamp": result.timestamp_ns,
                "count": result.count_by_label("person")
            })
    
    def on_person_detected(self, frame_seq, person):
        """
        Handle person detection
        
        This is where you implement your custom logic.
        """
        if self.debug:
            print(f"  Person detected with confidence {person.score:.2f}")
        
        # Example: Control device
        # self.device.set_white_light(100)
        
        # Example: Send alert
        # self.events.publish(f"app/{self.app_id}/alert", {
        #     "type": "person_detected",
        #     "confidence": person.score
        # })
    
    def cleanup(self):
        """Cleanup resources before exit"""
        print(f"[{self.app_id}] Cleaning up...")
        
        # Close connections
        self.inference.close()
        self.events.close()
        self.device.close()
        
        print(f"[{self.app_id}] Shutdown complete")


def main():
    """Main entry point"""
    app = TemplateApp()
    app.run()


if __name__ == "__main__":
    main()

