#!/usr/bin/env python3
"""
People Counting Application Example

Features:
1. Subscribe to person detection results
2. Count people in the scene
3. Send statistics events
4. Control lights for alerts when threshold exceeded
"""

import os
import signal
import time
from collections import deque
from neoruntime_ipc_sdk import InferenceClient, EventClient, DeviceClient


class PeopleCounter:
    """People counter"""
    
    def __init__(self, window_size=30):
        """
        Args:
            window_size: Statistics window size (in frames)
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
    def update(self, count):
        """Update count"""
        self.history.append(count)
    
    def get_average(self):
        """Get average count"""
        if not self.history:
            return 0
        return sum(self.history) / len(self.history)
    
    def get_current(self):
        """Get current count"""
        if not self.history:
            return 0
        return self.history[-1]


class PeopleCountingApp:
    """People counting application"""
    
    def __init__(self):
        # Initialize clients
        self.inference = InferenceClient()
        self.events = EventClient()
        self.device = DeviceClient()
        
        # Counter
        self.counter = PeopleCounter(window_size=30)
        
        # Configuration
        self.threshold = 10  # Alert threshold
        self.alert_active = False
        
        print("[PeopleCounter] Application started")
    
    def run(self):
        """Main loop"""
        try:
            # Subscribe to person detection results
            for frame, result in self.inference.subscribe(
                # Stream id from env (app.yaml STREAM_ID; devices expose
                # main/sub, dev rigs used cam0_main)
                stream=os.environ.get("STREAM_ID", "cam0_main"),
                # Resolved id from spec.models (falls back to the bundled id)
                model=os.environ.get("AIPC_MODEL_detector", "person_v1"),
                fps=10
            ):
                self.process_frame(frame, result)

        except (KeyboardInterrupt, SystemExit):
            print("\n[PeopleCounter] Received exit signal")
        except Exception as e:
            print(f"[PeopleCounter] Error: {e}")
        finally:
            self.cleanup()
    
    def process_frame(self, frame, result):
        """Process one frame"""
        # Count people
        person_count = result.count_by_label("person")
        self.counter.update(person_count)
        
        # Get statistics
        avg_count = self.counter.get_average()
        
        # Print log
        if person_count > 0:
            print(f"[Frame {frame}] "
                  f"Current: {person_count}, "
                  f"Average: {avg_count:.1f}")

        # Send statistics event
        self.events.publish("app/people_counting/stats", {
            "timestamp": result.timestamp_ns,
            "current_count": person_count,
            "average_count": avg_count,
            "threshold": self.threshold
        })
        
        # Alert logic
        self.check_alert(person_count)
    
    def check_alert(self, person_count):
        """Check if alert needed"""
        if person_count > self.threshold:
            if not self.alert_active:
                # Trigger alert
                print(f"[ALERT] People count exceeds threshold: {person_count} > {self.threshold}")
                
                # Turn on white light
                self.device.set_white_light(100)
                
                # Send alert event
                self.events.publish("app/people_counting/alert", {
                    "type": "over_threshold",
                    "count": person_count,
                    "threshold": self.threshold
                }, persistent=True)
                
                self.alert_active = True
        else:
            if self.alert_active:
                # Clear alert
                print(f"[INFO] People count back to normal: {person_count}")
                
                # Turn off white light
                self.device.set_white_light(0)
                
                # Send recovery event
                self.events.publish("app/people_counting/recovered", {
                    "count": person_count
                })
                
                self.alert_active = False
    
    def cleanup(self):
        """Cleanup resources"""
        print("[PeopleCounter] Cleaning up resources")
        
        # Turn off lights
        if self.alert_active:
            self.device.set_white_light(0)
        
        # Close connections
        self.inference.close()
        self.events.close()
        self.device.close()


def _handle_term(signum, _frame):
    """Translate SIGTERM into the normal shutdown path.

    The container runs this module as PID 1, and the kernel ignores
    default-disposition signals for PID 1: without an explicit handler,
    containerd's SIGTERM never reaches Python and every app-manager stop
    degrades to the grace-timeout wait followed by SIGKILL. The subscribe
    iterator yields at ~10 fps, so the raised SystemExit unwinds run()'s
    loop (and its cleanup) within a frame of the signal arriving.
    """
    print("[PeopleCounter] Received SIGTERM, shutting down", flush=True)
    raise SystemExit(0)


def main():
    """Main function"""
    signal.signal(signal.SIGTERM, _handle_term)
    app = PeopleCountingApp()
    app.run()
    # Exit deterministically: a stalled SDK close inside cleanup() must not
    # push container stop back into the kill-timeout path.
    os._exit(0)


if __name__ == "__main__":
    main()

