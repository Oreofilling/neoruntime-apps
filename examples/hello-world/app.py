#!/usr/bin/env python3
"""
Hello World Application for NeoRuntime Platform

A minimal example demonstrating the basic application structure.
Prints hello world with counter continuously.
"""

import os
import time
import signal
import sys

class HelloWorldApp:
    """Hello World application"""

    def __init__(self):
        """Initialize the application"""
        self.running = True
        self.app_id = os.environ.get("APP_ID", "hello_world")
        self.counter = 0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("=" * 50)
        print("  NeoRuntime Hello World Application")
        print(f"  App ID: {self.app_id}")
        print(f"  Platform: {os.uname().machine}")
        print("=" * 50)
        print()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[{self.app_id}] Received signal {signum}, shutting down...")
        self.running = False

    def run(self):
        """Main application loop - print hello world with counter"""
        print(f"[{self.app_id}] Starting main loop...")
        print(f"[{self.app_id}] Use 'docker exec' or 'kubectl exec' to enter container for debugging")
        print()

        try:
            while self.running:
                self.counter += 1
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] #{self.counter:06d} - Hello World from NeoRuntime!")
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n[{self.app_id}] Interrupted by user")
        except Exception as e:
            print(f"[{self.app_id}] Error: {e}")
            return 1
        finally:
            self.cleanup()

        return 0

    def cleanup(self):
        """Cleanup resources before exit"""
        print(f"[{self.app_id}] Cleaning up...")
        print(f"[{self.app_id}] Total messages: {self.counter}")
        print(f"[{self.app_id}] Goodbye!")


def main():
    """Main entry point"""
    app = HelloWorldApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
