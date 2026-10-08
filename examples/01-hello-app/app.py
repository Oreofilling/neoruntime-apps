#!/usr/bin/env python3
"""01-hello-app — the smallest useful NeoRuntime app.

Rung 1 of the examples ladder (01-05). No camera, no model, no device
daemons: it exercises only the application lifecycle that every later
rung reuses.

What it teaches
  * app identity via ``Config.get_app_id()`` — the platform injects it,
    and it works with no device connection at all
  * SIGTERM -> graceful shutdown -> exit 0 (what ``aipc-cli app stop``
    sends; person-detection and every long-running app use this shape)
  * an unrecoverable failure -> exit 1, which ``restart_policy:
    on-failure`` turns into a restart. Set DEMO_FAIL=1 in app.yaml to
    watch it happen (3 ticks, then a deliberate crash).
"""

import os
import signal
import sys
import time

from neoruntime_ipc_sdk import Config


class HelloApp:
    """Lifecycle demo: tick until told to stop, or crash on purpose."""

    def __init__(self):
        self.running = True
        self.app_id = Config.get_app_id()
        self.fail = os.environ.get("DEMO_FAIL", "") == "1"
        self.counter = 0
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, _frame):
        print(f"[{self.app_id}] signal {signum} -> graceful stop", flush=True)
        self.running = False

    def run(self):
        print(f"[{self.app_id}] hello from {os.uname().machine}", flush=True)
        while self.running:
            self.counter += 1
            print(f"[{self.app_id}] tick #{self.counter}", flush=True)
            if self.fail and self.counter >= 3:
                print(f"[{self.app_id}] simulated crash -> exit 1", flush=True)
                return 1
            time.sleep(1)
        print(f"[{self.app_id}] stopped after {self.counter} ticks", flush=True)
        return 0


def main():
    sys.exit(HelloApp().run())


if __name__ == "__main__":
    main()
