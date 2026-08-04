"""Wiegand access control and Alarm output via NE503 platform-api.

Provides gate unlock on member recognition and alarm trigger on
safety events (fall, exhaustion).
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class AccessControl:
    """Wiegand gate control and alarm output via platform-api.

    API endpoints:
      POST /api/v1/device/wiegand  {"channel": N, "enable": true/false}
      POST /api/v1/device/alarm-out  {"channel": N, "enable": true/false}
    """

    def __init__(self, api_base: str = "http://127.0.0.1:8080",
                 wiegand_channel: int = 0, alarm_channel: int = 0,
                 unlock_duration: float = 5.0):
        self.api_base = api_base.rstrip("/")
        self.wiegand_channel = wiegand_channel
        self.alarm_channel = alarm_channel
        self.unlock_duration = unlock_duration
        self._gate_open_since: float | None = None

    def _post(self, endpoint: str, payload: dict) -> bool:
        """POST JSON to platform-api, return True on 200."""
        try:
            import json
            import urllib.request
            url = f"{self.api_base}{endpoint}"
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning("POST %s failed: %s", endpoint, e)
            return False

    def unlock(self) -> bool:
        """Trigger Wiegand gate unlock (enable=true)."""
        ok = self._post("/api/v1/device/wiegand",
                        {"channel": self.wiegand_channel, "enable": True})
        if ok:
            self._gate_open_since = time.time()
            logger.info("Gate unlocked (channel %d)", self.wiegand_channel)
        return ok

    def lock(self) -> bool:
        """Trigger Wiegand gate lock (enable=false)."""
        ok = self._post("/api/v1/device/wiegand",
                        {"channel": self.wiegand_channel, "enable": False})
        if ok:
            self._gate_open_since = None
            logger.info("Gate locked (channel %d)", self.wiegand_channel)
        return ok

    def trigger_alarm(self, enable: bool = True) -> bool:
        """Trigger or clear alarm output."""
        ok = self._post("/api/v1/device/alarm-out",
                        {"channel": self.alarm_channel, "enable": enable})
        if ok:
            action = "triggered" if enable else "cleared"
            logger.info("Alarm %s (channel %d)", action, self.alarm_channel)
        return ok

    def tick(self, now: float | None = None) -> None:
        """Auto-relock gate after unlock_duration seconds."""
        now = now or time.time()
        if self._gate_open_since is not None:
            if now - self._gate_open_since >= self.unlock_duration:
                self.lock()
