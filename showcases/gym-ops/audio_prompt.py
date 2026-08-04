"""Audio TTS voice prompts via NE503 platform-api.

Uses pre-recorded WAV files for common gym prompts. Triggers playback
through the platform-api REST endpoint for audio output.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Default prompt WAV files (stored in /data/aipc/etc/gym-ops/audio/)
PROMPT_FILES = {
    "welcome": "welcome.wav",
    "rep_encourage": "rep_encourage.wav",
    "form_correction": "form_correction.wav",
    "set_complete": "set_complete.wav",
    "fall_alert": "fall_alert.wav",
    "exhaustion_alert": "exhaustion_alert.wav",
    "goodbye": "goodbye.wav",
}


class AudioPrompt:
    """Play pre-recorded voice prompts via platform-api audio playback.

    API endpoints:
      POST /api/v1/audio/playback/start  {"file": "<path>"}
      POST /api/v1/audio/playback/stop
    """

    def __init__(self, audio_dir: str, api_base: str = "http://127.0.0.1:8080",
                 cooldown_seconds: float = 5.0):
        self.audio_dir = Path(audio_dir)
        self.api_base = api_base.rstrip("/")
        self.cooldown_seconds = cooldown_seconds
        self._last_played: dict[str, float] = {}  # prompt_name → last_ts

    def _wav_path(self, prompt_name: str) -> str | None:
        """Resolve prompt name to full WAV path."""
        filename = PROMPT_FILES.get(prompt_name)
        if filename is None:
            return None
        full = self.audio_dir / filename
        if full.exists():
            return str(full)
        return None

    def play(self, prompt_name: str, now: float | None = None) -> bool:
        """Play a voice prompt if cooldown has elapsed.

        Args:
            prompt_name: Key in PROMPT_FILES (e.g. "welcome", "fall_alert").
            now: Current timestamp (defaults to time.time()).

        Returns:
            True if playback was triggered, False if skipped (cooldown or missing).
        """
        now = now or time.time()
        # Cooldown check
        last = self._last_played.get(prompt_name, 0.0)
        if now - last < self.cooldown_seconds:
            return False

        wav = self._wav_path(prompt_name)
        if wav is None:
            logger.debug("Prompt %s has no WAV file", prompt_name)
            return False

        try:
            import urllib.request
            url = f"{self.api_base}/api/v1/audio/playback/start"
            data = f'{{"file": "{wav}"}}'.encode()
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    self._last_played[prompt_name] = now
                    logger.info("Played prompt: %s", prompt_name)
                    return True
                logger.warning("Audio playback returned %d", resp.status)
                return False
        except Exception as e:
            logger.warning("Audio playback failed for %s: %s", prompt_name, e)
            return False

    def stop(self) -> None:
        """Stop any currently playing audio."""
        try:
            import urllib.request
            url = f"{self.api_base}/api/v1/audio/playback/stop"
            req = urllib.request.Request(url, method="POST",
                                         headers={"Content-Length": "0"})
            with urllib.request.urlopen(req, timeout=3.0):
                pass
        except Exception as e:
            logger.warning("Audio stop failed: %s", e)
