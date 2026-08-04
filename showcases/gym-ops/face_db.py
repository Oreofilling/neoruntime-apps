"""Local face embedding database with cosine similarity matching.

Stores member face embeddings in a JSON file on persistent storage.
Supports 1:N identification via cosine similarity.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FaceDatabase:
    """Local face feature database for member identification.

    Storage format (faces.json):
    {
      "members": [
        {
          "member_id": "M001",
          "name": "Jack",
          "type": "member",       # member | coach | visitor
          "embedding": [0.023, -0.015, ...],
          "registered_at": "2026-07-21T10:00:00Z"
        }
      ],
      "threshold": 0.6
    }
    """

    def __init__(self, path: str, threshold: float = 0.6):
        self.path = Path(path)
        self.threshold = threshold
        self.embeddings: dict[str, np.ndarray] = {}   # member_id → vector
        self.metadata: dict[str, dict] = {}            # member_id → {name, type, ...}
        self._load()

    def _load(self) -> None:
        """Load database from JSON file."""
        if not self.path.exists():
            logger.info("Face DB not found at %s — starting empty", self.path)
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.threshold = data.get("threshold", self.threshold)
            for m in data.get("members", []):
                mid = m["member_id"]
                emb = np.array(m["embedding"], dtype=np.float32)
                self.embeddings[mid] = emb
                self.metadata[mid] = {
                    "name": m.get("name", mid),
                    "type": m.get("type", "member"),
                    "registered_at": m.get("registered_at", ""),
                }
            logger.info("Loaded %d members from %s", len(self.embeddings), self.path)
        except Exception as e:
            logger.warning("Failed to load face DB: %s — starting empty", e)

    def _save(self) -> None:
        """Persist database to JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        members = []
        for mid, emb in self.embeddings.items():
            meta = self.metadata.get(mid, {})
            members.append({
                "member_id": mid,
                "name": meta.get("name", mid),
                "type": meta.get("type", "member"),
                "embedding": emb.tolist(),
                "registered_at": meta.get("registered_at", ""),
            })
        data = {"members": members, "threshold": self.threshold}
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """1:N cosine similarity search for best match.

        Args:
            embedding: Query embedding vector (float32).

        Returns:
            (member_id, score) if above threshold, else (None, best_score).
        """
        if not self.embeddings:
            return None, 0.0
        query = embedding / (np.linalg.norm(embedding) + 1e-10)
        best_id: str | None = None
        best_score: float = 0.0
        for mid, emb in self.embeddings.items():
            norm_emb = emb / (np.linalg.norm(emb) + 1e-10)
            score = float(np.dot(query, norm_emb))
            if score > best_score:
                best_score = score
                best_id = mid
        if best_score >= self.threshold:
            return best_id, best_score
        return None, best_score

    def register(self, member_id: str, embedding: np.ndarray,
                 name: str = "", member_type: str = "member") -> None:
        """Register a new member embedding."""
        self.embeddings[member_id] = embedding.astype(np.float32)
        self.metadata[member_id] = {
            "name": name or member_id,
            "type": member_type,
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._save()
        logger.info("Registered member %s (%s)", member_id, name)

    def remove(self, member_id: str) -> None:
        """Remove a member from the database."""
        self.embeddings.pop(member_id, None)
        self.metadata.pop(member_id, None)
        self._save()

    def get_member_info(self, member_id: str) -> dict | None:
        """Return metadata for a member, or None."""
        meta = self.metadata.get(member_id)
        if meta is None:
            return None
        return {"member_id": member_id, **meta}
