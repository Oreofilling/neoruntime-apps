"""shelf-ops vocabulary + CLIP text-embedding matrix (slot-anchored edition).

The detector path was removed after the device evidence chain
(docs/device-gate.md §6-§7): yolo_world_v2s class outputs are information-free
constants and the platform yolov8n head (person/vehicle/face) cannot see shelf
products. Recognition is now slot-anchored CLIP:

    crop slot polygon -> CLIP ViT-B/32 image embed (~48 ms/slot)
    cos(image, EncodeText(template.format(prompt))) -> argmax over vocabulary

The vocabulary maps display codes (A-E plus the vacancy code from slots.EMPTY_
CODE) to one CLIP prompt each. load_or_build_vocab_embeddings() encodes every
prompt once via device EncodeText (~2 s each) and caches the unit-norm float32
matrix next to a JSON sidecar pinning code+prompt order — a cache written for
a different vocabulary is detected and rebuilt rather than silently
mislabeling slots.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

DEFAULT_TEMPLATE = "a photo of {}"
EMBED_DIM = 512


@dataclass(frozen=True)
class VocabEntry:
    """One vocabulary row: display code -> CLIP text prompt."""

    code: str        # "A".."E" or the vacancy code (slots.EMPTY_CODE)
    label: str       # english display label
    label_cn: str    # chinese display label
    prompt: str      # CLIP prompt phrase; template fills the {} slot


class Vocabulary:
    """Display codes -> CLIP prompts, ordered like the embedding rows."""

    def __init__(self, groups: dict[str, dict] | None) -> None:
        raw = groups or {}
        self.entries: list[VocabEntry] = []
        seen: set[str] = set()
        for code in sorted(raw.keys()):
            g = raw[code] or {}
            entry = VocabEntry(
                code=str(code).upper(),
                label=str(g.get("label", "")),
                label_cn=str(g.get("label_cn", "")),
                prompt=str(g.get("prompt", "")),
            )
            if not entry.prompt:
                raise ValueError(
                    f"vocabulary entry '{entry.code}': 'prompt' is required")
            if entry.code in seen:
                raise ValueError(f"duplicate vocabulary code {entry.code}")
            seen.add(entry.code)
            self.entries.append(entry)

    def codes(self) -> list[str]:
        """Row-ordered display codes (index i matches embedding row i)."""
        return [e.code for e in self.entries]

    def entry_for(self, code: str) -> VocabEntry | None:
        for e in self.entries:
            if e.code == str(code).upper():
                return e
        return None

    def sorted_entries(self) -> list[VocabEntry]:
        return list(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _sidecar_path(path: str) -> str:
    return path + ".json"


def load_or_build_vocab_embeddings(
    path: str,
    entries: list[VocabEntry],
    encode: Callable[[str], np.ndarray],
    template: str = DEFAULT_TEMPLATE,
    verbose: bool = True,
) -> np.ndarray:
    """Return the (N, 512) float32 unit-norm text matrix for ``entries``.

    ``encode`` is called with each full prompt text (template applied) and
    must return a float vector; it is only invoked on a cache miss. Raises
    ValueError if an encode returns the wrong dimension. Cache = npy matrix +
    JSON sidecar; any vocabulary/template/file mismatch triggers a rebuild.
    """
    expected_n = len(entries)
    codes = [e.code for e in entries]
    texts = [template.format(e.prompt) for e in entries]

    if path and os.path.exists(path):
        mat = _load_cache(path, codes, texts, template, expected_n, verbose)
        if mat is not None:
            return mat

    if verbose:
        print(f"[vocab] encoding {expected_n} prompts via EncodeText "
              f"(~2 s each, one-time)")
    rows: list[np.ndarray] = []
    for i, text in enumerate(texts):
        emb = np.asarray(encode(text), dtype=np.float32).reshape(-1)
        if emb.size != EMBED_DIM:
            raise ValueError(
                f"EncodeText returned {emb.size} dims for '{text}', "
                f"expected {EMBED_DIM}")
        if verbose:
            print(f"  [{i + 1}/{expected_n}] {text}")
        rows.append(emb)

    mat = np.stack(rows).astype(np.float32)
    mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)

    if path:
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            np.save(path, mat)
            with open(_sidecar_path(path), "w", encoding="utf-8") as f:
                json.dump({"codes": codes, "prompts": texts, "template": template},
                          f, ensure_ascii=False, indent=1)
            if verbose:
                print(f"[vocab] cached embeddings -> {path}")
        except OSError as e:
            if verbose:
                print(f"[vocab] cache write failed ({e}); continuing in-memory")
    return mat


def _load_cache(
    path: str,
    codes: list[str],
    texts: list[str],
    template: str,
    expected_n: int,
    verbose: bool,
) -> np.ndarray | None:
    """Load a cached matrix only if its sidecar matches this vocabulary."""
    try:
        mat = np.asarray(np.load(path), dtype=np.float32)
    except (OSError, ValueError) as e:
        if verbose:
            print(f"[vocab] cache load failed ({e}); rebuilding")
        return None
    if mat.shape != (expected_n, EMBED_DIM):
        if verbose:
            print(f"[vocab] cached {path} shape {mat.shape} != "
                  f"{(expected_n, EMBED_DIM)}; rebuilding")
        return None
    try:
        with open(_sidecar_path(path), "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        if verbose:
            print(f"[vocab] sidecar unreadable ({e}); rebuilding")
        return None
    if (meta.get("codes") != codes or meta.get("prompts") != texts
            or meta.get("template") != template):
        if verbose:
            print("[vocab] cached vocabulary differs from config; rebuilding")
        return None
    if verbose:
        print(f"[vocab] loaded cached embeddings from {path}")
    return mat
