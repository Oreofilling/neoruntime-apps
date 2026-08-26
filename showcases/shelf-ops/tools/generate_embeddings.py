#!/usr/bin/env python3
"""shelf-ops vocabulary embedding generator (offline provisioning).

Writes the float32 (N,512) unit-norm CLIP text matrix for the configured
vocabulary so the app can run with EMBEDDING_MODE=file (no first-boot
EncodeText pass). Output = .npy matrix + JSON sidecar {codes, prompts,
template} pinning row order — the exact format vocab.py loads, so a
vocabulary change is detected and rebuilt rather than mislabeling slots.

Two modes:

  device : run ON the NE503 (or via tunnel) — uses the device EncodeText RPC
           (~2s/call, N calls), identical to the app's first-boot build.
  clip   : run on x86 with torch + open_clip — the same CLIP ViT-B/32 text
           encoder, reproduces the matrix offline.
           Requires: pip install open_clip_torch torch timm

Usage:
  python tools/generate_embeddings.py --mode device \
      [--out assets/vocab_embeddings_f32.npy]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
APP_DIR = TOOLS_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import load_config  # noqa: E402
from vocab import (  # noqa: E402
    DEFAULT_TEMPLATE, Vocabulary, load_or_build_vocab_embeddings,
)

DEFAULT_OUT = APP_DIR / "assets" / "vocab_embeddings_f32.npy"


def _device_encode(text: str) -> np.ndarray:
    from hailo_ipc_sdk.inference import InferenceClient

    endpoint = os.environ.get("AI_RUNTIME_ENDPOINT",
                              "unix:///run/aipc/ai-runtime.sock")
    client = InferenceClient(endpoint=endpoint)
    client.connect()
    # Fail fast on a bad endpoint before the N-call loop.
    probe = np.asarray(client.encode_text("a photo of a bottle",
                                          timeout_ms=20000))
    if probe.size != 512:
        raise RuntimeError(f"EncodeText probe returned {probe.size} dims")
    print(f"[gen] EncodeText probe ok (dim={probe.size})")
    return np.asarray(client.encode_text(text, timeout_ms=60000),
                      dtype=np.float32)


def _clip_encode_factory():
    try:
        import torch
        import open_clip
    except ImportError as e:
        sys.exit(
            f"[gen] clip mode needs torch + open_clip_torch (+ timm): {e}\n"
            "pip install torch open_clip_torch timm   (x86 only, not in app image)"
        )

    print("[gen] loading open_clip ViT-B/32...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B/32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B/32")

    def encode(text: str) -> np.ndarray:
        with torch.no_grad():
            feat = model.encode_text(tokenizer([text]))
            feat = feat / feat.norm(dim=-1, keepdim=True)  # CLIP UnitNorm
        return feat[0].detach().cpu().numpy().astype(np.float32)

    return encode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("device", "clip"), default="device")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    cfg = load_config()
    vocab = Vocabulary(cfg.vocabulary)
    encode = _device_encode if args.mode == "device" else _clip_encode_factory()

    mat = load_or_build_vocab_embeddings(
        path=args.out,
        entries=vocab.sorted_entries(),
        encode=encode,
        template=cfg.encode_template or DEFAULT_TEMPLATE,
    )
    print(f"[gen] matrix {mat.shape} dtype={mat.dtype} "
          f"row-norms≈{np.linalg.norm(mat, axis=1).mean():.4f}")
    print(f"[gen] cache -> {args.out} (+ .json sidecar)")
    if cfg.embedding_mode != "file":
        print("[gen] note: app EMBEDDING_MODE is 'device'; set it to 'file' "
              f"and EMBEDDING_PATH={args.out} to use this cache")


if __name__ == "__main__":
    main()
