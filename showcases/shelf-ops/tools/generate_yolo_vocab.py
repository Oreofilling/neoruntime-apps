#!/usr/bin/env python3
"""Generate the yolo_world detection vocabulary assets (offline, host-side).

Embeds a curated goods/negative prompt list with CLIP ViT-B/32 text encoder
and writes the pair the detector loads at runtime:

  assets/yolo_world_vocab.npy   float32 (80, 512)   — HEF layer2 contract
  assets/yolo_world_vocab.json  {labels, categories, n_goods, provenance}

The 80-row contract is fixed by the compiled HEF (input_layer2
[1, 80, 512] uint16). Goods prompts come first; remaining rows are padded
with negative (non-goods) prompts so on-chip NMS has an escape class for
shelf boards / price tags / people.

Encoding pipeline is byte-identical to the probe set that passed the
bus.jpg + focused-camera oracles on device (probes 50-53):
  transformers AutoTokenizer(openai/clip-vit-base-patch32)
  -> pad/truncate to SEQ_LEN 20 with PAD_VALUE 49407
  -> clip_text.onnx (CPU) -> raw (n, 512) float32 (NOT L2-normalized —
  the HEF quantization range was fitted to raw embeddings).

Prompt list is pinned by the probe53 measurement on /home/share/
stocked_shelf.jpg (2x2 tiles, threshold sweep): every goods prompt kept
here actually fires on a real shelf (or belongs to its visual family);
never-firing non-retail variants (shampoo/detergent/spray bottle, bare
"drink"/"snack"/"box" duplicates) were dropped. `categories` groups the
synonym families so counting can report per-category item tallies.

Usage (system python3 with transformers + onnxruntime + the text ONNX):
  python3 tools/generate_yolo_vocab.py \
      [--clip-onnx /tmp/clip_text.onnx] [--out-dir assets]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SEQ_LEN = 20
PAD_VALUE = 49407
EMB_ROWS = 80
EMB_DIM = 512
DEFAULT_TOKENIZER = os.environ.get(
    "CLIP_TOKENIZER", "openai/clip-vit-base-patch32")

# (prompt, category) — goods rows [0, n_goods). Order is part of the
# contract: changing it changes cls indices baked into any saved analysis.
GOODS: list[tuple[str, str]] = [
    ("bottle", "bottle"),
    ("water bottle", "bottle"),
    ("juice bottle", "bottle"),
    ("sauce bottle", "bottle"),
    ("condiment bottle", "bottle"),
    ("glass bottle", "bottle"),
    ("plastic bottle", "bottle"),
    ("soda can", "can"),
    ("beverage can", "can"),
    ("juice box", "carton"),
    ("milk carton", "carton"),
    ("snack bag", "bag"),
    ("bag of chips", "bag"),
    ("jar", "jar"),
    ("glass jar", "jar"),
    ("package", "box"),
    ("packaged goods", "box"),
    ("noodle package", "other"),
    ("cookie", "other"),
    ("candy", "other"),
    ("chocolate", "other"),
    ("product", "other"),
    ("retail product", "other"),
    ("grocery item", "other"),
]

# Environment / non-goods rows. "shelf*" and "person" matter most: probe53
# showed shelf boards fire up to 0.47 and must land on a negative row so
# they are excluded from goods counting.
NEGATIVES: list[str] = [
    "shelf", "shelf board", "shelf edge", "price tag",
    "label", "sticker", "barcode", "person",
    "hand", "arm", "face", "wall",
    "floor", "ceiling", "light", "lamp",
    "reflection", "background", "table", "chair",
    "cup", "bowl", "plate", "book",
    "clock", "vase", "tv", "laptop",
    "cell phone", "keyboard", "mouse", "remote",
    "plant", "door", "window", "trash can",
    "basket", "handbag", "backpack", "cloth",
    "sign", "poster", "monitor", "shopping cart",
    "checkout counter", "freezer door", "cooler door", "floor mat",
    "printer", "speaker", "cable", "power strip",
    "card reader", "receipt", "escalator", "cart wheel",
]

CATEGORIES = ("bottle", "can", "carton", "bag", "jar", "box", "other")


def build_rows() -> tuple[list[str], list[str], int]:
    """goods + negatives padded/truncated to exactly EMB_ROWS rows."""
    goods = GOODS[: EMB_ROWS]
    n_goods = len(goods)
    negs = NEGATIVES[: EMB_ROWS - n_goods]
    labels = [p for p, _ in goods] + negs
    categories = [c for _, c in goods] + ["negative"] * len(negs)
    if len(labels) != EMB_ROWS:
        raise SystemExit(
            f"vocab must be exactly {EMB_ROWS} rows, got {len(labels)} "
            f"({n_goods} goods + {len(negs)} negatives)")
    if len(set(labels)) != EMB_ROWS:
        dupes = {p for p in labels if labels.count(p) > 1}
        raise SystemExit(f"duplicate prompts: {sorted(dupes)}")
    return labels, categories, n_goods


def embed(prompts: list[str], clip_onnx: str, tokenizer: str) -> np.ndarray:
    """CLIP text embeddings, contract-identical to the probe pipeline."""
    from transformers import AutoTokenizer
    import onnxruntime as ort

    tok = AutoTokenizer.from_pretrained(tokenizer)
    enc = tok(list(prompts), add_special_tokens=True, padding=True)
    ids = np.array(enc["input_ids"], dtype=np.int64)
    n, sl = ids.shape
    if sl > SEQ_LEN:
        raise SystemExit(f"prompt batch too long: {sl} > {SEQ_LEN}")
    padded = np.full((n, SEQ_LEN), PAD_VALUE, dtype=np.int64)
    padded[:, :sl] = ids

    sess = ort.InferenceSession(clip_onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    rows = [sess.run(None, {iname: padded[i:i + 1]})[0][0]
            for i in range(n)]
    emb = np.stack(rows).astype(np.float32)
    if emb.shape != (EMB_ROWS, EMB_DIM):
        raise SystemExit(f"unexpected embedding shape {emb.shape}")
    norms = np.linalg.norm(emb, axis=1)
    if not np.isfinite(emb).all() or norms.min() < 1e-3:
        raise SystemExit("degenerate embeddings (nan/zero rows)")
    print(f"[vocab] embeds {emb.shape} range=[{emb.min():.4f},"
          f"{emb.max():.4f}] rownorm min/med/max={norms.min():.3f}/"
          f"{np.median(norms):.3f}/{norms.max():.3f}")
    return emb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip-onnx", default="/tmp/clip_text.onnx",
                    help="CLIP ViT-B/32 text encoder ONNX (default: "
                         "/tmp/clip_text.onnx)")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help=f"HF tokenizer id (default: {DEFAULT_TOKENIZER})")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: repo assets/ next to "
                         "this script)")
    args = ap.parse_args()

    if not Path(args.clip_onnx).is_file():
        raise SystemExit(f"clip onnx not found: {args.clip_onnx}")

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(__file__).resolve().parent.parent / "assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, categories, n_goods = build_rows()
    print(f"[vocab] {n_goods} goods + {EMB_ROWS - n_goods} negatives")

    emb = embed(labels, args.clip_onnx, args.tokenizer)
    npy_path = out_dir / "yolo_world_vocab.npy"
    with open(npy_path, "wb") as f:
        np.save(f, emb)          # proper .npy (headered) — detector np.load's it
    print(f"[vocab] wrote {npy_path} ({emb.size} f32)")

    prompts_sha = hashlib.sha256(
        "\n".join(labels).encode()).hexdigest()[:16]
    sidecar = {
        "labels": labels,
        "categories": categories,
        "n_goods": n_goods,
        "provenance": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tokenizer": args.tokenizer,
            "clip_onnx": args.clip_onnx,
            "seq_len": SEQ_LEN,
            "pad_value": PAD_VALUE,
            "prompts_sha256_16": prompts_sha,
            "pinned_by": "probe53 threshold/tile sweep on "
                         "stocked_shelf.jpg (device 192.168.93.72)",
        },
    }
    json_path = out_dir / "yolo_world_vocab.json"
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=1)
    print(f"[vocab] wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
