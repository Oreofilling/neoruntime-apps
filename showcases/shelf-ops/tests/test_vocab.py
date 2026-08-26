"""Unit tests for vocab.py (prompt vocabulary + cached text embeddings)."""
import json

import numpy as np
import pytest

from vocab import (
    DEFAULT_TEMPLATE,
    EMBED_DIM,
    Vocabulary,
    load_or_build_vocab_embeddings,
)

GROUPS = {
    "A": {"label_cn": "瓶装水", "label": "bottled water",
          "prompt": "a clear plastic bottle of water"},
    "B": {"label_cn": "罐装饮品", "label": "canned drink",
          "prompt": "an aluminum beverage can"},
    "EMPTY": {"label_cn": "空位", "label": "empty shelf",
              "prompt": "empty supermarket shelf with no products"},
}


class _Encoder:
    """Counts calls and returns distinguishable non-unit vectors."""

    def __init__(self, magnitude: float = 3.0) -> None:
        self.magnitude = magnitude
        self.calls: list[str] = []

    def __call__(self, text: str) -> np.ndarray:
        self.calls.append(text)
        v = np.full(EMBED_DIM, 0.1, dtype=np.float32)
        v[len(self.calls) % EMBED_DIM] = 1.0
        return v * self.magnitude


def _boom(text: str) -> np.ndarray:
    raise AssertionError("encode must not be called when the cache is valid")


def test_vocabulary_requires_prompt():
    with pytest.raises(ValueError):
        Vocabulary({"A": {"label": "x", "label_cn": "x"}})  # no prompt


def test_vocabulary_order_and_lookup():
    vocab = Vocabulary(GROUPS)
    assert vocab.codes() == ["A", "B", "EMPTY"]       # sorted, row-ordered
    assert len(vocab) == 3
    e = vocab.entry_for("empty")
    assert e is not None and e.prompt.startswith("empty ")
    assert vocab.entry_for("Z") is None
    assert [p.prompt for p in vocab.sorted_entries()] == [
        GROUPS[c]["prompt"] for c in vocab.codes()]


def test_builds_unit_norm_matrix_and_caches(tmp_path):
    enc = _Encoder()
    path = str(tmp_path / "emb.npy")
    entries = Vocabulary(GROUPS).sorted_entries()

    mat = load_or_build_vocab_embeddings(
        path=path, entries=entries, encode=enc,
        template=DEFAULT_TEMPLATE, verbose=False)

    assert mat.shape == (3, EMBED_DIM) and mat.dtype == np.float32
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert len(enc.calls) == 3
    # prompts reach the encoder template-applied
    assert enc.calls[0] == f"a photo of {GROUPS['A']['prompt']}"
    # cache artifacts on disk
    assert np.asarray(np.load(path)).shape == (3, EMBED_DIM)
    sidecar = json.load(open(path + ".json"))
    assert sidecar["codes"] == ["A", "B", "EMPTY"]
    assert sidecar["template"] == DEFAULT_TEMPLATE


def test_second_load_hits_cache_without_encoding(tmp_path):
    enc = _Encoder()
    path = str(tmp_path / "emb.npy")
    entries = Vocabulary(GROUPS).sorted_entries()
    load_or_build_vocab_embeddings(path=path, entries=entries, encode=enc,
                                   template=DEFAULT_TEMPLATE, verbose=False)
    first = list(enc.calls)

    mat2 = load_or_build_vocab_embeddings(path=path, entries=entries,
                                          encode=_boom,
                                          template=DEFAULT_TEMPLATE, verbose=False)
    assert mat2.shape == (3, EMBED_DIM)
    assert first  # sanity: the build pass did encode


def test_vocabulary_change_triggers_rebuild(tmp_path):
    path = str(tmp_path / "emb.npy")
    entries = Vocabulary(GROUPS).sorted_entries()
    load_or_build_vocab_embeddings(path=path, entries=entries, encode=_Encoder(),
                                   template=DEFAULT_TEMPLATE, verbose=False)

    changed = dict(GROUPS, **{
        "A": {"label_cn": "瓶装水", "label": "bottled water",
              "prompt": "a glass bottle of mineral water"}})
    enc2 = _Encoder()
    mat = load_or_build_vocab_embeddings(
        path=path, entries=Vocabulary(changed).sorted_entries(),
        encode=enc2, template=DEFAULT_TEMPLATE, verbose=False)
    assert len(enc2.calls) == 3                     # re-encoded everything
    assert mat.shape == (3, EMBED_DIM)


def test_template_change_triggers_rebuild(tmp_path):
    path = str(tmp_path / "emb.npy")
    entries = Vocabulary(GROUPS).sorted_entries()
    load_or_build_vocab_embeddings(path=path, entries=entries, encode=_Encoder(),
                                   template=DEFAULT_TEMPLATE, verbose=False)
    enc2 = _Encoder()
    load_or_build_vocab_embeddings(path=path, entries=entries, encode=enc2,
                                   template="a shelf photo of {}", verbose=False)
    assert len(enc2.calls) == 3


def test_corrupt_or_shape_mismatched_cache_rebuilds(tmp_path):
    path = str(tmp_path / "emb.npy")
    entries = Vocabulary(GROUPS).sorted_entries()

    # garbage bytes instead of a npy
    with open(path, "wb") as f:
        f.write(b"\x00not-an-npy")
    enc = _Encoder()
    load_or_build_vocab_embeddings(path=path, entries=entries, encode=enc,
                                   template=DEFAULT_TEMPLATE, verbose=False)
    assert len(enc.calls) == 3

    # valid sidecar but wrong matrix shape
    enc = _Encoder()
    sidecar = json.load(open(path + ".json"))
    np.save(path, np.zeros((3, 128), dtype=np.float32))
    with open(path + ".json", "w") as f:
        json.dump(sidecar, f)
    load_or_build_vocab_embeddings(path=path, entries=entries, encode=enc,
                                   template=DEFAULT_TEMPLATE, verbose=False)
    assert len(enc.calls) == 3

    # matrix fine but sidecar unreadable
    np.save(path, np.zeros((3, EMBED_DIM), dtype=np.float32))
    with open(path + ".json", "w") as f:
        f.write("{not json")
    enc = _Encoder()
    load_or_build_vocab_embeddings(path=path, entries=entries, encode=enc,
                                   template=DEFAULT_TEMPLATE, verbose=False)
    assert len(enc.calls) == 3


def test_wrong_dim_encode_raises(tmp_path):
    def bad(text: str) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    with pytest.raises(ValueError):
        load_or_build_vocab_embeddings(
            path=str(tmp_path / "emb.npy"),
            entries=Vocabulary(GROUPS).sorted_entries(),
            encode=bad, template=DEFAULT_TEMPLATE, verbose=False)


def test_empty_path_skips_cache():
    enc = _Encoder()
    mat = load_or_build_vocab_embeddings(
        path="", entries=Vocabulary(GROUPS).sorted_entries(),
        encode=enc, template=DEFAULT_TEMPLATE, verbose=False)
    assert mat.shape == (3, EMBED_DIM)
    assert len(enc.calls) == 3
