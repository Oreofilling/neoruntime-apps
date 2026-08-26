"""shelf-ops configuration (slot-anchored CLIP + per-item detection).

Two models: the CLIP ViT-B/32 image encoder for per-slot classification
(docs/device-gate.md §7), and the yolo_world v5.4.0 detector for per-item
goods boxes (§9 — §6's "information-free constants" verdict was about the
v5.3.0 raw-tensor artifact and is superseded for v5.4.0's NMS output).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from hailo_ipc_sdk.config import Config

from slots import EMPTY_CODE, build_grid_slots

# Inference timeout tiering: registration/first-inference can be slow on a
# busy runtime, steady-state classification is ~48 ms per slot crop.
INFER_TIMEOUT_MS = 3000
INFER_WARMUP_TIMEOUT_MS = 20000
INFER_WARMUP_COUNT = 12          # first N CLIP calls per boot
DEGRADED_FAILURE_THRESHOLD = 10

# Slot-crop defaults.
DEFAULT_SLOT_CROP_PAD = 0.05     # relative padding around a slot polygon
DEFAULT_CLIP_MIN_COS = 0.15      # below this a tick holds the last observation

# Per-item detection defaults (yolo_world v5.4.0; device-gate.md §9): the
# HEF's on-chip threshold is ~0, so the host thresholds at 0.25 and then
# runs a greedy cross-class NMS at IoU 0.45.
DEFAULT_DETECT_THRESHOLD = 0.25
DEFAULT_DETECT_NMS_IOU = 0.45
DEFAULT_DETECT_TIMEOUT_MS = 15000
DEFAULT_DETECT_MAX_ITEMS = 64

# One-shot photo imports get the accurate path: 2x2 tiled inference
# (~2x linear resolution for small goods) at the probe53-pinned threshold
# — stocked_shelf.jpg goes 22 items/2-of-4 rows -> ~63 items/all 4 rows.
# The live scan loop stays full-frame (tiled would cut it to ~0.5 fps).
DEFAULT_IMPORT_DETECT_TILES = 4
DEFAULT_IMPORT_DETECT_THRESHOLD = 0.15
DEFAULT_IMPORT_DETECT_MAX_ITEMS = 200

# Built-in demo video (assets/shelf_transition.mp4): loops the gapped ->
# restocked -> full shelf through the LIVE pipeline for customer demos
# (empty-slot detection, stockout/restock events, ITEMS, heatmap). The demo
# runs its own d-prefixed 4x6 grid so analytics rows stay separate from the
# live cells; pacing 0.5x makes one loop ~21 s (empty ~8.4 s, hands ~2.4 s,
# full ~10.2 s), so a 5 s stockout threshold + 3 s alert cooldown fire
# events every loop instead of being swallowed by the live 30 s cooldown.
DEFAULT_DEMO_VIDEO_PATH = "assets/demo.mp4"
DEFAULT_DEMO_SPEED = 0.5
# demo.mp4: portrait 540x960 fridge, 2 shelf rows (top ~8 items, bottom ~12),
# shelf fills ~90% height / full width. Gaps only open in the last third of
# the loop, so the sustained-empty window is short -> stockout after 3 s.
DEFAULT_DEMO_GRID_REGION = (0.02, 0.04, 0.98, 0.99)
DEFAULT_DEMO_GRID_ROWS = 2
DEFAULT_DEMO_GRID_COLS = 8
DEFAULT_DEMO_STOCKOUT_SECONDS = 3.0
DEFAULT_DEMO_ALERT_COOLDOWN_SECONDS = 3

# Grid mode defaults (path A: one big region + virtual cell slots; see
# docs/device-gate.md §8 — the geometry-only detector path was falsified).
SLOT_MODES = ("slots", "grid")
DEFAULT_GRID_REGION = (0.0, 0.0, 1.0, 1.0)
DEFAULT_GRID_ROWS = 3
DEFAULT_GRID_COLS = 4


@dataclass(frozen=True)
class ModelDef:
    name: str
    path: str
    model_type: str
    input_format: str
    input_size: tuple[int, int]
    variant: str | None
    inputs: list[dict[str, Any]] | None
    postprocess_json: str | None = None


MODEL_DEFS: dict[str, ModelDef] = {
    "clip_vit_b_32": ModelDef(
        name="clip_vit_b_32",
        path="clip/clip_vit_b_32_image_encoder_nv12.hef",
        model_type="embedding",
        input_format="nv12",
        input_size=(224, 224),
        variant='{"backend_function": "identity"}',
        inputs=[{"name": "input_layer1", "shape": [1, 224, 224, 3],
                 "dtype": "uint8"}],
        postprocess_json='{"backend_function": "identity"}',
    ),
    "yolo_world_540": ModelDef(
        name="yolo_world_540",
        path="yolo_world_v2s_540.hef",
        model_type="detection",
        input_format="rgb",
        input_size=(640, 640),
        variant='{"backend_function": "identity"}',
        inputs=[{"name": "yolo_world_v2s/input_layer1",
                 "shape": [1, 640, 640, 3], "dtype": "uint8"},
                {"name": "yolo_world_v2s/input_layer2",
                 "shape": [1, 80, 512], "dtype": "uint16"}],
    ),
}


def _default_slots() -> list[dict]:
    """2 rows x 5 columns demo grid (10 slots, capacity 1 each)."""
    slots: list[dict] = []
    codes = ["A", "B", "C", "D", "E"]
    rows = ((0.32, 0.62), (0.66, 0.96))
    margin, gap = 0.04, 0.015
    width = (1.0 - 2 * margin - 4 * gap) / 5.0
    i = 0
    for r, (y1, y2) in enumerate(rows):
        for c in range(5):
            x1 = margin + c * (width + gap)
            slots.append({
                "id": f"s{i + 1}",
                "name": f"{r + 1}-{c + 1} Shelf",
                "expected_code": codes[c],
                "capacity": 1,
                "polygon": [[x1, y1], [x1 + width, y1],
                            [x1 + width, y2], [x1, y2]],
            })
            i += 1
    return slots


_DEFAULT_VOCABULARY: dict[str, dict[str, str]] = {
    "A": {"label_cn": "瓶装水", "label": "bottled water",
          "prompt": "a clear plastic bottle of water"},
    "B": {"label_cn": "罐装饮品", "label": "canned drink",
          "prompt": "an aluminum beverage can"},
    "C": {"label_cn": "盒装零食", "label": "boxed snack",
          "prompt": "a cardboard box of snacks"},
    "D": {"label_cn": "瓶装饮品", "label": "soft drink bottle",
          "prompt": "a plastic bottle of soft drink"},
    "E": {"label_cn": "水果", "label": "fresh fruit",
          "prompt": "fresh fruit like apples and oranges"},
    EMPTY_CODE: {"label_cn": "空位", "label": "empty shelf",
                 "prompt": "empty supermarket shelf with no products"},
}


@dataclass
class ShelfConfig:
    web_port: int = 8891
    stream_id: str = "main"
    log_level: str = "INFO"
    infer_fps: float = 2.0
    preview_fps: int = 15
    preview_width: int = 1280

    model_root: str = "/data/aipc/models"
    clip_model: str = "clip_vit_b_32"
    clip_timeout_ms: int = 3000
    clip_register_retries: int = 3
    clip_min_cos: float = DEFAULT_CLIP_MIN_COS
    slot_crop_pad: float = DEFAULT_SLOT_CROP_PAD

    embedding_mode: str = "device"      # device | file
    embedding_path: str = "/data/aipc/etc/shelf-ops/vocab_embeddings_f32.npy"
    encode_template: str = "a photo of {}"

    # Per-item goods detection (yolo_world v5.4.0; device-gate.md §9).
    # detect_enabled=False rolls the app back to pure CLIP behavior.
    detect_enabled: bool = True
    detect_model: str = "yolo_world_540"
    detect_threshold: float = DEFAULT_DETECT_THRESHOLD
    detect_nms_iou: float = DEFAULT_DETECT_NMS_IOU
    detect_timeout_ms: int = DEFAULT_DETECT_TIMEOUT_MS
    detect_register_retries: int = 3
    detect_emb_path: str = ""       # empty = bundled assets/yolo_world_vocab.*
    detect_tiles: int = 1           # live scan: 1 = full-frame (2x2 opt-in)
    detect_max_items: int = DEFAULT_DETECT_MAX_ITEMS

    # One-shot photo imports: tiled + low threshold + high cap (probe53).
    import_detect_tiles: int = DEFAULT_IMPORT_DETECT_TILES
    import_detect_threshold: float = DEFAULT_IMPORT_DETECT_THRESHOLD
    import_detect_max_items: int = DEFAULT_IMPORT_DETECT_MAX_ITEMS

    # Offline photo import (media_import.py): upload cap per image.
    import_max_image_mb: int = 15

    # Offline video import: upload cap + sampling bounds. max_frames bounds
    # NPU time (one tiled detect per sampled frame); keyframes bounds the
    # result payload (annotated stills returned as base64 JPEG).
    import_max_video_mb: int = 200
    import_video_max_frames: int = 48
    import_video_keyframes: int = 6
    # Where a successful import is persisted for demo playback. Empty =
    # auto: a media/ folder next to the analytics db (the persistent config
    # volume on device). One file (imported.mp4), overwritten per import.
    import_media_dir: str = ""

    # Built-in demo video mode (POST /api/demo): loop the bundled
    # shelf_transition.mp4 through the live pipeline for customer demos.
    demo_video_path: str = DEFAULT_DEMO_VIDEO_PATH
    demo_speed: float = DEFAULT_DEMO_SPEED
    demo_grid_region: tuple[float, float, float, float] = DEFAULT_DEMO_GRID_REGION
    demo_grid_rows: int = DEFAULT_DEMO_GRID_ROWS
    demo_grid_cols: int = DEFAULT_DEMO_GRID_COLS
    demo_stockout_seconds: float = DEFAULT_DEMO_STOCKOUT_SECONDS
    demo_alert_cooldown_seconds: int = DEFAULT_DEMO_ALERT_COOLDOWN_SECONDS
    # import-grade detection settings so ITEMS visibly climbs on the restock
    demo_detect_tiles: int = DEFAULT_IMPORT_DETECT_TILES
    demo_detect_threshold: float = DEFAULT_IMPORT_DETECT_THRESHOLD
    demo_detect_max_items: int = DEFAULT_IMPORT_DETECT_MAX_ITEMS

    slots: list[dict] = field(default_factory=_default_slots)
    vocabulary: dict[str, dict[str, str]] = field(
        default_factory=lambda: dict(_DEFAULT_VOCABULARY))

    # "slots" = calibrated planogram polygons; "grid" = one big region tiled
    # with rows x cols virtual cells (path A big-frame counting).
    slot_mode: str = "slots"
    grid_region: tuple[float, float, float, float] = DEFAULT_GRID_REGION
    grid_rows: int = DEFAULT_GRID_ROWS
    grid_cols: int = DEFAULT_GRID_COLS
    # render-only: False hides the virtual cell borders/chips so the whole
    # region reads as ONE shelf (cells still drive CLIP counting internally)
    grid_show_cells: bool = True

    analytics_db: str = "/data/aipc/data/shelf-ops/shelf.db"
    bucket_seconds: int = 600
    retention_hours: int = 336
    heatmap_min_popularity: float = 0.1
    alert_cooldown_seconds: int = 30
    stockout_seconds: int = 120
    restock_cooldown_seconds: int = 60

    # platform-api hardware H.264 preview (HD mode; mirrors gym-ops)
    platform_api_ws_scheme: str = "wss"
    platform_api_port: int = 443
    platform_api_token: str = ""

    def clip_def(self) -> ModelDef:
        return MODEL_DEFS[self.clip_model]

    def clip_full_path(self) -> str:
        return os.path.join(self.model_root, self.clip_def().path)

    def detect_def(self) -> ModelDef:
        return MODEL_DEFS[self.detect_model]

    def detect_full_path(self) -> str:
        return os.path.join(self.model_root, self.detect_def().path)

    def grid_dict(self) -> dict:
        """Grid params for /api/config and the frontend overlay."""
        return {
            "region": [float(v) for v in self.grid_region],
            "rows": self.grid_rows,
            "cols": self.grid_cols,
            "show_cells": self.grid_show_cells,
        }


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (ValueError, AttributeError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (ValueError, AttributeError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _load_yaml_overlay(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    import yaml  # local import: only needed when an overlay exists
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as e:
        print(f"[config] overlay {path} unreadable ({e}); using defaults")
        return {}


def load_config() -> ShelfConfig:
    """Build the effective config from the YAML overlay + environment."""
    overlay_path = _env_str("CONFIG_PATH", "")
    data = _load_yaml_overlay(overlay_path)

    slots_cfg = data.get("slots")
    if not isinstance(slots_cfg, list) or not slots_cfg:
        slots_cfg = _default_slots()

    vocab_cfg = data.get("vocabulary")
    if not isinstance(vocab_cfg, dict) or not vocab_cfg:
        vocab_cfg = dict(_DEFAULT_VOCABULARY)

    slot_mode = _env_str("SLOT_MODE", str(data.get("slot_mode", "slots")))
    grid_cfg = data.get("grid") if isinstance(data.get("grid"), dict) else {}
    region_raw = grid_cfg.get("region", DEFAULT_GRID_REGION)
    try:
        region = tuple(float(v) for v in region_raw)
        if len(region) != 4:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"grid.region must be 4 numbers: {region_raw!r}")
    grid_rows = _env_int("GRID_ROWS", int(grid_cfg.get("rows", DEFAULT_GRID_ROWS)))
    grid_cols = _env_int("GRID_COLS", int(grid_cfg.get("cols", DEFAULT_GRID_COLS)))
    grid_show_cells = _env_bool(
        "GRID_SHOW_CELLS", bool(grid_cfg.get("show_cells", True)))

    # demo: overlay block tunes the built-in video demo without a rebuild
    # (video_path / speed / grid_region / grid_rows / grid_cols /
    # stockout_seconds / alert_cooldown_seconds / detect_tiles /
    # detect_threshold / detect_max_items).
    demo_cfg = data.get("demo") if isinstance(data.get("demo"), dict) else {}
    demo_region_raw = demo_cfg.get("grid_region", DEFAULT_DEMO_GRID_REGION)
    try:
        demo_region = tuple(float(v) for v in demo_region_raw)
        if len(demo_region) != 4:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"demo.grid_region must be 4 numbers: {demo_region_raw!r}")

    # grid mode replaces the calibrated planogram with generated virtual
    # cells; build_grid_slots validates region/rows/cols (raises on bad input).
    if slot_mode == "grid":
        slots_cfg = build_grid_slots(region, grid_rows, grid_cols)
    elif slot_mode != "slots":
        raise ValueError(
            f"slot_mode must be one of {SLOT_MODES}, got '{slot_mode}'")

    cfg = ShelfConfig(
        web_port=_env_int("WEB_PORT", 8891),
        stream_id=_env_str("STREAM_ID", "main"),
        log_level=_env_str("LOG_LEVEL", "INFO"),
        infer_fps=_env_float("INFER_FPS", 2.0),
        preview_fps=_env_int("PREVIEW_FPS", 15),
        preview_width=_env_int("PREVIEW_WIDTH", 1280),
        model_root=_env_str("MODEL_ROOT", "/data/aipc/models"),
        clip_model=_env_str("CLIP_MODEL", "clip_vit_b_32"),
        clip_timeout_ms=_env_int("CLIP_TIMEOUT_MS", 3000),
        clip_register_retries=_env_int("CLIP_REGISTER_RETRIES", 3),
        clip_min_cos=_env_float("CLIP_MIN_COS", DEFAULT_CLIP_MIN_COS),
        slot_crop_pad=_env_float("SLOT_CROP_PAD", DEFAULT_SLOT_CROP_PAD),
        embedding_mode=_env_str("EMBEDDING_MODE", "device"),
        embedding_path=_env_str(
            "EMBEDDING_PATH",
            "/data/aipc/etc/shelf-ops/vocab_embeddings_f32.npy"),
        encode_template=_env_str("ENCODE_TEMPLATE", "a photo of {}"),
        detect_enabled=_env_bool("DETECT_ENABLED", True),
        detect_model=_env_str("DETECT_MODEL", "yolo_world_540"),
        detect_threshold=_env_float("DETECT_THRESHOLD",
                                    DEFAULT_DETECT_THRESHOLD),
        detect_nms_iou=_env_float("DETECT_NMS_IOU", DEFAULT_DETECT_NMS_IOU),
        detect_timeout_ms=_env_int("DETECT_TIMEOUT_MS",
                                   DEFAULT_DETECT_TIMEOUT_MS),
        detect_register_retries=_env_int("DETECT_REGISTER_RETRIES", 3),
        detect_emb_path=_env_str("DETECT_EMB_PATH", ""),
        detect_tiles=_env_int("DETECT_TILES", 1),
        detect_max_items=_env_int("DETECT_MAX_ITEMS",
                                  DEFAULT_DETECT_MAX_ITEMS),
        import_detect_tiles=_env_int("IMPORT_DETECT_TILES",
                                     DEFAULT_IMPORT_DETECT_TILES),
        import_detect_threshold=_env_float("IMPORT_DETECT_THRESHOLD",
                                           DEFAULT_IMPORT_DETECT_THRESHOLD),
        import_detect_max_items=_env_int("IMPORT_DETECT_MAX_ITEMS",
                                         DEFAULT_IMPORT_DETECT_MAX_ITEMS),
        import_max_image_mb=_env_int("IMPORT_MAX_IMAGE_MB", 15),
        import_max_video_mb=_env_int("IMPORT_MAX_VIDEO_MB", 200),
        import_video_max_frames=_env_int("IMPORT_VIDEO_MAX_FRAMES", 48),
        import_video_keyframes=_env_int("IMPORT_VIDEO_KEYFRAMES", 6),
        import_media_dir=_env_str("IMPORT_MEDIA_DIR", ""),
        demo_video_path=_env_str(
            "DEMO_VIDEO_PATH",
            str(demo_cfg.get("video_path", DEFAULT_DEMO_VIDEO_PATH))),
        demo_speed=_env_float(
            "DEMO_SPEED", float(demo_cfg.get("speed", DEFAULT_DEMO_SPEED))),
        demo_grid_region=demo_region,
        demo_grid_rows=_env_int(
            "DEMO_GRID_ROWS",
            int(demo_cfg.get("grid_rows", DEFAULT_DEMO_GRID_ROWS))),
        demo_grid_cols=_env_int(
            "DEMO_GRID_COLS",
            int(demo_cfg.get("grid_cols", DEFAULT_DEMO_GRID_COLS))),
        demo_stockout_seconds=_env_float(
            "DEMO_STOCKOUT_SECONDS",
            float(demo_cfg.get("stockout_seconds",
                               DEFAULT_DEMO_STOCKOUT_SECONDS))),
        demo_alert_cooldown_seconds=_env_int(
            "DEMO_ALERT_COOLDOWN_SECONDS",
            int(demo_cfg.get("alert_cooldown_seconds",
                             DEFAULT_DEMO_ALERT_COOLDOWN_SECONDS))),
        demo_detect_tiles=_env_int(
            "DEMO_DETECT_TILES",
            int(demo_cfg.get("detect_tiles", DEFAULT_IMPORT_DETECT_TILES))),
        demo_detect_threshold=_env_float(
            "DEMO_DETECT_THRESHOLD",
            float(demo_cfg.get("detect_threshold",
                               DEFAULT_IMPORT_DETECT_THRESHOLD))),
        demo_detect_max_items=_env_int(
            "DEMO_DETECT_MAX_ITEMS",
            int(demo_cfg.get("detect_max_items",
                             DEFAULT_IMPORT_DETECT_MAX_ITEMS))),
        slots=slots_cfg,
        vocabulary=vocab_cfg,
        slot_mode=slot_mode,
        grid_region=region,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        grid_show_cells=grid_show_cells,
        analytics_db=_env_str("ANALYTICS_DB",
                              "/data/aipc/data/shelf-ops/shelf.db"),
        bucket_seconds=_env_int("BUCKET_SECONDS", 600),
        retention_hours=_env_int("RETENTION_HOURS", 336),
        heatmap_min_popularity=_env_float("HEATMAP_MIN_POPULARITY", 0.1),
        alert_cooldown_seconds=_env_int("ALERT_COOLDOWN_SECONDS", 30),
        stockout_seconds=_env_int("STOCKOUT_SECONDS", 120),
        restock_cooldown_seconds=_env_int("RESTOCK_COOLDOWN_SECONDS", 60),
        platform_api_ws_scheme=_env_str("PLATFORM_API_WS_SCHEME", "wss"),
        platform_api_port=_env_int("PLATFORM_API_PORT", 443),
        platform_api_token=_env_str("PLATFORM_API_TOKEN", ""),
    )
    if cfg.clip_model not in MODEL_DEFS:
        raise ValueError(
            f"unknown CLIP_MODEL '{cfg.clip_model}'; "
            f"known: {sorted(MODEL_DEFS)}")
    if cfg.embedding_mode not in ("device", "file"):
        raise ValueError(
            f"EMBEDDING_MODE must be 'device' or 'file', "
            f"got '{cfg.embedding_mode}'")
    if cfg.detect_model not in MODEL_DEFS:
        raise ValueError(
            f"unknown DETECT_MODEL '{cfg.detect_model}'; "
            f"known: {sorted(MODEL_DEFS)}")
    if not (0.0 < cfg.detect_threshold < 1.0):
        raise ValueError(
            f"DETECT_THRESHOLD must be in (0,1), got {cfg.detect_threshold}")
    if not (0.0 < cfg.detect_nms_iou < 1.0):
        raise ValueError(
            f"DETECT_NMS_IOU must be in (0,1), got {cfg.detect_nms_iou}")
    if cfg.detect_tiles not in (1, 4):
        raise ValueError(
            f"DETECT_TILES must be 1 or 4, got {cfg.detect_tiles}")
    if cfg.detect_max_items <= 0:
        raise ValueError(
            f"DETECT_MAX_ITEMS must be > 0, got {cfg.detect_max_items}")
    if cfg.import_detect_tiles not in (1, 4):
        raise ValueError(
            f"IMPORT_DETECT_TILES must be 1 or 4, got {cfg.import_detect_tiles}")
    if not (0.0 < cfg.import_detect_threshold < 1.0):
        raise ValueError(
            f"IMPORT_DETECT_THRESHOLD must be in (0,1), "
            f"got {cfg.import_detect_threshold}")
    if cfg.import_detect_max_items <= 0:
        raise ValueError(
            f"IMPORT_DETECT_MAX_ITEMS must be > 0, "
            f"got {cfg.import_detect_max_items}")
    if cfg.import_max_image_mb <= 0:
        raise ValueError(
            f"IMPORT_MAX_IMAGE_MB must be > 0, got {cfg.import_max_image_mb}")
    if cfg.import_max_video_mb <= 0:
        raise ValueError(
            f"IMPORT_MAX_VIDEO_MB must be > 0, got {cfg.import_max_video_mb}")
    if cfg.import_video_max_frames <= 0:
        raise ValueError(
            f"IMPORT_VIDEO_MAX_FRAMES must be > 0, "
            f"got {cfg.import_video_max_frames}")
    if cfg.import_video_keyframes <= 0:
        raise ValueError(
            f"IMPORT_VIDEO_KEYFRAMES must be > 0, "
            f"got {cfg.import_video_keyframes}")
    if cfg.demo_speed <= 0:
        raise ValueError(f"DEMO_SPEED must be > 0, got {cfg.demo_speed}")
    if cfg.demo_stockout_seconds <= 0:
        raise ValueError(
            f"DEMO_STOCKOUT_SECONDS must be > 0, got {cfg.demo_stockout_seconds}")
    if cfg.demo_alert_cooldown_seconds < 0:
        raise ValueError(
            f"DEMO_ALERT_COOLDOWN_SECONDS must be >= 0, "
            f"got {cfg.demo_alert_cooldown_seconds}")
    if cfg.demo_detect_tiles not in (1, 4):
        raise ValueError(
            f"DEMO_DETECT_TILES must be 1 or 4, got {cfg.demo_detect_tiles}")
    if not (0.0 < cfg.demo_detect_threshold < 1.0):
        raise ValueError(
            f"DEMO_DETECT_THRESHOLD must be in (0,1), "
            f"got {cfg.demo_detect_threshold}")
    if cfg.demo_detect_max_items <= 0:
        raise ValueError(
            f"DEMO_DETECT_MAX_ITEMS must be > 0, got {cfg.demo_detect_max_items}")
    return cfg


def get_app_id() -> str:
    return Config.get_app_id()
