"""Overlay contract tests: ttl on every publish, zone lease refresh,
clear-with-empty-lists, and cleanup ordering."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from sdk_helpers import OVERLAY_STREAM, OVERLAY_TTL_MS, ZONE_POLYGON

NV12 = np.zeros((576, 640), dtype=np.uint8)  # 640x384 model input


def obj(label, score):
    return SimpleNamespace(
        label=label, score=score, class_id=0,
        bbox=SimpleNamespace(x=10, y=12, width=30, height=40))


def make_out():
    return SimpleNamespace(
        objects=[obj("person", 0.9), obj("vehicle", 0.2)],
        tensor=NV12,
        meta=SimpleNamespace(scale=(1.0, 1.0), origin=(0.0, 0.0)),
        result=SimpleNamespace(infer_time_us=250_000),
        latency_ms=13.0,
        release=MagicMock())


def make_frame():
    return SimpleNamespace(sequence=3,
                           to_jpeg_bytes=lambda quality=80: b"raw",
                           to_array=lambda: NV12)


def process(engine, app, **overrides):
    """One B-loop tick on an existing engine (zone state persists)."""
    settings = {**app.settings.snapshot(), **overrides}
    engine._process(make_frame(), make_out(), settings)


def test_platform_path_publishes_with_ttl(app, teaching):
    engine = teaching.BFormEngine(app)
    process(engine, app, overlay_path="platform")
    call = app.overlay.annotate_result.call_args
    assert call.args[0] == OVERLAY_STREAM
    assert call.kwargs["ttl_ms"] == OVERLAY_TTL_MS


def test_zone_publishes_then_lease_then_clears(app, teaching):
    engine = teaching.BFormEngine(app)  # one engine: lease state persists

    # toggle on: polygon goes out immediately, with ttl
    process(engine, app, zone_on=True)
    call = app.overlay.annotate.call_args
    assert call.args[0] == OVERLAY_STREAM
    assert call.kwargs["polygons"] == [ZONE_POLYGON]
    assert call.kwargs["ttl_ms"] == OVERLAY_TTL_MS
    app.overlay.annotate.reset_mock()

    # lease cadence not due -> no extra publish on the next tick
    process(engine, app, zone_on=True)
    app.overlay.annotate.assert_not_called()

    # toggle off: cleared with EMPTY lists (None would keep the layer)
    process(engine, app, zone_on=False)
    call = app.overlay.annotate.call_args
    assert call.args[0] == OVERLAY_STREAM
    assert call.kwargs["polygons"] == []


def test_render_filter_does_not_touch_stats(app, teaching):
    engine = teaching.BFormEngine(app)
    process(engine, app)  # defaults: min_score .4 keeps only person .9
    assert app.stats.totals["frames"] == 1
    assert app.stats.totals["objects"] == 2  # full model output counted
    assert app.state.snapshot()["latest"]["count"] == 1  # render filtered
    assert app.buffer.frame_id == 1


def test_draw_off_uses_passthrough_jpeg(app, teaching):
    engine = teaching.BFormEngine(app)
    process(engine, app, draw_on=False)
    # buffer holds the frame's own to_jpeg_bytes output, not a drawn one
    assert app.buffer._jpeg == b"raw"


def test_cleanup_order_and_empty_lists(teaching, fake_sdk, helpers,
                                       monkeypatch):
    monkeypatch.setattr(helpers, "_sdk", lambda: fake_sdk)
    monkeypatch.setattr(teaching, "_sdk", lambda: fake_sdk)
    instance = teaching.TeachingApp()
    order = []
    instance.overlay = MagicMock()
    instance.overlay.annotate.side_effect = \
        lambda *a, **k: order.append("overlay_clear")
    instance.overlay.disable.side_effect = lambda: order.append("disable")
    instance.events = MagicMock()
    instance.events.close.side_effect = lambda: order.append("events_close")
    instance.bform = MagicMock()
    instance.bform.stop.side_effect = lambda: order.append("bform_stop")
    instance.aform._pipe = MagicMock()
    instance.aform._pipe.stop.side_effect = lambda: order.append("aform_stop")
    instance._overlay_enabled = True

    instance.cleanup()

    assert order == ["aform_stop", "overlay_clear", "disable",
                     "bform_stop", "events_close"]
    clear_call = instance.overlay.annotate.call_args
    assert clear_call.kwargs["detections"] == []
    assert clear_call.kwargs["polygons"] == []
