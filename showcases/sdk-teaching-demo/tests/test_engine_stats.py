"""Rolling-window math, cumulative totals and router health deltas."""

from sdk_helpers import WindowStats, health_deltas


def test_window_not_closed_early():
    stats = WindowStats(window_s=5.0)
    for _ in range(10):
        stats.note(objects=2, infer_ms=10.0, pipeline_ms=20.0, draw_ms=5.0)
    assert stats.poll(now=stats._start + 4.0) is None  # not due yet


def test_window_close_math():
    stats = WindowStats(window_s=5.0)
    for _ in range(10):
        stats.note(objects=2, infer_ms=10.0, pipeline_ms=20.0,
                   draw_ms=5.0, overlay_ms=4.0)
    window = stats.poll(now=stats._start + 5.0)
    assert window["fps"] == 2.0            # 10 frames / 5 s
    assert window["objects_per_frame"] == 2.0
    assert window["infer_ms"] == 10.0
    assert window["pipeline_ms"] == 20.0
    assert window["draw_ms"] == 5.0
    assert window["overlay_ms"] == 4.0     # mean over overlay calls
    assert window["frames"] == 10


def test_window_overlay_absent_is_none():
    stats = WindowStats(window_s=5.0)
    stats.note()
    window = stats.poll(now=stats._start + 5.0)
    assert window["overlay_ms"] is None


def test_poll_without_new_frames_keeps_last_window():
    stats = WindowStats(window_s=5.0)
    stats.note()
    first = stats.poll(now=stats._start + 5.0)
    assert stats.poll(now=stats._start + 12.0) is first


def test_totals_accumulate_across_windows():
    stats = WindowStats(window_s=5.0)
    for _ in range(4):
        stats.note(objects=3)
    stats.poll(now=stats._start + 5.0)
    for _ in range(6):
        stats.note(objects=1)
    stats.poll(now=stats._start + 10.0)
    assert stats.totals["frames"] == 10
    assert stats.totals["objects"] == 18


def test_health_deltas_math():
    prev = {"ops": {"jpeg_encode": {"backend": "dsp",
                                    "hardware_calls": 10,
                                    "software_calls": 2,
                                    "fallbacks": 1},
                    "gone_op": {"hardware_calls": 5}}}
    cur = {"ops": {"jpeg_encode": {"backend": "cpu",
                                   "hardware_calls": 12,
                                   "software_calls": 9,
                                   "fallbacks": 3}}}
    deltas = health_deltas(prev, cur)
    assert deltas["jpeg_encode"] == {"backend": "cpu", "hw": 2, "sw": 7,
                                     "fallbacks": 2}
    assert "gone_op" not in deltas


def test_health_deltas_prev_none():
    cur = {"ops": {"op": {"hardware_calls": 3, "software_calls": 0,
                          "fallbacks": 0}}}
    assert health_deltas(None, cur)["op"]["hw"] == 3
