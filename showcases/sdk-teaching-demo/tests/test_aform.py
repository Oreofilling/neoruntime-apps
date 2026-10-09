"""A-form control: construction kwargs, guards, stop, consumer, and the
exit-code contract (engine death without a signal -> run() returns 1)."""

from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

from sdk_helpers import validate_partial


def obj(label, score):
    return SimpleNamespace(
        label=label, score=score,
        bbox=SimpleNamespace(x=0, y=0, width=1, height=1))


def test_start_kwargs_default(app, fake_sdk):
    app.aform.start()
    fake_sdk.StreamPipeline.assert_called_once_with(
        "third", "yolov8n_384_640",
        fps=10, min_score=0.4, labels=["person", "vehicle"],
        overlay_config=ANY, ttl_ms=1000)
    pipe = fake_sdk.StreamPipeline.return_value
    pipe.start.assert_called_once()
    app.aform.stop()  # join the (already dead) consumer thread


def test_start_kwargs_include_zone_when_on(app, fake_sdk):
    app.settings.apply(validate_partial({"zone_on": True}))
    app.aform.start()
    kwargs = fake_sdk.StreamPipeline.call_args.kwargs
    assert "polygons" in kwargs
    assert kwargs["polygons"][0]["label"] == "teaching zone"
    assert kwargs["ttl_ms"] == 1000
    app.aform.stop()


def test_double_start_guard(app):
    app.aform._pipe = object()
    try:
        app.aform.start()
        assert False, "double start must raise"
    except RuntimeError:
        pass


def test_stop_calls_pipe_stop(app):
    pipe = MagicMock()
    app.aform._pipe = pipe
    app.aform.stop()
    pipe.stop.assert_called_once()
    assert app.aform.running is False


def test_stop_when_not_running_is_noop(app):
    app.aform.stop()  # must not raise


def test_consumer_filters_and_publishes(app):
    result = SimpleNamespace(objects=[obj("person", 0.9),
                                      obj("face", 0.8)])
    app.aform._pipe = SimpleNamespace(
        results=lambda: iter([(5, result)]),
        stop=lambda: None,
        status=lambda: SimpleNamespace(running=True, results_seen=1),
    )
    app.aform._consume()  # synchronous: one (sequence, result) pair

    assert app.aform.consumed == 1
    # face is filtered out by the default labels (person, vehicle)
    assert app.state.snapshot()["latest"]["count"] == 1
    topic = app.events.publish.call_args.args[0]
    assert topic == "app/sdk-teaching-demo/detection"
    payload = app.events.publish.call_args.args[1]
    assert payload["source"] == "aform"
    assert payload["frame_sequence"] == 5


def test_run_returns_1_when_bform_dies_unsignalled(app):
    app.settings.apply(validate_partial({"bform_on": False}))
    app._start_bform = lambda: None  # keep the fake dead engine in place
    app.bform = SimpleNamespace(running=False, last_error="boom",
                                stop=lambda: None)
    app.settings.apply(validate_partial({"bform_on": True}))
    assert app.run() == 1


def test_run_returns_0_on_clean_stop(app):
    app._start_bform = lambda: None  # no real engine thread in this test
    app.bform = SimpleNamespace(running=True, last_error="",
                                stop=lambda: None)
    app.running = False  # "signal already received"
    assert app.run() == 0
    app.overlay.annotate.assert_called_once_with(
        "main", detections=[], polygons=[])
