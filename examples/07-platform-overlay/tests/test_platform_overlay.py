"""07-platform-overlay — unit tests (offline, no SDK, no device)."""


def test_annotate_pushes_result_with_ttl(
        app_mod, overlay, make_object_, make_result_):
    result = make_result_(make_object_("person", 0.9),
                          make_object_("dog", 0.4))
    count = app_mod.annotate(overlay, 9, result)

    assert count == 2
    assert overlay.annotated == [("main", 2, app_mod.TTL_MS)]


def test_annotate_empty_result_still_sent(
        app_mod, overlay, make_result_):
    # An empty result publishes "no detections" — the clearing signal.
    count = app_mod.annotate(overlay, 10, make_result_())

    assert count == 0
    assert overlay.annotated == [("main", 0, app_mod.TTL_MS)]


def test_annotator_wiring(app_mod, wire):
    annotator = app_mod.Annotator()
    annotator.run()  # synchronous: the scripted results end the loop

    # Subscribed to the declared stream+model at fps<=2.
    assert wire.inference.subscribe_kwargs == {
        "stream": app_mod.STREAM_ID, "model": app_mod.MODEL_ID, "fps": 2}
    # Overlay enabled once, with the display knobs.
    assert wire.overlay.enabled_with == {
        "show_label": True, "show_confidence": True, "line_thickness": 2}
    # One annotate per result — the empty one included.
    assert wire.overlay.annotated == [
        (app_mod.STREAM_ID, 2, app_mod.TTL_MS),
        (app_mod.STREAM_ID, 0, app_mod.TTL_MS)]
    # Clean exit takes the boxes off and closes both clients.
    assert wire.overlay.disabled is True
    assert wire.overlay.closed is True
    assert wire.inference.closed is True
