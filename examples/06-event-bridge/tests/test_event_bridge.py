"""06-event-bridge — unit tests (offline, no SDK, no device)."""

import logging


def test_payload_filters_and_sorts(app_mod, make_object_, make_result_):
    result = make_result_(make_object_("person", 0.9),
                          make_object_("cat", 0.1),
                          make_object_("dog", 0.4))
    payload = app_mod.event_payload(7, result)
    # cat is under MIN_SCORE (0.3); labels come out sorted.
    assert payload == {"frame": 7, "objects": 2,
                       "labels": ["dog", "person"]}


def test_cooldown_gate(app_mod):
    cooldown = app_mod.Cooldown(2.0)
    assert cooldown.allow(0.0) is True   # first call always passes
    assert cooldown.allow(1.0) is False  # inside the window
    assert cooldown.allow(2.0) is True   # window elapsed


def test_maybe_publish_once_per_window(
        app_mod, events, make_object_, make_result_):
    cooldown = app_mod.Cooldown(2.0)
    result = make_result_(make_object_("person", 0.9))

    first = app_mod.maybe_publish(events, cooldown, 1, result, 0.0)
    second = app_mod.maybe_publish(events, cooldown, 2, result, 1.0)
    third = app_mod.maybe_publish(events, cooldown, 3, result, 2.0)

    assert first == {"frame": 1, "objects": 1, "labels": ["person"]}
    assert second is None          # suppressed by the cooldown
    assert third is not None       # window elapsed -> published
    assert len(events.published) == 2
    topics = [topic for topic, _payload in events.published]
    assert topics == [app_mod.PUBLISH_TOPIC, app_mod.PUBLISH_TOPIC]


def test_maybe_publish_skips_empty_without_burning_cooldown(
        app_mod, events, make_object_, make_result_):
    cooldown = app_mod.Cooldown(2.0)

    empty = app_mod.maybe_publish(events, cooldown, 1, make_result_(),
                                  0.0)
    # The very next frame, with objects, publishes IMMEDIATELY: the
    # empty frame did not consume the cooldown window.
    filled = app_mod.maybe_publish(
        events, cooldown, 2,
        make_result_(make_object_("person", 0.9)), 0.5)

    assert empty is None
    assert filled is not None
    assert len(events.published) == 1


def test_tap_loop_logs_all_incoming(app_mod, events, bus_event, caplog):
    incoming = [
        bus_event(payload={"frame": 1}),
        bus_event(topic="app/event-bridge/other", payload={"x": 2}),
        bus_event(payload={"frame": 3}),
    ]
    replays = type(events)(incoming)

    with caplog.at_level(logging.INFO, logger="event-bridge"):
        app_mod.tap_loop(replays)

    assert replays.subscribed_topic == "app/event-bridge/#"
    assert len(caplog.records) == 3
    assert all("bus:" in rec.message for rec in caplog.records)
