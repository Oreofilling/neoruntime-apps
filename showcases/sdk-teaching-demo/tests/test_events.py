"""Event decision logic: the cooldown gate and payload shaping."""

from types import SimpleNamespace

from sdk_helpers import CooldownGate, payload_from_objects


def obj(label, score):
    return SimpleNamespace(
        label=label, score=score,
        bbox=SimpleNamespace(x=1.0, y=2.0, width=30.0, height=40.0))


def test_gate_first_hit_publishes():
    gate = CooldownGate(cooldown_s=5)
    assert gate.should_publish(now=100.0, has_objects=True) is True


def test_gate_blocked_without_objects():
    gate = CooldownGate(cooldown_s=5)
    gate.record(100.0)
    assert gate.should_publish(now=102.0, has_objects=False) is False


def test_gate_blocked_within_cooldown():
    gate = CooldownGate(cooldown_s=5)
    gate.record(100.0)
    assert gate.should_publish(now=104.9, has_objects=True) is False


def test_gate_open_after_cooldown():
    gate = CooldownGate(cooldown_s=5)
    gate.record(100.0)
    assert gate.should_publish(now=105.0, has_objects=True) is True


def test_payload_shape_and_rounding():
    payload = payload_from_objects(42, [obj("person", 0.45678)])
    assert payload["frame_sequence"] == 42
    assert payload["count"] == 1
    entry = payload["objects"][0]
    assert entry["label"] == "person"
    assert entry["score"] == 0.457  # round(..., 3)
    assert entry["bbox"] == [1.0, 2.0, 30.0, 40.0]


def test_payload_empty_objects():
    payload = payload_from_objects(7, [])
    assert payload["count"] == 0
    assert payload["objects"] == []
