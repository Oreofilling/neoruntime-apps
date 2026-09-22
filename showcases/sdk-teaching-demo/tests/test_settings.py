"""Settings validation, clamping and the thread-guarded store."""

import threading

import pytest


def test_defaults(helpers):
    settings = helpers.TeachingSettings()
    assert settings.bform_on is True
    assert settings.draw_on is True
    assert settings.min_score == 0.4
    assert tuple(settings.labels) == ("person", "vehicle")
    assert settings.overlay_path == "off"
    assert settings.zone_on is False
    assert settings.policy == "prefer_hardware"
    assert settings.aform_on is False


def test_min_score_clamped_both_ends(helpers):
    assert helpers.validate_partial({"min_score": 0.01})["min_score"] == 0.05
    assert helpers.validate_partial({"min_score": 5})["min_score"] == 0.9
    assert helpers.validate_partial({"min_score": "0.55"})["min_score"] == 0.55


def test_labels_validation(helpers):
    with pytest.raises(ValueError, match="non-empty"):
        helpers.validate_partial({"labels": []})
    with pytest.raises(ValueError, match="unknown labels"):
        helpers.validate_partial({"labels": ["cat"]})
    out = helpers.validate_partial({"labels": ["face"]})
    assert out["labels"] == ("face",)


def test_enum_fields_reject_unknown(helpers):
    with pytest.raises(ValueError, match="policy"):
        helpers.validate_partial({"policy": "fastest"})
    with pytest.raises(ValueError, match="overlay_path"):
        helpers.validate_partial({"overlay_path": "gpu"})
    assert helpers.validate_partial(
        {"policy": "hardware_only"})["policy"] == "hardware_only"


def test_unknown_key_and_non_object_rejected(helpers):
    with pytest.raises(ValueError, match="unknown setting"):
        helpers.validate_partial({"turbo": True})
    with pytest.raises(ValueError, match="JSON object"):
        helpers.validate_partial([("min_score", 0.5)])


def test_bools_coerced(helpers):
    out = helpers.validate_partial({"bform_on": 0, "zone_on": 1})
    assert out["bform_on"] is False
    assert out["zone_on"] is True


def test_store_apply_returns_full_snapshot(helpers):
    store = helpers.SettingsStore(helpers.TeachingSettings())
    snapshot = store.apply({"min_score": 0.6})
    assert snapshot["min_score"] == 0.6
    assert snapshot["policy"] == "prefer_hardware"  # untouched keys intact
    assert store.snapshot() == snapshot


def test_store_concurrent_smoke(helpers):
    store = helpers.SettingsStore(helpers.TeachingSettings())

    def hammer(index):
        for _ in range(200):
            store.apply({"min_score": 0.1 + (index % 5) * 0.1})
            store.snapshot()

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    snapshot = store.snapshot()
    assert 0.1 <= snapshot["min_score"] <= 0.5
    assert isinstance(snapshot["labels"], tuple)
