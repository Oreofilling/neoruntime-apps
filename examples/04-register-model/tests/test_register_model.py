"""Lifecycle order: register -> list -> infer -> unregister (or skip)."""


def test_full_lifecycle_when_not_registered(app_mod, wire, caplog):
    with caplog.at_level("INFO"):
        assert app_mod.main() == app_mod.EXIT_OK

    ops = [c[0] for c in wire.inference.calls]
    # register precedes the post-register list; unregister precedes close
    assert ops.index("register") < ops.index("unregister")
    assert ops[-1] == "close"
    assert ("register", "yolov8n-demo", str(wire.bundled), "detection") \
        in wire.inference.calls
    assert ("unregister", "yolov8n-demo") in wire.inference.calls
    assert wire.pipeline.run.called
    assert wire.media.closed is True
    assert "[4/4] unregistered" in caplog.text


def test_skips_register_and_unregister_when_id_exists(app_mod, wire,
                                                      caplog):
    wire.inference.models.append("yolov8n-demo")

    with caplog.at_level("INFO"):
        assert app_mod.main() == app_mod.EXIT_OK

    ops = [c[0] for c in wire.inference.calls]
    assert "register" not in ops
    assert "unregister" not in ops
    assert "already registered - reusing" in caplog.text


def test_missing_model_file_exits_three(app_mod, wire, monkeypatch, caplog):
    monkeypatch.setattr(app_mod, "CANDIDATE_PATHS", ["/nonexistent/x.hef"])

    with caplog.at_level("ERROR"):
        assert app_mod.main() == app_mod.EXIT_ENV
    assert "allow_register_model" in caplog.text
    # nothing was registered, so nothing must be unregistered
    ops = [c[0] for c in wire.inference.calls]
    assert "unregister" not in ops


def test_register_refusal_exits_three_without_unregister(app_mod, wire,
                                                         monkeypatch, caplog):
    def refuse(**_kwargs):
        raise PermissionError("register_model not allowed")

    monkeypatch.setattr(wire.inference, "register_model", refuse)

    with caplog.at_level("ERROR"):
        assert app_mod.main() == app_mod.EXIT_ENV
    ops = [c[0] for c in wire.inference.calls]
    assert "unregister" not in ops
    assert ops[-1] == "close"
