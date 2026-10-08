"""First-result paths: report and exit 0, or exit 3 on failure."""


def test_first_result_reported_then_exit_zero(app_mod, inference_factory,
                                              fake_inference, object_maker,
                                              result_maker, caplog):
    result = result_maker(
        object_maker("person", 0.87),
        object_maker("dog", 0.05),
    )
    inference = fake_inference(result=result)
    inference_factory(inference)

    with caplog.at_level("INFO"):
        assert app_mod.main() == app_mod.EXIT_OK

    assert "frame 118: 1 object(s)" in caplog.text
    assert "person" in caplog.text and "dog" not in caplog.text
    assert inference.subscribed_with == ("main", "yolov8n", 2)
    assert inference.closed is True


def test_min_score_filters_rendering_only(app_mod, object_maker,
                                          result_maker, monkeypatch):
    monkeypatch.setattr(app_mod, "MIN_SCORE", 0.5)
    result = result_maker(object_maker("person", 0.9),
                          object_maker("cat", 0.2))

    assert app_mod.report(7, result) == 1
    assert app_mod.kept_objects(result)[0].label == "person"


def test_missing_model_warns_before_subscribing(app_mod, inference_factory,
                                                fake_inference, caplog):
    inference = fake_inference(models=["something-else"])
    inference_factory(inference)

    with caplog.at_level("WARNING"):
        assert app_mod.main() == app_mod.EXIT_OK
    assert "not registered yet" in caplog.text


def test_subscribe_failure_exits_three(app_mod, inference_factory,
                                       fake_inference, caplog):
    inference = fake_inference(error=RuntimeError("stream not found"))
    inference_factory(inference)

    with caplog.at_level("ERROR"):
        assert app_mod.main() == app_mod.EXIT_NO_RESULT
    assert "permissions" in caplog.text
    assert inference.closed is True
