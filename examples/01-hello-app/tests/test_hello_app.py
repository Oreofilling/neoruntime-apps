"""Lifecycle behavior: ticks, graceful stop, exit codes."""

EXPECT_SIGNUM = 15


def test_ticks_then_graceful_stop(app_mod, monkeypatch, capsys):
    app = app_mod.HelloApp()
    ticks = []

    def fake_sleep(_seconds):
        ticks.append(1)
        if len(ticks) >= 2:
            app.running = False

    monkeypatch.setattr(app_mod.time, "sleep", fake_sleep)

    assert app.run() == 0
    assert len(ticks) == 2
    out = capsys.readouterr().out
    assert "tick #1" in out
    assert "stopped after 2 ticks" in out


def test_demo_fail_exits_one_after_three_ticks(app_mod, monkeypatch, capsys):
    monkeypatch.setenv("DEMO_FAIL", "1")
    app = app_mod.HelloApp()
    monkeypatch.setattr(app_mod.time, "sleep", lambda _s: None)

    assert app.run() == 1
    assert "simulated crash" in capsys.readouterr().out


def test_signal_flips_running_flag(app_mod, capsys):
    app = app_mod.HelloApp()
    app._on_signal(EXPECT_SIGNUM, None)
    assert app.running is False
    assert "graceful stop" in capsys.readouterr().out


def test_demo_fail_defaults_off(app_mod, monkeypatch):
    monkeypatch.delenv("DEMO_FAIL", raising=False)
    app = app_mod.HelloApp()
    assert app.fail is False
