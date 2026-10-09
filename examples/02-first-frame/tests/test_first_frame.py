"""Frame grab paths: success writes the file, failures exit 3."""


def test_success_writes_jpeg_and_closes(app_mod, media_factory, fake_frame,
                                        fake_media, monkeypatch, tmp_path):
    out = tmp_path / "first-frame.jpg"
    monkeypatch.setattr(app_mod, "OUT_PATH", str(out))
    media = fake_media(frame=fake_frame(b"jpeg-bytes"))
    media_factory(media)

    assert app_mod.main() == app_mod.EXIT_OK
    assert out.read_bytes() == b"jpeg-bytes"
    assert media.closed is True


def test_no_frame_exits_three(app_mod, media_factory, fake_media,
                              monkeypatch, tmp_path):
    out = tmp_path / "first-frame.jpg"
    monkeypatch.setattr(app_mod, "OUT_PATH", str(out))
    media = fake_media(frame=None)
    media_factory(media)

    assert app_mod.main() == app_mod.EXIT_NO_FRAME
    assert not out.exists()
    assert media.closed is True


def test_missing_stream_exits_three_and_lists(app_mod, media_factory,
                                              fake_media, monkeypatch,
                                              caplog):
    monkeypatch.setattr(app_mod, "OUT_PATH", "/tmp/never-written.jpg")
    media = fake_media(info=None, streams=["sub"])
    media_factory(media)

    with caplog.at_level("ERROR"):
        assert app_mod.main() == app_mod.EXIT_NO_FRAME
    assert "sub" in caplog.text


def test_grab_exception_is_environment_failure(app_mod, media_factory,
                                                fake_media, monkeypatch,
                                                tmp_path):
    monkeypatch.setattr(app_mod, "OUT_PATH", str(tmp_path / "x.jpg"))

    class ExplodingMedia(fake_media):
        def get_frame(self, *_args, **_kwargs):
            raise ConnectionError("socket dead")

    media = ExplodingMedia()
    media_factory(media)

    assert app_mod.main() == app_mod.EXIT_NO_FRAME
    assert media.closed is True
