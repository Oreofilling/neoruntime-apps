"""Capstone pieces: buffer, frame wrapping, MJPEG framing, loop pass."""

import threading


def test_multipart_chunk_framing(app_mod):
    chunk = app_mod.multipart_chunk(b"ABC")

    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
    assert b"Content-Length: 3\r\n\r\nABC\r\n" in chunk
    assert chunk.endswith(b"ABC\r\n")


def test_as_frame_wraps_rgb(app_mod, wire):
    frame = app_mod.as_frame(type("A", (), {"ndim": 3,
                                            "shape": (1080, 1920, 3)})())
    assert frame.kwargs["format"] == "RGB"
    assert (frame.kwargs["width"], frame.kwargs["height"]) == (1920, 1080)


def test_as_frame_wraps_nv12_layout(app_mod, wire):
    # SDK NV12 layout: (h*3//2, w) -> 1080p is (1620, 1920)
    frame = app_mod.as_frame(type("A", (), {"ndim": 2,
                                            "shape": (1620, 1920)})())
    assert frame.kwargs["format"] == "NV12"
    assert (frame.kwargs["width"], frame.kwargs["height"]) == (1920, 1080)


def test_latest_jpeg_keeps_only_newest(app_mod):
    buffer = app_mod.LatestJpeg()
    buffer.update(b"old", sequence=1, objects=0)
    buffer.update(b"new", sequence=2, objects=3)

    jpeg, meta = buffer.snapshot()
    assert jpeg == b"new"
    assert meta == {"sequence": 2, "objects": 3}


def test_latest_jpeg_wait_unblocks_on_update(app_mod):
    buffer = app_mod.LatestJpeg()

    def updater():
        buffer.update(b"fresh", sequence=9)

    threading.Timer(0.05, updater).start()
    assert buffer.wait(2.0) == b"fresh"


def test_loop_pass_filters_draws_and_updates_buffer(app_mod, wire):
    buffer = app_mod.LatestJpeg()
    loop = app_mod.Loop(buffer)
    loop.run()  # synchronous: the fake subscribe yields 2 frames, ends

    # MIN_SCORE default 0.3 keeps "person" (0.9), drops "cat" (0.1)
    assert len(wire.drawn) == 2
    assert [obj.label for _array, objs in wire.drawn
            for obj in objs] == ["person", "person"]
    # every pipeline result was released (no leaked retained input)
    assert len(wire.released) == 2
    # buffer holds the LAST frame, encoded from the RGB wrap
    jpeg, meta = buffer.snapshot()
    assert meta["sequence"] == 11
    assert meta["objects"] == 1
    assert jpeg == b"jpeg-for-RGB"
    # frames wrapped with the stream geometry, not model input geometry
    assert wire.sdk_frames[-1].kwargs["width"] == 1920
