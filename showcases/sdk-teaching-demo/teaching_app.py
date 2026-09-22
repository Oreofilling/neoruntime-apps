"""Engines + orchestrator for the teaching app.

BFormEngine  — subscribe/InferencePipeline/draw_detections: pixels flow
               through the app (Station 1 + the local half of Station 2).
AFormControl — StreamPipeline on demand: the platform subscribes,
               infers and draws; the app only consumes results (Station 4).
TeachingApp   — client wiring, settings side effects, refusal demo,
               status assembly, cleanup ordering, exit-code contract.

All SDK constructions go through sdk_helpers._sdk() so tests can inject
a fake SDK namespace after loading this module.
"""

import signal
import threading
import time

from sdk_helpers import (
    AFORM_STATUS_FIELDS,
    ALL_LABELS,
    EVENT_COOLDOWN_S,
    JPEG_QUALITY,
    MODEL_ID,
    OVERLAY_STREAM,
    OVERLAY_TTL_MS,
    REFUSAL_OP,
    STREAM_ID,
    ZONE_POLYGON,
    ZONE_REFRESH_S,
    CooldownGate,
    DetectionState,
    FrameBuffer,
    SettingsStore,
    TeachingSettings,
    WindowStats,
    _sdk,
    as_frame,
    draw_badge,
    ensure_model_registered,
    health_deltas,
    log,
    payload_from_objects,
    to_input_box,
    to_numpy,
    validate_partial,
)


# --------------------------------------------------------------------------
# B-form engine (pixels flow through this app)
# --------------------------------------------------------------------------

class BFormEngine(threading.Thread):
    """subscribe -> InferencePipeline -> draw -> MJPEG; optional platform
    overlay path and zone lease refresh ride the same loop.

    Owns its FdMediaClient: stop() closes the client, which unblocks the
    blocking subscribe iterator from another thread.
    """

    def __init__(self, app):
        super().__init__(daemon=True, name="bform-engine")
        self.app = app
        self.running = False
        self.last_error = ""
        self._stop = threading.Event()
        self._media = None
        self._zone_published = False
        self._zone_last_pub = 0.0

    def stop(self):
        self._stop.set()
        media = self._media
        if media is not None:
            try:
                media.close()
            except Exception as exc:
                log(f"media close during stop: {exc!r}")

    def run(self):
        self.running = True
        try:
            self._loop()
        except Exception as exc:
            if not self._stop.is_set():
                self.last_error = repr(exc)
                log(f"B-form engine died: {self.last_error}")
        finally:
            self.running = False

    def _loop(self):
        self._media = _sdk().FdMediaClient()
        pipeline = self.app.pipeline
        for frame in self._media.subscribe(STREAM_ID, keep_fd=True):
            if self._stop.is_set() or not self.app.running:
                break
            settings = self.app.settings.snapshot()
            with frame:  # keep-fd lease: max 3 held per stream, 200 ms each
                out = pipeline.run(frame)
                try:
                    self._process(frame, out, settings)
                finally:
                    out.release()  # retained input: never leak DSP buffers

    def _process(self, frame, out, settings):
        drawn = [
            obj for obj in out.objects
            if float(obj.score) >= settings["min_score"]
            and obj.label in settings["labels"]
        ]
        # NOTE: min_score/labels filter RENDERING and EVENTS only — the
        # model's full output is always counted in the stats below.

        draw_start = time.perf_counter()
        if settings["draw_on"]:
            base = (to_numpy(out.tensor) if out.tensor is not None
                    else frame.to_array())
            annotated = _sdk().draw_detections(
                base, [to_input_box(obj, out.meta) for obj in drawn])
            annotated = draw_badge(
                annotated, f"{settings['policy']} | {len(drawn)} obj")
            jpeg = as_frame(annotated).to_jpeg_bytes(quality=JPEG_QUALITY)
        else:
            # Raw passthrough — to_jpeg_bytes itself routes through the
            # accel router (hardware JPEG when available): free routing demo.
            jpeg = frame.to_jpeg_bytes(quality=JPEG_QUALITY)
        draw_ms = (time.perf_counter() - draw_start) * 1000.0

        overlay_ms = 0.0
        if settings["overlay_path"] == "platform":
            overlay_start = time.perf_counter()
            self.app.overlay.annotate_result(
                OVERLAY_STREAM, out.result, ttl_ms=OVERLAY_TTL_MS)
            overlay_ms = (time.perf_counter() - overlay_start) * 1000.0

        now = time.monotonic()
        if settings["zone_on"]:
            if now - self._zone_last_pub >= ZONE_REFRESH_S:
                self.app.overlay.annotate(
                    OVERLAY_STREAM, polygons=[ZONE_POLYGON],
                    ttl_ms=OVERLAY_TTL_MS)
                self._zone_last_pub = now
                self._zone_published = True
        elif self._zone_published:
            self.app.overlay.annotate(OVERLAY_STREAM, polygons=[])
            self._zone_published = False

        self.app.buffer.update(jpeg)
        self.app.stats.note(
            objects=len(out.objects),
            infer_ms=float(out.result.infer_time_us) / 1000.0,
            pipeline_ms=float(out.latency_ms),
            draw_ms=draw_ms,
            overlay_ms=overlay_ms,
        )
        self.app.state.update(frame.sequence, drawn)
        self.app.maybe_publish("bform", frame.sequence, drawn)


# --------------------------------------------------------------------------
# A-form control (platform does everything; the app consumes results)
# --------------------------------------------------------------------------

class AFormControl:
    """Start/stop a StreamPipeline on demand; consume results + events."""

    def __init__(self, app):
        self.app = app
        self._lock = threading.Lock()
        self._pipe = None
        self._consumer = None
        self.consumed = 0
        self.last_error = ""

    @property
    def running(self):
        return self._pipe is not None

    def start(self):
        with self._lock:
            if self._pipe is not None:
                raise RuntimeError("A-form pipeline already running")
            settings = self.app.settings.snapshot()
            kwargs = dict(
                fps=10,
                min_score=settings["min_score"],
                labels=list(settings["labels"]),
                overlay_config=_sdk().OverlayConfig(
                    show_label=True, show_confidence=True, line_thickness=2),
                ttl_ms=OVERLAY_TTL_MS,
            )
            if settings["zone_on"]:
                kwargs["polygons"] = [ZONE_POLYGON]
            pipe = _sdk().StreamPipeline(STREAM_ID, MODEL_ID, **kwargs)
            pipe.start()
            self._pipe = pipe
            self.consumed = 0
            self.last_error = ""
        self._consumer = threading.Thread(
            target=self._consume, daemon=True, name="aform-consumer")
        self._consumer.start()
        log(f"A-form StreamPipeline started on '{STREAM_ID}' "
            f"(fps<=10, min_score={settings['min_score']}, "
            f"labels={list(settings['labels'])})")

    def stop(self):
        with self._lock:
            pipe, self._pipe = self._pipe, None
        if pipe is None:
            return
        try:
            pipe.stop()  # clears platform-side polygons AND detections
        except Exception as exc:
            log(f"A-form stop error: {exc!r}")
        consumer = self._consumer
        if consumer is not None and consumer.is_alive():
            consumer.join(timeout=2.0)
        self._consumer = None
        log("A-form StreamPipeline stopped (overlay cleared)")

    def _consume(self):
        try:
            for sequence, result in self._pipe.results():
                with self._lock:
                    self.consumed += 1
                settings = self.app.settings.snapshot()
                objects = [
                    obj for obj in result.objects
                    if float(obj.score) >= settings["min_score"]
                    and obj.label in settings["labels"]
                ]
                self.app.state.update(sequence, objects)
                self.app.maybe_publish("aform", sequence, objects)
        except Exception as exc:
            self.last_error = repr(exc)
            log(f"A-form consumer ended: {self.last_error}")
        # results() ending on its own = worker death (e.g. 10 consecutive
        # inference failures); the cause is visible in status().last_error.

    def status(self):
        with self._lock:
            pipe = self._pipe
            consumed = self.consumed
            last_error = self.last_error
        if pipe is None:
            return {"running": False, "consumed": consumed,
                    "last_error": last_error}
        raw = pipe.status()
        data = {field: getattr(raw, field, None)
                for field in AFORM_STATUS_FIELDS}
        data["running"] = bool(data.get("running"))
        data["consumed"] = consumed
        if last_error:
            data["consumer_error"] = last_error
        return data


# --------------------------------------------------------------------------
# application orchestrator
# --------------------------------------------------------------------------

class TeachingApp:
    def __init__(self):
        self.app_id = _sdk().Config.get_app_id()
        self.running = True
        self.started_monotonic = time.monotonic()

        self.inference = None
        self.overlay = None
        self.events = None
        self.pipeline = None
        self.router = None
        self._overlay_enabled = False
        self._health_mark = None
        self._httpd = None
        self.exit_code = 0

        self.settings = SettingsStore(TeachingSettings())
        self.buffer = FrameBuffer()
        self.stats = WindowStats()
        self.state = DetectionState()
        self.event_gate = CooldownGate(EVENT_COOLDOWN_S)
        self.bform = BFormEngine(self)
        self.aform = AFormControl(self)

    # -- lifecycle ---------------------------------------------------------

    def _signal_handler(self, signum, _frame):
        log(f"received signal {signum}, shutting down")
        self.running = False

    def connect(self):
        self.inference = _sdk().InferenceClient()
        ensure_model_registered(self.inference)
        preprocessor = _sdk().Preprocessor.from_model(self.inference, MODEL_ID)
        # retain_input=True keeps the input tensor alive for draw_detections;
        # every frame is then handed back with out.release().
        self.pipeline = _sdk().InferencePipeline(
            client=self.inference, model_id=MODEL_ID,
            preprocessor=preprocessor, retain_input=True)
        self.overlay = _sdk().OverlayClient()
        self.events = _sdk().EventClient()
        self.router = _sdk().get_default_router()
        _sdk().set_route_policy(self.settings.snapshot()["policy"])
        log(f"connected: model={MODEL_ID} stream={STREAM_ID} "
            f"overlay_target={OVERLAY_STREAM}")

    def run(self, server_factory=None):
        """Run until a signal; server_factory(app) builds the HTTP server
        (main.py injects make_server; tests omit it to skip serving)."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        try:
            self.connect()
        except Exception as exc:
            log(f"startup failed: {exc!r}")
            return 1
        try:
            with _sdk().AppClient() as client:
                client.register_web_url("/")
        except Exception as exc:
            log(f"register_web_url failed (non-fatal): {exc!r}")

        if self.settings.snapshot()["bform_on"]:
            self._start_bform()
        if server_factory is not None:
            self._httpd = server_factory(self)
            threading.Thread(target=self._httpd.serve_forever,
                             daemon=True, name="http-server").start()
            log("interactive teaching page is serving "
                "(stations: live view, render paths, routing, a-form, events)")

        try:
            while self.running:
                time.sleep(0.5)
                # Exit-code contract: an engine dying without a signal
                # must exit 1, or restart_policy on-failure never revives.
                if (self.settings.snapshot()["bform_on"]
                        and not self.bform.running
                        and self.bform.last_error):
                    log(f"B-form engine died unexpectedly "
                        f"({self.bform.last_error}); exiting 1 for restart")
                    self.exit_code = 1
                    break
        finally:
            self.cleanup()
        return self.exit_code

    def cleanup(self):
        # Order matters: clear the platform overlay while the client is
        # alive, with EMPTY lists (None would keep the stale layers).
        try:
            self.aform.stop()
        except Exception as exc:
            log(f"aform stop: {exc!r}")
        if self.overlay is not None:
            try:
                self.overlay.annotate(
                    OVERLAY_STREAM, detections=[], polygons=[])
            except Exception as exc:
                log(f"overlay clear: {exc!r}")
            if self._overlay_enabled:
                try:
                    self.overlay.disable()
                except Exception as exc:
                    log(f"overlay disable: {exc!r}")
        try:
            self.bform.stop()
        except Exception as exc:
            log(f"bform stop: {exc!r}")
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
        for name in ("events", "inference"):
            client = getattr(self, name, None)
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
        log("cleanup complete")

    # -- engines -----------------------------------------------------------

    def _start_bform(self):
        self.bform = BFormEngine(self)
        self.bform.start()

    def _ensure_overlay(self):
        if not self._overlay_enabled:
            self.overlay.enable(show_label=True, show_confidence=True,
                                line_thickness=2)
            self._overlay_enabled = True

    def apply_settings(self, partial):
        """Validate + apply a partial update, run side effects."""
        try:
            validated = validate_partial(partial)
        except ValueError as exc:
            return False, str(exc)
        before = self.settings.snapshot()
        try:
            after = self.settings.apply(validated)
            if validated.get("policy") and validated["policy"] != before["policy"]:
                _sdk().set_route_policy(validated["policy"])
                self._health_mark = None  # deltas restart at the switch
            if after["overlay_path"] == "platform" or after["zone_on"]:
                self._ensure_overlay()
            if "aform_on" in validated:
                if after["aform_on"] and not before["aform_on"]:
                    self.aform.start()
                elif not after["aform_on"] and before["aform_on"]:
                    self.aform.stop()
            if "bform_on" in validated:
                if after["bform_on"] and not before["bform_on"]:
                    self._start_bform()
                elif not after["bform_on"] and before["bform_on"]:
                    self.bform.stop()
        except Exception as exc:
            return False, f"side effect failed: {exc!r}"
        return True, after

    def refusal_demo(self):
        """Route an unregistered op; the refusal IS the exhibit."""
        try:
            self.router.run(REFUSAL_OP, 1)
            return "unexpectedly succeeded — the op should not exist"
        except _sdk().HardwareUnavailable as exc:
            return f"HardwareUnavailable: {exc}"
        except KeyError as exc:
            return f"KeyError (unregistered op): {exc}"
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    # -- event + status ----------------------------------------------------

    def maybe_publish(self, source, sequence, objects):
        now = time.monotonic()
        if not self.event_gate.should_publish(now, bool(objects)):
            return
        payload = {"source": source}
        payload.update(payload_from_objects(sequence, objects))
        try:
            self.events.publish(f"app/{self.app_id}/detection", payload,
                                ttl_ms=10000)
        except Exception as exc:
            log(f"event publish failed (non-fatal): {exc!r}")
            return
        self.event_gate.record(now)
        self.state.note_event()

    def routing_status(self):
        policy = self.settings.snapshot()["policy"]
        try:
            health = self.router.health() or {}
        except Exception as exc:
            return {"policy": policy, "error": repr(exc)}
        if self._health_mark is None:
            self._health_mark = health
        return {
            "policy": health.get("policy") or policy,
            "ops": health.get("ops") or {},
            "deltas_since_mark": health_deltas(self._health_mark, health),
            "recent_degradations": health.get("recent_degradations", []),
        }

    def status(self):
        state = self.state.snapshot()
        self.stats.poll()
        return {
            "app": self.app_id,
            "uptime_s": round(time.monotonic() - self.started_monotonic, 1),
            "settings": self.settings.snapshot(),
            "model": {"id": MODEL_ID, "stream": STREAM_ID,
                      "all_labels": list(ALL_LABELS)},
            "bform": {
                "running": self.bform.running,
                "last_error": self.bform.last_error,
                "stats": self.stats.status(),
            },
            "overlay": {"target_stream": OVERLAY_STREAM,
                        "ttl_ms": OVERLAY_TTL_MS,
                        "enabled": self._overlay_enabled},
            "routing": self.routing_status(),
            "aform": self.aform.status(),
            "events": {
                "topic": f"app/{self.app_id}/detection",
                "published": state["events_published"],
                "cooldown_s": EVENT_COOLDOWN_S,
                "results_consumed": state["results_consumed"],
                "latest": state["latest"],
            },
        }
