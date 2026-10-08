"""Teaching payloads served at /api/snippets: the real code behind each
station, the error lessons, and the next-steps signpost.

Every snippet carries `anchors` — exact lines that MUST still exist in
the referenced source file. tests/test_snippets.py asserts them, so a
refactor that changes the real call sequence fails the suite instead of
silently rotting the lesson shown on the page.

No SDK imports here: this module is pure data + one assembly function,
loadable anywhere (page payload, tests, CI).
"""

SDK_PYPI_URL = "https://pypi.org/project/neoruntime-ipc-sdk/"
SDK_REPO_URL = "https://github.com/camthink-ai/neoruntime-sdks"
SDK_PYTHON_DIR_URL = "https://github.com/camthink-ai/neoruntime-sdks/tree/main/python"
APPS_REPO_URL = "https://github.com/camthink-ai/neoruntime-apps"
TEMPLATE_URL = APPS_REPO_URL + "/tree/main/templates/basic"
DEMO_DIR_URL = APPS_REPO_URL + "/tree/main/showcases/sdk-teaching-demo"

_STATION_DOCS = [
    ("neoruntime-ipc-sdk (PyPI)", SDK_PYPI_URL),
    ("SDK source (python/)", SDK_PYTHON_DIR_URL),
]

STATION_SNIPPETS = [
    {
        "id": "s1",
        "file": "teaching_app.py",
        "title": {"en": "The engine loop", "zh": "引擎主循环"},
        "note": {
            "en": "the whole live feed is this loop: subscribe with "
                  "keep_fd, run the pipeline inside the frame lease, "
                  "always release the retained input",
            "zh": "整个实时画面就是这段循环：keep_fd 订阅、在帧租约内跑"
                  "管线、始终释放保留的输入",
        },
        "code": """\
media = _sdk().FdMediaClient()
pipeline = self.app.pipeline
for frame in media.subscribe(STREAM_ID, keep_fd=True):
    settings = self.app.settings.snapshot()
    with frame:                      # keep-fd lease: max 3 held, 200 ms each
        out = pipeline.run(frame)
        try:
            self._process(frame, out, settings)
        finally:
            out.release()            # retained input: never leak DSP buffers
""",
        "anchors": [
            "for frame in self._media.subscribe(STREAM_ID, keep_fd=True):",
            "out = pipeline.run(frame)",
            "out.release()",
        ],
        "docs": _STATION_DOCS,
    },
    {
        "id": "s2",
        "file": "teaching_app.py",
        "title": {"en": "Platform overlay vs pixels you own", "zh": "平台 overlay vs 自己的像素"},
        "note": {
            "en": "annotate_result ships the result to the platform — zero "
                  "pixels leave your app; cleanup clears with EMPTY lists, "
                  "not None (None keeps the stale layer)",
            "zh": "annotate_result 把结果交给平台画——零像素离开你的 app；"
                  "清理要用空列表而不是 None（None 会保留旧图层）",
        },
        "code": """\
# draw on the platform's console stream — no pixels leave the app
if settings["overlay_path"] == "platform":
    self.app.overlay.annotate_result(
        OVERLAY_STREAM, out.result, ttl_ms=OVERLAY_TTL_MS)

# cleanup: empty lists clear the layers; detections=None would NOT
self.app.overlay.annotate(OVERLAY_STREAM, polygons=[])

""",
        "anchors": [
            "self.app.overlay.annotate_result(",
            "OVERLAY_STREAM, out.result, ttl_ms=OVERLAY_TTL_MS)",
            "self.app.overlay.annotate(OVERLAY_STREAM, polygons=[])",
        ],
        "docs": _STATION_DOCS,
    },
    {
        "id": "s3",
        "file": "teaching_app.py",
        "title": {"en": "Route policy + the refusal", "zh": "路由策略与拒绝"},
        "note": {
            "en": "one call flips every op between NPU/DSP/CPU; under "
                  "hardware_only a refused op raises — routing errors are "
                  "loud, never silently swallowed",
            "zh": "一次调用把所有算子在 NPU/DSP/CPU 间切换；hardware_only "
                  "下被拒算子直接抛错——路由错误是响亮的，绝不静默吞掉",
        },
        "code": """\
# settings side effect: one call, every op follows
_sdk().set_route_policy(validated["policy"])

# the refusal demo: routing an op that has no hardware leg
def refusal_demo(self):
    try:
        self.router.run(REFUSAL_OP, 1)
        return "unexpectedly succeeded — the op should not exist"
    except _sdk().HardwareUnavailable as exc:
        return f"HardwareUnavailable: {exc}"
""",
        "anchors": [
            "self.router.run(REFUSAL_OP, 1)",
        ],
        "docs": _STATION_DOCS,
    },
    {
        "id": "s4",
        "file": "teaching_app.py",
        "title": {"en": "A form: the platform runs the loop", "zh": "A 形态：平台跑循环"},
        "note": {
            "en": "StreamPipeline subscribes, infers and draws on the "
                  "platform side; your app only consumes results — and "
                  "stop() clears the platform drawings for you",
            "zh": "StreamPipeline 在平台侧订阅、推理并绘制；你的 app 只消费"
                  "结果——stop() 还会替你清掉平台侧的绘制",
        },
        "code": """\
pipe = _sdk().StreamPipeline(STREAM_ID, MODEL_ID, **kwargs)   # geometry rule:
                                                               # stream == model input
pipe.start()
for sequence, result in pipe.results():                  # consume only
    self.app.state.update(sequence, result)

pipe.stop()  # clears platform-side polygons AND detections
""",
        "anchors": [
            "pipe = _sdk().StreamPipeline(STREAM_ID, MODEL_ID, **kwargs)",
            "pipe.stop()  # clears platform-side polygons AND detections",
        ],
        "docs": _STATION_DOCS,
    },
    {
        "id": "s5",
        "file": "teaching_app.py",
        "title": {"en": "Cooldown-gated publish", "zh": "带冷却门的发布"},
        "note": {
            "en": "the gate decides, publish only fires when it allows — "
                  "downstream consumers get at most one event per cooldown "
                  "and only when something was detected",
            "zh": "先过门再发布——下游每个冷却窗最多收到一条事件，且只在"
                  "确有检出时收到",
        },
        "code": """\
def maybe_publish(self, source, sequence, objects):
    now = time.monotonic()
    if not self.event_gate.should_publish(now, bool(objects)):
        return
    payload = {"source": source}
    payload.update(payload_from_objects(sequence, objects))
    self.events.publish(f"app/{self.app_id}/detection", payload,
                        ttl_ms=10000)
    self.event_gate.record(now)
""",
        "anchors": [
            "if not self.event_gate.should_publish(now, bool(objects)):",
            "self.events.publish(f\"app/{self.app_id}/detection\", payload,",
        ],
        "docs": _STATION_DOCS,
    },
]

ERROR_LESSONS = [
    {
        "id": "dma-2811",
        "station": "s4",
        "title": {"en": "DMA input rejected: -2811", "zh": "DMA 输入被拒：-2811"},
        "cause": {
            "en": "platform-side feeding has no scaler: the stream geometry "
                  "must equal the runtime-registered model's input. Feeding "
                  "1080p `main` into a 640×384 model rejects every frame.",
            "zh": "平台侧喂帧没有缩放器：流几何必须等于运行时注册的模型"
                  "输入。把 1080p 的 `main` 喂给 640×384 模型会逐帧被拒。",
        },
        "contract": {
            "en": "use `third` (640×384) — or register a platform-managed "
                  "model (.bin package in spec.models) for `main`.",
            "zh": "用 `third`（640×384）——或为 `main` 注册平台托管的模型"
                  "（spec.models 里的 .bin 包）。",
        },
    },
    {
        "id": "routing-refusal",
        "station": "s3",
        "title": {"en": "HardwareUnavailable / KeyError from the router",
                  "zh": "路由器的 HardwareUnavailable / KeyError"},
        "cause": {
            "en": "hardware_only never silently falls back — an op without a "
                  "hardware leg raises instead of quietly running on CPU.",
            "zh": "hardware_only 绝不静默回落——没有硬件腿的算子直接抛错，"
                  "而不是悄悄在 CPU 上跑。",
        },
        "contract": {
            "en": "catch it and decide: prefer_hardware if a graceful "
                  "fallback is acceptable, or fix the op registration.",
            "zh": "捕获它再做决定：允许优雅回落就用 prefer_hardware，否则"
                  "修正算子注册。",
        },
    },
    {
        "id": "overlay-none-vs-empty",
        "station": "s2",
        "title": {"en": "detections=None keeps stale overlay layers",
                  "zh": "detections=None 会保留旧 overlay 层"},
        "cause": {
            "en": "None means “leave that layer alone”; only an EMPTY list "
                  "means “clear it”. Publishing None on cleanup leaves the "
                  "last boxes on screen.",
            "zh": "None 表示“该层不动”；只有空列表才表示“清空”。清理时传 "
                  "None 会让最后一帧的框留在画面上。",
        },
        "contract": {
            "en": "always clear with detections=[], polygons=[] — and carry "
                  "ttl_ms on every annotate so layers self-expire.",
            "zh": "清理永远用 detections=[], polygons=[]——且每次 annotate "
                  "都带 ttl_ms 让图层自过期。",
        },
    },
    {
        "id": "exit-code",
        "station": "s1",
        "title": {"en": "exit 1 vs exit 0 under restart_policy: on-failure",
                  "zh": "restart_policy: on-failure 下的退出码 1 与 0"},
        "cause": {
            "en": "the supervisor only revives the app when it exits "
                  "non-zero. An engine that dies without a signal must "
                  "return 1, or the page sits as a zombie.",
            "zh": "监管器只在非零退出时复活 app。引擎无信号死亡必须返回 "
                  "1，否则页面变成僵尸。",
        },
        "contract": {
            "en": "die-without-signal → exit 1 (revive me); clean signal "
                  "stop → exit 0 (stay down).",
            "zh": "无信号死亡 → 退出 1（复活我）；信号干净停 → 退出 0"
                  "（保持停止）。",
        },
    },
]

NEXT_STEPS = {
    "title": {"en": "From this demo to your own app",
              "zh": "从 demo 到你自己的 app"},
    "steps": [
        {"en": "1. copy templates/basic — the minimal app skeleton "
               "(app.yaml + Dockerfile + build.sh)",
         "zh": "1. 复制 templates/basic —— 最小 app 骨架（app.yaml + "
               "Dockerfile + build.sh）"},
        {"en": "2. write your engine against the same SDK surface these "
               "five stations drive",
         "zh": "2. 用这五个站操练的同一套 SDK 接口写你的引擎"},
        {"en": "3. ./build.sh arm64 → a .neoapp bundle",
         "zh": "3. ./build.sh arm64 → 得到 .neoapp 包"},
        {"en": "4. aipc-cli app install <id> <app.yaml> <image.tar>, then "
               "start it",
         "zh": "4. aipc-cli app install <id> <app.yaml> <image.tar>，然后"
               "启动"},
    ],
    "links": [
        ("templates/basic", TEMPLATE_URL),
        ("this demo's source", DEMO_DIR_URL),
        ("neoruntime-ipc-sdk (PyPI)", SDK_PYPI_URL),
    ],
}


def payload():
    """JSON-safe payload for GET /api/snippets."""
    return {
        "stations": STATION_SNIPPETS,
        "errors": ERROR_LESSONS,
        "next": NEXT_STEPS,
    }
