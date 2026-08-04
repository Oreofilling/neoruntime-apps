"""
Web routes - Flask + SSE + HD preview for parallel benchmark dashboard.
"""

import json
import os
import time
import queue
import threading

from flask import Flask, Response, request, jsonify, render_template

# Raw protobuf: ListModels returns empty inputs/outputs, and the SDK
# ModelInfo wrapper drops them too. GetModelInfo (raw RPC) is the only path
# that returns populated input tensor specs.
import hailo_ipc_sdk.proto.inference_pb2 as inference_pb2

from .stats_collector import StatsCollector
from .benchmark import BenchmarkEngine


def _get_raw_stub(infer_client):
    """Return a connected raw gRPC stub (connect on demand)."""
    stub = getattr(infer_client, "stub", None)
    if stub is None:
        try:
            infer_client.connect()
        except Exception:
            return None
        stub = getattr(infer_client, "stub", None)
    return stub


# ---------------------------------------------------------------------------
# HD preview config (H.264 MSE via platform-api WebSocket)
# ---------------------------------------------------------------------------

def _build_hd_preview(req) -> dict:
    """Build H.264 HD preview config for the browser MSE player.

    platform-api enforces TLS: plain http://<host>:8080 301-redirects to
    https://<host>:443, including the H.264 WS endpoint, so the stream is only
    reachable over wss. A ws:// URL hits the 301 and the browser's WS upgrade
    fails (no frames -> black video). The device-served cert is self-signed
    (auto-generated, CN=ne503); the browser must trust it first by visiting
    https://<host>/api/v1/system/health and accepting the warning, otherwise
    wss fails silently (no "proceed" prompt for WebSocket).
    """
    host = req.host.split(':')[0]
    token = os.environ.get("PLATFORM_API_TOKEN", "")
    enabled = os.environ.get("HD_PREVIEW_ENABLED", "1") != "0"
    use_tls = os.environ.get("PLATFORM_API_TLS", "1") != "0"
    port = os.environ.get("PLATFORM_API_PORT", "443" if use_tls else "8080")
    ws_proto = "wss" if use_tls else "ws"
    # Omit the port for the TLS default (443) / empty so the URL stays clean.
    host_part = host if port in ("", "443") else f"{host}:{port}"
    ws_url = f"{ws_proto}://{host_part}/api/v1/h264/main"
    return {"enabled": enabled, "wsUrl": ws_url, "token": token}


# ---------------------------------------------------------------------------
# SSE hub
# ---------------------------------------------------------------------------

_SSE_CLIENTS: list = []
_SSE_LOCK = threading.Lock()


def sse_broadcast(data: dict):
    """Push a JSON event to all connected SSE clients."""
    payload = f"data: {json.dumps(data)}\n\n"
    dead = []
    with _SSE_LOCK:
        for q in _SSE_CLIENTS:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _SSE_CLIENTS.remove(q)


def _sse_subscribe():
    q = queue.Queue(maxsize=100)
    with _SSE_LOCK:
        _SSE_CLIENTS.append(q)
    return q


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_app(infer_client, stats_collector: StatsCollector,
               benchmark_engine: BenchmarkEngine):
    """Create Flask app with all routes."""

    app = Flask(__name__, template_folder='/app/templates')
    app.config['JSON_SORT_KEYS'] = False

    # Cache references
    _infer = infer_client
    _stats = stats_collector
    _bench = benchmark_engine

    # ---- Pages ----

    @app.route('/')
    def index():
        hd_preview = _build_hd_preview(request)
        return render_template('index.html', hd_preview=hd_preview)

    # ---- Stats API ----

    @app.route('/api/stats')
    def api_stats():
        snapshot = _stats.get_snapshot()
        return jsonify(snapshot or {})

    @app.route('/api/stats/history')
    def api_stats_history():
        count = request.args.get('count', 60, type=int)
        history = _stats.get_history(count)
        return jsonify(history)

    @app.route('/api/stats/stream')
    def api_stats_stream():
        """SSE stream of real-time stats."""
        q = _sse_subscribe()

        def generate():
            try:
                while True:
                    try:
                        data = q.get(timeout=30)
                        yield data
                    except queue.Empty:
                        yield ": keepalive\n\n"
            except GeneratorExit:
                with _SSE_LOCK:
                    if q in _SSE_CLIENTS:
                        _SSE_CLIENTS.remove(q)

        return Response(generate(), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache',
                                 'X-Accel-Buffering': 'no'})

    # ---- Models API ----

    @app.route('/api/models')
    def api_models():
        """List available models with populated input/output tensor specs.

        ListModels (and the SDK wrapper) return empty inputs/outputs, so we
        call the raw GetModelInfo RPC per model to get real shapes plus the
        ``estimated_memory`` field (renamed from ``memory_bytes`` in SDK 0.3.0).
        """
        try:
            models = _infer.list_models()
            stub = _get_raw_stub(_infer)
            result = []
            for m in (models or []):
                mid = m.model_id
                inputs, outputs = [], []
                estimated_tops = 0.0
                estimated_memory = 0
                # Best-effort enrichment; fall back to whatever ListModels gave.
                if stub is not None:
                    try:
                        info = stub.GetModelInfo(
                            inference_pb2.ModelInfo(model_id=mid), timeout=5)
                        inputs = [{'name': ts.name,
                                   'shape': [int(d) for d in ts.shape],
                                   'dtype': int(ts.dtype),
                                   'layout': ts.layout}
                                  for ts in info.inputs]
                        outputs = [{'name': ts.name,
                                    'shape': [int(d) for d in ts.shape]}
                                   for ts in info.outputs]
                        estimated_tops = float(info.estimated_tops or 0.0)
                        estimated_memory = int(info.estimated_memory or 0)
                    except Exception:
                        pass
                result.append({
                    'id': mid,
                    'name': mid,
                    'inputs': inputs,
                    'outputs': outputs,
                    'estimated_tops': estimated_tops,
                    'estimated_memory': estimated_memory,
                })
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ---- Benchmark API ----

    @app.route('/api/benchmark/start', methods=['POST'])
    def api_benchmark_start():
        body = request.get_json(force=True, silent=True) or {}
        model_ids = body.get('models', [])
        iterations = body.get('iterations', 50)

        if not model_ids:
            return jsonify({'error': 'No models selected'}), 400

        # Input buffers are resolved per model inside the engine (each HEF may
        # expect NV12 or RGB at its own HxW), so no image is passed here.
        ok, msg = _bench.start_benchmark(model_ids, iterations)
        if ok:
            return jsonify({'status': 'started', 'iterations': iterations, 'models': model_ids})
        return jsonify({'error': msg}), 409

    @app.route('/api/benchmark/stop', methods=['POST'])
    def api_benchmark_stop():
        _bench.stop()
        return jsonify({'status': 'stopped'})

    @app.route('/api/benchmark/status')
    def api_benchmark_status():
        return jsonify(_bench.get_status())

    @app.route('/api/benchmark/results')
    def api_benchmark_results():
        return jsonify(_bench.get_history())

    # ---- Health ----

    @app.route('/api/health')
    def api_health():
        return jsonify({'status': 'ok', 'timestamp': time.time()})

    return app
