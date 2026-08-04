"""
Stats Collector - Collects and maintains NPU/CPU/DSP/RAM performance metrics.
"""

import time
import threading
from collections import deque


class StatsCollector:
    """Collects system and per-model stats from ai-runtime."""

    def __init__(self, infer_client, window_size=120, interval_sec=1.0):
        self._infer = infer_client
        self._interval = interval_sec
        self._history = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._latest = None

    def start(self):
        """Start background collection thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def get_snapshot(self):
        """Return the latest stats snapshot."""
        with self._lock:
            return self._latest

    def get_history(self, count=None):
        """Return recent stats history (newest last)."""
        with self._lock:
            items = list(self._history)
        if count is not None:
            items = items[-count:]
        return items

    def _collect_loop(self):
        while self._running:
            try:
                snapshot = self._collect_once()
                with self._lock:
                    self._history.append(snapshot)
                    self._latest = snapshot
            except Exception as e:
                print(f"[StatsCollector] Error: {e}")
            time.sleep(self._interval)

    def _collect_once(self):
        ts = time.time()
        stats = self._infer.get_stats() or {}

        # Key names match the SSE contract consumed by the browser
        # (templates/index.html updateGauges reads *_utilization / fps /
        # avg_latency_ms). Renaming them silently zeroes every gauge.
        #
        # gRPC (grpc_service.cpp GetStats) stores device/cpu/dsp utilization as
        # 0-1 FRACTIONS (it divides HAL's 0-100 percent by 100.0f). The dashboard
        # gauge expects a 0-100 percentage (setGauge max=100), so multiply back
        # here — without this every utilization ring reads 0.15 as "0%".
        snapshot = {
            'timestamp': ts,
            'npu_utilization': stats.get('device_utilization', 0.0) * 100.0,
            'cpu_utilization': stats.get('cpu_utilization', 0.0) * 100.0,
            'dsp_utilization': stats.get('dsp_utilization', 0.0) * 100.0,
            'temp': stats.get('device_temperature', 0.0),
            'ram_total_kib': stats.get('ram_total_kib', 0),
            'ram_used_kib': stats.get('ram_used_kib', 0),
            'npu_mem_total': stats.get('total_memory_bytes', 0),
            'npu_mem_used': stats.get('used_memory_bytes', 0),
            'models': {},
            # Aggregate gauges: the dashboard has a single FPS + latency ring.
            # fps = total throughput across models; avg_latency_ms = mean of the
            # models that reported a non-zero latency this interval.
            'fps': 0.0,
            'avg_latency_ms': 0.0,
        }

        latencies = []
        for ms in stats.get('model_stats', []):
            model_id = ms.get('model_id', 'unknown')
            fps = float(ms.get('hw_fps', 0.0) or 0.0)
            latency_ms = float(ms.get('avg_latency_us', 0) or 0) / 1000.0
            snapshot['models'][model_id] = {
                'fps': fps,
                'latency_ms': latency_ms,
                'qps': ms.get('current_qps', 0.0),
                'queue_depth': ms.get('queue_depth', 0),
                'total_inferences': ms.get('total_inferences', 0),
                'total_errors': ms.get('total_errors', 0),
            }
            snapshot['fps'] += fps
            if latency_ms > 0:
                latencies.append(latency_ms)

        if latencies:
            snapshot['avg_latency_ms'] = sum(latencies) / len(latencies)

        return snapshot
