"""
Parallel Benchmark - Main entry point.

Demonstrates NPU parallel inference capabilities by running serial vs parallel
benchmarks and displaying real-time performance metrics.
"""

import os
import signal
import sys
import time
import logging
import threading

from neoruntime_ipc_sdk import InferenceClient

from .stats_collector import StatsCollector
from .benchmark import BenchmarkEngine
from .web import create_app, sse_broadcast

logger = logging.getLogger(__name__)


class ParallelBenchmark:
    """Main application class for parallel benchmark demo."""

    def __init__(self):
        self._infer = None
        self._stats_collector = None
        self._benchmark_engine = None
        self._shutdown = threading.Event()

    def run(self):
        """Initialize SDK clients and start the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
        )
        logger.info("Starting Parallel Benchmark")

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Initialize inference client
        self._infer = InferenceClient()

        # Wait for SDK connectivity (readiness probe)
        logger.info("Waiting for SDK connectivity...")
        for i in range(60):
            if self._shutdown.is_set():
                return
            try:
                models = self._infer.list_models()
                if models is not None:
                    logger.info("SDK connected, %d models available", len(models))
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            logger.warning("SDK not available after 60s, continuing with limited functionality")

        # Create stats collector
        self._stats_collector = StatsCollector(self._infer, interval_sec=1.0)
        self._stats_collector.start()

        # Create benchmark engine
        self._benchmark_engine = BenchmarkEngine(self._infer, self._stats_collector)

        # Start stats broadcast thread
        self._start_stats_broadcast()

        # Create Flask app
        app = create_app(
            self._infer,
            self._stats_collector,
            self._benchmark_engine,
        )

        # Run with waitress
        port = int(os.environ.get('WEB_PORT', '8080'))
        logger.info("Starting web server on port %d", port)
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=4)

    def _start_stats_broadcast(self):
        """Broadcast stats via SSE every second."""
        def _loop():
            last_state = None
            while not self._shutdown.is_set():
                try:
                    snapshot = self._stats_collector.get_snapshot()
                    if snapshot:
                        sse_broadcast({'type': 'stats', 'data': snapshot})

                    # Broadcast benchmark status while a run is active, and
                    # once on the transition into a terminal state so the UI
                    # updates immediately instead of waiting for the 10s
                    # results poll. Idle is never broadcast. (Previously this
                    # checked a nonexistent 'state' field — get_status never
                    # returned one — so `None != 'idle'` was always True and
                    # the loop spammed benchmark status every second even at
                    # idle.)
                    bench_status = self._benchmark_engine.get_status()
                    state = bench_status.get('state')
                    if (state == 'running'
                            or (state not in ('idle', None)
                                and state != last_state)):
                        sse_broadcast({'type': 'benchmark', 'data': bench_status})
                    last_state = state
                except Exception:
                    pass
                self._shutdown.wait(1.0)

        t = threading.Thread(target=_loop, name='stats-broadcast', daemon=True)
        t.start()

    def _handle_signal(self, signum, frame):
        logger.info("Signal %d received, shutting down", signum)
        self._shutdown.set()
        if self._benchmark_engine:
            self._benchmark_engine.stop()
        if self._stats_collector:
            self._stats_collector.stop()
        sys.exit(0)


def main():
    ParallelBenchmark().run()


if __name__ == '__main__':
    main()
