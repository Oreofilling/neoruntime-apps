"""
Benchmark Engine - Serial vs Parallel inference comparison.

The engine runs the SAME set of models two ways:
  * serial   — one model inference at a time, per iteration
  * parallel — all models submitted in one batch, per iteration

Both modes do ``iterations * len(models)`` inferences, so the comparison is a
fair throughput / wall-clock trade-off of NPU batching vs sequential dispatch.

Hailo HEF input note: a model whose HEF reports shape ``[1, H, W, 3]`` may
accept either an NV12 plane (``H*W*1.5`` bytes) or an RGB plane (``H*W*3``
bytes) depending on how the HEF was compiled. Shape alone is ambiguous, so
:func:`resolve_model_input` probes both and keeps whichever the NPU accepts.
"""

import time
import threading
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from neoruntime_ipc_sdk import InferenceClient, BatchInferItem

# Raw protobuf is needed because the SDK convenience wrapper does not surface
# per-input tensor specs (it drops inputs/outputs). Importing at module load is
# part of the installed wheel contract.
import neoruntime_ipc_sdk.proto.inference_pb2 as inference_pb2  # noqa: E402


# ---------------------------------------------------------------------------
# Per-model input resolution
# ---------------------------------------------------------------------------

@dataclass
class ModelInput:
    """A correctly-sized zero buffer + the format the NPU accepted for it."""
    model_id: str
    buffer: np.ndarray
    format: str          # "nv12" | "rgb"
    height: int
    width: int


def _ensure_stub(infer_client: InferenceClient):
    """Return a connected raw gRPC stub (connect on demand)."""
    stub = getattr(infer_client, "stub", None)
    if stub is None:
        infer_client.connect()
        stub = getattr(infer_client, "stub", None)
    return stub


def get_input_hw(infer_client: InferenceClient, model_id: str,
                 timeout: float = 5.0) -> Optional[Tuple[int, int]]:
    """Return ``(height, width)`` of the model's first input, or ``None``.

    Uses the raw ``GetModelInfo`` RPC because the SDK wrapper drops inputs.
    Hailo HEF inputs are conventionally ``[1, H, W, C]`` (NHWC).
    """
    stub = _ensure_stub(infer_client)
    if stub is None:
        return None
    try:
        resp = stub.GetModelInfo(inference_pb2.ModelInfo(model_id=model_id),
                                 timeout=timeout)
    except Exception:
        return None
    if not resp.inputs:
        return None
    shape = [int(d) for d in resp.inputs[0].shape]
    if len(shape) >= 4:
        return shape[1], shape[2]
    if len(shape) == 3:
        return shape[0], shape[1]
    return None


def resolve_model_input(infer_client: InferenceClient, model_id: str,
                        probe_timeout_ms: int = 15000) -> ModelInput:
    """Probe a model with a zero buffer to find the input format it accepts.

    Tries NV12 then RGB. Raises ``ValueError`` if the model rejects both (e.g.
    it expects a different source or the HEF is incompatible at runtime).
    """
    hw = get_input_hw(infer_client, model_id)
    if hw is None:
        raise ValueError(f"{model_id}: input shape unavailable "
                         f"(GetModelInfo returned no inputs)")
    height, width = hw

    candidates = [
        (height * width * 3 // 2, "nv12"),
        (height * width * 3, "rgb"),
    ]
    last_err = None
    for nbytes, fmt in candidates:
        if nbytes <= 0:
            continue
        buf = np.zeros(nbytes, dtype=np.uint8)
        try:
            infer_client.infer(buf, model_id=model_id,
                               timeout_ms=probe_timeout_ms)
            return ModelInput(model_id, buf, fmt, height, width)
        except Exception as exc:  # probe failure — try the next candidate
            last_err = exc
            continue

    sizes = ", ".join(f"{fmt}={nbytes}B" for nbytes, fmt in candidates)
    raise ValueError(f"{model_id}: rejected both candidate inputs "
                     f"({sizes}); last error: {last_err}")


# ---------------------------------------------------------------------------
# Result DTO (field names match the frontend contract in templates/index.html)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Frontend field contract (templates/index.html displayComparison /
    displayHistory): serial_total_ms, parallel_total_ms, serial_avg_latency_ms,
    parallel_avg_latency_ms, serial_fps, parallel_fps, throughput_speedup,
    status, models, iterations, timestamp. ``inputs`` / ``dropped`` / ``error``
    are extra debugging context the frontend ignores.
    """
    mode: str
    models: List[str]
    iterations: int
    serial_total_ms: float = 0.0
    parallel_total_ms: float = 0.0
    serial_avg_latency_ms: float = 0.0
    parallel_avg_latency_ms: float = 0.0
    serial_fps: float = 0.0
    parallel_fps: float = 0.0
    throughput_speedup: float = 0.0
    npu_util_avg: float = 0.0
    timestamp: float = 0.0
    status: str = "running"
    # Per-item failures from the parallel batch phase (infer_batch does not
    # raise on partial failure). Surfaced so silent model failures are not
    # miscounted as successful throughput.
    parallel_item_failures: int = 0
    inputs: List[Dict] = field(default_factory=list)
    dropped: List[Dict] = field(default_factory=list)
    error: str = ""
    # Raw per-iteration timings (seconds), kept for future distributions.
    serial_times: List[float] = field(default_factory=list)
    parallel_times: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'mode': self.mode,
            'models': self.models,
            'iterations': self.iterations,
            'serial_total_ms': round(self.serial_total_ms, 2),
            'parallel_total_ms': round(self.parallel_total_ms, 2),
            'serial_avg_latency_ms': round(self.serial_avg_latency_ms, 2),
            'parallel_avg_latency_ms': round(self.parallel_avg_latency_ms, 2),
            'serial_fps': round(self.serial_fps, 2),
            'parallel_fps': round(self.parallel_fps, 2),
            'throughput_speedup': round(self.throughput_speedup, 2),
            'npu_util_avg': round(self.npu_util_avg, 1),
            'timestamp': self.timestamp,
            'status': self.status,
            'parallel_item_failures': self.parallel_item_failures,
            'inputs': self.inputs,
            'dropped': self.dropped,
            'error': self.error,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BenchmarkEngine:
    """Runs serial vs parallel inference benchmarks on a background thread."""

    MAX_HISTORY = 20

    def __init__(self, infer_client: InferenceClient, stats_collector):
        self._infer = infer_client
        self._stats = stats_collector
        self._running = False
        self._current_result: Optional[BenchmarkResult] = None
        self._history: List[BenchmarkResult] = []
        self._progress = 0
        self._total_steps = 0
        self._last_error = ""
        self._phase = ""  # 'resolving' | 'serial' | 'parallel' | ''
        self._state = "idle"  # idle|running|completed|failed|stopped
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> Dict:
        # Frontend contract (templates/index.html updateBenchmarkProgress)
        # reads `state` / `phase` / `total`. `state` is an explicit engine
        # state machine (idle/running/completed/failed/stopped) so a stop()
        # during a blocking infer is reflected immediately instead of
        # waiting for the run thread to commit a terminal result. The legacy
        # `running` / `progress` / `total_steps` / `current` keys are kept.
        with self._lock:
            return {
                'state': self._state,
                'phase': self._phase or ("running" if self._running else ""),
                'running': self._running,
                'progress': self._progress,
                'total': self._total_steps,
                'total_steps': self._total_steps,
                'iterations': (self._current_result.iterations
                               if self._current_result else 0),
                'last_error': self._last_error,
                'current': (self._current_result.to_dict()
                            if self._current_result else None),
            }

    def get_history(self) -> List[Dict]:
        with self._lock:
            return [r.to_dict() for r in self._history]

    def start_benchmark(self, model_ids: List[str], iterations: int = 50
                        ) -> Tuple[bool, str]:
        """Start a benchmark comparing serial vs parallel inference.

        The image argument from the old signature is gone — input buffers are
        resolved per model (NV12/RGB probe) inside the run, since each HEF may
        expect a different format/size.
        """
        # Check-then-set under the lock: waitress serves this app with
        # threads=4, so two concurrent POST /api/benchmark/start requests
        # could both pass the _running guard and spawn duplicate run
        # threads. Hold the lock across the guard AND the state reset so
        # only one request can transition idle -> running.
        with self._lock:
            if self._running:
                return False, "Benchmark already running"

            iterations = max(1, int(iterations or 50))
            self._running = True
            self._progress = 0
            self._total_steps = iterations * 2  # serial + parallel phases
            self._last_error = ""
            self._phase = "resolving"
            self._state = "running"

        thread = threading.Thread(
            target=self._run_benchmark,
            args=(list(model_ids), iterations),
            daemon=True,
            name="benchmark-run",
        )
        thread.start()
        return True, "Benchmark started"

    def stop(self):
        # Flip the run-loop flag and reflect 'stopped' in get_status
        # immediately. The in-flight gRPC call is not pre-empted (the SDK
        # exposes no context-cancellation path), but every phase loop checks
        # _running before each iteration and exits promptly; the run thread
        # then commits a 'stopped' result via _stopped(). Setting _state
        # here (not only in _stopped) means the UI sees 'stopped' right
        # away instead of waiting for the blocking call to return.
        with self._lock:
            was_running = self._running
            self._running = False
            if was_running:
                self._state = "stopped"

    def _set_phase(self, phase: str):
        with self._lock:
            self._phase = phase

    def _set_state(self, state: str):
        with self._lock:
            self._state = state

    def _stopped(self, result: BenchmarkResult):
        """Record a partial result when the user stops a run mid-flight."""
        result.status = "stopped"
        self._set_state("stopped")
        self._set_phase("")
        self._commit_result(result, record=True)

    # -- internals ---------------------------------------------------------

    def _commit_result(self, result: BenchmarkResult, record: bool = True):
        with self._lock:
            self._current_result = result
            if record:
                self._history.append(result)
                if len(self._history) > self.MAX_HISTORY:
                    self._history = self._history[-self.MAX_HISTORY:]

    def _fail(self, result: BenchmarkResult, message: str):
        """Mark a result failed, surface the error, and record it in history."""
        result.status = "failed"
        result.error = message
        with self._lock:
            self._last_error = message
            self._state = "failed"
        self._commit_result(result, record=True)
        print(f"[Benchmark] FAILED: {message}\n{traceback.format_exc()}")

    def _run_benchmark(self, model_ids: List[str], iterations: int):
        result = BenchmarkResult(
            mode='comparison',
            models=model_ids,
            iterations=iterations,
            timestamp=time.time(),
        )
        run_start_ts = result.timestamp
        self._set_phase("resolving")
        try:
            # Phase 0: resolve a correctly-sized input buffer per model.
            # Models that reject both formats are dropped (not fatal) so one
            # incompatible HEF cannot sink the whole comparison.
            resolved: Dict[str, ModelInput] = {}
            for mid in model_ids:
                if not self._running:
                    self._stopped(result)
                    return
                try:
                    mi = resolve_model_input(self._infer, mid)
                    resolved[mid] = mi
                    result.inputs.append({
                        'model_id': mid,
                        'format': mi.format,
                        'height': mi.height,
                        'width': mi.width,
                        'bytes': int(mi.buffer.nbytes),
                    })
                except Exception as exc:
                    result.dropped.append({'model_id': mid, 'error': str(exc)})

            if not resolved:
                dropped = "; ".join(
                    f"{d['model_id']}: {d['error']}" for d in result.dropped
                ) or "no models selected"
                self._fail(result, f"No usable models ({dropped})")
                return

            # Narrow the run to the models we could actually feed.
            usable = [mid for mid in model_ids if mid in resolved]
            result.models = usable
            self._commit_result(result, record=False)

            # Phase 1: serial — one model inference at a time per iteration.
            self._set_phase("serial")
            serial_iter_times, serial_call_times = self._run_serial(
                resolved, usable, iterations)
            if not self._running:
                self._stopped(result)
                return

            # Phase 2: parallel — all models in one batch per iteration.
            self._set_phase("parallel")
            parallel_iter_times, parallel_failures = self._run_parallel(
                resolved, usable, iterations)
            result.parallel_item_failures = parallel_failures
            if not self._running:
                self._stopped(result)
                return

            # Compute aggregate metrics. Both modes perform
            # iterations * n_models inferences, so FPS is comparable.
            n_models = len(usable)
            total_inferences = iterations * n_models
            serial_total_s = float(np.sum(serial_iter_times)) if serial_iter_times else 0.0
            parallel_total_s = float(np.sum(parallel_iter_times)) if parallel_iter_times else 0.0

            result.serial_times = serial_iter_times
            result.parallel_times = parallel_iter_times
            result.serial_total_ms = serial_total_s * 1000.0
            result.parallel_total_ms = parallel_total_s * 1000.0
            # Per-inference latency (ms): serial = mean single call;
            # parallel = wall-clock per batch divided by models in the batch.
            result.serial_avg_latency_ms = (
                float(np.mean(serial_call_times)) * 1000.0
                if serial_call_times else 0.0
            )
            result.parallel_avg_latency_ms = (
                float(np.mean(parallel_iter_times)) * 1000.0 / max(1, n_models)
                if parallel_iter_times else 0.0
            )
            if serial_total_s > 0:
                result.serial_fps = total_inferences / serial_total_s
            if parallel_total_s > 0:
                result.parallel_fps = total_inferences / parallel_total_s
            # Frontend label: "Throughput Speedup (Parallel / Serial)" — >1
            # means the batched path is faster, <1 means it is slower.
            if result.serial_fps > 0:
                result.throughput_speedup = result.parallel_fps / result.serial_fps

            # Window-average NPU utilization over the actual run window. A
            # single end-of-run snapshot captures whatever the 1s stats
            # thread last polled — a run ending during a quiet dispatch gap
            # would report ~0%. Averaging every sample collected while the
            # run executed gives a faithful mean.
            if self._stats:
                history = self._stats.get_history()
                window = [h for h in history
                          if h.get('timestamp', 0) >= run_start_ts]
                if window:
                    vals = [float(h.get('npu_utilization', 0.0) or 0.0)
                            for h in window]
                    result.npu_util_avg = float(np.mean(vals))
                else:
                    snap = self._stats.get_snapshot()
                    if snap:
                        result.npu_util_avg = float(
                            snap.get('npu_utilization', 0.0) or 0.0)

            result.status = "completed"
            self._set_state("completed")
            self._set_phase("")
            self._commit_result(result, record=True)

        except Exception as exc:  # defensive — phases already swallow per-model
            self._fail(result, str(exc))
        finally:
            self._running = False
            self._set_phase("")

    def _run_serial(self, resolved: Dict[str, ModelInput], model_ids: List[str],
                    iterations: int) -> Tuple[List[float], List[float]]:
        """Serial inference: one model at a time per iteration.

        Returns (per-iteration wall-clock times, per-call wall-clock times).
        """
        iter_times: List[float] = []
        call_times: List[float] = []
        for i in range(iterations):
            if not self._running:
                break
            t_iter = time.perf_counter()
            for mid in model_ids:
                mi = resolved[mid]
                t_call = time.perf_counter()
                self._infer.infer(mi.buffer, model_id=mid, timeout_ms=10000)
                call_times.append(time.perf_counter() - t_call)
            iter_times.append(time.perf_counter() - t_iter)
            with self._lock:
                self._progress = i + 1
        return iter_times, call_times

    def _run_parallel(self, resolved: Dict[str, ModelInput],
                      model_ids: List[str],
                      iterations: int) -> Tuple[List[float], int]:
        """Parallel inference: all models in one batch per iteration.

        Returns ``(per-iteration wall-clock times, total per-item failures)``.
        ``infer_batch`` returns one ``InferenceResult`` per item and does NOT
        raise on partial failure — a failed item carries a non-empty
        ``status_message``. We tally those so a silently-failing model is not
        counted as successful throughput, and break early if an entire batch
        fails (no item succeeded → continuing is pointless).
        """
        times: List[float] = []
        item_failures = 0
        items = [
            BatchInferItem(image=resolved[mid].buffer, model_id=mid,
                           timeout_ms=10000)
            for mid in model_ids
        ]
        n_items = len(items)
        for i in range(iterations):
            if not self._running:
                break
            t0 = time.perf_counter()
            results = self._infer.infer_batch(items, timeout_ms=20000)
            times.append(time.perf_counter() - t0)
            failed = sum(1 for r in results
                         if getattr(r, 'status_message', ''))
            item_failures += failed
            with self._lock:
                self._progress = iterations + i + 1
            if n_items and failed >= n_items:
                break
        return times, item_failures
