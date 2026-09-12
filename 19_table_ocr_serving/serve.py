#!/usr/bin/env python3
"""Serve one PaddleOCR-VL crop per HTTP request.

The HTTP process never imports Torch. One spawned NPU process owns the
ContinuousRecognizer and its compiled-graph caches for the server lifetime.
The implementation is the validated 910B2 table-serving path. This endpoint
accepts table crops only, with the bundled 60,416-row decode vocabulary by
default or the full checkpoint vocabulary with --full-decode-lm-head.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse


PROMPTS = {"table": "Table Recognition:"}


# Application startup and operational arguments


@dataclass(frozen=True) # frozen means params can't be changed after init
class ServeConfig:
    """Server options, shared by the HTTP process and its NPU worker."""

    model_path: Path
    graph_cache_directory: Path
    log_folder: Path

    # Address on which clients can reach the HTTP endpoint.
    host: str = "127.0.0.1"
    port: int = 8765

    request_timeout_s: float = 900.0  # timeout for individual ocr image request
    max_image_bytes: int = 64 * 1024 * 1024 # Largest encoded image accepted in one HTTP request (64 MiB).
    queue_capacity: int = 256 # Maximum jobs waiting in the HTTP-to-worker queue.


    device: str = "npu:0"

    decode_batch_size: int = 2 # Max number of decode slots at one time.

    # Use all 103,424 vocabulary rows instead of the smaller 60,416-row vocabulary.
    # The smaller vocabulary includes all chinese/han tokens, and all basic latex, math,
    # language, and syntax tokens. There is no generation difference or quality degradation
    # for OmniDocBench v1.6.
    # The 103,424 vocabulary is original full paddle vocabulary.
    # Using the smaller 60,416 vocabulary is 3-6% faster for average and P95 latency.
    full_decode_lm_head: bool = False

    # Run without graph compilation. Will be much slower. run_eagerly=True can be used for checking
    # if pipeline works at initial installation on server. Normally leave it disabled.
    run_eagerly: bool = False

    # basic: save minimal request metrics.
    # scheduling: also record scheduling details (preprocessing, prefill, decode time, tok/s metrics, etc.)
    # detailed: also measure decode time with NPU events (adds timing overhead, only use for debugging). Includes 'scheduling' logs.
    metrics_level: Literal["basic", "scheduling", "detailed"] = "scheduling"


def main() -> None:
    serve_config: ServeConfig = parse_args()

    # Two processes run the service:
    # A is this HTTP process. It receives images and sends responses to clients.
    # B is the inference process. It owns the model and runs OCR on the NPU,
    # with a background CPU thread preparing images while NPU work progresses.
    inference_connection = InferenceConnection(serve_config)
    inference_connection.start_inference_process()

    # Model setup has finished. Process A can now accept HTTP connections.
    http_server = HttpServer(serve_config, inference_connection)
    try:
        run_http_process(http_server)
    finally:
        try:
            # Let B finish queued OCR work before it exits.
            summary = inference_connection.stop_inference_process()
            if summary is not None:
                summary_path = serve_config.log_folder / "service_summary.json"
                _write_service_summary(
                    summary_path,
                    configuration=inference_connection.worker_runtime_info,
                    worker_pid=inference_connection.worker_pid,
                    summary=summary,
                )
                print(f"SERVICE_SUMMARY {summary_path}", flush=True)
        finally:
            http_server.server_close()


def parse_args() -> ServeConfig:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default=ServeConfig.host)
    parser.add_argument("--port", type=int, default=ServeConfig.port)
    parser.add_argument("--request-timeout-s", type=float, default=ServeConfig.request_timeout_s)
    parser.add_argument("--max-image-bytes", type=int, default=ServeConfig.max_image_bytes)
    parser.add_argument("--queue-capacity", type=int, default=ServeConfig.queue_capacity)
    parser.add_argument(
        "--run-eagerly", action="store_true", default=ServeConfig.run_eagerly,
        help="Run the same NPU stages without TorchAir compilation.",
    )
    parser.add_argument(
        "--model-path", type=Path, required=True,
        help="Local directory containing the model weights and tokenizer.",
    )
    parser.add_argument(
        "--device",
        default=ServeConfig.device,
        help="Logical NPU within the process's visible-device set.",
    )
    parser.add_argument(
        "--decode-batch-size", type=int, default=ServeConfig.decode_batch_size,
        help="Physical decode batch B, not a client concurrency limit; no batch-filling wait.",
    )
    parser.add_argument(
        "--full-decode-lm-head", action="store_true", default=ServeConfig.full_decode_lm_head,
        help="Use all 103,424 checkpoint vocabulary rows instead of the bundled 60,416-row decode head.",
    )
    parser.add_argument(
        "--metrics-level", choices=("basic", "scheduling", "detailed"),
        default=ServeConfig.metrics_level,
        help="Basic request metrics, additional scheduling details, or both plus NPU decode timings.",
    )
    parser.add_argument(
        "--graph-cache-directory",
        type=Path,
        required=True,
        help="Root directory for all compiled graphs, separated into stage subdirectories.",
    )
    parser.add_argument(
        "--log-folder",
        type=Path,
        required=True,
        help="Directory for service_summary.json at shutdown; continuous logging is not implemented yet.",
    )
    args = parser.parse_args()
    # Resolve caller-supplied paths once, relative to the launch working directory.
    args.model_path = args.model_path.expanduser().resolve()
    args.graph_cache_directory = args.graph_cache_directory.expanduser().resolve()
    args.log_folder = args.log_folder.expanduser().resolve()
    return ServeConfig(**vars(args))


# Process A: receive HTTP requests and return OCR results.


def run_http_process(http_server: HttpServer) -> None:
    """Keep accepting client connections in process A until shutdown is requested."""
    stop_requested = threading.Event()

    def stop_accepting_connections(signum: int, frame: Any) -> None:
        del signum, frame
        if not stop_requested.is_set():
            stop_requested.set()
            # Python's HTTP server requires shutdown() to be called from another
            # thread; calling it from serve_forever()'s thread would deadlock.
            threading.Thread(target=http_server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_accepting_connections)
    signal.signal(signal.SIGINT, stop_accepting_connections)
    serve_config = http_server.serve_config
    print(
        f"READY http://{serve_config.host}:{serve_config.port} "
        f"worker_pid={http_server.inference_connection.worker_pid}",
        flush=True,
    )
    http_server.serve_forever(poll_interval=0.25)


class HttpServer(ThreadingHTTPServer):
    """Python's HTTP server, with the settings and inference connection used by A."""

    # Each connection gets its own thread, so waiting for one OCR result does
    # not prevent other clients from submitting images.
    daemon_threads = True
    # Pending HTTP connections, separate from the queue of OCR jobs sent to B.
    request_queue_size = 256

    def __init__(
        self, serve_config: ServeConfig, inference_connection: InferenceConnection
    ) -> None:
        self.serve_config = serve_config
        self.inference_connection = inference_connection
        super().__init__((serve_config.host, serve_config.port), HttpRequestHandler)


class HttpRequestHandler(BaseHTTPRequestHandler):
    """Part of A: read one client's image, wait for OCR, and send the response."""

    server_version = "PaddleOCRCropAPI/1"

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/v1/ocr":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        query = parse_qs(parsed.query)
        crop_type = query.get("crop_type", [""])[0].strip().lower()
        if crop_type not in PROMPTS:
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"crop_type must be one of {sorted(PROMPTS)}"},
            )
            return
        public_request_id = query.get("request_id", [uuid.uuid4().hex])[0].strip()
        source_request_id = query.get("source_request_id", [public_request_id])[
            0
        ].strip()
        request_id = f"{public_request_id}:{uuid.uuid4().hex}"
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > self.server.serve_config.max_image_bytes:
            self._json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "invalid image body size"},
            )
            return
        image_bytes = self.rfile.read(length)
        submitted = time.perf_counter()
        try:
            message = self.inference_connection.recognize(
                {
                    "request_id": request_id,
                    "source_request_id": source_request_id,
                    "crop_type": crop_type,
                    "prompt": PROMPTS[crop_type],
                    "image_bytes": image_bytes,
                }
            )
        except queue.Full:
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE, {"error": "recognition queue is full"}
            )
            return
        except queue.Empty:
            self._json(HTTPStatus.GATEWAY_TIMEOUT, {"error": "recognition timed out"})
            return
        except Exception as exc:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        if not message["ok"]:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, message)
            return
        message["payload"]["request_id"] = public_request_id
        message["payload"]["http_wall_s"] = time.perf_counter() - submitted
        self._json(HTTPStatus.OK, message["payload"])

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._json(
                HTTPStatus.OK,
                {
                    "ok": self.inference_connection.inference_process.is_alive(),
                    "worker_pid": self.inference_connection.worker_pid,
                },
            )
        elif path == "/ready":
            ok = (
                self.inference_connection.startup_finished.is_set()
                and self.inference_connection.startup_error is None
                and self.inference_connection.inference_process.is_alive()
            )
            self._json(
                HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "ready": ok,
                    "worker_pid": self.inference_connection.worker_pid,
                    "configuration": self.inference_connection.worker_runtime_info,
                    "startup_error": self.inference_connection.startup_error,
                },
            )
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    @property
    def inference_connection(self) -> InferenceConnection:
        return self.server.inference_connection

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"HTTP {self.address_string()} {format % args}", flush=True)


# Still process A: send images to B and deliver each returning result to its caller.


class InferenceConnection:
    """A's connection to process B, not another process or another inference engine."""

    def __init__(self, serve_config: ServeConfig) -> None:
        self.serve_config = serve_config

        # Create process B and its shared queues using the spawn startup method.
        # B starts with a fresh Python interpreter when start_inference_process() runs.
        inference_process_context = mp.get_context("spawn")
        self.jobs = inference_process_context.Queue(maxsize=serve_config.queue_capacity)
        self.results = inference_process_context.Queue()
        self.inference_process = inference_process_context.Process(
            target=run_inference_process,
            args=(self.jobs, self.results, serve_config),
            name="crop-ocr-npu-worker",
        )

        # These fields are filled when B reports that setup succeeded or failed.
        # worker_runtime_info is B's setup report, not the input ServeConfig.
        self.startup_finished = threading.Event()
        self.startup_error: dict[str, Any] | None = None
        self.worker_runtime_info: dict[str, Any] | None = None
        self.worker_pid: int | None = None

        # Each waiting HTTP request has its own queue for one OCR result.
        # HTTP threads and the result-reading thread share this dictionary.
        self.result_queues_by_request_id: dict[str, queue.Queue[dict[str, Any]]] = {}
        self.result_queues_lock = threading.Lock()

        # B sends the final summary after receiving the end-of-input marker.
        self.service_summary: dict[str, Any] | None = None
        self.service_summary_ready = threading.Event()
        self.stop_reading_results = threading.Event()
        self.result_reader = threading.Thread(
            target=self._receive_worker_messages, name="ocr-result-dispatch", daemon=True
        )

    def start_inference_process(self) -> None:
        self.inference_process.start()
        self.result_reader.start()

        # Do not accept HTTP connections until B has finished model setup.
        # This uses request_timeout_s from ServeConfig (900 seconds by default).
        print(f"Waiting for NPU worker pid={self.inference_process.pid}", flush=True)
        self.startup_finished.wait(timeout=self.serve_config.request_timeout_s)
        if self.startup_error is not None:
            print(self.startup_error["traceback"], file=sys.stderr)
            raise RuntimeError(self.startup_error["error"])
        if not self.startup_finished.is_set() or not self.inference_process.is_alive():
            raise RuntimeError("NPU worker did not become ready")

    def recognize(self, job: dict[str, Any]) -> dict[str, Any]:
        """Send one image to B and wait for its OCR result in the calling HTTP thread."""
        request_result_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self.result_queues_lock:
            self.result_queues_by_request_id[job["request_id"]] = request_result_queue
        try:
            job["submitted_monotonic_s"] = time.perf_counter()
            self.jobs.put(job, timeout=1.0)
            # The lock is not held while waiting; other requests can proceed.
            return request_result_queue.get(timeout=self.serve_config.request_timeout_s)
        finally:
            with self.result_queues_lock:
                self.result_queues_by_request_id.pop(job["request_id"], None)

    def _receive_worker_messages(self) -> None:
        """Read messages from B; deliver each OCR result to the request waiting for it."""
        while not self.stop_reading_results.is_set():
            try:
                message = self.results.get(timeout=0.25)
            except queue.Empty:
                continue
            kind = message.get("kind")
            if kind == "ready":
                self.worker_runtime_info = message.get("configuration")
                self.worker_pid = message.get("worker_pid")
                self.startup_finished.set()
            elif kind == "startup_error":
                self.startup_error = message
                self.startup_finished.set()
            elif kind == "result":
                with self.result_queues_lock:
                    request_result_queue = self.result_queues_by_request_id.get(message["request_id"])
                if request_result_queue is not None:
                    request_result_queue.put(message)
            elif kind == "service_summary":
                self.service_summary = message["payload"]
                self.service_summary_ready.set()

    def stop_inference_process(self) -> dict[str, Any] | None:
        """Finish pending OCR work, stop B, and return its summary if it arrives in time."""
        summary = None
        try:
            # None means no more requests will follow. B finishes the requests
            # already received before returning its final summary.
            self.jobs.put(None, timeout=1.0)
            if not self.service_summary_ready.wait(timeout=self.serve_config.request_timeout_s):
                raise queue.Empty("timed out waiting for inference shutdown")
            assert self.service_summary is not None
            summary = self.service_summary
        except (queue.Empty, queue.Full):
            self.inference_process.terminate()
        self.inference_process.join(timeout=10.0)
        if self.inference_process.is_alive():
            self.inference_process.terminate()
            self.inference_process.join(timeout=5.0)
        self.stop_reading_results.set()
        return summary


# Process B: own the model and execute OCR. No HTTP connections live here.


def run_inference_process(
    jobs: Any,
    results: Any,
    config: ServeConfig,
) -> None:
    """Run inside B: receive images from A, perform OCR, and send results back."""

    try:
        from serving_runtime import ContinuousRecognizer
        from _support.serving.types import RecognitionRequest
        from crop_processing import convert_otsl_to_html

        recognizer = ContinuousRecognizer(
            model=str(config.model_path),
            batch_size=config.decode_batch_size,
            full_decode_lm_head=config.full_decode_lm_head,
            decode_device_timing=config.metrics_level == "detailed",
            torchair_cache_dir=config.graph_cache_directory / "decode",
            eager=config.run_eagerly,
            vision_torchair_cache_dir=config.graph_cache_directory / "vision_prefill",
            text_torchair_cache_dir=config.graph_cache_directory / "text_prefill",
            device=config.device,
        )
        setup_gc = _freeze_setup_gc()
        if setup_gc["enabled"]:
            print("EXP09_SETUP_GC " + json.dumps(setup_gc), flush=True)
        configuration = recognizer.configuration()
        configuration["setup_gc"] = setup_gc
        configuration["request_scheduling_metrics"] = config.metrics_level != "basic"
        configuration["metrics_level"] = config.metrics_level
        configuration["max_prefill_interruptions"] = None
        configuration["open_prefill_admission"] = "free_decode_slots_only_cpu_lookahead"
        results.put(
            {
                "kind": "ready",
                "configuration": configuration,
                "worker_pid": __import__("os").getpid(),
            }
        )

        request_jobs: dict[str, dict[str, Any]] = {}

        class QueueRecognitionSource:
            def __init__(self) -> None:
                self._closed = False

            @property
            def closed(self) -> bool:
                return self._closed

            def pull(self, *, block: bool) -> Any | None:
                while not self._closed:
                    try:
                        job = jobs.get() if block else jobs.get_nowait()
                    except queue.Empty:
                        return None
                    if job is None:
                        self._closed = True
                        return None
                    request_id = job["request_id"]
                    request_jobs[request_id] = job
                    return RecognitionRequest(
                        request_id=request_id,
                        # Same Image.open/convert recipe, executed later on the
                        # preparation worker instead of pausing active decode.
                        # Preparation failures use the existing error callback.
                        crop=job["image_bytes"],
                        prompt=job["prompt"],
                        submitted_at=job["submitted_monotonic_s"],
                    )
                return None

        def emit_result(recognition: Any) -> None:
            request_id = recognition.request_id
            job = request_jobs.pop(request_id)
            payload = asdict(recognition)
            payload["raw_text"] = payload["text"]
            payload["text"] = convert_otsl_to_html(payload["raw_text"]) or payload["raw_text"]
            payload.update(
                {
                    "crop_type": job["crop_type"],
                    "worker_wall_s": (
                        time.perf_counter() - job["submitted_monotonic_s"]
                    ),
                }
            )
            results.put(
                {
                    "kind": "result",
                    "request_id": request_id,
                    "ok": True,
                    "payload": payload,
                }
            )

        def emit_error(request_id: str, exc: BaseException) -> None:
            request_jobs.pop(request_id, None)
            results.put(
                {
                    "kind": "result",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                }
            )

        try:
            run_summary = recognizer.serve(
                QueueRecognitionSource(),
                schedule_id="http:open",
                emit_result=emit_result,
                on_request_error=emit_error,
                collect_scheduling_metrics=config.metrics_level != "basic",
            )
            results.put(
                {
                    "kind": "service_summary",
                    "payload": asdict(run_summary),
                }
            )
        except BaseException as exc:
            for request_id in list(request_jobs):
                results.put(
                    {
                        "kind": "result",
                        "request_id": request_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
            raise
    except BaseException as exc:
        results.put(
            {
                "kind": "startup_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


# Setup garbage collection and service summary


def _freeze_setup_gc() -> dict[str, Any]:
    """Exclude persistent setup objects from later cyclic-GC scans.

    Called only in the dedicated model worker, before accepting any request.
    This does not disable collection of newly allocated request objects.
    Frozen setup objects intentionally share the worker's lifetime.
    """
    import gc

    started = time.perf_counter()
    collected = gc.collect()
    gc.freeze()
    return {
        "enabled": True,
        "setup_collected": collected,
        "frozen_objects": gc.get_freeze_count(),
        "gc_remains_enabled": gc.isenabled(),
        "setup_wall_s": time.perf_counter() - started,
    }


def _write_service_summary(
    path: Path,
    *,
    configuration: dict[str, Any] | None,
    worker_pid: int | None,
    summary: dict[str, Any],
) -> None:
    payload = {
        "format": "paddleocr_crop_service_summary_v1",
        "worker_pid": worker_pid,
        "configuration": configuration,
        "summary": summary,
    }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
