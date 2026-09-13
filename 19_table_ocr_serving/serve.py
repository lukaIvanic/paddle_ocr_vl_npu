#!/usr/bin/env python3
"""HTTP service that runs PaddleOCR-VL on one table crop per request.

Two processes: this one answers HTTP, and a child process owns the model and
the NPU. They talk through two queues, described on InferenceServer below.
The HTTP process never imports Torch.

Endpoints: POST /v1/ocr?crop_type=table with the encoded image as the body,
GET /health, and GET /ready.

Table crops decode with the bundled 60,416-row vocabulary by default, or with
the full checkpoint vocabulary when --full-decode-lm-head is given.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
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

from _support.serving.types import RecognitionRequest


# The prompt the model receives for each accepted crop_type. Supporting another
# crop type (text, formula) starts with a new line here.
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

    # Timeout for one OCR image request. The same value bounds the wait for the
    # model to load at startup and for the summary at shutdown.
    request_timeout_s: float = 900.0
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

    # main() owns both halves. HTTP runs here; inference runs in a child process.
    inference_server = InferenceServer(serve_config)
    http_server = HttpServer(serve_config, inference_server)
    try:
        inference_server.start()  # returns once the model is loaded, or raises
        http_server.run()  # returns on Ctrl+C or SIGTERM
    finally:
        try:
            http_server.close()  # finish the requests already accepted
        finally:
            inference_server.close()  # then stop the child and save its summary


# HTTP service: receive images and return OCR results.


class HttpServer(ThreadingHTTPServer):
    """Accept client connections; inference is a separate service supplied by main()."""

    # Python's ThreadingHTTPServer gives each connection its own thread, so
    # waiting for one OCR result does not block other clients.
    # Non-daemon threads: close() waits for accepted requests to finish, and
    # main() keeps inference running until they have.
    daemon_threads = False
    # TCP connections waiting to be accepted, not a limit on outstanding OCR images.
    request_queue_size = 256

    def __init__(
        self, serve_config: ServeConfig, inference_server: InferenceServer
    ) -> None:
        self.serve_config = serve_config
        self.inference_server = inference_server
        self.stop_requested = threading.Event()
        # Bind the port only in run(), after the model is loaded. A bound but
        # idle port would make clients hang during the multi-minute startup.
        super().__init__(
            (serve_config.host, serve_config.port), HttpRequestHandler,
            bind_and_activate=False,
        )

    def run(self) -> None:
        """Listen and accept HTTP requests until Ctrl+C or a termination signal."""
        self.server_bind()
        self.server_activate()
        signal.signal(signal.SIGTERM, self._stop_accepting_connections)
        signal.signal(signal.SIGINT, self._stop_accepting_connections)
        print(
            f"READY http://{self.serve_config.host}:{self.serve_config.port} "
            f"worker_pid={self.inference_server.worker_pid}",
            flush=True,
        )
        self.serve_forever(poll_interval=0.25)

    def _stop_accepting_connections(self, signum: int, frame: Any) -> None:
        del signum, frame
        if not self.stop_requested.is_set():
            self.stop_requested.set()
            # Python requires shutdown() to run outside serve_forever()'s thread,
            # otherwise that thread would wait for itself and deadlock.
            threading.Thread(target=self.shutdown, daemon=True).start()

    def close(self) -> None:
        """Finish accepted HTTP requests and close sockets; do not stop inference."""
        self.server_close()


class HttpRequestHandler(BaseHTTPRequestHandler):
    """Read one client's image, wait for OCR, and send the response."""

    server_version = "PaddleOCRCropAPI/1"

    @property
    def inference_server(self) -> InferenceServer:
        return self.server.inference_server

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

        # Clients may reuse a request_id; the internal id stays unique per request.
        client_request_id = query.get("request_id", [uuid.uuid4().hex])[0].strip()
        request_id = f"{client_request_id}:{uuid.uuid4().hex}"
        submitted = time.perf_counter()
        try:
            reply = self.inference_server.recognize(request_id, crop_type, image_bytes)
        except InferenceQueueFull:
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE, {"error": "recognition queue is full"}
            )
            return
        except InferenceTimeout:
            self._json(HTTPStatus.GATEWAY_TIMEOUT, {"error": "recognition timed out"})
            return
        except Exception as exc:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return

        if not reply["ok"]:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, reply)
            return
        payload = reply["payload"]
        payload["request_id"] = client_request_id
        payload["http_wall_s"] = time.perf_counter() - submitted
        self._json(HTTPStatus.OK, payload)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._json(
                HTTPStatus.OK,
                {
                    "ok": self.inference_server.is_alive,
                    "worker_pid": self.inference_server.worker_pid,
                },
            )
        elif path == "/ready":
            ready = self.inference_server.is_ready
            self._json(
                HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "ready": ready,
                    "worker_pid": self.inference_server.worker_pid,
                    "configuration": self.inference_server.worker_runtime_info,
                    "startup_error": self.inference_server.startup_error,
                },
            )
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

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


# Inference service: own the child process and expose recognize() to HTTP callers.


class InferenceQueueFull(Exception):
    """The inference worker already has queue_capacity images waiting."""


class InferenceTimeout(Exception):
    """No result arrived within request_timeout_s."""


class InferenceServer:
    """The HTTP process's handle on the inference child process.

    Messages to the child, one per image:
        {"request_id", "crop_type", "image_bytes", "submitted_monotonic_s"}
        None                         no more images; finish and send the summary

    Messages from the child, each a dict with a "kind":
        "ready"            {"configuration", "worker_pid"}    model is loaded
        "startup_error"    {"error", "traceback"}             model failed to load
        "result"           {"request_id", "ok": True, "payload"}
                           {"request_id", "ok": False, "error", "traceback"}
        "service_summary"  {"payload"}                        sent last, after None
    """

    def __init__(self, serve_config: ServeConfig) -> None:
        self.serve_config = serve_config

        # The model runs in a child process with a fresh Python interpreter.
        # These queues carry images to it and results back to this service.
        inference_process_context = mp.get_context("spawn")
        self.jobs = inference_process_context.Queue(maxsize=serve_config.queue_capacity)
        self.results = inference_process_context.Queue()
        self.inference_process = inference_process_context.Process(
            target=run_inference_process,
            args=(self.jobs, self.results, serve_config),
            name="paddleocr-inference-worker",
        )

        # These fields are filled when the inference worker reports setup completion.
        # worker_runtime_info is the worker's setup report, not the input ServeConfig.
        self.startup_finished = threading.Event()
        self.startup_error: dict[str, Any] | None = None
        self.worker_runtime_info: dict[str, Any] | None = None
        self.worker_pid: int | None = None

        # Each waiting HTTP request has its own queue for one OCR result.
        # HTTP threads and the result-reading thread share this dictionary.
        self.result_queues_by_request_id: dict[str, queue.Queue[dict[str, Any]]] = {}
        self.result_queues_lock = threading.Lock()

        # The worker sends its final summary after the end-of-input marker.
        self.service_summary: dict[str, Any] | None = None
        self.service_summary_ready = threading.Event()
        self.stop_reading_results = threading.Event()
        self.result_reader = threading.Thread(
            target=self._receive_results_and_status, name="ocr-result-dispatch", daemon=True
        )

    def start(self) -> None:
        """Start the child process and wait until its model is loaded."""
        self.inference_process.start()
        self.result_reader.start()

        print(f"Waiting for inference worker pid={self.inference_process.pid}", flush=True)
        self.startup_finished.wait(timeout=self.serve_config.request_timeout_s)
        if self.startup_error is not None:
            print(self.startup_error["traceback"], file=sys.stderr)
            raise RuntimeError(self.startup_error["error"])
        if not self.is_ready:
            raise RuntimeError("inference worker did not become ready")

    def recognize(self, request_id: str, crop_type: str, image_bytes: bytes) -> dict[str, Any]:
        """Send one image to the inference worker and wait for its reply."""
        reply_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self.result_queues_lock:
            self.result_queues_by_request_id[request_id] = reply_queue
        try:
            job = {
                "request_id": request_id,
                "crop_type": crop_type,
                "image_bytes": image_bytes,
                "submitted_monotonic_s": time.perf_counter(),
            }
            try:
                self.jobs.put(job, timeout=1.0)
            except queue.Full:
                raise InferenceQueueFull() from None
            try:
                # The lock is not held while waiting; other requests can proceed.
                return reply_queue.get(timeout=self.serve_config.request_timeout_s)
            except queue.Empty:
                raise InferenceTimeout() from None
        finally:
            with self.result_queues_lock:
                self.result_queues_by_request_id.pop(request_id, None)

    @property
    def is_alive(self) -> bool:
        return self.inference_process.is_alive()

    @property
    def is_ready(self) -> bool:
        """True once the model is loaded and the child process is still running."""
        return (
            self.startup_finished.is_set()
            and self.startup_error is None
            and self.is_alive
        )

    def close(self) -> dict[str, Any] | None:
        """Stop the child process and save its final summary when there is one."""
        if self.inference_process.pid is None:
            return None  # start() never created the process.
        summary = self._ask_worker_to_finish()
        if summary is None:
            self.inference_process.terminate()
        self.inference_process.join(timeout=10.0)
        if self.inference_process.is_alive():
            self.inference_process.terminate()
            self.inference_process.join(timeout=5.0)
        self.stop_reading_results.set()
        if self.result_reader.ident is not None:
            self.result_reader.join(timeout=1.0)
        if summary is not None:
            summary_path = self.serve_config.log_folder / "service_summary.json"
            _write_service_summary(
                summary_path,
                configuration=self.worker_runtime_info,
                worker_pid=self.worker_pid,
                summary=summary,
            )
            print(f"SERVICE_SUMMARY {summary_path}", flush=True)
        return summary

    def _ask_worker_to_finish(self) -> dict[str, Any] | None:
        """Send the end-of-input marker and wait for the summary; None if the worker cannot answer."""
        if self.startup_error is not None or not self.is_alive:
            return None
        try:
            # None means no more images will follow. The worker finishes the
            # requests already received, then sends its summary.
            self.jobs.put(None, timeout=1.0)
        except queue.Full:
            return None
        if not self.service_summary_ready.wait(timeout=self.serve_config.request_timeout_s):
            return None
        return self.service_summary

    def _receive_results_and_status(self) -> None:
        """Receive inference status and deliver each OCR result to its waiting request."""
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
                    reply_queue = self.result_queues_by_request_id.get(message["request_id"])
                if reply_queue is not None:
                    reply_queue.put(message)
            elif kind == "service_summary":
                self.service_summary = message["payload"]
                self.service_summary_ready.set()


# Inference child process: own the model and execute OCR.


def run_inference_process(jobs: Any, results: Any, serve_config: ServeConfig) -> None:
    """Entrypoint of the child process: build the worker and run it until closed."""
    try:
        InferenceWorker(jobs, results, serve_config).run()
    except BaseException as exc:
        results.put({"kind": "startup_error", **_describe_failure(exc)})


class InferenceWorker:
    """Load the model, then OCR every image the HTTP process sends until told to stop."""

    def __init__(self, jobs: Any, results: Any, serve_config: ServeConfig) -> None:
        self.jobs = jobs
        self.results = results
        self.serve_config = serve_config
        self.jobs_in_progress: dict[str, dict[str, Any]] = {}
        self._closed = False

    def run(self) -> None:
        recognizer = self._load_model_and_report_ready()
        self._serve_until_closed(recognizer)

    def _load_model_and_report_ready(self) -> Any:
        # Only the child process imports Torch and the model code.
        from serving_runtime import ContinuousRecognizer

        config = self.serve_config
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
        configuration = recognizer.configuration()
        configuration["setup_gc"] = _freeze_setup_gc()
        configuration["request_scheduling_metrics"] = config.metrics_level != "basic"
        configuration["metrics_level"] = config.metrics_level
        self.results.put(
            {"kind": "ready", "configuration": configuration, "worker_pid": os.getpid()}
        )
        return recognizer

    def _serve_until_closed(self, recognizer: Any) -> None:
        try:
            # The recognizer calls pull() and closed for input, and emit_result()
            # or emit_error() for each finished request. It owns all OCR scheduling.
            run_summary = recognizer.serve(
                self,
                schedule_id="http:open",
                emit_result=self.emit_result,
                on_request_error=self.emit_error,
                collect_scheduling_metrics=self.serve_config.metrics_level != "basic",
            )
        except BaseException as exc:
            # Every request still in progress gets an error instead of silence.
            for request_id in list(self.jobs_in_progress):
                self.results.put(
                    {"kind": "result", "request_id": request_id, "ok": False, **_describe_failure(exc)}
                )
            raise
        self.results.put({"kind": "service_summary", "payload": asdict(run_summary)})

    def pull(self, *, block: bool) -> RecognitionRequest | None:
        """Hand the recognizer the next image, or None when there is none right now."""
        if self._closed:
            return None
        try:
            job = self.jobs.get() if block else self.jobs.get_nowait()
        except queue.Empty:
            return None
        if job is None:
            self._closed = True
            return None
        self.jobs_in_progress[job["request_id"]] = job
        return RecognitionRequest(
            request_id=job["request_id"],
            crop=job["image_bytes"],  # decoded later on the recognizer's CPU preparation thread
            prompt=PROMPTS[job["crop_type"]],
            submitted_at=job["submitted_monotonic_s"],
        )

    @property
    def closed(self) -> bool:
        """True only after the HTTP process signals that no more images will arrive."""
        return self._closed

    def emit_result(self, recognition: Any) -> None:
        """Send one finished OCR result back, as raw model text and as HTML."""
        from crop_processing import convert_otsl_to_html  # Torch-backed; child-only import

        job = self.jobs_in_progress.pop(recognition.request_id)
        payload = asdict(recognition)
        payload["raw_text"] = payload["text"]
        payload["text"] = convert_otsl_to_html(payload["raw_text"]) or payload["raw_text"]
        payload["crop_type"] = job["crop_type"]
        payload["worker_wall_s"] = time.perf_counter() - job["submitted_monotonic_s"]
        self.results.put(
            {"kind": "result", "request_id": recognition.request_id, "ok": True, "payload": payload}
        )

    def emit_error(self, request_id: str, exc: BaseException) -> None:
        """Report one failed request so its HTTP caller receives an error."""
        self.jobs_in_progress.pop(request_id, None)
        self.results.put(
            {"kind": "result", "request_id": request_id, "ok": False, **_describe_failure(exc)}
        )


def _describe_failure(exc: BaseException) -> dict[str, str]:
    """The error fields of every failure message the child sends."""
    return {
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _freeze_setup_gc() -> dict[str, Any]:
    """Exclude persistent setup objects from later cyclic-GC scans.

    Called only in the inference worker, before accepting any request.
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


# Service summary file and CLI parsing: ServeConfig above is the readable list of options.


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
        "--device", default=ServeConfig.device,
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
        "--graph-cache-directory", type=Path, required=True,
        help="Root directory for all compiled graphs, separated into stage subdirectories.",
    )
    parser.add_argument(
        "--log-folder", type=Path, required=True,
        help="Directory for service_summary.json at shutdown; continuous logging is not implemented yet.",
    )
    args = parser.parse_args()
    # Resolve caller-supplied paths once, relative to the launch working directory.
    args.model_path = args.model_path.expanduser().resolve()
    args.graph_cache_directory = args.graph_cache_directory.expanduser().resolve()
    args.log_folder = args.log_folder.expanduser().resolve()
    return ServeConfig(**vars(args))


if __name__ == "__main__":
    main()
