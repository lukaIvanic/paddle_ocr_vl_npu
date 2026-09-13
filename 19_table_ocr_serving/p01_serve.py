#!/usr/bin/env python3
"""HTTP and asynchronous Python interfaces for one OCR crop per request.

Two processes: this one answers HTTP, and a child process owns the model and
the NPU. They talk through two queues, described on InferenceServer below.
The HTTP process never imports Torch.

Endpoints: POST /v1/ocr?crop_type=table|text|formula with encoded image bytes,
GET /health, and GET /ready. Python calls InferenceServer directly without HTTP.

Decoding uses the bundled 60,416-row vocabulary by default, or
the full checkpoint vocabulary when --full-decode-lm-head is given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from p02_serving_runtime import RecognitionRequest


# The caller supplies the crop type; we do not try to classify the image.
PROMPTS = {
    "table": "Table Recognition:",
    "text": "OCR:",
    "formula": "Formula Recognition:",
}

# Application startup and operational arguments


@dataclass(frozen=True) # frozen means params can't be changed after init
class ServeConfig:
    """Options for HTTP or Python serving and the shared NPU worker."""

    model_path: Path
    graph_cache_directory: Path
    log_folder: Path

    # Address of the HTTP endpoint. The Python interface does not use these.
    host: str = "127.0.0.1"
    port: int = 8765

    # How long the caller waits for OCR. A timeout does not cancel inference;
    # the image still counts toward the limit below until processing finishes.
    request_timeout_s: float = 60.0
    # Startup waits for model setup to finish, without a time limit.
    # Shutdown has its own limit for finishing already-accepted work.
    shutdown_timeout_s: float = 900.0
    max_image_bytes: int = 64 * 1024 * 1024 # Largest encoded image accepted in one HTTP request (64 MiB).
    # Accepted OCR images waiting, being prepared or running, across both APIs.
    # Checked once image bytes are supplied; not a limit on TCP connections/uploads.
    max_in_flight_requests: int = 64


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

    # Both levels log startup, each request, errors, and periodic heartbeats.
    # basic: request latency/output tokens and live request/token rates and occupancy.
    # detailed: also CPU/wait/prefill/decode-residency/formatting wall times, crop
    # dimensions and input tokens; heartbeat includes recent mean/P95 latency.
    metrics_level: Literal["basic", "detailed"] = "basic"
    # Seconds between live status reports, including while the server is idle.
    heartbeat_interval_s: float = 15.0


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
        self.inference_server._log("http_ready", {
            "url": f"http://{self.serve_config.host}:{self.serve_config.port}",
            "worker_pid": self.inference_server.worker_pid,
        })
        self.serve_forever(poll_interval=0.25)

    def close(self) -> None:
        """Finish accepted HTTP requests and close sockets; do not stop inference."""
        self.server_close()

    def _stop_accepting_connections(self, signum: int, frame: Any) -> None:
        del signum, frame
        if not self.stop_requested.is_set():
            self.stop_requested.set()
            # Python requires shutdown() to run outside serve_forever()'s thread,
            # otherwise that thread would wait for itself and deadlock.
            threading.Thread(target=self.shutdown, daemon=True).start()


class HttpRequestHandler(BaseHTTPRequestHandler):
    """Read one client's image, wait for OCR, and send the response."""

    server_version = "PaddleOCRCropAPI/1"

    @property
    def inference_server(self) -> InferenceServer:
        return self.server.inference_server

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/v1/ocr":
            self.inference_server._log("request_rejected", {"reason": "unknown endpoint"})
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        query = parse_qs(parsed.query)
        crop_type = query.get("crop_type", [""])[0].strip().lower()
        if crop_type not in PROMPTS:
            self.inference_server._log("request_rejected", {"reason": "invalid crop type"})
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
            self.inference_server._log("request_rejected", {"reason": "invalid image body size"})
            self._json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "invalid image body size"},
            )
            return
        image_bytes = self.rfile.read(length)

        # Clients may reuse a request_id; the internal id stays unique per request.
        client_request_id = query.get("request_id", [uuid.uuid4().hex])[0].strip()
        request_id = f"{client_request_id}:{uuid.uuid4().hex}"
        self.ocr_request_id = request_id
        submitted = time.perf_counter()
        try:
            reply = self.inference_server.recognize(request_id, crop_type, image_bytes)
        except InferenceCapacityFull:
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE, {"error": "maximum unfinished OCR requests reached"}
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
        try:
            self.wfile.write(data)
        except OSError as exc:
            self.inference_server._log("response_write_failed", {
                "request_id": getattr(self, "ocr_request_id", None), "error": str(exc),
            })
            raise
        if self.command == "POST":
            self.inference_server._log("response_sent", {
                "request_id": getattr(self, "ocr_request_id", None), "http_status": int(status),
            })

    def log_message(self, format: str, *args: Any) -> None:
        # OCR lifecycle events above replace the duplicate HTTP access line.
        pass


# Inference service: share admission, results and the child process across both APIs.


class InferenceCapacityFull(Exception):
    """The maximum number of accepted, unfinished OCR requests has been reached."""


class InferenceTimeout(Exception):
    """No result arrived within request_timeout_s."""


class InferenceServer:
    """Start inference once; accept images from HTTP threads or async Python calls.

    Python callers use start(), await recognize_async(image_bytes, crop_type=...),
    then close() in a finally block. Startup and shutdown are blocking operations;
    recognition calls can overlap without opening an HTTP socket.

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
        if not math.isfinite(serve_config.heartbeat_interval_s) or serve_config.heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be finite and greater than zero")
        self.serve_config = serve_config

        # The model runs in a child process with a fresh Python interpreter.
        # These queues carry images to it and results back to this service.
        inference_process_context = mp.get_context("spawn")
        # Admission below bounds all unfinished requests, not just this queue.
        # Leaving room for the shutdown marker also lets a full service drain.
        self.jobs = inference_process_context.Queue()
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

        # One future holds one request's eventual reply. HTTP waits on it;
        # Python awaits it. Timed-out/cancelled futures stay here until the
        # worker finishes, so abandoned requests still occupy capacity.
        self.pending_requests: dict[str, Future[dict[str, Any]]] = {}
        self.requests_lock = threading.Lock()
        self.accepting_requests = False

        # The worker sends its final summary after the end-of-input marker.
        self.service_summary: dict[str, Any] | None = None
        self.service_summary_ready = threading.Event()
        self.stop_reading_results = threading.Event()
        self.result_reader = threading.Thread(
            target=self._receive_results_and_status, name="ocr-result-dispatch", daemon=True
        )
        # Only this thread formats records and touches the log file/terminal.
        # Requests hand it their existing result; image bytes are never queued.
        self.log_queue = queue.Queue(maxsize=1024)
        self.logs_dropped = threading.Event()
        self.stop_logging = threading.Event()
        self.log_writer = threading.Thread(target=self._write_logs, name="ocr-log-writer", daemon=True)
        self.started_at = time.perf_counter()

    def start(self) -> None:
        """Start the child process and wait until its model is loaded."""
        self.started_at = time.perf_counter()
        self.log_writer.start()
        self._log("startup", self.serve_config)
        self.inference_process.start()
        self._log("waiting_for_model", {"worker_pid": self.inference_process.pid})
        self.result_reader.start()
        self.startup_finished.wait()
        if self.startup_error is not None:
            raise RuntimeError(self.startup_error["error"])
        if not self.is_ready:
            raise RuntimeError("inference worker did not become ready")
        with self.requests_lock:
            self.accepting_requests = True

    def recognize(self, request_id: str, crop_type: str, image_bytes: bytes) -> dict[str, Any]:
        """Send one image to the inference worker and wait for its reply."""
        reply = self._submit_request(request_id, crop_type, image_bytes)
        try:
            return reply.result(timeout=self.serve_config.request_timeout_s)
        except FutureTimeoutError:
            self._stop_waiting_for_reply(reply, request_id, "request_timeout")
            raise InferenceTimeout() from None

    async def recognize_async(
        self, image_bytes: bytes, *, crop_type: Literal["table", "text", "formula"],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """OCR one encoded image; return the HTTP payload without HTTP timing."""
        client_request_id = request_id if request_id is not None else uuid.uuid4().hex
        internal_request_id = f"{client_request_id}:{uuid.uuid4().hex}"
        reply = self._submit_request(internal_request_id, crop_type, image_bytes)
        try:
            # Shield prevents caller cancellation from cancelling the shared
            # future outside our lock. Inference itself is never cancelled.
            result = await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(reply)),
                timeout=self.serve_config.request_timeout_s,
            )
        except TimeoutError:
            self._stop_waiting_for_reply(reply, internal_request_id, "request_timeout")
            raise InferenceTimeout() from None
        except asyncio.CancelledError:
            self._stop_waiting_for_reply(reply, internal_request_id, "caller_cancelled")
            raise
        if not result["ok"]:
            raise RuntimeError(result["error"])
        payload = result["payload"]
        payload["request_id"] = client_request_id
        self._log("python_result_returned", {"request_id": internal_request_id})
        return payload

    def _submit_request(self, request_id: str, crop_type: str, image_bytes: bytes) -> Future[dict[str, Any]]:
        """Accept one image or reject it immediately; capacity lasts until its result arrives."""
        if crop_type not in PROMPTS:
            self._log("request_rejected", {"request_id": request_id, "reason": "invalid crop type"})
            raise ValueError(f"crop_type must be one of {sorted(PROMPTS)}")
        if not isinstance(image_bytes, bytes):
            self._log("request_rejected", {"request_id": request_id, "reason": "expected encoded image bytes"})
            raise TypeError("image_bytes must contain an encoded image as bytes")
        if not 0 < len(image_bytes) <= self.serve_config.max_image_bytes:
            self._log("request_rejected", {"request_id": request_id, "reason": "invalid image body size"})
            raise ValueError("invalid image body size")
        with self.requests_lock:
            if not self.accepting_requests or not self.is_alive:
                self._log("request_rejected", {"request_id": request_id, "reason": "service unavailable"})
                raise RuntimeError("inference service is not accepting requests")
            if len(self.pending_requests) >= self.serve_config.max_in_flight_requests:
                self._log("request_rejected", {"request_id": request_id, "reason": "capacity full"})
                raise InferenceCapacityFull()
            if request_id in self.pending_requests:
                raise ValueError(f"duplicate internal request id: {request_id}")
            reply: Future[dict[str, Any]] = Future()
            self.pending_requests[request_id] = reply
            job = {
                "request_id": request_id,
                "crop_type": crop_type,
                "image_bytes": image_bytes,
                "submitted_monotonic_s": time.perf_counter(),
            }
            try:
                self.jobs.put_nowait(job)
            except BaseException:
                self.pending_requests.pop(request_id)
                raise
            self._log("request_accepted", (request_id, crop_type, len(self.pending_requests)))
            return reply

    def _stop_waiting_for_reply(self, reply: Future[dict[str, Any]], request_id: str, event: str) -> None:
        # Do not remove the request. Only the worker's result frees its capacity.
        with self.requests_lock:
            reply.cancel()
            self._log(event, {"request_id": request_id})

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
        try:
            with self.requests_lock:
                self.accepting_requests = False
                self._log("shutdown_started", {"unfinished_requests": len(self.pending_requests)})
            if self.stop_reading_results.is_set():
                return self.service_summary
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
            self._fail_pending_requests("inference service stopped before returning a result")
            if summary is not None:
                summary_path = self.serve_config.log_folder / "service_summary.json"
                try:
                    _write_service_summary(summary_path, configuration=self.worker_runtime_info,
                                           worker_pid=self.worker_pid, summary=summary)
                except OSError as exc:
                    self._log("logging_warning", {"error": f"shutdown summary could not be written: {exc}"})
                self._log("shutdown_finished", summary)
            return summary
        finally:
            self._close_logging()


    def _ask_worker_to_finish(self) -> dict[str, Any] | None:
        """Send the end-of-input marker and wait for the summary; None if the worker cannot answer."""
        if self.startup_error is not None or not self.is_alive:
            return None
        # Admission is already closed. All accepted jobs precede this marker,
        # including timed-out jobs; the worker finishes them before its summary.
        self.jobs.put_nowait(None)
        if not self.service_summary_ready.wait(timeout=self.serve_config.shutdown_timeout_s):
            return None
        return self.service_summary

    def _fail_pending_requests(self, error: str) -> None:
        """Release waiters when inference cannot produce any more results."""
        with self.requests_lock:
            self.accepting_requests = False
            for request_id, reply in self.pending_requests.items():
                self._log("request_failed", {"request_id": request_id, "error": error})
                if not reply.cancelled():
                    reply.set_exception(RuntimeError(error))
            self.pending_requests.clear()

    def _receive_results_and_status(self) -> None:
        """Receive inference status and deliver each OCR result to its waiting request."""
        while not self.stop_reading_results.is_set():
            try:
                message = self.results.get(timeout=0.25)
            except queue.Empty:
                if not self.is_alive:
                    if not self.service_summary_ready.is_set():
                        self.startup_error = {"error": "inference worker exited", "traceback": ""}
                        self._log("inference_failed", self.startup_error)
                        self.startup_finished.set()
                        self._fail_pending_requests("inference worker exited")
                    return
                continue
            kind = message.get("kind")
            if kind == "ready":
                self.worker_runtime_info = message.get("configuration")
                self.worker_pid = message.get("worker_pid")
                message["startup_wall_s"] = time.perf_counter() - self.started_at
                self._log("inference_ready", message)
                self.startup_finished.set()
            elif kind == "startup_error":
                self._log("inference_failed", message)
                self.startup_error = message
                self.startup_finished.set()
                self._fail_pending_requests(message["error"])
            elif kind == "result":
                with self.requests_lock:
                    reply = self.pending_requests.pop(message["request_id"], None)
                    # Log before waking the caller, which may modify its payload's ID.
                    # The writer uses the immutable outer request_id, not that field.
                    self._log("request_finished" if message.get("ok") else "request_failed", message)
                    if reply is not None and reply.cancelled():
                        self._log("result_not_returned", {"request_id": message["request_id"], "reason": "caller no longer waiting"})
                    if reply is not None and not reply.cancelled():
                        reply.set_result(message)
            elif kind == "service_summary":
                self.service_summary = message["payload"]
                self.service_summary_ready.set()
            elif kind in ("setup", "heartbeat"):
                if kind == "heartbeat":
                    with self.requests_lock:
                        message["unfinished_requests"] = len(self.pending_requests)
                self._log(kind, message)

    def _log(self, event: str, data: Any = None) -> None:
        """Hand existing data to the writer without waiting for disk or console."""
        if self.stop_logging.is_set():
            return
        try:
            self.log_queue.put_nowait((time.time(), event, data))
        except queue.Full:
            self.logs_dropped.set()  # The writer warns once it can make progress.

    def _write_logs(self) -> None:
        """One JSON line per event, printed and flushed to events.jsonl immediately."""
        log_file = None
        warning_at = float('-inf')
        recent_latencies = deque(maxlen=2048)
        counts = dict(accepted=0, completed=0, failed=0, rejected=0, timed_out=0)
        previous_counts = dict(counts)
        previous_tokens = 0
        previous_observed = None
        count_events = dict(request_accepted='accepted', request_finished='completed',
                            request_failed='failed', request_rejected='rejected', request_timeout='timed_out')
        try:
            while not self.stop_logging.is_set() or not self.log_queue.empty():
                try:
                    timestamp, event, data = self.log_queue.get(timeout=.25)
                except queue.Empty:
                    continue
                record = {"timestamp": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(), "event": event}
                if event in count_events:
                    counts[count_events[event]] += 1
                if event == 'startup':
                    record.update(asdict(data))
                elif event == 'request_accepted':
                    record.update(zip(('request_id', 'crop_type', 'unfinished_requests'), data))
                elif event == 'request_finished':
                    # Reuse the received result. Never serialize recognized text,
                    # token arrays or images into operational logs.
                    result = data['payload']
                    record['request_id'] = data['request_id']
                    for key in ('crop_type', 'worker_wall_s', 'generated_tokens_including_eos', 'stop_reason'):
                        if key in result:
                            record[key] = result[key]
                    latency = result.get('worker_wall_s')
                    if latency is not None:
                        recent_latencies.append((timestamp, latency))
                    if self.serve_config.metrics_level == 'detailed':
                        for key in ('crop_size', 'input_tokens', 'projected_image_tokens', 'vision', 'timing_s'):
                            if key in result:
                                record[key] = result[key]
                elif data is not None:
                    record.update((key, value) for key, value in data.items() if key != 'kind')
                if event == 'heartbeat':
                    observed = record.pop('observed_monotonic_s')
                    tokens = record['output_tokens_including_eos']
                    elapsed = None if previous_observed is None else observed - previous_observed
                    record['interval_s'] = elapsed
                    record['output_tokens_per_s'] = (tokens - previous_tokens) / elapsed if elapsed else None
                    for name, total in counts.items():
                        record[name + '_since_last_heartbeat'] = total - previous_counts[name]
                    record['accepted_requests_per_s'] = (counts['accepted'] - previous_counts['accepted']) / elapsed if elapsed else None
                    record['completed_requests_per_s'] = (counts['completed'] - previous_counts['completed']) / elapsed if elapsed else None
                    record['max_in_flight_requests'] = self.serve_config.max_in_flight_requests
                    previous_observed, previous_tokens = observed, tokens
                    previous_counts = dict(counts)
                    if self.serve_config.metrics_level == 'detailed':
                        while recent_latencies and recent_latencies[0][0] < timestamp - 60:
                            recent_latencies.popleft()
                        values = sorted(value for _, value in recent_latencies)
                        record['recent_completed_count'] = len(values)
                        record['recent_window_s'] = 60
                        record['mean_latency_s'] = sum(values) / len(values) if values else None
                        if values:
                            position = .95 * (len(values) - 1)
                            lower = int(position)
                            record['p95_latency_s'] = values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)
                        else:
                            record['p95_latency_s'] = None
                line = json.dumps(record, ensure_ascii=False, default=str, separators=(',', ':'))
                problems = []
                if self.logs_dropped.is_set():
                    self.logs_dropped.clear()
                    problems.append('logging queue full; records were dropped')
                try:
                    print(line, flush=True)
                except (OSError, ValueError) as exc:
                    problems.append(f'console logging failed: {exc}')
                try:
                    if log_file is None:
                        self.serve_config.log_folder.mkdir(parents=True, exist_ok=True)
                        log_file = (self.serve_config.log_folder / 'events.jsonl').open('a', encoding='utf-8')
                    log_file.write(line + '\n')
                    log_file.flush()
                except (OSError, ValueError) as exc:
                    problems.append(f'file logging failed: {exc}')
                if problems and time.monotonic() - warning_at >= self.serve_config.heartbeat_interval_s:
                    warning_at = time.monotonic()
                    warning = json.dumps({'timestamp': record['timestamp'], 'event': 'logging_warning', 'warning': '; '.join(problems)})
                    try:
                        print(warning, file=sys.stderr, flush=True)
                    except (OSError, ValueError):
                        pass  # A broken stderr cannot be allowed to fail inference either.
                    if log_file is not None:
                        try:
                            log_file.write(warning + '\n')
                            log_file.flush()
                        except (OSError, ValueError):
                            pass
        finally:
            if log_file is not None:
                try:
                    log_file.close()
                except (OSError, ValueError):
                    pass

    def _close_logging(self) -> None:
        self.stop_logging.set()
        if self.log_writer.ident is not None:
            self.log_writer.join(timeout=2.0)
        # A stuck file/terminal must not prevent shutdown. The writer is daemonized.

# Inference child process: own the model and execute OCR.


def run_inference_process(jobs: Any, results: Any, serve_config: ServeConfig) -> None:
    """Entrypoint of the child process: build the worker and run it until closed."""
    try:
        InferenceWorker(jobs, results, serve_config).run()
    except BaseException as exc:
        results.put({"kind": "startup_error", **_describe_failure(exc)})


class InferenceWorker:
    """Load the model, then OCR every submitted image until told to stop."""

    def __init__(self, jobs: Any, results: Any, serve_config: ServeConfig) -> None:
        self.jobs = jobs
        self.results = results
        self.serve_config = serve_config
        self.jobs_in_progress: dict[str, dict[str, Any]] = {}
        self._closed = False
        self.last_status_at = float('-inf')

    def run(self) -> None:
        recognizer = self._load_model_and_report_ready()
        self._serve_until_closed(recognizer)

    def _load_model_and_report_ready(self) -> Any:
        # Only the child process imports Torch and the model code.
        from p02_serving_runtime import ContinuousRecognizer

        config = self.serve_config
        recognizer = ContinuousRecognizer(
            model=str(config.model_path),
            batch_size=config.decode_batch_size,
            full_decode_lm_head=config.full_decode_lm_head,
            setup_progress=self.report_setup,
            graph_cache_directory=config.graph_cache_directory,
            eager=config.run_eagerly,
            device=config.device,
        )
        self.recognizer = recognizer
        configuration = recognizer.configuration()
        configuration["setup_gc"] = _freeze_setup_gc()
        configuration["request_scheduling_metrics"] = False
        configuration["metrics_level"] = config.metrics_level
        configuration["heartbeat_interval_s"] = config.heartbeat_interval_s
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
                report_status=self.report_status,
            )
        except BaseException as exc:
            # Every request still in progress gets an error instead of silence.
            for request_id in list(self.jobs_in_progress):
                self.results.put(
                    {"kind": "result", "request_id": request_id, "ok": False, **_describe_failure(exc)}
                )
            raise
        self.report_status(force=True)
        self.results.put({"kind": "service_summary", "payload": asdict(run_summary)})

    def report_setup(self, stage: str, status: str, elapsed_s: float | None = None) -> None:
        """Forward the existing setup progress through the existing results queue."""
        self.results.put_nowait({"kind": "setup", "stage": stage, "status": status, "elapsed_s": elapsed_s})

    def report_status(self, *, force: bool = False) -> None:
        """Publish CPU-side counters at the configured heartbeat interval."""
        now = time.perf_counter()
        if not force and now - self.last_status_at < self.serve_config.heartbeat_interval_s:
            return
        self.last_status_at = now
        runtime = self.recognizer
        self.results.put_nowait({
            "kind": "heartbeat", "observed_monotonic_s": now,
            "output_tokens_including_eos": runtime.output_tokens,
            "active_decode_slots": runtime.decode_arena.num_active,
            "decode_batch_size": runtime.batch_size,
            "cpu_preparation_or_prefill_waiting": len(runtime.crops_awaiting_prefill),
            "prefilled_waiting_for_decode": len(runtime.ready_queue),
        })

    def pull(self, *, block: bool) -> RecognitionRequest | None:
        """Hand the recognizer the next image, or None when there is none right now."""
        # This runs in the inference child; the HTTP process never imports the model.
        from p02_serving_runtime import RecognitionRequest

        if self._closed:
            return None
        while True:
            self.report_status()
            try:
                # Wake during idle periods to report that the service is still
                # alive. No timeout or shutdown is applied to inference.
                remaining = max(.001, self.serve_config.heartbeat_interval_s - (time.perf_counter() - self.last_status_at))
                job = self.jobs.get(timeout=remaining) if block else self.jobs.get_nowait()
                break
            except queue.Empty:
                if not block:
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
        """True only after the inference service signals that no more images will arrive."""
        return self._closed

    def emit_result(self, recognition: Any) -> None:
        """Return raw and formatted text; only table crops are converted to HTML."""
        from p03_crop_processing import convert_otsl_to_html, normalize_math_delimiters  # Torch-backed; child-only import

        job = self.jobs_in_progress.pop(recognition.request_id)
        payload = asdict(recognition)
        payload["raw_text"] = payload["text"]
        formatting_started = time.perf_counter()
        formatted_text = normalize_math_delimiters(payload["raw_text"])
        payload["text"] = formatted_text
        if job["crop_type"] == "table":
            payload["text"] = convert_otsl_to_html(formatted_text) or formatted_text
        payload.setdefault("timing_s", {})["output_formatting"] = time.perf_counter() - formatting_started
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
    parser.add_argument("--shutdown-timeout-s", type=float, default=ServeConfig.shutdown_timeout_s)
    parser.add_argument("--max-image-bytes", type=int, default=ServeConfig.max_image_bytes)
    parser.add_argument("--max-in-flight-requests", type=int, default=ServeConfig.max_in_flight_requests)
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
        "--metrics-level", choices=("basic", "detailed"),
        default=ServeConfig.metrics_level,
        help="Basic lifecycle/rate logs, or detailed per-request wall timings; neither uses NPU profiling.",
    )
    parser.add_argument(
        "--heartbeat-interval-s", type=float, default=ServeConfig.heartbeat_interval_s,
        help="Seconds between live status reports (default: 15). Must be positive and finite.",
    )
    parser.add_argument(
        "--graph-cache-directory", type=Path, required=True,
        help="Root directory for all compiled graphs, separated into stage subdirectories.",
    )
    parser.add_argument(
        "--log-folder", type=Path, required=True,
        help="Directory for live events.jsonl and service_summary.json at shutdown.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.heartbeat_interval_s) or args.heartbeat_interval_s <= 0:
        parser.error("--heartbeat-interval-s must be finite and greater than zero")
    # Resolve caller-supplied paths once, relative to the launch working directory.
    args.model_path = args.model_path.expanduser().resolve()
    args.graph_cache_directory = args.graph_cache_directory.expanduser().resolve()
    args.log_folder = args.log_folder.expanduser().resolve()
    return ServeConfig(**vars(args))


if __name__ == "__main__":
    main()
