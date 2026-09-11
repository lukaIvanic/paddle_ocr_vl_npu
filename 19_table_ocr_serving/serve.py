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
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE
REPO_ROOT = EXPERIMENT_ROOT.parent

PROMPTS = {"table": "Table Recognition:"}


# Application startup and operational arguments


def main() -> None:
    args = parse_args()
    context = mp.get_context("spawn")
    jobs = context.Queue(maxsize=args.queue_capacity)
    results = context.Queue()
    config = {
        "model": str(args.model.expanduser().resolve()),
        "decode_batch_size": args.decode_batch_size,
        "full_decode_lm_head": args.full_decode_lm_head,
        "eager": args.eager,
        "decode_device_timing": not args.no_decode_device_timing,
        "request_scheduling_metrics": args.request_scheduling_metrics,
        "torchair_cache_dir": str(args.torchair_cache_dir.expanduser().resolve()),
        "vision_torchair_cache_dir": str(
            args.vision_torchair_cache_dir.expanduser().resolve()
        ),
        "text_torchair_cache_dir": str(
            args.text_torchair_cache_dir.expanduser().resolve()
        ),
        "device": args.device,
    }
    worker = context.Process(
        target=_worker_main, args=(jobs, results, config), name="crop-ocr-npu-worker"
    )
    worker.start()
    state = _State(
        jobs=jobs,
        results=results,
        worker=worker,
        timeout_s=args.request_timeout_s,
        max_image_bytes=args.max_image_bytes,
    )
    print(f"Waiting for NPU worker pid={worker.pid}", flush=True)
    state.ready.wait(timeout=args.request_timeout_s)
    if state.startup_error is not None:
        print(state.startup_error["traceback"], file=sys.stderr)
        raise RuntimeError(state.startup_error["error"])
    if not state.ready.is_set() or not worker.is_alive():
        raise RuntimeError("NPU worker did not become ready")

    server = _Server((args.host, args.port), _Handler)
    server.state = state  # type: ignore[attr-defined]
    stop_once = threading.Event()

    def stop_server(signum: int, frame: Any) -> None:
        del signum, frame
        if not stop_once.is_set():
            stop_once.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    print(
        f"READY http://{args.host}:{args.port} worker_pid={state.worker_pid}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        service_summary: dict[str, Any] | None = None
        try:
            service_summary = state.drain()
        except (queue.Empty, queue.Full):
            worker.terminate()
        if args.service_summary_output is not None and service_summary is not None:
            _write_service_summary(
                args.service_summary_output,
                configuration=state.configuration,
                worker_pid=state.worker_pid,
                summary=service_summary,
            )
            print(
                f"SERVICE_SUMMARY {args.service_summary_output.expanduser().resolve()}",
                flush=True,
            )
        worker.join(timeout=10.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5.0)
        state.stopping.set()
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--max-image-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--queue-capacity", type=int, default=256)
    parser.add_argument("--eager", action="store_true", help="Run the same NPU stages without TorchAir compilation.")
    parser.add_argument(
        "--model", type=Path, default=Path("/workspace/models/PaddleOCR-VL-1.6")
    )
    parser.add_argument(
        "--device",
        default="npu:0",
        help="Logical NPU within the process's visible-device set.",
    )
    parser.add_argument(
        "--decode-batch-size", type=int, default=2,
        help="Physical decode batch B, not a client concurrency limit; no batch-filling wait.",
    )
    parser.add_argument(
        "--full-decode-lm-head", action="store_true",
        help="Use all 103,424 checkpoint vocabulary rows instead of the bundled 60,416-row decode head.",
    )
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--no-decode-device-timing", action="store_true", default=True)
    timing.add_argument(
        "--decode-device-timing", dest="no_decode_device_timing", action="store_false"
    )
    parser.add_argument(
        "--request-scheduling-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--torchair-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/19_table_ocr_serving_torchair",
    )
    parser.add_argument(
        "--vision-torchair-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/19_table_ocr_serving_vision_torchair",
    )
    parser.add_argument(
        "--text-torchair-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/19_table_ocr_serving_text_torchair",
    )
    parser.add_argument("--service-summary-output", type=Path)
    return parser.parse_args()


# HTTP request handling


class _Handler(BaseHTTPRequestHandler):
    server_version = "PaddleOCRCropAPI/1"

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/v1/drain":
            self._json(
                HTTPStatus.OK,
                {"drained": False, "summary": {}, "server_running": True},
            )
            return
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
        if length <= 0 or length > self.state.max_image_bytes:
            self._json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "invalid image body size"},
            )
            return
        image_bytes = self.rfile.read(length)
        submitted = time.perf_counter()
        try:
            message = self.state.submit(
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
                    "ok": self.state.worker.is_alive(),
                    "worker_pid": self.state.worker_pid,
                },
            )
        elif path == "/ready":
            ok = (
                self.state.ready.is_set()
                and self.state.startup_error is None
                and self.state.worker.is_alive()
            )
            self._json(
                HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "ready": ok,
                    "worker_pid": self.state.worker_pid,
                    "configuration": self.state.configuration,
                    "startup_error": self.state.startup_error,
                },
            )
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    @property
    def state(self) -> _State:
        return self.server.state  # type: ignore[attr-defined]

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


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


# Request submission, result delivery, and shutdown


class _State:

    def submit(self, job: dict[str, Any]) -> dict[str, Any]:
        waiter: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self.lock:
            self.waiters[job["request_id"]] = waiter
        try:
            job["submitted_monotonic_s"] = time.perf_counter()
            self.jobs.put(job, timeout=1.0)
            return waiter.get(timeout=self.timeout_s)
        finally:
            with self.lock:
                self.waiters.pop(job["request_id"], None)

    def _dispatch(self) -> None:
        while not self.stopping.is_set():
            try:
                message = self.results.get(timeout=0.25)
            except queue.Empty:
                continue
            kind = message.get("kind")
            if kind == "ready":
                self.configuration = message.get("configuration")
                self.worker_pid = message.get("worker_pid")
                self.ready.set()
            elif kind == "startup_error":
                self.startup_error = message
                self.ready.set()
            elif kind == "result":
                with self.lock:
                    waiter = self.waiters.get(message["request_id"])
                if waiter is not None:
                    waiter.put(message)
            elif kind == "service_summary":
                self.service_summary = message["payload"]
                self.service_summary_ready.set()

    def drain(self) -> dict[str, Any]:
        self.jobs.put(None, timeout=1.0)
        if not self.service_summary_ready.wait(timeout=self.timeout_s):
            raise queue.Empty("timed out waiting for recognizer drain")
        assert self.service_summary is not None
        return self.service_summary

    def __init__(
        self,
        *,
        jobs: Any,
        results: Any,
        worker: Any,
        timeout_s: float,
        max_image_bytes: int,
    ):
        self.jobs = jobs
        self.results = results
        self.worker = worker
        self.timeout_s = timeout_s
        self.max_image_bytes = max_image_bytes
        self.ready = threading.Event()
        self.startup_error: dict[str, Any] | None = None
        self.configuration: dict[str, Any] | None = None
        self.worker_pid: int | None = None
        self.waiters: dict[str, queue.Queue[dict[str, Any]]] = {}
        self.lock = threading.Lock()
        self.service_summary: dict[str, Any] | None = None
        self.service_summary_ready = threading.Event()
        self.stopping = threading.Event()
        self.dispatcher = threading.Thread(
            target=self._dispatch, name="ocr-result-dispatch", daemon=True
        )
        self.dispatcher.start()


# NPU worker lifecycle


def _worker_main(
    jobs: Any,
    results: Any,
    config: dict[str, Any],
) -> None:
    """Own the NPU runtime and execute each queued crop immediately."""

    try:
        sys.path.insert(0, str(EXPERIMENT_ROOT))
        from serving_runtime import ContinuousRecognizer
        from _support.serving.types import RecognitionRequest
        from crop_processing import normalize_recognition_text

        recognizer = ContinuousRecognizer(
            model=config["model"],
            batch_size=config["decode_batch_size"],
            full_decode_lm_head=config["full_decode_lm_head"],
            decode_device_timing=config["decode_device_timing"],
            torchair_cache_dir=Path(config["torchair_cache_dir"]),
            eager=config["eager"],
            vision_torchair_cache_dir=Path(config["vision_torchair_cache_dir"]),
            text_torchair_cache_dir=Path(config["text_torchair_cache_dir"]),
            device=config["device"],
        )
        setup_gc = _freeze_setup_gc()
        if setup_gc["enabled"]:
            print("EXP09_SETUP_GC " + json.dumps(setup_gc), flush=True)
        configuration = recognizer.configuration()
        configuration["setup_gc"] = setup_gc
        configuration["request_scheduling_metrics"] = config[
            "request_scheduling_metrics"
        ]
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
                        min_pixels=recognizer.preprocessor_min_pixels_override,
                        max_pixels=recognizer.preprocessor_max_pixels_override,
                        submitted_at=job["submitted_monotonic_s"],
                    )
                return None

        def emit_result(recognition: Any) -> None:
            request_id = recognition.request_id
            job = request_jobs.pop(request_id)
            payload = asdict(recognition)
            payload["raw_text"] = payload["text"]
            payload["text"] = normalize_recognition_text(
                job["crop_type"],
                payload["raw_text"],
            )
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
                collect_scheduling_metrics=config["request_scheduling_metrics"],
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
