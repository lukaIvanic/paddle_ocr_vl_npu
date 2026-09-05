#!/usr/bin/env python3
"""Serve one PaddleOCR-VL crop per HTTP request.

The HTTP process never imports Torch. One spawned NPU process owns the
ContinuousRecognizer and its compiled-graph caches for the server lifetime.
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
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent

PROMPTS = {
    "text": "OCR:",
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}
DEFAULT_VISION_BUCKETS = "256,384,512,640,768,1408,1920,2048,2944,4096"
# A 4096-token vision input becomes at most about 1024 projected tokens.
# Five existing B1 shapes cover that full table-recognition range while
# avoiding startup work for thirteen redundant intermediate buckets.
DEFAULT_TEXT_BUCKETS = "128,256,512,1024,1312"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--max-image-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--queue-capacity", type=int, default=256)
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/PaddleOCR-VL-1.6"))
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--decode-backend", default="torchair")
    parser.add_argument("--decode-optimization", default="combined_apply_pse_sentinel")
    parser.add_argument(
        "--decode-vocab-token-ids",
        type=Path,
        help=(
            "Optional JSON file of native token IDs for a decode-only compact "
            "LM head. Prefill keeps the full checkpoint head."
        ),
    )
    parser.add_argument(
        "--token-selection",
        default="greedy",
        help="Direct-logit token selection policy; validated in the NPU worker.",
    )
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument(
        "--request-scheduling-metrics", action="store_true",
        help="Include per-request prefill interruption and decode occupancy records.",
    )
    parser.add_argument(
        "--max-prefill-interruptions", type=int,
        help=(
            "Optional maximum other-request prefills during a table's decode. "
            "At the limit, defer new preparation/prefill until protected requests finish."
        ),
    )
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--vision-buckets", default=DEFAULT_VISION_BUCKETS)
    parser.add_argument("--text-buckets", default=DEFAULT_TEXT_BUCKETS)
    parser.add_argument("--torchair-cache-dir", type=Path, default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair")
    parser.add_argument("--vision-torchair-cache-dir", type=Path, default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair")
    parser.add_argument("--text-torchair-cache-dir", type=Path, default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_torchair")
    parser.add_argument(
        "--service-summary-output",
        type=Path,
        help=(
            "Write the final decode scheduler and stage summary when the "
            "server drains during shutdown."
        ),
    )
    args = parser.parse_args()
    if args.max_prefill_interruptions is not None and args.max_prefill_interruptions < 0:
        parser.error("--max-prefill-interruptions must be nonnegative")
    return args


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


def _worker_main(
    jobs: Any,
    results: Any,
    config: dict[str, Any],
) -> None:
    """Own the NPU runtime and execute each queued crop immediately."""

    try:
        sys.path.insert(0, str(EXPERIMENT_ROOT))
        from paddleocr_vl.model.text_prefill import parse_text_buckets
        from paddleocr_vl.model.vision_prefill import parse_vision_buckets
        from paddleocr_vl.serving.engine import ContinuousRecognizer
        from paddleocr_vl.serving.types import RecognitionRequest
        from pipeline.layout_output import normalize_recognition_text

        recognizer = ContinuousRecognizer(
            model=config["model"],
            dtype=config["dtype"],
            decode_backend=config["decode_backend"],
            decode_optimization=config["decode_optimization"],
            decode_vocab_token_ids=(
                None
                if config["decode_vocab_token_ids"] is None
                else Path(config["decode_vocab_token_ids"])
            ),
            token_selection=config["token_selection"],
            batch_size=config["decode_batch_size"],
            cache_length=config["cache_length"],
            max_new_tokens=config["max_new_tokens"],
            torchair_cache_dir=Path(config["torchair_cache_dir"]),
            vision_backend="torchair",
            vision_attention="prompt_flash_attention",
            vision_promptfa_align_128=True,
            vision_mlp_intermediate_size=4352,
            vision_linear_weight_format="fractal_nz",
            vision_buckets=parse_vision_buckets(config["vision_buckets"]),
            vision_torchair_cache_dir=Path(config["vision_torchair_cache_dir"]),
            vision_padding="bucket",
            vision_packing="off",
            text_backend="torchair",
            text_buckets=parse_text_buckets(config["text_buckets"]),
            text_torchair_cache_dir=Path(config["text_torchair_cache_dir"]),
            text_padding="bucket",
            text_packing="off",
            preprocessor_min_pixels=config["min_pixels"],
            preprocessor_max_pixels=config["max_pixels"],
        )
        configuration = recognizer.configuration()
        configuration["request_scheduling_metrics"] = config["request_scheduling_metrics"]
        configuration["max_prefill_interruptions"] = config["max_prefill_interruptions"]
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
                        min_pixels=config["min_pixels"],
                        max_pixels=config["max_pixels"],
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
                max_prefill_interruptions=config["max_prefill_interruptions"],
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


class _State:
    def __init__(self, *, jobs: Any, results: Any, worker: Any, timeout_s: float, max_image_bytes: int):
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
        self.dispatcher = threading.Thread(target=self._dispatch, name="ocr-result-dispatch", daemon=True)
        self.dispatcher.start()

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

    def drain(self) -> dict[str, Any]:
        self.jobs.put(None, timeout=1.0)
        if not self.service_summary_ready.wait(timeout=self.timeout_s):
            raise queue.Empty("timed out waiting for recognizer drain")
        assert self.service_summary is not None
        return self.service_summary


class _Handler(BaseHTTPRequestHandler):
    server_version = "PaddleOCRCropAPI/1"

    @property
    def state(self) -> _State:
        return self.server.state  # type: ignore[attr-defined]

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._json(HTTPStatus.OK, {"ok": self.state.worker.is_alive(), "worker_pid": self.state.worker_pid})
        elif path == "/ready":
            ok = self.state.ready.is_set() and self.state.startup_error is None and self.state.worker.is_alive()
            self._json(HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE, {"ready": ok, "worker_pid": self.state.worker_pid, "configuration": self.state.configuration, "startup_error": self.state.startup_error})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

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
            self._json(HTTPStatus.BAD_REQUEST, {"error": f"crop_type must be one of {sorted(PROMPTS)}"})
            return
        public_request_id = query.get("request_id", [uuid.uuid4().hex])[0].strip()
        source_request_id = query.get(
            "source_request_id", [public_request_id]
        )[0].strip()
        request_id = f"{public_request_id}:{uuid.uuid4().hex}"
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > self.state.max_image_bytes:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid image body size"})
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
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "recognition queue is full"})
            return
        except queue.Empty:
            self._json(HTTPStatus.GATEWAY_TIMEOUT, {"error": "recognition timed out"})
            return
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})
            return
        if not message["ok"]:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, message)
            return
        message["payload"]["request_id"] = public_request_id
        message["payload"]["http_wall_s"] = time.perf_counter() - submitted
        self._json(HTTPStatus.OK, message["payload"])

    def log_message(self, format: str, *args: Any) -> None:
        print(f"HTTP {self.address_string()} {format % args}", flush=True)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def main() -> None:
    args = parse_args()
    context = mp.get_context("spawn")
    jobs = context.Queue(maxsize=args.queue_capacity)
    results = context.Queue()
    config = {
        "model": str(args.model.expanduser().resolve()),
        "dtype": args.dtype,
        "decode_backend": args.decode_backend,
        "decode_optimization": args.decode_optimization,
        "decode_vocab_token_ids": (
            None
            if args.decode_vocab_token_ids is None
            else str(args.decode_vocab_token_ids.expanduser().resolve())
        ),
        "token_selection": args.token_selection,
        "decode_batch_size": args.decode_batch_size,
        "request_scheduling_metrics": args.request_scheduling_metrics,
        "max_prefill_interruptions": args.max_prefill_interruptions,
        "cache_length": args.cache_length,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "vision_buckets": args.vision_buckets,
        "text_buckets": args.text_buckets,
        "torchair_cache_dir": str(args.torchair_cache_dir.expanduser().resolve()),
        "vision_torchair_cache_dir": str(args.vision_torchair_cache_dir.expanduser().resolve()),
        "text_torchair_cache_dir": str(args.text_torchair_cache_dir.expanduser().resolve()),
    }
    worker = context.Process(target=_worker_main, args=(jobs, results, config), name="crop-ocr-npu-worker")
    worker.start()
    state = _State(jobs=jobs, results=results, worker=worker, timeout_s=args.request_timeout_s, max_image_bytes=args.max_image_bytes)
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
    print(f"READY http://{args.host}:{args.port} worker_pid={state.worker_pid}", flush=True)
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


if __name__ == "__main__":
    main()
