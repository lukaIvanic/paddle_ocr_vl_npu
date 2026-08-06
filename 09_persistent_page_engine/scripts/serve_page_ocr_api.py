#!/usr/bin/env python3
"""Serve the persistent full-page PaddleOCR-VL pipeline over HTTP."""

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
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
VISION_BUCKETS = "256,384,512,640,768,1408,1920,2048,2944,4096"
TEXT_PACK_BUCKETS = "128,256,384,512,768,1024"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-image-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--queue-capacity", type=int, default=64)
    parser.add_argument(
        "--spool-dir",
        type=Path,
        default=REPO_ROOT / "tmp/09_persistent_page_engine/page_api_uploads",
    )
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=Path("/workspace/models/PP-DocLayoutV3_safetensors"),
    )
    parser.add_argument(
        "--recognizer-model",
        type=Path,
        default=Path("/workspace/models/PaddleOCR-VL-1.6"),
    )
    parser.add_argument(
        "--layout-graph-capture",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--text-max-pixels", type=int, default=None)
    parser.add_argument("--table-max-pixels", type=int, default=None)
    parser.add_argument("--text-crop-scale", type=float, default=0.5)
    parser.add_argument("--vision-buckets", default=VISION_BUCKETS)
    parser.add_argument("--vision-pack-target", type=int, default=768)
    parser.add_argument(
        "--vision-promptfa-align-128",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--text-buckets", default="1152")
    parser.add_argument("--text-pack-buckets", default=TEXT_PACK_BUCKETS)
    parser.add_argument(
        "--torchair-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair",
    )
    parser.add_argument(
        "--vision-torchair-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair",
    )
    parser.add_argument(
        "--text-torchair-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_torchair",
    )
    parser.add_argument(
        "--text-packed-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_packed_torchair",
    )
    return parser.parse_args(argv)


def worker_config(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {
        "host",
        "port",
        "request_timeout_s",
        "max_image_bytes",
        "queue_capacity",
        "spool_dir",
    }
    return {
        key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in excluded
    }


def _worker_main(jobs: Any, results: Any, config: dict[str, Any]) -> None:
    try:
        sys.path.insert(0, str(EXPERIMENT_ROOT))
        import torch
        import torch_npu  # noqa: F401

        from paddleocr_vl.model.text_prefill import parse_text_buckets
        from paddleocr_vl.model.vision_prefill import parse_vision_buckets
        from paddleocr_vl.serving.engine import ContinuousRecognizer
        from pipeline.layout_frontend import OwnedLayoutFrontend
        from pipeline.layout_mask_guard import install_layout_mask_guard
        from pipeline.layout_output import page_markdown
        from pipeline.persistent_page_service import PageSubmission, PersistentPageEngine

        torch.npu.config.allow_internal_format = True
        torch.npu.set_compile_mode(jit_compile=False)
        install_layout_mask_guard()
        frontend = OwnedLayoutFrontend(
            Path(config["layout_model"]),
            torch.device("npu:0"),
            graph_capture=config["layout_graph_capture"],
        )
        recognizer = ContinuousRecognizer(
            model=config["recognizer_model"],
            dtype="fp16",
            decode_backend="torchair",
            decode_optimization="combined_apply_pse_sentinel",
            batch_size=config["decode_batch_size"],
            cache_length=config["cache_length"],
            max_new_tokens=config["max_new_tokens"],
            torchair_cache_dir=Path(config["torchair_cache_dir"]),
            vision_backend="torchair",
            vision_attention="prompt_flash_attention",
            vision_promptfa_align_128=config["vision_promptfa_align_128"],
            vision_mlp_intermediate_size=4352,
            vision_linear_weight_format="fractal_nz",
            vision_buckets=parse_vision_buckets(config["vision_buckets"]),
            vision_torchair_cache_dir=Path(config["vision_torchair_cache_dir"]),
            vision_padding="bucket",
            vision_packing="greedy",
            vision_pack_target=config["vision_pack_target"],
            text_backend="torchair",
            text_buckets=parse_text_buckets(config["text_buckets"]),
            text_torchair_cache_dir=Path(config["text_torchair_cache_dir"]),
            text_padding="bucket",
            text_packing="production_group",
            text_pack_buckets=parse_text_buckets(config["text_pack_buckets"]),
            text_pack_max_members=32,
            text_packed_cache_dir=Path(config["text_packed_cache_dir"]),
            preprocessor_min_pixels=config["min_pixels"],
            preprocessor_max_pixels=config["max_pixels"],
        )
        engine = PersistentPageEngine(
            frontend,
            recognizer,
            min_pixels=config["min_pixels"],
            max_pixels=config["max_pixels"],
            text_max_pixels=config["text_max_pixels"],
            table_max_pixels=config["table_max_pixels"],
            text_crop_scale=config["text_crop_scale"],
        )

        class PageQueue:
            def __init__(self) -> None:
                self._closed = False

            @property
            def closed(self) -> bool:
                return self._closed

            def pull(self, *, block: bool) -> PageSubmission | None:
                if self._closed:
                    return None
                try:
                    job = jobs.get() if block else jobs.get_nowait()
                except queue.Empty:
                    return None
                if job is None:
                    self._closed = True
                    return None
                submitted_at = job.get("submitted_monotonic_s")
                if submitted_at is None:
                    submitted_at = time.perf_counter()
                return PageSubmission(
                    request_id=job["request_id"],
                    image_path=Path(job["image_path"]),
                    submitted_at=float(submitted_at),
                )

        results.put(
            {
                "kind": "ready",
                "worker_pid": __import__("os").getpid(),
                "configuration": recognizer.configuration(),
            }
        )

        def emit_page(request_id: str, result: Any, timing_s: dict[str, float]) -> None:
            markdown, _images = page_markdown(
                result.data["parsing_res_list"],
                page_width=result.data["width"],
            )
            results.put(
                {
                    "kind": "result",
                    "request_id": request_id,
                    "ok": True,
                    "payload": {
                        "request_id": request_id,
                        "markdown": markdown,
                        "result": result.json["res"],
                        "timing_s": timing_s,
                    },
                }
            )

        def emit_error(request_id: str, exc: BaseException) -> None:
            results.put(
                {
                    "kind": "result",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        service_started = time.perf_counter()
        with torch.inference_mode():
            summary = engine.serve(
                PageQueue(),
                emit_page=emit_page,
                emit_page_error=emit_error,
            )
        results.put(
            {
                "kind": "service_summary",
                "payload": {
                    "service_wall_s": time.perf_counter() - service_started,
                    "schedule": asdict(summary),
                    "device_modes": engine.last_mode_summary,
                },
            }
        )
    except BaseException as exc:
        results.put(
            {
                "kind": "service_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


class _State:
    def __init__(self, args: argparse.Namespace, jobs: Any, results: Any, worker: Any):
        self.jobs = jobs
        self.results = results
        self.worker = worker
        self.timeout_s = args.request_timeout_s
        self.max_image_bytes = args.max_image_bytes
        self.spool_dir = args.spool_dir.expanduser().resolve()
        self.ready = threading.Event()
        self.error: dict[str, Any] | None = None
        self.configuration: dict[str, Any] | None = None
        self.worker_pid: int | None = None
        self.waiters: dict[str, queue.Queue[dict[str, Any]]] = {}
        self.paths: dict[str, Path] = {}
        self.lock = threading.Lock()
        self.accepting = True
        self.worker_stop_sent = False
        self.summary: dict[str, Any] | None = None
        self.summary_ready = threading.Event()
        self.stopping = threading.Event()
        threading.Thread(target=self._dispatch, daemon=True).start()

    def _dispatch(self) -> None:
        while not self.stopping.is_set():
            try:
                message = self.results.get(timeout=0.25)
            except queue.Empty:
                continue
            kind = message.get("kind")
            if kind == "ready":
                self.configuration = message["configuration"]
                self.worker_pid = message["worker_pid"]
                self.ready.set()
            elif kind == "result":
                request_id = message["request_id"]
                with self.lock:
                    waiter = self.waiters.get(request_id)
                    path = self.paths.pop(request_id, None)
                if path is not None:
                    path.unlink(missing_ok=True)
                    try:
                        path.parent.rmdir()
                    except OSError:
                        pass
                if waiter is not None:
                    waiter.put(message)
            elif kind == "service_summary":
                self.summary = message["payload"]
                self.summary_ready.set()
            elif kind == "service_error":
                self.error = message
                self.ready.set()
                with self.lock:
                    waiters = list(self.waiters.values())
                for waiter in waiters:
                    waiter.put({"kind": "result", "ok": False, **message})

    def submit(self, request_id: str, filename: str, body: bytes) -> dict[str, Any]:
        waiter: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        request_dir = self.spool_dir / uuid.uuid4().hex
        request_dir.mkdir(parents=True, exist_ok=False)
        image_path = request_dir / (Path(filename).name or "page.png")
        image_path.write_bytes(body)
        with self.lock:
            self.waiters[request_id] = waiter
            self.paths[request_id] = image_path
        try:
            self.jobs.put(
                {
                    "request_id": request_id,
                    "image_path": str(image_path),
                    "submitted_monotonic_s": time.perf_counter(),
                },
                timeout=1.0,
            )
            return waiter.get(timeout=self.timeout_s)
        finally:
            with self.lock:
                self.waiters.pop(request_id, None)

    def shutdown_worker(self) -> dict[str, Any]:
        with self.lock:
            self.accepting = False
            first = not self.worker_stop_sent
            self.worker_stop_sent = True
        if first:
            self.jobs.put(None, timeout=1.0)
        if not self.summary_ready.wait(timeout=self.timeout_s):
            raise queue.Empty("timed out waiting for page worker shutdown")
        assert self.summary is not None
        return self.summary


class _Handler(BaseHTTPRequestHandler):
    server_version = "PaddleOCRPageAPI/1"

    @property
    def state(self) -> _State:
        return self.server.state  # type: ignore[attr-defined]

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/health":
            self._json(HTTPStatus.OK, {"ok": self.state.worker.is_alive()})
        elif urlparse(self.path).path == "/ready":
            ready = self.state.ready.is_set() and self.state.error is None
            self._json(
                HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                {"ready": ready, "configuration": self.state.configuration, "error": self.state.error},
            )
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/v1/pages", "/v1/parse"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        query = parse_qs(parsed.query)
        public_request_id = query.get("request_id", [uuid.uuid4().hex])[0]
        request_id = f"{public_request_id}:{uuid.uuid4().hex}"
        filename = query.get("filename", ["page.png"])[0]
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > self.state.max_image_bytes:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid image body size"})
            return
        started = time.perf_counter()
        try:
            message = self.state.submit(request_id, filename, self.rfile.read(length))
        except queue.Full:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "page queue is full"})
            return
        except queue.Empty:
            self._json(HTTPStatus.GATEWAY_TIMEOUT, {"error": "page request timed out"})
            return
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})
            return
        if not message.get("ok"):
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, message)
            return
        message["payload"]["request_id"] = public_request_id
        message["payload"]["http_wall_s"] = time.perf_counter() - started
        self._json(HTTPStatus.OK, message["payload"])

    def log_message(self, format: str, *args: Any) -> None:
        print(f"HTTP {self.address_string()} {format % args}", flush=True)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def main() -> None:
    args = parse_args()
    args.spool_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    jobs = context.Queue(maxsize=args.queue_capacity)
    results = context.Queue()
    config = worker_config(args)
    worker = context.Process(target=_worker_main, args=(jobs, results, config))
    worker.start()
    state = _State(args, jobs, results, worker)
    print(f"Waiting for NPU worker pid={worker.pid}", flush=True)
    state.ready.wait(timeout=args.request_timeout_s)
    if state.error is not None:
        print(state.error.get("traceback", ""), file=sys.stderr)
        raise RuntimeError(state.error["error"])
    if not state.ready.is_set() or not worker.is_alive():
        raise RuntimeError("NPU worker did not become ready")
    server = _Server((args.host, args.port), _Handler)
    server.state = state  # type: ignore[attr-defined]

    def stop_server(signum: int, frame: Any) -> None:
        del signum, frame
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    print(f"READY http://{args.host}:{args.port} worker_pid={state.worker_pid}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        try:
            state.shutdown_worker()
        except (queue.Empty, queue.Full):
            worker.terminate()
        worker.join(timeout=10.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5.0)
        state.stopping.set()
        server.server_close()


if __name__ == "__main__":
    main()
