#!/usr/bin/env python3
"""Serve the validated height-routed U8 table-speculation lane over HTTP."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import io
import multiprocessing as mp
from pathlib import Path
import queue
import signal
from types import SimpleNamespace
from typing import Any
import sys
import threading
import time
import traceback


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_TARGETS = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/table_spec_full_d1e6d00/"
    "whole/whole/tables.jsonl"
)
DEFAULT_COMPACT_VOCAB = (
    EXPERIMENT_ROOT
    / "presets/table_compact_vocab/b1_verifier_topfreq_16384.json"
)
sys.path.insert(0, str(HERE))

from serve_crop_ocr_api import _Handler, _Server, _State  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--max-image-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--queue-capacity", type=int, default=256)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/workspace/models/PaddleOCR-VL-1.6"),
    )
    parser.add_argument(
        "--decode-vocab-token-ids",
        type=Path,
        default=DEFAULT_COMPACT_VOCAB,
    )
    parser.add_argument("--height-threshold-px", type=int, default=384)
    parser.add_argument("--cold-request-id", default="page_000010_table_box_id_1")
    parser.add_argument("--draft-cache-length", type=int, default=768)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--decode-optimization",
        default="combined_apply_complete_layer_prefetch1_rope_lut",
    )
    parser.add_argument(
        "--verifier-optimization",
        default="combined_apply_spec_prefetch_mrope",
    )
    parser.add_argument("--k-values", default="7,15,31,63")
    parser.add_argument("--initial-k", type=int, default=15)
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument(
        "--decode-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair",
    )
    parser.add_argument(
        "--vision-cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / ".runtime_cache/09_persistent_page_engine_vision_torchair"
        ),
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / ".runtime_cache/09_persistent_page_engine_text_torchair"
        ),
    )
    parser.add_argument(
        "--text-packed-cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / ".runtime_cache/09_persistent_page_engine_text_packed_torchair"
        ),
    )
    return parser.parse_args()


def _live_args(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        model=Path(config["model"]),
        b1_decode_optimization=config["decode_optimization"],
        b1_decode_vocab_token_ids=Path(config["decode_vocab_token_ids"]),
        b1_cache_length=config["cache_length"],
        b1_max_new_tokens=config["max_new_tokens"],
        b1_vision_buckets="4096",
        b1_text_buckets="1152",
        draft_decode_optimization=config["decode_optimization"],
        draft_decode_vocab_token_ids=Path(config["decode_vocab_token_ids"]),
        draft_cache_length=config["draft_cache_length"],
        draft_row_count=8,
        draft_batch_size=8,
        draft_vision_packing="greedy",
        draft_vision_pack_target=2304,
        draft_prefill_layout="packed_b1",
        draft_batched_vision_shapes="8x640,8x768",
        draft_batched_text_shape="8x256",
        row_overlap_px=3,
        compact_uint8_preprocess=False,
        image_resize_backend="pillow",
        target_cpu_delay_ms=0.0,
        overlap_target_cpu_preparation=False,
        k_values=config["k_values"],
        initial_k=config["initial_k"],
        verifier_optimization=config["verifier_optimization"],
        per_call_device_timing=False,
        allow_compile=False,
        token_selection="greedy",
        min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
        vision_buckets="256,384,512,640,768,1408,1920,2048,2304,2944,4096",
        decode_cache_dir=Path(config["decode_cache_dir"]),
        vision_cache_dir=Path(config["vision_cache_dir"]),
        vision_batched_cache_dir=Path(config["vision_cache_dir"]),
        text_cache_dir=Path(config["text_cache_dir"]),
        text_packed_cache_dir=Path(config["text_packed_cache_dir"]),
        text_batched_cache_dir=Path(config["text_cache_dir"]),
        k_cache_root=[],
    )


def _worker_main(jobs: Any, results: Any, config: dict[str, Any]) -> None:
    try:
        sys.path.insert(0, str(EXPERIMENT_ROOT))
        from PIL import Image
        import torch
        import torch_npu  # noqa: F401
        import table_row_ocr_lab as row_lab
        import table_spec_adaptive_k_lab as adaptive_lab
        import table_spec_decode_lab as fixed_lab
        import table_spec_live_u8_adaptive_lab as live_lab
        from paddleocr_vl.model.text_spec_verify import (
            torchair_cache_dir_for_spec_shape,
        )
        from pipeline.layout_output import normalize_recognition_text

        torch.npu.config.allow_internal_format = True
        torch.npu.set_compile_mode(jit_compile=False)
        args = _live_args(config)
        targets = fixed_lab.read_jsonl(Path(config["targets"]))
        targets_by_id = {str(row["request_id"]): row for row in targets}

        print("TABLE_SPEC_API setup=draft_recognizer", flush=True)
        draft_recognizer = row_lab.build_recognizer(live_lab._draft_args(args))
        print("TABLE_SPEC_API setup=b1_recognizer", flush=True)
        b1_recognizer = fixed_lab.build_recognizer(live_lab._b1_args(args))
        k_values = adaptive_lab.parse_k_values(args.k_values)
        cache_roots = adaptive_lab.parse_k_cache_roots(
            [], k_values=k_values, default=args.decode_cache_dir
        )
        for value in k_values:
            spec_cache = torchair_cache_dir_for_spec_shape(
                cache_roots[value],
                draft_length=value,
                cache_length=args.b1_cache_length,
                dtype=b1_recognizer.dtype,
                device=b1_recognizer.device,
                model_dir=b1_recognizer.model_dir,
                linear_weight_format=str(
                    b1_recognizer.weight_format["effective_mode"]
                ),
                optimization=args.verifier_optimization,
                token_selection=args.token_selection,
                preferred_token_id=b1_recognizer.math_open_token_id,
                alternate_preferred_token_id=b1_recognizer.math_slash_token_id,
                cell_start_token_ids=b1_recognizer.table_cell_token_ids,
            )
            if not spec_cache.is_dir() or not any(spec_cache.iterdir()):
                raise RuntimeError(f"missing K{value} verifier cache: {spec_cache}")
        runtime = adaptive_lab.AdaptiveKTableSpeculativeDecodeRuntime(
            b1_recognizer,
            k_values=k_values,
            initial_k=args.initial_k,
            cache_roots=cache_roots,
            verifier_optimization=args.verifier_optimization,
            record_device_timing=False,
        )
        cpu_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="table-spec-api-cpu-overlap",
        )

        def run_spec(source: dict[str, Any], raw_image: Any) -> dict[str, Any]:
            return live_lab._run_one(
                source=source,
                raw_image=raw_image,
                crop_preload_wall_s=0.0,
                measured=True,
                args=args,
                b1_recognizer=b1_recognizer,
                draft_recognizer=draft_recognizer,
                runtime=runtime,
                cpu_executor=cpu_executor,
            )

        def run_b1(source: dict[str, Any], target_crop: Any) -> dict[str, Any]:
            emitted: list[Any] = []
            request = fixed_lab.request_for(source, target_crop, args)
            schedule = b1_recognizer.run(
                [request],
                schedule_id=f"table-spec-api:b1:{source['request_id']}",
                emit_result=emitted.append,
            )
            if len(emitted) != 1:
                raise RuntimeError(f"B1 route emitted {len(emitted)} results")
            recognition = emitted[0]
            payload = asdict(recognition)
            payload["raw_text"] = payload["text"]
            payload["text"] = normalize_recognition_text(
                "table", payload["raw_text"]
            )
            payload["schedule"] = asdict(schedule)
            return payload

        cold_id = config["cold_request_id"]
        if cold_id:
            cold_source = targets_by_id[cold_id]
            cold_image = live_lab.load_crop(
                cold_source, Path(config["images_dir"])
            )
            print(f"TABLE_SPEC_API cold_start={cold_id}", flush=True)
            run_spec(cold_source, cold_image)

        results.put(
            {
                "kind": "ready",
                "configuration": {
                    "lane": "height_routed_u8_adaptive_k",
                    "height_threshold_px": config["height_threshold_px"],
                    "draft_rows": 8,
                    "adaptive_k": list(k_values),
                    "initial_k": args.initial_k,
                    "decode_optimization": args.b1_decode_optimization,
                    "verifier_optimization": args.verifier_optimization,
                },
                "worker_pid": __import__("os").getpid(),
            }
        )

        while True:
            job = jobs.get()
            if job is None:
                results.put({"kind": "service_summary", "payload": {}})
                break
            internal_id = str(job["request_id"])
            try:
                if job.get("crop_type") != "table":
                    raise ValueError("speculative server accepts only table crops")
                source_id = str(job["source_request_id"])
                source = targets_by_id[source_id]
                with Image.open(io.BytesIO(job["image_bytes"])) as opened:
                    raw_image = opened.convert("RGB")
                service_started = time.perf_counter()
                target_crop = live_lab._exact_target_crop_from_raw(
                    source, raw_image
                )
                if target_crop.height < config["height_threshold_px"]:
                    route_lane = "b1"
                    full = run_b1(source, target_crop)
                    raw_text = str(full["raw_text"])
                    text = str(full["text"])
                    token_ids = list(full["token_ids"])
                    stop_reason = str(full["stop_reason"])
                    exact = token_ids == fixed_lab.target_tokens(source)
                    stage_timing = {}
                else:
                    route_lane = "spec"
                    full = run_spec(source, raw_image)
                    speculative = full["speculative"]
                    raw_text = str(speculative["text"])
                    text = str(full["pred_html"])
                    token_ids = list(speculative["token_ids"])
                    stop_reason = str(speculative["stop_reason"])
                    exact = bool(full["exact_saved_reference"])
                    stage_timing = dict(full["timing_s"])
                service_wall_s = time.perf_counter() - service_started
                results.put(
                    {
                        "kind": "result",
                        "request_id": internal_id,
                        "ok": True,
                        "payload": {
                            "request_id": source_id,
                            "crop_type": "table",
                            "route_lane": route_lane,
                            "raw_text": raw_text,
                            "text": text,
                            "token_ids": token_ids,
                            "output_tokens": len(token_ids),
                            "stop_reason": stop_reason,
                            "exact_saved_reference": exact,
                            "service_wall_s": service_wall_s,
                            "queue_wait_s": (
                                service_started
                                - float(job["submitted_monotonic_s"])
                            ),
                            "stage_timing_s": stage_timing,
                            "worker_wall_s": (
                                time.perf_counter()
                                - float(job["submitted_monotonic_s"])
                            ),
                        },
                    }
                )
            except BaseException as exc:
                results.put(
                    {
                        "kind": "result",
                        "request_id": internal_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
        cpu_executor.shutdown(wait=True)
    except BaseException as exc:
        results.put(
            {
                "kind": "startup_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def main() -> None:
    args = parse_args()
    context = mp.get_context("spawn")
    jobs = context.Queue(maxsize=args.queue_capacity)
    results = context.Queue()
    config = {
        "targets": str(args.targets.expanduser().resolve()),
        "images_dir": str(args.images_dir.expanduser().resolve()),
        "model": str(args.model.expanduser().resolve()),
        "decode_vocab_token_ids": str(
            args.decode_vocab_token_ids.expanduser().resolve()
        ),
        "height_threshold_px": args.height_threshold_px,
        "cold_request_id": args.cold_request_id,
        "draft_cache_length": args.draft_cache_length,
        "cache_length": args.cache_length,
        "max_new_tokens": args.max_new_tokens,
        "decode_optimization": args.decode_optimization,
        "verifier_optimization": args.verifier_optimization,
        "k_values": args.k_values,
        "initial_k": args.initial_k,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "decode_cache_dir": str(args.decode_cache_dir.expanduser().resolve()),
        "vision_cache_dir": str(args.vision_cache_dir.expanduser().resolve()),
        "text_cache_dir": str(args.text_cache_dir.expanduser().resolve()),
        "text_packed_cache_dir": str(
            args.text_packed_cache_dir.expanduser().resolve()
        ),
    }
    worker = context.Process(
        target=_worker_main,
        args=(jobs, results, config),
        name="table-speculative-npu-worker",
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
        try:
            state.drain()
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
