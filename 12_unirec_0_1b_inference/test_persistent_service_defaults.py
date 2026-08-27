from __future__ import annotations

import sys

import run_persistent_unirec_service_benchmark as benchmark


REQUIRED_ARGUMENTS = [
    "--openocr-root",
    "/openocr",
    "--model-path",
    "/model",
    "--layout-model",
    "/layout",
    "--input",
    "/input",
    "--output-dir",
    "/output",
    "--spool-dir",
    "/spool",
    "--layout-cache",
    "/layout-cache",
    "--vision-cache",
    "/vision-cache",
    "--decode-cache-parent",
    "/decode-cache",
]


def parse(monkeypatch, *extra: str):
    monkeypatch.setattr(
        sys,
        "argv",
        ["persistent-service", *REQUIRED_ARGUMENTS, *extra],
    )
    return benchmark.parse_args()


def test_canonical_resident_k20_defaults(monkeypatch) -> None:
    args = parse(monkeypatch)
    actual = {
        "warmup_pages": args.warmup_pages,
        "workers": args.workers,
        "recognition_threads": args.recognition_threads,
        "recognition_resize_chunk_size": args.recognition_resize_chunk_size,
        "layout_lanes": args.layout_lanes,
        "layout_batch_size": args.layout_batch_size,
        "layout_threshold": args.layout_threshold,
        "vision_bucket_preset": args.vision_bucket_preset,
        "vision_lanes": args.vision_lanes,
        "vision_graph_residency": args.vision_graph_residency,
        "require_all_warmup_vision_graphs": (
            args.require_all_warmup_vision_graphs
        ),
        "vision_same_key_shards": args.vision_same_key_shards,
        "vision_sharded_key_count": args.vision_sharded_key_count,
        "vision_record_budget": args.vision_record_budget,
        "vision_max_calls_per_key": args.vision_max_calls_per_key,
        "vision_queue_size": args.vision_queue_size,
        "vision_tall_fallback": args.vision_tall_fallback,
        "decode_batch_size": args.decode_batch_size,
        "cross_cache_length": args.cross_cache_length,
        "self_cache_length": args.self_cache_length,
        "max_length": args.max_length,
        "ready_queue_size": args.ready_queue_size,
        "progress_every": args.progress_every,
    }
    assert actual == {
        "warmup_pages": 512,
        "workers": 4,
        "recognition_threads": 8,
        "recognition_resize_chunk_size": 0,
        "layout_lanes": 1,
        "layout_batch_size": 2,
        "layout_threshold": 0.5,
        "vision_bucket_preset": "310p_k20_l4",
        "vision_lanes": 4,
        "vision_graph_residency": "all",
        "require_all_warmup_vision_graphs": True,
        "vision_same_key_shards": 1,
        "vision_sharded_key_count": 0,
        "vision_record_budget": 128,
        "vision_max_calls_per_key": 64,
        "vision_queue_size": 128,
        "vision_tall_fallback": "eager",
        "decode_batch_size": 128,
        "cross_cache_length": 1320,
        "self_cache_length": 2048,
        "max_length": 2048,
        "ready_queue_size": 128,
        "progress_every": 16,
    }


def test_small_cache_probe_can_disable_full_k20_warmup_gate(monkeypatch) -> None:
    args = parse(monkeypatch, "--no-require-all-warmup-vision-graphs")
    assert args.require_all_warmup_vision_graphs is False
