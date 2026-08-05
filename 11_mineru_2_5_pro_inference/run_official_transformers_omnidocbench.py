#!/usr/bin/env python3
"""Run an official MinerU local client over OmniDocBench pages.

The runner supports the stock Transformers and synchronous vLLM engines while
preserving the remaining model-card contract: the official
``mineru-vl-utils`` two-step client, BF16, greedy generation, image analysis
disabled, and official ``json2md`` output.  It adds only corpus selection,
deterministic sharding, durable checkpoints, and explicit progress/timing
records.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from run_transformers_recognition_smoke import configure_npu, synchronize


DEFAULT_MODEL = Path("/workspace/models/MinerU2.5-Pro-2605-1.2B")
DEFAULT_DATASET_JSON = Path("/workspace/datasets/OmniDocBench/OmniDocBench.json")
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm-engine"),
        default="transformers",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Maximum requests per backend call. Zero keeps the official "
            "backend default: B1 for Transformers and unbounded for vLLM."
        ),
    )
    parser.add_argument(
        "--page-batch-size",
        type=int,
        default=1,
        help=(
            "Pages passed to official batch_two_step_extract at once. One uses "
            "the page-at-a-time two_step_extract path."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--hash-model-files", action="store_true")
    parser.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable vLLM graph capture for the compatibility baseline.",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=64)
    return parser.parse_args()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def image_name(sample: dict[str, Any]) -> str:
    page_info = sample.get("page_info") or {}
    value = page_info.get("image_path")
    if not value:
        raise ValueError("OmniDocBench sample has no page_info.image_path")
    return Path(value).name


def main() -> None:
    args = parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit < 0):
        raise ValueError("offset and limit must be non-negative")
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.batch_size < 0 or args.page_batch_size <= 0:
        raise ValueError("batch-size must be non-negative and page-batch-size must be positive")

    model_dir = args.model.expanduser().resolve()
    dataset_json = args.dataset_json.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    predictions_dir = output_dir / "predictions"
    content_dir = output_dir / "content_lists"
    progress_dir = output_dir / "progress"
    failures_dir = output_dir / "failures"
    for directory in (predictions_dir, content_dir, progress_dir, failures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(dataset_json.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise TypeError("OmniDocBench dataset must be a JSON list")
    stop = None if args.limit is None else args.offset + args.limit
    selected = list(enumerate(dataset))[args.offset:stop]
    shard = [item for position, item in enumerate(selected) if position % args.shard_count == args.shard_index]
    if not shard:
        raise ValueError("selection produced an empty shard")

    configure_npu()
    import torch
    import torch_npu
    import transformers
    from mineru_vl_utils import MinerUClient, __version__ as mineru_utils_version
    from mineru_vl_utils.post_process import json2md

    setup_started = time.perf_counter()
    print(
        f"[setup] shard={args.shard_index}/{args.shard_count} pages={len(shard)} "
        f"model={model_dir}",
        flush=True,
    )
    if args.backend == "transformers":
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        # The checkpoint stores the tied embedding matrix once.  Transformers
        # otherwise initializes the absent lm_head independently.
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
        model = model.to("npu:0").eval()
        processor = AutoProcessor.from_pretrained(
            model_dir,
            use_fast=True,
            local_files_only=True,
        )
        client = MinerUClient(
            backend="transformers",
            model=model,
            processor=processor,
            image_analysis=False,
            batch_size=args.batch_size,
            use_tqdm=False,
        )
        attention = "eager"
        processor_fast: bool | None = True
    else:
        from mineru_vl_utils import MinerULogitsProcessor
        from vllm import LLM

        model = LLM(
            model=str(model_dir),
            dtype="bfloat16",
            enforce_eager=args.vllm_enforce_eager,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
            limit_mm_per_prompt={"image": 1},
            logits_processors=[MinerULogitsProcessor],
        )
        client = MinerUClient(
            backend="vllm-engine",
            vllm_llm=model,
            image_analysis=False,
            batch_size=args.batch_size,
            use_tqdm=False,
        )
        attention = "vllm-selected"
        processor_fast = None
    synchronize()
    setup_s = time.perf_counter() - setup_started

    model_hashes = {
        "config.json": sha256(model_dir / "config.json"),
    }
    if args.hash_model_files:
        model_hashes["model.safetensors"] = sha256(model_dir / "model.safetensors")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "backend": f"official_mineru_{args.backend}",
        "model": str(model_dir),
        "dataset_json": str(dataset_json),
        "images_dir": str(images_dir),
        "model_hashes": model_hashes,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "transformers": transformers.__version__,
        "mineru_vl_utils": mineru_utils_version,
        "dtype": "bfloat16",
        "attention": attention,
        "processor_fast": processor_fast,
        "npu_jit_compile": False,
        "image_analysis": False,
        "batch_size": args.batch_size,
        "page_batch_size": args.page_batch_size,
        "vllm_enforce_eager": (
            args.vllm_enforce_eager if args.backend == "vllm-engine" else None
        ),
        "vllm_gpu_memory_utilization": (
            args.vllm_gpu_memory_utilization if args.backend == "vllm-engine" else None
        ),
        "vllm_max_model_len": (
            args.vllm_max_model_len if args.backend == "vllm-engine" else None
        ),
        "vllm_max_num_seqs": (
            args.vllm_max_num_seqs if args.backend == "vllm-engine" else None
        ),
        "offset": args.offset,
        "limit": args.limit,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "selected_pages": len(selected),
        "shard_pages": len(shard),
        "setup_s": setup_s,
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    }
    atomic_write_text(
        output_dir / f"run_manifest_shard_{args.shard_index:02d}.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(f"[setup] complete elapsed_s={setup_s:.3f}", flush=True)

    progress_path = output_dir / f"progress_shard_{args.shard_index:02d}.jsonl"
    shard_started = time.perf_counter()
    completed = 0
    skipped = 0
    failed = 0
    page_times: list[float] = []
    batch_times: list[float] = []

    pending: list[tuple[int, int, dict[str, Any]]] = []
    for shard_position, (dataset_index, sample) in enumerate(shard, start=1):
        name = image_name(sample)
        stem = Path(name).stem
        markdown_path = predictions_dir / f"{stem}.md"
        content_path = content_dir / f"{stem}.json"
        page_record_path = progress_dir / f"{stem}.json"
        if args.resume and markdown_path.is_file() and content_path.is_file() and page_record_path.is_file():
            skipped += 1
            print(
                f"[page {shard_position}/{len(shard)}] SKIP dataset_index={dataset_index} image={name}",
                flush=True,
            )
            continue
        pending.append((shard_position, dataset_index, sample))

    if args.page_batch_size == 1:
        page_groups = [[item] for item in pending]
    else:
        page_groups = [
            pending[start : start + args.page_batch_size]
            for start in range(0, len(pending), args.page_batch_size)
        ]

    for group_index, group in enumerate(page_groups, start=1):
        names = [image_name(sample) for _, _, sample in group]
        dataset_indices = [dataset_index for _, dataset_index, _ in group]
        print(
            f"[group {group_index}/{len(page_groups)}] START pages={len(group)} "
            f"dataset_indices={dataset_indices}",
            flush=True,
        )
        group_started = time.perf_counter()
        try:
            images = []
            for name in names:
                with Image.open(images_dir / name) as source:
                    images.append(source.convert("RGB"))
            with torch.inference_mode():
                if len(group) == 1:
                    results = [client.two_step_extract(images[0])]
                else:
                    results = client.batch_two_step_extract(images)
            synchronize()
            group_elapsed_s = time.perf_counter() - group_started
            batch_times.append(group_elapsed_s)
            for (shard_position, dataset_index, _), name, blocks in zip(group, names, results):
                stem = Path(name).stem
                markdown = json2md(blocks)
                rendered_blocks = json.dumps(list(blocks), ensure_ascii=False, indent=2) + "\n"
                type_counts = dict(sorted(collections.Counter(block["type"] for block in blocks).items()))
                record = {
                    "status": "completed",
                    "dataset_index": dataset_index,
                    "image": name,
                    "group_index": group_index,
                    "group_size": len(group),
                    "group_elapsed_s": group_elapsed_s,
                    "block_count": len(blocks),
                    "block_types": type_counts,
                    "markdown_chars": len(markdown),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_write_text(predictions_dir / f"{stem}.md", markdown)
                atomic_write_text(content_dir / f"{stem}.json", rendered_blocks)
                atomic_write_text(
                    progress_dir / f"{stem}.json",
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                )
                append_jsonl(progress_path, record)
                completed += 1
                if len(group) == 1:
                    page_times.append(group_elapsed_s)
            elapsed_total = sum(batch_times)
            remaining_pages = len(pending) - completed
            pages_per_s = completed / elapsed_total
            remaining_s = remaining_pages / pages_per_s
            print(
                f"[group {group_index}/{len(page_groups)}] DONE pages={len(group)} "
                f"elapsed_s={group_elapsed_s:.3f} pages_per_s={pages_per_s:.5f} "
                f"eta_s={remaining_s:.1f}",
                flush=True,
            )
        except Exception as error:
            elapsed_s = time.perf_counter() - group_started
            for _, dataset_index, sample in group:
                name = image_name(sample)
                stem = Path(name).stem
                failed += 1
                record = {
                    "status": "failed",
                    "dataset_index": dataset_index,
                    "image": name,
                    "group_index": group_index,
                    "group_size": len(group),
                    "group_elapsed_s": elapsed_s,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_write_text(
                    failures_dir / f"{stem}.json",
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                )
                append_jsonl(progress_path, record)
            print(
                f"[group {group_index}/{len(page_groups)}] FAIL pages={len(group)} "
                f"elapsed_s={elapsed_s:.3f} "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            if args.fail_fast:
                raise

    wall_s = time.perf_counter() - shard_started
    summary = {
        **manifest,
        "pipeline_wall_s": wall_s,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "measured_page_mean_s": sum(page_times) / len(page_times) if page_times else None,
        "measured_pages_per_s": len(page_times) / sum(page_times) if page_times else None,
        "measured_group_count": len(batch_times),
        "measured_group_wall_s": sum(batch_times),
        "measured_group_pages_per_s": completed / sum(batch_times) if batch_times else None,
    }
    summary_path = output_dir / f"run_summary_shard_{args.shard_index:02d}.json"
    atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"[summary] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
