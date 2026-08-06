#!/usr/bin/env python3
"""Run an official MinerU local client over OmniDocBench pages.

The runner supports the stock Transformers and synchronous vLLM engines while
preserving the remaining model-card contract: the official
``mineru-vl-utils`` two-step client, FP16, greedy generation, image analysis
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
DEFAULT_LOCAL_TORCHAIR_CACHE_DIR = (
    Path(".runtime_cache")
    / "11_mineru_2_5_pro_inference"
    / "native_compiled_decode_b1_k8192_fp16"
)
DEFAULT_LOCAL_VISION_TORCHAIR_CACHE_DIR = (
    Path(".runtime_cache")
    / "11_mineru_2_5_pro_inference"
    / "vision_prefill_b1_fp16"
)
DEFAULT_LOCAL_VISION_BUCKETS = "384,512,768,1024,1536,2048,3072,4224,5632"
DEFAULT_LOCAL_TEXT_TORCHAIR_CACHE_DIR = (
    Path(".runtime_cache")
    / "11_mineru_2_5_pro_inference"
    / "text_prefill_packed_fp16"
)
DEFAULT_LOCAL_TEXT_BUCKETS = "128,256,512,1024"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=(
            "transformers",
            "local-correctness",
            "local-eager-client",
            "local-compiled-client",
            "local-fixed-batch-client",
            "local-continuous-client",
            "vllm-engine",
            "vllm-async-engine",
        ),
        default="transformers",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--warmup-pages",
        type=int,
        default=2,
        help=(
            "Run this many pages from the start of the selected shard before "
            "measurement, discard their outputs, and reset runtime counters. "
            "Use zero to disable warmup."
        ),
    )
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
    parser.add_argument(
        "--layout-image-size",
        type=int,
        nargs=2,
        default=(1036, 1036),
        metavar=("W", "H"),
        help="Square or rectangular image size used only for layout generation.",
    )
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help=(
            "Run and save raw layout generations plus parsed layout blocks; "
            "skip all crop recognition."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--hash-model-files", action="store_true")
    parser.add_argument(
        "--processor-min-pixels",
        type=int,
        help=(
            "Override the Qwen2-VL image processor minimum pixel area. "
            "By default, use the model checkpoint value."
        ),
    )
    parser.add_argument(
        "--local-dtype",
        choices=("float16",),
        default="float16",
        help=(
            "Model dtype for local MinerU backends. Use float16 for the "
            "PromptFA path shared by 910B and 310P."
        ),
    )
    parser.add_argument("--local-compiled-cache-length", type=int, default=8192)
    parser.add_argument(
        "--local-decode-attention",
        choices=("manual", "increfa"),
        default="manual",
        help="Attention implementation used only by static one-token decode.",
    )
    parser.add_argument(
        "--local-decode-weight-format",
        choices=("none", "decode_nz"),
        default="none",
        help="Weight layout used only by static one-token decode.",
    )
    parser.add_argument(
        "--local-decode-rotary-impl",
        choices=("manual", "npu_apply"),
        default="manual",
        help="RoPE implementation used only by static one-token decode.",
    )
    parser.add_argument(
        "--local-prepare-prefetch-depth",
        type=int,
        default=16,
        help=(
            "Continuous local lane only: maximum CPU-prepared requests kept "
            "ahead of slot admission. Zero disables background preparation."
        ),
    )
    parser.add_argument(
        "--local-prefill-metrics",
        action="store_true",
        help="Record opt-in NPU-event timings and token counts for local prefill.",
    )
    parser.add_argument(
        "--local-text-backend",
        choices=("eager", "torchair-packed"),
        default="eager",
        help="Run text prefill per request eagerly or through packed static graphs.",
    )
    parser.add_argument(
        "--local-text-buckets",
        default=DEFAULT_LOCAL_TEXT_BUCKETS,
        help="Comma-separated physical token lengths for packed text-prefill graphs.",
    )
    parser.add_argument("--local-text-max-members", type=int, default=32)
    parser.add_argument(
        "--local-text-torchair-cache-dir",
        type=Path,
        default=DEFAULT_LOCAL_TEXT_TORCHAIR_CACHE_DIR,
    )
    parser.add_argument(
        "--local-vision-attention",
        choices=("manual", "prompt_flash_attention"),
        default="manual",
        help="Vision attention implementation for local MinerU backends.",
    )
    parser.add_argument(
        "--local-vision-backend",
        choices=("eager", "torchair"),
        default="eager",
        help="Run B=1 vision transformer blocks eagerly or through padded static TorchAir graphs.",
    )
    parser.add_argument(
        "--local-vision-buckets",
        default=DEFAULT_LOCAL_VISION_BUCKETS,
        help="Comma-separated physical sequence lengths for compiled B=1 vision prefill.",
    )
    parser.add_argument(
        "--local-vision-torchair-cache-dir",
        type=Path,
        default=DEFAULT_LOCAL_VISION_TORCHAIR_CACHE_DIR,
    )
    parser.add_argument(
        "--local-vision-pack-target",
        type=int,
        default=768,
        help=(
            "Continuous packed-text lane only: physical B=1 bucket used to "
            "pack independent vision sequences. Single-member groups retain "
            "their ordinary smallest-bucket path."
        ),
    )
    parser.add_argument(
        "--local-torchair-cache-dir",
        type=Path,
        default=DEFAULT_LOCAL_TORCHAIR_CACHE_DIR,
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable vLLM graph capture for the compatibility baseline.",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=64)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int)
    parser.add_argument(
        "--vllm-full-decode-only",
        action="store_true",
        help="Capture only pure-decode batches with FULL_DECODE_ONLY ACLGraph.",
    )
    parser.add_argument(
        "--vllm-cudagraph-capture-sizes",
        default="1,2,4,8,16,32,64,128",
        help="Comma-separated batch sizes captured by FULL_DECODE_ONLY.",
    )
    return parser.parse_args()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Some OmniDocBench stems already approach the filesystem's 255-byte
    # component limit.  Repeating the destination name in the temporary file
    # can therefore fail even though the final path itself is valid.
    name_digest = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
    temporary = path.with_name(f".tmp-{os.getpid()}-{name_digest}")
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


def run_page_group(client: Any, images: list[Image.Image]) -> list[Any]:
    if len(images) == 1:
        return [client.two_step_extract(images[0])]
    return client.batch_two_step_extract(images)


def run_layout_group(client: Any, images: list[Image.Image]) -> list[dict[str, Any]]:
    """Run the same layout-generation path as MinerU while retaining raw text."""

    layout_images = client.helper.batch_prepare_for_layout(client.executor, images)
    prompt = client.prompts.get("[layout]") or client.prompts["[default]"]
    params = client.sampling_params.get("[layout]") or client.sampling_params.get(
        "[default]"
    )
    outputs = client._batch_predict(layout_images, prompt, params, None, False)
    texts = [output.text for output in outputs]
    blocks = client.helper.batch_parse_layout_output(client.executor, texts)
    return [
        {"raw_text": text, "blocks": page_blocks}
        for text, page_blocks in zip(texts, blocks)
    ]


def reset_measurement_counters(
    client: Any,
    vision_runtime: Any | None,
    text_runtime: Any | None,
) -> None:
    generation_metrics = getattr(client.client, "generation_metrics", None)
    if generation_metrics is not None:
        generation_metrics.clear()
    if vision_runtime is not None:
        vision_runtime.route_counts.clear()
        vision_runtime.real_tokens = 0
        vision_runtime.physical_tokens = 0
    if text_runtime is not None:
        text_runtime.route_counts.clear()
        text_runtime.real_tokens = 0
        text_runtime.physical_tokens = 0
        text_runtime.cache_copy_bytes = 0
        text_runtime.pack_count = 0


def install_bucket_input_recorder(runtime: Any, input_count: int, synchronize_fn: Any):
    """Capture real graph inputs while preserving the cache-backed callable."""

    original = runtime._compiled_for_bucket
    state: dict[str, Any] = {
        "seen": set(),
        "largest_bucket": -1,
        "inputs": None,
        "first_call_s": {},
    }

    def recorded_compiled_for_bucket(bucket: int):
        compiled = original(bucket)

        def recorded_call(*args: Any):
            first_call = bucket not in state["seen"]
            if first_call:
                synchronize_fn()
                started = time.perf_counter()
            output = compiled(*args)
            if first_call:
                synchronize_fn()
                elapsed_s = time.perf_counter() - started
                state["seen"].add(bucket)
                state["first_call_s"][str(bucket)] = elapsed_s
                record = runtime.compile_records.get(str(bucket))
                if record is not None and record.get("first_call_s") is None:
                    record["first_call_s"] = elapsed_s
                if bucket > state["largest_bucket"]:
                    state["largest_bucket"] = bucket
                    state["inputs"] = tuple(args[:input_count])
            return output

        return recorded_call

    runtime._compiled_for_bucket = recorded_compiled_for_bucket
    return original, state


def resize_vision_graph_inputs(
    torch: Any,
    inputs: tuple[Any, ...],
    target: int,
) -> tuple[Any, ...]:
    hidden, rope_cos, rope_sin, mask = inputs
    source = int(hidden.shape[1])
    copied = min(source, target)

    resized_hidden = hidden.new_zeros((hidden.shape[0], target, *hidden.shape[2:]))
    resized_hidden[:, :copied].copy_(hidden[:, :copied])
    resized_cos = rope_cos.new_ones((rope_cos.shape[0], target, *rope_cos.shape[2:]))
    resized_cos[:, :copied].copy_(rope_cos[:, :copied])
    resized_sin = rope_sin.new_zeros((rope_sin.shape[0], target, *rope_sin.shape[2:]))
    resized_sin[:, :copied].copy_(rope_sin[:, :copied])

    resized_mask = torch.ones(
        (mask.shape[0], mask.shape[1], target, target),
        device=mask.device,
        dtype=mask.dtype,
    )
    resized_mask[:, :, :copied, :copied].copy_(mask[:, :, :copied, :copied])
    if target > source:
        resized_mask[:, :, source:, source:] = False
    return resized_hidden, resized_cos, resized_sin, resized_mask.contiguous()


def resize_text_graph_inputs(
    inputs: tuple[Any, ...],
    target: int,
) -> tuple[Any, ...]:
    inputs_embeds, position_ids, segment_ids, local_positions = inputs
    source = int(inputs_embeds.shape[1])
    copied = min(source, target)

    resized_embeds = inputs_embeds.new_zeros(
        (inputs_embeds.shape[0], target, *inputs_embeds.shape[2:])
    )
    resized_embeds[:, :copied].copy_(inputs_embeds[:, :copied])
    resized_positions = position_ids.new_ones(
        (*position_ids.shape[:-1], target)
    )
    resized_positions[..., :copied].copy_(position_ids[..., :copied])
    resized_segments = segment_ids.new_full((target,), -1)
    resized_segments[:copied].copy_(segment_ids[:copied])
    resized_local_positions = local_positions.new_zeros((target,))
    resized_local_positions[:copied].copy_(local_positions[:copied])
    return (
        resized_embeds.contiguous(),
        resized_positions.contiguous(),
        resized_segments.contiguous(),
        resized_local_positions.contiguous(),
    )


def warm_all_static_buckets(
    torch: Any,
    synchronize_fn: Any,
    vision_runtime: Any | None,
    vision_original: Any | None,
    vision_state: dict[str, Any] | None,
    text_runtime: Any | None,
    text_original: Any | None,
    text_state: dict[str, Any] | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if vision_runtime is not None:
        if (
            vision_original is None
            or vision_state is None
            or vision_state["inputs"] is None
        ):
            raise RuntimeError("two-page warmup captured no compiled vision graph inputs")
        bucket_report: dict[str, Any] = {}
        for bucket in vision_runtime.buckets:
            if bucket in vision_state["seen"]:
                bucket_report[str(bucket)] = {
                    "source": "real_page_execution",
                    "first_call_s": vision_state["first_call_s"][str(bucket)],
                }
                continue
            compiled = vision_original(bucket)
            graph_inputs = resize_vision_graph_inputs(
                torch, vision_state["inputs"], bucket
            )
            synchronize_fn()
            started = time.perf_counter()
            compiled(*graph_inputs)
            synchronize_fn()
            elapsed_s = time.perf_counter() - started
            vision_runtime.compile_records[str(bucket)]["first_call_s"] = elapsed_s
            vision_state["seen"].add(bucket)
            bucket_report[str(bucket)] = {
                "source": "real_page_tensor_replay",
                "first_call_s": elapsed_s,
            }
        uncovered = [
            bucket
            for bucket in vision_runtime.buckets
            if vision_runtime.compile_records.get(str(bucket), {}).get("first_call_s")
            is None
        ]
        if uncovered:
            raise RuntimeError(f"vision warmup missed buckets: {uncovered}")
        report["vision"] = bucket_report

    if text_runtime is not None:
        if (
            text_original is None
            or text_state is None
            or text_state["inputs"] is None
        ):
            raise RuntimeError("two-page warmup captured no compiled text graph inputs")
        bucket_report = {}
        for bucket in text_runtime.buckets:
            if bucket in text_state["seen"]:
                bucket_report[str(bucket)] = {
                    "source": "real_page_execution",
                    "first_call_s": text_state["first_call_s"][str(bucket)],
                }
                continue
            compiled = text_original(bucket)
            graph_inputs = resize_text_graph_inputs(text_state["inputs"], bucket)
            scratch = text_runtime.scratch_caches[bucket]
            synchronize_fn()
            started = time.perf_counter()
            compiled(*graph_inputs, *scratch.flat_tensors())
            synchronize_fn()
            elapsed_s = time.perf_counter() - started
            text_runtime.compile_records[str(bucket)]["first_call_s"] = elapsed_s
            text_state["seen"].add(bucket)
            bucket_report[str(bucket)] = {
                "source": "real_page_tensor_replay",
                "first_call_s": elapsed_s,
            }
        uncovered = [
            bucket
            for bucket in text_runtime.buckets
            if text_runtime.compile_records.get(str(bucket), {}).get("first_call_s")
            is None
        ]
        if uncovered:
            raise RuntimeError(f"text warmup missed buckets: {uncovered}")
        report["text"] = bucket_report
    return report


def main() -> None:
    args = parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit < 0):
        raise ValueError("offset and limit must be non-negative")
    if args.warmup_pages < 0:
        raise ValueError("warmup-pages must be non-negative")
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.batch_size < 0 or args.page_batch_size <= 0:
        raise ValueError("batch-size must be non-negative and page-batch-size must be positive")
    if any(int(value) <= 0 for value in args.layout_image_size):
        raise ValueError("layout-image-size values must be positive")
    if args.local_prepare_prefetch_depth < 0:
        raise ValueError("local-prepare-prefetch-depth must be non-negative")
    capture_sizes = [
        int(value)
        for value in args.vllm_cudagraph_capture_sizes.split(",")
        if value.strip()
    ]
    if any(value <= 0 for value in capture_sizes):
        raise ValueError("vllm-cudagraph-capture-sizes must contain positive integers")
    if args.vllm_full_decode_only and args.vllm_enforce_eager:
        raise ValueError("FULL_DECODE_ONLY requires --no-vllm-enforce-eager")

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
    torch.npu.config.allow_internal_format = True
    from mineru_vl_utils import MinerUClient, __version__ as mineru_utils_version
    from mineru_vl_utils.post_process import json2md

    setup_started = time.perf_counter()
    local_vision_runtime = None
    local_text_runtime = None
    local_decode_setup = None
    print(
        f"[setup] shard={args.shard_index}/{args.shard_count} pages={len(shard)} "
        f"model={model_dir}",
        flush=True,
    )
    if args.backend in (
        "transformers",
        "local-correctness",
        "local-eager-client",
        "local-compiled-client",
        "local-fixed-batch-client",
        "local-continuous-client",
    ):
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_dir,
            use_fast=True,
            local_files_only=True,
        )
        if args.processor_min_pixels is not None:
            if args.processor_min_pixels <= 0:
                raise ValueError("processor-min-pixels must be positive")
            processor.image_processor.min_pixels = args.processor_min_pixels
            processor.image_processor.size["shortest_edge"] = (
                args.processor_min_pixels
            )
        if args.backend == "transformers":
            from transformers import Qwen2VLForConditionalGeneration

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_dir,
                dtype=torch.float16,
                attn_implementation="eager",
                local_files_only=True,
            )
            # The checkpoint stores the tied embedding matrix once.
            model.lm_head.weight = model.model.language_model.embed_tokens.weight
            model = model.to("npu:0").eval()
        else:
            if (
                args.backend
                not in ("local-fixed-batch-client", "local-continuous-client")
                and args.batch_size not in (0, 1)
            ):
                raise ValueError("local MinerU lanes currently require --batch-size 1")
            if (
                args.backend in ("local-fixed-batch-client", "local-continuous-client")
                and args.batch_size <= 1
            ):
                raise ValueError(
                    "local batched MinerU clients require --batch-size > 1"
                )
            from local_modeling_mineru import (
                LocalMinerU2_5ForConditionalGeneration,
                configure_decode_attention_impl,
                configure_decode_packed_projections,
                configure_decode_rotary_impl,
                configure_decode_weight_format,
            )
            from native_custom_backend import (
                LocalMinerUGenerateAdapter,
                make_local_compiled_vlm_client,
                make_local_eager_vlm_client,
                make_local_fixed_batch_vlm_client,
            )

            local_dtype = torch.float16
            local_model = LocalMinerU2_5ForConditionalGeneration.from_pretrained(
                model_dir,
                dtype=local_dtype,
                device="npu:0",
            )
            if args.backend in (
                "local-compiled-client",
                "local-fixed-batch-client",
                "local-continuous-client",
            ):
                decode_setup_started = time.perf_counter()
                packed_projections = configure_decode_packed_projections(local_model)
                decode_weight_format = configure_decode_weight_format(
                    local_model, args.local_decode_weight_format
                )
                decode_rotary_impl = configure_decode_rotary_impl(
                    local_model, args.local_decode_rotary_impl
                )
                decode_attention = configure_decode_attention_impl(
                    local_model, args.local_decode_attention
                )
                synchronize()
                local_decode_setup = {
                    "packed_projections": packed_projections,
                    "weight_format": decode_weight_format,
                    "rotary_impl": decode_rotary_impl,
                    "attention": decode_attention,
                    "setup_s": time.perf_counter() - decode_setup_started,
                }
            local_model.set_vision_attention_impl(args.local_vision_attention)
            if args.local_vision_backend == "torchair":
                if args.local_vision_attention != "prompt_flash_attention":
                    raise ValueError(
                        "compiled MinerU vision prefill currently requires "
                        "--local-vision-attention prompt_flash_attention"
                    )
                from vision_prefill_compile import MinerUVisionPrefillRuntime

                local_vision_runtime = MinerUVisionPrefillRuntime(
                    local_model.visual,
                    buckets=args.local_vision_buckets,
                    cache_root=args.local_vision_torchair_cache_dir,
                    model_dir=model_dir,
                    device=local_model.device,
                    dtype=local_dtype,
                )
                local_model.set_vision_prefill_runtime(local_vision_runtime)
            model = LocalMinerUGenerateAdapter(local_model)
        client = MinerUClient(
            backend="transformers",
            model=model,
            processor=processor,
            image_analysis=False,
            layout_image_size=tuple(int(value) for value in args.layout_image_size),
            batch_size=args.batch_size,
            use_tqdm=False,
        )
        if args.backend == "local-eager-client":
            client.client = make_local_eager_vlm_client(
                local_model,
                processor,
                batch_size=max(1, args.batch_size),
                system_prompt=client.client.system_prompt,
                allow_truncated_content=client.client.allow_truncated_content,
            )
        elif args.backend == "local-compiled-client":
            if args.local_compiled_cache_length <= 0:
                raise ValueError("local-compiled-cache-length must be positive")
            from run_local_model_two_step_extract import (
                CompiledSingleBatchRecognitionDecoder,
            )

            compiled_decoder = CompiledSingleBatchRecognitionDecoder(
                local_model,
                cache_root=args.local_torchair_cache_dir,
                cache_length=args.local_compiled_cache_length,
                decode_weight_format=args.local_decode_weight_format,
                decode_rotary_impl=args.local_decode_rotary_impl,
                decode_attention_impl=args.local_decode_attention,
            )
            client.client = make_local_compiled_vlm_client(
                local_model,
                processor,
                compiled_decoder,
                batch_size=max(1, args.batch_size),
                system_prompt=client.client.system_prompt,
                allow_truncated_content=client.client.allow_truncated_content,
            )
        elif args.backend in ("local-fixed-batch-client", "local-continuous-client"):
            if args.local_compiled_cache_length <= 0:
                raise ValueError("local-compiled-cache-length must be positive")
            from fixed_batch_engine import (
                ContinuousBatchDecodeEngine,
                FixedBatchDecodeEngine,
            )
            from run_local_model_two_step_extract import (
                CompiledSingleBatchRecognitionDecoder,
            )

            compiled_decoder = CompiledSingleBatchRecognitionDecoder(
                local_model,
                cache_root=args.local_torchair_cache_dir,
                cache_length=args.local_compiled_cache_length,
                decode_weight_format=args.local_decode_weight_format,
                decode_rotary_impl=args.local_decode_rotary_impl,
                decode_attention_impl=args.local_decode_attention,
            )
            if args.local_text_backend == "torchair-packed":
                if args.backend != "local-continuous-client":
                    raise ValueError(
                        "packed text prefill currently requires local-continuous-client"
                    )
                from text_prefill_compile import MinerUPackedTextPrefillRuntime

                local_text_runtime = MinerUPackedTextPrefillRuntime(
                    local_model,
                    buckets=args.local_text_buckets,
                    max_members=args.local_text_max_members,
                    cache_root=args.local_text_torchair_cache_dir,
                    model_dir=model_dir,
                    device=local_model.device,
                    dtype=local_dtype,
                )
            engine_cls = (
                ContinuousBatchDecodeEngine
                if args.backend == "local-continuous-client"
                else FixedBatchDecodeEngine
            )
            engine = engine_cls(
                local_model,
                compiled_decoder,
                batch_size=args.batch_size,
                cache_length=args.local_compiled_cache_length,
                eos_token_id=local_model.config.eos_token_id,
                pad_token_id=local_model.config.pad_token_id,
                collect_prefill_metrics=args.local_prefill_metrics,
                packed_text_prefill_runtime=local_text_runtime,
                vision_pack_target=args.local_vision_pack_target,
            )
            client.client = make_local_fixed_batch_vlm_client(
                local_model,
                processor,
                engine,
                batch_size=args.batch_size,
                continuous_refill=args.backend == "local-continuous-client",
                prepare_prefetch_depth=(
                    args.local_prepare_prefetch_depth
                    if args.backend == "local-continuous-client"
                    else 0
                ),
                system_prompt=client.client.system_prompt,
                allow_truncated_content=client.client.allow_truncated_content,
            )
        attention = (
            f"{args.local_vision_attention}-{args.local_vision_backend}-prefill-torchair-static-decode"
            if args.backend
            in (
                "local-compiled-client",
                "local-fixed-batch-client",
                "local-continuous-client",
            )
            else f"{args.local_vision_attention}-local"
            if args.backend.startswith("local-")
            else "eager"
        )
        processor_fast: bool | None = True
    else:
        from mineru_vl_utils import MinerULogitsProcessor
        compilation_config = None
        if args.vllm_full_decode_only:
            compilation_config = {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": capture_sizes,
            }
        engine_kwargs = dict(
            model=str(model_dir),
            dtype="float16",
            enforce_eager=args.vllm_enforce_eager,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            limit_mm_per_prompt={"image": 1},
            logits_processors=[MinerULogitsProcessor],
            # The checkpoint stores one tied embedding matrix and omits a
            # separate lm_head.  Qwen2-VL's root config does not expose that
            # tie, so vLLM otherwise leaves lm_head unloaded and emits uniform
            # logits.  Set both config levels before model construction.
            hf_overrides={
                "tie_word_embeddings": True,
                "text_config": {"tie_word_embeddings": True},
            },
        )
        if compilation_config is not None:
            engine_kwargs["compilation_config"] = compilation_config

        if args.backend == "vllm-engine":
            from mineru_vl_utils.vlm_client.vllm_engine_client import (
                VllmEngineVlmClient,
            )
            from vllm import LLM

            # mineru-vl-utils 1.0.5 sends an already-valid local vLLM
            # multimodal request through the renderer a second time.
            def _keep_raw_vllm_prompts(self, raw_prompts):
                return raw_prompts

            VllmEngineVlmClient._render_vllm_cmpl_inputs = _keep_raw_vllm_prompts
            model = LLM(**engine_kwargs)
            client = MinerUClient(
                backend="vllm-engine",
                vllm_llm=model,
                image_analysis=False,
                layout_image_size=tuple(int(value) for value in args.layout_image_size),
                batch_size=args.batch_size,
                use_tqdm=False,
            )
        else:
            from mineru_vl_utils.vlm_client.vllm_async_engine_client import (
                VllmAsyncEngineVlmClient,
            )
            from vllm import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            async def _keep_raw_vllm_prompt(self, raw_prompt):
                return raw_prompt

            VllmAsyncEngineVlmClient._render_vllm_cmpl_input = (
                _keep_raw_vllm_prompt
            )
            model = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))
            client = MinerUClient(
                backend="vllm-async-engine",
                vllm_async_llm=model,
                image_analysis=False,
                layout_image_size=tuple(int(value) for value in args.layout_image_size),
                batch_size=args.batch_size,
                max_concurrency=args.vllm_max_num_seqs,
                use_tqdm=False,
            )
        attention = "vllm-selected"
        processor_fast = None
    synchronize()
    setup_s = time.perf_counter() - setup_started

    warmup_count = min(args.warmup_pages, len(shard))
    warmup_report: dict[str, Any] = {
        "requested_pages": args.warmup_pages,
        "executed_pages": warmup_count,
        "dataset_indices": [index for index, _ in shard[:warmup_count]],
        "wall_s": 0.0,
        "measurement_counters_reset": False,
    }
    if warmup_count:
        print(
            f"[warmup] START pages={warmup_count} "
            f"dataset_indices={warmup_report['dataset_indices']}",
            flush=True,
        )
        warmup_started = time.perf_counter()
        warmup_items = shard[:warmup_count]
        vision_original = vision_state = None
        text_original = text_state = None
        if local_vision_runtime is not None:
            vision_original, vision_state = install_bucket_input_recorder(
                local_vision_runtime, 4, synchronize
            )
        if local_text_runtime is not None and not args.layout_only:
            text_original, text_state = install_bucket_input_recorder(
                local_text_runtime, 4, synchronize
            )
        page_warmup_started = time.perf_counter()
        try:
            for start in range(0, warmup_count, args.page_batch_size):
                warmup_group = warmup_items[start : start + args.page_batch_size]
                warmup_images: list[Image.Image] = []
                for _, sample in warmup_group:
                    with Image.open(images_dir / image_name(sample)) as source:
                        warmup_images.append(source.convert("RGB"))
                with torch.inference_mode():
                    if args.layout_only:
                        run_layout_group(client, warmup_images)
                    else:
                        run_page_group(client, warmup_images)
        finally:
            if local_vision_runtime is not None:
                local_vision_runtime._compiled_for_bucket = vision_original
            if local_text_runtime is not None and not args.layout_only:
                local_text_runtime._compiled_for_bucket = text_original
        synchronize()
        warmup_report["real_page_wall_s"] = (
            time.perf_counter() - page_warmup_started
        )
        bucket_warmup_started = time.perf_counter()
        with torch.inference_mode():
            warmup_report["static_bucket_warmup"] = warm_all_static_buckets(
                torch,
                synchronize,
                local_vision_runtime,
                vision_original,
                vision_state,
                None if args.layout_only else local_text_runtime,
                text_original,
                text_state,
            )
        warmup_report["static_bucket_wall_s"] = (
            time.perf_counter() - bucket_warmup_started
        )
        warmup_report["wall_s"] = time.perf_counter() - warmup_started
        generation_metrics = getattr(client.client, "generation_metrics", None)
        warmup_report["generation_calls"] = (
            len(generation_metrics) if generation_metrics is not None else None
        )
        if local_vision_runtime is not None:
            warmup_report["vision_runtime"] = local_vision_runtime.metadata()
        if local_text_runtime is not None:
            warmup_report["text_runtime"] = local_text_runtime.metadata()
        reset_measurement_counters(client, local_vision_runtime, local_text_runtime)
        warmup_report["measurement_counters_reset"] = True
        print(
            f"[warmup] DONE pages={warmup_count} "
            f"elapsed_s={warmup_report['wall_s']:.3f}; outputs discarded; "
            "measurement counters reset",
            flush=True,
        )

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
        "dtype": "float16",
        "attention": attention,
        "processor_fast": processor_fast,
        "processor_min_pixels": (
            int(processor.image_processor.size["shortest_edge"])
            if processor_fast
            else None
        ),
        "npu_jit_compile": False,
        "image_analysis": False,
        "batch_size": args.batch_size,
        "page_batch_size": args.page_batch_size,
        "layout_image_size": [int(value) for value in args.layout_image_size],
        "layout_only": bool(args.layout_only),
        "local_compiled_cache_length": (
            args.local_compiled_cache_length
            if args.backend
            in (
                "local-compiled-client",
                "local-fixed-batch-client",
                "local-continuous-client",
            )
            else None
        ),
        "local_decode_attention": (
            args.local_decode_attention if local_decode_setup is not None else None
        ),
        "local_decode_weight_format": (
            args.local_decode_weight_format if local_decode_setup is not None else None
        ),
        "local_decode_rotary_impl": (
            args.local_decode_rotary_impl if local_decode_setup is not None else None
        ),
        "local_decode_setup": local_decode_setup,
        "local_prepare_prefetch_depth": (
            args.local_prepare_prefetch_depth
            if args.backend == "local-continuous-client"
            else None
        ),
        "local_prefill_metrics": (
            args.local_prefill_metrics
            if args.backend == "local-continuous-client"
            else None
        ),
        "local_text_backend": (
            args.local_text_backend if args.backend == "local-continuous-client" else None
        ),
        "local_text_buckets": (
            args.local_text_buckets
            if args.backend == "local-continuous-client"
            and args.local_text_backend == "torchair-packed"
            else None
        ),
        "local_text_max_members": (
            args.local_text_max_members
            if args.backend == "local-continuous-client"
            and args.local_text_backend == "torchair-packed"
            else None
        ),
        "local_text_torchair_cache_dir": (
            str(args.local_text_torchair_cache_dir)
            if args.backend == "local-continuous-client"
            and args.local_text_backend == "torchair-packed"
            else None
        ),
        "local_vision_attention": (
            args.local_vision_attention if args.backend.startswith("local-") else None
        ),
        "local_vision_backend": (
            args.local_vision_backend if args.backend.startswith("local-") else None
        ),
        "local_vision_buckets": (
            args.local_vision_buckets if args.backend.startswith("local-") else None
        ),
        "local_vision_torchair_cache_dir": (
            str(args.local_vision_torchair_cache_dir)
            if args.backend.startswith("local-") and args.local_vision_backend == "torchair"
            else None
        ),
        "local_torchair_cache_dir": (
            str(args.local_torchair_cache_dir)
            if args.backend
            in (
                "local-compiled-client",
                "local-fixed-batch-client",
                "local-continuous-client",
            )
            else None
        ),
        "vllm_enforce_eager": (
            args.vllm_enforce_eager if args.backend.startswith("vllm-") else None
        ),
        "vllm_gpu_memory_utilization": (
            args.vllm_gpu_memory_utilization
            if args.backend.startswith("vllm-")
            else None
        ),
        "vllm_max_model_len": (
            args.vllm_max_model_len if args.backend.startswith("vllm-") else None
        ),
        "vllm_max_num_seqs": (
            args.vllm_max_num_seqs if args.backend.startswith("vllm-") else None
        ),
        "vllm_max_num_batched_tokens": (
            args.vllm_max_num_batched_tokens
            if args.backend.startswith("vllm-")
            else None
        ),
        "vllm_full_decode_only": (
            args.vllm_full_decode_only if args.backend.startswith("vllm-") else None
        ),
        "vllm_cudagraph_capture_sizes": (
            capture_sizes
            if args.backend.startswith("vllm-") and args.vllm_full_decode_only
            else None
        ),
        "vllm_force_tied_embeddings": args.backend.startswith("vllm-"),
        "vllm_raw_multimodal_prompts": args.backend.startswith("vllm-"),
        "offset": args.offset,
        "limit": args.limit,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "selected_pages": len(selected),
        "shard_pages": len(shard),
        "setup_s": setup_s,
        "warmup": warmup_report,
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
                results = (
                    run_layout_group(client, images)
                    if args.layout_only
                    else run_page_group(client, images)
                )
            synchronize()
            group_elapsed_s = time.perf_counter() - group_started
            batch_times.append(group_elapsed_s)
            for (shard_position, dataset_index, _), name, result in zip(group, names, results):
                stem = Path(name).stem
                if args.layout_only:
                    blocks = result["blocks"]
                    markdown = result["raw_text"]
                else:
                    blocks = result
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
    if local_vision_runtime is not None:
        summary["local_compiled_vision"] = local_vision_runtime.metadata()
    if local_text_runtime is not None:
        summary["local_compiled_text_prefill"] = local_text_runtime.metadata()
    generation_metrics = getattr(client.client, "generation_metrics", None)
    if generation_metrics is not None:
        decode_calls = sum(int(item["decode_calls"]) for item in generation_metrics)
        raw_decode_token_slots = sum(
            int(item.get("raw_decode_token_slots", item["decode_calls"]))
            for item in generation_metrics
        )
        decode_s = sum(float(item["decode_s"]) for item in generation_metrics)
        active_decode_token_slots = sum(
            int(item.get("active_decode_token_slots", item["decode_calls"]))
            for item in generation_metrics
        )
        summary["local_compiled_generation"] = {
            "calls": len(generation_metrics),
            "decode_calls": decode_calls,
            "raw_decode_token_slots": raw_decode_token_slots,
            "active_decode_token_slots": active_decode_token_slots,
            "idle_decode_token_slots": (
                raw_decode_token_slots - active_decode_token_slots
            ),
            "active_slot_fraction": (
                active_decode_token_slots / raw_decode_token_slots
                if raw_decode_token_slots > 0
                else 0.0
            ),
            "refill_count": sum(
                int(item.get("refill_count", 0)) for item in generation_metrics
            ),
            "prepare_prefetch_depth": max(
                (int(item.get("prepare_prefetch_depth", 0)) for item in generation_metrics),
                default=0,
            ),
            "cpu_prepare_worker_s": sum(
                float(item.get("cpu_prepare_worker_s", 0.0))
                for item in generation_metrics
            ),
            "cpu_prepare_wait_s": sum(
                float(item.get("cpu_prepare_wait_s", 0.0))
                for item in generation_metrics
            ),
            "request_h2d_submit_s": sum(
                float(item.get("request_h2d_submit_s", 0.0))
                for item in generation_metrics
            ),
            "decode_s": decode_s,
            "decode_tok_s": decode_calls / decode_s if decode_s > 0 else 0.0,
            "raw_decode_tok_s": (
                raw_decode_token_slots / decode_s if decode_s > 0 else 0.0
            ),
            "prefill_s": sum(float(item["prefill_s"]) for item in generation_metrics),
            "generation_wall_s": sum(
                float(
                    item.get(
                        "generation_wall_s",
                        float(item.get("prefill_s", 0.0))
                        + float(item.get("decode_s", 0.0)),
                    )
                )
                for item in generation_metrics
            ),
            "compile_wrapper_s": sum(
                float(item["compile_wrapper_s"])
                for item in generation_metrics
                if item.get("compile_warmup", {}).get("ran_this_call")
            ),
            "compiled_first_call_s": sum(
                float(item["compiled_first_call_s"])
                for item in generation_metrics
                if item.get("compile_warmup", {}).get("ran_this_call")
            ),
            "phase_calls": [
                {
                    "call_index": call_index,
                    "group_index": call_index // 2,
                    "phase": "layout" if call_index % 2 == 0 else "recognition",
                    "request_count": int(item.get("request_count", 0)),
                    "graph_calls": int(item.get("graph_calls", 0)),
                    "decode_calls": int(item.get("decode_calls", 0)),
                    "raw_decode_token_slots": int(
                        item.get("raw_decode_token_slots", item.get("decode_calls", 0))
                    ),
                    "active_decode_token_slots": int(
                        item.get("active_decode_token_slots", item.get("decode_calls", 0))
                    ),
                    "prefill_s": float(item.get("prefill_s", 0.0)),
                    "decode_s": float(item.get("decode_s", 0.0)),
                    "generation_wall_s": float(item.get("generation_wall_s", 0.0)),
                    "cpu_prepare_worker_s": float(
                        item.get("cpu_prepare_worker_s", 0.0)
                    ),
                    "cpu_prepare_wait_s": float(
                        item.get("cpu_prepare_wait_s", 0.0)
                    ),
                    "request_h2d_submit_s": float(
                        item.get("request_h2d_submit_s", 0.0)
                    ),
                    "prefill_metrics": item.get("prefill_metrics", {}),
                }
                for call_index, item in enumerate(generation_metrics)
            ],
        }
        prefill_metrics: dict[str, float | int] = {}
        for item in generation_metrics:
            for name, value in item.get("prefill_metrics", {}).items():
                prefill_metrics[name] = prefill_metrics.get(name, 0) + value
        if prefill_metrics:
            vision_s = float(prefill_metrics.get("vision_tower_and_merger", 0.0))
            text_s = float(prefill_metrics.get("text_transformer_prefill", 0.0))
            raw_vision_tokens = int(prefill_metrics.get("raw_vision_tokens", 0))
            merged_vision_tokens = int(prefill_metrics.get("merged_vision_tokens", 0))
            text_prefill_tokens = int(prefill_metrics.get("text_prefill_tokens", 0))
            physical_text_prefill_tokens = int(
                prefill_metrics.get("physical_text_prefill_tokens", 0)
            )
            prefill_metrics.update(
                {
                    "raw_vision_tok_s": raw_vision_tokens / vision_s if vision_s > 0 else 0.0,
                    "merged_vision_tok_s": merged_vision_tokens / vision_s if vision_s > 0 else 0.0,
                    "text_prefill_tok_s": text_prefill_tokens / text_s if text_s > 0 else 0.0,
                    "physical_text_prefill_tok_s": (
                        physical_text_prefill_tokens / text_s
                        if text_s > 0
                        else 0.0
                    ),
                }
            )
            summary["local_compiled_generation"]["prefill_metrics"] = prefill_metrics
            summary["local_compiled_generation"]["prefill_calls"] = [
                {
                    "call_index": call_index,
                    "request_count": int(item.get("request_count", 0)),
                    "prefill_s": float(item.get("prefill_s", 0.0)),
                    "prefill_metrics": item.get("prefill_metrics", {}),
                }
                for call_index, item in enumerate(generation_metrics)
            ]
    summary_path = output_dir / f"run_summary_shard_{args.shard_index:02d}.json"
    atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"[summary] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
