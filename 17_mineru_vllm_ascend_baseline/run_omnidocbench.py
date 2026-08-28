#!/usr/bin/env python3
"""Run the stock MinerU vLLM-Ascend 310P reference contract on one 910B2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image


EXPERIMENT = "17_mineru_vllm_ascend_baseline"
MODE_COMPILED_ASYNC = "compiled_async"
MODE_EAGER_SYNC = "eager_sync"
MODES = (MODE_COMPILED_ASYNC, MODE_EAGER_SYNC)

DEFAULT_MODEL = Path("/workspace/models/MinerU2.5-Pro-2605-1.2B")
DEFAULT_DATASET_JSON = Path("/workspace/datasets/OmniDocBench/OmniDocBench.json")
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")
DEFAULT_COMPILE_CACHE_DIR = Path(
    "/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/17_mineru_vllm_ascend_baseline/vllm_compile"
)
DEFAULT_STATIC_OFF_COMPILE_CACHE_DIR = Path(
    "/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/17_mineru_vllm_ascend_baseline/vllm_compile_static_kernel_off"
)
CAPTURE_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32]
STATIC_KERNEL_ON = "on"
STATIC_KERNEL_OFF = "off"
STATIC_KERNEL_CHOICES = (STATIC_KERNEL_ON, STATIC_KERNEL_OFF)
PACKAGE_NAMES = (
    "vllm",
    "vllm-ascend",
    "torch",
    "torch-npu",
    "transformers",
    "mineru-vl-utils",
)


@dataclass(frozen=True)
class InputPage:
    dataset_index: int
    image_name: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default=MODE_COMPILED_ASYNC)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument(
        "--image-list",
        type=Path,
        help=(
            "Optional newline-delimited image-name manifest. Use this for the "
            "exact historical 981-page corpus when it becomes available."
        ),
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--layout-image-size", type=int, nargs=2, default=(1036, 1036))
    parser.add_argument("--hash-model-files", action="store_true")
    parser.add_argument(
        "--static-kernel",
        choices=STATIC_KERNEL_CHOICES,
        default=STATIC_KERNEL_ON,
        help=(
            "Enable or disable fixed-shape static-kernel compilation in the "
            "compiled_async lane. The accepted baseline default remains on."
        ),
    )
    parser.add_argument(
        "--allow-physical-npu5",
        action="store_true",
        help="Override the historical NPU5 quarantine. Never use this casually.",
    )
    return parser.parse_args()


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name_digest = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
    temporary = path.with_name(f".tmp-{os.getpid()}-{name_digest}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _dataset_image_name(sample: dict[str, Any]) -> str:
    page_info = sample.get("page_info") or {}
    value = page_info.get("image_path")
    if not value:
        raise ValueError("dataset sample has no page_info.image_path")
    return Path(value).name


def load_input_pages(
    *,
    dataset_json: Path,
    image_list: Path | None,
    images_dir: Path,
    offset: int,
    limit: int | None,
) -> tuple[list[InputPage], str]:
    if offset < 0 or (limit is not None and limit <= 0):
        raise ValueError("offset must be non-negative and limit must be positive")

    if image_list is not None:
        names = [
            line.strip()
            for line in image_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        source = f"image_list:{image_list.resolve()}"
    else:
        raw = json.loads(dataset_json.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise TypeError("dataset JSON must contain a list")
        names = [_dataset_image_name(sample) for sample in raw]
        source = f"dataset_json:{dataset_json.resolve()}"

    stop = None if limit is None else offset + limit
    selected = list(enumerate(names))[offset:stop]
    if not selected:
        raise ValueError("input selection is empty")

    pages = [
        InputPage(
            dataset_index=index,
            image_name=name,
            image_path=(images_dir / name).resolve(),
        )
        for index, name in selected
    ]
    missing = [str(page.image_path) for page in pages if not page.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} selected images; first={missing[0]}")
    stems = [Path(page.image_name).stem for page in pages]
    if len(stems) != len(set(stems)):
        raise ValueError("selected images contain duplicate output stems")
    return pages, source


def compile_cache_dir(enable_static_kernel: bool) -> Path:
    return (
        DEFAULT_COMPILE_CACHE_DIR
        if enable_static_kernel
        else DEFAULT_STATIC_OFF_COMPILE_CACHE_DIR
    )


def preset_spec(mode: str, *, enable_static_kernel: bool = True) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    common = {
        "model": "MinerU2.5-Pro-2605-1.2B",
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "quantization": None,
        "batch_size": 0,
    }
    if mode == MODE_COMPILED_ASYNC:
        return {
            **common,
            "engine": "AsyncLLM",
            "client_backend": "vllm-async-engine",
            "two_step_method": "concurrent_two_step_extract",
            "image_analysis": True,
            "max_num_seqs": 512,
            "max_num_batched_tokens": 16384,
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enable_npugraph_ex": True,
            "enable_static_kernel": enable_static_kernel,
            "fuse_norm_quant": False,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": CAPTURE_SIZES,
            "compile_cache_dir": str(compile_cache_dir(enable_static_kernel)),
        }
    return {
        **common,
        "engine": "LLM",
        "client_backend": "vllm-engine",
        "two_step_method": "two_step_extract_page_by_page",
        "image_analysis": False,
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "enable_npugraph_ex": False,
        "enable_static_kernel": False,
        "fuse_norm_quant": None,
        "cudagraph_mode": None,
        "cudagraph_capture_sizes": None,
    }


def build_engine_kwargs(
    mode: str,
    model: Path,
    logits_processor: Any,
    *,
    enable_static_kernel: bool = True,
) -> dict[str, Any]:
    spec = preset_spec(mode, enable_static_kernel=enable_static_kernel)
    kwargs: dict[str, Any] = {
        "model": str(model),
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "limit_mm_per_prompt": {"image": 1},
        "logits_processors": [logits_processor],
        "disable_log_stats": False,
        "hf_overrides": {
            "tie_word_embeddings": True,
            "text_config": {"tie_word_embeddings": True},
        },
        "enforce_eager": bool(spec["enforce_eager"]),
        "enable_prefix_caching": bool(spec["enable_prefix_caching"]),
        "enable_chunked_prefill": bool(spec["enable_chunked_prefill"]),
    }
    if mode == MODE_COMPILED_ASYNC:
        kwargs.update(
            {
                "max_num_seqs": 512,
                "max_num_batched_tokens": 16384,
                "additional_config": {
                    "ascend_compilation_config": {
                        "fuse_norm_quant": False,
                        "enable_npugraph_ex": True,
                        "enable_static_kernel": enable_static_kernel,
                    }
                },
                "compilation_config": {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": CAPTURE_SIZES,
                    "cache_dir": str(compile_cache_dir(enable_static_kernel)),
                },
            }
        )
    return kwargs


def model_manifest(model_dir: Path, full_hash: bool) -> dict[str, Any]:
    files = sorted(path for path in model_dir.rglob("*") if path.is_file())
    if not full_hash:
        files = [
            path
            for path in files
            if path.suffix == ".json" or path.name.endswith(".safetensors.index.json")
        ]
    return {
        "root": str(model_dir),
        "full_hash": full_hash,
        "files": [
            {
                "path": str(path.relative_to(model_dir)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def input_manifest(pages: list[InputPage], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "count": len(pages),
        "pages": [
            {
                "dataset_index": page.dataset_index,
                "image": page.image_name,
                "size": page.image_path.stat().st_size,
                "sha256": sha256_file(page.image_path),
            }
            for page in pages
        ],
    }


def configure_npu(allow_physical_npu5: bool) -> tuple[Any, Any]:
    physical = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not physical:
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES is unset; source npu-setup first")
    if physical.strip() == "5" and not allow_physical_npu5:
        raise RuntimeError("physical NPU5 is quarantined; select another free device")
    import torch
    import torch_npu

    torch.npu.set_device("npu:0")
    torch.npu.config.allow_internal_format = True
    return torch, torch_npu


def patch_vllm_prompt_renderer(mode: str) -> None:
    if mode == MODE_COMPILED_ASYNC:
        from mineru_vl_utils.vlm_client.vllm_async_engine_client import (
            VllmAsyncEngineVlmClient,
        )

        async def keep_raw_prompt(self, raw_prompt):
            return raw_prompt

        VllmAsyncEngineVlmClient._render_vllm_cmpl_input = keep_raw_prompt
    else:
        from mineru_vl_utils.vlm_client.vllm_engine_client import VllmEngineVlmClient

        def keep_raw_prompts(self, raw_prompts):
            return raw_prompts

        VllmEngineVlmClient._render_vllm_cmpl_inputs = keep_raw_prompts


def create_engine(
    mode: str,
    model_dir: Path,
    *,
    enable_static_kernel: bool = True,
) -> Any:
    from mineru_vl_utils import MinerULogitsProcessor

    patch_vllm_prompt_renderer(mode)
    kwargs = build_engine_kwargs(
        mode,
        model_dir,
        MinerULogitsProcessor,
        enable_static_kernel=enable_static_kernel,
    )
    if mode == MODE_COMPILED_ASYNC:
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
    from vllm import LLM

    return LLM(**kwargs)


def create_client(mode: str, engine: Any, layout_image_size: tuple[int, int]) -> Any:
    from mineru_vl_utils import MinerUClient

    if mode == MODE_COMPILED_ASYNC:
        return MinerUClient(
            backend="vllm-async-engine",
            vllm_async_llm=engine,
            batch_size=0,
            image_analysis=True,
            layout_image_size=layout_image_size,
            max_concurrency=512,
            use_tqdm=True,
        )
    return MinerUClient(
        backend="vllm-engine",
        vllm_llm=engine,
        batch_size=0,
        image_analysis=False,
        layout_image_size=layout_image_size,
        use_tqdm=True,
    )


def open_images(pages: list[InputPage]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for page in pages:
        with Image.open(page.image_path) as source:
            images.append(source.convert("RGB"))
    return images


def run_two_step(mode: str, client: Any, images: list[Image.Image]) -> tuple[list[Any], list[float]]:
    if mode == MODE_COMPILED_ASYNC:
        return client.concurrent_two_step_extract(images), []
    results: list[Any] = []
    page_times: list[float] = []
    for image in images:
        started = time.perf_counter()
        results.append(client.two_step_extract(image))
        page_times.append(time.perf_counter() - started)
    return results, page_times


def main() -> None:
    args = parse_args()
    enable_static_kernel = args.static_kernel == STATIC_KERNEL_ON
    model_dir = args.model.expanduser().resolve()
    dataset_json = args.dataset_json.expanduser().resolve()
    image_list = args.image_list.expanduser().resolve() if args.image_list else None
    images_dir = args.images_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    if image_list is None and not dataset_json.is_file():
        raise FileNotFoundError(f"dataset JSON not found: {dataset_json}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    content_dir = output_dir / "content_lists"
    predictions_dir.mkdir()
    content_dir.mkdir()

    pages, source = load_input_pages(
        dataset_json=dataset_json,
        image_list=image_list,
        images_dir=images_dir,
        offset=args.offset,
        limit=args.limit,
    )
    inputs = input_manifest(pages, source)
    model_info = model_manifest(model_dir, args.hash_model_files)
    write_json(output_dir / "input_manifest.json", inputs)
    write_json(output_dir / "model_manifest.json", model_info)

    torch, _torch_npu = configure_npu(args.allow_physical_npu5)
    environment = {
        "experiment": EXPERIMENT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "mode": args.mode,
        "preset": preset_spec(
            args.mode,
            enable_static_kernel=enable_static_kernel,
        ),
        "static_kernel": args.static_kernel,
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": package_versions(),
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "input_source": source,
        "selected_pages": len(pages),
        "offset": args.offset,
        "limit": args.limit,
        "layout_image_size": list(args.layout_image_size),
    }
    write_json(output_dir / "run_manifest.json", environment)

    setup_started = time.perf_counter()
    engine = create_engine(
        args.mode,
        model_dir,
        enable_static_kernel=enable_static_kernel,
    )
    torch.npu.synchronize()
    engine_setup_s = time.perf_counter() - setup_started

    images: list[Image.Image] = []
    benchmark_started = time.perf_counter()
    try:
        client_started = time.perf_counter()
        client = create_client(args.mode, engine, tuple(args.layout_image_size))
        client_setup_s = time.perf_counter() - client_started

        image_load_started = time.perf_counter()
        images = open_images(pages)
        image_load_s = time.perf_counter() - image_load_started

        inference_started = time.perf_counter()
        content_lists, page_times = run_two_step(args.mode, client, images)
        torch.npu.synchronize()
        inference_s = time.perf_counter() - inference_started
        if len(content_lists) != len(pages):
            raise RuntimeError(
                f"result count {len(content_lists)} != selected pages {len(pages)}"
            )

        from mineru_vl_utils.post_process import json2md

        output_started = time.perf_counter()
        for page, content in zip(pages, content_lists):
            stem = Path(page.image_name).stem
            markdown = json2md(content)
            (predictions_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
            write_json(content_dir / f"{stem}.json", list(content))
        output_write_s = time.perf_counter() - output_started
        benchmark_wall_s = time.perf_counter() - benchmark_started

        summary = {
            **environment,
            "engine_setup_s": engine_setup_s,
            "client_setup_s": client_setup_s,
            "image_load_s": image_load_s,
            "inference_s": inference_s,
            "output_write_s": output_write_s,
            "benchmark_wall_s": benchmark_wall_s,
            "completed": len(content_lists),
            "failed": 0,
            "pages_per_s": len(content_lists) / benchmark_wall_s,
            "inference_pages_per_s": len(content_lists) / inference_s,
            "page_time_s": (
                {
                    "mean": sum(page_times) / len(page_times),
                    "min": min(page_times),
                    "max": max(page_times),
                }
                if page_times
                else None
            ),
        }
        write_json(output_dir / "run_summary.json", summary)
        print("RUN_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as error:
        failure = {
            **environment,
            "engine_setup_s": engine_setup_s,
            "benchmark_elapsed_s": time.perf_counter() - benchmark_started,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "failure.json", failure)
        raise
    finally:
        for image in images:
            image.close()


if __name__ == "__main__":
    main()
