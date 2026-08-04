#!/usr/bin/env python3
"""Production-faithful lab for PaddleOCR-VL text-decode experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.modeling import (
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
)
from paddleocr_vl.model.text_decode import (
    TextDecodeStage,
    TextDecodeRuntime,
    cast_decode_linear_weights_to_nz,
    decode_optimization_names,
    prepare_decode_optimization_modules,
    torchair_cache_dir_for_shape,
)
from paddleocr_vl.serving.continuous_decode import (
    ContinuousDecodeScheduler,
    DecodeArena,
    DecodeSlotState,
    ReadyDecodeRequest,
)
from utils.timing import DeviceTimeline, synchronize


DEFAULT_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_CORPUS = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_decode_lab"
    / "corpus_256p_b32_kv4096_7899d40.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp/09_persistent_page_engine/text_decode_lab"
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair"
)
MODES = (
    "simulate",
    "profile",
    "torch_profile",
    "boundary",
    "tail_invariance",
    "replay",
    "correctness",
)
NPU_PROFILE_METRICS = ("pipe", "memory", "l2", "memory_access")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="simulate")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--backend", choices=("torchair", "raw_eager"), default="torchair"
    )
    parser.add_argument(
        "--decode-optimization",
        choices=decode_optimization_names(),
        default="baseline",
        help=(
            "Named text-transformer implementation preset. The baseline "
            "matches the production implementation; other presets are "
            "lab-only candidates."
        ),
    )
    parser.add_argument("--active-slots", type=int)
    parser.add_argument("--profile-position", type=int, default=1024)
    parser.add_argument(
        "--tail-positions",
        default="64,80,113",
        help=(
            "Comma-separated non-boundary cache positions for "
            "tail_invariance mode."
        ),
    )
    parser.add_argument(
        "--tail-canary-value",
        type=float,
        default=0.25,
        help="Value written only into masked KV-tail positions.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "npu_profiles",
        help="Root for raw torch_npu profiler output in torch_profile mode.",
    )
    parser.add_argument(
        "--profile-metric",
        choices=NPU_PROFILE_METRICS,
        default="pipe",
        help="AI Core metric collected by torch_npu profiler.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--overflow-policy",
        choices=("error", "exclude"),
        default="error",
        help="How replay/simulation handles requests exceeding --cache-length.",
    )
    parser.add_argument("--ready-buffer-capacity", type=int)
    parser.add_argument("--ready-buffer-low-watermark", type=int)
    parser.add_argument("--correctness-items", type=int, default=1)
    parser.add_argument("--correctness-steps", type=int, default=16)
    parser.add_argument(
        "--allow-compile",
        action="store_true",
        help="Allow a missing static TorchAir shape to compile.",
    )
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.batch_size & (args.batch_size - 1):
        parser.error("--batch-size must be a positive power of two")
    if args.cache_length <= 0:
        parser.error("--cache-length must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    if args.active_slots is None:
        args.active_slots = args.batch_size
    if args.active_slots <= 0 or args.active_slots > args.batch_size:
        parser.error("--active-slots must be in [1, batch-size]")
    if args.profile_position < 0:
        parser.error("--profile-position must be non-negative")
    try:
        args.tail_positions = tuple(
            int(value) for value in args.tail_positions.split(",") if value
        )
    except ValueError:
        parser.error("--tail-positions must be comma-separated integers")
    if not args.tail_positions:
        parser.error("--tail-positions must not be empty")
    if any(position < 0 for position in args.tail_positions):
        parser.error("--tail-positions must be non-negative")
    if args.correctness_items <= 0 or args.correctness_items > args.batch_size:
        parser.error("--correctness-items must be in [1, batch-size]")
    if args.correctness_steps <= 0:
        parser.error("--correctness-steps must be positive")
    if args.mode == "torch_profile" and args.repeats > 8:
        parser.error(
            "torch_profile mode permits at most 8 captured steps to keep "
            "profiler output bounded"
        )
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_corpus(
    path: Path,
    max_items: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = path.expanduser().resolve()
    corpus = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        not isinstance(corpus, dict)
        or corpus.get("schema_version") != 1
        or corpus.get("kind") != "text_decode_trace_replay"
    ):
        raise ValueError(f"not a text-decode lab corpus: {resolved}")
    items = [dict(item) for item in corpus.get("items", ())]
    if max_items is not None:
        items = items[:max_items]
    if not items:
        raise ValueError("decode corpus contains no selected items")
    return corpus, items


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _per_second(count: int, seconds: float) -> float | None:
    return float(count) / float(seconds) if seconds > 0 else None


def _memory_baseline(device: torch.device) -> int:
    try:
        torch.npu.reset_peak_memory_stats(device)
        return int(torch.npu.memory_allocated(device))
    except Exception:
        return 0


def _peak_memory_delta(device: torch.device, baseline: int) -> int | None:
    try:
        return max(0, int(torch.npu.max_memory_allocated(device)) - baseline)
    except Exception:
        return None


def _npu_profiler_config(metric: str) -> Any:
    import torch_npu.profiler as npu_prof

    metrics = {
        "pipe": npu_prof.AiCMetrics.PipeUtilization,
        "memory": npu_prof.AiCMetrics.Memory,
        "l2": npu_prof.AiCMetrics.L2Cache,
        "memory_access": npu_prof.AiCMetrics.MemoryAccess,
    }
    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=metrics[metric],
        l2_cache=metric == "l2",
        export_type=npu_prof.ExportType.Text,
        data_simplification=False,
    )


def _filter_for_cache(
    items: list[dict[str, Any]],
    *,
    cache_length: int,
    overflow_policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fitting = [
        item
        for item in items
        if int(item["replay_required_cache_tokens"]) <= int(cache_length)
    ]
    overflow = [
        item
        for item in items
        if int(item["replay_required_cache_tokens"]) > int(cache_length)
    ]
    if overflow and overflow_policy == "error":
        worst = max(
            int(item["replay_required_cache_tokens"]) for item in overflow
        )
        raise ValueError(
            f"{len(overflow)} requests exceed cache_length={cache_length}; "
            f"maximum required={worst}. Use --overflow-policy exclude only for "
            "an explicitly partial workload."
        )
    if not fitting:
        raise ValueError("no requests fit the selected cache length")
    return fitting, overflow


@dataclass
class _SimSlot:
    item: dict[str, Any]
    remaining_iterations: int


def simulate_scheduler(
    items: list[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    source_index = 0
    slots: list[_SimSlot | None] = [None] * batch_size
    graph_calls = 0
    active_slots = 0
    completed = 0
    prefill_only = 0

    def fill() -> None:
        nonlocal source_index, prefill_only
        for slot_index, slot in enumerate(slots):
            if slot is not None:
                continue
            while source_index < len(items):
                item = items[source_index]
                source_index += 1
                iterations = int(item["active_decode_iterations"])
                if iterations == 0:
                    prefill_only += 1
                    continue
                slots[slot_index] = _SimSlot(item, iterations)
                break

    fill()
    while any(slot is not None for slot in slots):
        graph_calls += 1
        active_slots += sum(slot is not None for slot in slots)
        for slot_index, slot in enumerate(slots):
            if slot is None:
                continue
            slot.remaining_iterations -= 1
            if slot.remaining_iterations == 0:
                slots[slot_index] = None
                completed += 1
        fill()

    effective_tokens = sum(int(item["effective_decode_tokens"]) for item in items)
    raw_slots = graph_calls * batch_size
    idle_slots = raw_slots - active_slots
    lookahead_slots = active_slots - effective_tokens
    if min(idle_slots, lookahead_slots) < 0:
        raise AssertionError("decode simulation accounting went negative")
    if raw_slots != effective_tokens + idle_slots + lookahead_slots:
        raise AssertionError("decode simulation accounting does not balance")
    if completed + prefill_only != len(items):
        raise AssertionError("decode simulation did not complete every request")
    return {
        "requests": len(items),
        "batch_size": batch_size,
        "graph_calls": graph_calls,
        "raw_decode_token_slots": raw_slots,
        "active_decode_token_slots": active_slots,
        "effective_decode_tokens": effective_tokens,
        "idle_decode_token_slots": idle_slots,
        "lookahead_decode_token_slots": lookahead_slots,
        "prefill_only_completions": prefill_only,
        "decode_useful_token_fraction": effective_tokens / raw_slots,
        "active_slot_fraction": active_slots / raw_slots,
    }


class TextDecodeLab:
    def __init__(self, args: argparse.Namespace):
        import torch_npu  # noqa: F401

        # Match the production runner before the first NPU allocation.  On
        # torch-npu 2.10, setting this later leaves npu_format_cast in ND and
        # selects a different compiled graph/cache from production.
        torch.npu.config.allow_internal_format = True
        self.args = args
        self.device = torch.device("npu:0")
        if not torch.npu.is_available():
            raise RuntimeError("text-decode lab requires an available Ascend NPU")
        torch.npu.set_compile_mode(jit_compile=False)
        self.dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        self.model_dir = _resolve_model_dir(args.model)

        synchronize(self.device)
        setup_started = time.perf_counter()
        model_started = time.perf_counter()
        self.model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
            self.model_dir,
            dtype=self.dtype,
            device=self.device,
        ).eval()
        synchronize(self.device)
        self.model_load_s = time.perf_counter() - model_started

        self.optimization = prepare_decode_optimization_modules(
            self.model,
            args.decode_optimization,
        )
        format_started = time.perf_counter()
        self.weight_format = cast_decode_linear_weights_to_nz(self.model)
        synchronize(self.device)
        self.weight_format_s = time.perf_counter() - format_started
        self.linear_weight_format = str(self.weight_format["effective_mode"])
        if args.backend == "torchair":
            self._preflight_cache()

        runtime_started = time.perf_counter()
        self.runtime = TextDecodeRuntime(
            self.model,
            backend=args.backend,
            device=self.device,
            cache_root=args.cache_dir,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            dtype=self.dtype,
            model_dir=self.model_dir,
            linear_weight_format=self.linear_weight_format,
            optimization=self.optimization,
        )
        self.reference_stage = TextDecodeStage(
            self.model,
            optimization="baseline",
        ).eval()
        synchronize(self.device)
        self.runtime_setup_s = time.perf_counter() - runtime_started
        self.setup_s = time.perf_counter() - setup_started

    def _preflight_cache(self) -> None:
        shape_dir = torchair_cache_dir_for_shape(
            self.args.cache_dir,
            batch_size=self.args.batch_size,
            cache_length=self.args.cache_length,
            dtype=self.dtype,
            device=self.device,
            model_dir=self.model_dir,
            linear_weight_format=self.linear_weight_format,
            optimization=self.optimization,
        )
        if (
            not shape_dir.is_dir() or not any(shape_dir.iterdir())
        ) and not self.args.allow_compile:
            raise RuntimeError(
                "missing compiled text-decode graph cache; refusing accidental "
                f"compilation without --allow-compile:\n  - {shape_dir}"
            )

    def _dummy_arena(
        self,
        *,
        active_slots: int,
        cache_position: int,
    ) -> DecodeArena:
        arena = DecodeArena(
            cache=self.runtime.warm_cache,
            device=self.device,
            batch_size=self.args.batch_size,
            eos_token_id=int(self.model.config.eos_token_id),
        )
        arena.next_token.copy_(
            torch.arange(
                1,
                self.args.batch_size + 1,
                device=self.device,
                dtype=torch.int64,
            ).view(-1, 1)
        )
        arena.cache_position.fill_(cache_position)
        for slot_index in range(active_slots):
            ready = ReadyDecodeRequest(
                request_id=f"profile:{slot_index}",
                payload=None,
                cache=None,
                rope_deltas=None,
                cache_position=None,
                first_token_tensor=None,
                first_token=slot_index + 1,
                prompt_length=cache_position,
            )
            arena.slots[slot_index] = DecodeSlotState(
                slot_index=slot_index,
                epoch=1,
                ready=ready,
                token_ids=[slot_index + 1],
                admitted_at=time.perf_counter(),
            )
            arena.active_mask[slot_index].fill_(True)
        return arena

    def profile(self) -> dict[str, Any]:
        last_position = (
            self.args.profile_position + self.args.warmup + self.args.repeats
        )
        if last_position >= self.args.cache_length:
            raise ValueError(
                "profile position plus warmup/repeats reaches cache capacity: "
                f"{last_position} >= {self.args.cache_length}"
            )
        warm_arena = self._dummy_arena(
            active_slots=self.args.active_slots,
            cache_position=self.args.profile_position,
        )
        for iteration in range(self.args.warmup):
            warm_arena.step(self.runtime.fn, iteration=iteration)
        synchronize(self.device)

        arena = self._dummy_arena(
            active_slots=self.args.active_slots,
            cache_position=self.args.profile_position,
        )
        memory_baseline = _memory_baseline(self.device)
        timeline = DeviceTimeline(self.device)
        synchronize(self.device)
        wall_started = time.perf_counter()
        for iteration in range(self.args.repeats):
            timeline.measure(
                f"step_{iteration:04d}",
                lambda iteration=iteration: arena.step(
                    self.runtime.fn,
                    iteration=iteration,
                ),
            )
        spans = timeline.resolve()
        wall_s = time.perf_counter() - wall_started
        model_and_argmax_s, admission_s = arena.resolve_device_timing()
        if admission_s:
            raise AssertionError("profile mode unexpectedly performed admission")
        durations = list(spans.values())
        full_step_s = sum(durations)
        raw_slots = self.args.repeats * self.args.batch_size
        active_slots = self.args.repeats * self.args.active_slots
        return {
            "shape": {
                "batch_size": self.args.batch_size,
                "cache_length": self.args.cache_length,
                "active_slots": self.args.active_slots,
                "initial_cache_position": self.args.profile_position,
            },
            "warmup": self.args.warmup,
            "repeats": self.args.repeats,
            "device_s": {
                "full_production_step": full_step_s,
                "model_and_argmax": model_and_argmax_s,
                "post_graph_state_update": max(
                    0.0, full_step_s - model_and_argmax_s
                ),
            },
            "latency_ms": {
                "mean": statistics.mean(durations) * 1000.0,
                "median": statistics.median(durations) * 1000.0,
                "p95": _percentile(durations, 0.95) * 1000.0,
                "min": min(durations) * 1000.0,
                "max": max(durations) * 1000.0,
            },
            "throughput": {
                "raw_physical_tok_per_s": _per_second(raw_slots, full_step_s),
                "active_tok_per_s": _per_second(active_slots, full_step_s),
                "graph_calls_per_s": _per_second(self.args.repeats, full_step_s),
            },
            "memory_bytes": {
                "baseline_allocated": memory_baseline,
                "peak_delta": _peak_memory_delta(self.device, memory_baseline),
            },
            "host_wall_s": wall_s,
        }

    def torch_profile(self) -> dict[str, Any]:
        if self.device.type != "npu":
            raise ValueError("torch_profile mode requires an NPU device")
        if self.args.backend != "torchair":
            raise ValueError(
                "torch_profile mode requires --backend torchair so the "
                "capture contains the production compiled decode graph"
            )
        last_position = (
            self.args.profile_position + self.args.warmup + self.args.repeats
        )
        if last_position >= self.args.cache_length:
            raise ValueError(
                "profile position plus warmup/repeats reaches cache capacity: "
                f"{last_position} >= {self.args.cache_length}"
            )

        import torch_npu.profiler as npu_prof

        profile_root = self.args.profile_dir.expanduser().resolve()
        run_name = "_".join(
            [
                f"b{self.args.batch_size}",
                f"k{self.args.cache_length}",
                f"p{self.args.profile_position}",
                self.args.profile_metric,
                time.strftime("%Y%m%d_%H%M%S"),
            ]
        )
        profile_dir = profile_root / run_name
        shutil.rmtree(profile_dir, ignore_errors=True)
        profile_dir.mkdir(parents=True, exist_ok=True)

        arena = self._dummy_arena(
            active_slots=self.args.active_slots,
            cache_position=self.args.profile_position,
        )
        for iteration in range(self.args.warmup):
            arena.step(self.runtime.fn, iteration=iteration)
        synchronize(self.device)

        schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
        synchronize(self.device)
        wall_started = time.perf_counter()
        with npu_prof.profile(
            activities=[
                npu_prof.ProfilerActivity.CPU,
                npu_prof.ProfilerActivity.NPU,
            ],
            schedule=schedule,
            experimental_config=_npu_profiler_config(
                self.args.profile_metric
            ),
            on_trace_ready=npu_prof.tensorboard_trace_handler(
                str(profile_dir),
                analyse_flag=True,
            ),
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
            with_modules=False,
            with_flops=False,
        ) as profiler:
            with torch.profiler.record_function(
                "paddleocr_vl.compiled_decode_profile"
            ):
                for captured_step in range(self.args.repeats):
                    with torch.profiler.record_function(
                        "paddleocr_vl.compiled_decode_step"
                    ):
                        arena.step(
                            self.runtime.fn,
                            iteration=self.args.warmup + captured_step,
                        )
            synchronize(self.device)
            profiler.step()
        synchronize(self.device)
        profile_wall_s = time.perf_counter() - wall_started

        return {
            "shape": {
                "batch_size": self.args.batch_size,
                "cache_length": self.args.cache_length,
                "active_slots": self.args.active_slots,
                "initial_cache_position": self.args.profile_position,
                "captured_start_position": (
                    self.args.profile_position + self.args.warmup
                ),
            },
            "warmup_steps_outside_profiler": self.args.warmup,
            "captured_steps": self.args.repeats,
            "captured_raw_token_slots": (
                self.args.repeats * self.args.batch_size
            ),
            "metric": self.args.profile_metric,
            "profiler_level": "Level1",
            "record_shapes": True,
            "with_stack": True,
            "profile_memory": False,
            "profile_dir": str(profile_dir),
            "profile_wall_s": profile_wall_s,
            "profile_wall_is_throughput_measurement": False,
        }

    def boundary(self) -> dict[str, Any]:
        """Run one full decoder step with durable synchronization markers."""
        arena = self._dummy_arena(
            active_slots=self.args.active_slots,
            cache_position=self.args.profile_position,
        )
        synchronize(self.device)
        event = {
            "batch_size": self.args.batch_size,
            "active_slots": self.args.active_slots,
            "cache_length": self.args.cache_length,
            "cache_position": self.args.profile_position,
            "effective_length": self.args.profile_position + 1,
            "backend": self.args.backend,
            "decode_optimization": self.optimization.name,
            "attention": self.optimization.attention,
        }
        print(
            "TEXT_DECODE_BOUNDARY "
            + json.dumps({"event": "step_begin", **event}, sort_keys=True),
            flush=True,
        )
        started = time.perf_counter()
        arena.step(self.runtime.fn, iteration=0)
        print(
            "TEXT_DECODE_BOUNDARY "
            + json.dumps({"event": "step_returned", **event}, sort_keys=True),
            flush=True,
        )
        print(
            "TEXT_DECODE_BOUNDARY "
            + json.dumps({"event": "sync_begin", **event}, sort_keys=True),
            flush=True,
        )
        try:
            synchronize(self.device)
        except BaseException as exc:
            print(
                "TEXT_DECODE_BOUNDARY "
                + json.dumps(
                    {
                        "event": "sync_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        **event,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise
        elapsed_s = time.perf_counter() - started
        print(
            "TEXT_DECODE_BOUNDARY "
            + json.dumps(
                {"event": "sync_end", "elapsed_s": elapsed_s, **event},
                sort_keys=True,
            ),
            flush=True,
        )
        model_and_argmax_s, admission_s = arena.resolve_device_timing()
        if admission_s:
            raise AssertionError("boundary mode unexpectedly performed admission")
        return {
            "shape": {
                "batch_size": self.args.batch_size,
                "cache_length": self.args.cache_length,
                "active_slots": self.args.active_slots,
                "cache_position": self.args.profile_position,
                "effective_length": self.args.profile_position + 1,
            },
            "backend": self.args.backend,
            "decode_optimization": self.optimization.name,
            "attention": self.optimization.attention,
            "elapsed_s": elapsed_s,
            "model_and_argmax_device_s": model_and_argmax_s,
        }

    def tail_invariance(self) -> dict[str, Any]:
        """Prove whether masked KV-tail contents can affect compiled decode.

        Every compared lane has identical tokens, cache positions, RoPE deltas,
        and valid KV prefixes.  Only positions strictly after each row's
        current cache position differ.  The decode graph overwrites the current
        position before attention, so those differing values are all masked.
        """

        positions_to_test = tuple(int(value) for value in self.args.tail_positions)
        if any(position >= self.args.cache_length - 1 for position in positions_to_test):
            raise ValueError("tail test positions must leave a non-empty KV tail")
        pse_boundaries = tuple(
            position
            for position in positions_to_test
            if (position + 1) % 1280 == 0
        )
        if pse_boundaries:
            raise ValueError(
                "tail invariance deliberately excludes PSE-sentinel boundaries: "
                f"{pse_boundaries}"
            )

        # Reuse the runtime's shape-owning cache.  A second B64xKV4096 arena is
        # several GiB and would make this diagnostic unrepresentative or OOM on
        # a 21-GiB 310P even though production itself fits.
        cache = self.runtime.warm_cache
        input_ids = (
            torch.arange(
                1,
                self.args.batch_size + 1,
                device=self.device,
                dtype=torch.int64,
            )
            % int(self.model.config.vocab_size)
        ).view(-1, 1)
        rope_deltas = torch.zeros(
            (self.args.batch_size, 1),
            device=self.device,
            dtype=torch.int64,
        )
        physical_positions = torch.arange(
            self.args.cache_length,
            device=self.device,
            dtype=torch.int64,
        ).view(1, 1, self.args.cache_length, 1)

        def reset_cache() -> None:
            for tensor in cache.flat_tensors():
                tensor.zero_()

        def set_masked_tail(
            cache_positions: torch.Tensor,
            *,
            rows: str,
        ) -> None:
            tail_mask = physical_positions > cache_positions.view(-1, 1, 1, 1)
            if rows == "row0":
                row_mask = torch.zeros(
                    (self.args.batch_size, 1, 1, 1),
                    device=self.device,
                    dtype=torch.bool,
                )
                row_mask[0].fill_(True)
                tail_mask = tail_mask & row_mask
            elif rows != "all":
                raise ValueError(f"unknown tail row selection: {rows}")
            for tensor in cache.flat_tensors():
                tensor.masked_fill_(tail_mask, self.args.tail_canary_value)

        def execute(cache_positions: torch.Tensor) -> torch.Tensor:
            logits = self.runtime.fn(
                input_ids,
                cache_positions,
                rope_deltas,
                *cache.flat_tensors(),
            )
            synchronize(self.device)
            return logits[:, -1, :].float().cpu()

        def compare(
            reference: torch.Tensor,
            candidate: torch.Tensor,
        ) -> dict[str, Any]:
            diff = (candidate - reference).abs()
            reference_top1 = torch.argmax(reference, dim=-1)
            candidate_top1 = torch.argmax(candidate, dim=-1)
            per_row_max = diff.amax(dim=-1)
            changed_rows = torch.nonzero(per_row_max > 0, as_tuple=False).reshape(-1)
            return {
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
                "exact": bool(torch.equal(reference, candidate)),
                "top1_matches": int((reference_top1 == candidate_top1).sum().item()),
                "top1_total": self.args.batch_size,
                "changed_rows": [int(value) for value in changed_rows.tolist()],
                "per_row_max_abs_first_16": [
                    float(value) for value in per_row_max[:16].tolist()
                ],
            }

        scenarios: list[tuple[str, torch.Tensor]] = []
        for position in positions_to_test:
            scenarios.append(
                (
                    f"uniform_{position}",
                    torch.full(
                        (self.args.batch_size,),
                        position,
                        device=self.device,
                        dtype=torch.int64,
                    ),
                )
            )
        if self.args.batch_size > 1 and len(positions_to_test) > 1:
            scenarios.append(
                (
                    "mixed",
                    torch.tensor(
                        [
                            positions_to_test[index % len(positions_to_test)]
                            for index in range(self.args.batch_size)
                        ],
                        device=self.device,
                        dtype=torch.int64,
                    ),
                )
            )

        rows = []
        for scenario, cache_positions in scenarios:
            print(
                "DECODE_TAIL_INVARIANCE "
                f"scenario={scenario} state=zero_reference_begin",
                flush=True,
            )
            reset_cache()
            zero_reference = execute(cache_positions)

            reset_cache()
            set_masked_tail(cache_positions, rows="row0")
            row0_stale = execute(cache_positions)

            reset_cache()
            set_masked_tail(cache_positions, rows="all")
            all_stale = execute(cache_positions)

            reset_cache()
            zero_repeat = execute(cache_positions)
            row = {
                "scenario": scenario,
                "cache_positions": [int(value) for value in cache_positions.cpu().tolist()],
                "zero_repeat": compare(zero_reference, zero_repeat),
                "row0_stale_tail": compare(zero_reference, row0_stale),
                "all_rows_stale_tail": compare(zero_reference, all_stale),
            }
            rows.append(row)
            print(
                "DECODE_TAIL_INVARIANCE "
                f"scenario={scenario} "
                f"zero_repeat_exact={row['zero_repeat']['exact']} "
                f"row0_exact={row['row0_stale_tail']['exact']} "
                f"all_exact={row['all_rows_stale_tail']['exact']}",
                flush=True,
            )

        return {
            "contract": {
                "graph": "production compiled 27-layer text decode",
                "identical": (
                    "tokens, cache positions, RoPE deltas, and valid KV prefix"
                ),
                "only_difference": (
                    "masked KV positions strictly after cache_position"
                ),
                "tail_canary_value": self.args.tail_canary_value,
            },
            "batch_size": self.args.batch_size,
            "cache_length": self.args.cache_length,
            "decode_optimization": self.optimization.name,
            "scenarios": rows,
            "all_zero_repeats_exact": all(
                row["zero_repeat"]["exact"] for row in rows
            ),
            "all_row0_stale_exact": all(
                row["row0_stale_tail"]["exact"] for row in rows
            ),
            "all_rows_stale_exact": all(
                row["all_rows_stale_tail"]["exact"] for row in rows
            ),
        }

    def replay(
        self,
        items: list[dict[str, Any]],
        overflow: list[dict[str, Any]],
    ) -> dict[str, Any]:
        memory_baseline = _memory_baseline(self.device)
        source_cache = self.model.allocate_static_cache(
            batch_size=1,
            cache_length=self.args.cache_length,
            device=self.device,
            dtype=self.dtype,
            init_mode="zeros",
        )
        first_tokens = torch.tensor(
            [int(item["first_token"]) for item in items],
            device=self.device,
            dtype=torch.int64,
        ).view(-1, 1)
        cache_positions = torch.tensor(
            [int(item["prompt_tokens"]) for item in items],
            device=self.device,
            dtype=torch.int64,
        )
        rope_deltas = torch.zeros(
            (len(items), 1),
            device=self.device,
            dtype=torch.int64,
        )
        ready = [
            ReadyDecodeRequest(
                request_id=str(item["request_id"]),
                payload=item,
                cache=source_cache,
                rope_deltas=rope_deltas[index : index + 1],
                cache_position=cache_positions[index : index + 1],
                first_token_tensor=first_tokens[index : index + 1],
                first_token=int(item["first_token"]),
                prompt_length=int(item["prompt_tokens"]),
            )
            for index, item in enumerate(items)
        ]
        targets = {
            str(item["request_id"]): (
                int(item["generated_tokens"]),
                str(item["stop_reason"]),
            )
            for item in items
        }

        def recorded_completion(
            state: DecodeSlotState,
            _token_id: int,
        ) -> str | None:
            target_length, stop_reason = targets[state.ready.request_id]
            return stop_reason if len(state.token_ids) >= target_length else None

        arena = DecodeArena(
            cache=self.runtime.warm_cache,
            device=self.device,
            batch_size=self.args.batch_size,
            eos_token_id=int(self.model.config.eos_token_id),
        )
        scheduler = ContinuousDecodeScheduler(
            arena=arena,
            decode_fn=self.runtime.fn,
            max_new_tokens=max(int(item["generated_tokens"]) for item in items),
            completion_policy=recorded_completion,
        )
        capacity = (
            self.args.ready_buffer_capacity
            if self.args.ready_buffer_capacity is not None
            else self.args.batch_size * 4
        )
        low_watermark = (
            self.args.ready_buffer_low_watermark
            if self.args.ready_buffer_low_watermark is not None
            else self.args.batch_size
        )
        synchronize(self.device)
        wall_started = time.perf_counter()
        run = scheduler.run_stream(
            ready,
            ready_buffer_capacity=capacity,
            ready_buffer_low_watermark=low_watermark,
        )
        synchronize(self.device)
        wall_s = time.perf_counter() - wall_started
        simulation = simulate_scheduler(items, batch_size=self.args.batch_size)
        replay_accounting = {
            "graph_calls": run.graph_calls,
            "raw_decode_token_slots": run.raw_decode_token_slots,
            "active_decode_token_slots": run.active_decode_token_slots,
            "effective_decode_tokens": run.effective_decode_tokens,
            "idle_decode_token_slots": run.idle_decode_token_slots,
            "lookahead_decode_token_slots": run.lookahead_decode_token_slots,
            "prefill_only_completions": run.prefill_only_completions,
        }
        expected_accounting = {
            name: simulation[name] for name in replay_accounting
        }
        if replay_accounting != expected_accounting:
            raise AssertionError(
                "device replay diverged from trace simulation:\n"
                f"replay={replay_accounting}\nexpected={expected_accounting}"
            )
        decode_wall_s = float(run.timing_s["continuous_decode_wall"])
        model_device_s = float(
            run.timing_s["decode_model_and_argmax_device"]
        )
        return {
            "workload": {
                "requests": len(items),
                "excluded_overflow_requests": len(overflow),
                "arrival_mode": "all_ready",
                "completion_mode": "recorded_request_lifetimes",
                "prompt_kv_values": "shared deterministic zero prefix",
                "rope_deltas": "zeros",
            },
            "scheduler": {
                **replay_accounting,
                "ready_buffer_capacity": run.ready_buffer_capacity,
                "ready_buffer_low_watermark": run.ready_buffer_low_watermark,
                "max_ready_queue_depth": run.max_ready_queue_depth,
                "ready_source_refill_count": run.ready_source_refill_count,
                "initial_admissions": run.initial_admissions,
                "hot_swap_admissions": run.hot_swap_admissions,
                "kv_prefix_bytes_copied": run.kv_prefix_bytes_copied,
            },
            "timing_s": dict(run.timing_s),
            "throughput": {
                "continuous_decode_effective_tok_per_s": _per_second(
                    run.effective_decode_tokens, decode_wall_s
                ),
                "continuous_decode_raw_physical_tok_per_s": _per_second(
                    run.raw_decode_token_slots, decode_wall_s
                ),
                "model_device_effective_tok_per_s": _per_second(
                    run.effective_decode_tokens, model_device_s
                ),
                "model_device_raw_physical_tok_per_s": _per_second(
                    run.raw_decode_token_slots, model_device_s
                ),
                "run_scoped_effective_tok_per_s": _per_second(
                    run.effective_decode_tokens,
                    float(run.timing_s["run_scoped_scheduler_wall"]),
                ),
            },
            "useful_token_fraction": (
                run.effective_decode_tokens / run.raw_decode_token_slots
            ),
            "memory_bytes": {
                "baseline_allocated": memory_baseline,
                "peak_delta": _peak_memory_delta(self.device, memory_baseline),
            },
            "lab_wall_s": wall_s,
            "simulation_match": True,
            "excludes": [
                "vision and text prefill execution",
                "real prompt KV values",
                "page/frontend arrival timing",
                "result assembly and artifact writing",
            ],
        }

    def correctness(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = [
            item
            for item in items
            if (
                int(item["prompt_tokens"]) + self.args.correctness_steps
                < self.args.cache_length
                and int(item["generated_tokens"]) > self.args.correctness_steps
            )
        ]
        candidates.sort(
            key=lambda item: int(item["generated_tokens"]),
            reverse=True,
        )
        selected = candidates[: self.args.correctness_items]
        if len(selected) < self.args.correctness_items:
            raise ValueError("not enough corpus requests fit correctness shape")
        while len(selected) < self.args.batch_size:
            selected.append(selected[len(selected) % self.args.correctness_items])

        eager_cache = self.model.allocate_static_cache(
            batch_size=self.args.batch_size,
            cache_length=self.args.cache_length,
            device=self.device,
            dtype=self.dtype,
            init_mode="zeros",
        )
        compiled_cache = self.model.allocate_static_cache(
            batch_size=self.args.batch_size,
            cache_length=self.args.cache_length,
            device=self.device,
            dtype=self.dtype,
            init_mode="zeros",
            num_key_value_heads=self.runtime.cache_num_key_value_heads,
        )
        positions = torch.tensor(
            [int(item["prompt_tokens"]) for item in selected],
            device=self.device,
            dtype=torch.int64,
        )
        rope_deltas = torch.zeros(
            (self.args.batch_size, 1),
            device=self.device,
            dtype=torch.int64,
        )
        input_ids = torch.tensor(
            [int(item["first_token"]) for item in selected],
            device=self.device,
            dtype=torch.int64,
        ).view(-1, 1)

        logit_max_abs = 0.0
        logit_abs_sum = 0.0
        logit_count = 0
        kv_max_abs = 0.0
        kv_abs_sum = 0.0
        kv_count = 0
        argmax_matches = 0
        argmax_total = 0
        step_rows: list[dict[str, Any]] = []
        batch_indices = torch.arange(
            self.args.batch_size,
            device=self.device,
            dtype=torch.int64,
        )

        for step in range(self.args.correctness_steps):
            written_positions = positions.clone()
            eager_logits = self.reference_stage(
                input_ids,
                positions,
                rope_deltas,
                *eager_cache.flat_tensors(),
            )
            compiled_logits = self.runtime.fn(
                input_ids,
                positions,
                rope_deltas,
                *compiled_cache.flat_tensors(),
            )
            synchronize(self.device)
            logit_diff = (eager_logits.float() - compiled_logits.float()).abs()
            step_max = float(logit_diff.max().item())
            step_mean = float(logit_diff.mean().item())
            eager_argmax = torch.argmax(eager_logits[:, -1, :].float(), dim=-1)
            compiled_argmax = torch.argmax(
                compiled_logits[:, -1, :].float(), dim=-1
            )
            matches = int((eager_argmax == compiled_argmax).sum().item())
            argmax_matches += matches
            argmax_total += self.args.batch_size
            logit_max_abs = max(logit_max_abs, step_max)
            logit_abs_sum += float(logit_diff.sum().item())
            logit_count += logit_diff.numel()

            step_kv_max = 0.0
            step_kv_sum = 0.0
            step_kv_count = 0
            for eager_tensor, compiled_tensor in zip(
                eager_cache.flat_tensors(),
                compiled_cache.flat_tensors(),
            ):
                eager_written = eager_tensor[
                    batch_indices, :, written_positions, :
                ]
                compiled_written = compiled_tensor[
                    batch_indices, :, written_positions, :
                ]
                if compiled_written.shape[1] != eager_written.shape[1]:
                    groups = (
                        int(compiled_written.shape[1])
                        // int(eager_written.shape[1])
                    )
                    eager_written = eager_written[:, :, None, :].expand(
                        eager_written.shape[0],
                        eager_written.shape[1],
                        groups,
                        eager_written.shape[2],
                    ).reshape_as(compiled_written)
                diff = (eager_written.float() - compiled_written.float()).abs()
                step_kv_max = max(step_kv_max, float(diff.max().item()))
                step_kv_sum += float(diff.sum().item())
                step_kv_count += diff.numel()
            kv_max_abs = max(kv_max_abs, step_kv_max)
            kv_abs_sum += step_kv_sum
            kv_count += step_kv_count
            step_rows.append(
                {
                    "step": step,
                    "logit_max_abs": step_max,
                    "logit_mean_abs": step_mean,
                    "argmax_matches": matches,
                    "kv_written_max_abs": step_kv_max,
                    "kv_written_mean_abs": (
                        step_kv_sum / step_kv_count if step_kv_count else 0.0
                    ),
                }
            )

            next_tokens = []
            for item in selected:
                token_ids = [int(value) for value in item["token_ids"]]
                next_tokens.append(token_ids[min(step + 1, len(token_ids) - 1)])
            input_ids = torch.tensor(
                next_tokens,
                device=self.device,
                dtype=torch.int64,
            ).view(-1, 1)
            positions = positions + 1

        return {
            "contract": {
                "inputs": "recorded token paths and prompt positions",
                "prompt_kv_values": "independent zero prefixes",
                "rope_deltas": "zeros",
                "comparison": (
                    "baseline eager production stage versus selected "
                    "compiled optimization"
                ),
                "selected_optimization": self.optimization.name,
                "integration_gate": (
                    "real-crop end-to-end token parity remains required before "
                    "promoting a decode optimization"
                ),
            },
            "selected_request_ids": [
                str(item["request_id"])
                for item in selected[: self.args.correctness_items]
            ],
            "steps": self.args.correctness_steps,
            "batch_size": self.args.batch_size,
            "logits": {
                "max_abs": logit_max_abs,
                "mean_abs": logit_abs_sum / logit_count,
                "argmax_matches": argmax_matches,
                "argmax_total": argmax_total,
                "argmax_fraction": argmax_matches / argmax_total,
            },
            "written_kv": {
                "max_abs": kv_max_abs,
                "mean_abs": kv_abs_sum / kv_count,
            },
            "per_step": step_rows,
        }


def _reference_comparison(
    corpus: dict[str, Any],
    items: list[dict[str, Any]],
    simulation: dict[str, Any],
    *,
    batch_size: int,
    overflow: list[dict[str, Any]],
) -> dict[str, Any]:
    production_config = dict(corpus.get("production_configuration") or {})
    reference = dict(corpus.get("production_reference") or {})
    comparable = (
        not overflow
        and len(items) == int(corpus["distribution"]["requests"])
        and batch_size == int(production_config.get("batch_size", -1))
    )
    fields = (
        "graph_calls",
        "raw_decode_token_slots",
        "active_decode_token_slots",
        "effective_decode_tokens",
        "idle_decode_token_slots",
        "lookahead_decode_token_slots",
    )
    deltas = {}
    if comparable:
        for field in fields:
            reference_field = (
                "decode_graph_calls" if field == "graph_calls" else field
            )
            if reference_field not in reference:
                continue
            expected = int(reference[reference_field])
            actual = int(simulation[field])
            deltas[field] = {
                "lab": actual,
                "production": expected,
                "delta": actual - expected,
                "matches": actual == expected,
            }
    return {
        "comparable": comparable,
        "accounting": deltas,
        "all_accounting_matches": (
            all(row["matches"] for row in deltas.values()) if deltas else None
        ),
    }


def _write_report(
    args: argparse.Namespace,
    corpus: dict[str, Any],
    result: dict[str, Any],
    *,
    lab: TextDecodeLab | None,
) -> Path:
    uses_corpus = args.mode in ("simulate", "replay", "correctness")
    if args.output is not None:
        output = args.output.expanduser().resolve()
    else:
        name = args.name or time.strftime(f"{args.mode}_%Y%m%d_%H%M%S")
        output = (DEFAULT_OUTPUT_ROOT / f"{name}.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "text_decode_lab_result",
        "configuration": {
            "mode": args.mode,
            "corpus": (
                str(args.corpus.expanduser().resolve()) if uses_corpus else None
            ),
            "corpus_sha256": (
                _sha256(args.corpus.expanduser().resolve())
                if uses_corpus
                else None
            ),
            "batch_size": args.batch_size,
            "cache_length": args.cache_length,
            "dtype": args.dtype,
            "backend": args.backend,
            "decode_optimization": args.decode_optimization,
            "cache_dir": str(args.cache_dir.expanduser().resolve()),
            "active_slots": args.active_slots,
            "profile_position": args.profile_position,
            "tail_positions": list(args.tail_positions),
            "tail_canary_value": args.tail_canary_value,
            "profile_dir": str(args.profile_dir.expanduser().resolve()),
            "profile_metric": args.profile_metric,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "overflow_policy": args.overflow_policy,
            "allow_compile": bool(args.allow_compile),
        },
        "corpus_contract": corpus.get("contract"),
        "result": result,
    }
    if lab is not None:
        payload["setup"] = {
            "total_s": lab.setup_s,
            "model_load_s": lab.model_load_s,
            "weight_format_s": lab.weight_format_s,
            "runtime_setup_s": lab.runtime_setup_s,
            "weight_format": lab.weight_format,
            "runtime_metadata": lab.runtime.metadata,
            "runtime_setup_detail_s": lab.runtime.setup_timing_s,
        }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _print_result(mode: str, result: dict[str, Any]) -> None:
    if mode == "simulate":
        row = result["simulation"]
        print(
            "DECODE_SIMULATION "
            f"requests={row['requests']} calls={row['graph_calls']} "
            f"raw_slots={row['raw_decode_token_slots']} "
            f"effective={row['effective_decode_tokens']} "
            f"useful={row['decode_useful_token_fraction']:.4f}"
        )
    elif mode == "profile":
        throughput = result["throughput"]
        latency = result["latency_ms"]
        print(
            "DECODE_PROFILE "
            f"mean_ms={latency['mean']:.4f} "
            f"raw_tok_s={throughput['raw_physical_tok_per_s']:.1f} "
            f"active_tok_s={throughput['active_tok_per_s']:.1f}"
        )
    elif mode == "torch_profile":
        print(
            "DECODE_TORCH_PROFILE "
            f"steps={result['captured_steps']} "
            f"metric={result['metric']} "
            f"profile_dir={result['profile_dir']}"
        )
    elif mode == "boundary":
        print(
            "DECODE_BOUNDARY "
            f"position={result['shape']['cache_position']} "
            f"effective_length={result['shape']['effective_length']} "
            f"attention={result['attention']} "
            f"elapsed_s={result['elapsed_s']:.6f}"
        )
    elif mode == "tail_invariance":
        print(
            "DECODE_TAIL_INVARIANCE_RESULT "
            f"batch_size={result['batch_size']} "
            f"zero_repeat_exact={result['all_zero_repeats_exact']} "
            f"row0_stale_exact={result['all_row0_stale_exact']} "
            f"all_stale_exact={result['all_rows_stale_exact']}"
        )
    elif mode == "replay":
        scheduler = result["scheduler"]
        throughput = result["throughput"]
        print(
            "DECODE_REPLAY "
            f"requests={result['workload']['requests']} "
            f"calls={scheduler['graph_calls']} "
            "effective_tok_s="
            f"{throughput['continuous_decode_effective_tok_per_s']:.1f} "
            "raw_tok_s="
            f"{throughput['continuous_decode_raw_physical_tok_per_s']:.1f} "
            f"useful={result['useful_token_fraction']:.4f}"
        )
    else:
        logits = result["logits"]
        print(
            "DECODE_CORRECTNESS "
            f"steps={result['steps']} "
            f"logit_mean_abs={logits['mean_abs']:.6f} "
            f"logit_max_abs={logits['max_abs']:.6f} "
            f"argmax={logits['argmax_matches']}/{logits['argmax_total']}"
        )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode in ("profile", "torch_profile", "boundary", "tail_invariance"):
        corpus: dict[str, Any] = {
            "contract": {
                "corpus_used": False,
                "scope": (
                    "compiled full-decoder masked-tail invariance"
                    if args.mode == "tail_invariance"
                    else (
                        "synthetic full-decoder cache-position boundary"
                        if args.mode == "boundary"
                        else "synthetic full-decoder throughput profile"
                    )
                ),
            }
        }
        selected_items: list[dict[str, Any]] = []
    else:
        corpus, selected_items = _load_corpus(args.corpus, args.max_items)

    lab: TextDecodeLab | None = None
    if args.mode in ("profile", "torch_profile", "boundary", "tail_invariance"):
        lab = TextDecodeLab(args)
        if args.mode == "profile":
            result: dict[str, Any] = lab.profile()
        elif args.mode == "torch_profile":
            result = lab.torch_profile()
        elif args.mode == "boundary":
            result = lab.boundary()
        else:
            result = lab.tail_invariance()
    else:
        items, overflow = _filter_for_cache(
            selected_items,
            cache_length=args.cache_length,
            overflow_policy=args.overflow_policy,
        )
        simulation = simulate_scheduler(items, batch_size=args.batch_size)
        reference = _reference_comparison(
            corpus,
            items,
            simulation,
            batch_size=args.batch_size,
            overflow=overflow,
        )

    if args.mode == "simulate":
        result: dict[str, Any] = {
            "simulation": simulation,
            "overflow": {
                "policy": args.overflow_policy,
                "requests": len(overflow),
                "maximum_required_cache_tokens": (
                    max(
                        int(item["replay_required_cache_tokens"])
                        for item in overflow
                    )
                    if overflow
                    else None
                ),
            },
            "production_reference": reference,
        }
    elif args.mode not in ("profile", "torch_profile", "boundary", "tail_invariance"):
        lab = TextDecodeLab(args)
        if args.mode == "replay":
            result = lab.replay(items, overflow)
            result["production_reference"] = reference
        else:
            result = lab.correctness(items)

    output = _write_report(args, corpus, result, lab=lab)
    _print_result(args.mode, result)
    print(f"output={output}")


if __name__ == "__main__":
    main()
