#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from statistics import mean

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings(
    "ignore",
    message=r"The following torchair config or properties may not take effect.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"TypedStorage is deprecated.*",
    category=UserWarning,
)

import torch

try:
    import torch_npu
    import torch_npu.profiler as npu_prof
except ModuleNotFoundError:
    torch_npu = None
    npu_prof = None


BYTES_PER_GIB = 1024**3


def require_torch_npu() -> None:
    if torch_npu is None or npu_prof is None:
        raise RuntimeError("torch_npu is required for NPU benchmark/profiling runs")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local Qwen 3.0 prefill and decode.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Write a tiny Python function that adds two numbers.")
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--static-kv-cache-len", type=int, default=65536)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--compile-decode", action="store_true")
    parser.add_argument("--compile-decode-dynamic", action="store_true")
    parser.add_argument("--npugraph-decode", action="store_true")
    parser.add_argument("--decode-increfa-mode", choices=("mask", "actual_seq_lengths"), default="mask")
    parser.add_argument("--prefill-warmups", type=int, default=1)
    parser.add_argument("--prefill-repeats", type=int, default=3)
    parser.add_argument("--decode-warmups", type=int, default=1)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--profile", choices=("none", "prefill", "decode", "both"), default="none")
    parser.add_argument("--profile-dir", default="/tmp/qwen3_0_bench_profile")
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--json-out")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class StageLogger:
    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self.started = time.perf_counter()
        self.last = self.started
        self.count = 0
        if self.enabled:
            print(f"[start] {datetime.now().isoformat(timespec='seconds')}", flush=True)

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self.count += 1
        print(
            f"[stage {self.count:02d} +{now - self.last:.3f}s total={now - self.started:.3f}s]\t{message}",
            flush=True,
        )
        self.last = now


def configure_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"The following torchair config or properties may not take effect.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"TypedStorage is deprecated.*",
        category=UserWarning,
    )


@contextlib.contextmanager
def suppress_stdout():
    sys.stdout.flush()
    saved_stdout_fd = os.dup(1)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
        yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)
        sys.stdout.flush()


def call_with_filtered_stderr(fn, *, suppressed_substrings: tuple[str, ...]):
    sys.stderr.flush()
    saved_stderr_fd = os.dup(2)
    with tempfile.TemporaryFile(mode="w+t") as captured:
        os.dup2(captured.fileno(), 2)
        try:
            result = fn()
        except BaseException:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
            captured.seek(0)
            sys.stderr.write(captured.read())
            sys.stderr.flush()
            raise
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        captured.seek(0)
        for line in captured:
            if any(needle in line for needle in suppressed_substrings):
                continue
            sys.stderr.write(line)
        sys.stderr.flush()
        return result


def sync() -> None:
    torch.npu.synchronize()


def time_call(fn) -> float:
    sync()
    started = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - started


def gib(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return float(num_bytes) / BYTES_PER_GIB


def npu_memory_snapshot(label: str) -> dict[str, int | float | str]:
    sync()
    free_bytes, total_bytes = torch.npu.mem_get_info()
    allocated_bytes = int(torch.npu.memory_allocated())
    reserved_bytes = int(torch.npu.memory_reserved())
    max_allocated_bytes = int(torch.npu.max_memory_allocated())
    max_reserved_bytes = int(torch.npu.max_memory_reserved())
    return {
        "label": label,
        "allocated_bytes": allocated_bytes,
        "reserved_bytes": reserved_bytes,
        "max_allocated_bytes": max_allocated_bytes,
        "max_reserved_bytes": max_reserved_bytes,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_gib": gib(allocated_bytes),
        "reserved_gib": gib(reserved_bytes),
        "max_allocated_gib": gib(max_allocated_bytes),
        "max_reserved_gib": gib(max_reserved_bytes),
        "free_gib": gib(free_bytes),
        "total_gib": gib(total_bytes),
    }


def print_memory_snapshot(snapshot: dict[str, int | float | str]) -> None:
    print(
        f"memory {snapshot['label']}: "
        f"alloc={snapshot['allocated_gib']:.2f}GiB "
        f"reserved={snapshot['reserved_gib']:.2f}GiB "
        f"max_alloc={snapshot['max_allocated_gib']:.2f}GiB "
        f"max_reserved={snapshot['max_reserved_gib']:.2f}GiB "
        f"free={snapshot['free_gib']:.2f}GiB/{snapshot['total_gib']:.2f}GiB"
    )


def reset_dynamo_counters() -> None:
    torch._dynamo.utils.counters.clear()


def dynamo_counters_snapshot() -> dict[str, dict[str, int]]:
    counters = torch._dynamo.utils.counters
    snapshot = {}
    for group, values in counters.items():
        if not values:
            continue
        snapshot[str(group)] = {str(name): int(value) for name, value in values.items()}
    return snapshot


def print_dynamo_counters(counters: dict[str, dict[str, int]]) -> None:
    if not counters:
        print("dynamo counters: <empty>")
        return
    print("dynamo counters:")
    for group, values in sorted(counters.items()):
        compact = ", ".join(f"{name}={value}" for name, value in sorted(values.items()))
        print(f"  {group}: {compact}")


def summarize_seconds(
    values: list[float],
    *,
    tokens: int,
    batches: int = 1,
) -> dict[str, float]:
    total = sum(values)
    return {
        "runs": len(values),
        "mean_sec": mean(values) if values else 0.0,
        "min_sec": min(values) if values else 0.0,
        "max_sec": max(values) if values else 0.0,
        "total_sec": total,
        "tok_s": (tokens * len(values) / total) if total > 0.0 else 0.0,
        "batch_s": (batches * len(values) / total) if total > 0.0 else 0.0,
    }


def build_prefill_input_ids(
    runner: LocalQwen30Runner,
    prompt: str,
    prefill_tokens: int,
    batch_size: int,
) -> torch.Tensor:
    input_ids = runner.encode_prompt(prompt)
    repeats = (prefill_tokens + input_ids.shape[1] - 1) // input_ids.shape[1]
    input_ids = input_ids.repeat(1, repeats)[:, :prefill_tokens]
    return input_ids.expand(batch_size, -1).contiguous()


def run_prefill_once(runner: LocalQwen30Runner, input_ids: torch.Tensor):
    with torch.inference_mode():
        return runner.model.prefill(input_ids, static_kv_cache_len=runner.static_kv_cache_len)


def prepare_decode_state(runner: LocalQwen30Runner, input_ids: torch.Tensor):
    with torch.inference_mode():
        key_caches, value_caches = runner.model.prefill(input_ids, static_kv_cache_len=runner.static_kv_cache_len)
        decode_input = runner.make_initial_decode_input(input_ids)
    runner.mark_static_decode_state(key_caches, value_caches)
    return decode_input, key_caches, value_caches


def run_decode_loop(
    runner: LocalQwen30Runner,
    input_ids: torch.Tensor,
    next_id: torch.Tensor,
    key_caches: tuple[torch.Tensor, ...],
    value_caches: tuple[torch.Tensor, ...],
    *,
    decode_steps: int,
    decode_one=None,
) -> torch.Tensor:
    if decode_one is None:
        decode_one = runner.decode_one
    with torch.inference_mode():
        generated = []
        for decode_position in range(input_ids.shape[1] - 1, input_ids.shape[1] - 1 + decode_steps):
            cache_position = torch.full(
                (input_ids.shape[0],),
                decode_position,
                device=input_ids.device,
                dtype=torch.long,
            )
            actual_seq_length = decode_position + 1 if runner.decode_increfa_mode == "actual_seq_lengths" else None
            next_id, key_caches, value_caches = decode_one(
                next_id,
                cache_position,
                key_caches,
                value_caches,
                actual_seq_length=actual_seq_length,
            )
            generated.append(next_id)
        return torch.cat(generated, dim=-1)


class NPUGraphDecodeRunner:
    def __init__(
        self,
        runner: LocalQwen30Runner,
        input_ids: torch.Tensor,
        next_id: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        *,
        decode_steps: int,
    ):
        require_torch_npu()
        if runner.decode_increfa_mode != "mask":
            raise ValueError("NPUGraph decode currently supports only decode_increfa_mode='mask'")
        self.runner = runner
        self.decode_steps = int(decode_steps)
        self.static_input_ids = next_id.clone()
        self.static_cache_position = torch.empty((1,), device=input_ids.device, dtype=torch.long)
        self.decode_positions = torch.arange(
            input_ids.shape[1] - 1,
            input_ids.shape[1] - 1 + decode_steps,
            device=input_ids.device,
            dtype=torch.long,
        )
        self.initial_next_id = next_id.clone()
        self.key_caches = key_caches
        self.value_caches = value_caches
        self.graph = torch_npu.npu.NPUGraph()
        self.output_next_id = None
        self.output_key_caches = None
        self.output_value_caches = None
        self.capture_sec = self.capture()

    def capture(self) -> float:
        self.static_cache_position.copy_(self.decode_positions[:1])
        side_stream = torch.npu.Stream(device=self.static_input_ids.device)
        sync()
        started = time.perf_counter()
        with torch.npu.stream(side_stream):
            with torch.inference_mode():
                self.graph.capture_begin()
                self.output_next_id = self.runner.model.decode(
                    self.static_input_ids,
                    self.static_cache_position,
                    self.key_caches,
                    self.value_caches,
                    None,
                )
                self.graph.capture_end()
        sync()
        return time.perf_counter() - started

    def run_loop(self) -> torch.Tensor:
        if self.output_next_id is None:
            raise RuntimeError("NPUGraph decode has not been captured")
        next_id = self.initial_next_id
        generated = torch.empty(
            (next_id.shape[0], self.decode_steps),
            device=next_id.device,
            dtype=next_id.dtype,
        )
        for step in range(self.decode_steps):
            self.static_input_ids.copy_(next_id)
            self.static_cache_position.copy_(self.decode_positions[step : step + 1])
            self.graph.replay()
            next_id = self.output_next_id
            generated[:, step : step + 1].copy_(next_id)
        return generated


def prepare_npugraph_decode_runner(
    runner: LocalQwen30Runner,
    input_ids: torch.Tensor,
    *,
    decode_steps: int,
) -> NPUGraphDecodeRunner:
    next_id, key_caches, value_caches = prepare_decode_state(runner, input_ids)
    return NPUGraphDecodeRunner(
        runner,
        input_ids,
        next_id,
        key_caches,
        value_caches,
        decode_steps=decode_steps,
    )


def benchmark_decode_npugraph(
    runner: LocalQwen30Runner,
    input_ids: torch.Tensor,
    *,
    warmups: int,
    repeats: int,
    decode_steps: int,
    log: StageLogger,
) -> dict:
    log.log(f"checking NPUGraph decode correctness: steps={decode_steps}")
    ref_next_id, ref_key_caches, ref_value_caches = prepare_decode_state(runner, input_ids)
    reference = run_decode_loop(
        runner,
        input_ids,
        ref_next_id,
        ref_key_caches,
        ref_value_caches,
        decode_steps=decode_steps,
    )
    check_runner = prepare_npugraph_decode_runner(runner, input_ids, decode_steps=decode_steps)
    candidate = check_runner.run_loop()
    sync()
    token_mismatch_count = int((candidate != reference).sum().item())
    if token_mismatch_count:
        raise RuntimeError(f"NPUGraph decode output mismatch: token_mismatch_count={token_mismatch_count}")

    log.log(f"warming up NPUGraph decode: runs={warmups} steps={decode_steps}")
    capture_timings = []
    for _ in range(warmups):
        graph_runner = prepare_npugraph_decode_runner(runner, input_ids, decode_steps=decode_steps)
        capture_timings.append(graph_runner.capture_sec)
        time_call(graph_runner.run_loop)
    log.log(f"timing NPUGraph decode: repeats={repeats} steps={decode_steps}")
    timings = []
    for _ in range(repeats):
        graph_runner = prepare_npugraph_decode_runner(runner, input_ids, decode_steps=decode_steps)
        capture_timings.append(graph_runner.capture_sec)
        timings.append(time_call(graph_runner.run_loop))
    summary = summarize_seconds(
        timings,
        tokens=int(input_ids.shape[0]) * decode_steps,
        batches=decode_steps,
    )
    summary["npugraph_capture_sec_mean"] = mean(capture_timings) if capture_timings else 0.0
    summary["npugraph_capture_sec_min"] = min(capture_timings) if capture_timings else 0.0
    summary["npugraph_capture_sec_max"] = max(capture_timings) if capture_timings else 0.0
    summary["npugraph_eager_exact_match"] = True
    summary["npugraph_eager_token_mismatch_count"] = token_mismatch_count
    return summary


def benchmark_prefill(
    runner: LocalQwen30Runner,
    input_ids: torch.Tensor,
    *,
    warmups: int,
    repeats: int,
    log: StageLogger,
) -> dict:
    log.log(f"warming up prefill: runs={warmups}")
    for _ in range(warmups):
        run_prefill_once(runner, input_ids)
    log.log(f"timing prefill: repeats={repeats}")
    timings = [time_call(lambda: run_prefill_once(runner, input_ids)) for _ in range(repeats)]
    return summarize_seconds(
        timings,
        tokens=int(input_ids.numel()),
        batches=1,
    )


def benchmark_decode(
    runner: LocalQwen30Runner,
    input_ids: torch.Tensor,
    *,
    warmups: int,
    repeats: int,
    decode_steps: int,
    log: StageLogger,
) -> dict:
    compile_first_call_sec = None
    parity = None
    if runner.compile_decode:
        log.log(f"checking compiled/eager decode parity: steps={decode_steps}")
        eager_next_id, eager_key_caches, eager_value_caches = prepare_decode_state(
            runner, input_ids
        )
        compiled_next_id, compiled_key_caches, compiled_value_caches = (
            prepare_decode_state(runner, input_ids)
        )
        eager_tokens = run_decode_loop(
            runner,
            input_ids,
            eager_next_id,
            eager_key_caches,
            eager_value_caches,
            decode_steps=decode_steps,
            decode_one=runner.decode_one_eager,
        )
        sync()
        started = time.perf_counter()
        compiled_tokens = run_decode_loop(
            runner,
            input_ids,
            compiled_next_id,
            compiled_key_caches,
            compiled_value_caches,
            decode_steps=decode_steps,
        )
        sync()
        compile_first_call_sec = time.perf_counter() - started
        token_mismatch_count = int((compiled_tokens != eager_tokens).sum().item())
        kv_max_abs = 0.0
        for eager_cache, compiled_cache in zip(
            (*eager_key_caches, *eager_value_caches),
            (*compiled_key_caches, *compiled_value_caches),
        ):
            kv_max_abs = max(
                kv_max_abs,
                float((compiled_cache.float() - eager_cache.float()).abs().max().item()),
            )
        parity = {
            "steps": int(decode_steps),
            "token_mismatch_count": token_mismatch_count,
            "token_exact_match": token_mismatch_count == 0,
            "kv_max_abs": kv_max_abs,
        }
        if token_mismatch_count:
            raise RuntimeError(
                "compiled decode token mismatch: "
                f"token_mismatch_count={token_mismatch_count}"
            )

    log.log(f"warming up decode: runs={warmups} steps={decode_steps}")
    warmup_timings = []
    for _ in range(warmups):
        next_id, key_caches, value_caches = prepare_decode_state(runner, input_ids)
        warmup_timings.append(
            time_call(
                lambda: run_decode_loop(
                    runner,
                    input_ids,
                    next_id,
                    key_caches,
                    value_caches,
                    decode_steps=decode_steps,
                )
            )
        )
    log.log(f"timing decode: repeats={repeats} steps={decode_steps}")
    timings = []
    for _ in range(repeats):
        next_id, key_caches, value_caches = prepare_decode_state(runner, input_ids)
        timings.append(
            time_call(
                lambda: run_decode_loop(
                    runner,
                    input_ids,
                    next_id,
                    key_caches,
                    value_caches,
                    decode_steps=decode_steps,
                )
            )
    )
    summary = summarize_seconds(
        timings,
        tokens=int(input_ids.shape[0]) * decode_steps,
        batches=decode_steps,
    )
    summary["compile_decode_first_call_sec"] = compile_first_call_sec
    if parity is not None:
        summary["compiled_eager_parity"] = parity
    return summary


def profiler_experimental_config():
    require_torch_npu()
    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
        export_type=npu_prof.ExportType.Text,
    )


def find_profile_root(profile_dir: Path) -> Path:
    roots = sorted(path for path in profile_dir.iterdir() if path.is_dir())
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one profiler run directory in {profile_dir}, found {len(roots)}")
    return roots[0]


def parse_shape_groups(raw: str | None) -> list[tuple[int, ...]]:
    if not raw:
        return []
    text = raw.strip().strip('"')
    if not text or text == "N/A":
        return []
    groups = []
    for group in text.split(";"):
        dims = []
        for dim in group.strip().strip('"').split(","):
            dim = dim.strip()
            if not dim or dim == "-1":
                return []
            try:
                dims.append(int(dim))
            except ValueError:
                return []
        if dims:
            groups.append(tuple(dims))
    return groups


def last_dim(shape: tuple[int, ...] | None) -> int | None:
    return shape[-1] if shape else None


class KernelAttributor:
    def __init__(self, config):
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.intermediate_size)
        self.q_dim = int(config.num_attention_heads * config.head_dim)
        self.kv_dim = int(config.num_key_value_heads * config.head_dim)

    def guess(self, *, name: str, op_type: str, input_shapes: str | None, output_shapes: str | None) -> str:
        lowered = f"{name} {op_type}".lower()
        inputs = parse_shape_groups(input_shapes)
        outputs = parse_shape_groups(output_shapes)
        output_last = last_dim(outputs[0] if outputs else None)

        if "increflashattention" in lowered:
            return "incre_flash_attention"
        if "scatter" in lowered:
            return "scatter_update"
        if "cast" in lowered:
            return "cast"
        if "transpose" in lowered:
            return "transpose"
        if "concat" in lowered or "cat" in lowered:
            return "concat"
        if "softmax" in lowered:
            return "attention_softmax"
        if "where" in lowered or "less" in lowered or "greater" in lowered or "triu" in lowered:
            return "mask_or_compare"
        if "gather" in lowered:
            if inputs and inputs[0] == (self.vocab_size, self.hidden_size):
                return "embed"
            return "other"
        if "batchmatmul" in lowered:
            if inputs and last_dim(inputs[0]) == self.q_dim and output_last == self.hidden_size:
                return "attn.o_proj"
            return "attention_batch_matmul"
        if "reducemean" in lowered or "rsqrt" in lowered or "sqrt" in lowered:
            return "rmsnorm_or_reduce"
        if "automaticbufferfusionop" in lowered:
            return "elementwise_fusion"
        if "copy" in lowered or "assign" in lowered or "update" in lowered:
            return "copy_or_update"
        if "matmul" in lowered:
            return self.guess_matmul(inputs, output_last)
        if any(token in lowered for token in ("mul", "add", "sub", "div", "neg", "pow", "floor")):
            return "elementwise"
        return "other"

    def guess_matmul(self, inputs: list[tuple[int, ...]], output_last: int | None) -> str:
        weight = inputs[1] if len(inputs) >= 2 else None
        if output_last == self.vocab_size:
            return "lm_head"
        if weight == (self.vocab_size, self.hidden_size):
            return "lm_head"
        if weight == (self.intermediate_size, self.hidden_size):
            return "mlp.gate_or_up"
        if weight == (self.hidden_size, self.intermediate_size):
            return "mlp.down"
        if self.kv_dim != self.hidden_size and weight == (self.kv_dim, self.hidden_size):
            return "attn.kv_proj"
        if weight == (self.q_dim, self.hidden_size):
            return "attn.qkv_or_o_or_square"
        if weight == (self.hidden_size, self.hidden_size):
            return "attn.qkv_or_o_or_square"
        if inputs and last_dim(inputs[0]) == self.intermediate_size and output_last == self.hidden_size:
            return "mlp.down"
        if inputs and last_dim(inputs[0]) == self.hidden_size and output_last == self.intermediate_size:
            return "mlp.gate_or_up"
        return "other"


def load_top_op_types(csv_path: Path, topn: int) -> list[dict[str, float | str]]:
    rows = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                total_time = float((row.get("Total Time(us)", "0") or "0").strip())
            except ValueError:
                continue
            rows.append(
                {
                    "op_type": row.get("OP Type", "").strip(),
                    "core_type": row.get("Core Type", "").strip(),
                    "count": int(float((row.get("Count", "0") or "0").strip())),
                    "total_time_us": total_time,
                    "avg_time_us": float((row.get("Avg Time(us)", "0") or "0").strip()),
                    "ratio_percent": float((row.get("Ratio(%)", "0") or "0").strip()),
                }
            )
    return sorted(rows, key=lambda row: row["total_time_us"], reverse=True)[:topn]


def sum_rows_by_key(rows: Iterable[dict], key: str, value_key: str) -> list[dict[str, float | int | str]]:
    totals: dict[str, dict[str, float | int | str]] = {}
    for row in rows:
        name = str(row.get(key) or "other")
        value = float(row.get(value_key) or 0.0)
        item = totals.setdefault(name, {key: name, "total_time_us": 0.0, "count": 0})
        item["total_time_us"] = float(item["total_time_us"]) + value
        item["count"] = int(item["count"]) + int(row.get("count", 1) or 1)
    return sorted(totals.values(), key=lambda item: float(item["total_time_us"]), reverse=True)


def load_kernel_details(
    csv_path: Path,
    *,
    attributor: KernelAttributor,
    topn: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, float | int | str]] = {}
    core_rows = []
    module_rows = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                duration = float((row.get("Duration(us)", "0") or "0").strip())
            except ValueError:
                continue
            name = row.get("Name", "").strip() or "<empty>"
            op_type = row.get("Type", "").strip() or "<empty>"
            core_type = row.get("Accelerator Core", "").strip() or "<empty>"
            input_shapes = row.get("Input Shapes", "").strip()
            output_shapes = row.get("Output Shapes", "").strip()
            module_guess = attributor.guess(
                name=name,
                op_type=op_type,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
            )
            core_rows.append({"core_type": core_type, "total_time_us": duration})
            module_rows.append({"module_guess": module_guess, "total_time_us": duration})
            key = (name, op_type, core_type, input_shapes, output_shapes, module_guess)
            item = grouped.setdefault(
                key,
                {
                    "name": name,
                    "op_type": op_type,
                    "core_type": core_type,
                    "input_shapes": input_shapes,
                    "output_shapes": output_shapes,
                    "module_guess": module_guess,
                    "total_time_us": 0.0,
                    "count": 0,
                    "avg_time_us": 0.0,
                },
            )
            item["total_time_us"] = float(item["total_time_us"]) + duration
            item["count"] = int(item["count"]) + 1
            item["avg_time_us"] = float(item["total_time_us"]) / int(item["count"])
    top_kernels = sorted(grouped.values(), key=lambda item: float(item["total_time_us"]), reverse=True)[:topn]
    core_summary = sum_rows_by_key(core_rows, "core_type", "total_time_us")
    module_summary = sum_rows_by_key(module_rows, "module_guess", "total_time_us")
    return top_kernels, core_summary, module_summary


def load_top_operator_rows(
    csv_path: Path,
    *,
    attributor: KernelAttributor,
    topn: int,
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str, str], dict[str, float | int | str]] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                duration = float((row.get("Device Total Duration(us)", "0") or "0").strip())
            except ValueError:
                continue
            name = row.get("Name", "").strip() or "<empty>"
            input_shapes = row.get("Input Shapes", "").strip()
            module_guess = attributor.guess(name=name, op_type=name, input_shapes=input_shapes, output_shapes=None)
            key = (name, input_shapes, module_guess)
            item = grouped.setdefault(
                key,
                {
                    "name": name,
                    "input_shapes": input_shapes,
                    "module_guess": module_guess,
                    "total_time_us": 0.0,
                    "count": 0,
                    "avg_time_us": 0.0,
                },
            )
            item["total_time_us"] = float(item["total_time_us"]) + duration
            item["count"] = int(item["count"]) + 1
            item["avg_time_us"] = float(item["total_time_us"]) / int(item["count"])
    return sorted(grouped.values(), key=lambda item: float(item["total_time_us"]), reverse=True)[:topn]


def load_trace_event_summary(trace_path: Path, *, topn: int) -> list[dict[str, float | int | str]]:
    payload = json.loads(trace_path.read_text())
    grouped: dict[str, dict[str, float | int | str]] = {}
    for event in payload.get("traceEvents", []):
        name = str(event.get("name") or "<empty>")
        duration = event.get("dur")
        if duration is None:
            continue
        try:
            duration_us = float(duration)
        except (TypeError, ValueError):
            continue
        item = grouped.setdefault(name, {"name": name, "total_time_us": 0.0, "count": 0, "avg_time_us": 0.0})
        item["total_time_us"] = float(item["total_time_us"]) + duration_us
        item["count"] = int(item["count"]) + 1
        item["avg_time_us"] = float(item["total_time_us"]) / int(item["count"])
    return sorted(grouped.values(), key=lambda item: float(item["total_time_us"]), reverse=True)[:topn]


def collect_profile_summary(profile_dir: Path, *, topn: int, attributor: KernelAttributor) -> dict:
    run_root = find_profile_root(profile_dir)
    summary_dir = run_root / "ASCEND_PROFILER_OUTPUT"
    operator_csv = summary_dir / "operator_details.csv"
    kernel_csv = summary_dir / "kernel_details.csv"
    op_stat_csv = summary_dir / "op_statistic.csv"
    trace_json = run_root / "trace_view.json"
    summary = {
        "profile_root": str(run_root),
        "operator_details_csv": str(operator_csv),
        "kernel_details_csv": str(kernel_csv),
        "op_statistic_csv": str(op_stat_csv),
        "trace_view_json": str(trace_json),
        "profile_parse_mode": "kernel_details" if kernel_csv.exists() and op_stat_csv.exists() else "trace_view",
        "top_op_types_total_time_us": [],
        "top_operator_device_total_us": [],
        "top_kernel_duration_us": [],
        "core_type_summary_total_us": [],
        "module_guess_summary_total_us": [],
        "top_trace_events_total_time_us": [],
    }
    if not kernel_csv.exists() or not op_stat_csv.exists():
        if not trace_json.exists():
            raise RuntimeError(
                "Profiler output did not include kernel_details.csv/op_statistic.csv or trace_view.json "
                f"under {run_root}"
            )
        summary["top_trace_events_total_time_us"] = load_trace_event_summary(trace_json, topn=topn)
        if operator_csv.exists():
            summary["top_operator_device_total_us"] = load_top_operator_rows(
                operator_csv,
                attributor=attributor,
                topn=topn,
            )
        return summary

    top_kernels, core_summary, module_summary = load_kernel_details(kernel_csv, attributor=attributor, topn=topn)
    summary.update(
        {
            "top_op_types_total_time_us": load_top_op_types(op_stat_csv, topn),
            "top_operator_device_total_us": load_top_operator_rows(operator_csv, attributor=attributor, topn=topn)
            if operator_csv.exists()
            else [],
            "top_kernel_duration_us": top_kernels,
            "core_type_summary_total_us": core_summary,
            "module_guess_summary_total_us": module_summary,
        }
    )
    return {
        **summary,
    }


def profile_once(label: str, fn, *, profile_dir: Path, topn: int, attributor: KernelAttributor) -> dict:
    require_torch_npu()
    phase_dir = profile_dir / label
    shutil.rmtree(phase_dir, ignore_errors=True)
    phase_dir.mkdir(parents=True, exist_ok=True)
    schedule = npu_prof.schedule(wait=0, warmup=1, active=1, repeat=1)
    with suppress_stdout():
        elapsed = time_call(
            lambda: _profile_body(fn, phase_dir=phase_dir, schedule=schedule)
        )
    return {
        "phase": label,
        "profiled_sec": elapsed,
        **collect_profile_summary(phase_dir, topn=topn, attributor=attributor),
    }


def _profile_body(fn, *, phase_dir: Path, schedule) -> None:
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        experimental_config=profiler_experimental_config(),
        on_trace_ready=npu_prof.tensorboard_trace_handler(str(phase_dir), analyse_flag=True),
    ) as prof:
        fn()
        sync()
        prof.step()
        fn()
        sync()
        prof.step()
    sync()


def print_timing(label: str, summary: dict) -> None:
    print(
        f"{label}: mean={summary['mean_sec']:.6f}s "
        f"min={summary['min_sec']:.6f}s max={summary['max_sec']:.6f}s "
        f"tok/s={summary['tok_s']:.2f} batch/s={summary['batch_s']:.2f}"
    )
    if summary.get("compile_decode_first_call_sec") is not None:
        print(f"{label}: compile_first_call={summary['compile_decode_first_call_sec']:.6f}s")
    if summary.get("npugraph_capture_sec_mean") is not None:
        print(
            f"{label}: npugraph_capture_mean={summary['npugraph_capture_sec_mean']:.6f}s "
            f"min={summary['npugraph_capture_sec_min']:.6f}s "
            f"max={summary['npugraph_capture_sec_max']:.6f}s"
        )


def print_profile(label: str, summary: dict) -> None:
    print(f"\n{label} profile: {summary['profile_root']}")
    if summary.get("profile_parse_mode") == "trace_view":
        print("profile parse mode: trace_view")
        print("top trace events:")
        for row in summary["top_trace_events_total_time_us"]:
            print(f"  {row['name']} total_us={row['total_time_us']:.1f} count={row['count']}")
        if summary["top_operator_device_total_us"]:
            print("top operators:")
            for row in summary["top_operator_device_total_us"]:
                print(
                    f"  {row['name']} guess={row['module_guess']} "
                    f"total_us={row['total_time_us']:.1f} count={row['count']}"
                )
        return
    print("core type summary:")
    for row in summary["core_type_summary_total_us"]:
        print(f"  {row['core_type']} total_us={row['total_time_us']:.1f} count={row['count']}")
    print("top module guesses:")
    for row in summary["module_guess_summary_total_us"]:
        print(f"  {row['module_guess']} total_us={row['total_time_us']:.1f} count={row['count']}")
    print("top op types:")
    for row in summary["top_op_types_total_time_us"]:
        print(
            f"  {row['op_type']} {row['core_type']} "
            f"total_us={row['total_time_us']:.1f} ratio={row['ratio_percent']:.2f}% count={row['count']}"
        )
    print("top operators:")
    for row in summary["top_operator_device_total_us"]:
        print(
            f"  {row['name']} guess={row['module_guess']} "
            f"total_us={row['total_time_us']:.1f} count={row['count']}"
        )
    print("top kernels:")
    for row in summary["top_kernel_duration_us"]:
        print(
            f"  {row['name']} {row['op_type']} {row['core_type']} "
            f"guess={row['module_guess']} total_us={row['total_time_us']:.1f} count={row['count']}"
        )


def main() -> None:
    args = parse_args()
    configure_warnings()
    from run_local_qwen3_0 import LocalQwen30Runner

    log = StageLogger(enabled=args.verbose)
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    if device.type == "npu":
        require_torch_npu()
        torch.npu.set_device(device)
        torch.npu.set_compile_mode(jit_compile=False)
    if args.npugraph_decode and args.compile_decode:
        raise ValueError("--npugraph-decode and --compile-decode are mutually exclusive")
    if args.npugraph_decode and args.compile_decode_dynamic:
        raise ValueError("--npugraph-decode and --compile-decode-dynamic are mutually exclusive")
    if args.npugraph_decode and args.decode_increfa_mode != "mask":
        raise ValueError("--npugraph-decode currently requires --decode-increfa-mode mask")
    if args.prefill_tokens + args.decode_steps > args.static_kv_cache_len:
        raise ValueError(
            "prefill_tokens + decode_steps exceeds static_kv_cache_len "
            f"({args.prefill_tokens} + {args.decode_steps} > {args.static_kv_cache_len})"
        )
    log.log("loading runner and model")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        runner = call_with_filtered_stderr(
            lambda: LocalQwen30Runner(
                args.model_dir,
                device=device,
                dtype=dtype,
                compile_decode=args.compile_decode,
                compile_decode_dynamic=args.compile_decode_dynamic,
                decode_increfa_mode=args.decode_increfa_mode,
                static_kv_cache_len=args.static_kv_cache_len,
            ),
            suppressed_substrings=(
                "compiler_config.py:74: UserWarning: The following torchair config",
                'warnings.warn("The following torchair config',
            ),
        )
    memory_snapshots = {"after_load": npu_memory_snapshot("after_load")}
    print_memory_snapshot(memory_snapshots["after_load"])
    log.log("building benchmark input ids")
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    input_ids = build_prefill_input_ids(
        runner,
        args.prompt,
        args.prefill_tokens,
        args.batch_size,
    )
    attributor = KernelAttributor(runner.config)

    result = {
        "model_dir": str(args.model_dir),
        "device": args.device,
        "dtype": args.dtype,
        "compile_decode": bool(args.compile_decode),
        "compile_decode_dynamic": bool(args.compile_decode_dynamic),
        "npugraph_decode": bool(args.npugraph_decode),
        "decode_increfa_mode": args.decode_increfa_mode,
        "batch_size": int(args.batch_size),
        "prefill_tokens": int(args.prefill_tokens),
        "decode_steps": int(args.decode_steps),
        "static_kv_cache_len": int(args.static_kv_cache_len),
    }

    log.log("running prefill benchmark")
    result["prefill"] = benchmark_prefill(
        runner,
        input_ids,
        warmups=args.prefill_warmups,
        repeats=args.prefill_repeats,
        log=log,
    )
    memory_snapshots["after_prefill_benchmark"] = npu_memory_snapshot("after_prefill_benchmark")
    print_timing("prefill", result["prefill"])
    print_memory_snapshot(memory_snapshots["after_prefill_benchmark"])

    log.log("running decode benchmark")
    reset_dynamo_counters()
    if args.npugraph_decode:
        result["decode"] = benchmark_decode_npugraph(
            runner,
            input_ids,
            warmups=args.decode_warmups,
            repeats=args.decode_repeats,
            decode_steps=args.decode_steps,
            log=log,
        )
    else:
        result["decode"] = benchmark_decode(
            runner,
            input_ids,
            warmups=args.decode_warmups,
            repeats=args.decode_repeats,
            decode_steps=args.decode_steps,
            log=log,
        )
    result["dynamo_counters_after_decode"] = dynamo_counters_snapshot()
    memory_snapshots["after_decode_benchmark"] = npu_memory_snapshot("after_decode_benchmark")
    print_timing("decode", result["decode"])
    print_dynamo_counters(result["dynamo_counters_after_decode"])
    print_memory_snapshot(memory_snapshots["after_decode_benchmark"])

    profiles = {}
    profile_dir = Path(args.profile_dir)
    if args.profile in ("prefill", "both"):
        log.log("running prefill profiler")
        profiles["prefill"] = profile_once(
            "prefill",
            lambda: run_prefill_once(runner, input_ids),
            profile_dir=profile_dir,
            topn=args.topn,
            attributor=attributor,
        )
        print_profile("prefill", profiles["prefill"])
    if args.profile in ("decode", "both"):
        log.log("preparing decode profiler state")
        if args.npugraph_decode:
            graph_runner = prepare_npugraph_decode_runner(runner, input_ids, decode_steps=args.decode_steps)
            decode_profile_fn = graph_runner.run_loop
        else:
            next_id, key_caches, value_caches = prepare_decode_state(runner, input_ids)
            decode_profile_fn = lambda: run_decode_loop(
                runner,
                input_ids,
                next_id,
                key_caches,
                value_caches,
                decode_steps=args.decode_steps,
            )
        log.log("running decode profiler")
        profiles["decode"] = profile_once(
            "decode",
            decode_profile_fn,
            profile_dir=profile_dir,
            topn=args.topn,
            attributor=attributor,
        )
        print_profile("decode", profiles["decode"])
    if profiles:
        result["profiles"] = profiles
        memory_snapshots["after_profiles"] = npu_memory_snapshot("after_profiles")
        print_memory_snapshot(memory_snapshots["after_profiles"])

    result["npu_memory"] = memory_snapshots

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2) + "\n")
        log.log(f"wrote json output to {json_path}")

    log.log("benchmark complete")


if __name__ == "__main__":
    main()
