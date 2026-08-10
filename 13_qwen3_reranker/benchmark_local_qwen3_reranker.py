#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import sys
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

from transformers_rerank import DEFAULT_TASK


BYTES_PER_GIB = 1024**3
npu_prof = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local Qwen3 reranker forward.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--query", default="What is the capital of China?")
    parser.add_argument(
        "--documents",
        nargs="+",
        default=[
            "The capital of China is Beijing.",
            "Gravity attracts two bodies.",
        ],
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--compile-forward", action="store_true")
    parser.add_argument("--attention-impl", choices=("eager", "prompt_flash_attention"), default="eager")
    parser.add_argument("--ffn-weight-mode", choices=("dense", "w8a8", "all_w8a8"), default="dense")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", choices=("none", "forward"), default="none")
    parser.add_argument("--profile-dir", default="/tmp/qwen3_reranker_bench_profile")
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
    snapshot = {}
    for group, values in torch._dynamo.utils.counters.items():
        if values:
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


def static_document_batch(documents: list[str], batch_size: int) -> list[str]:
    if not documents:
        raise ValueError("--documents must contain at least one document")
    repeats = (batch_size + len(documents) - 1) // len(documents)
    return (documents * repeats)[:batch_size]


def summarize_seconds(values: list[float], *, batch_size: int, sequence_length: int) -> dict[str, float]:
    total = sum(values)
    runs = len(values)
    return {
        "runs": runs,
        "mean_sec": mean(values) if values else 0.0,
        "min_sec": min(values) if values else 0.0,
        "max_sec": max(values) if values else 0.0,
        "total_sec": total,
        "samples_s": (batch_size * runs / total) if total > 0.0 else 0.0,
        "tok_s": (batch_size * sequence_length * runs / total) if total > 0.0 else 0.0,
    }


def run_forward_once(runner, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return runner.score_ids(input_ids, attention_mask)


def benchmark_forward(
    runner,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    warmups: int,
    repeats: int,
    log: StageLogger,
) -> dict:
    log.log(f"warming up forward: runs={warmups}")
    warmup_timings = [
        time_call(lambda: run_forward_once(runner, input_ids, attention_mask))
        for _ in range(warmups)
    ]
    log.log(f"timing forward: repeats={repeats}")
    timings = [
        time_call(lambda: run_forward_once(runner, input_ids, attention_mask))
        for _ in range(repeats)
    ]
    summary = summarize_seconds(
        timings,
        batch_size=int(input_ids.shape[0]),
        sequence_length=int(input_ids.shape[1]),
    )
    summary["compile_forward_first_call_sec"] = warmup_timings[0] if runner.compile_forward and warmup_timings else None
    return summary


def profiler_experimental_config():
    require_npu_profiler()
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
        if output_last == self.vocab_size or weight == (self.vocab_size, self.hidden_size):
            return "lm_head"
        if weight == (self.intermediate_size, self.hidden_size):
            return "mlp.gate_or_up"
        if weight == (self.hidden_size, self.intermediate_size):
            return "mlp.down"
        if self.kv_dim != self.hidden_size and weight == (self.kv_dim, self.hidden_size):
            return "attn.kv_proj"
        if weight == (self.q_dim, self.hidden_size):
            return "attn.q_proj"
        if weight == (self.hidden_size, self.hidden_size):
            return "attn.q_or_o_or_square"
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
    return (
        top_kernels,
        sum_rows_by_key(core_rows, "core_type", "total_time_us"),
        sum_rows_by_key(module_rows, "module_guess", "total_time_us"),
    )


def load_top_operators(
    csv_path: Path,
    *,
    attributor: KernelAttributor,
    topn: int,
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str], dict[str, float | int | str]] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                total_time = float((row.get("Total Time(us)", "0") or "0").strip())
            except ValueError:
                continue
            name = row.get("Name", "").strip() or "<empty>"
            input_shapes = row.get("Input Shapes", "").strip()
            output_shapes = row.get("Output Shapes", "").strip()
            module_guess = attributor.guess(
                name=name,
                op_type=name,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
            )
            key = (name, module_guess)
            item = grouped.setdefault(
                key,
                {"name": name, "module_guess": module_guess, "total_time_us": 0.0, "count": 0},
            )
            item["total_time_us"] = float(item["total_time_us"]) + total_time
            item["count"] = int(item["count"]) + int(float((row.get("Calls", "1") or "1").strip()))
    return sorted(grouped.values(), key=lambda item: float(item["total_time_us"]), reverse=True)[:topn]


def summarize_profile(profile_dir: Path, *, attributor: KernelAttributor, topn: int) -> dict:
    root = find_profile_root(profile_dir)
    ascend_pt = root / "ASCEND_PROFILER_OUTPUT"
    op_stat = ascend_pt / "op_statistic.csv"
    kernel_details = ascend_pt / "kernel_details.csv"
    operator_details = ascend_pt / "operator_details.csv"
    summary = {"profile_root": str(root)}
    if op_stat.exists():
        summary["top_op_types"] = load_top_op_types(op_stat, topn)
    if kernel_details.exists():
        top_kernels, core_summary, module_summary = load_kernel_details(
            kernel_details,
            attributor=attributor,
            topn=topn,
        )
        summary["top_kernels"] = top_kernels
        summary["core_type_summary"] = core_summary
        summary["module_summary"] = module_summary
    if operator_details.exists():
        summary["top_operators"] = load_top_operators(operator_details, attributor=attributor, topn=topn)
    return summary


def run_profile_once(
    runner,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    profile_dir: Path,
    topn: int,
) -> dict:
    require_npu_profiler()
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True)
    with suppress_stdout():
        with npu_prof.profile(
            activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
            on_trace_ready=npu_prof.tensorboard_trace_handler(str(profile_dir)),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            experimental_config=profiler_experimental_config(),
        ):
            run_forward_once(runner, input_ids, attention_mask)
            sync()
    return summarize_profile(profile_dir, attributor=KernelAttributor(runner.config), topn=topn)


def require_npu_profiler() -> None:
    if npu_prof is None:
        raise RuntimeError("torch_npu.profiler is required for --profile forward")


def print_forward_summary(summary: dict) -> None:
    print(
        "forward: "
        f"mean={summary['mean_sec']:.6f}s "
        f"min={summary['min_sec']:.6f}s "
        f"max={summary['max_sec']:.6f}s "
        f"samples/s={summary['samples_s']:.2f} "
        f"tok/s={summary['tok_s']:.2f}"
    )
    first_call = summary.get("compile_forward_first_call_sec")
    if first_call is not None:
        print(f"forward: compile_first_call={first_call:.6f}s")


def print_profile(summary: dict) -> None:
    print(f"\nforward profile: {summary['profile_root']}")
    if summary.get("core_type_summary"):
        print("core type summary:")
        for row in summary["core_type_summary"]:
            print(f"  {row['core_type']} total_us={row['total_time_us']:.1f} count={row['count']}")
    if summary.get("module_summary"):
        print("top module guesses:")
        for row in summary["module_summary"][:20]:
            print(f"  {row['module_guess']} total_us={row['total_time_us']:.1f} count={row['count']}")
    if summary.get("top_op_types"):
        print("top op types:")
        for row in summary["top_op_types"]:
            print(
                f"  {row['op_type']} {row['core_type']} "
                f"total_us={row['total_time_us']:.1f} "
                f"ratio={row['ratio_percent']:.2f}% count={row['count']}"
            )
    if summary.get("top_operators"):
        print("top operators:")
        for row in summary["top_operators"]:
            print(f"  {row['name']} guess={row['module_guess']} total_us={row['total_time_us']:.1f} count={row['count']}")
    if summary.get("top_kernels"):
        print("top kernels:")
        for row in summary["top_kernels"]:
            print(
                f"  {row['name']} {row['op_type']} {row['core_type']} "
                f"guess={row['module_guess']} total_us={row['total_time_us']:.1f} count={row['count']}"
            )


def main() -> None:
    global npu_prof
    args = parse_args()
    import torch_npu  # noqa: F401
    import torch_npu.profiler as imported_npu_prof
    from run_local_qwen3_reranker import LocalQwen3RerankerRunner

    npu_prof = imported_npu_prof
    log = StageLogger(enabled=args.verbose)
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]

    log.log("loading runner and model")
    runner = LocalQwen3RerankerRunner(
        args.model_dir,
        device=device,
        dtype=dtype,
        max_length=args.max_length,
        batch_size=args.batch_size,
        compile_forward=args.compile_forward,
        attention_impl=args.attention_impl,
        ffn_weight_mode=args.ffn_weight_mode,
    )
    memory_snapshots = [npu_memory_snapshot("after_load")]
    print_memory_snapshot(memory_snapshots[-1])

    log.log("building benchmark input tensors")
    benchmark_documents = static_document_batch(args.documents, args.batch_size)
    encoded = runner.encode_pairs(args.query, benchmark_documents, args.task)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if args.ffn_weight_mode != "dense":
        log.log(f"calibrating {args.ffn_weight_mode} input scales")
        runner.calibrate_ffn_input_scales(input_ids, attention_mask)

    log.log("running forward benchmark")
    reset_dynamo_counters()
    forward_summary = benchmark_forward(
        runner,
        input_ids,
        attention_mask,
        warmups=args.warmups,
        repeats=args.repeats,
        log=log,
    )
    print_forward_summary(forward_summary)
    dynamo_counters = dynamo_counters_snapshot()
    print_dynamo_counters(dynamo_counters)
    memory_snapshots.append(npu_memory_snapshot("after_forward_benchmark"))
    print_memory_snapshot(memory_snapshots[-1])

    scores = run_forward_once(runner, input_ids, attention_mask)
    ranked = sorted(enumerate(scores.detach().float().cpu().tolist()), key=lambda item: item[1], reverse=True)
    print(f"scores={scores.detach().float().cpu().tolist()}")
    print(f"ranked={ranked}")

    profile_summary = None
    if args.profile == "forward":
        log.log("running forward profiler")
        profile_summary = run_profile_once(
            runner,
            input_ids,
            attention_mask,
            profile_dir=Path(args.profile_dir) / "forward",
            topn=args.topn,
        )
        print_profile(profile_summary)
        memory_snapshots.append(npu_memory_snapshot("after_profile"))
        print_memory_snapshot(memory_snapshots[-1])

    result = {
        "model_dir": args.model_dir,
        "query": args.query,
        "documents": args.documents,
        "benchmark_documents": benchmark_documents,
        "task": args.task,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": args.device,
        "compile_forward": args.compile_forward,
        "attention_impl": args.attention_impl,
        "ffn_weight_mode": args.ffn_weight_mode,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "forward": forward_summary,
        "dynamo_counters": dynamo_counters,
        "memory": memory_snapshots,
        "scores": scores.detach().float().cpu().tolist(),
        "ranked": ranked,
        "profile": profile_summary,
    }
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(result, handle, indent=2)
    log.log("benchmark complete")


if __name__ == "__main__":
    main()
