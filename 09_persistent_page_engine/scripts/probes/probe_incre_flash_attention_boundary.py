#!/usr/bin/env python3
"""Probe IncreFlashAttention alone at exact mixed cache positions.

The dense decode probe still runs the whole 18-layer decode step.  This one
runs a single ``npu_incre_flash_attention`` call with the production decode
shapes and nothing else: no weights, no projections, no MLP, no lm_head.  If a
device stalls here, the reproduction is small enough to hand to a vendor.

The production decode call attends over the whole physical KV cache with
``actual_seq_lengths=None`` and excludes the unwritten tail with a boolean
mask built from ``cache_position``.  Every step is therefore shape-identical;
only the mask contents and the cache contents change.  The lanes isolate that:

``--backend``   ``torchair`` compiles the step; ``eager`` does not.
``--mask``      ``position`` is production; ``none`` passes ``atten_mask=None``
                so the same step runs with no mask at all.
``--cache-init````zeros`` is a clean arena; ``random`` emulates a reused slot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "09_persistent_page_engine"

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.compile_utils import import_torchair  # noqa: E402
from paddleocr_vl.model.config import PaddleOCRVLConfig  # noqa: E402
from paddleocr_vl.model.text_decode import (  # noqa: E402
    build_static_decode_bool_mask,
)


DEFAULT_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_decode_lab"
    / "incre_flash_attention_boundary.json"
)


class IncreFlashAttentionStep(torch.nn.Module):
    """The production decode attention call, and nothing around it."""

    def __init__(
        self,
        *,
        num_heads: int,
        num_key_value_heads: int,
        cache_length: int,
        scale_value: float,
        masked: bool,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.cache_length = int(cache_length)
        self.scale_value = float(scale_value)
        self.masked = bool(masked)

    def forward(
        self,
        query: torch.Tensor,
        cache_position: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        atten_mask = None
        if self.masked:
            atten_mask = build_static_decode_bool_mask(
                cache_position, self.cache_length
            ).contiguous()
        return torch_npu.npu_incre_flash_attention(
            query.contiguous(),
            key_cache.contiguous(),
            value_cache.contiguous(),
            atten_mask=atten_mask,
            actual_seq_lengths=None,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            input_layout="BNSD",
            scale_value=self.scale_value,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_incre_fa_boundary",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--target-row", type=int, default=3)
    parser.add_argument("--inactive-position", type=int, default=0)
    parser.add_argument("--positions", default="1277,1278,1279,1280,1281")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--backend", default="torchair", choices=("torchair", "eager")
    )
    parser.add_argument("--mask", default="position", choices=("position", "none"))
    parser.add_argument(
        "--cache-init", default="zeros", choices=("zeros", "random")
    )
    parser.add_argument("--pipeline-depth", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        args.positions = tuple(
            int(value.strip())
            for value in args.positions.split(",")
            if value.strip()
        )
    except ValueError as exc:
        parser.error(f"invalid --positions: {exc}")
    if args.batch_size <= 0 or args.cache_length <= 0:
        parser.error("--batch-size and --cache-length must be positive")
    if not 0 <= args.target_row < args.batch_size:
        parser.error("--target-row must be inside the physical batch")
    if not args.positions:
        parser.error("--positions must not be empty")
    if args.pipeline_depth < 1:
        parser.error("--pipeline-depth must be at least 1")
    all_positions = (*args.positions, args.inactive_position)
    if min(all_positions) < 0 or max(all_positions) >= args.cache_length:
        parser.error("all positions must be inside the physical KV cache")
    return args


class ProgressRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        row = {"event": event, "monotonic_s": time.monotonic(), **fields}
        line = json.dumps(row, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        print("INCRE_FA_BOUNDARY " + line, flush=True)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("probe requires an available Ascend NPU")

    import torch_npu

    device = torch.device("npu:0")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(args.seed)

    output = args.output.expanduser().resolve()
    progress = ProgressRecorder(output.with_suffix(".progress.jsonl"))

    config = PaddleOCRVLConfig.from_model_dir(args.model).text_config
    num_heads = int(config.num_attention_heads)
    num_key_value_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    scale_value = 1.0 / math.sqrt(head_dim)

    progress.emit(
        "setup_begin",
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        scale_value=scale_value,
        backend=args.backend,
        mask=args.mask,
        cache_init=args.cache_init,
        pipeline_depth=args.pipeline_depth,
        positions=list(args.positions),
    )

    query = torch.randn(
        (args.batch_size, num_heads, 1, head_dim), device=device, dtype=dtype
    )
    cache_shape = (
        args.batch_size,
        num_key_value_heads,
        args.cache_length,
        head_dim,
    )
    key_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
    value_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
    if args.cache_init == "random":
        key_cache.normal_(mean=0.0, std=0.05)
        value_cache.normal_(mean=0.0, std=0.05)

    step_module = IncreFlashAttentionStep(
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        cache_length=args.cache_length,
        scale_value=scale_value,
        masked=args.mask == "position",
    )
    compile_metadata: dict[str, Any] = {"backend": args.backend}
    if args.backend == "torchair":
        torchair, CompilerConfig = import_torchair()
        shape_cache_dir = (
            args.cache_dir.expanduser().resolve()
            / f"b{args.batch_size}_cache{args.cache_length}_mask{args.mask}"
        )
        shape_cache_dir.mkdir(parents=True, exist_ok=True)
        step_fn = torchair.inference.cache_compile(
            step_module.forward,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(shape_cache_dir),
            ge_cache=True,
        )
        compile_metadata["torchair_cache_dir"] = str(shape_cache_dir)
    else:
        step_fn = step_module.forward

    torch.npu.synchronize()
    progress.emit("setup_end", **compile_metadata)

    steps: list[dict[str, Any]] = []
    pending: deque[dict[str, Any]] = deque()

    def drain_one() -> None:
        entry = pending.popleft()
        progress.emit(
            "step_sync_begin",
            step=entry["step"],
            target_position=entry["target_position"],
            effective_length=entry["target_position"] + 1,
            in_flight=len(pending) + 1,
        )
        entry["event"].synchronize()
        elapsed_s = time.perf_counter() - entry["started"]
        target = entry["attention_output"][args.target_row].float()
        step = {
            "step": entry["step"],
            "target_position": entry["target_position"],
            "effective_length": entry["target_position"] + 1,
            "elapsed_s": elapsed_s,
            "target_finite": bool(torch.isfinite(target).all().cpu()),
            "target_abs_mean": float(target.abs().mean().cpu()),
        }
        steps.append(step)
        progress.emit("step_sync_end", **step)

    for step_index, target_position in enumerate(args.positions):
        cache_position = torch.full(
            (args.batch_size,),
            args.inactive_position,
            device=device,
            dtype=torch.int64,
        )
        cache_position[args.target_row].fill_(target_position)
        progress.emit(
            "step_enqueue_begin",
            step=step_index,
            target_position=target_position,
            effective_length=target_position + 1,
        )
        started = time.perf_counter()
        attention_output = step_fn(query, cache_position, key_cache, value_cache)
        event = torch_npu.npu.current_stream().record_event()
        pending.append(
            {
                "step": step_index,
                "target_position": target_position,
                "started": started,
                "attention_output": attention_output,
                "event": event,
            }
        )
        progress.emit(
            "step_enqueue_end",
            step=step_index,
            target_position=target_position,
            in_flight=len(pending),
        )
        if len(pending) >= args.pipeline_depth:
            drain_one()
    while pending:
        drain_one()

    result = {
        "schema_version": 1,
        "kind": "incre_flash_attention_position_boundary",
        "passed": all(step["target_finite"] for step in steps),
        "configuration": {
            "model": str(args.model.expanduser().resolve()),
            "batch_size": args.batch_size,
            "cache_length": args.cache_length,
            "num_heads": num_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "scale_value": scale_value,
            "input_layout": "BNSD",
            "actual_seq_lengths": None,
            "target_row": args.target_row,
            "inactive_position": args.inactive_position,
            "positions": list(args.positions),
            "backend": args.backend,
            "mask": args.mask,
            "cache_initialization": args.cache_init,
            "pipeline_depth": args.pipeline_depth,
        },
        "compile": compile_metadata,
        "steps": steps,
        "progress_jsonl": str(progress.path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    progress.emit("probe_end", passed=result["passed"], output=str(output))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
