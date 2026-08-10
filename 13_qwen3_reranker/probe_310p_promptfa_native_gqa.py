#!/usr/bin/env python3
"""Compare expanded and native-GQA PromptFA with real Qwen3 reranker weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path

import torch
from torch import nn

from local_modeling_qwen3_reranker import (
    PROMPT_FA_FULL_ATTENTION_TOKENS,
    build_left_padded_causal_bool_mask,
    repeat_kv_bsnd,
)
from run_local_qwen3_reranker import LocalQwen3RerankerRunner, _import_cache_compile
from transformers_rerank import DEFAULT_TASK


class ExpandedGQAPromptFA(nn.Module):
    """Current 310P-safe route: expand compact GQA K/V before PromptFA."""

    def __init__(self, *, num_heads: int, num_kv_heads: int, scale: float):
        super().__init__()
        self.num_heads = int(num_heads)
        self.groups = int(num_heads) // int(num_kv_heads)
        self.scale = float(scale)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        expanded_key = repeat_kv_bsnd(key, self.groups).contiguous()
        expanded_value = repeat_kv_bsnd(value, self.groups).contiguous()
        return torch_npu.npu_prompt_flash_attention(
            query.contiguous(),
            expanded_key,
            expanded_value,
            atten_mask=attention_mask.contiguous(),
            num_heads=self.num_heads,
            input_layout="BSND",
            scale_value=self.scale,
            pre_tokens=PROMPT_FA_FULL_ATTENTION_TOKENS,
            next_tokens=PROMPT_FA_FULL_ATTENTION_TOKENS,
            sparse_mode=0,
        )


class NativeGQAPromptFA(nn.Module):
    """Experimental route: keep compact K/V and declare their head count."""

    def __init__(self, *, num_heads: int, num_kv_heads: int, scale: float):
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.scale = float(scale)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        return torch_npu.npu_prompt_flash_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            atten_mask=attention_mask.contiguous(),
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSND",
            scale_value=self.scale,
            pre_tokens=PROMPT_FA_FULL_ATTENTION_TOKENS,
            next_tokens=PROMPT_FA_FULL_ATTENTION_TOKENS,
            sparse_mode=0,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(".runtime_cache/13_qwen3_reranker/native_gqa_promptfa"),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def timed(device: torch.device, fn) -> tuple[float, torch.Tensor]:
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = fn()
    synchronize(device)
    return time.perf_counter() - started, output


def output_diff(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    difference = (reference.float() - candidate.float()).abs()
    reference_abs = reference.float().abs()
    return {
        "max_abs": float(difference.max().cpu()),
        "mean_abs": float(difference.mean().cpu()),
        "reference_max_abs": float(reference_abs.max().cpu()),
        "allclose_atol_5e_2_rtol_5e_2": bool(
            torch.allclose(reference.float(), candidate.float(), atol=5e-2, rtol=5e-2)
        ),
        "allclose_atol_1e_1_rtol_1e_1": bool(
            torch.allclose(reference.float(), candidate.float(), atol=1e-1, rtol=1e-1)
        ),
    }


def manual_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    groups: int,
    scale: float,
) -> torch.Tensor:
    query_bnsd = query.transpose(1, 2)
    key_bnsd = repeat_kv_bsnd(key, groups).transpose(1, 2)
    value_bnsd = repeat_kv_bsnd(value, groups).transpose(1, 2)
    scores = torch.matmul(query_bnsd, key_bnsd.transpose(-2, -1)) * float(scale)
    scores = scores.masked_fill(attention_mask, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(probabilities, value_bnsd).transpose(1, 2).contiguous()


def lane_result(
    *,
    name: str,
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    manual_output: torch.Tensor,
    expanded_eager_output: torch.Tensor | None,
    device: torch.device,
    cache_dir: Path,
    warmups: int,
    repeats: int,
    token_count: int,
) -> tuple[dict[str, object], torch.Tensor | None]:
    result: dict[str, object] = {"name": name}
    eager_output: torch.Tensor | None = None
    print(f"LANE_START {name}", flush=True)
    try:
        eager_s, eager_output = timed(device, lambda: module(*inputs))
        result["eager_first_call_s"] = eager_s
        result["eager_vs_manual"] = output_diff(manual_output, eager_output)
        if expanded_eager_output is not None:
            result["eager_vs_expanded"] = output_diff(expanded_eager_output, eager_output)

        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        cache_dir.mkdir(parents=True, exist_ok=True)
        compiled = _import_cache_compile()(
            module.forward,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        first_call_s, compiled_output = timed(device, lambda: compiled(*inputs))
        for _ in range(warmups):
            timed(device, lambda: compiled(*inputs))
        timings = [timed(device, lambda: compiled(*inputs))[0] for _ in range(repeats)]
        median_s = statistics.median(timings)
        result.update(
            {
                "status": "passed",
                "compile_first_call_s": first_call_s,
                "post_warmup_s": {
                    "runs": repeats,
                    "median": median_s,
                    "mean": statistics.mean(timings),
                    "min": min(timings),
                    "max": max(timings),
                },
                "attention_query_tok_s": token_count / median_s,
                "compiled_vs_eager": output_diff(eager_output, compiled_output),
                "compiled_vs_manual": output_diff(manual_output, compiled_output),
            }
        )
        if expanded_eager_output is not None:
            result["compiled_vs_expanded"] = output_diff(expanded_eager_output, compiled_output)
        print("LANE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return result, eager_output
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print("LANE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return result, eager_output


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.sequence_length <= 0:
        raise ValueError("batch size and sequence length must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")

    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    torch.npu.set_device(device)

    load_started = time.perf_counter()
    runner = LocalQwen3RerankerRunner(
        args.model_dir,
        device=device,
        dtype=torch.float16,
        max_length=args.sequence_length,
        batch_size=args.batch_size,
        compile_forward=False,
        attention_impl="prompt_flash_attention",
        prefill_optimization="combined_bsnd",
    )
    model_load_s = time.perf_counter() - load_started
    documents = [
        f"Document {index}: native grouped-query attention contract test."
        for index in range(args.batch_size)
    ]
    encoded = runner.encode_pairs(
        "Does this document describe an attention implementation?",
        documents,
        DEFAULT_TASK,
    )
    input_ids = encoded["input_ids"]
    token_mask = encoded["attention_mask"]
    position_ids = (token_mask.to(dtype=torch.long).cumsum(dim=-1) - 1).clamp(min=0)

    with torch.inference_mode():
        hidden_states = model_hidden = runner.model.embed_tokens(input_ids)
        hidden_states = runner.model.layers[0].input_layernorm(hidden_states)
        cos, sin = runner.model.rotary_emb(
            position_ids,
            dtype=model_hidden.dtype,
            device=model_hidden.device,
        )
        attention = runner.model.layers[0].self_attn
        query, key, value = attention.project_qkv(
            hidden_states,
            cos,
            sin,
            output_layout="BSND",
        )
        attention_mask = build_left_padded_causal_bool_mask(token_mask).contiguous()
        manual_output = manual_attention(
            query,
            key,
            value,
            attention_mask,
            groups=attention.num_key_value_groups,
            scale=attention.scaling,
        )

    inputs = (query, key, value, attention_mask)
    common = {
        "num_heads": attention.num_heads,
        "num_kv_heads": attention.num_key_value_heads,
        "scale": attention.scaling,
    }
    expanded_result, expanded_eager_output = lane_result(
        name="expanded_gqa",
        module=ExpandedGQAPromptFA(**common).to(device),
        inputs=inputs,
        manual_output=manual_output,
        expanded_eager_output=None,
        device=device,
        cache_dir=args.compile_cache_dir / "expanded",
        warmups=args.warmups,
        repeats=args.repeats,
        token_count=args.batch_size * args.sequence_length,
    )
    native_result, _native_eager_output = lane_result(
        name="native_gqa",
        module=NativeGQAPromptFA(**common).to(device),
        inputs=inputs,
        manual_output=manual_output,
        expanded_eager_output=expanded_eager_output,
        device=device,
        cache_dir=args.compile_cache_dir / "native",
        warmups=args.warmups,
        repeats=args.repeats,
        token_count=args.batch_size * args.sequence_length,
    )

    speedup = None
    if expanded_result.get("status") == native_result.get("status") == "passed":
        expanded_median = float(expanded_result["post_warmup_s"]["median"])
        native_median = float(native_result["post_warmup_s"]["median"])
        speedup = expanded_median / native_median

    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    result = {
        "device": {
            "requested": str(device),
            "name": torch.npu.get_device_name(device),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "model": {
            "directory": str(args.model_dir.resolve()),
            "load_s": model_load_s,
            "weights": "full real checkpoint loaded; layer-0 embedding, norm, Q/K/V, and RoPE used",
        },
        "shape": {
            "batch": args.batch_size,
            "sequence": args.sequence_length,
            "query": list(query.shape),
            "compact_key": list(key.shape),
            "compact_value": list(value.shape),
            "mask": list(attention_mask.shape),
            "num_heads": attention.num_heads,
            "num_key_value_heads": attention.num_key_value_heads,
        },
        "source_hash": source_digest,
        "expanded_gqa": expanded_result,
        "native_gqa": native_result,
        "native_vs_expanded_speedup": speedup,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print("NATIVE_GQA_PROMPTFA_PROBE " + json.dumps(result, sort_keys=True), flush=True)
    print(f"OUTPUT_JSON {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
