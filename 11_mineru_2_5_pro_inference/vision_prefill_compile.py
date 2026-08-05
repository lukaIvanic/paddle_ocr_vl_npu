#!/usr/bin/env python3
"""Static B=1 bucketed MinerU vision-transformer execution.

Patch embedding and position construction stay eager at the real image shape.
Only the 32 transformer blocks are padded and compiled.  Padded rows are
attention-isolated from real rows, and are removed before the stock patch
merger consumes the result.
"""

from __future__ import annotations

import hashlib
import importlib
import time
import types
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from local_modeling_mineru import (
    _activation,
    apply_rotary_pos_emb_vision,
    vision_prompt_flash_attention_bnsd,
)


DEFAULT_VISION_BUCKETS = (384, 512, 768, 1024, 1536, 2048, 3072, 4224, 5632)
VISION_MASK_SPARSE_MODE = 1


def parse_vision_buckets(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        values = tuple(int(item) for item in value)
    if not values or any(item <= 0 for item in values):
        raise ValueError("vision buckets must contain positive integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("vision buckets must be unique and strictly increasing")
    return values


def select_vision_bucket(real_seq_len: int, buckets: Iterable[int]) -> int | None:
    for bucket in buckets:
        if int(real_seq_len) <= int(bucket):
            return int(bucket)
    return None


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _import_torchair():
    try:
        import torchair

        CompilerConfig = torchair.CompilerConfig
    except Exception as direct_error:
        try:
            from torch_npu.dynamo import torchair
            from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig
        except Exception as fallback_error:
            raise RuntimeError(
                "TorchAir is unavailable: direct import failed with "
                f"{direct_error!r}; torch_npu fallback failed with {fallback_error!r}"
            ) from fallback_error
    if not hasattr(torchair, "inference"):
        torchair.inference = importlib.import_module(f"{torchair.__name__}.inference")
    return torchair, CompilerConfig


def _short_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


class StaticMinerUVisionBlocks(nn.Module):
    """Compiler-safe equivalent of the stock MinerU vision block stack."""

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.blocks = visual.blocks
        self.num_heads = int(visual.config.num_heads)
        self.head_dim = int(visual.config.embed_dim) // self.num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        for block in self.blocks:
            residual = hidden_states
            normed = block.norm1(hidden_states)
            qkv = block.attn.qkv(normed).view(
                batch_size,
                seq_len,
                3,
                self.num_heads,
                self.head_dim,
            )
            query_states, key_states, value_states = qkv.unbind(dim=2)
            query_states, key_states = apply_rotary_pos_emb_vision(
                query_states,
                key_states,
                rope_cos,
                rope_sin,
            )
            query_states = query_states.transpose(1, 2).contiguous()
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
            attention_output = vision_prompt_flash_attention_bnsd(
                query_states,
                key_states,
                value_states,
                num_heads=self.num_heads,
                scale=float(block.attn.scaling),
                atten_mask=attention_mask,
                sparse_mode=VISION_MASK_SPARSE_MODE,
            )
            attention_output = (
                attention_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, -1)
            )
            hidden_states = residual + block.attn.proj(attention_output)
            mlp_input = block.norm2(hidden_states)
            hidden_states = hidden_states + block.mlp.fc2(
                _activation(block.mlp.hidden_act, block.mlp.fc1(mlp_input))
            )
        return hidden_states


def _unique_bucket_forward(
    module: StaticMinerUVisionBlocks,
    bucket: int,
) -> Callable[..., torch.Tensor]:
    """Give every static shape a distinct Dynamo code object."""

    original = module.forward.__func__
    name = f"mineru_vision_blocks_bucket_{int(bucket)}"
    function = types.FunctionType(
        original.__code__.replace(co_name=name),
        original.__globals__,
        name,
        original.__defaults__,
        original.__closure__,
    )
    function.__annotations__ = dict(original.__annotations__)
    function.__kwdefaults__ = original.__kwdefaults__
    return types.MethodType(function, module)


class MinerUVisionPrefillRuntime:
    """Route one image at a time through padded static TorchAir graphs."""

    def __init__(
        self,
        visual: nn.Module,
        *,
        buckets: str | Iterable[int] = DEFAULT_VISION_BUCKETS,
        cache_root: Path,
        model_dir: Path,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if device.type != "npu":
            raise ValueError("MinerU compiled vision prefill requires an NPU device")
        self.visual = visual
        self.buckets = parse_vision_buckets(buckets)
        self.cache_root = cache_root.expanduser().resolve()
        self.model_dir = model_dir.expanduser().resolve()
        self.device = device
        self.dtype = dtype
        self.compiled: dict[int, Callable[..., torch.Tensor]] = {}
        self.modules: dict[int, StaticMinerUVisionBlocks] = {}
        self.compile_records: dict[str, dict[str, Any]] = {}
        self.route_counts: dict[str, int] = {}
        self.real_tokens = 0
        self.physical_tokens = 0

    def _cache_dir(self, bucket: int) -> Path:
        source_path = Path(__file__).resolve()
        config_path = self.model_dir / "config.json"
        try:
            import torch_npu

            torch_npu_version = str(torch_npu.__version__)
        except Exception:
            torch_npu_version = "unknown"
        key = "_".join(
            (
                "mineru_vision_blocks_promptfa_mask_sparse1",
                "bs1",
                f"seq{int(bucket)}",
                f"dtype{str(self.dtype).replace('torch.', '')}",
                f"model{_short_hash(config_path)}",
                f"torch{torch.__version__}",
                f"torchnpu{torch_npu_version}",
                f"src{_short_hash(source_path)}",
            )
        )
        return self.cache_root / key.replace("/", "_")

    def _compiled_for_bucket(self, bucket: int) -> Callable[..., torch.Tensor]:
        if bucket in self.compiled:
            return self.compiled[bucket]
        torchair, CompilerConfig = _import_torchair()
        module = StaticMinerUVisionBlocks(self.visual).eval()
        entrypoint = _unique_bucket_forward(module, bucket)
        cache_dir = self._cache_dir(bucket)
        cache_dir.mkdir(parents=True, exist_ok=True)
        config = CompilerConfig()
        _synchronize(self.device)
        started = time.perf_counter()
        compiled = torchair.inference.cache_compile(
            entrypoint,
            config=config,
            dynamic=False,
            cache_dir=str(cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        _synchronize(self.device)
        self.modules[bucket] = module
        self.compiled[bucket] = compiled
        self.compile_records[str(bucket)] = {
            "compile_wrapper_s": float(time.perf_counter() - started),
            "first_call_s": None,
            "cache_dir": str(cache_dir),
        }
        return compiled

    def _prepare(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        bucket: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        real_seq_len = int(hidden_states.shape[0])
        pad_tokens = int(bucket) - real_seq_len
        prefix = F.pad(hidden_states, (0, 0, 0, pad_tokens)).unsqueeze(0).contiguous()
        rope_cos, rope_sin = position_embeddings
        if pad_tokens:
            rope_cos = F.pad(rope_cos, (0, 0, 0, pad_tokens), value=1.0)
            rope_sin = F.pad(rope_sin, (0, 0, 0, pad_tokens), value=0.0)
        indices = torch.arange(bucket, device=hidden_states.device)
        is_real = indices < real_seq_len
        mask = (is_real[:, None] != is_real[None, :]).view(1, 1, bucket, bucket)
        return (
            prefix,
            rope_cos.unsqueeze(0).contiguous(),
            rope_sin.unsqueeze(0).contiguous(),
            mask.contiguous(),
        )

    def run(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        real_seq_len = int(hidden_states.shape[0])
        if tuple(cu_seqlens.shape) != (2,):
            raise ValueError(
                "compiled MinerU vision prefill supports one image per call; "
                f"got cu_seqlens shape {tuple(cu_seqlens.shape)}"
            )
        bucket = select_vision_bucket(real_seq_len, self.buckets)
        if bucket is None:
            eager_hidden = hidden_states
            for block in self.visual.blocks:
                eager_hidden = block(
                    eager_hidden,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )
            self.route_counts["eager_overflow"] = (
                self.route_counts.get("eager_overflow", 0) + 1
            )
            self.real_tokens += real_seq_len
            self.physical_tokens += real_seq_len
            return eager_hidden
        prepared = self._prepare(hidden_states, position_embeddings, bucket)
        compiled = self._compiled_for_bucket(bucket)
        record = self.compile_records[str(bucket)]
        first_call = record["first_call_s"] is None
        if first_call:
            _synchronize(self.device)
            started = time.perf_counter()
        output = compiled(*prepared)
        if first_call:
            _synchronize(self.device)
            record["first_call_s"] = float(time.perf_counter() - started)
        key = str(bucket)
        self.route_counts[key] = self.route_counts.get(key, 0) + 1
        self.real_tokens += real_seq_len
        self.physical_tokens += bucket
        return output[0, :real_seq_len].contiguous()

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "torchair",
            "boundary": "32_vision_transformer_blocks",
            "batch_size": 1,
            "attention": "prompt_flash_attention",
            "mask_sparse_mode": VISION_MASK_SPARSE_MODE,
            "padding": "bucket",
            "buckets": list(self.buckets),
            "overflow": "eager_unpadded",
            "route_counts": dict(self.route_counts),
            "real_tokens": int(self.real_tokens),
            "physical_tokens": int(self.physical_tokens),
            "useful_token_fraction": (
                self.real_tokens / self.physical_tokens if self.physical_tokens else None
            ),
            "compile_records": dict(self.compile_records),
        }
