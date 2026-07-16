"""Bucketed static TorchAir execution for PaddleOCR-VL's vision encoder.

Patch embedding and absolute-position interpolation remain eager and operate at
the crop's real shape.  This module pads that prefix to a configured static
bucket, runs the complete vision encoder plus final LayerNorm, and slices the
real rows before the existing projector consumes them.
"""

from __future__ import annotations

import hashlib
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from local_modeling_paddleocr_vl import (
    LocalPaddleOCRVLForConditionalGeneration,
    _activation,
    apply_rotary_pos_emb_vision,
    attention_softmax,
    get_vision_attention_impl,
    get_vision_softmax_dtype_mode,
)
from probe_static_compile import (
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from timing import synchronize


DEFAULT_VISION_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048)
VISION_BACKEND_CHOICES = ("raw_eager", "torchair")


def parse_vision_buckets(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if not pieces:
            raise ValueError("vision compile buckets cannot be empty")
        try:
            buckets = tuple(int(piece) for piece in pieces)
        except ValueError as exc:
            raise ValueError(f"invalid vision compile buckets: {value!r}") from exc
    else:
        buckets = tuple(int(item) for item in value)
    if not buckets:
        raise ValueError("vision compile buckets cannot be empty")
    if any(bucket <= 0 or bucket & (bucket - 1) for bucket in buckets):
        raise ValueError("every vision compile bucket must be a positive power of two")
    if tuple(sorted(set(buckets))) != buckets:
        raise ValueError("vision compile buckets must be unique and strictly increasing")
    return buckets


def select_vision_bucket(real_seq_len: int, buckets: Iterable[int]) -> int | None:
    real_seq_len = int(real_seq_len)
    if real_seq_len <= 0:
        raise ValueError("real vision sequence length must be positive")
    for bucket in buckets:
        if real_seq_len <= int(bucket):
            return int(bucket)
    return None


class StaticManualVisionEncoder(torch.nn.Module):
    """Compiler-safe manual-attention vision encoder plus post LayerNorm."""

    def __init__(self, model: LocalPaddleOCRVLForConditionalGeneration):
        super().__init__()
        self.transformer = model.visual.vision_model
        self.softmax_dtype_mode = get_vision_softmax_dtype_mode()

    def _attention(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_length, _hidden = hidden_states.shape
        qkv = torch.cat(
            [
                attention.q_proj(hidden_states),
                attention.k_proj(hidden_states),
                attention.v_proj(hidden_states),
            ],
            dim=-1,
        )
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        num_heads = int(attention.num_heads)
        head_dim = int(attention.head_dim)
        query_states = query_states.view(batch_size, seq_length, num_heads, head_dim)
        key_states = key_states.view(batch_size, seq_length, num_heads, head_dim)
        value_states = value_states.view(batch_size, seq_length, num_heads, head_dim)
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            rope_cos,
            rope_sin,
        )
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        # Explicit [B*H, S, D] bmm is equivalent to the stock 4-D matmul but
        # prevents GE from treating the attention head as a broadcast axis.
        query_bh = query_states.reshape(batch_size * num_heads, seq_length, head_dim)
        key_bh = key_states.reshape(batch_size * num_heads, seq_length, head_dim)
        value_bh = value_states.reshape(batch_size * num_heads, seq_length, head_dim)
        scores = torch.bmm(query_bh, key_bh.transpose(1, 2)).view(
            batch_size,
            num_heads,
            seq_length,
            seq_length,
        ) * attention.scaling
        scores = scores.masked_fill(attention_mask, torch.finfo(scores.dtype).min)
        probs = attention_softmax(
            scores,
            dim=-1,
            output_dtype=query_states.dtype,
            mode=self.softmax_dtype_mode,
        )
        attention_output = torch.bmm(
            probs.reshape(batch_size * num_heads, seq_length, seq_length),
            value_bh,
        ).view(batch_size, num_heads, seq_length, head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_length,
            -1,
        )
        return attention.out_proj(attention_output)

    def forward(
        self,
        prefix_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = prefix_hidden_states
        for encoder_layer in self.transformer.encoder.layers:
            attention_input = encoder_layer.layer_norm1(hidden_states)
            hidden_states = hidden_states + self._attention(
                encoder_layer.self_attn,
                attention_input,
                rope_cos,
                rope_sin,
                attention_mask,
            )
            mlp_input = encoder_layer.layer_norm2(hidden_states)
            hidden_states = hidden_states + encoder_layer.mlp.fc2(
                _activation(
                    encoder_layer.mlp.hidden_act,
                    encoder_layer.mlp.fc1(mlp_input),
                )
            )
        return self.transformer.post_layernorm(hidden_states)


def unique_bucket_forward(
    module: StaticManualVisionEncoder,
    bucket: int,
) -> Callable[..., torch.Tensor]:
    """Clone ``forward``'s code object so Dynamo caches shapes independently.

    TorchDynamo keys recompilation state by Python code object. Passing the same
    class method to eight ``cache_compile`` wrappers therefore makes later
    static shapes look like recompilations and TorchAir skips their persistent
    caches. Each bucket needs a semantically identical but distinct entry code
    object.
    """

    original = module.forward.__func__
    name = f"vision_encoder_bucket_{int(bucket)}"
    code = original.__code__.replace(co_name=name)
    function = types.FunctionType(
        code,
        original.__globals__,
        name,
        original.__defaults__,
        original.__closure__,
    )
    function.__annotations__ = dict(original.__annotations__)
    function.__kwdefaults__ = original.__kwdefaults__
    return types.MethodType(function, module)


@dataclass(frozen=True)
class PreparedVisionBucket:
    prefix_hidden_states: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    attention_mask: torch.Tensor
    real_seq_len: int
    physical_seq_len: int


def build_vision_rope(
    model: LocalPaddleOCRVLForConditionalGeneration,
    image_grid_thw: torch.Tensor,
    *,
    real_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = image_grid_thw.detach().cpu().reshape(-1, 3)
    if int(grid.shape[0]) != 1:
        raise ValueError(f"compiled B=1 vision expects one grid row, got {tuple(grid.shape)}")
    t, h, w = (int(value) for value in grid[0].tolist())
    if t * h * w != int(real_seq_len):
        raise ValueError(
            f"image grid has {t * h * w} tokens but embeddings have {int(real_seq_len)} rows"
        )
    encoder = model.visual.vision_model.encoder
    image_pids = torch.arange(int(real_seq_len), device=device, dtype=torch.int64) % int(h * w)
    pids = torch.stack((image_pids // int(w), image_pids % int(w)), dim=-1)
    rotary_max = encoder.rotary_pos_emb(max(h, w))
    rotary_embeddings = rotary_max[pids].flatten(1).repeat(1, 2)
    return rotary_embeddings.cos().contiguous(), rotary_embeddings.sin().contiguous()


def prepare_vision_bucket(
    model: LocalPaddleOCRVLForConditionalGeneration,
    prefix_hidden_states: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    physical_seq_len: int,
) -> PreparedVisionBucket:
    if prefix_hidden_states.ndim != 2:
        raise ValueError(
            f"vision prefix must have shape [S, H], got {tuple(prefix_hidden_states.shape)}"
        )
    real_seq_len = int(prefix_hidden_states.shape[0])
    physical_seq_len = int(physical_seq_len)
    if real_seq_len > physical_seq_len:
        raise ValueError(
            f"real vision sequence {real_seq_len} exceeds bucket {physical_seq_len}"
        )
    rope_cos, rope_sin = build_vision_rope(
        model,
        image_grid_thw,
        real_seq_len=real_seq_len,
        device=prefix_hidden_states.device,
    )
    pad_tokens = physical_seq_len - real_seq_len
    prefix = F.pad(prefix_hidden_states, (0, 0, 0, pad_tokens)).unsqueeze(0).contiguous()
    if pad_tokens:
        rope_cos = torch.cat(
            [
                rope_cos,
                torch.ones(
                    (pad_tokens, rope_cos.shape[-1]),
                    device=rope_cos.device,
                    dtype=rope_cos.dtype,
                ),
            ],
            dim=0,
        )
        rope_sin = torch.cat(
            [
                rope_sin,
                torch.zeros(
                    (pad_tokens, rope_sin.shape[-1]),
                    device=rope_sin.device,
                    dtype=rope_sin.dtype,
                ),
            ],
            dim=0,
        )
    indices = torch.arange(physical_seq_len, device=prefix_hidden_states.device)
    is_real = indices < real_seq_len
    attention_mask = (is_real[:, None] != is_real[None, :]).view(
        1,
        1,
        physical_seq_len,
        physical_seq_len,
    )
    return PreparedVisionBucket(
        prefix_hidden_states=prefix,
        rope_cos=rope_cos.unsqueeze(0).contiguous(),
        rope_sin=rope_sin.unsqueeze(0).contiguous(),
        attention_mask=attention_mask.contiguous(),
        real_seq_len=real_seq_len,
        physical_seq_len=physical_seq_len,
    )


def vision_source_hash() -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    for name in ("local_modeling_paddleocr_vl.py", "vision_compile.py"):
        digest.update(name.encode("utf-8"))
        digest.update(short_file_hash(here / name).encode("utf-8"))
    return digest.hexdigest()[:12]


def vision_cache_dir_for_bucket(
    cache_root: Path,
    *,
    bucket: int,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
) -> Path:
    key = "_".join(
        [
            "encoder_postln_manual_bmm",
            f"softmax{cache_key_part(get_vision_softmax_dtype_mode())}",
            "bs1",
            f"seq{int(bucket)}",
            f"dtype{cache_key_part(dtype)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{vision_source_hash()}",
        ]
    )
    return cache_root.expanduser().resolve() / key


class BucketedVisionEncoderRuntime:
    """Own one static compiled graph per configured sequence bucket."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        backend: str,
        buckets: Iterable[int],
        cache_root: Path,
        device: torch.device,
        dtype: torch.dtype,
        model_dir: Path,
    ):
        self.model = model
        self.backend = str(backend)
        self.buckets = parse_vision_buckets(buckets)
        self.device = device
        self.dtype = dtype
        self.cache_root = cache_root.expanduser().resolve()
        self.compiled: dict[int, Callable[..., torch.Tensor]] = {}
        self.entrypoints: dict[int, Callable[..., torch.Tensor]] = {}
        self.modules: dict[int, StaticManualVisionEncoder] = {}
        self.compile_metadata: dict[str, Any] = {
            "backend": self.backend,
            "enabled": self.backend == "torchair",
            "boundary": "vision_encoder_layers_plus_post_layernorm",
            "buckets": list(self.buckets),
            "overflow": "eager_unpadded",
        }
        if self.backend not in VISION_BACKEND_CHOICES:
            raise ValueError(f"vision backend must be one of {VISION_BACKEND_CHOICES}, got {backend!r}")
        if self.backend == "raw_eager":
            return
        if self.device.type != "npu":
            raise ValueError("compiled vision backend torchair requires an NPU device")
        if get_vision_attention_impl() != "manual":
            raise ValueError(
                "compiled vision currently preserves the manual-attention path; "
                "set PADDLE_OCR_VL_VISION_ATTENTION=manual"
            )

        torchair, CompilerConfig = import_torchair()
        hidden_size = int(model.config.vision_config.hidden_size)
        head_dim = hidden_size // int(model.config.vision_config.num_attention_heads)
        per_bucket: dict[str, Any] = {}
        wrapper_total_s = 0.0
        first_call_total_s = 0.0
        for bucket in self.buckets:
            module = StaticManualVisionEncoder(model).eval()
            cache_dir = vision_cache_dir_for_bucket(
                self.cache_root,
                bucket=bucket,
                dtype=self.dtype,
                device=self.device,
                model_dir=model_dir,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            config = CompilerConfig()
            entrypoint = unique_bucket_forward(module, bucket)
            synchronize(self.device)
            started = time.perf_counter()
            compiled = torchair.inference.cache_compile(
                entrypoint,
                config=config,
                dynamic=False,
                cache_dir=str(cache_dir),
                ge_cache=True,
            )
            synchronize(self.device)
            wrapper_s = time.perf_counter() - started

            warm_prefix = torch.zeros(
                (1, bucket, hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
            warm_cos = torch.ones(
                (1, bucket, head_dim),
                device=self.device,
                # The stock rotary table is derived from an fp32 inv_freq, so
                # real calls supply fp32 cos/sin even when hidden states are fp16.
                dtype=torch.float32,
            )
            warm_sin = torch.zeros_like(warm_cos)
            warm_mask = torch.zeros(
                (1, 1, bucket, bucket),
                device=self.device,
                dtype=torch.bool,
            )
            synchronize(self.device)
            started = time.perf_counter()
            warm_output = compiled(warm_prefix, warm_cos, warm_sin, warm_mask)
            synchronize(self.device)
            first_call_s = time.perf_counter() - started
            del warm_output, warm_prefix, warm_cos, warm_sin, warm_mask

            self.modules[bucket] = module
            self.entrypoints[bucket] = entrypoint
            self.compiled[bucket] = compiled
            wrapper_total_s += wrapper_s
            first_call_total_s += first_call_s
            per_bucket[str(bucket)] = {
                "compile_wrapper_s": float(wrapper_s),
                "compile_first_call_s": float(first_call_s),
                "torchair_cache_dir": str(cache_dir),
            }
        self.compile_metadata.update(
            {
                "compile_api": "torchair.inference.cache_compile",
                "dynamic": False,
                "fullgraph": True,
                "torchair_ge_cache": True,
                "compile_wrapper_total_s": float(wrapper_total_s),
                "compile_first_call_total_s": float(first_call_total_s),
                "per_bucket": per_bucket,
                "cache_key_fields": {
                    "dtype": str(dtype),
                    "model_config_hash": short_file_hash(model_dir / "config.json"),
                    "torch": str(torch.__version__),
                    "torch_npu": torch_npu_version_label(device),
                    "torchair": torchair_version_label(device),
                    "vision_source_hash": vision_source_hash(),
                    "attention": "manual_bmm",
                    "softmax_dtype": get_vision_softmax_dtype_mode(),
                },
            }
        )

    def route(self, real_seq_len: int) -> dict[str, Any]:
        real_seq_len = int(real_seq_len)
        if self.backend == "raw_eager":
            return {
                "execution": "eager",
                "real_vision_tokens": real_seq_len,
                "physical_vision_tokens": real_seq_len,
                "padding_vision_tokens": 0,
                "useful_token_fraction": 1.0,
                "bucket": None,
            }
        bucket = select_vision_bucket(real_seq_len, self.buckets)
        if bucket is None:
            return {
                "execution": "eager_overflow",
                "real_vision_tokens": real_seq_len,
                "physical_vision_tokens": real_seq_len,
                "padding_vision_tokens": 0,
                "useful_token_fraction": 1.0,
                "bucket": None,
            }
        return {
            "execution": "compiled",
            "real_vision_tokens": real_seq_len,
            "physical_vision_tokens": bucket,
            "padding_vision_tokens": bucket - real_seq_len,
            "useful_token_fraction": float(real_seq_len) / float(bucket),
            "bucket": bucket,
        }

    def prepare(
        self,
        prefix_hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor,
        *,
        bucket: int,
    ) -> PreparedVisionBucket:
        return prepare_vision_bucket(
            self.model,
            prefix_hidden_states,
            image_grid_thw,
            physical_seq_len=int(bucket),
        )

    def run_prepared(self, prepared: PreparedVisionBucket) -> torch.Tensor:
        output = self.compiled[prepared.physical_seq_len](
            prepared.prefix_hidden_states,
            prepared.rope_cos,
            prepared.rope_sin,
            prepared.attention_mask,
        )
        return output[0, : prepared.real_seq_len].contiguous()
