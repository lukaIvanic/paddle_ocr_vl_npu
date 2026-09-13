"""Unified eager and compiled model execution for vision prefill.

Patch embedding and absolute-position interpolation remain eager and operate at
the crop's real shape. This module prepares zero-padded or exact-shape inputs,
runs one compiler-safe vision stage either directly or through TorchAir, and
slices the real rows before the existing projector consumes them.
"""

from __future__ import annotations

import time
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch_npu
import torch.nn.functional as F
from torch import nn

from p06_text_prefill_and_decode import TEXT_HIDDEN_SIZE

if TYPE_CHECKING:
    from p04_paddle_ocr_vl_1_6_modeling import LocalPaddleOCRVLForConditionalGeneration


# Fixed PaddleOCR-VL-1.6 vision architecture.
VISION_HIDDEN_SIZE = 1152
VISION_INTERMEDIATE_SIZE = 4304
VISION_LAYERS = 27
VISION_HEADS = 16
VISION_CHANNELS = 3
VISION_IMAGE_SIZE = 384
VISION_PATCH_SIZE = 14
VISION_MERGE_SIZE = 2
VISION_NORM_EPS = 1e-6


VISION_BUCKETS = (256, 384, 512, 640, 768, 1408, 1920, 2048, 2944, 4096)
VISION_SEQUENCE_ALIGNMENT = 128
VISION_PROMPT_FA_FULL_ATTENTION_TOKENS = (1 << 31) - 1
VISION_FRACTAL_NZ_FORMAT = 29


# Per-image computation comes first: embed patches, run the vision encoder,
# then project its output into text-model embeddings. The serving runtime
# calls these parts in that order. Execution/bucketing follows; checkpoint
# parameter containers and one-time weight preparation are near the bottom.


# Patch and position embeddings


class PaddleOCRVisionEmbeddings(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed_dim = VISION_HIDDEN_SIZE
        self.image_size = VISION_IMAGE_SIZE
        self.patch_size = VISION_PATCH_SIZE
        self.patch_embedding = nn.Conv2d(
            in_channels=VISION_CHANNELS,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_len, channel, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(
            batch_size * sequence_len, channel, height, width
        )
        embeddings = self.project_patches(pixel_values)
        embeddings = embeddings.reshape(batch_size, sequence_len, -1).squeeze(0)
        start = 0
        tmp_embeddings = []
        for image_grid in image_grid_thw:
            t, h, w = [int(v.item()) for v in image_grid]
            end = start + t * h * w
            image_embeddings = embeddings[start:end, :]
            pos = (
                self.interpolate_pos_encoding(image_embeddings, h, w)
                .squeeze(0)
                .repeat(t, 1)
            )
            tmp_embeddings.append(image_embeddings + pos)
            start = end
        return torch.cat(tmp_embeddings, dim=0)

    def project_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixels = pixel_values.to(dtype=self.patch_embedding.weight.dtype)
        conv = self.patch_embedding
        # Preserve channel/row/column order, original parameter storage and bias.
        # One patch covers the entire convolution kernel, so there is exactly
        # one spatial output. No pixels or vision tokens are added or removed.
        return F.linear(pixels.flatten(1), conv.weight.flatten(1), conv.bias)

    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        num_positions = self.position_embedding.weight.shape[0]
        dim = embeddings.shape[-1]
        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        ).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)


# Vision encoder computation


class VisionPrefillStage(torch.nn.Module):
    """Vision encoder plus post LayerNorm for eager or compiled use."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
    ):
        super().__init__()
        self.transformer = model.visual.vision_model

    def forward(
        self,
        prefix_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = prefix_hidden_states
        rope_cos = pad_vision_rope_halves(rope_cos, 1.0)
        rope_sin = pad_vision_rope_halves(rope_sin, 0.0)
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
                F.gelu(
                    encoder_layer.mlp.fc1(mlp_input),
                    approximate="tanh",
                )
            )
        return self.transformer.post_layernorm(hidden_states)

    def _attention(
        self, attention, hidden_states, rope_cos, rope_sin, attention_mask,
    ):
        batch, seq, _ = hidden_states.shape
        heads = int(attention.num_heads)
        qk = torch.cat((attention.q_proj(hidden_states),
                        attention.k_proj(hidden_states)), dim=-1).view(batch, seq, 2 * heads, 80)
        value = attention.v_proj(hidden_states).view(batch, seq, heads, 80)
        qk_fp32 = qk.float()
        rotated = torch.cat((-qk_fp32[..., 40:], qk_fp32[..., :40]), dim=-1)
        qk = (qk_fp32 * rope_cos.unsqueeze(-2).float()
              + rotated * rope_sin.unsqueeze(-2).float()).to(qk.dtype)
        query, key = (qk.view(batch, seq, 2, heads, 80)
                      .permute(2, 0, 3, 1, 4).contiguous().unbind(0))
        output = vision_prompt_flash_attention_bnsd(
            query, key, value.transpose(1, 2).contiguous(),
            num_heads=heads, scale=float(attention.scaling), atten_mask=attention_mask,
        )
        return attention.out_proj(output.transpose(1, 2).contiguous().view(batch, seq, heads * 80))


def vision_prompt_flash_attention_bnsd(
    q_bnsd: torch.Tensor,
    k_bnsd: torch.Tensor,
    v_bnsd: torch.Tensor,
    *,
    num_heads: int,
    scale: float,
    atten_mask: torch.Tensor,
) -> torch.Tensor:
    """Full bidirectional vision attention with the prepared padding mask."""

    attention_mask = atten_mask.to(torch.bool).contiguous()
    return torch_npu.npu_prompt_flash_attention(
        q_bnsd.contiguous(),
        k_bnsd.contiguous(),
        v_bnsd.contiguous(),
        num_heads=int(num_heads),
        input_layout="BNSD",
        scale_value=float(scale),
        # Preserve the window arguments of the validated serving call.
        pre_tokens=VISION_PROMPT_FA_FULL_ATTENTION_TOKENS,
        next_tokens=VISION_PROMPT_FA_FULL_ATTENTION_TOKENS,
        sparse_mode=1,
        atten_mask=attention_mask,
    )


def pad_vision_rope_halves(value: torch.Tensor, fill: float) -> torch.Tensor:
    # Neutral coordinates in each half, not a new D80 frequency table.
    first, second = value.chunk(2, dim=-1)
    return torch.cat((F.pad(first, (0, 4), value=fill),
                      F.pad(second, (0, 4), value=fill)), dim=-1)


# Projection into text-model embeddings


class PaddleOCRProjector(nn.Module):

    def __init__(self):
        super().__init__()
        merge = VISION_MERGE_SIZE
        hidden_size = VISION_HIDDEN_SIZE * merge * merge
        self.merge_kernel_size = (merge, merge)
        self.pre_norm = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=1e-5)
        self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_2 = nn.Linear(
            hidden_size, TEXT_HIDDEN_SIZE, bias=True
        )

    def forward(
        self,
        image_features: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        chunks = image_features.split(image_grid_thw.prod(dim=1).tolist(), dim=0)
        m1, m2 = self.merge_kernel_size
        processed = []
        for image_feature, image_grid in zip(chunks, image_grid_thw):
            image_feature = self.pre_norm(image_feature)
            t, h, w = [int(v.item()) for v in image_grid]
            d = image_feature.shape[-1]
            h_block = h // m1
            w_block = w // m2
            image_feature = image_feature.reshape(
                t, h_block, m1, w_block, m2, d
            )
            image_feature = image_feature.transpose(2, 3)
            image_feature = image_feature.reshape(
                t * h_block * w_block, m1 * m2 * d
            )
            hidden_states = self.linear_1(image_feature)
            hidden_states = F.gelu(hidden_states)
            hidden_states = self.linear_2(hidden_states)
            processed.append(hidden_states)
        return torch.cat(processed, dim=0)


# Request execution, input preparation, and bucket setup


class VisionPrefillRuntime:
    """Run one vision-prefill stage eagerly or through static bucket graphs."""

    def run_prepared(self, prepared: PreparedVisionPrefill) -> torch.Tensor:
        run = (
            self.compiled[prepared.physical_seq_len]
            if prepared.execution == "compiled"
            else self.eager_stage
        )
        output = run(
            prepared.prefix_hidden_states,
            prepared.rope_cos,
            prepared.rope_sin,
            prepared.attention_mask,
        )
        return output[0, : prepared.real_seq_len].contiguous()

    def route(self, real_seq_len: int) -> dict[str, Any]:
        real_seq_len = int(real_seq_len)
        bucket = select_vision_bucket(real_seq_len)
        if bucket is None:
            physical_seq_len = align_vision_seq_len(real_seq_len)
            return {
                "execution": "eager_overflow",
                "real_vision_tokens": real_seq_len,
                "physical_vision_tokens": physical_seq_len,
                "padding_vision_tokens": physical_seq_len - real_seq_len,
                "useful_token_fraction": (
                    float(real_seq_len) / float(physical_seq_len)
                ),
                "bucket": None,
            }
        return {
            "execution": (
                "eager_padded" if self.eager else "compiled"
            ),
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
        route: dict[str, Any],
    ) -> PreparedVisionPrefill:
        return prepare_vision_prefill(
            self.model,
            prefix_hidden_states,
            image_grid_thw,
            physical_seq_len=int(route["physical_vision_tokens"]),
            execution=str(route["execution"]),
        )

    # Startup only: construct and warm each bucket before serving requests.
    # The request methods above reuse these stages and compiled entrypoints.

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        graph_directories: dict[int, Path],
        device: torch.device,
        eager: bool = False,
        setup_progress: Callable[[str, str, float | None], None] | None = None,
    ):
        self.model = model
        self.eager = eager
        self.buckets = VISION_BUCKETS
        self.device = device
        self.dtype = torch.float16
        hidden_size = int(VISION_HIDDEN_SIZE)
        head_dim = hidden_size // int(VISION_HEADS)
        self.compiled: dict[int, Callable[..., torch.Tensor]] = {}
        self.entrypoints: dict[int, Callable[..., torch.Tensor]] = {}
        self.eager_stage = VisionPrefillStage(model).eval()
        self.modules: dict[int, VisionPrefillStage] = {}
        self.metadata = {
            "backend": "raw_eager" if eager else "torchair",
            "enabled": not eager,
            "boundary": "vision_encoder_layers_plus_post_layernorm",
            "buckets": list(self.buckets),
            "sequence_alignment": VISION_SEQUENCE_ALIGNMENT,
            "padding": "bucket",
            "overflow": "eager_same_stage_unpadded",
        }
        if eager:
            return

        # Eager mode does not import the compiler.
        import torchair.inference
        from torchair import CompilerConfig
        per_bucket: dict[str, Any] = {}
        wrapper_total_s = 0.0
        first_call_total_s = 0.0
        for index, bucket in enumerate(self.buckets, 1):
            graph_label = f"vision graph {index}/{len(self.buckets)}, tokens={bucket}"
            if setup_progress is not None:
                setup_progress(graph_label, "start", None)
            module = VisionPrefillStage(model).eval()
            cache_dir = graph_directories[bucket]
            cache_dir.mkdir(parents=True, exist_ok=True)
            config = CompilerConfig()
            entrypoint = unique_bucket_forward(module, bucket)
            torch_npu.npu.synchronize(self.device)
            started = time.perf_counter()
            compiled = torchair.inference.cache_compile(
                entrypoint,
                config=config,
                dynamic=False,
                cache_dir=str(cache_dir),
                ge_cache=True,
            )
            torch_npu.npu.synchronize(self.device)
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
            torch_npu.npu.synchronize(self.device)
            started = time.perf_counter()
            warm_output = compiled(warm_prefix, warm_cos, warm_sin, warm_mask)
            torch_npu.npu.synchronize(self.device)
            first_call_s = time.perf_counter() - started
            del warm_output, warm_prefix, warm_cos, warm_sin, warm_mask

            self.modules[bucket] = module
            self.entrypoints[bucket] = entrypoint
            self.compiled[bucket] = compiled
            wrapper_total_s += wrapper_s
            first_call_total_s += first_call_s
            if setup_progress is not None:
                setup_progress(graph_label, "done", wrapper_s + first_call_s)
            per_bucket[str(bucket)] = {
                "compile_wrapper_s": float(wrapper_s),
                "compile_first_call_s": float(first_call_s),
                "torchair_cache_dir": str(cache_dir),
            }
        self.metadata.update(
            {
                "compile_api": "torchair.inference.cache_compile",
                "dynamic": False,
                "fullgraph": True,
                "torchair_ge_cache": True,
                "compile_wrapper_total_s": float(wrapper_total_s),
                "compile_first_call_total_s": float(first_call_total_s),
                "per_bucket": per_bucket,
                "cache_key_fields": {
                    "dtype": str(self.dtype),
                    "torch": str(torch.__version__),
                    "attention": "prompt_flash_attention",
                    "prompt_flash_attention_layout": 'bnsd',
                    "prompt_flash_attention_mask_sparse_mode": 1,
                    "softmax_dtype": 'fp32',
                    "execution_mode": "inference",
                },
            }
        )


@dataclass(frozen=True)
class PreparedVisionPrefill:
    prefix_hidden_states: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    attention_mask: torch.Tensor
    real_seq_len: int
    physical_seq_len: int
    execution: str


def prepare_vision_prefill(
    model: LocalPaddleOCRVLForConditionalGeneration,
    prefix_hidden_states: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    physical_seq_len: int,
    execution: str,
) -> PreparedVisionPrefill:
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
    return PreparedVisionPrefill(
        prefix_hidden_states=prefix,
        rope_cos=rope_cos.unsqueeze(0).contiguous(),
        rope_sin=rope_sin.unsqueeze(0).contiguous(),
        attention_mask=attention_mask.contiguous(),
        real_seq_len=real_seq_len,
        physical_seq_len=physical_seq_len,
        execution=str(execution),
    )


def select_vision_bucket(real_seq_len: int) -> int | None:
    for bucket in VISION_BUCKETS:
        if real_seq_len <= bucket:
            return bucket
    return None


def align_vision_seq_len(seq_len: int) -> int:
    """Round a request's physical vision length to the fixed alignment."""
    return ((seq_len + 127) // 128) * 128


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


# Checkpoint structure: these containers own the layers used by the
# computations above. Their nested names match the checkpoint weight names.


class PaddleOCRVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = PaddleOCRVisionTransformer()

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.embeddings.patch_embedding.weight.dtype


class PaddleOCRVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = PaddleOCRVisionEmbeddings()
        self.encoder = PaddleOCRVisionEncoder()
        self.post_layernorm = nn.LayerNorm(
            VISION_HIDDEN_SIZE, eps=VISION_NORM_EPS
        )


class PaddleOCRVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PaddleOCRVisionEncoderLayer()
                for _ in range(VISION_LAYERS)
            ]
        )
        head_dim = VISION_HIDDEN_SIZE // VISION_HEADS
        self.rotary_pos_emb = PaddleOCRVisionRotaryEmbedding(head_dim // 2)


class PaddleOCRVisionEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(
            VISION_HIDDEN_SIZE, eps=VISION_NORM_EPS
        )
        self.self_attn = PaddleOCRVisionAttention()
        self.layer_norm2 = nn.LayerNorm(
            VISION_HIDDEN_SIZE, eps=VISION_NORM_EPS
        )
        self.mlp = PaddleOCRVisionMLP()


class PaddleOCRVisionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = VISION_HIDDEN_SIZE
        self.num_heads = VISION_HEADS
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)


class PaddleOCRVisionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(VISION_HIDDEN_SIZE, VISION_INTERMEDIATE_SIZE)
        self.fc2 = nn.Linear(VISION_INTERMEDIATE_SIZE, VISION_HIDDEN_SIZE)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class PaddleOCRVisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = int(dim)
        self.theta = float(theta)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def forward(self, seqlen: int | torch.Tensor) -> torch.Tensor:
        seq = torch.arange(
            int(seqlen),
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        return torch.outer(seq, self.inv_freq)

    def reset_inv_freq(self, device: torch.device | None = None) -> None:
        self.register_buffer(
            "inv_freq",
            self._compute_inv_freq().to(device=device),
            persistent=False,
        )

    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )


# One-time weight preparation


@torch.no_grad()
def prepare_vision_attention_weight_padding(model: nn.Module) -> None:
    """Zero-extend D72 projections to D80, preserving both RoPE halves.

    Load-time only, before NZ conversion. The stage below owns execution of
    these weights; original head_dim/scaling and rotary frequencies stay D72.
    This is the joint-manual formulation measured in vision_matmul_lab.
    """
    layers = model.visual.vision_model.encoder.layers
    for layer in layers:
        attn = layer.self_attn
        heads = int(attn.num_heads)
        indices = torch.tensor(
            [h * 80 + i + (4 if i >= 36 else 0)
             for h in range(heads) for i in range(72)],
            device=attn.q_proj.weight.device, dtype=torch.long,
        )
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            source = getattr(attn, name)
            output_projection = name == "out_proj"
            replacement = nn.Linear(
                heads * 80 if output_projection else source.in_features,
                source.out_features if output_projection else heads * 80,
                bias=True,
                device=source.weight.device, dtype=source.weight.dtype,
            )
            replacement.weight.zero_().index_copy_(
                1 if output_projection else 0, indices, source.weight
            )
            if output_projection:
                replacement.bias.copy_(source.bias)
            else:
                replacement.bias.zero_().index_copy_(0, indices, source.bias)
            setattr(attn, name, replacement)


def prepare_vision_mlp_intermediate(
    model: LocalPaddleOCRVLForConditionalGeneration,
) -> dict[str, Any]:
    """Zero-extend every vision MLP without changing the represented function."""
    layers = tuple(model.visual.vision_model.encoder.layers)
    source_intermediate_size = int(layers[0].mlp.fc1.out_features)
    target = 4352

    for layer in layers:
        source_fc1 = layer.mlp.fc1
        source_fc2 = layer.mlp.fc2
        hidden_size = int(source_fc1.in_features)
        device = source_fc1.weight.device
        dtype = source_fc1.weight.dtype
        fc1 = nn.Linear(
            hidden_size,
            target,
            bias=True,
            device=device,
            dtype=dtype,
        )
        fc2 = nn.Linear(
            target,
            hidden_size,
            bias=True,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            fc1.weight.zero_()
            fc1.weight[:source_intermediate_size].copy_(source_fc1.weight)
            fc1.bias.zero_()
            fc1.bias[:source_intermediate_size].copy_(source_fc1.bias)
            fc2.weight.zero_()
            fc2.weight[:, :source_intermediate_size].copy_(source_fc2.weight)
            fc2.bias.copy_(source_fc2.bias)
        layer.mlp.fc1 = fc1
        layer.mlp.fc2 = fc2

    return {
        "source_intermediate_size": source_intermediate_size,
        "target_intermediate_size": target,
        "layer_count": len(layers),
        "zero_extended": True,
    }


def prepare_vision_linear_weight_format(
    model: LocalPaddleOCRVLForConditionalGeneration,
) -> dict[str, Any]:
    """Precast all six vision Linear weights per layer to FRACTAL_NZ."""

    layers = tuple(model.visual.vision_model.encoder.layers)
    modules: list[tuple[str, nn.Linear]] = []
    for layer_index, layer in enumerate(layers):
        modules.extend(
            (
                (f"layers.{layer_index}.self_attn.q_proj", layer.self_attn.q_proj),
                (f"layers.{layer_index}.self_attn.k_proj", layer.self_attn.k_proj),
                (f"layers.{layer_index}.self_attn.v_proj", layer.self_attn.v_proj),
                (f"layers.{layer_index}.self_attn.out_proj", layer.self_attn.out_proj),
                (f"layers.{layer_index}.mlp.fc1", layer.mlp.fc1),
                (f"layers.{layer_index}.mlp.fc2", layer.mlp.fc2),
            )
        )

    def histogram() -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    str(int(torch_npu.get_npu_format(module.weight)))
                    for _name, module in modules
                ).items()
            )
        )

    before = histogram()
    converted = 0
    for name, module in modules:
        before_code = int(torch_npu.get_npu_format(module.weight))
        if before_code == VISION_FRACTAL_NZ_FORMAT:
            continue
        module.weight.data = torch_npu.npu_format_cast(
            module.weight.data, VISION_FRACTAL_NZ_FORMAT,
        )
        after_code = int(torch_npu.get_npu_format(module.weight))
        if after_code != VISION_FRACTAL_NZ_FORMAT:
            raise RuntimeError(
                "vision linear format cast did not produce FRACTAL_NZ: "
                f"module={name} before={before_code} after={after_code}"
            )
        converted += 1
    after = histogram()
    all_after_are_nz = all(
        int(torch_npu.get_npu_format(module.weight))
        == VISION_FRACTAL_NZ_FORMAT
        for _name, module in modules
    )
    if not all_after_are_nz:
        raise RuntimeError(
            "not all vision Linear weights are FRACTAL_NZ after conversion: "
            f"{after}"
        )
    return {
        "target_format": "FRACTAL_NZ",
        "target_format_code": VISION_FRACTAL_NZ_FORMAT,
        "linear_weight_count": len(modules),
        "before_format_histogram": before,
        "after_format_histogram": after,
        "converted_count": converted,
        "all_after_are_nz": all_after_are_nz,
    }


# Compilation helper: each bucket keeps its own Python entrypoint identity.
# VisionPrefillRuntime uses this during its one-time setup above.

def unique_bucket_forward(
    module: VisionPrefillStage,
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
