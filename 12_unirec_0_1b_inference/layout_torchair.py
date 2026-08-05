"""Static fullgraph TorchAir runtime for PP-DocLayoutV2 B1/800x800."""

from __future__ import annotations

import hashlib
import math
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.pp_doclayout_v2 import modeling_pp_doclayout_v2 as layout_mod


def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, device="cpu", dtype=torch.float32):
    if spatial_shapes is None:
        spatial_shapes = [
            [int(self.config.anchor_image_size[0] / stride), int(self.config.anchor_image_size[1] / stride)]
            for stride in self.config.feat_strides
        ]
    anchors = []
    for level, (height, width) in enumerate(spatial_shapes):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(end=height, device=device).to(dtype),
            torch.arange(end=width, device=device).to(dtype),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], -1).unsqueeze(0) + 0.5
        grid_xy = torch.cat(
            [grid_xy[..., 0:1] / width, grid_xy[..., 1:2] / height], dim=-1
        )
        wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
        anchors.append(torch.cat([grid_xy, wh], -1).reshape(-1, height * width, 4))
    anchors = torch.cat(anchors, 1)
    valid_components = (anchors > 1e-2).to(torch.int32) * (anchors < 1 - 1e-2).to(torch.int32)
    valid_mask = valid_components.sum(dim=-1, keepdim=True) == 4
    anchors = torch.log(anchors / (1 - anchors))
    replacement = torch.tensor(torch.finfo(dtype).max, dtype=dtype, device=device)
    return torch.where(valid_mask, anchors, replacement), valid_mask


def _reading_order(self, boxes, labels=None, mask=None, **kwargs):
    batch_size, seq_len = mask.shape
    num_pred = mask.sum(dim=1)
    positions = torch.arange(seq_len + 2, device=mask.device).unsqueeze(0)
    input_ids = torch.full(
        (batch_size, seq_len + 2), self.config.pad_token_id, dtype=torch.long, device=boxes.device
    )
    input_ids = torch.where(
        positions == 0, torch.full_like(input_ids, self.config.start_token_id), input_ids
    )
    pred_mask = (positions >= 1) & (positions <= num_pred.unsqueeze(1))
    input_ids = torch.where(
        pred_mask, torch.full_like(input_ids, self.config.pred_token_id), input_ids
    )
    input_ids = torch.where(
        positions == (num_pred + 1).unsqueeze(1),
        torch.full_like(input_ids, self.config.end_token_id),
        input_ids,
    )
    pad_box = torch.zeros(
        (boxes.shape[0], 1, boxes.shape[-1]), dtype=boxes.dtype, device=boxes.device
    )
    pad_boxes = torch.cat([pad_box, boxes, pad_box], dim=1)
    bbox_embedding = self.embeddings(input_ids=input_ids, bbox=pad_boxes.long())
    if labels is not None:
        label_proj = self.label_features_projection(self.label_embeddings(labels))
        pad = torch.zeros(
            (label_proj.shape[0], 1, label_proj.shape[-1]),
            dtype=label_proj.dtype,
            device=labels.device,
        )
        label_proj = torch.cat([pad, label_proj, pad], dim=1)
    else:
        label_proj = torch.zeros_like(bbox_embedding)
    embeddings = self.embeddings.dropout(self.embeddings.norm(bbox_embedding + label_proj))
    attention_mask = positions < (num_pred + 2).unsqueeze(1)
    attention_mask = layout_mod.create_bidirectional_mask(
        config=self.config, inputs_embeds=embeddings, attention_mask=attention_mask
    )
    encoded = self.encoder(
        hidden_states=embeddings, bbox=pad_boxes, attention_mask=attention_mask
    ).last_hidden_state
    return self.relative_head(encoded[:, 1 : 1 + seq_len, :])


def _cogview_attention(self, attention_scores, alpha=32):
    scaled = attention_scores / alpha
    maximum = torch.max(scaled, dim=-1, keepdim=True).values
    return torch.softmax((scaled - maximum) * alpha, dim=-1)


def _model_self_attention(
    self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs
):
    batch_size, sequence_length, hidden_size = hidden_states.shape
    head_dim = self.head_dim
    num_heads = hidden_size // head_dim
    query_key_input = (
        hidden_states + position_embeddings if position_embeddings is not None else hidden_states
    )
    query = self.q_proj(query_key_input).reshape(
        batch_size, sequence_length, num_heads, head_dim
    ).permute(0, 2, 1, 3).contiguous()
    key = self.k_proj(query_key_input).reshape(
        batch_size, sequence_length, num_heads, head_dim
    ).permute(0, 2, 1, 3).contiguous()
    value = self.v_proj(hidden_states).reshape(
        batch_size, sequence_length, num_heads, head_dim
    ).permute(0, 2, 1, 3).contiguous()
    query = query.reshape(batch_size * num_heads, sequence_length, head_dim)
    key = key.reshape(batch_size * num_heads, sequence_length, head_dim)
    value = value.reshape(batch_size * num_heads, sequence_length, head_dim)
    weights = torch.bmm(query, key.transpose(1, 2)) * self.scaling
    weights = weights.reshape(batch_size, num_heads, sequence_length, sequence_length)
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = torch.softmax(weights, dim=-1)
    output = torch.bmm(
        weights.reshape(batch_size * num_heads, sequence_length, sequence_length), value
    ).reshape(batch_size, num_heads, sequence_length, head_dim)
    output = output.transpose(1, 2).contiguous().reshape(
        batch_size, sequence_length, hidden_size
    )
    return self.o_proj(output), weights


def _reading_order_attention(
    self, hidden_states, attention_mask=None, rel_pos=None, rel_2d_pos=None, **kwargs
):
    batch_size, sequence_length, _ = hidden_states.shape
    num_heads = self.num_attention_heads
    head_dim = self.attention_head_size
    query = self.query(hidden_states).reshape(
        batch_size, sequence_length, num_heads, head_dim
    ).permute(0, 2, 1, 3).contiguous()
    key = self.key(hidden_states).reshape(
        batch_size, sequence_length, num_heads, head_dim
    ).permute(0, 2, 1, 3).contiguous()
    value = self.value(hidden_states).reshape(
        batch_size, sequence_length, num_heads, head_dim
    ).permute(0, 2, 1, 3).contiguous()
    query = (query / math.sqrt(head_dim)).reshape(
        batch_size * num_heads, sequence_length, head_dim
    )
    key = key.reshape(batch_size * num_heads, sequence_length, head_dim)
    value = value.reshape(batch_size * num_heads, sequence_length, head_dim)
    scores = torch.bmm(query, key.transpose(1, 2)).reshape(
        batch_size, num_heads, sequence_length, sequence_length
    )
    if rel_2d_pos is not None:
        scores = scores + rel_2d_pos
    elif self.has_relative_attention_bias:
        scores = scores + rel_pos / math.sqrt(head_dim)
    if attention_mask is not None:
        scores = scores + attention_mask
    probabilities = _cogview_attention(self, scores)
    probabilities = self.dropout(probabilities)
    context = torch.bmm(
        probabilities.reshape(batch_size * num_heads, sequence_length, sequence_length),
        value,
    ).reshape(batch_size, num_heads, sequence_length, head_dim)
    context = context.permute(0, 2, 1, 3).contiguous().reshape(
        batch_size, sequence_length, self.all_head_size
    )
    return context, probabilities


def _global_pointer(self, inputs):
    batch_size, sequence_length, _ = inputs.shape
    projected = self.dropout(self.dense(inputs)).reshape(
        batch_size, sequence_length, 2, self.head_size
    )
    queries, keys = torch.unbind(projected, dim=2)
    logits = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.head_size)
    mask = torch.tril(torch.ones(sequence_length, sequence_length, device=logits.device)).bool()
    return logits.masked_fill(mask.unsqueeze(0), -1e4)


def _sine_position(self, width, height, device, dtype):
    grid_w = torch.arange(width, device=device).to(dtype)
    grid_h = torch.arange(height, device=device).to(dtype)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")
    pos_dim = self.embed_dim // 4
    omega = torch.arange(pos_dim, device=device).to(dtype) / pos_dim
    omega = 1.0 / (self.temperature**omega)
    out_w = grid_w.flatten().unsqueeze(-1) * omega.unsqueeze(0)
    out_h = grid_h.flatten().unsqueeze(-1) * omega.unsqueeze(0)
    return torch.cat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1).unsqueeze(0)


def _linear_2d(self, input_tensor):
    if input_tensor.ndim <= 2:
        return F.linear(input_tensor, self.weight, self.bias)
    leading_shape = input_tensor.shape[:-1]
    output = F.linear(
        input_tensor.reshape(-1, self.in_features), self.weight, self.bias
    )
    return output.reshape(*leading_shape, self.out_features)


def _bind(instance: Any, method: Any) -> None:
    instance.forward = types.MethodType(method, instance)


def make_compile_compatible(model: nn.Module) -> None:
    """Apply algebraically equivalent rewrites only to this layout model."""
    layout_mod.torch_compilable_check = lambda *args, **kwargs: None
    model.model.generate_anchors = types.MethodType(_generate_anchors, model.model)
    _bind(model.reading_order, _reading_order)
    for module in model.modules():
        if isinstance(module, layout_mod.PPDocLayoutV2SelfAttention):
            _bind(module, _model_self_attention)
        elif isinstance(module, layout_mod.PPDocLayoutV2ReadingOrderSelfAttention):
            _bind(module, _reading_order_attention)
        elif isinstance(module, layout_mod.PPDocLayoutV2GlobalPointer):
            _bind(module, _global_pointer)
        elif isinstance(module, layout_mod.PPDocLayoutV2SinePositionEmbedding):
            _bind(module, _sine_position)
        elif isinstance(module, nn.Linear):
            _bind(module, _linear_2d)


class LayoutFullGraph(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        output = self.model(pixel_values=pixel_values)
        return output.logits, output.pred_boxes, output.order_logits


def _cache_compile():
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile
    except ImportError:
        from torchair.inference import cache_compile
    return cache_compile


class LayoutFullGraphRuntime:
    def __init__(
        self,
        model: nn.Module,
        *,
        cache_root: Path,
        dtype: torch.dtype,
        device: torch.device,
        warmup_passes: int = 2,
    ) -> None:
        make_compile_compatible(model)
        self.stage = LayoutFullGraph(model).eval()
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        dtype_name = str(dtype).removeprefix("torch.")
        self.cache_dir = cache_root.expanduser().resolve() / (
            f"layout_b1_800x800_{dtype_name}_src{source_hash}"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        config.mode.value = "max-autotune"
        self.compiled = _cache_compile()(
            self.stage.forward,
            config=config,
            dynamic=False,
            cache_dir=str(self.cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        self.warmup = self._warmup(
            passes=warmup_passes, dtype=dtype, device=device
        )

    def _warmup(self, *, passes: int, dtype: torch.dtype, device: torch.device) -> dict[str, Any]:
        sample = torch.zeros((1, 3, 800, 800), dtype=dtype, device=device)
        pass_wall_s = []
        with torch.inference_mode():
            for index in range(passes):
                started = time.perf_counter()
                self.compiled(sample)
                torch.npu.synchronize()
                elapsed = time.perf_counter() - started
                pass_wall_s.append(elapsed)
                print(
                    f"LAYOUT_GRAPH_WARMUP pass={index + 1}/{passes} wall_s={elapsed:.3f}",
                    flush=True,
                )
        return {
            "passes": passes,
            "pass_wall_s": pass_wall_s,
            "cache_dir": str(self.cache_dir),
            "dynamic": False,
            "fullgraph": True,
        }

    def __call__(self, pixel_values: torch.Tensor):
        return self.compiled(pixel_values)
