"""Static fullgraph TorchAir runtime for PP-DocLayoutV2 at 800x800."""

from __future__ import annotations

import hashlib
import math
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.pp_doclayout_v2 import modeling_pp_doclayout_v2 as layout_mod


COGVIEW_ATTENTION_IMPL_CHOICES = (
    "stabilized",
    "direct_softmax",
)


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
    # Ascend 310P rejects torch.where when one branch is a 0-D device tensor.
    # Materialize the broadcast explicitly. This is algebraically identical.
    replacement = torch.full_like(anchors, torch.finfo(dtype).max)
    return torch.where(valid_mask, anchors, replacement), valid_mask


def _reading_order_attention_bias(
    valid_tokens: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Build the eager bidirectional key-padding bias without indexed masks.

    Transformers 5.5.4 routes ``create_bidirectional_mask`` through a mask
    function that reads ``padding_mask[batch_idx, kv_idx]`` with broadcast
    tensor indices, then converts the result with a scalar-tensor
    ``torch.where`` branch. Both forms can dispatch to the failing AICPU Index
    path on Ascend 310P. Reading-order attention is fully bidirectional, so its
    only restriction is whether each key token is valid. Materialize that bias
    directly with full-shape branches and view expansion.
    """
    valid_tokens = valid_tokens.bool()
    zeros = torch.zeros_like(valid_tokens, dtype=dtype)
    negative = torch.full_like(valid_tokens, torch.finfo(dtype).min, dtype=dtype)
    key_bias = torch.where(valid_tokens, zeros, negative)
    sequence_length = valid_tokens.shape[1]
    return key_bias.unsqueeze(1).unsqueeze(1).expand(
        valid_tokens.shape[0], 1, sequence_length, sequence_length
    )


def _table_lookup(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Replace tensor advanced indexing with the supported embedding lookup."""
    if table.ndim == 1:
        return F.embedding(indices, table.unsqueeze(-1)).squeeze(-1)
    return F.embedding(indices, table)


def _reading_order_cal_1d_pos_emb(self, position_ids):
    """Build relative-position bias without ``weight[relative_positions]``."""
    relative_positions = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
    buckets = self.relative_position_bucket(
        relative_positions,
        num_buckets=self.rel_pos_bins,
        max_distance=self.max_rel_pos,
    )
    with torch.no_grad():
        relative_bias = _table_lookup(self.rel_pos_bias.weight.t(), buckets)
    return relative_bias.permute(0, 3, 1, 2).contiguous()


def _reading_order(self, boxes, labels=None, mask=None, **kwargs):
    # The detector and reading-order head may deliberately use different
    # floating-point dtypes.  Box coordinates are integerized for the text
    # embeddings, but the reading-order encoder also consumes the padded box
    # tensor directly.  Keep that tensor in the head's parameter dtype.
    head_dtype = self.label_embeddings.weight.dtype
    if boxes.is_floating_point() and boxes.dtype != head_dtype:
        boxes = boxes.to(dtype=head_dtype)
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
    valid_tokens = positions < (num_pred + 2).unsqueeze(1)
    attention_mask = _reading_order_attention_bias(
        valid_tokens, embeddings.dtype
    )
    encoded = self.encoder(
        hidden_states=embeddings, bbox=pad_boxes, attention_mask=attention_mask
    ).last_hidden_state
    return self.relative_head(encoded[:, 1 : 1 + seq_len, :])


def _object_detection_forward(
    self,
    pixel_values,
    pixel_mask=None,
    encoder_outputs=None,
    inputs_embeds=None,
    decoder_inputs_embeds=None,
    labels=None,
    **kwargs,
):
    """Run the inference forward without data-dependent tensor indexing.

    Transformers 5.5.4 uses three advanced table reads in this path: class
    thresholds, class-order remapping, and reading-order relative-position
    bias. Atlas 310P lowers those reads to its unsupported ``IndexByTensor``
    AICPU kernel. The first two are rewritten here as embedding lookups; the
    relative-position read is patched on its owning encoder above.
    """
    outputs = self.model(
        pixel_values,
        pixel_mask=pixel_mask,
        encoder_outputs=encoder_outputs,
        inputs_embeds=inputs_embeds,
        decoder_inputs_embeds=decoder_inputs_embeds,
        labels=labels,
        **kwargs,
    )

    raw_bboxes = outputs.intermediate_reference_points[:, -1]
    logits = outputs.intermediate_logits[:, -1]

    box_centers, box_sizes = raw_bboxes.split(2, dim=-1)
    bboxes = torch.cat(
        [box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes],
        dim=-1,
    ) * 1000
    bboxes = bboxes.clamp_(0.0, 1000.0)

    max_logits, class_ids = logits.max(dim=-1)
    max_probs = max_logits.sigmoid()
    class_thresholds = torch.tensor(
        self.config.class_thresholds,
        dtype=torch.float32,
        device=logits.device,
    )
    thresholds = _table_lookup(class_thresholds, class_ids)
    mask = max_probs >= thresholds

    indices = torch.argsort(mask.to(torch.int8), dim=1, descending=True)
    sorted_class_ids = torch.take_along_dim(class_ids, indices, dim=1)
    expanded_box_indices = indices.unsqueeze(-1).expand(-1, -1, 4)
    sorted_boxes = torch.take_along_dim(bboxes, expanded_box_indices, dim=1)
    pred_boxes = torch.take_along_dim(raw_bboxes, expanded_box_indices, dim=1)
    expanded_logit_indices = indices.unsqueeze(-1).expand(
        -1, -1, logits.shape[-1]
    )
    logits = torch.take_along_dim(logits, expanded_logit_indices, dim=1)
    sorted_mask = torch.take_along_dim(mask, indices, dim=1)

    pad_boxes = torch.where(
        sorted_mask.unsqueeze(-1), sorted_boxes, torch.zeros_like(sorted_boxes)
    )
    pad_class_ids = torch.where(
        sorted_mask, sorted_class_ids, torch.zeros_like(sorted_class_ids)
    )
    class_order = torch.tensor(
        self.config.class_order,
        dtype=torch.float32,
        device=logits.device,
    )
    pad_class_ids = _table_lookup(class_order, pad_class_ids).to(torch.int32)

    order_logits = self.reading_order(
        boxes=pad_boxes,
        labels=pad_class_ids,
        mask=mask,
    )
    order_logits = order_logits[:, :, : self.num_queries]

    if labels is not None:
        raise ValueError("PPDocLayoutV2ForObjectDetection does not support training")

    return layout_mod.PPDocLayoutV2ForObjectDetectionOutput(
        logits=logits,
        pred_boxes=pred_boxes,
        order_logits=order_logits,
        last_hidden_state=outputs.last_hidden_state,
        intermediate_hidden_states=outputs.intermediate_hidden_states,
        intermediate_logits=outputs.intermediate_logits,
        intermediate_reference_points=outputs.intermediate_reference_points,
        intermediate_predicted_corners=outputs.intermediate_predicted_corners,
        initial_reference_points=outputs.initial_reference_points,
        decoder_hidden_states=outputs.decoder_hidden_states,
        decoder_attentions=outputs.decoder_attentions,
        cross_attentions=outputs.cross_attentions,
        encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        encoder_hidden_states=outputs.encoder_hidden_states,
        encoder_attentions=outputs.encoder_attentions,
        init_reference_points=outputs.init_reference_points,
        enc_topk_logits=outputs.enc_topk_logits,
        enc_topk_bboxes=outputs.enc_topk_bboxes,
        enc_outputs_class=outputs.enc_outputs_class,
        enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
        denoising_meta_values=outputs.denoising_meta_values,
    )


def _cogview_attention_stabilized(attention_scores, alpha=32):
    scaled = attention_scores / alpha
    maximum = torch.max(scaled, dim=-1, keepdim=True).values
    return torch.softmax((scaled - maximum) * alpha, dim=-1)


def _cogview_attention_direct_softmax(attention_scores):
    """Use Softmax's own stable maximum subtraction.

    For positive ``alpha``, the stabilized expression above is algebraically
    equivalent to ``softmax(attention_scores)``. Keep this as an explicit
    diagnostic lane until NPU output parity is validated on both target chips.
    """
    return torch.softmax(attention_scores, dim=-1)


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
    if self._unirec_cogview_direct_softmax:
        probabilities = _cogview_attention_direct_softmax(scores)
    else:
        probabilities = _cogview_attention_stabilized(scores)
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


def _nearest_upsample2d_2x_exact(inputs: torch.Tensor) -> torch.Tensor:
    """Duplicate each NCHW value into a 2x2 block without tensor indexing."""
    batch, channels, height, width = inputs.shape
    return (
        inputs.reshape(batch, channels, height, 1, width, 1)
        .expand(batch, channels, height, 2, width, 2)
        .reshape(batch, channels, height * 2, width * 2)
    )


def _hybrid_encoder(self, inputs_embeds=None, **kwargs):
    """Run the stock hybrid encoder with an index-free exact 2x upsample."""
    feature_maps = inputs_embeds

    if self.config.encoder_layers > 0:
        for index, encoder_index in enumerate(self.encode_proj_layers):
            feature_maps[encoder_index] = self.aifi[index](
                feature_maps[encoder_index],
                **kwargs,
            )

    fpn_feature_maps = [feature_maps[-1]]
    for index, (lateral_conv, fpn_block) in enumerate(
        zip(self.lateral_convs, self.fpn_blocks)
    ):
        backbone_feature_map = feature_maps[self.num_fpn_stages - index - 1]
        top_fpn_feature_map = lateral_conv(fpn_feature_maps[-1])
        fpn_feature_maps[-1] = top_fpn_feature_map
        top_fpn_feature_map = _nearest_upsample2d_2x_exact(
            top_fpn_feature_map
        )
        fused_feature_map = torch.concat(
            [top_fpn_feature_map, backbone_feature_map],
            dim=1,
        )
        fpn_feature_maps.append(fpn_block(fused_feature_map))

    fpn_feature_maps.reverse()
    pan_feature_maps = [fpn_feature_maps[0]]
    for index, (downsample_conv, pan_block) in enumerate(
        zip(self.downsample_convs, self.pan_blocks)
    ):
        downsampled_feature_map = downsample_conv(pan_feature_maps[-1])
        fused_feature_map = torch.concat(
            [downsampled_feature_map, fpn_feature_maps[index + 1]],
            dim=1,
        )
        pan_feature_maps.append(pan_block(fused_feature_map))

    return layout_mod.BaseModelOutput(last_hidden_state=pan_feature_maps)


def _bind(instance: Any, method: Any) -> None:
    instance.forward = types.MethodType(method, instance)


def make_eager_npu_compatible(model: nn.Module) -> None:
    """Apply only the rewrites required by eager PP-DocLayoutV2 on NPU.

    Ascend 310P rejects the scalar ``torch.where`` branch in anchor generation,
    the data-dependent indexed writes in the upstream reading-order input
    construction, and the broadcast advanced-indexing path used by the upstream
    reading-order mask helper. The remaining rewrites below are TorchAir
    fullgraph accommodations and must not change the eager path unnecessarily.
    """
    model.model.generate_anchors = types.MethodType(_generate_anchors, model.model)
    _bind(model.reading_order, _reading_order)


def make_compile_compatible(
    model: nn.Module,
    *,
    cogview_attention_impl: str = "stabilized",
) -> None:
    """Apply algebraically equivalent rewrites only to this layout model."""
    if cogview_attention_impl not in COGVIEW_ATTENTION_IMPL_CHOICES:
        raise ValueError(
            "Unsupported CogView attention implementation: "
            f"{cogview_attention_impl}"
        )
    layout_mod.torch_compilable_check = lambda *args, **kwargs: None
    make_eager_npu_compatible(model)
    _bind(model, _object_detection_forward)
    for module in model.modules():
        if isinstance(module, layout_mod.PPDocLayoutV2SelfAttention):
            _bind(module, _model_self_attention)
        elif isinstance(module, layout_mod.PPDocLayoutV2ReadingOrderEncoder):
            module._cal_1d_pos_emb = types.MethodType(
                _reading_order_cal_1d_pos_emb,
                module,
            )
        elif isinstance(module, layout_mod.PPDocLayoutV2ReadingOrderSelfAttention):
            module._unirec_cogview_direct_softmax = (
                cogview_attention_impl == "direct_softmax"
            )
            _bind(module, _reading_order_attention)
        elif isinstance(module, layout_mod.PPDocLayoutV2GlobalPointer):
            _bind(module, _global_pointer)
        elif isinstance(module, layout_mod.PPDocLayoutV2SinePositionEmbedding):
            _bind(module, _sine_position)
        elif isinstance(module, layout_mod.PPDocLayoutV2HybridEncoder):
            _bind(module, _hybrid_encoder)
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
        batch_size: int = 1,
        freeze_parameters: bool = False,
        cogview_attention_impl: str = "stabilized",
    ) -> None:
        if batch_size < 1:
            raise ValueError("layout batch size must be >= 1")
        make_compile_compatible(
            model,
            cogview_attention_impl=cogview_attention_impl,
        )
        self.batch_size = int(batch_size)
        self.stage = LayoutFullGraph(model).eval()
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        dtype_name = str(dtype).removeprefix("torch.")
        self.cache_dir = cache_root.expanduser().resolve() / (
            f"layout_b{self.batch_size}_800x800_{dtype_name}_src{source_hash}"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        config.mode.value = "max-autotune"
        config.experimental_config.frozen_parameter.value = bool(freeze_parameters)
        self.compiled = _cache_compile()(
            self.stage.forward,
            config=config,
            dynamic=False,
            cache_dir=str(self.cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
    def __call__(self, pixel_values: torch.Tensor):
        return self.compiled(pixel_values)
