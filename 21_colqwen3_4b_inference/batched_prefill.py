"""Equal-length image batches using existing FP16 PromptFA and text stages.

Separate from the B1/cache contract. No token padding, bucketing, graph compilation,
resolution changes or cross-page attention. Mixed orientations are allowed.
"""
import torch
from torch import nn

from local_modeling_colqwen3 import image_positions, rotate_half
from optimized_prefill import VisionBlock, prompt_attention
from prepared_prefill import PreparedInputs, gelu_tanh


class BatchedVisionBlock(VisionBlock):
    def forward(self, hidden, cos, sin):
        batch, length, _ = hidden.shape
        q, k, v = self.qkv(self.norm(self.norm1, hidden)).reshape(
            batch, length, 3, self.heads, -1).permute(2, 0, 1, 3, 4).unbind(0)
        q = (q.float()*cos + rotate_half(q.float())*sin).to(q.dtype)
        k = (k.float()*cos + rotate_half(k.float())*sin).to(k.dtype)
        q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
        out = prompt_attention(q, k, v, self.scale).reshape(batch, length, -1)
        hidden = hidden + self.proj(out)
        return hidden + self.fc2(gelu_tanh(self.fc1(self.norm(self.norm2, hidden))))


class BatchedVisionStage(nn.Module):
    def __init__(self, model, options):
        super().__init__()
        self.blocks = nn.ModuleList([BatchedVisionBlock(b, options) for b in model.visual.blocks])
        self.taps = model.config.vision_config.deepstack_visual_indexes

    def forward(self, hidden, cos, sin):
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        taps = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin)
            if index in self.taps:
                taps.append(hidden)
        # Existing mergers consume page-major flattened patches; every page's
        # length is divisible by merge_size**2, so no merge crosses a boundary.
        return tuple(x.flatten(0, 1).contiguous() for x in (hidden, *taps))


def prepare_batched_images(model, inputs, patch_embed):
    ids, valid = inputs['input_ids'], inputs['attention_mask']
    if ids.ndim != 2 or not ids.shape[0] or valid.shape != ids.shape:
        raise ValueError('Expected nonempty batched IDs and matching mask')
    if not bool((valid == 1).all()):
        raise ValueError('Image batch probe requires equal unpadded text lengths')
    if bool((ids == model.config.video_token_id).any()):
        raise ValueError('Video input is unsupported')
    grid, pixels = inputs['image_grid_thw'], inputs['pixel_values']
    batch = ids.shape[0]
    if grid.shape != (batch, 3) or pixels.ndim != 3 or pixels.shape[0] != batch:
        raise ValueError('Expected one still-image grid and pixel row per page')
    grids = grid.cpu().tolist()
    merge = model.config.vision_config.spatial_merge_size
    lengths = [h*w for _, h, w in grids]
    if len(set(lengths)) != 1 or any(t != 1 or h <= 0 or w <= 0 or
            h % merge or w % merge for t, h, w in grids):
        raise ValueError('Image batch probe requires equal valid still-image patch lengths')
    length = lengths[0]
    if length > pixels.shape[1] or not bool(((ids == model.config.image_token_id).sum(-1)
                                            == length // merge**2).all()):
        raise ValueError('Pixel/placeholder counts do not match grids')
    hidden = patch_embed(pixels[:, :length].contiguous().flatten(0, 1))
    absolute, cos, sin = model.visual.position_embeddings(grids)
    hidden = hidden + absolute
    vision_args = tuple(a.reshape(batch, length, -1).contiguous() for a in (hidden, cos, sin))
    return PreparedInputs(inputs, grids, image_positions(ids, valid, grids, model.config), vision_args)


def pad_page_batch(inputs, batch_size):
    """Fill unused batch slots with the last valid page; never mask out its tokens.

    Fully masked dummy sequences can violate PromptFA's row contract. Repeated
    valid inputs keep attention well defined; the caller discards their outputs.
    Tensor slots are padded after real-page preprocessing, before NPU transfer.
    """
    real = inputs['input_ids'].shape[0]
    if not 0 < real <= batch_size:
        raise ValueError('Expected 1..batch_size real pages')
    if any(v.ndim == 0 or v.shape[0] != real for v in inputs.values()):
        raise ValueError('Every processor tensor must have one leading row per page')
    if real == batch_size:
        return inputs
    return {k:torch.cat((v, v[-1:].expand(batch_size-real, *v.shape[1:])), dim=0)
            for k,v in inputs.items()}
