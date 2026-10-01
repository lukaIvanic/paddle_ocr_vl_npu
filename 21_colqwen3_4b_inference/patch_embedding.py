"""Patch Conv3D -> Linear, outside transformer graphs and their cache identity."""
from math import prod
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from prepared_prefill import prepare_inputs


class LinearPatchEmbed(nn.Module):
    def __init__(self, patch_embed):
        super().__init__()
        conv = patch_embed.proj
        if (not isinstance(conv, nn.Conv3d) or conv.stride != conv.kernel_size
                or conv.padding != (0,0,0) or conv.dilation != (1,1,1)
                or conv.groups != 1 or conv.padding_mode != 'zeros'):
            raise ValueError('Requires an ungrouped whole-patch Conv3D with no padding/dilation')
        self.in_features = conv.in_channels * prod(conv.kernel_size)
        self.out_features = conv.out_channels
        # Same C,T,H,W flatten order as the reference. Prepare once, not per call.
        self.weight = nn.Parameter(conv.weight.detach().reshape(
            self.out_features,self.in_features).contiguous(), requires_grad=False)
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach(),requires_grad=False)

    def forward(self, patches):
        if patches.ndim != 2 or patches.shape[-1] != self.in_features:
            raise ValueError('Expected pre-extracted [patches, C*T*H*W] input')
        return F.linear(patches.to(self.weight.dtype),self.weight,self.bias)


def prepare_linear_patch_inputs(model, inputs, patch_embed):
    # A read-only view substitutes only the patch projector. No mutation of the
    # reference module tree, and all validation/position preparation is reused.
    view = SimpleNamespace(config=model.config, visual=SimpleNamespace(
        patch_embed=patch_embed, position_embeddings=model.visual.position_embeddings))
    return prepare_inputs(view,inputs)
