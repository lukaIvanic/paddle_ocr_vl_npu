"""Exact uint8 RGB staging for the production Qwen2-VL fast image processor.

Resize and patch order follow installed Qwen2VLImageProcessor._preprocess.
The 256-entry lookup pattern follows experiment 09's _uint8_normalization_table,
but values come from THIS processor's rescale_and_normalize, preserving its
fused torchvision rounding rather than assuming the NumPy recipe is equivalent.
"""
import copy
import importlib
from types import MethodType
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import SizeDict


def _compact_preprocess(self, images, do_resize, size, resample, patch_size,
                        temporal_patch_size, merge_size, return_tensors=None, **kwargs):
    if len(images)!=1 or images[0].ndim!=3 or images[0].shape[0]!=3 or images[0].dtype!=torch.uint8:
        raise ValueError('compact staging requires one uint8 RGB still image')
    if (patch_size,temporal_patch_size,merge_size)!=(14,2,2):
        raise ValueError('unsupported compact Qwen2-VL patch contract')
    image=images[0].unsqueeze(0)
    height,width=image.shape[-2:]
    if do_resize:
        smart_resize=importlib.import_module(type(self).__module__).smart_resize
        height,width=smart_resize(height,width,factor=patch_size*merge_size,
                                  min_pixels=size.shortest_edge,max_pixels=size.longest_edge)
        image=self.resize(image=image,size=SizeDict(height=height,width=width),resample=resample)
    if image.dtype!=torch.uint8:
        raise ValueError('resizer did not preserve uint8; no approximation allowed')
    return BatchFeature(data={'pixel_values':image[0].contiguous(),
        'image_grid_thw':torch.tensor([[1,height//patch_size,width//patch_size]],dtype=torch.long)},
        tensor_type=return_tensors)


class CompactVisionInputs:
    def __init__(self,processor,device):
        ip=processor.image_processor
        if type(ip).__name__ not in ('Qwen2VLImageProcessor','Qwen2VLImageProcessorFast'):
            raise ValueError(f'unsupported image processor: {type(ip)}')
        if (ip.patch_size,ip.temporal_patch_size,ip.merge_size)!=(14,2,2):
            raise ValueError('unsupported patch contract')
        ramp=torch.arange(256,dtype=torch.uint8).reshape(1,1,16,16).expand(1,3,16,16).contiguous()
        table=ip.rescale_and_normalize(ramp,ip.do_rescale,ip.rescale_factor,
                                      ip.do_normalize,ip.image_mean,ip.image_std)
        if table.dtype!=torch.float32 or table.shape!=(1,3,16,16):
            raise ValueError('unexpected normalization table contract')
        self.table=table.reshape(-1).to(device)
        self.channel_offsets=(torch.arange(3,device=device,dtype=torch.int64)*256).reshape(3,1,1)
        self.processor=copy.copy(processor)
        self.processor.image_processor=copy.deepcopy(ip)
        self.processor.image_processor._preprocess=MethodType(_compact_preprocess,self.processor.image_processor)

    def decode(self,rgb,*,dtype):
        if rgb.dtype!=torch.uint8 or rgb.ndim!=3 or rgb.shape[0]!=3:
            raise ValueError('compact NPU input must be CHW uint8')
        _,height,width=rgb.shape
        gh,gw=height//14,width//14
        index=(rgb.to(torch.int64)+self.channel_offsets).reshape(-1)
        normalized=self.table.index_select(0,index).reshape(3,height,width)
        # Exact Qwen2-VL still-image temporal duplication and merge-aware order.
        # Omit singleton batch/grid_t axes: Ascend Copy rejects rank > 8.
        patches=normalized[None].expand(2,3,height,width)
        patches=patches.reshape(2,3,gh//2,2,14,gw//2,2,14)
        patches=patches.permute(2,5,3,6,1,0,4,7)
        return patches.reshape(gh*gw,3*2*14*14).to(dtype).contiguous()
