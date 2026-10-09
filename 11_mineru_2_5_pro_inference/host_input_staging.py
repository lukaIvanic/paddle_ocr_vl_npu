"""Opt-in pinned FP32 staging, following experiment 09 serving/engine.py.

Uses the same pin_memory and non_blocking Tensor.to mechanism as
_pin_memory_or_keep / _stage_prefill_group there. Unlike that helper's optional
fallback, an explicitly requested pinned lane fails if pinning is unavailable.
No CPU FP16 conversion or pixel preprocessing replacement is introduced.
"""
from dataclasses import dataclass
import time
from typing import Any

import torch


def _pin(tensor):
    if not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu':
        raise TypeError('pinned staging requires CPU processor tensors')
    result=tensor if tensor.is_pinned() else tensor.pin_memory()
    if not result.is_pinned():
        raise RuntimeError('pinned staging requested but pin_memory did not produce pinned storage')
    return result


def pin_processor_outputs(inputs,position_ids,rope_deltas):
    for name,value in list(inputs.items()):inputs[name]=_pin(value)
    return _pin(position_ids),_pin(rope_deltas)


def move_pinned_inputs(inputs,position_ids,rope_deltas,*,device,dtype,compact_codec=None):
    sources=tuple(inputs.values())+(position_ids,rope_deltas)
    if not all(t.device.type=='cpu' and t.is_pinned() for t in sources):
        raise RuntimeError('non-blocking lane received an unpinned processor output')
    pixels=inputs.get('pixel_values')
    if compact_codec is None and pixels is not None and pixels.dtype!=torch.float32:
        raise RuntimeError('C2 requires unchanged FP32 processor pixel_values')
    if compact_codec is not None:
        pixels=inputs.pop("pixel_values")
        compact=pixels.to(device=device,non_blocking=True)
    inputs=inputs.to(device=device,dtype=dtype,non_blocking=True)
    if compact_codec is not None:
        inputs["pixel_values"]=compact_codec.decode(compact,dtype=dtype)
        sources=sources+(compact,)
    position_ids=position_ids.to(device=device,non_blocking=True)
    rope_deltas=rope_deltas.to(device=device,non_blocking=True)
    return inputs,position_ids,rope_deltas,sources


@dataclass
class StagedInputs:
    inputs: Any
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    cpu_grid: torch.Tensor | None
    sources: tuple[torch.Tensor, ...]
    ready_event: Any
    submit_s: float


@torch.inference_mode()
def stage_pinned_inputs(inputs, position_ids, rope_deltas, *, stream, device, dtype, keep_grid_on_cpu, compact_codec=None):
    """Experiment 09's _stage_prefill_group stream/event pattern, per request.

    Copy the mapping, not its tensors: the prep worker/trace keeps its original
    CPU inputs, and StagedInputs retains the DMA sources until prefill finishes.
    """
    from transformers.feature_extraction_utils import BatchFeature
    import torch_npu
    started=time.perf_counter()
    staged=BatchFeature(data=dict(inputs))
    cpu_grid=staged.pop('image_grid_thw',None) if keep_grid_on_cpu else None
    with torch_npu.npu.stream(stream):
        moved,pos,delta,sources=move_pinned_inputs(staged,position_ids,rope_deltas,device=device,dtype=dtype,compact_codec=compact_codec)
        ready=stream.record_event()
    return StagedInputs(moved,pos,delta,cpu_grid,sources,ready,time.perf_counter()-started)
