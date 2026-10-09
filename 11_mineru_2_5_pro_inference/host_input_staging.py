"""Opt-in pinned FP32 staging, following experiment 09 serving/engine.py.

Uses the same pin_memory and non_blocking Tensor.to mechanism as
_pin_memory_or_keep / _stage_prefill_group there. Unlike that helper's optional
fallback, an explicitly requested pinned lane fails if pinning is unavailable.
No CPU FP16 conversion or pixel preprocessing replacement is introduced.
"""
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


def move_pinned_inputs(inputs,position_ids,rope_deltas,*,device,dtype):
    sources=tuple(inputs.values())+(position_ids,rope_deltas)
    if not all(t.device.type=='cpu' and t.is_pinned() for t in sources):
        raise RuntimeError('non-blocking lane received an unpinned processor output')
    pixels=inputs.get('pixel_values')
    if pixels is not None and pixels.dtype!=torch.float32:
        raise RuntimeError('C2 requires unchanged FP32 processor pixel_values')
    inputs=inputs.to(device=device,dtype=dtype,non_blocking=True)
    position_ids=position_ids.to(device=device,non_blocking=True)
    rope_deltas=rope_deltas.to(device=device,non_blocking=True)
    return inputs,position_ids,rope_deltas,sources
