#!/usr/bin/env python3

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch_npu.dynamo import torchair
from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig


class CollectiveProbe(nn.Module):
    def __init__(self, world_size: int):
        super().__init__()
        self.world_size = int(world_size)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reduced = value + 1
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        local_pair = torch.stack((reduced, value), dim=-1).reshape(-1, 2)
        gathered = torch.empty(
            (self.world_size * local_pair.shape[0], 2),
            device=value.device,
            dtype=value.dtype,
        )
        dist.all_gather_into_tensor(gathered, local_pair.contiguous())
        return reduced, gathered


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"This probe requires WORLD_SIZE=2, got {world_size}")

    torch.npu.set_device(local_rank)
    torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group("hccl")
    torchair.patch_for_hcom()
    device = torch.device(f"npu:{local_rank}")

    eager = CollectiveProbe(world_size).to(device)
    backend = torchair.get_npu_backend(compiler_config=CompilerConfig())
    compiled = torch.compile(eager, backend=backend, dynamic=False, fullgraph=True)
    value = torch.tensor([float(rank + 1)], device=device)
    reduced, gathered = compiled(value)
    torch.npu.synchronize()
    expected_reduced = torch.tensor([5.0], device=device)
    expected_gathered = torch.tensor(
        [[5.0, 1.0], [5.0, 2.0]], device=device
    )
    passed = bool(
        torch.equal(reduced, expected_reduced)
        and torch.equal(gathered, expected_gathered)
    )
    print(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "reduced": reduced.cpu().tolist(),
                "gathered": gathered.cpu().tolist(),
                "passed": passed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not passed:
        raise RuntimeError("TorchAir HCCL collective probe failed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
