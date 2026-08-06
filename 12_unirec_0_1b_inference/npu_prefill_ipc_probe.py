#!/usr/bin/env python3
"""Measure direct spawned-process NPU IPC for a UniRec-shaped cache payload."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from typing import Any


def _producer(result_queue: Any, ack_queue: Any, device: int) -> None:
    import torch
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(device)
    started = time.perf_counter()
    cross_key = []
    cross_value = []
    self_key = []
    self_value = []
    for layer in range(6):
        value = float(layer + 1)
        cross_key.append(
            torch.full((1, 6, 1320, 128), value, dtype=torch.float16, device="npu:0")
        )
        cross_value.append(
            torch.full((1, 6, 1320, 128), -value, dtype=torch.float16, device="npu:0")
        )
        self_key.append(
            torch.full((1, 6, 256, 128), value + 0.25, dtype=torch.float16, device="npu:0")
        )
        self_value.append(
            torch.full((1, 6, 256, 128), -value - 0.25, dtype=torch.float16, device="npu:0")
        )
    torch.npu.synchronize()
    ready_s = time.perf_counter() - started
    tensors = cross_key + cross_value + self_key + self_value
    result_queue.put(
        {
            "cross_key": cross_key,
            "cross_value": cross_value,
            "self_key": self_key,
            "self_value": self_value,
            "producer_ready_s": ready_s,
            "payload_bytes": sum(t.numel() * t.element_size() for t in tensors),
            "sent_at": time.perf_counter(),
        }
    )
    acknowledgement = ack_queue.get(timeout=120)
    if acknowledgement != "received":
        raise RuntimeError(f"unexpected acknowledgement: {acknowledgement!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    ack_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_producer,
        args=(result_queue, ack_queue, args.device),
    )
    process.start()
    receive_started = time.perf_counter()
    payload = result_queue.get(timeout=180)
    received_s = time.perf_counter() - receive_started
    torch.npu.set_device(args.device)
    torch.npu.synchronize()
    import_lag_s = time.perf_counter() - float(payload["sent_at"])

    tensors = (
        payload["cross_key"]
        + payload["cross_value"]
        + payload["self_key"]
        + payload["self_value"]
    )
    if any(t.device.type != "npu" for t in tensors):
        raise RuntimeError("IPC rebuilt at least one tensor off-device")
    checks = []
    for layer in range(6):
        checks.extend(
            (
                float(payload["cross_key"][layer].flatten()[0].item()),
                float(payload["cross_value"][layer].flatten()[0].item()),
                float(payload["self_key"][layer].flatten()[0].item()),
                float(payload["self_value"][layer].flatten()[0].item()),
            )
        )
    expected = []
    for layer in range(6):
        value = float(layer + 1)
        expected.extend((value, -value, value + 0.25, -value - 0.25))
    if checks != expected:
        raise RuntimeError(f"IPC values differ: {checks} != {expected}")

    copy_started = time.perf_counter()
    local = [torch.empty_like(t) for t in tensors]
    for target, source in zip(local, tensors):
        target.copy_(source)
    torch.npu.synchronize()
    copy_s = time.perf_counter() - copy_started
    ack_queue.put("received")
    process.join(timeout=30)
    if process.exitcode != 0:
        raise RuntimeError(f"producer exit code: {process.exitcode}")
    print(
        json.dumps(
            {
                "status": "pass",
                "payload_bytes": int(payload["payload_bytes"]),
                "tensor_count": len(tensors),
                "producer_ready_s": float(payload["producer_ready_s"]),
                "queue_receive_wall_s": received_s,
                "post_send_receive_and_sync_s": import_lag_s,
                "device_to_device_copy_s": copy_s,
                "producer_exitcode": process.exitcode,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
