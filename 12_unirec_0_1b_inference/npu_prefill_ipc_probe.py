#!/usr/bin/env python3
"""Measure direct spawned-process NPU IPC for a UniRec-shaped cache payload."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import Any


def _producer(
    result_queue: Any,
    ack_queue: Any,
    device: int,
    transport: str,
) -> None:
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
    d2h_s = 0.0
    packed_tensor = None
    shared_storage = None
    shared_descriptor = None
    tensor_shapes = [list(tensor.shape) for tensor in tensors]
    if transport == "host":
        d2h_started = time.perf_counter()
        tensors = [tensor.cpu() for tensor in tensors]
        d2h_s = time.perf_counter() - d2h_started
        cross_key = tensors[:6]
        cross_value = tensors[6:12]
        self_key = tensors[12:18]
        self_value = tensors[18:24]
    elif transport == "host_packed":
        d2h_started = time.perf_counter()
        packed_tensor = torch.cat([tensor.reshape(-1) for tensor in tensors]).cpu()
        d2h_s = time.perf_counter() - d2h_started
        cross_key = []
        cross_value = []
        self_key = []
        self_value = []
    elif transport == "shared_packed":
        packed_npu = torch.cat([tensor.reshape(-1) for tensor in tensors])
        shared_storage = SharedMemory(
            create=True,
            size=packed_npu.numel() * packed_npu.element_size(),
        )
        shared_array = torch.frombuffer(
            shared_storage.buf,
            dtype=packed_npu.dtype,
            count=packed_npu.numel(),
        )
        d2h_started = time.perf_counter()
        shared_array.copy_(packed_npu)
        torch.npu.synchronize()
        d2h_s = time.perf_counter() - d2h_started
        shared_descriptor = {
            "name": shared_storage.name,
            "elements": packed_npu.numel(),
        }
        cross_key = []
        cross_value = []
        self_key = []
        self_value = []
    result_queue.put(
        {
            "cross_key": cross_key,
            "cross_value": cross_value,
            "self_key": self_key,
            "self_value": self_value,
            "packed_tensor": packed_tensor,
            "shared_descriptor": shared_descriptor,
            "tensor_shapes": tensor_shapes,
            "producer_ready_s": ready_s,
            "producer_d2h_s": d2h_s,
            "transport": transport,
            "payload_bytes": sum(t.numel() * t.element_size() for t in tensors),
            "sent_at": time.perf_counter(),
        }
    )
    acknowledgement = ack_queue.get(timeout=120)
    if acknowledgement != "received":
        raise RuntimeError(f"unexpected acknowledgement: {acknowledgement!r}")
    if shared_storage is not None:
        shared_storage.close()
        resource_tracker.unregister(shared_storage._name, "shared_memory")  # noqa: SLF001


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--transport",
        choices=("npu_ipc", "host", "host_packed", "shared_packed"),
        default="npu_ipc",
    )
    args = parser.parse_args()

    import torch
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    ack_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_producer,
        args=(result_queue, ack_queue, args.device, args.transport),
    )
    process.start()
    receive_started = time.perf_counter()
    payload = result_queue.get(timeout=180)
    received_s = time.perf_counter() - receive_started
    torch.npu.set_device(args.device)
    torch.npu.synchronize()
    import_lag_s = time.perf_counter() - float(payload["sent_at"])

    shared_storage = None
    if args.transport == "shared_packed":
        descriptor = payload["shared_descriptor"]
        shared_storage = SharedMemory(name=descriptor["name"])
        shared_storage.unlink()
        packed = torch.frombuffer(
            shared_storage.buf,
            dtype=torch.float16,
            count=int(descriptor["elements"]),
        )
        tensors = []
        cursor = 0
        for shape in payload["tensor_shapes"]:
            elements = 1
            for dimension in shape:
                elements *= int(dimension)
            tensors.append(packed[cursor : cursor + elements].view(shape))
            cursor += elements
    elif args.transport == "host_packed":
        packed = payload["packed_tensor"]
        tensors = []
        cursor = 0
        for shape in payload["tensor_shapes"]:
            elements = 1
            for dimension in shape:
                elements *= int(dimension)
            tensors.append(packed[cursor : cursor + elements].view(shape))
            cursor += elements
        if cursor != packed.numel():
            raise RuntimeError(f"packed cursor mismatch: {cursor} != {packed.numel()}")
    else:
        tensors = (
            payload["cross_key"]
            + payload["cross_value"]
            + payload["self_key"]
            + payload["self_value"]
        )
    expected_device = "npu" if args.transport == "npu_ipc" else "cpu"
    if any(t.device.type != expected_device for t in tensors):
        raise RuntimeError(
            f"IPC rebuilt at least one tensor off {expected_device}"
        )
    checks = []
    for layer in range(6):
        checks.extend(
            (
                float(tensors[layer].flatten()[0].item()),
                float(tensors[6 + layer].flatten()[0].item()),
                float(tensors[12 + layer].flatten()[0].item()),
                float(tensors[18 + layer].flatten()[0].item()),
            )
        )
    expected = []
    for layer in range(6):
        value = float(layer + 1)
        expected.extend((value, -value, value + 0.25, -value - 0.25))
    if checks != expected:
        raise RuntimeError(f"IPC values differ: {checks} != {expected}")

    copy_started = time.perf_counter()
    if args.transport == "host":
        local = [tensor.to("npu:0") for tensor in tensors]
    elif args.transport in {"host_packed", "shared_packed"}:
        source_packed = (
            payload["packed_tensor"]
            if args.transport == "host_packed"
            else packed
        )
        local_packed = source_packed.to("npu:0")
        local = []
        cursor = 0
        for shape in payload["tensor_shapes"]:
            elements = 1
            for dimension in shape:
                elements *= int(dimension)
            local.append(local_packed[cursor : cursor + elements].view(shape))
            cursor += elements
    else:
        local = [torch.empty_like(t) for t in tensors]
        for target, source in zip(local, tensors):
            target.copy_(source)
    torch.npu.synchronize()
    copy_s = time.perf_counter() - copy_started
    ack_queue.put("received")
    process.join(timeout=30)
    if process.exitcode != 0:
        raise RuntimeError(f"producer exit code: {process.exitcode}")
    if shared_storage is not None:
        shared_storage.close()
    print(
        json.dumps(
            {
                "status": "pass",
                "payload_bytes": int(payload["payload_bytes"]),
                "tensor_count": len(tensors),
                "producer_ready_s": float(payload["producer_ready_s"]),
                "producer_d2h_s": float(payload["producer_d2h_s"]),
                "transport": args.transport,
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
