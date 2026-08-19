#!/usr/bin/env python3
"""Fast eager contract probes for GLM-5.2 dense-path fused NPU operators."""

from __future__ import annotations

import json

import torch
import torch_npu


def manual_interleave_rope(
    value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
    return value * cos + rotated * sin


def tensor_diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    diff = actual.float().sub(expected.float()).abs()
    return {
        "shape": list(actual.shape),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "exact": bool(torch.equal(actual, expected)),
        "allclose_1e_3": bool(torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)),
    }


def probe_rope(device: torch.device) -> dict[str, object]:
    torch.manual_seed(52)
    query = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16, device=device)
    key = torch.randn(1, 1, 1, 64, dtype=torch.bfloat16, device=device)
    angles = torch.randn(1, 1, 1, 32, dtype=torch.float32, device=device)
    cos = angles.cos().repeat_interleave(2, dim=-1).to(torch.bfloat16)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(torch.bfloat16)
    expected_q = manual_interleave_rope(query, cos, sin)
    expected_k = manual_interleave_rope(key, cos, sin)
    variants: dict[str, object] = {}
    calls = {
        "npu_rotary_mul": lambda: (
            torch_npu.npu_rotary_mul(
                query, cos, sin, rotary_mode="interleave"
            ),
            torch_npu.npu_rotary_mul(
                key, cos, sin, rotary_mode="interleave"
            ),
        ),
        "npu_interleave_rope": lambda: (
            torch_npu.npu_interleave_rope(query, cos, sin),
            torch_npu.npu_interleave_rope(key, cos, sin),
        ),
        "npu_apply_rotary_pos_emb": lambda: torch_npu.npu_apply_rotary_pos_emb(
            query,
            key,
            cos,
            sin,
            layout="BSND",
            rotary_mode="interleave",
        ),
    }
    for name, call in calls.items():
        try:
            actual_q, actual_k = call()
            torch.npu.synchronize()
            variants[name] = {
                "status": "ok",
                "query": tensor_diff(actual_q, expected_q),
                "key": tensor_diff(actual_k, expected_k),
            }
        except Exception as exc:
            variants[name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return variants


def cache_slice(cache: torch.Tensor, position: int) -> torch.Tensor:
    if cache.ndim == 3:
        return cache[:, position : position + 1]
    if cache.ndim == 4:
        return cache[:, :, position : position + 1]
    raise ValueError(cache.shape)


def probe_kv_cache(device: torch.device) -> dict[str, object]:
    torch.manual_seed(53)
    position_value = 3
    kv = torch.randn(1, 1, 1, 576, dtype=torch.bfloat16, device=device)
    gamma = torch.randn(512, dtype=torch.bfloat16, device=device)
    angles = torch.randn(1, 1, 1, 32, dtype=torch.float32, device=device)
    cos = angles.cos().repeat_interleave(2, dim=-1).to(torch.bfloat16)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(torch.bfloat16)
    index = torch.tensor([position_value], dtype=torch.int64, device=device)
    expected_ckv = torch_npu.npu_rms_norm(kv[..., :512], gamma, 1e-6)[0]
    expected_k = manual_interleave_rope(kv[..., 512:], cos, sin)
    results: dict[str, object] = {}
    for rank in (3, 4):
        k_shape = (1, 8, 64) if rank == 3 else (1, 1, 8, 64)
        ckv_shape = (1, 8, 512) if rank == 3 else (1, 1, 8, 512)
        for version in ("v1", "v2"):
            name = f"{version}_rank{rank}_norm"
            k_cache = torch.zeros(k_shape, dtype=torch.bfloat16, device=device)
            ckv_cache = torch.zeros(ckv_shape, dtype=torch.bfloat16, device=device)
            try:
                if version == "v1":
                    outputs = torch_npu.npu_kv_rmsnorm_rope_cache(
                        kv,
                        gamma,
                        cos,
                        sin,
                        index,
                        k_cache,
                        ckv_cache,
                        epsilon=1e-6,
                        cache_mode="Norm",
                        is_output_kv=True,
                    )
                else:
                    outputs = torch_npu.npu_kv_rmsnorm_rope_cache_v2(
                        kv,
                        gamma,
                        cos,
                        sin,
                        index,
                        k_cache,
                        ckv_cache,
                        epsilon=1e-6,
                        cache_mode="Norm",
                        is_output_kv=True,
                    )
                torch.npu.synchronize()
                actual_k = cache_slice(k_cache, position_value).reshape_as(expected_k)
                actual_ckv = cache_slice(ckv_cache, position_value).reshape_as(expected_ckv)
                results[name] = {
                    "status": "ok",
                    "output_shapes": [list(value.shape) for value in outputs],
                    "key_cache": tensor_diff(actual_k, expected_k),
                    "compressed_cache": tensor_diff(actual_ckv, expected_ckv),
                }
            except Exception as exc:
                results[name] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
    return results


def main() -> None:
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    result = {
        "chip": torch.npu.get_device_name(0),
        "rope": probe_rope(device),
        "kv_cache": probe_kv_cache(device),
    }
    print("GLM52_FUSED_OP_PROBE " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
