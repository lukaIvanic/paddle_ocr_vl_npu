#!/usr/bin/env python3
"""Benchmark direct CPU cross-K/V admission policies without loading UniRec."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


LAYERS = 6
HEADS = 6
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--self-cache-length", type=int, default=1024)
    parser.add_argument("--cross-cache-length", type=int, default=512)
    parser.add_argument("--warmup-admissions", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help=(
            "Optional unirec_cross_kv_v1 artifact directory. When set, replay "
            "its real, unique cross-K/V rows in addition to the synthetic "
            "length-distribution policies."
        ),
    )
    parser.add_argument(
        "--synthetic-policies",
        default="full_reset,no_self_reset,kv_reuse,packed_copy,masked_reuse",
        help="Comma-separated synthetic policies, or 'none'.",
    )
    parser.add_argument(
        "--artifact-policies",
        default=(
            "mask_fill_zero,mask_template,direct_pageable,"
            "direct_pageable_template,pinned_staging"
        ),
        help="Comma-separated artifact policies, or 'none'.",
    )
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device.startswith("npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def load_lengths(path: Path, cross_cache_length: int) -> list[int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows.sort(key=lambda row: int(row["admission_index"]))
    lengths = [int(row["text_prefill_real_source_tokens"]) for row in rows]
    if not lengths:
        raise ValueError("trace contains no admissions")
    if max(lengths) > cross_cache_length:
        raise ValueError(
            f"trace source length {max(lengths)} exceeds {cross_cache_length}"
        )
    return lengths


@dataclass(frozen=True)
class ArtifactRow:
    source_len: int
    array: np.ndarray


def load_artifact_rows(
    artifact_dir: Path,
    cross_cache_length: int,
) -> tuple[list[ArtifactRow], np.memmap, int]:
    metadata_path = artifact_dir / "crops.jsonl"
    rows = [
        json.loads(line)
        for line in metadata_path.read_text().splitlines()
        if line
    ]
    if not rows:
        raise ValueError(f"artifact contains no crops: {metadata_path}")
    data_names = {str(row["cross_kv"]["file"]) for row in rows}
    if len(data_names) != 1:
        raise ValueError(f"artifact must use one data file, got {data_names}")
    data_path = artifact_dir / next(iter(data_names))
    mapped = np.memmap(data_path, dtype=np.uint8, mode="r")
    artifact_rows = []
    payload_bytes = 0
    for index, row in enumerate(rows):
        cross = row["cross_kv"]
        shape = tuple(int(value) for value in cross["shape"])
        dtype = np.dtype(cross["dtype"])
        offset = int(cross["offset"])
        nbytes = int(cross["nbytes"])
        source_len = int(cross["source_length"])
        if dtype != np.dtype(np.float16) or len(shape) != 5:
            raise ValueError(
                f"unexpected artifact row {index}: shape={shape} dtype={dtype}"
            )
        if shape != (2 * LAYERS, 1, HEADS, source_len, HEAD_DIM):
            raise ValueError(
                f"unexpected artifact row {index} shape: {shape}"
            )
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes != expected_nbytes:
            raise ValueError(
                f"artifact row {index} byte mismatch: {nbytes} != {expected_nbytes}"
            )
        if source_len > cross_cache_length:
            raise ValueError(
                f"artifact row {index} exceeds cross cache: "
                f"{source_len} > {cross_cache_length}"
            )
        if offset < 0 or offset + nbytes > mapped.nbytes:
            raise ValueError(
                f"artifact row {index} exceeds data file: "
                f"offset={offset} nbytes={nbytes} file={mapped.nbytes}"
            )
        array = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=mapped,
            offset=offset,
        )
        artifact_rows.append(ArtifactRow(source_len=source_len, array=array))
        payload_bytes += nbytes
    return artifact_rows, mapped, payload_bytes


def parse_policy_list(value: str) -> list[str]:
    if value.strip().lower() == "none":
        return []
    policies = [item.strip() for item in value.split(",") if item.strip()]
    if not policies:
        raise ValueError("policy list must contain at least one policy or 'none'")
    return policies


def allocate_arena(
    *,
    device: str,
    batch_size: int,
    self_cache_length: int,
    cross_cache_length: int,
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    torch.Tensor,
    torch.Tensor,
]:
    self_shape = (batch_size, HEADS, self_cache_length, HEAD_DIM)
    cross_shape = (batch_size, HEADS, cross_cache_length, HEAD_DIM)
    negative_inf = torch.finfo(torch.float32).min
    with torch.inference_mode():
        self_keys = tuple(
            torch.zeros(self_shape, dtype=torch.float16, device=device)
            for _ in range(LAYERS)
        )
        self_values = tuple(
            torch.zeros(self_shape, dtype=torch.float16, device=device)
            for _ in range(LAYERS)
        )
        packed_cross = torch.zeros(
            (2 * LAYERS, *cross_shape),
            dtype=torch.float16,
            device=device,
        )
        cross_keys = tuple(packed_cross[layer] for layer in range(LAYERS))
        cross_values = tuple(
            packed_cross[LAYERS + layer] for layer in range(LAYERS)
        )
        cross_mask = torch.full(
            (batch_size, 1, 1, cross_cache_length),
            negative_inf,
            dtype=torch.float32,
            device=device,
        )
    return (
        self_keys,
        self_values,
        cross_keys,
        cross_values,
        cross_mask,
        packed_cross,
    )


def run_policy(
    policy: str,
    *,
    lengths: list[int],
    host_by_length: dict[int, np.ndarray],
    arena: tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
    ],
    mask_templates: dict[int, torch.Tensor],
    device: str,
    batch_size: int,
) -> float:
    (
        self_keys,
        self_values,
        cross_keys,
        cross_values,
        cross_mask,
        packed_cross,
    ) = arena
    negative_inf = torch.finfo(cross_mask.dtype).min
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for admission, source_len in enumerate(lengths):
            slot = admission % batch_size
            packed = host_by_length[source_len]
            if policy in {
                "full_reset",
                "no_self_reset",
                "kv_reuse",
                "packed_copy",
            }:
                cross_mask[slot : slot + 1].fill_(negative_inf)
                cross_mask[slot : slot + 1, :, :, :source_len].zero_()
            elif policy == "masked_reuse":
                cross_mask[slot : slot + 1].copy_(mask_templates[source_len])
            else:
                raise ValueError(f"unknown policy: {policy}")
            if policy == "packed_copy":
                packed_cross[
                    :, slot : slot + 1, :, :source_len, :
                ].copy_(torch.from_numpy(packed))
                continue
            for layer in range(LAYERS):
                if policy == "full_reset":
                    self_keys[layer][slot : slot + 1].zero_()
                    self_values[layer][slot : slot + 1].zero_()
                if policy in {"full_reset", "no_self_reset"}:
                    cross_keys[layer][
                        slot : slot + 1, :, source_len:, :
                    ].zero_()
                    cross_values[layer][
                        slot : slot + 1, :, source_len:, :
                    ].zero_()
                cross_keys[layer][slot : slot + 1, :, :source_len, :].copy_(
                    torch.from_numpy(packed[layer])
                )
                cross_values[layer][slot : slot + 1, :, :source_len, :].copy_(
                    torch.from_numpy(packed[LAYERS + layer])
                )
    synchronize(device)
    return time.perf_counter() - started


def run_artifact_policy(
    policy: str,
    *,
    rows: list[ArtifactRow],
    arena: tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
    ],
    mask_templates: dict[int, torch.Tensor],
    device: str,
    batch_size: int,
    pinned_flat: torch.Tensor | None,
) -> float:
    *_, cross_mask, packed_cross = arena
    negative_inf = torch.finfo(cross_mask.dtype).min
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for admission, row in enumerate(rows):
            slot = admission % batch_size
            source_len = row.source_len
            if policy in {
                "mask_fill_zero",
                "direct_pageable",
                "pinned_staging",
            }:
                cross_mask[slot : slot + 1].fill_(negative_inf)
                cross_mask[slot : slot + 1, :, :, :source_len].zero_()
            elif policy in {"mask_template", "direct_pageable_template"}:
                cross_mask[slot : slot + 1].copy_(mask_templates[source_len])
            else:
                raise ValueError(f"unknown artifact policy: {policy}")

            if policy in {"mask_fill_zero", "mask_template"}:
                continue
            source = torch.from_numpy(row.array)
            if policy == "pinned_staging":
                if pinned_flat is None:
                    raise RuntimeError("pinned_staging requires a pinned buffer")
                elements = source.numel()
                staged = pinned_flat[:elements].view(source.shape)
                staged.copy_(source)
                source = staged
            packed_cross[
                :, slot : slot + 1, :, :source_len, :
            ].copy_(source)
    synchronize(device)
    return time.perf_counter() - started


def main() -> None:
    args = parse_args()
    if args.rounds < 1 or args.warmup_admissions < 0:
        raise ValueError("rounds must be positive and warmup-admissions nonnegative")
    synthetic_policies = parse_policy_list(args.synthetic_policies)
    artifact_policies = parse_policy_list(args.artifact_policies)
    if args.artifact_dir is None and artifact_policies:
        artifact_policies = []
    lengths = load_lengths(args.trace, args.cross_cache_length)
    unique_lengths = sorted(set(lengths))
    host_by_length = {
        length: np.zeros(
            (2 * LAYERS, 1, HEADS, length, HEAD_DIM),
            dtype=np.float16,
        )
        for length in unique_lengths
    }
    negative_inf = torch.finfo(torch.float32).min
    mask_templates = {}
    with torch.inference_mode():
        for length in unique_lengths:
            template = torch.full(
                (1, 1, 1, args.cross_cache_length),
                negative_inf,
                dtype=torch.float32,
                device=args.device,
            )
            template[..., :length].zero_()
            mask_templates[length] = template
    arena = allocate_arena(
        device=args.device,
        batch_size=args.batch_size,
        self_cache_length=args.self_cache_length,
        cross_cache_length=args.cross_cache_length,
    )
    warmup_lengths = lengths[: args.warmup_admissions]
    results = {}
    known_synthetic_policies = {
        "full_reset",
        "no_self_reset",
        "kv_reuse",
        "packed_copy",
        "masked_reuse",
    }
    unknown_synthetic = set(synthetic_policies) - known_synthetic_policies
    if unknown_synthetic:
        raise ValueError(f"unknown synthetic policies: {sorted(unknown_synthetic)}")
    for policy in synthetic_policies:
        if warmup_lengths:
            run_policy(
                policy,
                lengths=warmup_lengths,
                host_by_length=host_by_length,
                arena=arena,
                mask_templates=mask_templates,
                device=args.device,
                batch_size=args.batch_size,
            )
        samples = [
            run_policy(
                policy,
                lengths=lengths,
                host_by_length=host_by_length,
                arena=arena,
                mask_templates=mask_templates,
                device=args.device,
                batch_size=args.batch_size,
            )
            for _ in range(args.rounds)
        ]
        results[policy] = {
            "samples_s": samples,
            "median_s": statistics.median(samples),
            "admissions_per_s": len(lengths) / statistics.median(samples),
        }
    if "full_reset" in results:
        baseline = results["full_reset"]["median_s"]
        for value in results.values():
            value["speedup_vs_full_reset"] = baseline / value["median_s"]

    artifact_summary = None
    if args.artifact_dir is not None:
        artifact_rows, mapped, artifact_payload_bytes = load_artifact_rows(
            args.artifact_dir,
            args.cross_cache_length,
        )
        artifact_lengths = [row.source_len for row in artifact_rows]
        missing_templates = set(artifact_lengths) - set(mask_templates)
        with torch.inference_mode():
            for length in sorted(missing_templates):
                template = torch.full(
                    (1, 1, 1, args.cross_cache_length),
                    negative_inf,
                    dtype=torch.float32,
                    device=args.device,
                )
                template[..., :length].zero_()
                mask_templates[length] = template
        max_elements = max(row.array.size for row in artifact_rows)
        pinned_flat = torch.empty(
            max_elements,
            dtype=torch.float16,
            pin_memory=True,
        )
        known_artifact_policies = {
            "mask_fill_zero",
            "mask_template",
            "direct_pageable",
            "direct_pageable_template",
            "pinned_staging",
        }
        unknown_artifact = set(artifact_policies) - known_artifact_policies
        if unknown_artifact:
            raise ValueError(
                f"unknown artifact policies: {sorted(unknown_artifact)}"
            )
        artifact_results = {}
        for policy in artifact_policies:
            warmup_rows = artifact_rows[: args.warmup_admissions]
            if warmup_rows:
                run_artifact_policy(
                    policy,
                    rows=warmup_rows,
                    arena=arena,
                    mask_templates=mask_templates,
                    device=args.device,
                    batch_size=args.batch_size,
                    pinned_flat=pinned_flat,
                )
            samples = [
                run_artifact_policy(
                    policy,
                    rows=artifact_rows,
                    arena=arena,
                    mask_templates=mask_templates,
                    device=args.device,
                    batch_size=args.batch_size,
                    pinned_flat=pinned_flat,
                )
                for _ in range(args.rounds)
            ]
            median_s = statistics.median(samples)
            copies_payload = policy not in {"mask_fill_zero", "mask_template"}
            artifact_results[policy] = {
                "samples_s": samples,
                "median_s": median_s,
                "admissions_per_s": len(artifact_rows) / median_s,
                "payload_gib_per_s": (
                    artifact_payload_bytes / (1 << 30) / median_s
                    if copies_payload
                    else None
                ),
            }
        artifact_summary = {
            "directory": str(args.artifact_dir),
            "admissions": len(artifact_rows),
            "payload_bytes": artifact_payload_bytes,
            "mapped_file_bytes": int(mapped.nbytes),
            "source_length_histogram": dict(
                sorted(Counter(artifact_lengths).items())
            ),
            "pinned_staging_bytes": int(
                pinned_flat.numel() * pinned_flat.element_size()
            ),
            "pinned_staging_is_pinned": bool(pinned_flat.is_pinned()),
            "results": artifact_results,
        }
    summary = {
        "device": args.device,
        "admissions": len(lengths),
        "batch_size": args.batch_size,
        "self_cache_length": args.self_cache_length,
        "cross_cache_length": args.cross_cache_length,
        "source_length_histogram": dict(sorted(Counter(lengths).items())),
        "host_payload_bytes": sum(host_by_length[length].nbytes for length in lengths),
        "results": results,
        "artifact": artifact_summary,
    }
    print("UNIREC_ARENA_ADMISSION_BENCH " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
