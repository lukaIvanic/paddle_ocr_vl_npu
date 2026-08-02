#!/usr/bin/env python3
"""Localize pre-transformer vision drift and correlate it with generation.

``exact`` captures full tensors for the clean Phase-39 table-token-zero case.
``corpus`` captures deterministic samples for all 106 crops in the fixed
seven-page Phase-39 corpus and, when given a reference bundle, correlates each
pre-transformer boundary's numerical error with generation divergence.

Both modes use the owned layout frontend and ``ContinuousRecognizer`` CPU
preparation.  They stop before transformer layer 0; no vision-transformer,
projector, text-prefill, or decode execution is launched.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Sequence

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.serving.engine import ContinuousRecognizer
from pipeline.layout_frontend import OwnedLayoutFrontend
from table_token0_replay import tensor_comparison, tensor_summary


DEFAULT_CASES = EXPERIMENT_ROOT / "accuracy_lab/cases.json"
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")
DEFAULT_LAYOUT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")
DEFAULT_RECOGNIZER_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_CACHE_ROOT = REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair"
DEFAULT_VISION_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair"
)
DEFAULT_TEXT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_torchair"
)
DEFAULT_PACKED_TEXT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_packed_torchair"
)
DEFAULT_REFERENCE_GENERATION = (
    REPO_ROOT / "tmp/09_persistent_page_engine/910b_accuracy_lab_7p_8e19fdc/output"
)

VISION_BUCKETS = (128, 256, 384, 512, 640, 768, 1408, 1920, 2944, 4992)
TEXT_BUCKETS = (
    32,
    64,
    96,
    128,
    160,
    176,
    192,
    208,
    224,
    256,
    320,
    384,
    448,
    576,
    640,
    768,
    896,
    1024,
    1152,
    1280,
    1312,
)
TEXT_PACK_BUCKETS = (128, 256, 512, 1024)
OUTPUT_BOUNDARIES = (
    "patch_embeddings",
    "position_embeddings",
    "summed_embeddings",
    "rotary_base_angles",
    "rotary_selected_angles",
    "rope_cos",
    "rope_sin",
)
CONTROL_TENSORS = (
    "conv_input",
    "patch_embedding_weight",
    "patch_embedding_bias",
    "position_embedding_weight",
    "rotary_inv_freq",
)
EXPECTED_EXACT_CASE = {
    "case_id": "table_token0_11_3",
    "crop_sha256": (
        "62359561cc5557d9a1972c23f0915ab37c77feb7b6c1aac4f86d320bfde1af2f"
    ),
    "prepared_inputs_sha256": (
        "ad925263b4f156d3d11b3367f7fc2b09c77abdbac53cb5ec9bf37007e38c291e"
    ),
    "input_tokens": 1021,
    "projected_image_tokens": 1008,
    "real_vision_tokens": 4032,
    "physical_vision_tokens": 4992,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("exact", "corpus"))
    parser.add_argument("--case-id", default="table_token0_11_3")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--layout-model", type=Path, default=DEFAULT_LAYOUT_MODEL)
    parser.add_argument(
        "--recognizer-model", type=Path, default=DEFAULT_RECOGNIZER_MODEL
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, default=None)
    parser.add_argument(
        "--generation-output",
        type=Path,
        default=None,
        help=(
            "Phase-39-style output containing recognition_trace.jsonl. "
            "Required in corpus mode."
        ),
    )
    parser.add_argument(
        "--sample-elements",
        type=int,
        default=8192,
        help="Deterministic flattened samples retained per corpus boundary.",
    )
    parser.add_argument(
        "--torchair-cache-dir", type=Path, default=DEFAULT_CACHE_ROOT
    )
    parser.add_argument(
        "--vision-torchair-cache-dir",
        type=Path,
        default=DEFAULT_VISION_CACHE_ROOT,
    )
    parser.add_argument(
        "--text-torchair-cache-dir",
        type=Path,
        default=DEFAULT_TEXT_CACHE_ROOT,
    )
    parser.add_argument(
        "--text-packed-cache-dir",
        type=Path,
        default=DEFAULT_PACKED_TEXT_CACHE_ROOT,
    )
    args = parser.parse_args(argv)
    if args.sample_elements <= 0:
        parser.error("--sample-elements must be positive")
    if args.mode == "corpus" and args.generation_output is None:
        parser.error("--generation-output is required in corpus mode")
    return args


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("accuracy manifest must use schema_version=1")
    return payload


def selected_case(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [row for row in manifest["cases"] if row["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"expected one case {case_id!r}, got {len(matches)}")
    return dict(matches[0])


def build_recognizer(args: argparse.Namespace) -> ContinuousRecognizer:
    return ContinuousRecognizer(
        model=str(args.recognizer_model.expanduser().resolve()),
        dtype="fp16",
        decode_backend="torchair",
        decode_optimization="combined_apply_static_actual",
        batch_size=32,
        cache_length=4096,
        max_new_tokens=4096,
        torchair_cache_dir=args.torchair_cache_dir.expanduser().resolve(),
        vision_backend="torchair",
        vision_attention="prompt_flash_attention",
        vision_buckets=VISION_BUCKETS,
        vision_torchair_cache_dir=args.vision_torchair_cache_dir.expanduser().resolve(),
        vision_padding="bucket",
        vision_promptfa_align_128=True,
        vision_packing="greedy",
        vision_pack_target=1920,
        vision_router_lookahead=32,
        text_backend="torchair",
        text_buckets=TEXT_BUCKETS,
        text_torchair_cache_dir=args.text_torchair_cache_dir.expanduser().resolve(),
        text_padding="auto",
        text_packing="production_group",
        text_pack_buckets=TEXT_PACK_BUCKETS,
        text_pack_max_members=32,
        text_packed_cache_dir=args.text_packed_cache_dir.expanduser().resolve(),
        preprocessor_min_pixels=28_224,
        recognition_input_fingerprints=True,
    )


def load_pages(
    frontend: OwnedLayoutFrontend,
    manifest: dict[str, Any],
    images_dir: Path,
) -> list[dict[str, Any]]:
    page_corpus = manifest["page_corpus"]
    indices = [int(value) for value in page_corpus["source_page_indices"]]
    names = [str(value) for value in page_corpus["source_images"]]
    if len(indices) != len(names):
        raise ValueError("accuracy page indices and image names do not align")
    pages: list[dict[str, Any]] = []
    for position, (source_index, image_name) in enumerate(zip(indices, names)):
        print(
            f"VISION_EMBED step=layout_page position={position + 1}/{len(names)} "
            f"source_index={source_index} image={image_name}",
            flush=True,
        )
        prepared = frontend.prepare_page(
            images_dir / image_name,
            source_index,
            min_pixels=28_224,
            max_pixels=1_003_520,
        )
        pages.append(
            {
                "source_page_index": source_index,
                "source_image_name": image_name,
                "prepared": prepared,
            }
        )
    return pages


def stable_requests(pages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        prepared = page["prepared"]
        for request, block_index in zip(
            prepared.requests,
            prepared.request_block_indices,
        ):
            rows.append(
                {
                    "source_page_index": page["source_page_index"],
                    "source_image_name": page["source_image_name"],
                    "block_index": int(block_index),
                    "request": request,
                }
            )
    return rows


def capture_pretransformer(
    recognizer: ContinuousRecognizer,
    request: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], Any]:
    embedding = recognizer.model.visual.vision_model.embeddings
    rotary = recognizer.model.visual.vision_model.encoder.rotary_pos_emb
    held: dict[str, torch.Tensor] = {}

    def hold(name: str, tensor: torch.Tensor) -> None:
        if name in held:
            raise RuntimeError(f"captured {name!r} more than once")
        held[name] = tensor.detach()

    def patch_pre_hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        hold("conv_input", inputs[0])

    def patch_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        patch = output.flatten(-2).squeeze(-1)
        hold("patch_embeddings", patch)

    def embedding_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        hold("summed_embeddings", output)

    def rotary_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        hold("rotary_base_angles", output)

    handles = [
        embedding.patch_embedding.register_forward_pre_hook(patch_pre_hook),
        embedding.patch_embedding.register_forward_hook(patch_hook),
        embedding.register_forward_hook(embedding_hook),
        rotary.register_forward_hook(rotary_hook),
    ]
    original_interpolate = embedding.interpolate_pos_encoding

    def interpolate(
        _embedding: Any,
        embeddings: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        output = original_interpolate(embeddings, height, width)
        hold("position_embeddings", output.squeeze(0))
        return output

    embedding.interpolate_pos_encoding = MethodType(interpolate, embedding)
    try:
        prepared = recognizer._prepare_cpu(request, time.perf_counter())
        pixels = prepared.pixel_values.to(
            recognizer.device,
            non_blocking=True,
        )
        hidden = embedding(
            pixels.unsqueeze(0),
            image_grid_thw=prepared.image_grid_thw,
        )
        route = recognizer.vision_prefill.route(int(hidden.shape[0]))
        prepared_vision = recognizer.vision_prefill.prepare(
            hidden,
            prepared.image_grid_thw,
            route=route,
        )
        grid = prepared.image_grid_thw.detach().cpu().reshape(-1, 3)
        if tuple(grid.shape) != (1, 3):
            raise RuntimeError(f"expected one image grid, got {tuple(grid.shape)}")
        t, h, w = (int(value) for value in grid[0].tolist())
        image_pids = torch.arange(
            int(hidden.shape[0]),
            device=hidden.device,
            dtype=torch.int64,
        ) % int(h * w)
        pids = torch.stack((image_pids // int(w), image_pids % int(w)), dim=-1)
        selected_angles = held["rotary_base_angles"][pids].flatten(1).repeat(1, 2)
        hold("rotary_selected_angles", selected_angles)
        hold("rope_cos", prepared_vision.rope_cos[0, : hidden.shape[0]])
        hold("rope_sin", prepared_vision.rope_sin[0, : hidden.shape[0]])
        held["patch_embedding_weight"] = embedding.patch_embedding.weight.detach()
        held["patch_embedding_bias"] = embedding.patch_embedding.bias.detach()
        held["position_embedding_weight"] = embedding.position_embedding.weight.detach()
        held["rotary_inv_freq"] = rotary.inv_freq.detach()
        torch.npu.synchronize()
    finally:
        embedding.interpolate_pos_encoding = original_interpolate
        for handle in handles:
            handle.remove()

    missing = [
        name for name in (*CONTROL_TENSORS, *OUTPUT_BOUNDARIES) if name not in held
    ]
    if missing:
        raise RuntimeError(f"missing pre-transformer captures: {missing}")
    patch = held["patch_embeddings"]
    position = held["position_embeddings"]
    summed = held["summed_embeddings"]
    if patch.shape != position.shape or patch.shape != summed.shape:
        raise RuntimeError(
            "patch/position/sum shape mismatch: "
            f"patch={tuple(patch.shape)} position={tuple(position.shape)} "
            f"sum={tuple(summed.shape)}"
        )
    reconstruction = patch + position
    reconstruction_exact = bool(torch.equal(reconstruction, summed))
    metadata = {
        "route": route,
        "grid_thw": [t, h, w],
        "real_vision_tokens": int(hidden.shape[0]),
        "physical_vision_tokens": int(prepared_vision.physical_seq_len),
        "sum_reconstruction_exact": reconstruction_exact,
        "sum_reconstruction_max_abs": float(
            (reconstruction.float() - summed.float()).abs().max().item()
        ),
    }
    return held, metadata, prepared


def to_cpu(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in tensors.items()
    }


def deterministic_sample(tensor: torch.Tensor, limit: int) -> torch.Tensor:
    flat = tensor.detach().to(device="cpu").contiguous().reshape(-1)
    if flat.numel() <= limit:
        return flat
    if limit == 1:
        return flat[:1]
    # Integer arithmetic gives the same evenly-spaced indices on every host.
    positions = torch.arange(limit, dtype=torch.int64)
    indices = torch.div(
        positions * (flat.numel() - 1),
        limit - 1,
        rounding_mode="floor",
    )
    return flat.index_select(0, indices).contiguous()


def load_bundle(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(
            path.expanduser().resolve(),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(path.expanduser().resolve(), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid vision embedding bundle: {path}")
    return payload


def input_hashes(prepared: Any) -> dict[str, Any]:
    fingerprints = prepared.input_fingerprints
    return {
        "crop_sha256": (fingerprints.get("crop") or {}).get("sha256"),
        "prepared_inputs_sha256": fingerprints.get("prepared_inputs_sha256"),
        "tensor_sha256": {
            name: row.get("sha256")
            for name, row in (fingerprints.get("tensors") or {}).items()
        },
    }


def generation_trace(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    root = root.expanduser().resolve()
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    images = [str(value) for value in summary["images"]]
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for line in (root / "recognition_trace.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = str(
            row.get("source_image_name")
            or images[int(row["page_input_index"])]
        )
        block = int(row["block_index"])
        key = (image, block)
        if key in rows:
            raise ValueError(f"duplicate generation trace key: {key}")
        rows[key] = {
            "source_image_name": image,
            "block_index": block,
            "label": row.get("label"),
            "prompt": row.get("prompt"),
            "token_ids": [int(value) for value in row.get("token_ids", ())],
            "text": str(row.get("text", "")),
            "stop_reason": row.get("stop_reason"),
        }
    return rows


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def average(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else None


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0
        for cursor in range(position, end):
            result[order[cursor]] = rank
        position = end
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(ranks(left), ranks(right))


def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    score_ranks = ranks(scores)
    positive_rank_sum = sum(
        rank for rank, label in zip(score_ranks, labels) if label
    )
    # Ranks are zero-based; this is the Mann-Whitney U formulation.
    positive_rank_sum += positives
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def compare_generation(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    left = reference["token_ids"]
    right = candidate["token_ids"]
    first = first_difference(left, right)
    ratio = difflib.SequenceMatcher(
        None,
        left,
        right,
        autojunk=False,
    ).ratio()
    return {
        "token_exact": left == right,
        "first_token_divergence": first == 0,
        "first_difference_index": first,
        "reference_tokens": len(left),
        "candidate_tokens": len(right),
        "token_sequence_ratio": ratio,
        "generation_distance": 1.0 - ratio,
        "reference_stop_reason": reference.get("stop_reason"),
        "candidate_stop_reason": candidate.get("stop_reason"),
    }


def exact_mode(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    pages: list[dict[str, Any]],
    recognizer: ContinuousRecognizer,
    output_dir: Path,
) -> None:
    case = selected_case(manifest, args.case_id)
    requests = stable_requests(pages)
    matches = [
        row
        for row in requests
        if row["source_image_name"] == case["source_image_name"]
        and row["block_index"] == int(case["block_index"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact case request, got {len(matches)}")
    captured_npu, metadata, prepared = capture_pretransformer(
        recognizer,
        matches[0]["request"],
    )
    captured = to_cpu(captured_npu)
    hashes = input_hashes(prepared)
    expected = EXPECTED_EXACT_CASE
    contract = {
        "case_id": args.case_id == expected["case_id"],
        "crop_sha256": hashes["crop_sha256"] == expected["crop_sha256"],
        "prepared_inputs_sha256": (
            hashes["prepared_inputs_sha256"]
            == expected["prepared_inputs_sha256"]
        ),
        "input_tokens": int(prepared.input_ids.shape[1]) == expected["input_tokens"],
        "projected_image_tokens": (
            int(prepared.image_token_count) == expected["projected_image_tokens"]
        ),
        "real_vision_tokens": (
            metadata["real_vision_tokens"] == expected["real_vision_tokens"]
        ),
        "physical_vision_tokens": (
            metadata["physical_vision_tokens"]
            == expected["physical_vision_tokens"]
        ),
        "sum_reconstruction_exact": metadata["sum_reconstruction_exact"],
    }
    failed = [name for name, passed in contract.items() if not passed]
    if failed:
        raise RuntimeError(f"exact vision embedding contract failed: {failed}")

    reference = load_bundle(args.reference_bundle) if args.reference_bundle else None
    comparisons = None
    if reference is not None:
        if reference.get("mode") != "exact" or reference.get("case_id") != args.case_id:
            raise ValueError("reference is not the matching exact embedding bundle")
        comparisons = {
            name: tensor_comparison(captured[name], reference["tensors"][name])
            for name in (*CONTROL_TENSORS, *OUTPUT_BOUNDARIES)
        }

    bundle = {
        "schema_version": 1,
        "kind": "experiment09_pretransformer_vision_embedding_debug",
        "mode": "exact",
        "case_id": args.case_id,
        "case": case,
        "contract": contract,
        "input_hashes": hashes,
        "metadata": metadata,
        "tensors": captured,
    }
    bundle_path = output_dir / "tensor_bundle.pt"
    torch.save(bundle, bundle_path)
    report = {
        "schema_version": 1,
        "kind": bundle["kind"],
        "mode": "exact",
        "case_id": args.case_id,
        "contract": contract,
        "input_hashes": hashes,
        "metadata": metadata,
        "tensor_summaries": {
            name: tensor_summary(captured[name])
            for name in (*CONTROL_TENSORS, *OUTPUT_BOUNDARIES)
        },
        "reference_bundle": str(args.reference_bundle) if args.reference_bundle else None,
        "comparisons": comparisons,
        "tensor_bundle": str(bundle_path),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def corpus_mode(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    pages: list[dict[str, Any]],
    recognizer: ContinuousRecognizer,
    output_dir: Path,
) -> None:
    generations = generation_trace(args.generation_output)
    requests = stable_requests(pages)
    if len(requests) != 106 or len(generations) != 106:
        raise RuntimeError(
            f"fixed corpus must contain 106 requests: requests={len(requests)} "
            f"generation={len(generations)}"
        )
    control_summaries: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    for position, row in enumerate(requests):
        key = (row["source_image_name"], row["block_index"])
        if key not in generations:
            raise KeyError(f"generation trace is missing request {key}")
        print(
            f"VISION_EMBED step=capture_crop position={position + 1}/{len(requests)} "
            f"image={key[0]} block={key[1]}",
            flush=True,
        )
        captured_npu, metadata, prepared = capture_pretransformer(
            recognizer,
            row["request"],
        )
        captured = to_cpu(captured_npu)
        current_controls = {
            name: tensor_summary(captured[name]) for name in CONTROL_TENSORS[1:]
        }
        if control_summaries is None:
            control_summaries = current_controls
        else:
            for name in current_controls:
                if current_controls[name]["sha256"] != control_summaries[name]["sha256"]:
                    raise RuntimeError(f"model control tensor changed during corpus: {name}")
        generation = generations[key]
        records.append(
            {
                "source_page_index": row["source_page_index"],
                "source_image_name": row["source_image_name"],
                "block_index": row["block_index"],
                "request_id": row["request"].request_id,
                "prompt": row["request"].prompt,
                "label": generation.get("label"),
                "input_hashes": input_hashes(prepared),
                "input_tokens": int(prepared.input_ids.shape[1]),
                "projected_image_tokens": int(prepared.image_token_count),
                "metadata": metadata,
                "boundary_metadata": {
                    name: {
                        "shape": list(captured[name].shape),
                        "dtype": str(captured[name].dtype).removeprefix("torch."),
                        "numel": int(captured[name].numel()),
                    }
                    for name in OUTPUT_BOUNDARIES
                },
                "samples": {
                    name: deterministic_sample(captured[name], args.sample_elements)
                    for name in OUTPUT_BOUNDARIES
                },
                "sample_summaries": {
                    name: tensor_summary(
                        deterministic_sample(captured[name], args.sample_elements)
                    )
                    for name in OUTPUT_BOUNDARIES
                },
                "generation": generation,
            }
        )
        del captured_npu, captured

    reference = load_bundle(args.reference_bundle) if args.reference_bundle else None
    comparison = None
    if reference is not None:
        if reference.get("mode") != "corpus":
            raise ValueError("reference is not a corpus embedding bundle")
        reference_by_key = {
            (row["source_image_name"], int(row["block_index"])): row
            for row in reference["records"]
        }
        per_crop: list[dict[str, Any]] = []
        for candidate in records:
            key = (
                candidate["source_image_name"],
                int(candidate["block_index"]),
            )
            reference_row = reference_by_key.get(key)
            if reference_row is None:
                raise KeyError(f"reference bundle is missing crop {key}")
            boundary: dict[str, Any] = {}
            for name in OUTPUT_BOUNDARIES:
                candidate_metadata = candidate["boundary_metadata"][name]
                reference_metadata = reference_row["boundary_metadata"][name]
                shape_exact = (
                    candidate_metadata["shape"] == reference_metadata["shape"]
                    and candidate_metadata["numel"] == reference_metadata["numel"]
                )
                if not shape_exact:
                    boundary[name] = {
                        "shape_exact": False,
                        "candidate": candidate_metadata,
                        "reference": reference_metadata,
                    }
                else:
                    boundary[name] = {
                        **tensor_comparison(
                            candidate["samples"][name],
                            reference_row["samples"][name],
                        ),
                        "full_shape": candidate_metadata["shape"],
                        "sample_elements": int(
                            candidate["samples"][name].numel()
                        ),
                    }
            crop_hash_exact = (
                candidate["input_hashes"]["crop_sha256"]
                == reference_row["input_hashes"]["crop_sha256"]
            )
            prepared_inputs_exact = (
                candidate["input_hashes"]["prepared_inputs_sha256"]
                == reference_row["input_hashes"]["prepared_inputs_sha256"]
            )
            route_exact = all(
                candidate["metadata"].get(field)
                == reference_row["metadata"].get(field)
                for field in (
                    "grid_thw",
                    "real_vision_tokens",
                    "physical_vision_tokens",
                )
            )
            boundary_shapes_exact = all(
                bool(boundary[name]["shape_exact"])
                for name in OUTPUT_BOUNDARIES
            )
            per_crop.append(
                {
                    "source_page_index": candidate["source_page_index"],
                    "source_image_name": key[0],
                    "block_index": key[1],
                    "label": candidate["label"],
                    "crop_hash_exact": crop_hash_exact,
                    "prepared_inputs_exact": prepared_inputs_exact,
                    "route_exact": route_exact,
                    "boundary_shapes_exact": boundary_shapes_exact,
                    "model_execution_eligible": (
                        crop_hash_exact
                        and prepared_inputs_exact
                        and route_exact
                        and boundary_shapes_exact
                    ),
                    "generation": compare_generation(
                        reference_row["generation"],
                        candidate["generation"],
                    ),
                    "boundaries": boundary,
                }
            )

        eligible = [row for row in per_crop if row["model_execution_eligible"]]
        if not eligible:
            raise RuntimeError("no exact-input crops are eligible for correlation")
        correlation: dict[str, Any] = {}
        generation_distance = [
            float(row["generation"]["generation_distance"]) for row in eligible
        ]
        token_divergent = [
            not bool(row["generation"]["token_exact"]) for row in eligible
        ]
        first_token_divergent = [
            bool(row["generation"]["first_token_divergence"]) for row in eligible
        ]
        for name in OUTPUT_BOUNDARIES:
            relative_l2 = [
                float(row["boundaries"][name]["relative_l2"]) for row in eligible
            ]
            cosine = [
                float(row["boundaries"][name]["cosine_similarity"])
                for row in eligible
            ]
            correlation[name] = {
                "relative_l2_vs_generation_distance_spearman": spearman(
                    relative_l2, generation_distance
                ),
                "cosine_vs_generation_distance_spearman": spearman(
                    cosine, generation_distance
                ),
                "relative_l2_auc_token_divergent": roc_auc(
                    relative_l2, token_divergent
                ),
                "relative_l2_auc_first_token_divergent": roc_auc(
                    relative_l2, first_token_divergent
                ),
                "relative_l2_mean_token_exact": average(
                    value
                    for value, divergent in zip(relative_l2, token_divergent)
                    if not divergent
                ),
                "relative_l2_mean_token_divergent": average(
                    value
                    for value, divergent in zip(relative_l2, token_divergent)
                    if divergent
                ),
                "relative_l2_mean_first_token_exact": average(
                    value
                    for value, divergent in zip(
                        relative_l2, first_token_divergent
                    )
                    if not divergent
                ),
                "relative_l2_mean_first_token_divergent": average(
                    value
                    for value, divergent in zip(
                        relative_l2, first_token_divergent
                    )
                    if divergent
                ),
                "top_10_relative_l2": [
                    {
                        "source_image_name": row["source_image_name"],
                        "block_index": row["block_index"],
                        "label": row["label"],
                        "relative_l2": row["boundaries"][name]["relative_l2"],
                        "cosine": row["boundaries"][name]["cosine_similarity"],
                        "generation": row["generation"],
                    }
                    for row in sorted(
                        eligible,
                        key=lambda value: value["boundaries"][name]["relative_l2"],
                        reverse=True,
                    )[:10]
                ],
            }
        comparison = {
            "requests": len(per_crop),
            "crop_hash_exact": sum(row["crop_hash_exact"] for row in per_crop),
            "prepared_inputs_exact": sum(
                row["prepared_inputs_exact"] for row in per_crop
            ),
            "route_exact": sum(row["route_exact"] for row in per_crop),
            "boundary_shapes_exact": sum(
                row["boundary_shapes_exact"] for row in per_crop
            ),
            "model_execution_eligible": len(eligible),
            "eligible_token_exact": sum(
                row["generation"]["token_exact"] for row in eligible
            ),
            "eligible_token_divergent": sum(token_divergent),
            "eligible_first_token_divergent": sum(first_token_divergent),
            "correlation": correlation,
            "per_crop": per_crop,
        }

    bundle = {
        "schema_version": 1,
        "kind": "experiment09_pretransformer_vision_embedding_debug",
        "mode": "corpus",
        "sample_elements": args.sample_elements,
        "page_corpus": manifest["page_corpus"],
        "control_summaries": control_summaries,
        "records": records,
    }
    bundle_path = output_dir / "corpus_bundle.pt"
    torch.save(bundle, bundle_path)
    report = {
        "schema_version": 1,
        "kind": bundle["kind"],
        "mode": "corpus",
        "requests": len(records),
        "sample_elements": args.sample_elements,
        "control_summaries": control_summaries,
        "reference_bundle": str(args.reference_bundle) if args.reference_bundle else None,
        "comparison": comparison,
        "corpus_bundle": str(bundle_path),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.cases)
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("vision embedding debug requires an available NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    started = time.perf_counter()
    print("VISION_EMBED step=layout_setup_begin", flush=True)
    frontend = OwnedLayoutFrontend(
        args.layout_model.expanduser().resolve(),
        torch.device("npu:0"),
        graph_capture=False,
    )
    if args.mode == "exact":
        case = selected_case(manifest, args.case_id)
        page_manifest = {
            "page_corpus": {
                "source_page_indices": [case["source_page_index"]],
                "source_images": [case["source_image_name"]],
            }
        }
    else:
        page_manifest = manifest
    pages = load_pages(
        frontend,
        page_manifest,
        args.images_dir.expanduser().resolve(),
    )
    print("VISION_EMBED step=recognizer_setup_begin", flush=True)
    recognizer = build_recognizer(args)
    print(f"VISION_EMBED step={args.mode}_capture_begin", flush=True)
    if args.mode == "exact":
        exact_mode(args, manifest, pages, recognizer, output_dir)
    else:
        corpus_mode(args, manifest, pages, recognizer, output_dir)
    print(
        f"VISION_EMBED status=PASS mode={args.mode} "
        f"elapsed_s={time.perf_counter() - started:.3f} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
