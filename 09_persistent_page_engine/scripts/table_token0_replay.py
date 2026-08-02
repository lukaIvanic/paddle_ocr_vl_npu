#!/usr/bin/env python3
"""Replay one table crop through the production prefill path to token zero.

The default case is the clean Phase-39 table divergence: OmniDocBench page 11,
block 3.  It is a singleton in both production routing stages (vision
4032->4992, text 1021->1024), so its intermediate tensors are not influenced by
pack companions.  Decode is intentionally not launched: the script stops after
the production prefill chain computes the first-token logits and argmax.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.serving.engine import ContinuousRecognizer
from pipeline.layout_frontend import OwnedLayoutFrontend


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
CAPTURE_ORDER = (
    "vision_embeddings",
    "vision_graph_input",
    "vision_prefill_output",
    "projector_output",
    "multimodal_inputs_embeds",
    "packed_text_graph_input",
    "text_prefill_last_hidden",
    "token0_logits",
)

# These hashes were exact on both devices in Phase 39.  Requiring them keeps a
# changed crop or preprocessing contract from being mistaken for model drift.
EXPECTED_INPUTS = {
    "table_token0_11_3": {
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
        "physical_text_tokens": 1024,
    }
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="table_token0_11_3")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--layout-model", type=Path, default=DEFAULT_LAYOUT_MODEL)
    parser.add_argument(
        "--recognizer-model", type=Path, default=DEFAULT_RECOGNIZER_MODEL
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-bundle",
        type=Path,
        default=None,
        help="Optional 910B tensor bundle to compare against in the same run.",
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
    return parser.parse_args(argv)


def load_case(path: Path, case_id: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    matches = [row for row in payload.get("cases", ()) if row.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one case {case_id!r}, got {len(matches)}")
    return dict(matches[0])


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().to(device="cpu").contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu").contiguous()
    as_float = value.float()
    finite = torch.isfinite(as_float)
    finite_values = as_float[finite]
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "numel": int(value.numel()),
        "bytes": int(value.numel() * value.element_size()),
        "sha256": hashlib.sha256(tensor_bytes(value)).hexdigest(),
        "finite_fraction": (
            float(finite.float().mean().item()) if value.numel() else 1.0
        ),
        "min": (
            float(finite_values.min().item()) if finite_values.numel() else None
        ),
        "max": (
            float(finite_values.max().item()) if finite_values.numel() else None
        ),
        "mean": (
            float(finite_values.mean().item()) if finite_values.numel() else None
        ),
        "abs_mean": (
            float(finite_values.abs().mean().item())
            if finite_values.numel()
            else None
        ),
        "l2": (
            float(torch.linalg.vector_norm(finite_values).item())
            if finite_values.numel()
            else None
        ),
    }


def tensor_comparison(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, Any]:
    if tuple(candidate.shape) != tuple(reference.shape):
        return {
            "shape_exact": False,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
        }
    left = candidate.float().reshape(-1)
    right = reference.float().reshape(-1)
    delta = (left - right).abs()
    left_l2 = torch.linalg.vector_norm(left)
    right_l2 = torch.linalg.vector_norm(right)
    delta_l2 = torch.linalg.vector_norm(left - right)
    denominator = left_l2 * right_l2
    cosine = (
        float(torch.dot(left, right).item() / denominator.item())
        if denominator.item() != 0.0
        else None
    )
    quantiles = (
        torch.quantile(delta, torch.tensor([0.5, 0.95, 0.99]))
        if delta.numel()
        else torch.zeros(3)
    )
    return {
        "shape_exact": True,
        "dtype_exact": candidate.dtype == reference.dtype,
        "byte_exact": tensor_bytes(candidate) == tensor_bytes(reference),
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "rms_abs": (
            float(torch.sqrt(torch.mean(delta.square())).item())
            if delta.numel()
            else 0.0
        ),
        "p50_abs": float(quantiles[0].item()),
        "p95_abs": float(quantiles[1].item()),
        "p99_abs": float(quantiles[2].item()),
        "relative_l2": (
            float(delta_l2.item() / right_l2.item())
            if right_l2.item() != 0.0
            else None
        ),
        "cosine_similarity": cosine,
    }


def topk_rows(
    logits: torch.Tensor,
    tokenizer: Any,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    vector = logits.float().reshape(-1, logits.shape[-1])[-1]
    values, indices = torch.topk(vector, k=min(limit, vector.numel()))
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "logit": float(logit),
            "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
        }
        for rank, (token_id, logit) in enumerate(
            zip(indices.tolist(), values.tolist()),
            start=1,
        )
    ]


def token_decision(
    candidate: torch.Tensor,
    tokenizer: Any,
    reference: torch.Tensor | None,
) -> dict[str, Any]:
    vector = candidate.float().reshape(-1, candidate.shape[-1])[-1]
    top_values, top_indices = torch.topk(vector, k=2)
    top_id = int(top_indices[0].item())
    payload: dict[str, Any] = {
        "top1_id": top_id,
        "top1_token": tokenizer.decode([top_id], skip_special_tokens=False),
        "top1_logit": float(top_values[0].item()),
        "top1_margin": float((top_values[0] - top_values[1]).item()),
        "top20": topk_rows(candidate, tokenizer),
    }
    if reference is None:
        return payload
    reference_vector = reference.float().reshape(-1, reference.shape[-1])[-1]
    reference_id = int(torch.argmax(reference_vector).item())
    candidate_order = torch.argsort(vector, descending=True)
    reference_rank = int((candidate_order == reference_id).nonzero()[0].item()) + 1
    payload.update(
        {
            "same_top1": top_id == reference_id,
            "reference_top1_id": reference_id,
            "reference_top1_token": tokenizer.decode(
                [reference_id], skip_special_tokens=False
            ),
            "reference_token_candidate_rank": reference_rank,
            "candidate_logit_for_reference_token": float(vector[reference_id].item()),
            "candidate_top1_minus_reference_token": float(
                (vector[top_id] - vector[reference_id]).item()
            ),
        }
    )
    return payload


def load_reference(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(
            path.expanduser().resolve(), map_location="cpu", weights_only=True
        )
    except TypeError:
        payload = torch.load(path.expanduser().resolve(), map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("tensors"), dict):
        raise ValueError("reference bundle is not a table-token0 replay bundle")
    return payload


def graph_input_comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    groups = sorted(set(candidate) | set(reference))
    result: dict[str, Any] = {}
    for group in groups:
        candidate_group = candidate.get(group) or {}
        reference_group = reference.get(group) or {}
        names = sorted(set(candidate_group) | set(reference_group))
        result[group] = {}
        for name in names:
            left = candidate_group.get(name)
            right = reference_group.get(name)
            result[group][name] = {
                "present_on_both": left is not None and right is not None,
                "shape_exact": (
                    left is not None
                    and right is not None
                    and left.get("shape") == right.get("shape")
                ),
                "dtype_exact": (
                    left is not None
                    and right is not None
                    and left.get("dtype") == right.get("dtype")
                ),
                "sha256_exact": (
                    left is not None
                    and right is not None
                    and left.get("sha256") == right.get("sha256")
                ),
                "candidate_sha256": None if left is None else left.get("sha256"),
                "reference_sha256": None if right is None else right.get("sha256"),
            }
    return result


def route_signature(
    route: dict[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    if kind == "vision":
        keys = (
            "execution",
            "real_vision_tokens",
            "physical_vision_tokens",
            "padding_vision_tokens",
            "bucket",
            "packing",
            "pack_crops",
            "pack_real_vision_tokens",
            "pack_physical_vision_tokens",
            "pack_batch_size",
            "pack_sequence_length",
            "pack_row_sizes",
        )
    elif kind == "text":
        keys = (
            "execution",
            "real_text_tokens",
            "physical_text_tokens",
            "padding_text_tokens",
            "bucket",
            "packing",
            "pack_members",
            "segment_lengths",
            "pack_real_text_tokens",
            "pack_physical_text_tokens",
        )
    else:
        raise ValueError(f"unsupported route signature kind: {kind}")
    return {key: route.get(key) for key in keys}


def capture_prefill(
    recognizer: ContinuousRecognizer,
    request: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], Any]:
    captured: dict[str, torch.Tensor] = {}
    prepared_metadata: dict[str, Any] = {}

    def hold(name: str, value: torch.Tensor) -> None:
        if name in captured:
            raise RuntimeError(f"captured boundary {name!r} more than once")
        captured[name] = value.detach()

    vision_embeddings = recognizer.model.visual.vision_model.embeddings
    projector = recognizer.model.mlp_AR
    lm_head = recognizer.model.lm_head
    handles = [
        vision_embeddings.register_forward_hook(
            lambda _module, _inputs, output: hold("vision_embeddings", output)
        ),
        projector.register_forward_hook(
            lambda _module, _inputs, output: hold("projector_output", output)
        ),
        lm_head.register_forward_hook(
            lambda _module, _inputs, output: hold("token0_logits", output)
        ),
    ]

    original_vision_run = recognizer.vision_prefill.run_prepared
    original_text_prepare = recognizer.packed_text_prefill.prepare
    original_text_run = recognizer.packed_text_prefill.run_prepared

    def vision_run(_runtime: Any, prepared: Any) -> torch.Tensor:
        hold("vision_graph_input", prepared.prefix_hidden_states)
        prepared_metadata["vision_graph_inputs"] = {
            "prefix_hidden_states": prepared.prefix_hidden_states,
            "rope_cos": prepared.rope_cos,
            "rope_sin": prepared.rope_sin,
            "attention_mask": prepared.attention_mask,
        }
        output = original_vision_run(prepared)
        hold("vision_prefill_output", output)
        return output

    def text_prepare(
        _runtime: Any,
        inputs_embeds: list[torch.Tensor],
        position_ids: list[torch.Tensor],
        *,
        route: dict[str, Any],
    ) -> Any:
        if len(inputs_embeds) != 1:
            raise RuntimeError(
                "the default table token-zero case must remain a singleton text pack"
            )
        hold("multimodal_inputs_embeds", inputs_embeds[0])
        prepared = original_text_prepare(inputs_embeds, position_ids, route=route)
        hold("packed_text_graph_input", prepared.inputs_embeds)
        prepared_metadata["packed_text_graph_inputs"] = {
            "inputs_embeds": prepared.inputs_embeds,
            "position_ids": prepared.position_ids,
            "segment_ids": prepared.segment_ids,
            "local_positions": prepared.local_positions,
            "last_token_indices": prepared.last_token_indices,
        }
        prepared_metadata["packed_text_segments"] = {
            "segment_lengths": list(prepared.segment_lengths),
            "segment_offsets": list(prepared.segment_offsets),
            "real_seq_len": int(prepared.real_seq_len),
            "physical_seq_len": int(prepared.physical_seq_len),
        }
        return prepared

    def text_run(_runtime: Any, prepared: Any) -> torch.Tensor:
        output = original_text_run(prepared)
        hold("text_prefill_last_hidden", output)
        return output

    recognizer.vision_prefill.run_prepared = MethodType(
        vision_run, recognizer.vision_prefill
    )
    recognizer.packed_text_prefill.prepare = MethodType(
        text_prepare, recognizer.packed_text_prefill
    )
    recognizer.packed_text_prefill.run_prepared = MethodType(
        text_run, recognizer.packed_text_prefill
    )

    finalized: list[Any] = []
    try:
        submitted_at = time.perf_counter()
        prepared = recognizer._prepare_cpu(request, submitted_at)
        group = recognizer._prepared_group([(prepared, 0.0)], row_sizes=(1,))
        staged = recognizer._stage_prefill_group(group)
        inflight = recognizer._enqueue_staged_prefill_group(staged)
        finalized = recognizer._finalize_prefill_group(inflight)
    finally:
        recognizer.vision_prefill.run_prepared = original_vision_run
        recognizer.packed_text_prefill.prepare = original_text_prepare
        recognizer.packed_text_prefill.run_prepared = original_text_run
        for handle in handles:
            handle.remove()

    if len(finalized) != 1:
        raise RuntimeError(f"expected one finalized prefill, got {len(finalized)}")
    state = finalized[0]
    _, _, _, _, release = state.take_device_state()
    if release is not None:
        release()
    return captured, prepared_metadata, state


def validate_contract(
    case: dict[str, Any],
    state: Any,
    captured: dict[str, torch.Tensor],
) -> dict[str, Any]:
    missing = [name for name in CAPTURE_ORDER if name not in captured]
    if missing:
        raise RuntimeError(f"missing captured boundaries: {missing}")
    expected = EXPECTED_INPUTS.get(str(case["case_id"]))
    checks: dict[str, Any] = {
        "prompt": state.prompt == case["reference_prompt"],
        "input_tokens": state.input_tokens == int(case["reference_input_tokens"]),
        "projected_image_tokens": state.projected_image_tokens
        == int(case["reference_projected_image_tokens"]),
        "vision_singleton": state.vision.get("packing") == "single",
        "vision_real_tokens": int(state.vision.get("real_vision_tokens", -1))
        == int(captured["vision_embeddings"].shape[0]),
        "text_singleton": int(state.text_prefill.get("pack_members", -1)) == 1,
    }
    if expected is not None:
        fingerprints = state.input_fingerprints
        checks.update(
            {
                "crop_sha256": (
                    (fingerprints.get("crop") or {}).get("sha256")
                    == expected["crop_sha256"]
                ),
                "prepared_inputs_sha256": (
                    fingerprints.get("prepared_inputs_sha256")
                    == expected["prepared_inputs_sha256"]
                ),
                "expected_input_tokens": state.input_tokens
                == expected["input_tokens"],
                "expected_projected_image_tokens": state.projected_image_tokens
                == expected["projected_image_tokens"],
                "expected_vision_route": (
                    int(state.vision.get("real_vision_tokens", -1))
                    == expected["real_vision_tokens"]
                    and int(state.vision.get("physical_vision_tokens", -1))
                    == expected["physical_vision_tokens"]
                ),
                "expected_text_route": int(
                    state.text_prefill.get("physical_text_tokens", -1)
                )
                == expected["physical_text_tokens"],
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"table token-zero replay contract failed: {failed}")
    return {"status": "PASS", "checks": checks}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case = load_case(args.cases, args.case_id)
    image_path = args.images_dir.expanduser().resolve() / case["source_image_name"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("table token-zero replay requires an available NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    started = time.perf_counter()
    print("PHASE45 step=layout_setup_begin", flush=True)
    layout = OwnedLayoutFrontend(
        args.layout_model.expanduser().resolve(),
        device,
        graph_capture=False,
    )
    print("PHASE45 step=layout_prepare_begin", flush=True)
    page = layout.prepare_page(
        image_path,
        int(case["source_page_index"]),
        min_pixels=28_224,
        max_pixels=1_003_520,
    )
    block_index = int(case["block_index"])
    try:
        request_position = page.request_block_indices.index(block_index)
    except ValueError as exc:
        raise RuntimeError(
            f"layout did not produce requested block {block_index}; "
            f"available={page.request_block_indices}"
        ) from exc
    request = page.requests[request_position]
    print(
        "PHASE45 step=recognizer_setup_begin "
        f"request_id={request.request_id} crop={request.crop.size}",
        flush=True,
    )
    recognizer = ContinuousRecognizer(
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
    print("PHASE45 step=prefill_replay_begin", flush=True)
    captured_npu, prepared_metadata, state = capture_prefill(recognizer, request)
    torch.npu.synchronize()
    captured = {
        name: captured_npu[name].detach().to(device="cpu").contiguous()
        for name in CAPTURE_ORDER
    }
    graph_input_summaries = {
        group: {name: tensor_summary(tensor) for name, tensor in tensors.items()}
        for group, tensors in prepared_metadata.items()
        if group.endswith("_graph_inputs")
    }
    contract = validate_contract(case, state, captured)

    reference = (
        load_reference(args.reference_bundle)
        if args.reference_bundle is not None
        else None
    )
    comparisons: dict[str, Any] | None = None
    graph_input_comparisons: dict[str, Any] | None = None
    route_comparisons: dict[str, Any] | None = None
    reference_logits: torch.Tensor | None = None
    if reference is not None:
        if reference.get("case_id") != args.case_id:
            raise ValueError(
                "reference case mismatch: "
                f"{reference.get('case_id')!r} != {args.case_id!r}"
            )
        reference_tensors = reference["tensors"]
        missing_reference = [
            name for name in CAPTURE_ORDER if name not in reference_tensors
        ]
        if missing_reference:
            raise ValueError(
                f"reference bundle is missing boundaries: {missing_reference}"
            )
        comparisons = {
            name: tensor_comparison(captured[name], reference_tensors[name])
            for name in CAPTURE_ORDER
        }
        graph_input_comparisons = graph_input_comparison(
            graph_input_summaries,
            reference.get("graph_input_summaries") or {},
        )
        candidate_vision_signature = route_signature(state.vision, kind="vision")
        reference_vision_signature = route_signature(
            reference.get("vision_route") or {}, kind="vision"
        )
        candidate_text_signature = route_signature(
            state.text_prefill, kind="text"
        )
        reference_text_signature = route_signature(
            reference.get("text_prefill_route") or {}, kind="text"
        )
        route_comparisons = {
            "vision": {
                "exact": candidate_vision_signature == reference_vision_signature,
                "candidate": candidate_vision_signature,
                "reference": reference_vision_signature,
            },
            "text_prefill": {
                "exact": candidate_text_signature == reference_text_signature,
                "candidate": candidate_text_signature,
                "reference": reference_text_signature,
            },
        }
        reference_logits = reference_tensors["token0_logits"]

    bundle_path = output_dir / "tensor_bundle.pt"
    torch.save(
        {
            "schema_version": 1,
            "kind": "experiment09_table_token0_prefill_replay",
            "case_id": args.case_id,
            "case": case,
            "contract": contract,
            "input_fingerprints": state.input_fingerprints,
            "vision_route": state.vision,
            "text_prefill_route": state.text_prefill,
            "graph_input_summaries": graph_input_summaries,
            "tensors": captured,
        },
        bundle_path,
    )
    decision = token_decision(
        captured["token0_logits"], recognizer.tokenizer, reference_logits
    )
    report = {
        "schema_version": 1,
        "kind": "experiment09_table_token0_prefill_replay_report",
        "case_id": args.case_id,
        "source_image_name": case["source_image_name"],
        "source_page_index": int(case["source_page_index"]),
        "block_index": block_index,
        "request_id": request.request_id,
        "elapsed_s": time.perf_counter() - started,
        "contract": contract,
        "input_fingerprints": state.input_fingerprints,
        "vision_route": state.vision,
        "text_prefill_route": state.text_prefill,
        "first_token": int(state.first_token),
        "first_token_decoded": recognizer.tokenizer.decode(
            [int(state.first_token)], skip_special_tokens=False
        ),
        "decision": decision,
        "tensor_summaries": {
            name: tensor_summary(captured[name]) for name in CAPTURE_ORDER
        },
        "graph_input_summaries": graph_input_summaries,
        "reference_bundle": (
            str(args.reference_bundle.expanduser().resolve())
            if args.reference_bundle is not None
            else None
        ),
        "comparisons": comparisons,
        "graph_input_comparisons": graph_input_comparisons,
        "route_comparisons": route_comparisons,
        "tensor_bundle": str(bundle_path),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "PHASE45 status=PASS "
        f"case={args.case_id} token={decision['top1_id']} "
        f"decoded={decision['top1_token']!r} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
