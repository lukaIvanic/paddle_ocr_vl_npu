#!/usr/bin/env python3
"""Fingerprint the recognizer checkpoint and isolate embedding-op numerics.

The probe separates five possible sources of cross-machine drift:

1. model/config file bytes;
2. source safetensors and deterministic CPU dtype conversion;
3. CPU -> NPU -> CPU transfer and direct safetensors-to-NPU loading;
4. the production model loader's resident parameters;
5. NPU Conv2d, bilinear interpolation, addition, and trigonometric outputs.

It uses the committed Phase-46 exact-case bundle for the input tensor and grid,
but loads all weights from ``--model-dir``.  No TorchAir graph, transformer
layer, projector, text prefill, or decode is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.modeling import LocalPaddleOCRVLForConditionalGeneration
from table_token0_replay import tensor_comparison, tensor_summary
from vision_embedding_debug import deterministic_sample


DEFAULT_MODEL_DIR = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_EXACT_BUNDLE = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/"
    "910b_phase46_vision_embedding_exact_edc0e49/output/tensor_bundle.pt"
)
IMPORTANT_FILES = (
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "generation_config.json",
)
SELECTED_TENSORS = (
    "visual.vision_model.embeddings.patch_embedding.weight",
    "visual.vision_model.embeddings.patch_embedding.bias",
    "visual.vision_model.embeddings.position_embedding.weight",
    "visual.vision_model.encoder.layers.0.self_attn.q_proj.weight",
    "visual.vision_model.encoder.layers.0.mlp.fc1.weight",
    "mlp_AR.linear_1.weight",
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "lm_head.weight",
)
EMBEDDING_TENSOR_NAMES = {
    "patch_embedding_weight": SELECTED_TENSORS[0],
    "patch_embedding_bias": SELECTED_TENSORS[1],
    "position_embedding_weight": SELECTED_TENSORS[2],
}
DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
OP_BOUNDARIES = (
    "patch_embeddings",
    "position_embeddings",
    "summed_embeddings",
    "rope_cos",
    "rope_sin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--phase46-exact-bundle", type=Path, default=DEFAULT_EXACT_BUNDLE
    )
    parser.add_argument("--reference-bundle", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--sample-elements", type=int, default=65536)
    parser.add_argument(
        "--production-load-dtypes",
        default="fp16,bf16,fp32",
        help="Comma-separated production model-load dtypes.",
    )
    args = parser.parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.phase46_exact_bundle = args.phase46_exact_bundle.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.reference_bundle is not None:
        args.reference_bundle = args.reference_bundle.expanduser().resolve()
    if args.sample_elements <= 0:
        parser.error("--sample-elements must be positive")
    load_dtypes = tuple(
        value.strip() for value in args.production_load_dtypes.split(",") if value.strip()
    )
    unknown = [value for value in load_dtypes if value not in DTYPES]
    if unknown:
        parser.error(f"unknown production load dtypes: {unknown}")
    args.production_load_dtypes = load_dtypes
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def compact_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    summary = tensor_summary(tensor.detach().cpu())
    return {
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "numel": summary["numel"],
        "sha256": summary["sha256"],
        "finite_fraction": summary["finite_fraction"],
        "min": summary["min"],
        "max": summary["max"],
        "mean": summary["mean"],
        "abs_mean": summary["abs_mean"],
        "l2": summary["l2"],
    }


def hash_files(model_dir: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name in IMPORTANT_FILES:
        path = model_dir / name
        print(f"CHECKPOINT_PROBE step=file_hash name={name}", flush=True)
        if not path.is_file():
            rows[name] = {"exists": False}
            continue
        rows[name] = {
            "exists": True,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return rows


def source_manifest(checkpoint: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    manifest: dict[str, Any] = {}
    selected: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        missing = sorted(set(SELECTED_TENSORS) - set(keys))
        if missing:
            raise KeyError(f"checkpoint is missing selected tensors: {missing}")
        for position, name in enumerate(keys, start=1):
            if position == 1 or position % 50 == 0 or position == len(keys):
                print(
                    f"CHECKPOINT_PROBE step=tensor_manifest "
                    f"position={position}/{len(keys)}",
                    flush=True,
                )
            tensor = handle.get_tensor(name).contiguous()
            fp16 = tensor.to(torch.float16).contiguous()
            manifest[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "numel": tensor.numel(),
                "source_sha256": tensor_sha256(tensor),
                "fp16_sha256": tensor_sha256(fp16),
            }
            if name in SELECTED_TENSORS:
                selected[name] = tensor.clone()
            del tensor, fp16
    return manifest, selected


def selected_casts(selected: dict[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, source in selected.items():
        result[name] = {
            dtype_name: compact_tensor(source.to(dtype))
            for dtype_name, dtype in DTYPES.items()
        }
    return result


def npu_roundtrips(
    selected: dict[str, torch.Tensor], device: torch.device
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, source in selected.items():
        print(f"CHECKPOINT_PROBE step=npu_roundtrip tensor={name}", flush=True)
        per_dtype: dict[str, Any] = {}
        for dtype_name, dtype in DTYPES.items():
            cpu_value = source.to(dtype).contiguous()
            try:
                npu_value = cpu_value.to(device)
                torch.npu.synchronize()
                returned = npu_value.cpu()
                per_dtype[dtype_name] = {
                    "supported": True,
                    "cpu_sha256": tensor_sha256(cpu_value),
                    "roundtrip_sha256": tensor_sha256(returned),
                    "byte_exact": torch.equal(cpu_value, returned),
                    "comparison": tensor_comparison(returned, cpu_value),
                }
                del npu_value, returned
            except Exception as exc:
                per_dtype[dtype_name] = {
                    "supported": False,
                    "error": repr(exc),
                }
            torch.npu.empty_cache()
        result[name] = per_dtype
    return result


def direct_npu_safetensors(
    checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    print("CHECKPOINT_PROBE step=direct_safetensors_npu", flush=True)
    result: dict[str, Any] = {}
    try:
        with safe_open(checkpoint, framework="pt", device=str(device)) as handle:
            for name in SELECTED_TENSORS:
                value = handle.get_tensor(name)
                torch.npu.synchronize()
                returned = value.cpu()
                result[name] = {
                    "supported": True,
                    "summary": compact_tensor(returned),
                }
                del value, returned
    except Exception as exc:
        result = {"supported": False, "error": repr(exc)}
    torch.npu.empty_cache()
    return result


def model_weight(module: LocalPaddleOCRVLForConditionalGeneration, name: str) -> torch.Tensor:
    if name == SELECTED_TENSORS[0]:
        return module.visual.vision_model.embeddings.patch_embedding.weight
    if name == SELECTED_TENSORS[1]:
        return module.visual.vision_model.embeddings.patch_embedding.bias
    if name == SELECTED_TENSORS[2]:
        return module.visual.vision_model.embeddings.position_embedding.weight
    if name == SELECTED_TENSORS[3]:
        return module.visual.vision_model.encoder.layers[0].self_attn.q_proj.weight
    if name == SELECTED_TENSORS[4]:
        return module.visual.vision_model.encoder.layers[0].mlp.fc1.weight
    if name == SELECTED_TENSORS[5]:
        return module.mlp_AR.linear_1.weight
    if name == SELECTED_TENSORS[6]:
        return module.model.embed_tokens.weight
    if name == SELECTED_TENSORS[7]:
        return module.model.layers[0].self_attn.q_proj.weight
    if name == SELECTED_TENSORS[8]:
        return module.lm_head.weight
    raise KeyError(name)


def production_model_loads(
    model_dir: Path,
    device: torch.device,
    dtype_names: Iterable[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dtype_name in dtype_names:
        print(
            f"CHECKPOINT_PROBE step=production_model_load dtype={dtype_name}",
            flush=True,
        )
        started = time.perf_counter()
        try:
            model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
                model_dir,
                dtype=DTYPES[dtype_name],
                device=device,
            )
            torch.npu.synchronize()
            weights = {
                name: compact_tensor(model_weight(model, name).detach().cpu())
                for name in SELECTED_TENSORS
            }
            result[dtype_name] = {
                "supported": True,
                "elapsed_s": time.perf_counter() - started,
                "weights": weights,
            }
            del model
        except Exception as exc:
            result[dtype_name] = {
                "supported": False,
                "elapsed_s": time.perf_counter() - started,
                "error": repr(exc),
            }
        torch.npu.empty_cache()
    return result


def cpu_fp32_reference(
    conv_input: torch.Tensor,
    patch_weight: torch.Tensor,
    patch_bias: torch.Tensor,
    position_weight: torch.Tensor,
    selected_angles: torch.Tensor,
    height: int,
    width: int,
) -> dict[str, torch.Tensor]:
    print("CHECKPOINT_PROBE step=cpu_fp32_reference", flush=True)
    conv_flat = conv_input.float().reshape(conv_input.shape[0], -1)
    weight_flat = patch_weight.float().reshape(patch_weight.shape[0], -1)
    patch = F.linear(conv_flat, weight_flat, patch_bias.float())
    dim = position_weight.shape[-1]
    side = int(position_weight.shape[0] ** 0.5)
    position = F.interpolate(
        position_weight.float()
        .reshape(1, side, side, dim)
        .permute(0, 3, 1, 2),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).reshape(-1, dim)
    return {
        "patch_embeddings": patch,
        "position_embeddings": position,
        "summed_embeddings": patch + position,
        "rope_cos": selected_angles.float().cos(),
        "rope_sin": selected_angles.float().sin(),
    }


@torch.inference_mode()
def run_embedding_ops(
    device: torch.device,
    conv_input: torch.Tensor,
    patch_weight: torch.Tensor,
    patch_bias: torch.Tensor,
    position_weight: torch.Tensor,
    selected_angles: torch.Tensor,
    height: int,
    width: int,
    cpu_reference: dict[str, torch.Tensor],
    sample_elements: int,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    report: dict[str, Any] = {}
    samples: dict[str, dict[str, torch.Tensor]] = {}
    for dtype_name, dtype in DTYPES.items():
        print(f"CHECKPOINT_PROBE step=embedding_ops dtype={dtype_name}", flush=True)
        started = time.perf_counter()
        try:
            inputs = conv_input.to(device=device, dtype=dtype)
            weight = patch_weight.to(device=device, dtype=dtype)
            bias = patch_bias.to(device=device, dtype=dtype)
            pos_weight = position_weight.to(device=device, dtype=dtype)
            angles = selected_angles.to(device=device, dtype=dtype)

            patch = F.conv2d(inputs, weight, bias=bias, stride=(14, 14))
            patch = patch.flatten(-2).squeeze(-1)
            dim = pos_weight.shape[-1]
            side = int(pos_weight.shape[0] ** 0.5)
            position = F.interpolate(
                pos_weight.reshape(1, side, side, dim).permute(0, 3, 1, 2),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).reshape(-1, dim)
            outputs = {
                "patch_embeddings": patch,
                "position_embeddings": position,
                "summed_embeddings": patch + position,
                "rope_cos": angles.cos(),
                "rope_sin": angles.sin(),
            }
            torch.npu.synchronize()
            cpu_outputs = {name: value.cpu() for name, value in outputs.items()}
            report[dtype_name] = {
                "supported": True,
                "elapsed_s": time.perf_counter() - started,
                "boundaries": {
                    name: {
                        "summary": compact_tensor(value),
                        "vs_cpu_fp32": tensor_comparison(
                            value.float(), cpu_reference[name]
                        ),
                    }
                    for name, value in cpu_outputs.items()
                },
            }
            samples[dtype_name] = {
                name: deterministic_sample(value, sample_elements)
                for name, value in cpu_outputs.items()
            }
            del inputs, weight, bias, pos_weight, angles, outputs, cpu_outputs
        except Exception as exc:
            report[dtype_name] = {
                "supported": False,
                "elapsed_s": time.perf_counter() - started,
                "error": repr(exc),
            }
            samples[dtype_name] = {}
        torch.npu.empty_cache()
    return report, samples


def compare_manifests(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    all_names = sorted(set(candidate) | set(reference))
    rows = []
    for name in all_names:
        left = candidate.get(name)
        right = reference.get(name)
        rows.append(
            {
                "name": name,
                "present_candidate": left is not None,
                "present_reference": right is not None,
                "shape_exact": left is not None and right is not None and left["shape"] == right["shape"],
                "dtype_exact": left is not None and right is not None and left["dtype"] == right["dtype"],
                "source_exact": left is not None and right is not None and left["source_sha256"] == right["source_sha256"],
                "fp16_exact": left is not None and right is not None and left["fp16_sha256"] == right["fp16_sha256"],
            }
        )
    return {
        "candidate_tensors": len(candidate),
        "reference_tensors": len(reference),
        "common_tensors": sum(row["present_candidate"] and row["present_reference"] for row in rows),
        "source_exact": sum(row["source_exact"] for row in rows),
        "fp16_exact": sum(row["fp16_exact"] for row in rows),
        "different": [row for row in rows if not row["source_exact"] or not row["fp16_exact"]],
    }


def compare_reference(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    operation_comparisons: dict[str, Any] = {}
    for dtype_name, candidate_samples in candidate["operation_samples"].items():
        reference_samples = reference["operation_samples"].get(dtype_name, {})
        operation_comparisons[dtype_name] = {
            name: tensor_comparison(candidate_samples[name], reference_samples[name])
            for name in OP_BOUNDARIES
            if name in candidate_samples and name in reference_samples
        }
    roundtrip_comparisons: dict[str, Any] = {}
    for name in SELECTED_TENSORS:
        roundtrip_comparisons[name] = {}
        for dtype_name in DTYPES:
            candidate_row = candidate["npu_roundtrips"][name][dtype_name]
            reference_row = reference["npu_roundtrips"][name][dtype_name]
            roundtrip_comparisons[name][dtype_name] = {
                "candidate_supported": candidate_row["supported"],
                "reference_supported": reference_row["supported"],
                "returned_hash_exact": (
                    candidate_row.get("roundtrip_sha256")
                    == reference_row.get("roundtrip_sha256")
                ),
                "candidate_byte_exact_to_its_cpu": candidate_row.get("byte_exact"),
                "reference_byte_exact_to_its_cpu": reference_row.get("byte_exact"),
            }
    production_comparisons: dict[str, Any] = {}
    for dtype_name in DTYPES:
        candidate_load = candidate["production_model_loads"].get(dtype_name, {})
        reference_load = reference["production_model_loads"].get(dtype_name, {})
        production_comparisons[dtype_name] = {
            "candidate_supported": candidate_load.get("supported", False),
            "reference_supported": reference_load.get("supported", False),
            "weights": {
                name: {
                    "candidate_sha256": candidate_load.get("weights", {})
                    .get(name, {})
                    .get("sha256"),
                    "reference_sha256": reference_load.get("weights", {})
                    .get(name, {})
                    .get("sha256"),
                    "exact": (
                        candidate_load.get("weights", {}).get(name, {}).get("sha256")
                        == reference_load.get("weights", {}).get(name, {}).get("sha256")
                    ),
                }
                for name in SELECTED_TENSORS
            },
        }
    direct_candidate = candidate["direct_npu_safetensors"]
    direct_reference = reference["direct_npu_safetensors"]
    direct_comparisons = {
        name: {
            "candidate_supported": direct_candidate.get(name, {}).get(
                "supported", False
            ),
            "reference_supported": direct_reference.get(name, {}).get(
                "supported", False
            ),
            "candidate_sha256": direct_candidate.get(name, {})
            .get("summary", {})
            .get("sha256"),
            "reference_sha256": direct_reference.get(name, {})
            .get("summary", {})
            .get("sha256"),
            "exact": (
                direct_candidate.get(name, {}).get("summary", {}).get("sha256")
                == direct_reference.get(name, {}).get("summary", {}).get("sha256")
            ),
        }
        for name in SELECTED_TENSORS
    }
    return {
        "file_comparisons": {
            name: {
                "candidate": candidate["files"].get(name),
                "reference": reference["files"].get(name),
                "exact": candidate["files"].get(name) == reference["files"].get(name),
            }
            for name in IMPORTANT_FILES
        },
        "tensor_manifest": compare_manifests(
            candidate["tensor_manifest"], reference["tensor_manifest"]
        ),
        "selected_casts_exact": {
            name: {
                dtype_name: (
                    candidate["selected_casts"][name][dtype_name]["sha256"]
                    == reference["selected_casts"][name][dtype_name]["sha256"]
                )
                for dtype_name in DTYPES
            }
            for name in SELECTED_TENSORS
        },
        "npu_roundtrips": roundtrip_comparisons,
        "direct_npu_safetensors": direct_comparisons,
        "production_model_loads": production_comparisons,
        "operation_samples": operation_comparisons,
    }


def strip_samples(bundle: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bundle.items() if key != "operation_samples"}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.model_dir / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    phase46 = torch.load(
        args.phase46_exact_bundle,
        map_location="cpu",
        weights_only=False,
    )
    if phase46.get("case_id") != "table_token0_11_3":
        raise ValueError("the Phase-46 bundle is not the fixed exact case")
    tensors = phase46["tensors"]
    conv_input = tensors["conv_input"].contiguous()
    selected_angles = tensors["rotary_selected_angles"].contiguous()
    _, height, width = phase46["metadata"]["grid_thw"]

    print("CHECKPOINT_PROBE step=setup_npu", flush=True)
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)

    files = hash_files(args.model_dir)
    tensor_manifest, selected = source_manifest(checkpoint)
    selected_values = selected_casts(selected)
    roundtrips = npu_roundtrips(selected, device)
    direct_npu = direct_npu_safetensors(checkpoint, device)
    production = production_model_loads(
        args.model_dir,
        device,
        args.production_load_dtypes,
    )

    patch_weight = selected[EMBEDDING_TENSOR_NAMES["patch_embedding_weight"]]
    patch_bias = selected[EMBEDDING_TENSOR_NAMES["patch_embedding_bias"]]
    position_weight = selected[EMBEDDING_TENSOR_NAMES["position_embedding_weight"]]
    cpu_reference = cpu_fp32_reference(
        conv_input,
        patch_weight,
        patch_bias,
        position_weight,
        selected_angles,
        height,
        width,
    )
    cpu_reference_report = {
        name: compact_tensor(value) for name, value in cpu_reference.items()
    }
    operation_matrix, operation_samples = run_embedding_ops(
        device,
        conv_input,
        patch_weight,
        patch_bias,
        position_weight,
        selected_angles,
        height,
        width,
        cpu_reference,
        args.sample_elements,
    )

    bundle = {
        "schema_version": 1,
        "kind": "checkpoint_embedding_ops_probe",
        "model_dir": str(args.model_dir),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "phase46_exact_bundle": str(args.phase46_exact_bundle),
        "sample_elements": args.sample_elements,
        "files": files,
        "tensor_manifest": tensor_manifest,
        "selected_casts": selected_values,
        "npu_roundtrips": roundtrips,
        "direct_npu_safetensors": direct_npu,
        "production_model_loads": production,
        "cpu_fp32_reference": cpu_reference_report,
        "operation_matrix": operation_matrix,
        "operation_samples": operation_samples,
    }
    reference_comparison = None
    if args.reference_bundle is not None:
        reference = torch.load(
            args.reference_bundle,
            map_location="cpu",
            weights_only=False,
        )
        if reference.get("kind") != bundle["kind"]:
            raise ValueError("reference bundle kind mismatch")
        reference_comparison = compare_reference(bundle, reference)

    bundle_path = args.output_dir / "probe_bundle.pt"
    torch.save(bundle, bundle_path)
    report = {
        **strip_samples(bundle),
        "reference_bundle": (
            str(args.reference_bundle) if args.reference_bundle is not None else None
        ),
        "reference_comparison": reference_comparison,
        "probe_bundle": str(bundle_path),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"CHECKPOINT_PROBE status=PASS output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
