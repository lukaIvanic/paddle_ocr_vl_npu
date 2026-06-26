#!/usr/bin/env python3
"""Validate fixed-S static visual candidates through real OCR generation.

This script stays inside experiment 07. It compares:

  stored eager PromptFA baseline visual_features -> projector/text prefill/decode
  candidate static visual tower -> projector/text prefill/decode

The headline speed metric remains the candidate visual tower only:
device-loaded pixel tensor, sync, visual tower call, sync. Projector/text prefill/decode
timings are reported separately as correctness context.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from vision_prefill_bench import (
    DEFAULT_MODEL,
    DEFAULT_VISION_TORCHAIR_CACHE_DIR,
    DTYPE_CHOICES,
    NPU_JIT_COMPILE_CHOICES,
    STATIC_VISUAL_LN_IMPL_CHOICES,
    STATIC_VISUAL_LN_LINEAR_MODE_CHOICES,
    TIMING_MODE_CHOICES,
    TORCHAIR_GRAPH_DUMP_TYPE_CHOICES,
    TORCHAIR_MODE_CHOICES,
    TORCHAIR_MSIT_DUMP_KIND_CHOICES,
    TORCHAIR_MSIT_DUMP_MODE_CHOICES,
    VISION_ATTENTION_CHOICES,
    VISION_COMPILE_BACKEND_CHOICES,
    add_common_args,
    add_torchair_diagnostic_args,
    aggregate_diff,
    aggregate_timed_token_rate,
    apply_runtime_env,
    build_inputs_from_manifest,
    clean_json,
    compute_prefill_state_from_visual_features,
    compute_visual_tower_only,
    diff_stats,
    first_mismatch,
    generate_from_prefill_state,
    input_row,
    json_default,
    load_baseline_manifest,
    load_model_for_args,
    load_preprocessor_config,
    maybe_sync,
    prepare_candidate_vision_forward,
    resolve_dataset_dir,
    sha256_file,
    stats,
    topk_summary,
    trim_after_eos,
    vision_tokens,
)


def decode_safely(tokenizer: Tokenizer, token_ids: list[int]) -> tuple[str, list[int]]:
    vocab_size = int(tokenizer.get_vocab_size())
    invalid = [int(value) for value in token_ids if int(value) < 0 or int(value) >= vocab_size]
    if invalid:
        return "", invalid[:16]
    return tokenizer.decode(token_ids, skip_special_tokens=True), []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, timing_default="standard")
    parser.add_argument("--baseline", default=str(Path(__file__).resolve().parent / "baselines" / "promptfa_fp16_eager_64"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "outputs" / "static_visual_generation.json"))
    parser.add_argument("--candidate-name", default="static_visual_generation_candidate")
    parser.add_argument("--max-items", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--vision-compile-backend", default="torchair", choices=VISION_COMPILE_BACKEND_CHOICES)
    parser.add_argument(
        "--vision-use-torchair-cache-compile",
        action="store_true",
        help="Use torchair.inference.cache_compile with GE cache for --vision-compile-backend torchair.",
    )
    parser.add_argument(
        "--vision-torchair-cache-dir",
        default=str(DEFAULT_VISION_TORCHAIR_CACHE_DIR),
        help="Root directory for static-visual TorchAir GE cache entries.",
    )
    parser.add_argument("--static-visual-fixed-physical-seq-len", type=int, default=1024)
    parser.add_argument("--static-visual-ln-impl", default="manual_fp32", choices=STATIC_VISUAL_LN_IMPL_CHOICES)
    parser.add_argument(
        "--static-visual-ln-linear-mode",
        default="grouped_qkv_mlp_fc1",
        choices=STATIC_VISUAL_LN_LINEAR_MODE_CHOICES,
    )
    parser.add_argument("--static-visual-promptfa-pad-head-dim-to", type=int, default=80)
    parser.add_argument("--validate-compiled-against-static-eager", action="store_true")
    add_torchair_diagnostic_args(parser)
    parser.add_argument("--debug-static-visual-no-padding", action="store_true")
    parser.add_argument("--debug-static-visual-min-pad-tokens", type=int, default=0)
    parser.add_argument("--debug-static-visual-pad-to-multiple", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if str(args.timing_mode) != "standard":
        raise ValueError("generation validation expects --timing-mode standard so visual tower timing stays isolated")
    apply_runtime_env(args)

    baseline_path = Path(args.baseline).expanduser().resolve()
    manifest = load_baseline_manifest(baseline_path)
    baseline_dir = baseline_path if baseline_path.is_dir() else baseline_path.parent
    tensor_dir = baseline_dir / str(manifest.get("tensor_dir", "tensors"))

    model, model_dir, device, dtype = load_model_for_args(args)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    eos_token_id = int(model.config.eos_token_id)
    dataset_dir = resolve_dataset_dir(args.dataset_dir or manifest["build_summary"]["page"]["dataset_dir"])
    inputs = build_inputs_from_manifest(manifest=manifest, model_dir=model_dir, tokenizer=tokenizer, dataset_dir=dataset_dir)
    merge_size = int(load_preprocessor_config(model_dir)["merge_size"])

    fixed_physical_seq_len = max(0, int(args.static_visual_fixed_physical_seq_len))
    input_pairs = list(enumerate(inputs))
    excluded_bucket_rows: list[dict[str, Any]] = []
    if fixed_physical_seq_len:
        eligible_pairs: list[tuple[int, Any]] = []
        for manifest_index, item in input_pairs:
            real_tokens = int(vision_tokens(item))
            if real_tokens > fixed_physical_seq_len:
                excluded_bucket_rows.append(
                    {
                        "manifest_index": int(manifest_index),
                        "id": str(item.entry.get("id")),
                        "layout_label": str(item.entry.get("layout_label", "")),
                        "image_grid_thw": [int(value) for value in item.image_grid_thw.flatten().tolist()],
                        "vision_tokens": int(real_tokens),
                        "reason": "real_visual_tokens_exceed_fixed_physical_seq_len",
                    }
                )
                continue
            eligible_pairs.append((manifest_index, item))
        input_pairs = eligible_pairs
    if int(args.max_items) > 0:
        input_pairs = input_pairs[: int(args.max_items)]

    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, float]] = []
    for compare_idx, (manifest_index, item) in enumerate(input_pairs):
        baseline_item = manifest["items"][manifest_index]
        tensor_path = tensor_dir / str(baseline_item["tensor_file"])
        if sha256_file(tensor_path) != str(baseline_item["tensor_sha256"]):
            raise RuntimeError(f"baseline tensor sha256 mismatch: {tensor_path}")
        baseline_payload = torch.load(tensor_path, map_location="cpu")
        baseline_tensors = baseline_payload["tensors"]
        reference_visual = baseline_tensors["visual_features"].to(device=device, dtype=model.visual.dtype)

        vision_forward, vision_compile = prepare_candidate_vision_forward(
            args=args,
            model=model,
            item=item,
            device=device,
        )

        candidate_visual, visual_timing = compute_visual_tower_only(
            model=model,
            item=item,
            device=device,
            vision_forward=vision_forward,
        )
        real_visual_tokens = int(vision_tokens(item))
        candidate_physical_vision_tokens = int(vision_compile.get("static_visual_physical_seq_len", real_visual_tokens))
        visual_tower_s = float(visual_timing.get("visual_tower_e2e_s", 0.0) or 0.0)
        if visual_tower_s > 0.0:
            visual_timing["visual_tower_effective_tokens_per_s"] = float(real_visual_tokens) / visual_tower_s
            visual_timing["visual_tower_physical_tokens_per_s"] = float(candidate_physical_vision_tokens) / visual_tower_s

        maybe_sync(device)
        ref_prefill_start = time.perf_counter()
        reference_prefill = compute_prefill_state_from_visual_features(
            model=model,
            item=item,
            device=device,
            cache_length=int(args.cache_length),
            visual_features=reference_visual,
        )
        maybe_sync(device)
        ref_prefill_s = float(time.perf_counter() - ref_prefill_start)

        maybe_sync(device)
        cand_prefill_start = time.perf_counter()
        candidate_prefill = compute_prefill_state_from_visual_features(
            model=model,
            item=item,
            device=device,
            cache_length=int(args.cache_length),
            visual_features=candidate_visual,
        )
        maybe_sync(device)
        cand_prefill_s = float(time.perf_counter() - cand_prefill_start)

        maybe_sync(device)
        ref_decode_start = time.perf_counter()
        reference_ids = generate_from_prefill_state(
            model=model,
            prefill=reference_prefill,
            max_new_tokens=int(args.max_new_tokens),
            eos_token_id=eos_token_id,
        )
        maybe_sync(device)
        ref_decode_s = float(time.perf_counter() - ref_decode_start)

        maybe_sync(device)
        cand_decode_start = time.perf_counter()
        candidate_ids = generate_from_prefill_state(
            model=model,
            prefill=candidate_prefill,
            max_new_tokens=int(args.max_new_tokens),
            eos_token_id=eos_token_id,
        )
        maybe_sync(device)
        cand_decode_s = float(time.perf_counter() - cand_decode_start)

        reference_token_list = [int(value) for value in reference_ids[0].detach().cpu().tolist()]
        candidate_token_list = [int(value) for value in candidate_ids[0].detach().cpu().tolist()]
        reference_trimmed = trim_after_eos(reference_token_list, eos_token_id)
        candidate_trimmed = trim_after_eos(candidate_token_list, eos_token_id)
        reference_text, reference_invalid = decode_safely(tokenizer, reference_trimmed)
        candidate_text, candidate_invalid = decode_safely(tokenizer, candidate_trimmed)
        token_mismatch = first_mismatch(reference_trimmed, candidate_trimmed)
        row_timing = {
            **visual_timing,
            "reference_projector_prefill_s": ref_prefill_s,
            "candidate_projector_prefill_s": cand_prefill_s,
            "reference_static_decode_s": ref_decode_s,
            "candidate_static_decode_s": cand_decode_s,
        }
        timing_rows.append(row_timing)

        diffs = {
            "visual_features": diff_stats(candidate_visual.cpu(), baseline_tensors["visual_features"]),
            "image_embeds": diff_stats(candidate_prefill["image_embeds"].cpu(), reference_prefill["image_embeds"].cpu()),
            "prefill_logits": diff_stats(candidate_prefill["prefill_logits"].cpu(), reference_prefill["prefill_logits"].cpu()),
        }
        reference_topk = topk_summary(reference_prefill["prefill_logits"])
        candidate_topk = topk_summary(candidate_prefill["prefill_logits"])
        rows.append(
            {
                "index": int(manifest_index),
                "compare_index": int(compare_idx),
                **input_row(item, merge_size=merge_size),
                "candidate_physical_vision_tokens": int(candidate_physical_vision_tokens),
                "diffs": diffs,
                "reference_topk": reference_topk,
                "candidate_topk": candidate_topk,
                "argmax_match": bool(int(reference_topk["argmax"]) == int(candidate_topk["argmax"])),
                "vision_compile": clean_json(vision_compile),
                "timing_s": {
                    key: stats([float(value)])
                    for key, value in sorted(row_timing.items())
                },
                "generation": {
                    "reference_trimmed_token_count": int(len(reference_trimmed)),
                    "candidate_trimmed_token_count": int(len(candidate_trimmed)),
                    "reference_generated_trimmed": reference_trimmed,
                    "candidate_generated_trimmed": candidate_trimmed,
                    "generated_trimmed_match": bool(reference_trimmed == candidate_trimmed),
                    "first_mismatch": token_mismatch,
                    "reference_invalid_token_ids_sample": reference_invalid,
                    "candidate_invalid_token_ids_sample": candidate_invalid,
                    "invalid_token_count": int(len(reference_invalid) + len(candidate_invalid)),
                    "reference_eos_hit": bool(eos_token_id in reference_trimmed),
                    "candidate_eos_hit": bool(eos_token_id in candidate_trimmed),
                    "reference_length_cap_hit": bool(eos_token_id not in reference_trimmed and len(reference_trimmed) >= int(args.max_new_tokens)),
                    "candidate_length_cap_hit": bool(eos_token_id not in candidate_trimmed and len(candidate_trimmed) >= int(args.max_new_tokens)),
                },
                "texts": {
                    "reference": reference_text,
                    "candidate": candidate_text,
                    "match": bool(reference_text == candidate_text),
                    "ground_truth_sample": str(item.entry.get("ground_truth", ""))[:500],
                },
            }
        )
        print(
            f"GEN_ITEM {compare_idx + 1}/{len(input_pairs)} manifest_index={manifest_index} "
            f"id={item.entry.get('id')} token_match={rows[-1]['generation']['generated_trimmed_match']} "
            f"text_match={rows[-1]['texts']['match']} logits_max_abs={diffs['prefill_logits'].get('max_abs_diff')}",
            file=sys.stderr,
            flush=True,
        )

    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "static_visual_generation_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "name": str(args.candidate_name),
            "dtype": str(dtype),
            "device": str(device),
            "vision_attention": str(args.vision_attention),
            "vision_prompt_fa_layout": str(args.vision_prompt_fa_layout),
            "vision_prompt_fa_mask_sparse_mode": int(args.vision_prompt_fa_mask_sparse_mode),
            "vision_compile_backend": str(args.vision_compile_backend),
            "compile_api": (
                "torchair.inference.cache_compile"
                if bool(args.vision_use_torchair_cache_compile) and str(args.vision_compile_backend) == "torchair"
                else "torch.compile"
                if str(args.vision_compile_backend) != "none"
                else None
            ),
            "uses_torchair_cache_compile": bool(
                args.vision_use_torchair_cache_compile and str(args.vision_compile_backend) == "torchair"
            ),
            "vision_torchair_cache_dir": str(args.vision_torchair_cache_dir),
            "static_visual_fixed_physical_seq_len": int(args.static_visual_fixed_physical_seq_len),
            "static_visual_ln_impl": str(args.static_visual_ln_impl),
            "static_visual_ln_linear_mode": str(args.static_visual_ln_linear_mode),
            "static_visual_promptfa_pad_head_dim_to": int(args.static_visual_promptfa_pad_head_dim_to),
            "max_new_tokens": int(args.max_new_tokens),
            "cache_length": int(args.cache_length),
            "timing_boundary": (
                "visual_tower_e2e_s is device-loaded pixel tensor -> sync -> candidate static visual tower "
                "-> sync. Projector/text prefill/decode are separate correctness timings."
            ),
        },
        "baseline": {
            "path": str(baseline_path),
            "reference_contract": manifest.get("reference_contract", {}),
            "item_count": int(manifest.get("item_count", len(manifest.get("items", [])))),
        },
        "compared_count": int(len(rows)),
        "summary": {
            "argmax_match_count": int(sum(bool(row["argmax_match"]) for row in rows)),
            "generated_trimmed_match_count": int(sum(bool(row["generation"]["generated_trimmed_match"]) for row in rows)),
            "text_match_count": int(sum(bool(row["texts"]["match"]) for row in rows)),
            "invalid_token_count": int(sum(int(row["generation"]["invalid_token_count"]) for row in rows)),
            "length_cap_hit_count": int(
                sum(
                    bool(row["generation"]["reference_length_cap_hit"] or row["generation"]["candidate_length_cap_hit"])
                    for row in rows
                )
            ),
            "bucket_filter": {
                "fixed_physical_seq_len": int(fixed_physical_seq_len),
                "manifest_item_count": int(len(inputs)),
                "eligible_count_before_max_items": int(len(inputs) - len(excluded_bucket_rows)),
                "excluded_count": int(len(excluded_bucket_rows)),
                "selected_count": int(len(rows)),
                "excluded_reason_counts": dict(sorted(Counter(row["reason"] for row in excluded_bucket_rows).items())),
                "first_excluded": clean_json(excluded_bucket_rows[:16]),
            },
            "visual_features": aggregate_diff(rows, "visual_features"),
            "image_embeds": aggregate_diff(rows, "image_embeds"),
            "prefill_logits": aggregate_diff(rows, "prefill_logits"),
            "visual_tower_effective_tokens_per_s": aggregate_timed_token_rate(
                rows,
                token_key="vision_tokens",
                time_key="visual_tower_e2e_s",
            ),
            "visual_tower_physical_tokens_per_s": aggregate_timed_token_rate(
                rows,
                token_key="candidate_physical_vision_tokens",
                time_key="visual_tower_e2e_s",
            ),
            "phase_timing_s": {
                key: stats([float(row[key]) for row in timing_rows if key in row])
                for key in sorted({key for row in timing_rows for key in row})
            },
            "vision_tokens": stats([float(row["vision_tokens"]) for row in rows]),
            "candidate_physical_vision_tokens": stats([float(row["candidate_physical_vision_tokens"]) for row in rows]),
            "generated_trimmed_token_count": stats(
                [float(row["generation"]["candidate_trimmed_token_count"]) for row in rows]
            ),
            "first_generation_mismatches": [
                {
                    "index": int(row["index"]),
                    "id": str(row["id"]),
                    "first_mismatch": row["generation"]["first_mismatch"],
                    "reference_text": str(row["texts"]["reference"])[:300],
                    "candidate_text": str(row["texts"]["candidate"])[:300],
                }
                for row in rows
                if not bool(row["generation"]["generated_trimmed_match"])
            ][:8],
        },
        "items": rows,
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps({"generation_output": str(output_path), "summary": output["summary"]}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
