#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

python_bin="/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python"
model="/workspace/models/PaddleOCR-VL-1.6"
cache_dir="$repo_root/.runtime_cache/09_persistent_page_engine_torchair"
compact_vocab="$repo_root/09_persistent_page_engine/presets/table_compact_vocab/b1_verifier_topfreq_16384.json"
decode_optimization="combined_apply_complete_layer_prefetch1_rope_lut"
spec_optimization="combined_apply_spec_prefetch_mrope"
spec_attention="manual_grouped_legal_scaled_masked_softmax_fp16_combined_qkv_post_rope"
cache_length=4096
profile_position=1249
batch_sizes="1,2,4,8"
draft_lengths="7"
warmup=10
repeats=50

if [[ ! -x "$python_bin" ]]; then
  echo "ERROR: production Python is missing: $python_bin" >&2
  exit 1
fi
if [[ ! -d "$model" ]]; then
  echo "ERROR: production model is missing: $model" >&2
  exit 1
fi
if [[ ! -f "$compact_vocab" ]]; then
  echo "ERROR: production compact vocabulary is missing: $compact_vocab" >&2
  exit 1
fi
if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: source npu-setup before running this benchmark" >&2
  exit 1
fi

commit="$(git rev-parse --short=8 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="$repo_root/tmp/09_persistent_page_engine/table_q1_q8_batch_sweep_${commit}_${timestamp}"
output="$output_dir/results.json"
mkdir -p "$output_dir"

compile_arg=()
if [[ "${ALLOW_COMPILE:-0}" == "1" ]]; then
  compile_arg+=(--allow-compile)
fi

export SPEC_VERIFY_ATTENTION="$spec_attention"

printf '%s\n' \
  "TABLE_Q1_Q8_LOCKED_CONTRACT" \
  "model=$model" \
  "dtype=float16" \
  "cache_length=$cache_length" \
  "profile_position=$profile_position" \
  "batch_sizes=$batch_sizes" \
  "query_lengths=1,8" \
  "compact_vocab=$compact_vocab" \
  "decode_optimization=$decode_optimization" \
  "spec_optimization=$spec_optimization" \
  "spec_attention=$spec_attention" \
  "warmup=$warmup" \
  "repeats=$repeats" \
  "allow_compile=${ALLOW_COMPILE:-0}" \
  "output=$output"

"$python_bin" 09_persistent_page_engine/scripts/text_spec_verify_lab.py \
  --model "$model" \
  --cache-dir "$cache_dir" \
  --output "$output" \
  --cache-length "$cache_length" \
  --profile-position "$profile_position" \
  --decode-vocab-token-ids "$compact_vocab" \
  --decode-optimization "$decode_optimization" \
  --spec-optimization "$spec_optimization" \
  --draft-lengths "$draft_lengths" \
  --batch-sizes "$batch_sizes" \
  --warmup "$warmup" \
  --repeats "$repeats" \
  "${compile_arg[@]}"

"$python_bin" -c '
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
expected_batches = [1, 2, 4, 8]
expected_attention = os.environ["SPEC_VERIFY_ATTENTION"]
contract = payload["contract"]
assert payload["status"] == "complete"
assert contract["batch_sizes"] == expected_batches
assert contract["cache_length"] == 4096
assert contract["profile_position"] == 1249
assert contract["draft_lengths"] == [7]
assert contract["decode_optimization"] == "combined_apply_complete_layer_prefetch1_rope_lut"
assert contract["spec_optimization"] == "combined_apply_spec_prefetch_mrope"
assert contract["spec_attention"] == expected_attention
assert payload["setup"]["decode_vocab"]["selected_vocab_size"] == 16384
q1 = payload["decode_q1"]
q8 = payload["spec_verify"]
assert [row["batch_size"] for row in q1] == expected_batches
assert [row["batch_size"] for row in q8] == expected_batches
assert all(row["query_length"] == 1 for row in q1)
assert all(row["query_length"] == 8 for row in q8)
assert all(
    row["batch_vs_b1_decode_target_agreement"]["fraction"] == 1.0
    for row in q1
)
assert all(
    row["batch_vs_b1_spec_target_agreement"]["fraction"] == 1.0
    for row in q8
)
print("TABLE_Q1_Q8_VALIDATED " + json.dumps({
    "output": str(path),
    "q1": [
        {
            "batch": row["batch_size"],
            "median_ms": row["latency_ms"]["median"],
            "iters_s": row["graph_calls_per_s"],
            "physical_tok_s": row["physical_generated_tok_per_s"],
        }
        for row in q1
    ],
    "q8": [
        {
            "batch": row["batch_size"],
            "median_ms": row["latency_ms"]["median"],
            "iters_s": row["graph_calls_per_s"],
            "physical_tok_s": row["physical_verified_tok_per_s"],
        }
        for row in q8
    ],
}, separators=(",", ":")))
' "$output"
