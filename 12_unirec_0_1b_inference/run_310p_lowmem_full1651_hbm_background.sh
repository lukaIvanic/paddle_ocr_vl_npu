#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LOWMEM_RUNNER="$SCRIPT_DIR/run_low_memory_unirec.py"
MEMORY_SAMPLER="$SCRIPT_DIR/run_with_process_tree_memory.py"
CACHE_LOCATOR="$SCRIPT_DIR/locate_unirec_production_caches.py"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then
    command -v "$value"
    return
  fi
  local directory basename
  directory="$(dirname "$value")"
  basename="$(basename "$value")"
  printf '%s/%s\n' "$(cd "$directory" && pwd -P)" "$basename"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 310P venv python_nosym}"
  : "${MODEL:?export the OpenDoc unirec-0.1b model directory}"
  : "${LAYOUT_MODEL:?export the PP-DocLayoutV2 model directory}"
  : "${OPENOCR_ROOT:?export the OpenOCR checkout root}"
  : "${IMAGES_DIR:?export the OmniDocBench v1.6 images directory}"
  : "${COMPILE_CACHE:?export the passed current-source K20 cache parent}"
  : "${LAYOUT_CACHE_ROOT:?export the passed compiled-FP32 B2 layout cache}"
  : "${DECODE_CACHE_PARENT:?export the passed B128 C1320 S2048 decode cache parent}"
  : "${CANONICAL_TRACE:?export the accepted 310P full-1651 recognition trace}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device from 0-3}"
  : "${CPUSET:=0-63}"
  : "${NPU_HBM_INTERVAL_MS:=2000}"
  : "${ALLOW_DECODE_CACHE_BUILD:=0}"

  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  MODEL="$(readlink -f "$MODEL")"
  LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  COMPILE_CACHE="$(readlink -f "$COMPILE_CACHE")"
  LAYOUT_CACHE_ROOT="$(readlink -f "$LAYOUT_CACHE_ROOT")"
  mkdir -p "$DECODE_CACHE_PARENT"
  DECODE_CACHE_PARENT="$(readlink -f "$DECODE_CACHE_PARENT")"
  CANONICAL_TRACE="$(readlink -f "$CANONICAL_TRACE")"

  test "$(basename "$PYTHON_BIN")" = python_nosym
  test -x "$PYTHON_BIN"
  test -f "$MODEL/model.pth"
  test -d "$LAYOUT_MODEL"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -d "$COMPILE_CACHE"
  test -d "$LAYOUT_CACHE_ROOT"
  test -d "$DECODE_CACHE_PARENT"
  test -s "$CANONICAL_TRACE"
  test -f "$LOWMEM_RUNNER"
  test -f "$MEMORY_SAMPLER"
  test -f "$CACHE_LOCATOR"
  [[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-3]$ ]]
  [[ "$NPU_HBM_INTERVAL_MS" =~ ^[1-9][0-9]*$ ]]
  [[ "$ALLOW_DECODE_CACHE_BUILD" =~ ^[01]$ ]]
  taskset -c "$CPUSET" true

  export PYTHON_BIN MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR
  export COMPILE_CACHE LAYOUT_CACHE_ROOT DECODE_CACHE_PARENT CANONICAL_TRACE
  export ASCEND_RT_VISIBLE_DEVICES CPUSET NPU_HBM_INTERVAL_MS
  export ALLOW_DECODE_CACHE_BUILD
  export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$DECODE_CACHE_PARENT"
}

decode_shape_path() {
  printf '%s/%s/%s\n' \
    "$DECODE_CACHE_PARENT" \
    decode_weight_nz_lmhead57344_semantic56371 \
    decode_selfkv2048_cross1320_increfa_all_b128_wnz
}

decode_cache_counts() {
  local decode_shape="$1"
  local module_count=0 om_count=0
  if [[ -d "$decode_shape" ]]; then
    module_count="$(find "$decode_shape" -name compiled_module | wc -l)"
    om_count="$(find "$decode_shape" -type f -name '*.om' | wc -l)"
  fi
  printf '%s %s\n' "$module_count" "$om_count"
}

vision_layout_om_inventory() {
  local output="$1"
  {
    find "$COMPILE_CACHE" -type f -name '*.om' -printf 'vision %p %s %T@\n'
    find "$LAYOUT_CACHE_ROOT" -type f -name '*.om' -printf 'layout %p %s %T@\n'
  } | sort -u >"$output"
}

om_inventory() {
  local output="$1"
  {
    find "$COMPILE_CACHE" -type f -name '*.om' -printf 'vision %p %s %T@\n'
    find "$LAYOUT_CACHE_ROOT" -type f -name '*.om' -printf 'layout %p %s %T@\n'
    find "$DECODE_CACHE_PARENT" -type f -name '*.om' -printf 'decode %p %s %T@\n'
  } | sort -u >"$output"
}

preflight() {
  local run_root="$1"
  "$PYTHON_BIN" -c 'import kornia_rs, torch, torch_npu; print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")'
  npu-smi info >"$run_root/npu_smi_before.txt"
  PHYSICAL_NPU="$ASCEND_RT_VISIBLE_DEVICES" SAMPLER="$MEMORY_SAMPLER" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
from pathlib import Path

sampler = Path(os.environ["SAMPLER"])
sys.path.insert(0, str(sampler.parent))
from run_with_process_tree_memory import query_npu_hbm

row = query_npu_hbm("npu-smi", int(os.environ["PHYSICAL_NPU"]), 10.0)
print(
    "UNIREC_310P_HBM_PREFLIGHT "
    + json.dumps(
        {"used_mb": row["used_mb"], "total_mb": row["total_mb"]},
        sort_keys=True,
    )
)
PY

  "$PYTHON_BIN" "$CACHE_LOCATOR" \
    --search-root "$COMPILE_CACHE" \
    --search-root "$LAYOUT_CACHE_ROOT" \
    --output "$run_root/cache_locator.json" \
    | tee "$run_root/cache_locator.log"

  local selected_vision selected_layout
  selected_vision="$(
    sed -n 's/^UNIREC_K20_COMPILE_CACHE=//p' \
      "$run_root/cache_locator.log" | tail -n 1
  )"
  selected_layout="$(
    sed -n 's/^UNIREC_FP32_B2_LAYOUT_CACHE=//p' \
      "$run_root/cache_locator.log" | tail -n 1
  )"
  test "$(readlink -f "$selected_vision")" = "$COMPILE_CACHE"
  test "$(readlink -f "$selected_layout")" = "$LAYOUT_CACHE_ROOT"

  local decode_shape decode_module_count decode_om_count decode_cache_mode
  decode_shape="$(decode_shape_path)"
  read -r decode_module_count decode_om_count < <(
    decode_cache_counts "$decode_shape"
  )
  if [[ "$decode_module_count" == 0 && "$decode_om_count" == 0 ]]; then
    test "$ALLOW_DECODE_CACHE_BUILD" = 1
    decode_cache_mode=cold_real_arena_build
  elif [[ "$decode_module_count" == 1 && "$decode_om_count" -ge 1 ]]; then
    decode_cache_mode=hot_reuse
  else
    printf 'UNIREC_310P_DECODE_CACHE_INCOMPLETE path=%s compiled_modules=%s oms=%s\n' \
      "$decode_shape" "$decode_module_count" "$decode_om_count" >&2
    return 1
  fi
  export DECODE_CACHE_MODE="$decode_cache_mode"
  printf 'UNIREC_310P_DECODE_CACHE_PREFLIGHT path=%s mode=%s compiled_modules=%s oms=%s\n' \
    "$decode_shape" "$decode_cache_mode" "$decode_module_count" "$decode_om_count"

  {
    printf 'project_commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'physical_npu=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'python=%s\n' "$PYTHON_BIN"
    printf 'cpuset=%s\n' "$CPUSET"
    printf 'model=%s\n' "$MODEL"
    printf 'layout_model=%s\n' "$LAYOUT_MODEL"
    printf 'images=%s\n' "$IMAGES_DIR"
    printf 'vision_cache=%s\n' "$COMPILE_CACHE"
    printf 'layout_cache=%s\n' "$LAYOUT_CACHE_ROOT"
    printf 'decode_cache_parent=%s\n' "$DECODE_CACHE_PARENT"
    printf 'decode_cache_mode=%s\n' "$DECODE_CACHE_MODE"
    printf 'canonical_trace=%s\n' "$CANONICAL_TRACE"
    printf 'hbm_interval_ms=%s\n' "$NPU_HBM_INTERVAL_MS"
    taskset -pc $$
    df -h /dev/shm
    grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
    if [[ -f "${ASCEND_HOME_PATH:-}/opp/version.info" ]]; then
      cat "$ASCEND_HOME_PATH/opp/version.info"
    fi
    "$PYTHON_BIN" -c 'import torch, torch_npu; print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")'
  } >"$run_root/preflight.txt"
}

run_real_decode_cache_lane() {
  local run_root="$1"
  local label="$2"
  local knowledge_processes="$3"
  local lane_root="$run_root/$label"
  mkdir -p "$lane_root"
  local command=(
    taskset -c "$CPUSET"
    "$PYTHON_BIN" "$LOWMEM_RUNNER"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --input "$IMAGES_DIR"
    --output-dir "$lane_root/output"
    --spool-dir "$lane_root/spool"
    --layout-cache "$LAYOUT_CACHE_ROOT"
    --vision-cache "$COMPILE_CACHE"
    --decode-cache-parent "$DECODE_CACHE_PARENT"
    --device npu:0
    --offset 0
    --limit 32
    --workers 4
    --recognition-threads 8
    --recognition-resize-chunk-size 0
    --layout-lanes 1
    --layout-batch-size 2
    --layout-threshold 0.5
    --vision-bucket-preset 310p_k20_l4
    --vision-lanes 4
    --vision-same-key-shards 1
    --vision-sharded-key-count 0
    --recognition-schedule streaming
    --vision-tall-fallback eager
    --decode-batch-size 128
    --cross-cache-length 1320
    --self-cache-length 2048
    --max-length 2048
    --ready-queue-size 128
    --progress-every 8
    --defer-output-write
  )
  printf '%q ' "${command[@]}" >"$lane_root/command.sh"
  printf '\n' >>"$lane_root/command.sh"
  printf 'UNIREC_310P_REAL_DECODE_CACHE_LANE_BEGIN label=%s knowledge_processes=%s\n' \
    "$label" "$knowledge_processes"
  CANN_KNOWLEDGE_BANK_PROCESS_NUM="$knowledge_processes" \
    "${command[@]}" 2>&1 | tee "$lane_root/run.log"
  printf 'UNIREC_310P_REAL_DECODE_CACHE_LANE_END label=%s\n' "$label"
}

prepare_real_decode_cache() {
  local run_root="$1"
  if [[ "$DECODE_CACHE_MODE" == hot_reuse ]]; then
    printf 'UNIREC_310P_REAL_DECODE_CACHE_BUILD skipped=already_hot\n'
    return
  fi

  local decode_shape module_count om_count
  decode_shape="$(decode_shape_path)"
  vision_layout_om_inventory "$run_root/vision_layout_before_decode_build.txt"
  run_real_decode_cache_lane "$run_root" decode_cache_cold_build 1
  read -r module_count om_count < <(decode_cache_counts "$decode_shape")
  test "$module_count" = 1
  test "$om_count" -ge 1
  printf 'UNIREC_310P_REAL_DECODE_CACHE_BUILT path=%s compiled_modules=%s oms=%s\n' \
    "$decode_shape" "$module_count" "$om_count"
  om_inventory "$run_root/om_after_decode_build.txt"

  run_real_decode_cache_lane "$run_root" decode_cache_fresh_replay 0
  vision_layout_om_inventory "$run_root/vision_layout_after_decode_replay.txt"
  diff -u \
    "$run_root/vision_layout_before_decode_build.txt" \
    "$run_root/vision_layout_after_decode_replay.txt" \
    >"$run_root/vision_layout_decode_build.diff"
  om_inventory "$run_root/om_after_decode_replay.txt"
  diff -u \
    "$run_root/om_after_decode_build.txt" \
    "$run_root/om_after_decode_replay.txt" \
    >"$run_root/decode_fresh_replay_om.diff"
  "$PYTHON_BIN" - \
    "$run_root/decode_cache_cold_build/output/recognition_trace.jsonl" \
    "$run_root/decode_cache_fresh_replay/output/recognition_trace.jsonl" <<'PY'
import json
import sys

def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return {
        str(row["request_id"]): (str(row["text"]), tuple(row["generated_ids"]))
        for row in rows
    }

left = load(sys.argv[1])
right = load(sys.argv[2])
assert left == right, "cold-build and fresh-replay traces differ"
print(f"UNIREC_310P_REAL_DECODE_CACHE_TRACE_PARITY rows={len(left)}")
PY
  printf 'UNIREC_310P_REAL_DECODE_CACHE_FRESH_REPLAY: PASS\n'
}

write_report() {
  local run_root="$1"
  RUN_ROOT="$run_root" REFERENCE_910B="$SCRIPT_DIR/references/unirec_910b_lowmem_full1651_hbm_4fc7311/process_tree_and_hbm.json" \
    "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
memory = json.loads((root / "process_tree_and_hbm.json").read_text())
run = json.loads((root / "output/run_summary.json").read_text())
reference_memory = json.loads(Path(os.environ["REFERENCE_910B"]).read_text())

def trace(path):
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    normalized = {
        str(row["request_id"]): (str(row["text"]), tuple(row["generated_ids"]))
        for row in rows
    }
    if len(normalized) != len(rows):
        raise RuntimeError(f"duplicate request IDs in {path}")
    payload = json.dumps(
        sorted((key, text, ids) for key, (text, ids) in normalized.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return rows, normalized, hashlib.sha256(payload).hexdigest()

reference_rows, reference, reference_sha = trace(os.environ["CANONICAL_TRACE"])
candidate_rows, candidate, candidate_sha = trace(
    root / "output/recognition_trace.jsonl"
)
all_ids_match = set(reference) == set(candidate)
mismatches = []
for request_id in sorted(set(reference) | set(candidate)):
    if reference.get(request_id) != candidate.get(request_id):
        mismatches.append(request_id)

hbm = memory["npu_hbm"]
assert memory["exit_code"] == 0
assert not hbm["errors"]
assert hbm["sample_count"] > 0
assert run["status"] == "pass"
assert (run["page_count"], run["crop_count"]) == (1651, 32110)
assert run["settings"]["workers"] == 4
assert run["settings"]["recognition_threads"] == 8
assert run["settings"]["layout_batch_size"] == 2
assert run["settings"]["vision_bucket_preset"] == "310p_k20_l4"
assert run["settings"]["decode_batch_size"] == 128
assert run["settings"]["cross_cache_length"] == 1320
assert run["settings"]["self_cache_length"] == 2048
assert len(reference_rows) == len(candidate_rows) == 32110
assert all_ids_match
assert not mismatches

report = {
    "schema": "unirec_310p_lowmem_full1651_hbm_report_v1",
    "status": "pass",
    "commit": run["commit"],
    "chip": run["chip"],
    "pages": run["page_count"],
    "crops": run["crop_count"],
    "internal_pipeline_wall_s": run["process_wall_s"],
    "internal_pages_per_s": run["pages_per_s"],
    "external_process_wall_s": memory["wall_s"],
    "external_pages_per_s": run["page_count"] / memory["wall_s"],
    "peak_host_pss_bytes": memory["peak"]["total_pss_bytes"],
    "peak_host_rss_bytes": memory["peak"]["total_rss_bytes"],
    "peak_host_pss_elapsed_s": memory["peak"]["elapsed_s"],
    "host_sample_count": memory["sample_count"],
    "peak_host_processes": sorted(
        memory["peak"]["processes"],
        key=lambda row: row["pss_bytes"],
        reverse=True,
    )[:12],
    "physical_npu": hbm["physical_npu"],
    "hbm_baseline_mb": hbm["baseline"]["used_mb"],
    "hbm_peak_mb": hbm["peak"]["used_mb"],
    "hbm_total_mb": hbm["peak"]["total_mb"],
    "hbm_peak_increase_mb": hbm["peak_increase_from_baseline_mb"],
    "hbm_peak_elapsed_s": hbm["peak"]["elapsed_s"],
    "hbm_sample_count": hbm["sample_count"],
    "hbm_errors": hbm["errors"],
    "hbm_peak_process_rows": [
        line
        for line in hbm["peak"]["raw_npu_smi"].splitlines()
        if line.lstrip().startswith(f"| {hbm['physical_npu']}")
    ],
    "trace_reference_sha256": reference_sha,
    "trace_candidate_sha256": candidate_sha,
    "trace_request_ids_match": all_ids_match,
    "trace_mismatch_count": len(mismatches),
    "first_trace_mismatches": mismatches[:20],
    "timing": {
        "frontend_wall_s": run["frontend_wall_s"],
        "layout_owner_wall_s": run["layout"]["owner_wall_s"],
        "vision_phase_wall_s": run["vision_phase_wall_s"],
        "vision_graph_wall_s": run["vision"]["wall_s"],
        "text_prefill_wall_s": run["text_prefill"]["wall_s"],
        "decode_wall_s": run["decode_wall_s"],
    },
    "decode": {
        "iterations": run["decode"]["decode_iterations"],
        "raw_token_slots": run["decode"]["raw_decode_token_slots"],
        "effective_tokens": run["decode"]["effective_decode_tokens"],
        "decode_s": run["decode"]["decode_s"],
        "raw_tokens_per_s": run["decode"]["raw_decode_tokens_per_s"],
        "effective_tokens_per_s": run["decode"]["effective_decode_tokens_per_s"],
    },
    "reference_910b": {
        "peak_host_pss_bytes": reference_memory["peak"]["total_pss_bytes"],
        "hbm_baseline_mb": reference_memory["npu_hbm"]["baseline"]["used_mb"],
        "hbm_peak_mb": reference_memory["npu_hbm"]["peak"]["used_mb"],
        "hbm_peak_increase_mb": reference_memory["npu_hbm"]["peak_increase_from_baseline_mb"],
    },
}
(root / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
print("UNIREC_310P_LOWMEM_FULL1651_HBM: PASS")
print(json.dumps(report, indent=2))
PY
}

worker_main() {
  local run_root="$1"
  local run_log="$run_root/run.log"
  local status=0 target_pid started
  trap 'status=$?; printf "%s\n" "$status" >"'"$run_root"'/exit_code.txt"; exit "$status"' EXIT

  resolve_inputs
  preflight "$run_root"
  prepare_real_decode_cache "$run_root"
  om_inventory "$run_root/om_before.txt"

  local command=(
    taskset -c "$CPUSET"
    "$PYTHON_BIN" "$MEMORY_SAMPLER"
    --output "$run_root/process_tree_and_hbm.json"
    --interval-ms 50
    --npu-id "$ASCEND_RT_VISIBLE_DEVICES"
    --npu-interval-ms "$NPU_HBM_INTERVAL_MS"
    --
    "$PYTHON_BIN" "$LOWMEM_RUNNER"
    --openocr-root "$OPENOCR_ROOT"
    --model-path "$MODEL"
    --layout-model "$LAYOUT_MODEL"
    --input "$IMAGES_DIR"
    --output-dir "$run_root/output"
    --spool-dir "$run_root/spool"
    --layout-cache "$LAYOUT_CACHE_ROOT"
    --vision-cache "$COMPILE_CACHE"
    --decode-cache-parent "$DECODE_CACHE_PARENT"
    --device npu:0
    --offset 0
    --limit 1651
    --workers 4
    --recognition-threads 8
    --recognition-resize-chunk-size 0
    --layout-lanes 1
    --layout-batch-size 2
    --layout-threshold 0.5
    --vision-bucket-preset 310p_k20_l4
    --vision-lanes 4
    --vision-same-key-shards 1
    --vision-sharded-key-count 0
    --recognition-schedule streaming
    --vision-tall-fallback eager
    --decode-batch-size 128
    --cross-cache-length 1320
    --self-cache-length 2048
    --max-length 2048
    --ready-queue-size 128
    --progress-every 32
    --defer-output-write
  )
  printf '%q ' "${command[@]}" >"$run_root/command.sh"
  printf '\n' >>"$run_root/command.sh"
  export PYTHONUNBUFFERED=1
  export UNIREC_VISION_DIAGNOSTIC_GRAPH_LOG=1

  started="$SECONDS"
  CANN_KNOWLEDGE_BANK_PROCESS_NUM=0 "${command[@]}" &
  target_pid="$!"
  printf '%s\n' "$target_pid" >"$run_root/target_pid.txt"
  taskset -pc "$target_pid" >"$run_root/target_affinity.txt"
  while kill -0 "$target_pid" 2>/dev/null; do
    sleep 15
    printf 'UNIREC_310P_LOWMEM_HEARTBEAT elapsed_s=%s target_pid=%s compiler_processes=%s om_count=%s last_marker=%q\n' \
      "$((SECONDS - started))" "$target_pid" \
      "$(pgrep -af 'atc|ccec|compiler|tbe' | wc -l)" \
      "$(find "$COMPILE_CACHE" "$LAYOUT_CACHE_ROOT" "$DECODE_CACHE_PARENT" -type f -name '*.om' -print | sort -u | wc -l)" \
      "$(grep -E 'UNIREC_LOWMEM_|UNIREC_GRAPH_WARMUP|Traceback|ERROR|recompil' "$run_log" | tail -n 1 || true)"
  done
  set +e
  wait "$target_pid"
  status="$?"
  set -e
  test "$status" -eq 0

  om_inventory "$run_root/om_after.txt"
  if diff -u "$run_root/om_before.txt" "$run_root/om_after.txt" \
      >"$run_root/om_inventory.diff"; then
    printf 'UNIREC_310P_LOWMEM_OM_INVENTORY_UNCHANGED\n'
  else
    printf 'UNIREC_310P_LOWMEM_OM_INVENTORY_CHANGED\n' >&2
    cat "$run_root/om_inventory.diff" >&2
    return 1
  fi
  write_report "$run_root" | tee "$run_root/final_report.txt"
}

worker_entry() {
  local run_root="$1"
  worker_main "$run_root"
}

report_only_entry() {
  : "${PYTHON_BIN:?export an environment Python}"
  : "${CANONICAL_TRACE:?export the accepted full-1651 recognition trace}"
  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  CANONICAL_TRACE="$(readlink -f "$CANONICAL_TRACE")"
  export PYTHON_BIN CANONICAL_TRACE
  write_report "${1:?completed run root is required}"
}

launch_main() {
  resolve_inputs
  local short timestamp run_root run_log
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  run_root="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_lowmem_full1651_hbm_${short}_${timestamp}}"
  run_log="$run_root/run.log"
  mkdir -p "$run_root"
  nohup bash "$0" --worker "$run_root" >"$run_log" 2>&1 </dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$run_root/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\nTAIL_COMMAND=tail -f %q\n' \
    "$run_root" "$run_log" "$pid" "$run_log"
}

if [[ "${1:-}" == --worker ]]; then
  shift
  worker_entry "${1:?run root is required}"
elif [[ "${1:-}" == --report-only ]]; then
  shift
  report_only_entry "${1:?completed run root is required}"
else
  launch_main
fi
