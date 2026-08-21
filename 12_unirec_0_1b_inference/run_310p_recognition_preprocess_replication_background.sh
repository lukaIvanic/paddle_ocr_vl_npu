#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
REFERENCE_DIR="$SCRIPT_DIR/references/unirec_hard32_preprocess_corpus"

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then
    command -v "$value"
    return
  fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

resolve_inputs() {
  : "${PYTHON_BIN:?export the validated 310P venv python_nosym}"
  : "${OPENOCR_ROOT:?export the matching OpenOCR checkout}"
  : "${IMAGES_DIR:?export the OmniDocBench v1.6 images directory}"
  : "${ASCEND_RT_VISIBLE_DEVICES:?select one free physical 310P device, 0-3}"

  PYTHON_BIN="$(absolute_executable_path "$PYTHON_BIN")"
  OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
  IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
  case "$ASCEND_RT_VISIBLE_DEVICES" in
    0|1|2|3) ;;
    *) printf '310P_DEVICE_MUST_BE_0_TO_3=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" >&2; exit 1 ;;
  esac
  [[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]]
  test -x "$PYTHON_BIN"
  test "$(basename "$PYTHON_BIN")" = python_nosym
  test -d "$OPENOCR_ROOT"
  test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
  test -d "$IMAGES_DIR"
  test -f "$REFERENCE_DIR/pages.jsonl.gz"
  test -f "$REFERENCE_DIR/crops.jsonl.gz"
  test -f "$REFERENCE_DIR/summary.json"
  export PYTHON_BIN OPENOCR_ROOT IMAGES_DIR
}

materialize_corpus() {
  local corpus_dir="$1"
  mkdir -p "$corpus_dir"
  "$PYTHON_BIN" - "$REFERENCE_DIR" "$IMAGES_DIR" "$corpus_dir" <<'PY'
import gzip
import json
import shutil
import sys
from pathlib import Path

reference_dir = Path(sys.argv[1])
images_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

page_count = 0
seen_names = set()
with gzip.open(reference_dir / "pages.jsonl.gz", "rt", encoding="utf-8") as source:
    with (output_dir / "pages.jsonl").open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            page = json.loads(line)
            name = Path(page["image_path"]).name
            if name in seen_names:
                raise RuntimeError(f"duplicate image basename: {name}")
            seen_names.add(name)
            rebased = images_dir / name
            if not rebased.is_file():
                raise FileNotFoundError(rebased)
            page["image_path"] = str(rebased)
            target.write(json.dumps(page, ensure_ascii=False, separators=(",", ":")) + "\n")
            page_count += 1

crop_count = 0
with gzip.open(reference_dir / "crops.jsonl.gz", "rt", encoding="utf-8") as source:
    with (output_dir / "crops.jsonl").open("w", encoding="utf-8") as target:
        for line in source:
            if line.strip():
                json.loads(line)
                target.write(line)
                crop_count += 1

shutil.copy2(reference_dir / "summary.json", output_dir / "summary.json")
if page_count != 32 or crop_count != 1564:
    raise RuntimeError(f"corpus mismatch: pages={page_count} crops={crop_count}")
print(f"UNIREC_310P_PREPROCESS_CORPUS pages={page_count} crops={crop_count} dir={output_dir}")
PY
}

run_phase() {
  local name="$1"
  shift
  printf 'UNIREC_310P_PREPROCESS_PHASE_BEGIN phase=%s epoch_s=%s\n' "$name" "$(date +%s)"
  "$@"
  printf 'UNIREC_310P_PREPROCESS_PHASE_END phase=%s epoch_s=%s\n' "$name" "$(date +%s)"
}

worker_main() {
  local run_root="$1"
  resolve_inputs
  local corpus_dir="$run_root/corpus"
  materialize_corpus "$corpus_dir" | tee "$run_root/corpus.log"

  {
    printf 'commit=%s\nphysical_device=%s\npython=%s\nopenocr=%s\nimages=%s\n' \
      "$(git -C "$REPO" rev-parse HEAD)" "$ASCEND_RT_VISIBLE_DEVICES" \
      "$PYTHON_BIN" "$OPENOCR_ROOT" "$IMAGES_DIR"
    "$PYTHON_BIN" - <<'PY'
import os
import platform
import cv2
import numpy
import PIL
import torch
import torch_npu
affinity = sorted(os.sched_getaffinity(0))
print(f"cpu_affinity_count={len(affinity)}")
print(f"cpu_affinity={affinity}")
print(f"platform={platform.platform()}")
print(f"torch={torch.__version__}")
print(f"torch_npu={torch_npu.__version__}")
print(f"numpy={numpy.__version__}")
print(f"pillow={PIL.__version__}")
print(f"opencv={cv2.__version__}")
if len(affinity) < 16:
    raise RuntimeError(f"need at least 16 allowed CPUs, got {len(affinity)}")
PY
    npu-smi info
  } >"$run_root/preflight.log" 2>&1

  local sequential_command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_cpu_recognition_preprocess.py"
    --artifact-dir "$corpus_dir"
    --openocr-root "$OPENOCR_ROOT"
    --rounds 3
    --warmup-crops 64
    --lanes pillow_no_convert_uint8_hwc
  )
  local threaded_command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_cpu_recognition_bucket_pack.py"
    --artifact-dir "$corpus_dir"
    --openocr-root "$OPENOCR_ROOT"
    --workers 16
    --rounds 5
    --warmup-crops 64
    --verify
  )
  local npu_command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_npu_compact_batch_input.py"
    --summary "$corpus_dir/summary.json"
    --device npu:0
    --rounds 3
  )

  {
    printf 'TORCH_DEVICE_BACKEND_AUTOLOAD=0 '
    printf '%q ' "${sequential_command[@]}"
    printf '\nTORCH_DEVICE_BACKEND_AUTOLOAD=0 '
    printf '%q ' "${threaded_command[@]}"
    printf '\n'
    printf '%q ' "${npu_command[@]}"
    printf '\n'
  } >"$run_root/command.sh"

  run_phase sequential env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    "${sequential_command[@]}" 2>&1 | tee "$run_root/sequential.log"
  run_phase threaded env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    "${threaded_command[@]}" 2>&1 | tee "$run_root/threaded.log"
  run_phase npu "${npu_command[@]}" 2>&1 | tee "$run_root/npu.log"

  "$PYTHON_BIN" - \
    "$run_root/sequential.log" "$run_root/threaded.log" "$run_root/npu.log" \
    "$run_root/final_report.json" <<'PY' | tee "$run_root/final_report.txt"
import json
import sys
from pathlib import Path

sequential_path, threaded_path, npu_path, output_path = map(Path, sys.argv[1:])

def prefixed_json(path, prefix):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            rows.append(json.loads(line[len(prefix):]))
    if len(rows) != 1:
        raise RuntimeError(f"expected one {prefix!r} row in {path}, got {len(rows)}")
    return rows[0]

sequential = prefixed_json(sequential_path, "UNIREC_CPU_PREPROCESS_SUMMARY ")
threaded = prefixed_json(threaded_path, "UNIREC_CPU_BUCKET_PACK_SUMMARY ")
npu = prefixed_json(npu_path, "UNIREC_NPU_COMPACT_BATCH_INPUT ")

if sequential["crop_count"] != 1564:
    raise RuntimeError("sequential crop count mismatch")
if sequential["threading"] != {
    "outer_workers": 1,
    "opencv_threads": 1,
    "torch_threads": 1,
    "torch_interop_threads": 1,
}:
    raise RuntimeError(f"sequential threading mismatch: {sequential['threading']}")
original = sequential["lanes"]["pillow_reference"]
compact_one = sequential["lanes"]["pillow_no_convert_uint8_hwc"]
if not compact_one["model_input_fp16_parity"]["all_exact"]:
    raise RuntimeError("compact one-thread model input is not exact")

if threaded["page_count"] != 32 or threaded["crop_count"] != 1564:
    raise RuntimeError("threaded workload mismatch")
if not threaded["verification"]["all_exact"]:
    raise RuntimeError("threaded bucket packing is not exact")
if len(threaded["results"]) != 1 or threaded["results"][0]["workers"] != 16:
    raise RuntimeError("threaded result is not the 16-worker lane")
compact_sixteen = threaded["results"][0]

if npu["total_input_call_count"] != 198 or npu["physical_npu"] not in {"0", "1", "2", "3"}:
    raise RuntimeError("NPU workload or physical device mismatch")
if npu.get("npu_jit_compile") is not False:
    raise RuntimeError("NPU JIT compile was not explicitly disabled")
if not npu["normalization_fp16_parity"]["all_exact"]:
    raise RuntimeError("NPU normalization parity failed")
original_npu = npu["results"]["current_float32_chw"]
compact_npu = npu["results"]["compact_uint8_hwc"]

crop_count = 1564
rows = {
    "original": {
        "cpu_s": original["median_s"],
        "npu_s": original_npu["median_s"],
    },
    "compact_one_thread": {
        "cpu_s": compact_one["median_s"],
        "npu_s": compact_npu["median_s"],
    },
    "compact_sixteen_threads": {
        "cpu_s": compact_sixteen["median_s"],
        "npu_s": compact_npu["median_s"],
    },
}
for row in rows.values():
    row["combined_s"] = row["cpu_s"] + row["npu_s"]
    row["crops_per_s"] = crop_count / row["combined_s"]
baseline = rows["original"]["combined_s"]
for row in rows.values():
    row["speedup_vs_original"] = baseline / row["combined_s"]

reference_910b = {
    "chip": "Ascend910B2",
    "physical_npu": 7,
    "commit": "f13d14d",
    "original": {"cpu_s": 11.443913526833057, "npu_s": 0.33457157714292407},
    "compact_one_thread": {"cpu_s": 5.749039839021862, "npu_s": 0.1027975669130683},
    "compact_sixteen_threads": {"cpu_s": 0.9722642251290381, "npu_s": 0.1027975669130683},
}
for name, row in rows.items():
    reference_total = (
        reference_910b[name]["cpu_s"] + reference_910b[name]["npu_s"]
    )
    row["ratio_vs_910b"] = row["combined_s"] / reference_total

report = {
    "schema": "unirec_310p_recognition_preprocess_replication_v1",
    "status": "pass",
    "crop_count": crop_count,
    "page_count": 32,
    "physical_npu": int(npu["physical_npu"]),
    "rows": rows,
    "reference_910b": reference_910b,
    "parity": {
        "compact_one_thread_model_input_exact": True,
        "compact_sixteen_thread_bucket_output_exact": True,
        "npu_normalization_exact": True,
        "npu_jit_compile_disabled": True,
    },
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
for name, row in rows.items():
    print(
        "UNIREC_310P_RECOGNITION_PREPROCESS_LANE "
        f"lane={name} cpu_s={row['cpu_s']:.6f} npu_s={row['npu_s']:.6f} "
        f"combined_s={row['combined_s']:.6f} crops_per_s={row['crops_per_s']:.3f} "
        f"speedup={row['speedup_vs_original']:.6f} "
        f"ratio_vs_910b={row['ratio_vs_910b']:.6f}"
    )
print("UNIREC_310P_RECOGNITION_PREPROCESS_PARITY: PASS")
print("UNIREC_310P_RECOGNITION_PREPROCESS_REPLICATION: PASS")
PY
}

worker_entry() {
  local run_root="$1" status=0 started="$SECONDS"
  set +e
  (set -e; worker_main "$run_root")
  status=$?
  set -e
  printf '%s\n' "$status" >"$run_root/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$run_root/process_wall_s.txt"
  printf 'UNIREC_310P_RECOGNITION_PREPROCESS_WORKER_END status=%s run_log=%s\n' \
    "$status" "$run_root/run.log"
  exit "$status"
}

launch_main() {
  resolve_inputs
  local short timestamp
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-$REPO/tmp/12_unirec_0_1b_inference/310p_recognition_preprocess_replication_${short}_${timestamp}}"
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
  test ! -e "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHON_BIN="$PYTHON_BIN" OPENOCR_ROOT="$OPENOCR_ROOT" IMAGES_DIR="$IMAGES_DIR" \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    bash "$0" worker "$RUN_ROOT" >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"
  printf 'RUN_ROOT=%s\nRUN_LOG=%s\nPID=%s\nTAIL_COMMAND=tail -f %q\n' \
    "$RUN_ROOT" "$RUN_ROOT/run.log" "$!" "$RUN_ROOT/run.log"
}

if [[ "${1:-}" == worker ]]; then
  worker_entry "$2"
else
  launch_main
fi
