#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ARCHIVE="$SCRIPT_DIR/references/unirec_full1651_910b_470d8a6_text_outputs.tar.gz"
CDM_RUNNER="$REPO/09_persistent_page_engine/scripts/run_cdm_from_matched_formulas.py"
FINGERPRINTER="$SCRIPT_DIR/fingerprint_unirec_cdm_runtime.py"
ANALYZER="$SCRIPT_DIR/analyze_unirec_cdm_same_host.py"
REFERENCE_FINGERPRINT="$SCRIPT_DIR/references/unirec_cdm_runtime_910b2_20260814.json"
EVAL_ENV="$REPO/09_persistent_page_engine/scripts/omnidocbench_eval_env.sh"
EVAL_SETUP="$REPO/09_persistent_page_engine/scripts/setup_omnidocbench_eval_runtime.sh"
EVALUATOR_COMMIT=2b161d010d2e3aff77a0edef359ea3a6411d23cd
TEXLIVE_INSTALLER_SHA256=311df9f1477fd90c520159d1feddc2d6270f010d8349d1f6bdb9461a93b48a5c

absolute_executable_path() {
  local value="$1"
  if [[ "$value" != */* ]]; then command -v "$value"; return; fi
  printf '%s/%s\n' "$(cd "$(dirname "$value")" && pwd -P)" "$(basename "$value")"
}

resolve() {
  : "${RUN_ROOT:?set the completed full-1651 310P run root}"
  : "${EVAL_PYTHON:?set the same CDM-capable Python used by the completed run}"
  : "${EVALUATOR_ROOT:?set the same OmniDocBench evaluator root used by the completed run}"
  : "${CDM_WORKERS:=64}"
  : "${REPAIR_RUNTIME:=0}"
  if [[ "$REPAIR_RUNTIME" != 0 && "$REPAIR_RUNTIME" != 1 ]]; then
    printf 'INVALID_REPAIR_RUNTIME=%s\n' "$REPAIR_RUNTIME" >&2
    exit 1
  fi
  RUN_ROOT="$(readlink -f "$RUN_ROOT")"
  EVAL_PYTHON="$(absolute_executable_path "$EVAL_PYTHON")"
  EVALUATOR_ROOT="$(readlink -f "$EVALUATOR_ROOT")"

  export OMNIDOCBENCH_EVAL_PYTHON="$EVAL_PYTHON"
  export OMNIDOCBENCH_EVALUATOR_ROOT="$EVALUATOR_ROOT"
  # shellcheck source=../09_persistent_page_engine/scripts/omnidocbench_eval_env.sh
  source "$EVAL_ENV"
  EVAL_PYTHON="$OMNIDOCBENCH_EVAL_PYTHON"
  EVALUATOR_ROOT="$OMNIDOCBENCH_EVALUATOR_ROOT"

  test -x "$EVAL_PYTHON"
  test -f "$EVALUATOR_ROOT/pdf_validation.py"
  test "$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)" = "$EVALUATOR_COMMIT"
  test -f "$RUN_ROOT/evaluation_image_tags_stripped/cdm/result/predictions_quick_match_cdm_result.json"
  test -f "$RUN_ROOT/evaluation_image_tags_stripped/work/result/predictions_quick_match_display_formula_result.json"
  test -s "$ARCHIVE" && test -s "$REFERENCE_FINGERPRINT"
}

runtime_preflight() {
  PYTHONPATH="$(dirname "$CDM_RUNNER")" "$EVAL_PYTHON" -c \
    'from run_cdm_from_matched_formulas import _configure_cdm_runtime; print(_configure_cdm_runtime())'
}

select_texlive_mirror() {
  local root="$1"
  local race_root="$root/texlive_mirror_race"
  local cache="$REPO/.runtime_cache/omnidocbench_eval/tools/cache/downloads"
  local final_installer="$cache/install-tl-unx-2025.tar.gz"
  local -a mirrors=(
    "https://ftp.tu-chemnitz.de/pub/tug/historic/systems/texlive/2025/tlnet-final"
    "https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final"
  )
  mkdir -p "$race_root" "$cache"

  if ! command -v wget >/dev/null; then
    export OMNIDOCBENCH_TEXLIVE_REPOSITORY_URL="${mirrors[0]}"
    printf 'CDM_RUNTIME_MIRROR_RACE_SKIPPED reason=no_wget selected=%s\n' \
      "$OMNIDOCBENCH_TEXLIVE_REPOSITORY_URL"
    return 0
  fi

  printf 'CDM_RUNTIME_MIRROR_RACE_BEGIN candidates=%s,%s\n' \
    "${mirrors[0]}" "${mirrors[1]}"
  local -a pids=()
  local index
  for index in 0 1; do
    (
      local started_ns ended_ns elapsed_ms archive
      started_ns="$(date +%s%N)"
      archive="$race_root/candidate_${index}.tar.gz"
      wget --timeout=20 --tries=2 --progress=dot:giga \
        -O "$archive.part" "${mirrors[$index]}/install-tl-unx.tar.gz"
      printf '%s  %s\n' "$TEXLIVE_INSTALLER_SHA256" "$archive.part" | sha256sum -c -
      mv "$archive.part" "$archive"
      ended_ns="$(date +%s%N)"
      elapsed_ms="$(( (ended_ns - started_ns) / 1000000 ))"
      printf '%s\t%s\n' "$elapsed_ms" "${mirrors[$index]}" \
        >"$race_root/candidate_${index}.result"
    ) >"$race_root/candidate_${index}.log" 2>&1 &
    pids+=("$!")
  done

  local selected_index="" selected_url="" selected_ms=""
  local race_started="$SECONDS"
  while [[ -z "$selected_index" ]]; do
    for index in 0 1; do
      if [[ -s "$race_root/candidate_${index}.result" ]]; then
        IFS=$'\t' read -r selected_ms selected_url \
          <"$race_root/candidate_${index}.result"
        selected_index="$index"
        break
      fi
    done
    [[ -n "$selected_index" ]] && break
    local alive=0
    for index in 0 1; do
      kill -0 "${pids[$index]}" 2>/dev/null && alive=1
    done
    if [[ "$alive" == 0 ]]; then
      printf 'CDM_RUNTIME_MIRROR_RACE_FAIL logs=%s\n' "$race_root" >&2
      return 1
    fi
    local bytes0=0 bytes1=0 elapsed rate0 rate1
    [[ -f "$race_root/candidate_0.tar.gz.part" ]] && \
      bytes0="$(wc -c <"$race_root/candidate_0.tar.gz.part")"
    [[ -f "$race_root/candidate_1.tar.gz.part" ]] && \
      bytes1="$(wc -c <"$race_root/candidate_1.tar.gz.part")"
    elapsed="$((SECONDS - race_started))"
    (( elapsed > 0 )) || elapsed=1
    rate0="$((bytes0 / elapsed))"
    rate1="$((bytes1 / elapsed))"
    printf 'CDM_RUNTIME_MIRROR_PROGRESS elapsed_s=%s bytes=%s,%s avg_Bps=%s,%s\n' \
      "$elapsed" "$bytes0" "$bytes1" "$rate0" "$rate1"
    sleep 2
  done

  for index in 0 1; do
    if [[ "$index" != "$selected_index" ]]; then
      kill "${pids[$index]}" 2>/dev/null || true
    fi
    wait "${pids[$index]}" 2>/dev/null || true
  done
  cp "$race_root/candidate_${selected_index}.tar.gz" "$final_installer"
  export OMNIDOCBENCH_TEXLIVE_REPOSITORY_URL="$selected_url"
  printf 'CDM_RUNTIME_MIRROR_SELECTED index=%s elapsed_ms=%s url=%s installer=%s\n' \
    "$selected_index" "$selected_ms" "$selected_url" "$final_installer"
}

repair_runtime() {
  local root="$1"
  local repair_log="$root/runtime_repair.log"
  local status_file="$root/runtime_repair_exit_code.txt"
  local started="$SECONDS"

  export OMNIDOCBENCH_WORKSPACE_ROOT="$(dirname "$REPO")"
  export OMNIDOCBENCH_EVAL_TOOLS_ROOT="$REPO/.runtime_cache/omnidocbench_eval/tools"
  export OMNIDOCBENCH_EVAL_PYTHON="$EVAL_PYTHON"
  export OMNIDOCBENCH_EVALUATOR_ROOT="$EVALUATOR_ROOT"
  export OMNIDOCBENCH_BUILD_JOBS="${OMNIDOCBENCH_BUILD_JOBS:-16}"
  select_texlive_mirror "$root"

  # Avoid apt and an ImageMagick rebuild when the host already has the exact
  # official binaries. In the common failure, only TeX Live 2025 is absent.
  local system_magick="" system_gs=""
  system_magick="$(command -v magick 2>/dev/null || true)"
  system_gs="$(command -v gs 2>/dev/null || true)"
  if [[ -n "$system_magick" ]] && \
     "$system_magick" --version 2>/dev/null | head -n 1 | \
       grep -q 'ImageMagick 7.1.1-47'; then
    export OMNIDOCBENCH_IMAGEMAGICK_ROOT
    OMNIDOCBENCH_IMAGEMAGICK_ROOT="$(cd "$(dirname "$system_magick")/.." && pwd -P)"
  fi
  if [[ -n "$system_magick" && -n "$system_gs" ]] && \
     "$system_magick" --version 2>/dev/null | head -n 1 | \
       grep -q 'ImageMagick 7.1.1-47' && \
     [[ "$("$system_gs" --version 2>/dev/null)" == 9.55.0 ]] && \
     command -v git >/dev/null && command -v perl >/dev/null && \
     command -v tar >/dev/null && command -v wget >/dev/null; then
    export OMNIDOCBENCH_SKIP_APT=1
  else
    export OMNIDOCBENCH_SKIP_APT=0
  fi

  printf 'CDM_RUNTIME_REPAIR_BEGIN skip_apt=%s imagemagick_root=%s jobs=%s timeout_s=1200\n' \
    "$OMNIDOCBENCH_SKIP_APT" "${OMNIDOCBENCH_IMAGEMAGICK_ROOT:-build}" \
    "$OMNIDOCBENCH_BUILD_JOBS"
  : >"$repair_log"
  rm -f "$status_file"
  (
    set +e
    set -o pipefail
    timeout --signal=TERM --kill-after=30s 1200 \
      bash "$EVAL_SETUP" 2>&1 | tee "$repair_log"
    printf '%s\n' "${PIPESTATUS[0]}" >"$status_file"
  ) &
  local repair_pid="$!"
  (
    local previous_bytes=0 idle_s=0
    while sleep 15; do
      kill -0 "$repair_pid" 2>/dev/null || exit 0
      local bytes rate stage last
      bytes="$(wc -c <"$repair_log" 2>/dev/null || printf 0)"
      rate="$(( (bytes - previous_bytes) / 15 ))"
      if (( bytes == previous_bytes )); then idle_s="$((idle_s + 15))"; else idle_s=0; fi
      previous_bytes="$bytes"
      stage="$(grep -E '^\[[1-5]/5\]' "$repair_log" 2>/dev/null | tail -n 1 || true)"
      [[ -n "$stage" ]] || stage=startup
      last="$(tail -n 1 "$repair_log" 2>/dev/null | tr '\r\n' ' ' | tr -s ' ' | cut -c1-200)"
      printf 'CDM_RUNTIME_REPAIR_PROGRESS elapsed_s=%s stage=%q log_bytes=%s log_rate_Bps=%s idle_s=%s last=%q\n' \
        "$((SECONDS - started))" "$stage" "$bytes" "$rate" "$idle_s" "$last"
    done
  ) &
  local heartbeat_pid="$!"
  wait "$repair_pid" || true
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  local status
  status="$(cat "$status_file" 2>/dev/null || printf 125)"
  if [[ "$status" != 0 ]]; then
    printf 'CDM_RUNTIME_REPAIR_FAIL exit=%s wall_s=%s log=%s\n' \
      "$status" "$((SECONDS - started))" "$repair_log" >&2
    return "$status"
  fi

  # Re-source the paths created by setup, then require the exact canonical
  # runtime before any full CDM pass begins.
  unset OMNIDOCBENCH_TEXLIVE_ROOT OMNIDOCBENCH_TEXLIVE_BIN \
    OMNIDOCBENCH_PDFLATEX OMNIDOCBENCH_KPSEWHICH CDM_TEXLIVE_ROOT \
    CDM_TEXLIVE_BIN CDM_PDFLATEX CDM_KPSEWHICH
  # shellcheck source=../09_persistent_page_engine/scripts/omnidocbench_eval_env.sh
  source "$EVAL_ENV"
  runtime_preflight
  printf 'CDM_RUNTIME_REPAIR_PASS wall_s=%s log=%s\n' \
    "$((SECONDS - started))" "$repair_log"
}

ensure_runtime() {
  local root="$1"
  local preflight_log="$root/runtime_preflight_initial.log"
  if runtime_preflight >"$preflight_log" 2>&1; then
    cat "$preflight_log"
    printf 'CDM_RUNTIME_REPAIR_NOT_NEEDED\n'
    return 0
  fi
  cat "$preflight_log" >&2
  if [[ "$REPAIR_RUNTIME" != 1 ]]; then
    printf 'CDM_RUNTIME_MISSING set_REPAIR_RUNTIME=1\n' >&2
    return 1
  fi
  repair_runtime "$root"
}

worker() {
  local root="$1"
  local canonical_evaluator="$root/evaluator_clean"
  mkdir -p "$root/inputs"
  # Use the committed evaluator source without modifying the agent's dirty
  # checkout. A local clone is fast and excludes its unstaged mathcolor patch.
  git clone --quiet --no-checkout "$EVALUATOR_ROOT" "$canonical_evaluator"
  git -C "$canonical_evaluator" checkout --quiet --detach "$EVALUATOR_COMMIT"
  test -z "$(git -C "$canonical_evaluator" status --porcelain=v1 --untracked-files=all)"
  EVALUATOR_ROOT="$canonical_evaluator"
  export OMNIDOCBENCH_EVALUATOR_ROOT="$canonical_evaluator"
  ensure_runtime "$root"
  tar -xOf "$ARCHIVE" \
    evaluation_image_tags_stripped/cdm/result/predictions_quick_match_cdm_result.json \
    >"$root/inputs/reference_original_cdm_result.json"
  tar -xOf "$ARCHIVE" \
    evaluation_image_tags_stripped/work/result/predictions_quick_match_display_formula_result.json \
    >"$root/inputs/reference_matched_formulas.json"

  "$EVAL_PYTHON" "$FINGERPRINTER" \
    --evaluator-root "$canonical_evaluator" --output "$root/candidate_runtime_fingerprint.json"

  printf '[same-host] candidate CDM replay begin workers=%s\n' "$CDM_WORKERS"
  PYTHONUNBUFFERED=1 "$EVAL_PYTHON" "$CDM_RUNNER" \
    --input "$RUN_ROOT/evaluation_image_tags_stripped/work/result/predictions_quick_match_display_formula_result.json" \
    --output-dir "$root/candidate_recheck" --evaluator-root "$canonical_evaluator" \
    --workers "$CDM_WORKERS" >"$root/candidate_recheck.log" 2>&1
  printf '[same-host] candidate CDM replay done\n'

  printf '[same-host] 910B output CDM replay on 310P begin workers=%s\n' "$CDM_WORKERS"
  PYTHONUNBUFFERED=1 "$EVAL_PYTHON" "$CDM_RUNNER" \
    --input "$root/inputs/reference_matched_formulas.json" \
    --output-dir "$root/reference_recheck" --evaluator-root "$canonical_evaluator" \
    --workers "$CDM_WORKERS" >"$root/reference_recheck.log" 2>&1
  printf '[same-host] 910B output CDM replay on 310P done\n'

  "$EVAL_PYTHON" "$ANALYZER" \
    --reference-original "$root/inputs/reference_original_cdm_result.json" \
    --candidate-original "$RUN_ROOT/evaluation_image_tags_stripped/cdm/result/predictions_quick_match_cdm_result.json" \
    --reference-recheck "$root/reference_recheck/result/predictions_quick_match_cdm_result.json" \
    --candidate-recheck "$root/candidate_recheck/result/predictions_quick_match_cdm_result.json" \
    --reference-fingerprint "$REFERENCE_FINGERPRINT" \
    --candidate-fingerprint "$root/candidate_runtime_fingerprint.json" \
    --output-dir "$root"
  cat "$root/report.md"
}

entry() {
  local root="$1" status=0
  set +e
  (
    set -e
    worker "$root"
  )
  status="$?"
  set -e
  printf '%s\n' "$status" >"$root/exit_code.txt"
  exit "$status"
}

launch() {
  resolve
  local short timestamp root
  short="$(git -C "$REPO" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  root="${AUDIT_ROOT:-$RUN_ROOT/cdm_same_host_${short}_${timestamp}}"
  test ! -e "$root"
  mkdir -p "$root"
  nohup env PYTHONUNBUFFERED=1 RUN_ROOT="$RUN_ROOT" EVAL_PYTHON="$EVAL_PYTHON" \
    EVALUATOR_ROOT="$EVALUATOR_ROOT" CDM_WORKERS="$CDM_WORKERS" \
    REPAIR_RUNTIME="$REPAIR_RUNTIME" \
    OMNIDOCBENCH_EVAL_TOOLS_ROOT="$OMNIDOCBENCH_EVAL_TOOLS_ROOT" \
    OMNIDOCBENCH_TOOL_ROOT="$OMNIDOCBENCH_TOOL_ROOT" \
    CDM_PDFLATEX="$CDM_PDFLATEX" CDM_KPSEWHICH="$CDM_KPSEWHICH" \
    "$0" worker "$root" >"$root/run.log" 2>&1 &
  printf '%s\n' "$!" >"$root/pid.txt"
  printf 'AUDIT_ROOT=%s\nRUN_LOG=%s\nPID=%s\n' "$root" "$root/run.log" "$(cat "$root/pid.txt")"
}

if [[ "${1:-}" == worker ]]; then
  resolve
  entry "$2"
else
  launch
fi
