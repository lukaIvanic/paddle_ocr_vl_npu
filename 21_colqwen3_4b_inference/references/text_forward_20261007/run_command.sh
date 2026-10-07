set -e
cd /workspace/repos/paddle_ocr_vl_npu
git pull --ff-only origin codex/colqwen-warm-forward-profile
source npu-setup
TEXT_RUN_ROOT='tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817'
mkdir -p "$TEXT_RUN_ROOT/cache"
cp -a .runtime_cache/21_colqwen3/prepared/optimized_text_70f3b5326c3bbfd343571e21 "$TEXT_RUN_ROOT/cache/"
/workspace/venvs/colqwen3_hf_py312/bin/python -m unittest discover -s 21_colqwen3_4b_inference -p test_forward_profile_analysis.py -v
for EXECUTION in raw_eager torchair; do
  mkdir -p "$TEXT_RUN_ROOT/$EXECUTION"
  FROZEN_ARGS=()
  if [ "$EXECUTION" = torchair ]; then
    FROZEN_ARGS=(--frozen-inputs "$TEXT_RUN_ROOT/raw_eager/output/text_inputs.pt")
  fi
  TEXT_COMMAND=(/workspace/venvs/colqwen3_hf_py312/bin/python -u 21_colqwen3_4b_inference/profile_warm_text.py
    --model /workspace/models/Ops-Colqwen3-4B
    --anchor tmp/21_colqwen3_4b_inference/replicate_20261007T102343Z_f9bb6837/hr_hf/output/image_00.pt
    --execution "$EXECUTION" --output-dir "$TEXT_RUN_ROOT/$EXECUTION/output"
    --cache-root "$TEXT_RUN_ROOT/cache" --warmups 3 --repeats 30 --profile-steps 3 --metrics pipe memory
    "${FROZEN_ARGS[@]}")
  {
    git rev-parse HEAD
    hostname
    printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf '%q ' "${TEXT_COMMAND[@]}"
    printf '\n'
  } > "$TEXT_RUN_ROOT/$EXECUTION/command.txt"
  printf 'START %s %s\n' "$EXECUTION" "$TEXT_RUN_ROOT"
  set +e
  "${TEXT_COMMAND[@]}" > "$TEXT_RUN_ROOT/$EXECUTION/run.log" 2>&1
  TEXT_STATUS=$?
  set -e
  printf '%s\n' "$TEXT_STATUS" > "$TEXT_RUN_ROOT/$EXECUTION/exit_code.txt"
  if [ "$TEXT_STATUS" != 0 ]; then tail -n 65 "$TEXT_RUN_ROOT/$EXECUTION/run.log"; exit "$TEXT_STATUS"; fi
  /workspace/venvs/colqwen3_hf_py312/bin/python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({k:r[k] for k in ("status","execution","physical_npu","text_tokens","text_input_shapes","before_profile","after_profile","vs_frozen_eager")},indent=2))' "$TEXT_RUN_ROOT/$EXECUTION/output/result.json"
done

/workspace/venvs/colqwen3_hf_py312/bin/python 21_colqwen3_4b_inference/analyze_text_profile.py --run-dir "$TEXT_RUN_ROOT" > "$TEXT_RUN_ROOT/analysis.log"
/workspace/venvs/colqwen3_hf_py312/bin/python - <<'PY'
import json, pathlib
root=pathlib.Path("tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817")
r=json.loads((root/'comparison.json').read_text())
print('TEXT_ONLY_COMPARISON',r['speedup'],r['latency_reduction_percent'])
for mode in ('raw_eager','torchair'):
 print(mode,r[mode]['warm_wall_ms'],r[mode]['profiles']['pipe'])
PY
