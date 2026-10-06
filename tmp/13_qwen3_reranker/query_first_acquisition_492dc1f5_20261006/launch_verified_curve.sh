exec > /tmp/qwen_curve_launcher.log 2>&1
sample_ready=0
for attempt in {1..120}; do
  if [ -f /tmp/qwen-bge-filtered-mixture.incoming.gz ]; then
    sample_checksum=$(sha256sum /tmp/qwen-bge-filtered-mixture.incoming.gz)
    if [ "${sample_checksum%% *}" = "153f4fc8e8297d057563ef87fc0add7e094f63aa644b25490951ff9a58103efd" ]; then
      mv /tmp/qwen-bge-filtered-mixture.incoming.gz /tmp/qwen-bge-filtered-mixture.json.gz
      sample_ready=1
      break
    fi
  fi
  sleep 5
done
if [ "$sample_ready" -ne 1 ]; then echo "Sample verification gate timed out; training was not started"; exit 1; fi
source npu-setup
setup_rc=$?
if [ "$setup_rc" -ne 0 ] || [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then exit 1; fi
set -eu
cd /tmp/qwen-document-first-training-smoke-small
run_dir=tmp/13_qwen3_reranker/query_first_curve_6f381fb3_20261006
mkdir -p "$run_dir"
git rev-parse HEAD > "$run_dir/command.txt"
hostname >> "$run_dir/command.txt"
printf 'physical_npu=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" >> "$run_dir/command.txt"
printf '%s\n' '/usr/local/python3.12.13/bin/python3 -u 13_qwen3_reranker/run_query_first_training_curve.py --model /workspace/models/Qwen3-Reranker-0.6B --dataset /tmp/qwen-bge-filtered-mixture.json.gz --fixture /workspace/results/clef_touche_full/fixture.json --output tmp/13_qwen3_reranker/query_first_curve_6f381fb3_20261006 --mode curve --steps 250 --wall-time-limit 2400' >> "$run_dir/command.txt"
set +e
/usr/local/python3.12.13/bin/python3 -u 13_qwen3_reranker/run_query_first_training_curve.py --model /workspace/models/Qwen3-Reranker-0.6B --dataset /tmp/qwen-bge-filtered-mixture.json.gz --fixture /workspace/results/clef_touche_full/fixture.json --output "$run_dir" --mode curve --steps 250 --wall-time-limit 2400 > "$run_dir/run.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$run_dir/exit_code.txt"
tail -n 25 "$run_dir/run.log"
exit "$rc"
