# 310P UniRec prefill gap analysis from existing summaries

Do not rerun prefill. Analyze the W1/T16 and W8/T8 first-128 summaries that
already completed on this 310P. The repository now contains the exact matched
910B controls and a comparison tool.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Do not run inference, compile graphs, or allocate an NPU.
- Do not write an agent report.
- Return only three short statements: primary sequential cause, scaling loss,
  and the single next experiment.

## Run the comparison

```sh
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"

W1_SUMMARY="${W1_SUMMARY:-$REPO/tmp/12_unirec_0_1b_inference/310p_compiled_layout_prefill_128_w1_w8_7821ad5/w1_t16/output/summary.json}"
W8_SUMMARY="${W8_SUMMARY:-$REPO/tmp/12_unirec_0_1b_inference/310p_compiled_layout_prefill_128_w1_w8_7821ad5/w8_t8/output/summary.json}"

test -x "$PYTHON_BIN"
test -f "$W1_SUMMARY"
test -f "$W8_SUMMARY"
PYTHONPYCACHEPREFIX=/tmp/unirec_gap_pycache \
  "$PYTHON_BIN" -m unittest \
  "$REPO/12_unirec_0_1b_inference/test_analyze_prefill_chip_gap.py"
"$PYTHON_BIN" \
  "$REPO/12_unirec_0_1b_inference/analyze_prefill_chip_gap.py" \
  --npu310-w1 "$W1_SUMMARY" \
  --npu310-w8 "$W8_SUMMARY"
```

If the summary paths differ, set `W1_SUMMARY` and `W8_SUMMARY` to the two
existing files. Do not rerun the benchmark.

## Analyze the result

Use the printed ratios and inspect the same two JSON files. Respect these
measurement rules:

- W1 stage totals are the primary causal comparison because they do not mix
  eight concurrent workers.
- `prefill_including_d2h` already contains `d2h_substage`; never add them.
- CPU detail timers can overlap inside the thread pool; use `cpu_crop` wall.
- W8 worker stage sums overlap in wall time and include NPU/process contention.
  Use them to identify scaling symptoms, not as an additive critical path.
- The tool rejects a workload mismatch before calculating chip ratios.

Return exactly three short statements in this form:

```text
Primary W1 cause: <stage>, <ratio>x slower, explaining <share>% of the producer gap.
Scaling loss: 310P scales <x>x versus 910B <x>x; this adds a <x>x penalty at W8.
Next experiment: <one focused experiment and why it distinguishes the leading hypotheses>.
```

Choose only one next experiment. Prefer an existing isolated lab or a W2/W4
reuse-cache run. Do not propose another full benchmark or unrelated decode work.
