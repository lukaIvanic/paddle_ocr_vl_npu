# Recalculate the completed 310P K10/L1 versus K10/L4-all vision result

## Objective

Do not rerun inference or NPU profiling. Read the two completed W1/T8 trace
artifact sets and produce one concise, arithmetically reconciled comparison.
The report must answer whether slot efficiency, spatial pixel efficiency,
physical padded pixels, graph time, and device-input time improved.

Use this commit or a descendant. Do not edit tracked files on the work server.

## Select the two exact output directories

`BASELINE_OUTPUT` is the completed 310P W1/T8, K10/L1,
one-page-lookahead trace. `CANDIDATE_OUTPUT` is the completed 310P W1/T8,
K10/L4-all, four-page-lookahead trace that measured about 40 seconds.

Each directory must contain all four files:

```text
run_summary.json
prefill_distributions.json
prefill_iterations.jsonl
prefill_pages.jsonl
```

Set absolute paths. Use the existing files; do not copy or regenerate them.

```bash
export BASELINE_OUTPUT=/absolute/path/to/prior/k10_l1/output
export CANDIDATE_OUTPUT=/absolute/path/to/new/k10_l4_all/output
test -f "$BASELINE_OUTPUT/prefill_iterations.jsonl"
test -f "$CANDIDATE_OUTPUT/prefill_iterations.jsonl"
```

## Run the single comparator

Any working Python 3 interpreter is sufficient because this script uses only
the standard library. If using the UniRec venv, preserve its `python_nosym`
path and do not apply `readlink -f` to it.

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
OUTPUT_JSON="$CANDIDATE_OUTPUT/vision_plan_ab_recalculated.json"
OUTPUT_LOG="$CANDIDATE_OUTPUT/vision_plan_ab_recalculated.log"
"$PYTHON_BIN" 12_unirec_0_1b_inference/compare_vision_plan_ab.py \
  --baseline-dir "$BASELINE_OUTPUT" \
  --candidate-dir "$CANDIDATE_OUTPUT" \
  --output "$OUTPUT_JSON" \
  --top-buckets 5 | tee "$OUTPUT_LOG"
printf 'OUTPUT_JSON=%s\nOUTPUT_LOG=%s\n' "$OUTPUT_JSON" "$OUTPUT_LOG"
```

The comparator intentionally permits only these plan changes:

- `vision_page_lookahead`
- `vision_bucket_preset`

It fails if worker count, CPU threads, layout settings, cache lengths, model
execution modes, or page identities differ. Do not weaken this gate. If it
fails, paste the exact mismatch instead of comparing incompatible runs.

Different crop counts are permitted but must be highlighted through
`page_workload_exact=false` and the reported crop/effective-pixel deltas.

## Return only the useful summary

Paste the five line groups emitted by the script:

1. `UNIREC_VISION_PLAN_AB`
2. `UNIREC_VISION_PLAN_HEADLINE`
3. `UNIREC_VISION_PLAN_WORK`
4. `UNIREC_VISION_PLAN_DEVICE`
5. `UNIREC_VISION_PLAN_RECONCILIATION`

Then paste the five candidate `UNIREC_VISION_PLAN_BUCKET` lines. Do not paste
setup inventories, general counts, unrelated CPU stages, or the full JSON.

Finish with three plain-language sentences:

1. Did total physical padded pixels and pixel efficiency improve?
2. Did total graph time (`compiled + eager fallback`) improve, and by how much?
3. What exactly offset the graph saving inside full vision wall time?

Also return the absolute `OUTPUT_JSON` and `OUTPUT_LOG` paths.
