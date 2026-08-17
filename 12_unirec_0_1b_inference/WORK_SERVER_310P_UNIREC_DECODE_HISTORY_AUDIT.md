# 310P UniRec production-decode history audit

## Purpose

Before changing decode again, recover the exact saved 310P production runs and
separate four effects that have been conflated:

1. physical decode-graph speed;
2. cross-KV/self-KV capacities and B64/B128 graph shape;
3. excess iterations caused by length-capped repetition;
4. CPU affinity, workers, ingestion, and scheduler overhead.

This audit is CPU-only.  Do not reserve or initialize an NPU.  Do not rerun
inference.  Do not use remembered terminal summaries as evidence.

## Pull and run

Pull the commit named in Luka's message.  Do not edit tracked files, create a
branch, commit, or push.

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"

AUDIT_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/decode_history_audit_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$AUDIT_ROOT"

PYTHON_BIN="${PYTHON_BIN:?reuse the validated UniRec Python executable}"
"$PYTHON_BIN" 12_unirec_0_1b_inference/audit_unirec_decode_run_history.py \
  --search-root "$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference" \
  --min-pages 512 \
  --output "$AUDIT_ROOT/history.json" \
  | tee "$AUDIT_ROOT/history.txt"

printf 'AUDIT_ROOT=%s\n' "$AUDIT_ROOT"
```

This recursively finds saved `run_summary.json` and decode-replay `clean.json`
files.  For large production runs it recomputes token-length, cap, and
single-token-repetition statistics from `recognition_trace.jsonl`.  It does not
print OCR text or token values.

If the known full runs were archived outside the repository's `tmp` directory,
do not copy or modify them.  Run the same script again with that archive parent
as `--search-root`, writing a second JSON under `AUDIT_ROOT`.  State the exact
additional search root.

## Runs that must be identified

From the audit rows, identify by artifacts rather than memory:

1. The canonical full-1,651 310P accuracy run that produced approximately
   90.13 Overall.
2. The earlier full-1,651 B128 run reported around 10.3k raw token slots/s and
   roughly 192 seconds of decode-graph time.
3. Its B64 companion, reported around 7.7k raw token slots/s and roughly
   249 seconds of decode-graph time.
4. The current 957-crop replay at B128/self-KV-2048/cross-KV-1320, around 6.23k
   raw token slots/s and 210 seconds of graph time.

Do not assume these used the same KV capacities.  The exact command and summary
must establish B, self-KV, cross-KV, maximum length, page/crop count, and commit.

## Required evidence for each identified run

Paste or attach:

- absolute run root and `run_summary.json`/`clean.json` path;
- file modification time;
- exact project commit from `preflight.log` or the saved command;
- physical chip/device and CANN/Torch/Torch-NPU/Python lines;
- exact saved command, including `taskset` or CPU affinity;
- page and crop counts;
- W/T settings, layout batch, decode batch;
- self-KV, cross-KV, and maximum length;
- decode iterations and graph seconds;
- mean graph step milliseconds;
- raw slots/s, effective tokens/s, and slot efficiency;
- decode including ingress and every available `timing_detail` field;
- generated-token sum/mean/p50/p90/p99/max;
- number of 2,047-token caps;
- number of caps that are one token repeated to the limit;
- whether a recognition trace exists and its SHA-256;
- evaluation-summary path if present.

## Comparison report

After pasting every relevant `UNIREC_DECODE_HISTORY_ROW`, write one compact
comparison with these normalized quantities:

```text
per_step_hardware_factor = candidate_mean_step_ms / historical_mean_step_ms
iteration_factor = candidate_iterations / historical_iterations
graph_time_factor = candidate_graph_s / historical_graph_s
```

Then answer, using evidence:

1. Was the historical ~10.3k result cross-KV 512/768 or cross-KV 1320?
2. Did it use self-KV 1024 or 2048?
3. Did it already contain repeated length-cap outputs?
4. Was it B64 or B128, and how many crops were active/effective?
5. Did its decode graph use the same compiled module settings?
6. Is today's lower raw rate explained by graph shape/KV length, or is there a
   same-shape regression?
7. Could CPU affinity explain measured graph time, or only ingress/scheduler
   time? Quote the timing fields.

## Hard rules

- No NPU work.
- No cache deletion or compilation.
- No source edits.
- Do not call a run comparable unless B, self-KV, cross-KV, maximum length,
  model, dtype, and decode backend match.
- Missing artifacts are a result.  Report them instead of reconstructing
  numbers from memory.
