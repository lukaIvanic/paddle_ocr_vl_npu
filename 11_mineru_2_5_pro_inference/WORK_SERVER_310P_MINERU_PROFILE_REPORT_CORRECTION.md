# 310P: correct the saved vision-profile report, without rerunning anything

Luka authorizes this read-only analysis of the already completed 310P vision
profiles. This new brief supersedes the launch instructions in the prior
profiling handoff for this task. **Do not launch inference, profiling, model
loading, compilation, evaluation, or any NZ-format experiment.**

Report directly to Luka in plain text, with compact tables if useful. Do not
create a narrative report file, edit tracked source, commit, push, branch,
reset, stash, change packages/model configs, or alter/clear caches. Do not
propose a code patch. Reading saved artifacts and printing calculations is all
that is needed. If evidence is missing, report the missing paths/fields to Luka;
do not regenerate it by rerunning the model.

## 1. Resolve the existing run and reference

Read `CLAUDE.md` and `AGENTS.md` for lane rules; do not execute historical runs.
Inspect tracked changes and preserve them. Pull `main` with `git pull --ff-only
origin main`; stop on conflicts rather than discarding work.

```bash
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git merge-base --is-ancestor 4124c5fa HEAD
```

The report Luka relayed identifies:

```text
profile commit: 447176de7f88
device: 310P3 index 3, healthy
control dir (relative to YOUR checkout):
tmp/11_mineru_2_5_pro_inference/310p_vision_profiles_447176de7f88_20260905T191727Z/
```

Verify that directory, its `endpoints` and `middle` result files, their run logs
and exit files. Resolve actual names from the existing artifacts if needed;
do not substitute another run. No access to the Mac or 910B host is available
or needed. Set:

```bash
export PROFILE_CONTROL="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/310p_vision_profiles_447176de7f88_20260905T191727Z"
export PROFILE_ENDPOINTS="$PROFILE_CONTROL/endpoints"
export PROFILE_MIDDLE="$PROFILE_CONTROL/middle"
# Use your existing Python executable; the analysis below needs only stdlib.
export ANALYSIS_PYTHON=python3
test -s "$PROFILE_ENDPOINTS/result.json"
test -s "$PROFILE_MIDDLE/result.json"
```

The committed 910B reference is
`11_mineru_2_5_pro_inference/references/vision_profiles_910b_20260905/analysis.json`.
Its `captures.pipe.kernel_ms`, `attention_ms` and `groups_ms` are already
normalized **per forward**, using kernel `duration_us`, not AICore time.

## 2. Fix the denominator and metric mismatch

Your previous report compared 310P totals over three forwards against 910B
per-forward figures. The 96 attention calls confirm 32 layers × 3 forwards;
384 projection calls confirm 4 projections × 32 layers × 3 forwards.

Recalculate from saved parsed JSON/CSV, not rounded numbers in your old reply.
For every route and each pipe/memory capture:

```text
N = result.json profile_steps (expected 3)
attention calls = 32 * N (expected 96)
linear calls = 128 * N (expected 384)
kernel milliseconds/forward = kernel_details.total_duration_us / (1000 * N)
category milliseconds/forward = sum(category duration_us) / (1000 * N)
share = sum(category duration_us) / kernel_details.total_duration_us
slowdown = 310P milliseconds/forward / 910B milliseconds/forward
```

Use `duration_us` consistently on both sides. Do not mix
`aicore_time_us` / `total_aicore_time_us` with kernel duration, wall time or
device-event time. AICore and PMU counters may be separately reported with their
own labels. Do not divide by 32 a second time unless explicitly reporting a
single-layer latency. Do not divide the already-normalized 910B numbers again.

This stdlib-only check prints normalized totals and exact kernel-type rows:

```bash
"$ANALYSIS_PYTHON" - <<'PY'
import json, os
from pathlib import Path
repo = Path(os.environ['WORK_SERVER_REPO'])
reference = json.loads((repo/'11_mineru_2_5_pro_inference/references/vision_profiles_910b_20260905/analysis.json').read_text())
ref = {row['route']: row for row in reference}
seen = set()
for stage in ['PROFILE_ENDPOINTS', 'PROFILE_MIDDLE']:
    root = Path(os.environ[stage])
    result = json.loads((root/'result.json').read_text())
    assert not result['missing_routes']
    n = result['profile_steps']
    assert n == 3
    for route, record in result['results'].items():
        assert route not in seen
        seen.add(route)
        for metric in ['pipe', 'memory']:
            path = root/route/metric/'parsed_profile_summary.json'
            parsed = json.loads(path.read_text())
            assert len(parsed['runs']) == 1, f'ambiguous capture count: {path}'
            kernel = parsed['runs'][0]['kernel_details']
            types = kernel['top_kernel_types']
            assert sum(row['count'] for row in types) == kernel['row_count'], 'truncated type list; inspect raw CSV'
            assert abs(sum(row['duration_us'] for row in types)-kernel['total_duration_us']) < 1e-3
            attention = [row for row in types if row['name'] == 'PromptFlashAttention']
            linears = [row for row in types if row['name'].startswith('MatMul')]
            assert sum(row['count'] for row in attention) == 32*n
            assert sum(row['count'] for row in linears) == 128*n
            total_ms = kernel['total_duration_us']/(1000*n)
            attn_ms = sum(row['duration_us'] for row in attention)/(1000*n)
            linear_ms = sum(row['duration_us'] for row in linears)/(1000*n)
            base = ref[route]['captures'][metric]
            groups = base['groups_ms']
            base_linear = sum(groups[key] for key in ['qkv_linear', 'attention_output_linear', 'mlp_fc1', 'mlp_fc2'])
            print(json.dumps(dict(route=route, metric=metric, forwards=n,
                real_tokens=record['tags']['real_tokens'], members=record['tags']['member_lengths'],
                members_match_910b=record['tags']['member_lengths']==ref[route]['members'],
                kernel_ms=total_ms, attention_ms=attn_ms, linear_ms=linear_ms,
                kernel_ratio=total_ms/base['kernel_ms'], attention_ratio=attn_ms/base['attention_ms'],
                linear_ratio=linear_ms/base_linear,
                attention_share=attn_ms/total_ms, linear_share=linear_ms/total_ms), indent=2))
            for row in types:
                print(json.dumps(dict(kernel_type=row['name'], count=row['count'],
                    ms_per_forward=row['duration_us']/(1000*n),
                    formats=row.get('input_format_samples'), shapes=row.get('input_shape_samples'))))
assert seen == set(ref), f'route coverage mismatch: {seen}'
print('NORMALIZED_KERNEL_ACCOUNTING: PASS')
PY
```

If a type name differs or lists are truncated, inspect the existing raw CSV
and explain the mapping; do not relax the gates silently or load the model.

## 3. Align semantic groups

The previous report's "rotary/slice/cast" 910B denominator included both
`rotary_elementwise_slice_cast` AND `qkv_split_and_attention_layout` (for S5632,
13.92952 + 2.98144 ms). It was not the narrower rotary-only group.

Either separate these groups on both chips, or label the combined group
"rotary + QKV split + attention layout" on both. List which types/shapes and
fused operations you attributed to LayerNorm, activation and residual work.
Do not group all Add/Mul/Cast operations under LayerNorm: rotary and residual
paths also use them. Prefer an explicit unassigned/mixed-fusion category over
guessing. Kernel-type totals are exact evidence; semantic grouping based on
source/shapes is an attribution and must be labeled accordingly.

Check category sums reconcile with total kernel duration and that pipe and
memory captures agree. Keep the trustworthy unprofiled route timings and their
denominators separate. Combine all 60 raw samples for p50/p99/max; never average
two p99s. Do not treat the distorted nested page-run pg/s as a benchmark.

## 4. Inspect existing weight-format evidence (read-only)

Luka is considering NZ pre-formatting of vision weights. The current custom
`configure_decode_weight_format` touches text decoder linears and a separate
LM-head copy, not `model.visual`. On 910B, the saved S5632 profile shows ND
weights for QKV, output projection, MLP FC1 and MLP FC2, with no separate
TransData kernel records in that capture.

From your existing 310P `kernel_details.csv` or parsed
`top_matmul_shape_signatures`, print for each of those four projection shapes:

- kernel name/type, weight input index and logical shape;
- input formats and dtypes (is the weight ND or FRACTAL_NZ?);
- total calls, duration per forward and duration per call;
- any separate TransData/format-conversion kernel names, shapes and durations.

Do this for S384, S768 and S5632; include other shapes if their paths differ.
An ND input label does not reveal all packing inside a matmul kernel. Absence
of a separate TransData kernel does not establish absence of internal packing
cost. Report exactly what the saved capture exposes, not what you assume.

**Do not convert weights, enable a new runtime setting, run an NZ test, or
change cache identities in this task.** This evidence will inform a separately
authorized production-path experiment.

## 5. Correct the conclusions and reply directly

Explain the three-forward denominator correction plainly. Replace the old
68.2x attention / 52.1x matmul / 86.8x LayerNorm claims with the recomputed,
same-field, per-forward ratios. Do not merely copy their values divided by
three: the metric-field mismatch must also be corrected.

Retract these unsupported inferences:

- high busy percentage proves compute-bound rather than memory-bound;
- high cube utilization proves an optimal kernel;
- low between-kernel free time rules out internal data-movement stalls;
- zero cube utilization for vector operations demonstrates their inefficiency;
- the measured ratios establish a particular complexity exponent or prove the
  slowdown is solely raw chip capacity.

Busy time shows scheduled activity, not which internal resource limits it.
Preserve PMU fields, but do not interpret their aggregation/unit semantics
without evidence. Present likely explanations as hypotheses.

Reply with: corrected route/kernel tables, the explicit denominator and metric
field, exact-vs-inferred grouping, weight-format evidence, material profiler
warnings, and the revised bottleneck assessment. Matching member lengths and
within-chip exact replay are not proof of cross-chip identical input tensors
or model-output parity; say what was actually checked. Give existing artifact
paths and finish. No narrative report file, rerun, code change or new experiment.
