# 310P UniRec known-good decode-lifecycle recreation

## Question

The historical accuracy anchor at project commit `470d8a6` warmed the decoder
on a separate scratch arena, then allocated and decoded real requests on a
fresh untouched arena.  Commit `8444f8e` later moved two warmup calls onto the
live, already-admitted arena to diagnose a 310P TorchAir cache/recompile issue.
Those calls write self-KV position zero before measured generation.

The current 256-crop diagnostic found that almost every 310P output matched the
910B2 token-sequence digest, but a small set generated one token repeatedly to
the 2,047-token cap.  This lane asks whether touching the live arena before real
decode caused those failures.

This is a one-variable A/B against the completed 256-crop result:

- same persistent cross-KV artifact;
- same crop order and first 256 rows;
- same B128, self-KV 2048, cross-KV 1320, and maximum length 2048;
- same compiled OM and cache parent;
- same inference-tensor input contract;
- **zero live-arena warmup calls instead of two**.

The existing OM is already warm and validated.  Do not recreate the historical
scratch warmup because its old normal-tensor contract can select another graph
on current 310P.  Zero live calls preserves the important known-good property:
the real arena is untouched before token 1.

## Inputs

Pull the commit named in Luka's message.  Do not edit tracked files, create a
branch, commit, or push.  Reuse the exact variables and completed run root from
the preceding output-parity diagnostic:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"

export PYTHON_BIN=/absolute/path/to/the/validated/venv/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export COMPILE_CACHE=/absolute/path/to/the/existing/production/cache/parent
export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE=/absolute/path/reported/by/the/passed/cache/gate
export ASCEND_RT_VISIBLE_DEVICES=0  # one free physical 310P, 0-3 only
export CPUSET=0-63
export PRIOR_RUN_ROOT=/absolute/path/to/the/completed/310p_production_decode_diagnostic_run
export DECODE_WARMUP_PASSES=0
```

`PRIOR_RUN_ROOT` is the original completed 957-crop decode diagnostic, not the
256-crop follow-up directory.  The runner resolves its persistent artifact from
`PRIOR_RUN_ROOT/clean.json`.

Keep `PYTHON_BIN` inside the validated venv.  Do not apply `readlink -f` to a
venv symlink that resolves outside it.

## Run

```bash
bash 12_unirec_0_1b_inference/run_310p_decode_output_parity_followup_background.sh
```

Immediately paste the launcher's absolute `RUN_ROOT`, `RUN_LOG`, and this exact
follow command back to Luka:

```bash
tail -f /absolute/RUN_LOG/from/the/launcher
```

This does not run prefill or layout.  It must reuse the existing B128 OM.  The
first decode call may include ordinary cached-model first-use latency because
the lane deliberately performs no live warmup; output parity is the headline,
not this lane's throughput.

## Hard stops

Stop and report the complete log if:

- TorchDynamo reports `recompiled`;
- the OM inventory or hash changes;
- the runner starts prefill or layout;
- it selects a device outside physical 310P 0-3;
- `replay.json` does not report
  `production_graph_warmup.passes == 0`.

Do not delete caches, compile a replacement, change batch/KV sizes, or fall
back to B64.

## Required report

Paste back verbatim:

1. `PRIOR_CLEAN_SUMMARY`
2. `UNIREC_PRODUCTION_DECODE_REPLAY_END`
3. `replay.json` fields:
   - `decode.production_graph_warmup`
   - `validation`
   - `workload.generated_length`
   - `decode.decode_iterations`
   - `decode.decode_s`
4. `UNIREC_DECODE_OUTPUT_PARITY: PASS`
5. `DECODE_OUTPUT_PARITY_REPORT`
6. `DECODE_CACHE_OM_INVENTORY_UNCHANGED`
7. `RUN_ROOT`, `RUN_LOG`, and `exit_code.txt`

The decisive comparison is the exact-token digest count and the number of new
single-token 2,047-token caps:

- If zero live warmups removes the five runaway outputs, the live-arena warmup
  caused the regression.
- If the same request IDs still run away, the issue predates the live warmup;
  next compare them against the canonical 310P full-run recognition trace and
  test them as fresh initial rows versus reused rows.
