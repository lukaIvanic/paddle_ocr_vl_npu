# MinerU: approximate PromptFA improvement by real crop length (310P)

This brief is self-contained. The question is whether the approximate PromptFA
gain seen at short input lengths persists or grows for recognition crops up to
3,072 raw vision tokens. Run on **310P**, FP16. Do not substitute isolated
attention, synthetic hidden states, decoder tokens/s, or a 910B result.

## What is already known

These are **historical 310P3** results from the September 6 matrix, all 32 vision
blocks, compiled D80, warm NPU-event timings (30 samples). Retained evidence is
the transcription of six work-server screenshots, not raw profiler receipts.

| Useful / physical tokens | Baseline ms | Approximate ms | Baseline tok/s | Approximate tok/s | Throughput gain |
|---|---:|---:|---:|---:|---:|
| 640 / 768, single crop | 101.2 | 91.3 | 6,322 | 7,008 | +10.9% |
| 672 / 768, packed 480+192 | 101.2 | 91.3 | 6,640 | 7,362 | +10.9% |
| 5,476 / 5,632, layout image | 2,169.9 | 1,604.5 | 2,524 | 3,413 | +35.2% |

The 5k result is valid full-stack evidence. It is a layout image, beyond the
602,112-pixel recognition crop cap, and does not establish the missing 1k–3k
crop results. Keep it as a historical reference, not a new measurement.
No retained approximate-PromptFA crop measurements fill those middle lengths.

## Method and scope

New choices for this experiment: two distinct real crops per physical bucket
384, 512, 768, 1024, 1536, 2048, 3072; selected near one-third and two-thirds of
each bucket interval from the tracked 100-crop manifest. The installed **slow
AutoProcessor** determines actual grids. Images are never resized to artificial
target token counts; the normal processor uses min_pixels=25,088 and
max_pixels=602,112. Selection fails if a bucket lacks enough real crops.
The inventory records every candidate's actual token count, grid, original
dimensions and image hash. Preserve selection.json with the results.

The baseline is compiled PromptFA D80, FP16 ordinary linear weights (ND),
manual FP32 LayerNorm, internal formats enabled, full attention within the real
crop and a separately masked padding segment. The configuration is complete in
`vision_length_config.json`; no production defaults are silently inferred.
Approximate changes only the GE PromptFA innerPrecise setting to 4, through the
existing verified converter in
`09_persistent_page_engine/scripts/vision_matmul_lab.py`,
`_register_promptfa_inner_precise_converter(4)`, reused by
`bench_production_vision_attention.py:approximate_converter`. It is the same
implementation used for the September 310P approximate lane. Do not invent an
unsupported torch-npu keyword or replace the operator.

Capture runs the real processor → patch embedding / positions → full compiled
32-block encoder → merger, saving the inputs and expected encoder outputs.
The measured replay includes **all 32 vision blocks**, including their
attention, projections, MLPs and LayerNorm. It excludes CPU image processing,
H2D, patch embedding, initial positions, merger, text prefill, decode and page
layout. Useful vision tokens count the input patches **once**, not 32 times,
not padding tokens or merged image tokens. These are encoder tok/s, not page/s.

Each crop runs baseline, approximate, baseline, approximate, with **30 warm
samples per lane**. Compile/first-call and two warmup passes occur before the
timer; wall and NPU-event mean/p50/p99 are retained separately. NPU-event time
is the stream region, not a sum of kernel durations. Compare adjacent pairs,
show both repeats, and retain numerical feature drift independently of speed.
Single-crop results omit production packing for small crops. This deliberately
isolates the length dependence requested here; **do not frequency-weight these
chosen crops or claim a production-wide speedup from their arithmetic mean**.
Production weighting needs the actual capped run's per-group token lengths,
packing and timing distribution. The alleged 0.289 pg/s run's original receipt
has not been recovered; do not call this its verified configuration.

910B checks the capture/replay harness and baseline across these buckets.
Approximate innerPrecise=4 is 310P-only; on 910B the script explicitly records
`unsupported_on_this_device`. That is not a measured zero gain.

## Source and environment

Repository is private. Branch: `codex/mineru-vision-length-sweep`.
Preserve other work. With a clean tracked checkout:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git fetch origin codex/mineru-vision-length-sweep
git checkout --detach FETCH_HEAD
git merge-base --is-ancestor 5335a3cd HEAD
git rev-parse HEAD
```

The pin names the first harness commit; later evidence/brief commits are
descendants. Record actual HEAD. No installs, downloads, source edits, commits,
pushes, branches, resets, or termination of unrelated processes on this server.
Use the existing successful MinerU torch/torch-npu/TorchAir environment and
checkpoint. Set `PYTHON` to its absolute interpreter and `MODEL` to the existing
MinerU2.5-Pro-2605-1.2B directory. Do not assume 910B paths or npu-setup exist.
Select a healthy free 310P, export its physical ID as ASCEND_RT_VISIBLE_DEVICES;
children use logical npu:0. Record installed versions, device name and hashes.
Reference checkpoint SHA-256:

- config.json: `22097df08750242647a513043636a8dff16820a09757e9271e220bdea378df28`
- model.safetensors: `abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0`

If the checkpoint differs, report that before interpreting comparison with the
historical reference; don't silently change it. If an import/contract fails,
report the minimal proposed fix with command/log, rather than editing source.

## Run and poll

Use new directories. The selection stage is CPU preprocessing only. The run
stage wraps capture and every replay with immutable launch receipts plus
before/after device health, power/mode/clock queries, load, CPU count and jobs.
An unavailable clock/mode query remains explicitly unavailable.

```bash
EXP=11_mineru_2_5_pro_inference
RUN_ROOT="$WORK_SERVER_REPO/tmp/$EXP/vision_lengths_310P_$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)"
mkdir -p "$RUN_ROOT"
"$PYTHON" -m unittest discover -s "$EXP" -p test_vision_length_sweep.py
"$PYTHON" -m unittest discover -s "$EXP" -p test_production_vision_attention.py
nohup "$PYTHON" -u "$EXP/vision_diagnostic_runner.py" \
  --output-dir "$RUN_ROOT/selection.receipt" --timeout-s 900 -- \
  "$PYTHON" -u "$EXP/vision_length_sweep.py" select \
  --model "$MODEL" --per-bucket 2 --output "$RUN_ROOT/selection.json" \
  > "$RUN_ROOT/selection.driver.log" 2>&1 </dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/selection.pid"
```

Poll with `tail -n 20 "$RUN_ROOT/selection.driver.log"` and
`cat "$RUN_ROOT/selection.receipt/exit.json"` once it exists. No blocking SSH
call around a cold compile. Proceed only after exit status completed. Inspect
the selected actual useful lengths before running; empty bucket is a failure,
not permission to generate dummy inputs or alter the pixel cap.

```bash
nohup "$PYTHON" -u "$EXP/vision_length_sweep.py" run \
  --selection "$RUN_ROOT/selection.json" --output-dir "$RUN_ROOT/sweep" \
  --steps 30 --repeats 2 --capture-timeout-s 3600 --lane-timeout-s 1800 \
  > "$RUN_ROOT/sweep.driver.log" 2>&1 </dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/sweep.pid"
```

Poll every 15–30 seconds with short tail/process checks. Each cold lane has an
1800-second deadline; capture has 3600 seconds. Timeout kills only that lane's
owned process group, preserves artifacts and stops. No automatic fallback or
retry. Stop and report on device error, baseline mismatch with capture,
nonfinite/nondeterministic output, or compilation inside the measured timer.
Finite approximate drift does **not** disqualify throughput; report it.

The last block of the driver log prints the tables. To regenerate after a stop:

```bash
"$PYTHON" "$EXP/vision_length_sweep.py" report --output-dir "$RUN_ROOT/sweep"
```

If capture failed before sweep.json was written, report capture.receipt/run.log
and exit.json instead. Do not relabel an incomplete run as complete.

## Report directly in chat

Paste the tables, not just a filename:

1. Chip, actual useful tokens, physical bucket, crop ID and original dimensions.
2. Every lane: repeat, baseline/approximate, wall ms, event ms, useful wall and
   event tok/s, status. Include mean/p50/p99 from length_results.json when
   variability matters. Keep the historical 310P 5k row separately labelled.
3. Per adjacent pair: wall and event throughput gain percentages. Summarize
   the range of both repeats by actual length; don't hide an unfavorable pair.
4. Feature relative L2 / maximum absolute difference / exactness for each lane;
   no downstream OCR-quality claim.
5. Configuration, chip/runtime/checkpoint, host load and other jobs before/after;
   failures, unsupported cases and skips with reasons. Identify any departure
   before dependent work. Attach the complete ignored run artifacts for relay.

No 310P run is performed from the authoring session. The larger diagnostic
handoff remains `VISION_CROP_310P_HANDOFF.md`; this focused request does not
replace its matmul/PromptFA gap investigation.
