# MinerU 310P vision: diagnose the matmul and attention gap

This is a self-contained brief for a pull-only 310P agent with no chat history.
The open question is why **plain vision matmuls as well as PromptFA are so slow
on 310P**. Recover the underlying receipts before making cross-chip ratios.
The following are already-known results relayed by Luka:

| Evidence | 310P | 910B | Interpretation |
|---|---:|---:|---|
| Sep 5, 384-page run, real vision tok/s | 3,160 | about 46,000 | The production vision gap behind the shorthand “3.5k” |
| Corrected Sep 21 S5632 PromptFA, ms/full forward | 1,325.7 | 54.8 | 24.2×, same duration field and per-forward denominator |
| Corrected Sep 21 S5632 four linear projections, ms/full forward | 430.4 | 23.1 | 18.7×; ordinary matmuls are an unexplained major cost |

The Sep 6 310P matrix already found unpad D128 about 13% faster at S5632 and
slightly slower at S768; compiled approximate PromptFA was reported at
1,604.5 ms. Locate that matrix and preserve its exact route/execution/timing
boundary. Unpad, approximate precision and D128 on crop-sized inputs are
**confirmation lanes**, not the main investigation. S5632 in that historical
vision evidence is a MinerU layout image, distinct from the PP-DocLayoutV3
hybrid's recognition crops. The calibration includes M5632 to connect to the
old profile; the real-crop replay does not pretend it is a crop at the 3072 cap.

Main hypotheses to distinguish: (a) slow kernel/tiling choice, (b) too few AI
cores used, (c) waiting between kernels, (d) hidden format conversion/repacking,
(e) throttling or a low-power device. Diagnostic lanes come first below.
The existing 910B result remains a negative result for NZ weights, D128 padding
and the stock op contract on that chip. It does not establish the 310P ranking.

## 1. Source, environment and historical configuration

Branch: `codex/mineru-vision-attention-investigation`. Repository is private.
Resolve the existing checkout with `git rev-parse --show-toplevel`. Preserve
tracked changes; stop rather than stash/overwrite. With clean tracked source:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git fetch origin codex/mineru-vision-attention-investigation
git checkout --detach FETCH_HEAD
git merge-base --is-ancestor 9d153d90 HEAD
git rev-parse HEAD
```

Use this server's existing successful MinerU environment and checkpoint. Do not
assume the 910B interpreter, `npu-setup`, CANN/ATB, TorchAir, package versions,
SSH access or model paths exist here. No installs, downloads, tracked edits,
commits, pushes, branches, device resets or unrelated process termination.
Ignored `tmp/` artifacts and configuration files for this diagnostic are allowed.
Select one healthy free 310P and export `ASCEND_RT_VISIBLE_DEVICES` with its
physical ID. Child workers use logical `npu:0`. Record environment versions,
model/config hashes and health; FP16 throughout. The scripts use torch-npu and
TorchAir and do not import vLLM.

Set `PYTHON` to the absolute working interpreter and `MODEL` to the existing
MinerU2.5-Pro-2605-1.2B checkpoint. Reference hashes from 910B:

- config.json: `22097df08750242647a513043636a8dff16820a09757e9271e220bdea378df28`
- model.safetensors: `abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0`

Before inference, locate the **exact Sep 5 384-page production run** behind the
3,160/“3.5k” figure, and its `command.txt`, source commit, summary and launch
wrapper. Search existing `tmp/11_mineru_2_5_pro_inference/` receipts; do not
substitute a nearby run. If “3.5k” also refers to a later run, identify it
separately. Preserve the original files and hashes. `command.txt` is the
invocation authority. For a setting absent from argv, inspect the entrypoint,
wrapper and parser defaults at **that recorded commit** with `git show`, and
corroborate with logs. “Not specified” does not mean the current default.

Fill the configuration-match table in section 6 **before running**. It must
cover attention/op variant, approximate precision (including a GE converter),
projection mode, LayerNorm mode, head padding, buckets, processor min/max caps,
internal-format setting, backend/execution, dtype and packing. In particular,
`allow_internal_format` may be set inside the recorded source rather than an
argv flag. State the source assignment and commit if that is the evidence.
If still unknown, mark UNKNOWN and report the missing evidence. Do not claim a
production-matched baseline or invent a production configuration.

The explicit controlled 910B configuration is in
`vision_diagnostic_910b_config.json`. If the recovered production configuration
differs, report the differences first and create a **complete**
`$RUN_ROOT/production_config.json` with the same keys and the recovered values.
The loader rejects missing keys. Run an additional capture and baseline under
that configuration, as specified below. Keep processor/bucket differences
visible; different token grids are not matched cross-chip inputs. Packing/page
scheduling is recorded as a scope difference: this replay uses direct real
crops, not the 384-page scheduler. If an attention/backend contract cannot be
expressed by these controls, report the unsupported mapping and a minimal
proposed change; never substitute another implementation silently.

```bash
RUN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/vision_diagnostics_310P_$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)"
mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT"
cp 11_mineru_2_5_pro_inference/vision_diagnostic_910b_config.json "$RUN_ROOT/control_config.json"
"$PYTHON" -m unittest discover -s 11_mineru_2_5_pro_inference -p test_vision_diagnostics.py
"$PYTHON" -m unittest discover -s 11_mineru_2_5_pro_inference -p test_production_vision_attention.py
```

These tests check bookkeeping, not NPU inference. If the historical matrix's
raw CSVs are available, use the generic analyzer in section 5 on it now. Report
its Block Num, wait, core and format findings before new attention retests.

## 2. Capture and establish the diagnostic baselines

Every long run uses `nohup`, a new output directory and short polls. The wrapper
writes an immutable argv/cwd/commit/environment receipt **before launch**, with
host/device snapshots before and after each child. Capture deadline is 3600 s;
every replay/calibration case has an 1800 s deadline and 15 s heartbeats.

```bash
nohup "$PYTHON" -u 11_mineru_2_5_pro_inference/vision_diagnostic_runner.py \
  --output-dir "$RUN_ROOT/control_capture.receipt" --timeout-s 3600 -- \
  "$PYTHON" -u 11_mineru_2_5_pro_inference/bench_production_vision_attention.py capture-crops \
  --model "$MODEL" --config-json "$RUN_ROOT/control_config.json" \
  --image crops/hotswap_001_code_txt_p0001_box_id_3.png \
  --image crops/hotswap_002_code_txt_p1474_11.png \
  --cache-root "$RUN_ROOT/control_cache" --output-dir "$RUN_ROOT/control_capture" \
  > "$RUN_ROOT/control_capture.driver.log" 2>&1 </dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/control_capture.pid"
```

Poll the driver and receipt with tool calls lasting at most 30 seconds:

```bash
tail -n 12 "$RUN_ROOT/control_capture.driver.log"
test ! -f "$RUN_ROOT/control_capture.receipt/exit.json" || cat "$RUN_ROOT/control_capture.receipt/exit.json"
```

Proceed only after a completed exit-zero receipt. The control crops should
produce 720/bucket768 and 3036/bucket3072 raw vision tokens. Verify the actual
manifest grids, model/image/tensor hashes, masks and independent filler segments.
Capture really executes patch embedding, positions, all 32 blocks and merger.
For a production-config capture use its actual shapes, without forcing these
expected control shapes. Derive route names from each manifest:

```bash
ROUTES=$("$PYTHON" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["routes"]))' "$RUN_ROOT/control_capture/manifest.json")
nohup "$PYTHON" -u 11_mineru_2_5_pro_inference/run_production_attention_matrix.py \
  --capture-dir "$RUN_ROOT/control_capture" --cache-root "$RUN_ROOT/control_cache" \
  --output-dir "$RUN_ROOT/control_baseline" --routes "$ROUTES" --variants baseline \
  --steps 30 --profile --timeout-s 1800 \
  > "$RUN_ROOT/control_baseline.driver.log" 2>&1 </dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/control_baseline.pid"
```

Require every baseline to be bit-exact with **its own capture**, with zero
compilation inside the warm timer. Finish and inspect health before launching
the next matrix. Run serially on the selected device.

If production differs, repeat the two commands above as an additional lane:
use `production_config.json`, `production_capture.receipt`, `production_capture`,
`production_cache`, `production_baseline` and distinct driver/PID filenames.
Derive `PRODUCTION_ROUTES` from its manifest. Do not reuse the control capture
for a baseline with a different config; this is explicitly rejected. If the
production internal-format setting was false, also run its baseline capture
through variant `internal_format_opposite`, in a separate matrix directory,
to label the false→true comparison. The controlled matrix below always supplies
the complementary true→false comparison. Each has a distinct cache identity.

## 3. Main diagnostics: projection calibration and full-stack changes

First run the standalone calibration in the **same activated environment**,
physical device and process settings as the controlled replay. It selects
FP16, JIT off and the requested internal-format flag explicitly. It uses
bounded seeded synthetic operands, which are appropriate only for calibration.
It is not OCR, a model result, or a replacement for the real-input full stack.

```bash
nohup "$PYTHON" -u 11_mineru_2_5_pro_inference/probe_vision_matmul.py suite \
  --output-dir "$RUN_ROOT/matmul_on" --internal-format on \
  --ms 768,3072,5632 --projections qkv,proj,fc1,fc2 \
  --executions eager,compiled --weight-formats nd,nz --steps 30 --timeout-s 1800 \
  > "$RUN_ROOT/matmul_on.driver.log" 2>&1 </dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/matmul_on.pid"
```

This is 48 cases: M ∈ {768,3072,5632}; (K,N) = (1280,3840), (1280,1280),
(1280,5120), (5120,1280); eager/compiled; ND/NZ. Each case is a fresh process,
with its own cache and before/after telemetry. It calls F.linear with bias,
like the vision linears. Report actual kernel names, Block Num, formats,
kernel duration and achieved TFLOPS = `2*M*K*N / (kernel_us * 1e6)`.
For decomposed matmuls, use the sum of matmul kernel durations per forward;
do not assign the full problem's FLOPs to every partial kernel. Profiling uses
three calls and divides by three. Wall/event latency is retained separately;
short eager calibration may be launch-bound. Never turn these numbers into
model tok/s. A small FP32 sample check is calibration validation only, not an
FP32 reference investigation of the unexplained full-vision unpad drift.

If production used internal formats off, run the same calibration with
`--internal-format off --weight-formats nd` and a fresh `matmul_off` output/log.
NZ/off is an incompatible request and is reported as skipped, never silently
cast back to ND. The on run above supplies the explicit opposite-setting ND/NZ
comparison. Preserve process/environment differences if the old production
runtime cannot be reproduced; do not upgrade it to match 910B.

Then test projection and format choices **inside the complete vision stack**:

```bash
nohup "$PYTHON" -u 11_mineru_2_5_pro_inference/run_production_attention_matrix.py \
  --capture-dir "$RUN_ROOT/control_capture" --cache-root "$RUN_ROOT/control_cache" \
  --output-dir "$RUN_ROOT/projection_diagnostics" --routes "$ROUTES" \
  --variants pfa_nz_weights,internal_format_opposite,grouped_qkv,grouped_qkv_mlp_fc1 \
  --steps 30 --profile --timeout-s 1800 \
  > "$RUN_ROOT/projection_diagnostics.driver.log" 2>&1 </dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/projection_diagnostics.pid"
```

The grouped modes are **exploratory, not known speedups**. They originated as
310P compile-failure workarounds (README's contract matrix). They are included
because UniRec work found 310P TransData/weight-repacking cost, and calibration
may reveal a similar issue here. Report timing, actual grouped-kernel names,
conversions and feature drift like any other candidate. An unsupported grouped
operator is a compatibility result, not permission to change source.

## 4. Secondary confirmation lanes

After the diagnostics above, confirm the already-known attention choices on
crop-sized inputs:

```bash
nohup "$PYTHON" -u 11_mineru_2_5_pro_inference/run_production_attention_matrix.py \
  --capture-dir "$RUN_ROOT/control_capture" --cache-root "$RUN_ROOT/control_cache" \
  --output-dir "$RUN_ROOT/attention_confirmation" --routes "$ROUTES" \
  --variants eager_pfa,unpad_d128,unpad_d128_nz_weights,pfa_d128,pfa_approx,pfa_d128_approx \
  --steps 30 --profile --timeout-s 1800 \
  > "$RUN_ROOT/attention_confirmation.driver.log" 2>&1 </dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/attention_confirmation.pid"
```

“Stock vLLM-Ascend” means the stock `_npu_flash_attention_unpad` op inside our
eager stack, **not the vLLM-Ascend vision path**. D80 is padded to D128 with
D80 scale preserved. Fresh Q/K/V are not persistent decode KV caches. CPU
lengths are prepared once outside the timer; the stock wrapper's repeated
`torch.diff(...).to("cpu")` is not reproduced. Approximate lanes use the owned
310P GE `innerPrecise=4` converter, not an invented Python API keyword.

All full-stack timings execute 32 blocks. CPU preprocessing/H2D, initial patch
embedding/positions, merger, format conversion at load and compile are outside
warm timing. Useful raw vision tokens are counted once per full forward,
excluding filler. Report drift independently; candidate numerical differences
must not hide timing. The old 910B unpad layer-0 relative L2 of 2.8e-4 versus
full-encoder 2.15% remains unexplained. No OCR-quality equivalence is claimed.
The 910B small-crop eager results are host-bound; do not report unpad-versus-PFA
speed differences there. Diagnose whether a 310P eager case is also host-bound
before using its whole-stack timing to compare operators.

## 5. Required accounting and stop conditions

Run the chip-independent analyzer on every completed or partial run tree:

```bash
"$PYTHON" 11_mineru_2_5_pro_inference/analyze_vision_diagnostics.py \
  --run-dir "$RUN_ROOT" --profile-forwards 3 --output "$RUN_ROOT/diagnostic_analysis.json"
```

It has no Ascend910B2 check or fixed 96/384 kernel-type assertion. It recognizes
Attention-containing types, MatMul/BatchMatMul/GroupedMatmul/GEMM equivalents,
and TransData/format casts, reporting actual names. Unclassified operations
remain visible by type. Missing fields are missing, not zero measurements.
It preserves per-call Block Num, Mix Block Num, Wait Time(us), Accelerator Core,
HF32 Eligible, Input Shapes and Input/Output Formats, plus per-type aggregates.
Conversions are grouped by direction. Every duration/wait total is divided by
**three profiled forwards**, unless result metadata explicitly records another
count. Validate unexpected counts against the trace; do not rename kernels to
make an assertion pass.

Total CSV wait is its own bucket. Remaining means remaining **kernel duration**,
not wall-minus-kernels. Observed gaps are also recorded per stream/profile-step.
Wait counters/gaps alone do not prove host delay; dependencies, instrumentation
and stream scheduling can contribute. They overlap other measures and must not
be added to kernel time as elapsed time. Block Num is a launch-grid indicator,
not proof of utilization. Compare it with available cores and the calibration,
and report what is missing before attributing the gap to under-parallelism.

The analyzer was checked against the committed 910B archive: all 12 attention,
matmul and total kernel durations match `analysis.json` within 1e-6 ms. Reproduce
this read-only check if needed by extracting its `raw_evidence.tar.gz` into a
fresh ignored directory and adding `--compare-analysis` with that reference's
`analysis.json`. Do not overwrite either original artifact.

The runner records `uptime`, load average, CPU count/affinity, process names/PIDs,
full `npu-smi info`, and selected-card health/power/usages/sensors/common/work-mode
queries before and after **each lane**. Report AI-core clock and performance
mode only if actually exposed and identified. A successful query saying “not
supported” is unavailable data; say “not exposed”. Keep raw responses and
other-job snapshots. If running inside a container, label its process table as
container-visible; NPU PIDs may belong to other namespaces. Obtain host process
context through an existing authorized host connection if available, otherwise
report missing other-container CPU job identities. Do not infer a clock or governor from power alone. Pre/post-lane power and
health snapshots cannot rule out throttling during a timed forward; leave that
hypothesis open if no under-load clock/performance evidence is exposed.

Stop and report immediately if baseline replay is not bit-exact with its own
capture, any lane compiles inside the warm timer, an NPU/device error occurs,
outputs are nonfinite, or a deadline is exceeded. The driver terminates only
its own child process group on timeout and preserves evidence. Inspect health
and the last phase; do not wait indefinitely, clear caches, kill unrelated
processes, relaunch automatically or substitute eager for compiled. A healthy
operator/compiler compatibility rejection is recorded as a failed/skipped lane;
report the minimal proposed fix without changing tracked source.

The revised tooling was exercised on 910B: 48 on-format calibration cases,
two off-format ND checks and eight full-vision lanes completed. The explicit
off/NZ incompatibilities and the initial corrected compiler-wrapper failure
are preserved in [the validation results](references/vision_diagnostics_910b_20261007/RESULTS.md).
These validate the tools; they do not answer the 310P hardware question.

## 6. Exact report format to Luka

Reply directly in chat with the following tables and a plain failure/skip list.
Attach/reference machine-readable artifacts as evidence, not as a replacement
for the explanation. Every numerical row must state **310P or 910B**; never
present a calibration TFLOPS figure as a full-model result.

Configuration-match table (present first if anything differs):

| Field | Historical 310P production value | Authority: command.txt literal or source/default at recorded commit | Controlled replay value | Production replay value | Match/difference/unknown |
|---|---|---|---|---|---|
| Run ID, timestamp, source commit | | | | | |
| Attention implementation / op | | | PromptFA | | |
| Approximate precision / GE converter | | | off | | |
| Projection mode | | | linear | | |
| LayerNorm mode | | | manual_fp32 | | |
| Head dimension/padding | | | D80/native | | |
| Buckets | | | 384,768,3072 | | |
| Processor min/max pixels | | | 25088/602112 | | |
| allow_internal_format | | | true | | |
| Execution/backend and dtype | | | compiled / FP16 | | |
| Packing/cohort and timing/token denominator | | | direct crops / 32 blocks / useful raw tokens | | |

One lane table, including failures and unsupported cases:

| Chip | Lane ID / purpose | Execution mode | Internal format / projection | Real/padded tokens | Wall mean ms | Event mean ms | Useful vision tok/s | Relative L2 drift | Status |
|---|---|---|---|---|---:|---:|---:|---:|---|

One kernel-bucket table per lane; expand attention and matmul into rows for
**each actual type** and retain per-call columns in JSON/CSV:

| Chip / lane | Bucket and actual type | Profile forwards | Calls/forward | Kernel ms/forward | Wait ms/forward | Block Num / Mix histogram | Accelerator Core | HF32 | Input shapes / formats | Conversion direction |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| | Attention: each type | 3 | | | | | | | | |
| | Matmul: each type | 3 | | | | | | | | |
| | TransData / format cast | 3 | | | | | | | | |
| | Remaining: each unclassified type | 3 | | | | | | | | |
| | TOTAL WAIT — separate, non-additive | 3 | — | — | | — | — | — | — | — |
| | Observed per-stream gaps — separate | 3 | — | — | — | — | — | — | report gap ms and stream IDs | — |

Calibration table (explicitly title it “Synthetic matmul calibration only”):

| Chip | Case ID | Mode | Internal formats | Weight requested/actual | M | K | N | Kernel type(s) | Block Num / Mix | Kernel us/forward | Wait us/forward | Achieved TFLOPS | Status |
|---|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|

Device/host context table, before and after every lane:

| Chip / lane | Health | Power / temperature | AI-core clock | Performance mode | Load averages | CPU count / affinity | Other NPU/host jobs (PIDs, names) | Missing fields |
|---|---|---|---|---|---|---|---|---|

Then a plain list of every failure, timeout, skip, missing measurement or
unsupported contract, with exact command, first causal error and artifact path.
Finish with an evidence-based assessment of hypotheses (a)–(e): observed,
ruled out by which measurement, or still unknown. Keep kernel selection,
parallelism, waits, conversion and device-state explanations distinct.
Preserve raw CSVs, immutable receipts, capture manifests, results and caches.
The receiving agent does not commit or push; Luka relays the findings.
