# MinerU 310P vision bottleneck: real-crop full-encoder handoff

This brief is self-contained. The receiving agent has no conversation history.
The active target is **vision encoding**: Luka reports roughly 3.5k raw vision
tokens/s on 310P versus 45k+ on 910B. Do not benchmark text decode or persistent
KV caches for this task. Read `VISION_310P_TARGET.md` for the source audit and
measurement boundary. The authoring branch is
`codex/mineru-vision-attention-investigation`; benchmark source starts at
`82575a91`. The repository is private.

## Purpose and constraints

Compare complete **32-block FP16 MinerU vision forwards** on two actual crop
images. The control is our compiled masked PromptFA D80 stack. The alternative
uses `_npu_flash_attention_unpad` with D80 padded to D128, D80 scale retained,
CPU int32 sequence lengths and output sliced back to D80. This matches the
audited vLLM-Ascend 310P attention operator contract. Compare ND versus NZ
vision projection weights separately. Vision has fresh Q/K/V in each layer;
there is no autoregressive KV cache.

- This server is pull-only for source. No tracked edits, commits, pushes,
  branches, installs, model downloads, device resets, service shutdowns or
  changes to vLLM. Write runtime artifacts only under ignored `tmp/`.
- Discover the existing working environment. Do not assume our 910B Python,
  CANN/ATB, torch-npu, TorchAir, model paths or setup scripts exist here.
- Select one healthy, unoccupied **310P**, using this server's procedure.
  Record physical visibility and device name. Use logical `npu:0` in the worker.
- Use the existing FP16 `MinerU2.5-Pro-2605-1.2B` checkpoint, actual repository
  images and existing dependencies. No BF16 or CPU inference fallback.
- If source is dirty or access/dependencies/model/operator support is missing,
  preserve the failing command and logs. Propose a minimal fix without applying
  tracked changes. Stop after device errors or timeouts and inspect health.

## Source and environment

Resolve the existing checkout without assuming its absolute path:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short
```

Stop for tracked source modifications; do not stash or overwrite them. Otherwise:

```bash
git fetch origin codex/mineru-vision-attention-investigation
git checkout --detach FETCH_HEAD
git merge-base --is-ancestor 82575a91 HEAD
git rev-parse HEAD
```

Require the ancestor check to succeed. Activate the **existing local** working
NPU environment. Set `PYTHON` to its absolute interpreter and `MODEL` to its
existing checkpoint directory. Record torch, torch-npu, Transformers, PIL,
TorchAir (top-level or bundled), CANN/ATB, `npu-smi info`, hostname and selected
physical device. Check that `_npu_flash_attention_unpad`, PromptFA and
`npu_format_cast` exist. These scripts use the owned model plus torch-npu,
TorchAir, PIL and Transformers processing; they do not import vLLM.

Reference checkpoint SHA256 values from 910B:

- config.json: `22097df08750242647a513043636a8dff16820a09757e9271e220bdea378df28`
- model.safetensors: `abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0`

Report any mismatch before interpreting a cross-chip comparison. Preserve
environment version differences instead of installing our versions.

## Capture actual crop inputs

Run the existing mask/segment and repacking contract checks first:

```bash
"$PYTHON" -m unittest discover -s 11_mineru_2_5_pro_inference \
  -p test_production_vision_attention.py
```

Create a unique run directory; do not reuse or overwrite another run's caches.
Save exact commands, HEAD, environment and exit status alongside every log.

```bash
RUN_ROOT="$WORK_SERVER_REPO/tmp/11_mineru_2_5_pro_inference/vision_crop_310P_$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)"
mkdir -p "$RUN_ROOT"
npu-smi info > "$RUN_ROOT/occupancy_before.txt"
git rev-parse HEAD > "$RUN_ROOT/source_commit.txt"
"$PYTHON" 11_mineru_2_5_pro_inference/bench_production_vision_attention.py capture-crops \
  --model "$MODEL" \
  --image crops/hotswap_001_code_txt_p0001_box_id_3.png \
  --image crops/hotswap_002_code_txt_p1474_11.png \
  --cache-root "$RUN_ROOT/vision_cache" \
  --output-dir "$RUN_ROOT/capture" \
  > "$RUN_ROOT/capture.log" 2>&1
CAPTURE_RC=$?
printf '%s\n' "$CAPTURE_RC" > "$RUN_ROOT/capture_exit.txt"
```

Proceed only for exit zero. Capture runs patch embedding, position preparation,
the full compiled 32 blocks and merger on actual crops. It saves actual block
inputs, rotary factors, masks and baseline outputs. No synthetic hidden states
or manufactured Q/K/V. Processor cap is 602112 pixels / 3072 raw vision tokens.
On 910B these images produce 720/bucket768 and 3036/bucket3072 tokens. Verify
the manifest's image/model hashes, grids and segments. A changed preprocessing
result is a different input; report it rather than force the expected shape.

## Full-encoder timing and profiles

The baseline replay must exactly match its own compiled capture before any
candidate timings are interpreted. The sequential driver stops at the first
failed lane and records its command, log, exit status and completed lanes.

```bash
"$PYTHON" -u 11_mineru_2_5_pro_inference/run_production_attention_matrix.py \
  --capture-dir "$RUN_ROOT/capture" \
  --cache-root "$RUN_ROOT/vision_cache" \
  --output-dir "$RUN_ROOT/matrix" \
  --routes crop_0_bucket_768,crop_1_bucket_3072 \
  --variants baseline,pfa_nz_weights,eager_pfa,unpad_d128,unpad_d128_nz_weights \
  --steps 30 --timeout-s 900 --profile \
  > "$RUN_ROOT/matrix_launcher.log" 2>&1
MATRIX_RC=$?
printf '%s\n' "$MATRIX_RC" > "$RUN_ROOT/matrix_exit.txt"
npu-smi info > "$RUN_ROOT/occupancy_after.txt"
```

This gives ten lanes. `baseline` and `pfa_nz_weights` use fullgraph static
TorchAir. `eager_pfa` and both `unpad_d128` lanes use raw eager full-stack
dispatch. Do not call an eager-versus-compiled difference an isolated operator
improvement. Compare weights within the same path, and attention contracts
within eager mode. If a healthy device rejects a candidate contract, retain
that compatibility result; never substitute another operator silently.

After that matrix completes successfully, isolate D128 padding inside the same
compiled PromptFA path, using the same capture and a separate output directory:

```bash
"$PYTHON" -u 11_mineru_2_5_pro_inference/run_production_attention_matrix.py \
  --capture-dir "$RUN_ROOT/capture" --cache-root "$RUN_ROOT/vision_cache" \
  --output-dir "$RUN_ROOT/padding_matrix" \
  --routes crop_0_bucket_768,crop_1_bucket_3072 --variants pfa_d128 \
  --steps 30 --timeout-s 900 --profile \
  > "$RUN_ROOT/padding_launcher.log" 2>&1
PADDING_RC=$?
printf '%s\n' "$PADDING_RC" > "$RUN_ROOT/padding_exit.txt"
```

Existing `pfa_approx` / `pfa_d128_approx` variants select the owned GE converter
for `innerPrecise=4`, a 310P-only contract that cannot be validated on 910B.
If the native control and primary matrix pass, run those as a separately labeled
310P-only secondary matrix with the same routes/steps/profile and fresh output
directory:

```bash
"$PYTHON" -u 11_mineru_2_5_pro_inference/run_production_attention_matrix.py \
  --capture-dir "$RUN_ROOT/capture" --cache-root "$RUN_ROOT/vision_cache" \
  --output-dir "$RUN_ROOT/approx_310P_only_matrix" \
  --routes crop_0_bucket_768,crop_1_bucket_3072 \
  --variants pfa_approx,pfa_d128_approx --steps 30 --timeout-s 900 --profile \
  > "$RUN_ROOT/approx_310P_only_launcher.log" 2>&1
APPROX_RC=$?
printf '%s\n' "$APPROX_RC" > "$RUN_ROOT/approx_310P_only_exit.txt"
```

Do not invent an `inner_precise` Python keyword or report the
910B unsupported-device skip as a performance measurement. Preserve candidate
feature drift alongside its timings. Report any converter rejection without
changing the installed software or tracked source.

Each warm sample includes all 32 blocks, their linears, rotary, norms, residuals
and attention. Initial CPU processing/H2D, patch embedding, positions, merger,
weight conversion and compilation are outside replay timing. The blocks are
the production `vision_transformer_blocks` boundary; this is not page/s or
complete crop pipeline throughput. Useful rate counts each real raw vision
token **once per entire 32-block forward**, excludes filler, and divides by
total elapsed time. Keep synchronized wall and NPU-event rates separate.

Lengths for unpad are prepared once before timing; the op's per-layer CPU
metadata consumption stays timed. Stock vLLM's per-forward length construction
and device-to-CPU copy are not reproduced. This is its operator contract
inside our actual vision stack, not a stock vLLM engine throughput result.

Record relative L2, cosine, max/mean absolute feature error independently of
timing. Candidate drift does not veto its timing. Nonfinite outputs, baseline
failure, nondeterministic repeated execution or compilation inside the warm
timer invalidate the corresponding run. NZ lanes must retain format29 and
exact logical weight values. Inspect **actual kernel input formats**; the
loaded descriptor alone cannot prove compiled matmuls consume NZ.

## Report to Luka

Return the run directory, actual commit, exact commands and exit codes,
runtime/device versions, checkpoint and image hashes, crop grids and masks.
Include every completed or failed lane, not just improvements:

- Warm mean/p50/p90 wall and NPU-event milliseconds per full encoder; useful
  raw vision tok/s and physical padded tok/s; 30 sample arrays and timing gates.
- ND/NZ stored formats, conversion/setup times, actual matmul input formats,
  TransData counts and conversion directions; feature drift separately.
- Profiles contain **three complete forwards**: normally 96 attention calls
  and 384 projection matmuls. Divide totals by three for per-forward costs.
  Show attention, linears, rotary, LayerNorm, conversions and launch/host gaps.
  Use kernel duration_us consistently. Do not add overlapping AICore/AIV
  engine counters or compare them to elapsed duration_us as the same quantity.
- Diagnose which stage accounts for the 310P gap on matched crops. Compare
  with the committed 910B evidence, with execution mode and inputs explicit.
  Two crop diagnostics do not establish corpus-average or page throughput.

Preserve manifest, result JSONs, launcher/lane logs, exact commands, parsed
profiles and operator/kernel CSVs. Leave captures and compiler caches locally;
do not push binary caches. The agent has no Git write authorization. Luka will
relay the report back to the authoring lane.
