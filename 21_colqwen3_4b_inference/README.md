# 21 — Ops-ColQwen3-4B HF reference and local eager model

**Current performance entrypoint:** `run_hr_evaluation.py`, defaulting to the
fixed 111-page/32-query HR development workload. See the
[single-observer contract](VIDORE_V3.md#current-performance-testing-contract-experiment-21-only).
Timing/logging is always enabled; only profiler capture is optional. Full HR
requires an explicit user request. Historical isolated-stage scripts below are
retained as evidence, not used for new performance conclusions: all new timing
and profiling goes through complete real-image/query pipeline executions.

Validated on one 910B2 at `2b1cafc3`: 111 pages + 32 queries, 31 CPU contract
tests, exact observed/control score parity, and three real-item profiler traces.
Final ABBA means showed +0.72% page time and +2.04% total job time; the prior
repeat showed no consistent whole-job difference. These are noisy measured
overheads, not a guarantee of zero cost. See [validation evidence](references/pipeline_observer_910b/README.md).
No full-HR rerun or 310P validation was performed for this instrumentation.

The reference uses checkpoint-supplied Hugging Face `AutoModel` and `AutoProcessor`
on Ascend NPU. That lane has no custom model implementation, vLLM, TorchAir compilation, NZ
conversion, quantization, or processor-resolution overrides. FP16 and HF eager
attention are explicit; this is a correctness anchor, not an optimized path.

Model: https://huggingface.co/OpenSearch-AI/Ops-Colqwen3-4B

The checkpoint's reviewed local Python code is loaded with
`trust_remote_code=True, local_files_only=True`. Its processor owns query
augmentation and image prompts; its model owns projection, masking and
normalization. Full-dimensional embeddings (2560) are retained. MaxSim uses
the supplied processor with an explicit NPU device, avoiding its auto-CPU fallback.
The model card's CUDA FlashAttention-2 example is not used on Ascend.

## Earlier work

`qwen_vllm_research/02_ops_colqwen3_embedding_smoke` recorded a 910B vLLM-Ascend
query/synthetic-image smoke with block size 128. It was not a Hugging Face
baseline or retrieval evaluation. The cached model remains at
`/workspace/models/Ops-Colqwen3-4B`. No 310P result is claimed here.

## Run on 910B

Prepare the experiment-owned environment once:

```sh
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  bash 21_colqwen3_4b_inference/setup_environment.sh
```

This inherits the container's torch/torch_npu binaries but locally pins
Transformers 4.57.1 (the checkpoint's recorded version), tokenizers 0.22.2,
HF Hub 0.36.2 and PyArrow 21.0.0. Existing environments are unchanged. Inherited
vLLM packages require Transformers 5.5.4 and are deliberately not used here;
this is an HF-only environment, not a vLLM environment.

Transformers 5.5.4 was tested: text inference worked, but image inference failed
because the checkpoint wrapper does not forward the newly required
`mm_token_type_ids`. Pinning the reference version avoids patching the model.

Use a healthy, free device selected by `npu-setup`; check health before running.
The current container can be reached through host SSH plus `docker exec`;
the historical port-22021 SSH shortcut may be unavailable. Source changes
still follow local commit/push and server pull only.

```sh
source npu-setup
cd /workspace/repos/paddle_ocr_vl_npu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/workspace/.cache/huggingface
RUN_ROOT="tmp/21_colqwen3_4b_inference/hf_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"
set -o pipefail
/workspace/venvs/colqwen3_hf_py312/bin/python \
  21_colqwen3_4b_inference/run_hf_baseline.py \
  --model /workspace/models/Ops-Colqwen3-4B \
  --output-dir "$RUN_ROOT/output" --hash-weights \
  2>&1 | tee "$RUN_ROOT/run.log"
RUN_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$RUN_EXIT" > "$RUN_ROOT/exit_code.txt"
```

Default inputs are two text queries and two real committed document crops.
Images run individually; queries form one batch. No synthetic image, crop
resize override, model modification, or CPU model fallback is permitted.
`--images` and `--queries` accept explicit alternative inputs.

Outputs include source/model/input hashes, package versions, exact loading
diagnostics, embedding dimensions and norms, repeat differences, MaxSim scores,
and CPU `.pt` copies of processor tensors and embeddings for future parity.
First forward and three warm repeats are reported separately. Forward timing
uses NPU synchronization and excludes preprocessing, transfer, and saving.
No ranking/accuracy conclusion follows from these smoke inputs or scores.

Reject missing/unexpected checkpoint weights, nonfinite/empty embeddings,
unit-norm error over 0.02, or repeat max-absolute drift over 0.001. Record
failures rather than silently patching checkpoint code or changing attention.

## Status

Validated on one **910B2 (physical device 7), 2026-10-01**. Both the two-crop
smoke and a two-full-page ViDoRe smoke passed with FP16, Transformers 4.57.1,
checkpoint-supplied model/processor, and true uncompiled eager attention.
All 715 checkpoint tensors matched the expected keys/shapes. Embeddings were
finite, approximately unit-normalized, and bit-exact across three warm repeats.

The two original ViDoRe pages took 0.4734 s and 0.4581 s per warm image forward
(B1); preprocessing, transfers and scoring are excluded. This is **not** a
full-corpus throughput measurement or retrieval accuracy evaluation.

See [validated results and caveats](references/RESULTS.md),
[full-page run JSON](references/vidore_smoke_910b/result.json), and
[crop run JSON](references/hf4571_smoke_910b/result.json).
Processor inputs and embeddings remain on the server for future parity tests.
Existing experiments and their environments are unchanged.

Target evaluation dataset and download procedure: [ViDoRe v3](VIDORE_V3.md).

## Local modeling replacement

`local_modeling_colqwen3.py` owns the complete still-image/text embedding forward
in ordinary PyTorch, with no Transformers imports. `config.py` validates the
specific checkpoint architecture instead of guessing missing model dimensions.
The unchanged HF processor remains the preprocessing/tokenization boundary.

The implementation preserves the 24-layer D64 vision tower, learned spatial
position interpolation, 2×2 merger, DeepStack taps at blocks 5/11/17 and their
injection after text layers 0/1/2, 36-layer causal GQA text backbone with Q/K
RMSNorm, interleaved MRoPE, and the learned 2560-dimensional retrieval projection.
Attention uses explicit matmul and FP32 softmax, as in the HF eager reference.
No LM head, token generation, KV cache, TorchAir, quantization, or NZ conversion
is involved. Video input, generation/cache arguments, and alternative model
architectures are intentionally unsupported; they must not be silently accepted.

`LocalColQwen3.from_pretrained` loads all safetensors weights strictly, including
projection bias and DeepStack-specific normalization shapes. Its forward accepts
the checkpoint processor's `input_ids`, `attention_mask`, padded `pixel_values`
and `image_grid_thw`, returning `[batch, sequence, 2560]` normalized embeddings.

### Parity validation

The validation runner imports HF only for the independent reference and unchanged
processor. It compares saved anchors, fresh left/right-padded text and image
batches, exact position IDs, 22 selected intermediate module outputs (when used),
final embeddings, repeat stability and MaxSim scores. It saves processor tensors
and both outputs for each case. Instrumented wall times include CPU trace copies
and are **not** performance measurements.

```sh
source npu-setup
cd /workspace/repos/paddle_ocr_vl_npu
/workspace/venvs/colqwen3_hf_py312/bin/python -u \
  21_colqwen3_4b_inference/run_local_parity.py \
  --model /workspace/models/Ops-Colqwen3-4B \
  --reference-dirs \
    tmp/21_colqwen3_4b_inference/hf4571_smoke_b25734d4/output \
    tmp/21_colqwen3_4b_inference/vidore_smoke_247c121b/output \
  --output-dir tmp/21_colqwen3_4b_inference/local_parity_new/output
```

Those anchor paths are specific to our 910B workspace. Elsewhere, regenerate
anchors with `run_hf_baseline.py`; do not substitute synthetic image tensors.
CPU unit/structural tests (`python3 -m unittest discover -s
21_colqwen3_4b_inference -p 'test_*.py'`) are separate from NPU inference parity.

**Local eager validation passed on 910B2, physical device 7, 2026-10-01,
source `038ce189`.** All 10 cases had bit-exact embeddings, all 168 captured
intermediate comparisons were exact, both MaxSim matrices were exact, and all
position IDs matched. Both saved HF anchor groups were reproduced exactly.
Maximum active embedding-row norm error was 0.000452, with zero padding rows.
See the [successful report](references/local_parity_910b/result.json),
[command](references/local_parity_910b/command.txt), and
[log](references/local_parity_910b/run.log).

The [initial failed report](references/local_parity_910b/initial_failed_result.json)
is retained. It isolated a numerical-fidelity pitfall: computing rotary inverse
frequencies on NPU differed from HF's CPU FP32 initialization by up to
3.73e-9 (vision) / 7.45e-9 (text). Those small differences propagated through
the network. Matching CPU initialization restored exact parity without relaxing
any tolerance or changing the reference. Keep this initialization detail when
adding later optimizations; do not infer bit parity merely from equivalent math.

This is not a 310P result, a full ViDoRe retrieval score, or an optimized runtime.
The next gate is broader retrieval evaluation; speed work must remain a separate
lane from this eager correctness anchor.

## Prepared transformer stages

`prepared_prefill.py` adds a separate exact-shape B1 path, sharing the original
weights without modifying `local_modeling_colqwen3.py`:

- Eager vision preparation: validation, image grid, patch projection, absolute
  position interpolation, CPU-initialized FP32 rotary constants, cos/sin and mask.
- `PreparedVisionStage`: all 24 transformer blocks, returning a fixed tuple of
  final raw hidden states and the three raw DeepStack taps.
- Eager multimodal preparation: all four mergers, token embeddings, initial
  image scatter, multimodal positions, causal mask and text rotary tensors.
  DeepStack features become three `[1, S_text, 2560]` tensors, zero outside image
  positions; no dynamic boolean indexing remains in the text graph.
- `PreparedTextStage`: all 36 transformer layers, dense additions after layers
  0/1/2, and final RMSNorm. It returns every token's hidden state; no KV writes.
- Eager finish: retrieval projection, normalization and output masking.

Both stages use 2D token-matrix Linear calls and explicit `[B*H,S,D]` attention
BMMs with FP32 softmax. No fused attention, NZ weights, quantization, buckets,
resolution overrides or silent eager fallback are enabled. Initially only one
unpadded query/image sequence is accepted. Query rows can be trimmed from a
saved padded anchor by the benchmark; this is recorded as a B1 comparison and
also checked against its saved HF embeddings.

Vision GELU explicitly uses `npu_gelu(approximate="tanh")` / GE `GeluV2` on NPU;
the installed TorchAir `aten.gelu` converter drops the approximation argument.
The compiler uses `compile_fusion_switch.json` to disable only
`AddLayerNormFusionPass`. This is a GE process-global policy, applied consistently
to vision and text, not a per-call toggle. All other default compiler settings
remain unchanged. The switch file is included in cache identity.

Why the fusion switch: the initial compiled crop failed the unchanged embedding
gate (max absolute error 0.006744). A full-stack diagnostic with first-block taps
found exact QKV, rotary, attention, output projection and residual-add output;
the first discrepancy was `norm2` (max absolute error 0.0078125). The generated
OM contained `AddLayerNorm` fusions. Explicit GELU alone did not change that
failure. Disabling the add/LayerNorm fusion restored bit-exact crop embeddings.
This isolates a numerical-fidelity issue at that fused boundary; it is not
evidence that attention or model weights were wrong. Extra diagnostic outputs
can affect optimization, so acceptance is measured again with the ordinary
production-stage outputs, not just the diagnostic graph.

`bench_prepared_prefill.py` checks reference eager -> prepared eager before
compiling each case. Hidden-state differences are reported as diagnostics;
compiled acceptance checks final embeddings against both the original local
model and saved HF reference (`atol=rtol=0.002`), finite/unit-norm output, and
MaxSim comparisons (`atol=0.02, rtol=0.001`). These checks are not a substitute
for full retrieval evaluation. Use `--eager-only` to validate all chosen shapes
before spending time compiling any of them.

```sh
source npu-setup
cd /workspace/repos/paddle_ocr_vl_npu
/workspace/venvs/colqwen3_hf_py312/bin/python -u \
  21_colqwen3_4b_inference/bench_prepared_prefill.py \
  --model /workspace/models/Ops-Colqwen3-4B \
  --anchors \
    tmp/21_colqwen3_4b_inference/hf4571_smoke_b25734d4/output/queries.pt \
    tmp/21_colqwen3_4b_inference/hf4571_smoke_b25734d4/output/image_01.pt \
    tmp/21_colqwen3_4b_inference/vidore_smoke_247c121b/output/image_01.pt \
  --cache-root .runtime_cache/21_colqwen3/prepared \
  --output-dir tmp/21_colqwen3_4b_inference/prepared_new \
  --repeats 5
```

The persistent cache key includes stage, tensor shapes/strides/dtypes, model
config/path and weight-file metadata, source hashes, chip and runtime versions.
Every signature receives a distinct Dynamo entry code object. Reuse the same
cache root on restarts; do not clear it. `om_present_before` reports artifact
presence, not proof that the runtime successfully loaded it.

`PREFILL` phase markers distinguish wrapper creation from first graph execution.
First-call time is reported separately from a subsequent warmup and repeated
warm stage calls. Warm timing excludes preparation, mergers, output projection,
comparison and serialization; it is **not end-to-end pages/s**. Five repeats are
an initial latency check, not a statistically robust tail-latency study.

### 910B validation and remaining numerical question (2026-10-01)

The prepared eager path is bit-exact on an 18-token query, a 512-vision-token
crop / 142-token text sequence, and a 4960-vision-token page / 1254-token text
sequence. Compiled text was exact at all three lengths across the initial runs.
With `AddLayerNormFusionPass` disabled, the ordinary compiled crop path is also
bit-exact. The large-page path still exceeds the provisional elementwise
embedding tolerance; **this is not a demonstrated retrieval-quality failure**.
The full compile matrix and warm-restart validation are not yet complete.

The [ordinary-stage report](references/prepared_prefill_910b/add_layernorm_off/result.json)
records final page embedding max/mean absolute error 0.0575285 / 0.000254219,
RMSE 0.000921809 and global cosine 0.99891233. The CPU-FP32 post-hoc MaxSim
comparison using the two saved crop-smoke queries changed scores by +0.002110
(+0.0212%) and +0.002067 (+0.0245%). That is only two queries against one page,
not a ViDoRe ranking/evaluation result. Some individual embedding rows have
larger discrepancies; aggregate cosine alone is not an acceptance criterion.

The [full-stack attention diagnostic](references/prepared_prefill_910b/full_probability_diagnostic/result.json)
at `c439ce1d` establishes, on that real page:

- First-block normalization, complete QKV and rotary outputs are exact.
- All **393,625,600** FP16 probabilities consumed by P×V are exact, not merely
  sampled rows. Sampled QK, scaled/masked scores and FP32 probabilities are exact
  too. The complete V tensor is exact through the QKV check.
- P×V is the first differing operation: `[16,4960,4960] @ [16,4960,64]`.
  Output max/mean absolute error is 0.0009765625 / 4.18687e-9; cosine is
  0.9999999999749. This is a very sparse initial difference that propagates
  through the remaining vision blocks and text model.
- Extra diagnostic outputs retain the same final vision max/mean error as the
  ordinary production-stage graph. All compared outputs are finite.

This isolates the first difference to compiled versus eager matrix-product
execution. A different reduction/tiling order at long K is a hypothesis, not a
proven kernel defect. An extra `ZZMatMulToMatmulV3FusionPass` was observed in the
large graph, but that observation does not establish it as the cause: the
first QKV projection is exact. Do not disable unrelated fusions on that basis.

Historical analogues were checked: experiment 07's native-D72 compiled
PromptFA drift/NaNs and D80 workaround (`5e698684`), MinerU's explicit 3D-BMM
manual attention (`396a36de`), and Paddle's 4304→4352 MLP alignment. None is
directly the current contract: ColQwen uses manual 3D BMM, D64, and a 4096-wide
vision MLP. LayerNorm fusion was a separate first issue, fixed for the crop;
the remaining full-page P×V discrepancy needs model-level ranking/score
validation before either accepting it or paying to enforce stricter arithmetic.

Evidence also preserves the [initial failure](references/prepared_prefill_910b/initial_compiled_failed.json),
[GELU-only failed attempt](references/prepared_prefill_910b/geluv2_only_failed.json),
and [initial LayerNorm diagnostic](references/prepared_prefill_910b/vision_diagnostic.json).
No tolerance was silently widened and no reference weights/model code changed.

## Portable prefill optimization candidates

`optimized_prefill.py` is a separate opt-in stage implementation. It does not
modify `local_modeling_colqwen3.py` or the prepared manual-attention baseline.
Its initial candidates are:

- FP16 PromptFA with BNSD inputs, no host sequence-length lists, no paged cache,
  and no 910B-only `npu_fusion_attention`. Vision is full bidirectional attention
  at the exact unpadded image length, so it needs no mask. Text uses a prepared
  square bool causal mask, full pre/next windows and `sparse_mode=0`.
- Native compact GQA (32 query / 8 KV heads) following experiment 13's
  directly 310P-tested contract (`73339436`). `--gqa repeat` is a separately
  keyed, explicit compatibility experiment, never an automatic fallback.
  Installed CANN/torch-npu support still needs checking on the target 310P.
- Setup-fused text Q/K/V and gate/up weights: seven projections become four
  per text layer. Splitting, per-head Q/K normalization, rotary, original scale
  and activation order are preserved. Vision QKV was already fused.
- Explicit FP32 vision LayerNorm statistics/normalization, cast back to FP16
  before the separate FP16 affine operations (the MinerU recipe), to
  avoid the fused LayerNorm/MatMul path that failed on 310P in MinerU. This is
  still a numerical candidate, not assumed identical to `nn.LayerNorm`.
  `--vision-norm module` provides an ablation.
- Optional setup-only FRACTAL_NZ Linear weights. Every candidate weight must
  report format 29; failure is fatal, not a hidden native-format fallback.
  Embeddings, mergers, norms and retrieval projection stay unchanged. Native
  is the default because NZ did not materially help the 910B reranker.

No quantization, resolution reduction, token dropping, sequence truncation or
model architecture change is involved. This remains exact-shape B1 prefill;
reusable buckets, batching and an optimized end-to-end retrieval service are
separate work. Candidate fusion/NZ copies coexist with the reference model in
this comparison harness, so its memory footprint is not a minimal serving
footprint. The benchmark records PyTorch allocator memory, not total device use.

The optimized runner also defaults to `--patch-embedding linear`;
`--patch-embedding conv3d` retains the original projector. ColQwen's patch
Conv3D consumes one pre-extracted `[3,2,16,16]` patch per sample and produces
one spatial output, so its weight can be flattened once from
`[1024,3,2,16,16]` to `[1024,1536]` and applied with `F.linear`, preserving
the bias and C/T/H/W flatten order. This is an algebraic replacement, not a
literal 1x1 convolution or a change to patch resolution.
`patch_embedding.py` leaves the reference model untouched. Projection remains
outside the transformer graphs, so this change does not invalidate their caches.
The runner compares both projectors with alternating synchronized warm calls
(`--patch-repeats 50`), and records downstream Linear-versus-Conv3D embeddings
through the same candidate transformer path as an isolated numerical check.

Validated at `adbf0b6d` on **910B2 physical NPU 7**, FP16, native weights,
internal formats enabled, with 50 alternating warm measurements per projector:

| Input | Vision tokens | Conv3D mean | Linear mean | Projection speedup |
|---|---:|---:|---:|---:|
| Table crop | 512 | 0.1687 ms | 0.1280 ms | 1.32x |
| ViDoRe full page | 4960 | 0.1900 ms | 0.1586 ms | 1.20x |

Patch outputs and final embeddings through the same compiled candidate
transformers were **bit-exact between Conv3D and Linear** for both images.
The candidate's existing MaxSim differences versus the untouched reference
remained -0.0563% and -0.1066%, with unchanged two-document ranking. The patch
replacement itself added no observed drift. The absolute saving is only
0.03–0.04 ms/image, not a meaningful end-to-end throughput gain by itself.
These are synchronized wall timings, not isolated kernel-device timings.
Query and full-page transformer graphs reused existing caches; missing crop
variants compiled normally. No cache was deleted. The run exited 0;
[raw report](references/patch_linear_910b/result.json),
[command](references/patch_linear_910b/command.txt), and
[log](references/patch_linear_910b/run.log) retain the measurements.
All 25 CPU algebra/contract tests passed. 310P validation and full ViDoRe v3
retrieval evaluation remain pending.

`bench_optimized_prefill.py` measures the same tensor-only stage boundaries as
the prepared benchmark, including manual eager, candidate eager, and candidate
compiled timings. All first calls/setup are separate from warm means. Cache
identity includes source, options, formats, runtime/chip and exact input shape.
Reuse the cache root; do not clear it when switching candidates.

Validation deliberately distinguishes **numerical diagnostics** from
**retrieval accuracy**: the untouched reference must match its HF anchor;
candidate outputs must be finite and unit-normalized; all embedding/hidden
differences remain in the report, but elementwise allclose is not the acceptance
gate. Query/document MaxSim differences are compared with the original diagnostic
limits (`atol=0.02`, `rtol=0.001`), but exceeding them does **not** fail the run
or establish an accuracy regression. The score report's `passed` fields refer
only to these numerical tolerances. Completed runs are labeled
`completed_experimental`, with quality explicitly not yet evaluated on ViDoRe v3.
This policy follows Luka's explicit approval on 2026-10-01; earlier reports
that stopped at the score threshold are retained as historical diagnostics.
Both reference and candidate scores are calculated
post hoc in FP32 on CPU from their NPU-produced embeddings. Rankings are also
reported. ViDoRe v3 evaluation, not these few anchors, is the quality test.

```sh
# After source npu-setup, from the repo root on the 910B workspace:
PYTHON=/workspace/venvs/colqwen3_hf_py312/bin/python
"$PYTHON" -u 21_colqwen3_4b_inference/bench_optimized_prefill.py \
  --model /workspace/models/Ops-Colqwen3-4B \
  --anchors \
    tmp/21_colqwen3_4b_inference/hf4571_smoke_b25734d4/output/queries.pt \
    tmp/21_colqwen3_4b_inference/hf4571_smoke_b25734d4/output/image_01.pt \
    tmp/21_colqwen3_4b_inference/vidore_smoke_247c121b/output/image_01.pt \
  --repeats 10 --cache-root .runtime_cache/21_colqwen3/prepared \
  --output-dir tmp/21_colqwen3_4b_inference/optimized_new
```

First add `--eager-only` with a distinct output directory to test the candidate
before compiling. For projection fusion ablation, add `--unfused`. For a fair
native/NZ comparison, use `--enable-internal-format` in **both** separate
processes and `--weight-format fractal_nz` only in the NZ process. Record every
variant separately; do not transfer a 910B speed or validity result to 310P.

### Initial optimized measurements on 910B2

Source `c6cfc8f8`, physical NPU 7, FP16/B1, ten warm repetitions, native weights,
native GQA, fused text projections and MinerU-style manual vision LayerNorm:

| Stage | Tokens | Manual eager, same run | Candidate eager | Candidate compiled | Previous manual compiled* |
|---|---:|---:|---:|---:|---:|
| Query text | 18 | 68.67 ms | 70.72 ms | 14.50 ms | 13.70 ms |
| Crop vision | 512 | 28.57 ms | 43.40 ms | 9.30 ms | 8.84 ms |
| Crop text | 142 | 74.29 ms | 76.31 ms | 22.02 ms | 21.61 ms |
| Page vision | 4960 | 309.59 ms | 60.89 ms | 51.93 ms | 271.49 ms |
| Page text | 1254 | 120.08 ms | 76.67 ms | 62.70 ms | 101.73 ms |

*The previous manual compiled column is from the earlier prepared-stage runs,
not an alternating same-process ablation. The small-shape candidate is not an
improvement in those observations; no automatic length-based routing is claimed.
The large-page transformer sum is 114.62 ms versus the previous 373.22 ms
(3.26x). Neither sum includes eager preparation, mergers or retrieval projection,
and neither is end-to-end page throughput. Fusion's individual contribution is
not isolated by this combined-candidate comparison.

Every compiled stage output was bit-exact against its **own optimized eager**
counterpart, including all four vision outputs. Compared with the unchanged
reference, the two MaxSim changes were -0.0563% and -0.1066%; both document
rankings agreed. The report predates the status-label change and says
`passed_score_smoke`, which is not a retrieval-accuracy claim. Full ViDoRe v3
evaluation remains pending. See the [report](references/optimized_prefill_910b/native/result.json)
and [command](references/optimized_prefill_910b/native/command.txt).

The [initial eager report](references/optimized_prefill_910b/initial_eager/result.json)
used FP32 LayerNorm affine before changing to MinerU's FP16 affine boundary.
It stopped at the old numerical score threshold (-0.328% on the page, unchanged
ranking). That recorded `failed` status means only the historical threshold was
exceeded, not that ViDoRe accuracy failed.

This comparison process held both the reference and candidate weight sets:
13.72 GB allocated after setup and 18.05 GB peak PyTorch allocation on the page
(20.74 GB reserved). CANN/driver memory is additional. In particular, do not
assume the dual-path full-page harness fits a 310P just because the production
model would; start with small inputs there. No ColQwen optimized 310P inference
has been validated from this machine.

Fresh-process [cache reuse](references/optimized_prefill_910b/warm_native/result.json)
was checked at `aac6216a`: all five graphs used the same existing cache paths.
First-call times were 4.94 s for the initial query, then 0.585/0.895 s for crop
vision/text and 0.642/0.939 s for page vision/text (versus 31–52 s cold calls).
Warm page stage means remained 51.80/62.56 ms. No fresh cache root or cache
deletion was used. Cache keys intentionally change for changed source/options;
the later format-report compatibility adjustment is separately keyed.

At `fd4fad8f`, separate native/NZ processes both enabled internal formats:

| Stage | Tokens | Native | FRACTAL_NZ |
|---|---:|---:|---:|
| Query text | 18 | 14.42 ms | 15.39 ms |
| Page vision | 4960 | 51.57 ms | 51.26 ms |
| Page text | 1254 | 61.93 ms | 63.92 ms |

All 240 candidate Linear weights reported code 2 in the native control and
code 29 in the NZ run. The stage sum was 113.50 vs 115.18 ms, so this sequential
single-device measurement does not support adopting NZ on 910B. It remains an
explicit 310P candidate, not a claimed cross-chip speedup. The page MaxSim
deltas versus reference were -0.1066% (native) and -0.2111% (NZ), with finite,
normalized outputs. NZ introduced additional compiled-versus-own-eager drift;
the earlier native bit-exact observation must not be generalized to NZ.
These two-process format runs each had one query and one document, so their
trivial one-document ranking is not informative. See the
[native report](references/optimized_prefill_910b/native_internal/result.json) and
[NZ report](references/optimized_prefill_910b/nz/result.json).
The NZ comparison harness peaked at **21.17 GB allocated / 24.36 GB reserved**
by PyTorch alone. Do not send that full dual-reference page run unchanged to a
310P: first use small shapes, or add a staged-reference/candidate-only harness
that releases the original projection weights before candidate execution.

Local checks: 22 CPU algebra/contract tests passed, including fused-weight
non-mutation, mocked native/repeated GQA, causal-mask semantics, strict format
report decoding and full tiny-model candidate algebra. Those tests do not
validate NPU kernels. Real 910B runs above cover the default native-GQA/fused
candidate and both weight formats; `--unfused`, `--gqa repeat` and module-norm
ablations have not yet been benchmarked on NPU. No 310P run has been performed.
The subsequent HR-only retrieval evaluation is recorded below.

### Full HR English retrieval — 910B2

At `fbfaedc5`, the instrumented candidate evaluated **all 1,110 HR pages and
318 English queries**, using the exact pinned MTEB-format dataset associated
with the published score. FP16/B1, 2560 embedding dimensions, default image
resolution, native weights, PromptFA, fused text projections, manual FP32
vision LayerNorm statistics and Linear patch projection; no token reduction.

| Metric, percent | Published model | This run | Difference, percentage points |
|---|---:|---:|---:|
| nDCG@10 | 66.088 | 66.472 | +0.384 |
| Recall@10 | 70.720 | 70.847 | +0.127 |
| MAP@10 | 51.613 | 52.086 | +0.473 |

This closely matches the reference on HR, with no observed HR regression. It
does not establish a statistically significant improvement, full eight-domain
accuracy, or 310P behavior. Published numbers use CUDA FlashAttention2 rather
than our Ascend implementation. The scorer's check against the checkpoint's
FP32 MaxSim formula differed by only `9.54e-7` on the tested first column.

The complete run exited 0 in **462.9 s (7m43s)**: 23.1 s setup, 33.9 s query
encoding, 397.5 s page encoding, 7.18 s scoring, plus final metrics/output.
Page encoding was **2.793 pg/s**, including preprocessing, transfers, validation
and saving; dividing page count by the entire job gives **2.398 pg/s**.
No new graphs were compiled. All 1,110 pages used optimized raw eager at
**5040 vision / 1274 text tokens** (1260 image + 14 prompt tokens). Only the
18-token query shape reused an existing compiled graph. This is **not** an
all-compiled page throughput result.

| Page section | Mean | p50 | p99 | Maximum | Wall tok/s |
|---|---:|---:|---:|---:|---:|
| Image preprocessing | 157.28 ms | 155.37 ms | 207.61 ms | 293.59 ms | — |
| Vision preparation | 14.72 ms | 14.38 ms | 18.88 ms | 67.27 ms | 342,307 |
| Vision transformer | 64.84 ms | 64.68 ms | 67.02 ms | 101.93 ms | 77,725 |
| Text preparation / mergers | 6.82 ms | 6.72 ms | 7.65 ms | 70.19 ms | 186,688 |
| Text transformer | 80.66 ms | 79.41 ms | 91.47 ms | 105.15 ms | 15,794 |
| Validation and serialization | 23.83 ms | 23.20 ms | 33.85 ms | 46.63 ms | — |
| Whole page | 357.85 ms | 353.99 ms | 419.88 ms | 655.21 ms | — |

NPU-event interval throughput for vision/text transformers was 78,110 / 15,852
tok/s; these intervals include eager launch gaps, not just kernel-active time.
Totals were 5,594,400 vision tokens and 1,414,140 text tokens. Preprocessing
accounted for about 44% of measured page time, so transformer-only speeds are
not a proxy for end-to-end throughput. All 14,755 section starts had matching
finishes; no stall was observed.

Evidence: [result](references/hr_910b/result.json),
[per-item timings](references/hr_910b/items.jsonl),
[per-query metrics](references/hr_910b/per_query_metrics.json),
[run log](references/hr_910b/run.log), and
[command](references/hr_910b/command.txt).
The 6.8 GB embeddings, full score matrix and rankings remain on the 910B under
`/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/hr_validated_run/full/output/`;
they are not committed to Git. See [the run protocol](VIDORE_V3.md) for replay.
The corrected 8-page/8-query NPU preflight passed before this run. Local tests
now total 29, including score reduction and progress-clock regression coverage.

### Multilingual HR evaluation

`run_hr_evaluation.py --workload full --languages all` evaluates English,
French, German, Italian, Portuguese and Spanish (318 queries each, 1,908 total)
against the same 1,110 pages. English remains the default; the fixed development
workload remains English-only. FP16/B1 encoding and the observer are unchanged.
Page embeddings stay in CPU memory and are reused across all languages. Scoring
runs one language at a time, retaining the original 318-query FP32 MaxSim shape
for English and bounding temporary NPU memory.

First run `download_hr_reference.py --root <dataset-root> --languages all`.
The revision and SHA256 hashes are pinned. All six upstream corpus and qrels
files have identical content hashes, so the downloader retains one English-named
copy of those shared components and downloads only the additional query files.
The loader verifies query counts, unique IDs and relevance references.

Results include `metrics_by_language`, equally weighted `macro_metrics`, the
matching per-language published references, and timings by language. With 318
queries in each language, the language macro mean equals the query mean.
`query_languages.json` identifies every row in the combined score matrix.
A full HR multilingual score is still only one domain of ViDoRe v3.

### Full HR, all six languages — 910B2

At `f5d58289`, the multilingual evaluator processed the same **1,110 pages
and all 1,908 queries** (318 per language). This uses the same FP16/B1 model,
2560 dimensions, default resolution, optimized attention and warm-cache policy
as the English baseline. Every page had 5040 vision tokens, 1260 merged image
tokens and 1274 text-transformer tokens. The page corpus was encoded once.

| Language | Recall@10, % | Published | nDCG@10, % | Published | MAP@10, % | Published |
|---|---:|---:|---:|---:|---:|---:|
| English | 70.8470 | 70.7200 | 66.4722 | 66.0880 | 52.0862 | 51.6130 |
| French | 66.3101 | 66.3850 | 61.2989 | 61.4660 | 46.8801 | 47.0150 |
| German | 65.3382 | 65.2500 | 60.7893 | 60.8730 | 46.7263 | 46.8110 |
| Italian | 65.0865 | 65.0270 | 60.5123 | 60.4790 | 46.2067 | 46.1830 |
| Portuguese | 65.5788 | 65.5750 | 61.8161 | 61.7550 | 47.7917 | 47.6820 |
| Spanish | 65.1043 | 65.2500 | 60.5392 | 60.2190 | 46.5528 | 46.1550 |
| Average | 66.3775 | 66.3678 | 61.9047 | 61.8133 | 47.7073 | 47.5765 |

Mean deltas versus the published six-language reference were **+0.0097 percentage
points Recall@10, +0.0913 nDCG@10 and +0.1308 MAP@10**. Some individual metrics
were lower (including French), so this supports close agreement across HR
languages, not a universal or statistically established improvement.

Evaluation time was **582.0 s (9m42s)**; script import/CLI included was 584.2 s,
and the complete child-process wall time including shutdown was 592.4 s (9m52s).
Setup took 24.4 s, page encoding 363.8 s (3.051 pages/s), query encoding 165.6 s,
and scoring 22.4 s. The job was 25.7% longer than the historical 462.9 s English
run, but that older run also used the earlier observer and embedding
serialization; this is not an isolated measurement of language overhead.
No new graphs were compiled: all pages were optimized eager, and only one
18-token query reused a compiled graph.

All **34 tests passed**. The English development check retained bit-exact
scores, IDs, selection, rankings and per-query metrics. More strongly, the
full 318-query English submatrix matched the historical full-English matrix
**bit for bit**, with all per-query metrics unchanged. The observer closed
with 56,036 matched section start/finish pairs and zero pending device events.

See [summary](references/hr_multilingual_910b/report.txt),
[full result](references/hr_multilingual_910b/result.json),
[artifact audit](references/hr_multilingual_910b/artifact_audit.json), and
[exact commands](references/hr_multilingual_910b/command.txt).
This validates HR across six languages on 910B2; it is not an eight-domain
ViDoRe average or a 310P result.

### Image-token budget sweep

`--max-image-tokens` caps merged image tokens through the checkpoint processor's
existing resize path, before patch extraction. It sets maximum pixel area to
`budget * (patch_size * merge_size)**2`, retaining the original minimum area,
RGB conversion, interpolation, normalization, full-page content and prompt.
For this checkpoint, budgets 1280/640/320/160 correspond to pixel-area caps
1310720/655360/327680/163840. Fractions refer to area/token budget, not each
side's length. Aspect-ratio/grid rounding can produce fewer tokens than the cap.
Omitting the option preserves checkpoint defaults. Query processing, FP16 model
weights and 2560-dimensional outputs are unchanged.

The full multilingual eighth-budget run uses `--workload full --languages all
--max-image-tokens 160`. Run the same command at 320 and 640 for the remaining
sweep points, comparing with the existing 1280-budget baseline. Record actual
resized dimensions, vision/image/text token counts, page pg/s and query-level
metrics. The requested override is saved separately from the checkpoint's
original `processor_image_config`. Existing matching caches may be reused;
unseen shapes use optimized eager, with no new graph compilation.

### Resolution sweep results — 910B2, six-language HR

Ran budgets **160 first, then 320 and 640** at `cb9eef32`, comparing the
prior full-resolution run at `f5d58289`. Each run encoded the same 1,110 pages
and evaluated all 1,908 queries, with FP16/B1 and 2560 dimensions unchanged.

| Budget | Actual image tokens | Pages/s | Page speedup | Recall@10, % | Recall delta, pp | Evaluation time |
|---|---:|---:|---:|---:|---:|---:|
| Full (1280) | 1260 | 3.051 | 1.000x | 66.3775 | 0.0000 | 9m42s |
| Eighth (160) | 150 | 4.302 | 1.410x | 45.2612 | -21.1163 | 7m29s |
| Quarter (320) | 315 | 4.083 | 1.338x | 60.8235 | -5.5540 | 7m51s |
| Half (640) | 630 | 3.799 | 1.245x | 64.8780 | -1.4995 | 8m14s |

Half budget retained the most accuracy among the reduced settings: 24.5% more
pages/s with a 1.50 percentage-point Recall@10 loss. Eighth budget lost 21.12
points for 41.0% more pages/s. None preserved baseline recall. This evaluates
standalone retrieval; candidate Recall@K for a coarse-to-fine system has not
been measured. Timings are single-run observations on 910B2, and quality covers
HR only. Query time also varied, so total evaluation speedup cannot all be
attributed to page resolution. Evaluation time excludes process startup/shutdown.

All 35 unit/contract tests passed. The explicit 1280 override produced
bit-identical default-processor inputs on the checked page. Workload manifests
and IDs matched the baseline; scores were finite and observer events closed.
Every page's actual token count was checked against the requested cap. The
initial attempt stopped on page two when the processor consumed a reused
options dictionary; the successful source constructs fresh options per page.
No failed-attempt measurements enter this table, and no new graphs were compiled.

See [full report](references/hr_resolution_910b/report.txt),
[comparison data](references/hr_resolution_910b/summary.json),
[environment and source](references/hr_resolution_910b/preflight.json), and
[exact first-run command](references/hr_resolution_910b/budget_160.command.txt).
Each budget's full result and command are retained alongside these files.

### Development page-batch probe

`run_hr_evaluation.py --workload dev --page-batch-size N` explicitly enables
page batching; queries and scoring retain the B1 evaluator contract. Omit the
image-token override to use the full checkpoint resolution. The fixed dev
selection is 111 pages spread across the corpus and 32 English queries.

The separate `batched_prefill.py` path uses the existing FP16 PromptFA kernels
with a true batch dimension for both vision and text. It requires equal patch
and unpadded text lengths; different orientations with equal lengths are valid.
Pages retain corpus order. The final batch is padded to the requested size by
repeating the last valid page's processor tensors before NPU transfer. These
slots run through the model normally and their outputs are discarded; throughput
counts only real pages while including the padding compute cost. For B32, the
111-page dev set runs 128 slots (17 padded). No token padding is used.
No resolution reduction, prefetch, new graph compilation or cross-page attention
is introduced. B1 keeps its existing path and compatible cache policy. Timed
page encoding includes processing, transfers, model inference, CPU output
materialization, validation and observation. Peak allocated/reserved NPU memory
is recorded. Batches have one timing record with all member page IDs.

Before throughput runs, execute `check_hr_batch_parity.py` against the same
model/data. It compares full-resolution portrait/landscape pages at B1/B2/B3 and a padded B4 batch
with independent optimized B1 embeddings and checks processor inputs exactly.
Then run the fixed dev evaluation at B1, B2, B4, B8, B16 and B32, comparing
scores/rankings/metrics with the fresh B1 result. These are development results;
they do not replace full-corpus multilingual quality evaluation.

The initial real-page NPU parity gate found batch-dependent embedding drift,
despite exact processor inputs, rotary encodings and attention masks. Use
`--record-drift` on the parity probe only for an explicitly exploratory quality
and performance measurement: it records the unchanged strict equivalence verdict,
still requires finite embeddings/exact inputs, and strictly checks same-shape
cross-page isolation. A completed drift probe is not a passed equivalence gate.

### Fixed padded batch sweep — 910B2 development results

At `d8aedf0f`, tested the fixed 111-page / 32-English-query HR development set
at full checkpoint resolution (1260 actual image tokens per page). Each batch
size ran in a separate process; page throughput includes real-page preprocessing,
model inference, padding compute, CPU materialization and validation. B1 is a
fresh same-subset reference. Every final batch is padded to the requested size.

| Batch | Padded slots | Pages/s | Speedup | Page encoding, s | Evaluation, s | Peak allocated NPU, GiB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 2.992 | 1.000x | 37.103 | 69.721 | 13.126 |
| 2 | 1 | 3.113 | 1.041x | 35.656 | 66.573 | 13.334 |
| 4 | 1 | 3.134 | 1.048x | 35.414 | 66.607 | 13.886 |
| 8 | 1 | 3.274 | 1.094x | 33.901 | 65.335 | 14.987 |
| 16 | 1 | 3.062 | 1.023x | 36.256 | 68.015 | 17.196 |
| 32 | 17 | 2.672 | 0.893x | 41.546 | 73.122 | 21.609 |

Recall@10 was **90.8854% at every batch size**. B1/B2/B8/B16/B32 also shared
nDCG@10 71.5552% and MAP@10 63.4189%; B4 measured 71.6643% and 63.6012%.
These are small development-set scores against 111 candidate pages, not the
full-corpus multilingual scores. B8 was fastest in this single sweep (+9.4%);
B32 was slower than B1, including its 17 filler slots. No repeated-run confidence
estimate or general batch-size optimum is established.

**Strict embedding equivalence did not pass.** The mixed-orientation NPU
probe found exact processor inputs, rotary encodings and attention masks, with
numerical differences inside the forward pass. Same-shape replacement of a
batch neighbor left the first page's embeddings bit-exact, confirming isolation
for that check. Batched scores and some top-10 sets/orders changed. The dev
metrics above measure the consequence; they do not turn the failed numerical
equivalence verdict into a pass. The original strict thresholds are preserved.

All 38 unit/contract tests passed. Completed runs retained the exact workload
and IDs, finite scores, full-resolution tokens, and zero pending observer events.
No new transformer graphs were compiled. Setup/query/scoring time is included
in evaluation time; process startup/shutdown is recorded separately.

See [report](references/hr_batch_padded_910b/report.txt),
[comparisons](references/hr_batch_padded_910b/comparisons.json),
[numerical probe](references/hr_batch_padded_910b/parity.json), and
[stage diagnostic](references/hr_batch_padded_910b/diagnostic.json).
Exact commands, environment hashes, per-query metrics and rankings are retained
alongside them; full observer artifacts remain in the documented run directory.

### NPU profiler explanation of batch scaling

At `7b490801`, captured two warmed real-page batches each at B1, B8 and B32.
These use full resolution and no padded slots, allowing B32 kernel cost to be
examined separately from the development sweep's final-batch padding cost.

| Kernel duration, ms per real page | B1 | B8 | B32 |
|---|---:|---:|---:|
| Vision transformer | 63.53 | 62.60 | 68.73 |
| Text transformer | 75.56 | 59.48 | 64.87 |
| All kernels | 144.34 | 126.33 | 137.99 |

B8 reduces summed kernel duration/page by 12.5%. Launches/page fall from 3772
to 473, but vision work barely improves; most savings come from text. At B8,
attention and matrix multiplications account for about 64% of kernel duration.
B32 incurs higher per-page cast and pointwise operator costs even without
padding. This does not establish memory-bandwidth saturation or peak utilization.

The unprofiled sweep spent 15.96 of 33.90 seconds in CPU preprocessing at B8,
with no overlap of next-batch preprocessing and current NPU work. Together,
limited kernel scaling and preprocessing explain the modest 9.4% end-to-end
gain. CPU/NPU overlap and the costly vision/cast/pointwise operators are the
next measurement-driven targets. Host spans containing asynchronous waits must
not be treated as kernel execution or added to device durations.

All three traces exported successfully and observer events closed. Profiled
wall times are diagnostic, not throughput claims. See the
[report](references/hr_batch_profile_910b/report.txt),
[kernel summary](references/hr_batch_profile_910b/kernel_summary.json), and
[exact invocation](references/hr_batch_profile_910b/command.txt). Full trace
artifacts and operator-detail CSVs are retained at the paths listed in the report.
