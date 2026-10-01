# 21 — Ops-ColQwen3-4B HF reference and local eager model

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
