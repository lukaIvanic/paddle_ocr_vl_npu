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
