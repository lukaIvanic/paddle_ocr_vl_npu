# TorchAir text-model donor audit

2026-10-07. Read-only audit of owned source, retained 910B2 measurements and
official Ascend source. No new model implementation or NPU inference was run
for this audit. Compact server reports are copied under `source_reports/`;
`manifest.json` records their original paths and SHA-256 hashes.

## Decision

There is no evidence that one existing model implementation is the fastest
general TorchAir implementation. Use components whose complete forward and
profiles were measured, then validate them on ColQwen's frozen text inputs.

Qwen3-Reranker-4B is the closest whole-block architecture. PaddleOCR-VL,
MinerU and the separate GLM-OCR roadmap are additional donors for measured
native norms, residual handling and multimodal rotary preparation. UniRec is
excluded from the donor ranking because of its scale and different architecture.
Follow selected run configurations rather than a model's default forward.
ColQwen already has
packed QKV and gate/up projections, PromptFA and native GQA; preserve these
when borrowing components.

Current matched ColQwen control: FP16, B1, S1274, H2560, 36 layers, Ascend910B2,
native weights/internal formats enabled. Warm text-only forward means are
77.560 ms raw eager and 62.328 ms TorchAir, with bit-exact hidden output.
See [isolated text evidence](../text_forward_20261007/README.md).

Its compiled per-forward kernel sums identify the work to target:

| Family | Kernel time | Calls |
|---|---:|---:|
| Dense projections | 35.430 ms | 144 |
| PromptFA | 11.087 ms | 36 |
| Manual RMSNorm and supporting casts | 6.528 ms | 653 |
| Projection splits and attention layout | 3.546 ms | 216 |
| Rotary | 2.694 ms | 288 |
| SiLU × up | 1.651 ms | 36 |

These are summed profiled kernel durations, not predicted removable wall time.
MLP gate/up plus down accounts for 26.008 ms, about 42% of the kernel sum.
The text-only scope has no image insertion, vision kernels or retrieval head.

## Model-by-model findings

### MinerU2.5-Pro: trace the selected packed/native execution paths

The retained real config in experiment 11 is Qwen2-VL: 24 layers, H896,
intermediate4864, Q14/KV2, D64, RMSNorm epsilon1e-6, SiLU-gated MLP,
MRoPE sections [8,12,12]. Unlike ColQwen, it has no learned Q/K RMSNorm and
no DeepStack additions. Its attention projections carry biases.

`local_modeling_mineru.py` contains packed decode QKV/gate-up, native RMSNorm,
fused add-RMSNorm and native rotary alternatives. The actual retained B1/KV4096
profile at `b0f0cde` uses native ApplyRotaryPosEmb, and a separate clean
100-step loop averages 1.804 ms. Its two captured forwards contain 96
InplaceAddRmsNorm calls and 48 ApplyRotary calls: 48 residual norms and 24
joint Q/K rotary calls per forward. These are evidence that the native
operations lower to compact graph kernels, not a latency comparison with S1274.

The selected `run_page_pipeline.py` options explicitly set IncreFA,
`decode_nz`, `npu_apply` rotary and `torchair-packed` text. Its call chain is
`run_official_transformers_omnidocbench.py` → compiled decode wrapper and
continuous-batch engine → `local_modeling_mineru.py`'s native decode path.
The same setup passes `local_text_runtime` to the engine's
`packed_text_prefill_runtime`; that distinct stage is
`PackedMinerUTextPrefillStage`. The model's selected decode is not manual
attention, and vision uses PromptFA. These are different stages.

That packed text-prefill stage currently uses repeated KV heads, two
manual attention BMMs and FP32 softmax, manual norms/rotary and scratch-cache
writes. Copying that stage would replace ColQwen's existing flash attention
with a more expensive attention representation.

The packed-NZ prefill experiment at `43725ec` was **rejected and reverted**:
on 128 pages transformer prefill changed only 22.338→22.207 s (0.6%); prefill
wall increased 78.585→80.723 s; 126/128 Markdown outputs were byte-identical.
This is a corpus aggregate, not a matched single-forward timing. The retained
report is copied into `source_reports/11_mineru_2_5_pro_inference/optimization_ladder/`.

Native SwiGLU has historical timing in `ca7ee1e`, but commit `af932225`
removed it as unsupported. Current MinerU decode uses packed projection then
split/SiLU/multiply. Do not advertise that historical native-SwiGLU experiment
as the accepted current implementation. Paddle/table code is the clearer donor
for testing native packed SwiGLU in the current ColQwen environment.

### PaddleOCR-VL / table engine: mature native operators, strong profiling lessons

Experiment 09's ERNIE text model has 18 layers, H1024, intermediate3072,
Q16/KV2, explicit D128, RMSNorm epsilon1e-5 and MRoPE [16,24,24]. It shares
the residual/RMSNorm, gated MLP and multimodal rotary structure, but lacks
ColQwen's learned Q/K norms and DeepStack.

The decode optimization matrix is real full-step evidence: B1/KV1024/position768
median 2.182→1.332 ms for baseline versus `combined_apply`, about 39% lower.
It combines factor hoisting, packed QKV, native norms and rotary, and fused
residual norms. The boundary includes model/argmax and post-graph state update.
It does not establish an S1274 prefill gain or isolate each component's saving.

The locked optimized serving run at `be691de1`, recorded in
`09_persistent_page_engine/repro/table_latency_20260908/README.md`, selects
`combined_apply_complete_layer_prefetch1_rope_lut_packed_mlp`, rather than
the older `combined_apply` matrix point above. The runtime resolves this
preset to native RMSNorm/AddRMSNorm, packed QKV and gate/up, native SwiGLU,
RoPE lookup, next-layer prefetch and stock IncreFA with native GQA.
Its NZ weights and reduced LM head are generation-specific choices, not a
recommendation for the cache-free ColQwen text tower.

The generic donor helpers are `_decode_rms_norm`, `_decode_add_rms_norm` and
the packed `npu_swiglu` MLP path in
`09_persistent_page_engine/paddleocr_vl/model/text_decode.py`; experiment 19
has a smaller version in `p06_text_prefill_and_decode.py`. Test the stock
SwiGLU call against ColQwen's already packed gate/up output before considering
specialized custom kernels. The B4 decode matrix's packed-MLP versus native
SwiGLU medians are 2.898 versus 2.877 ms; that small difference is not proof
of a long-prefill gain.

The locked serving configuration explicitly records manual causal FP32-softmax
text prefill. This does **not** characterize every optimized text path:
`text_spec_verify.py` has a native-GQA `promptfa_gqa` branch, and
`text_mixed_q.py` implements packed/two-row/padded BSND PromptFA and
replicated IncreFA alternatives. The re-audit below measures these as complete
forwards. Select an attention implementation from its measured workload,
not simply from whether its name contains flash attention.

The ordinary prefill packing work reduces the number of
requests/graph calls; it does not imply better per-layer operators. The saved
Paddle S1024 packing lab explicitly excludes scratch-KV redistribution and
LM-head work and substitutes synthetic image embeddings. Keep those scope
limitations when interpreting its 2.99× production-group projection.

The [mixed-M16 re-audit](../../../09_persistent_page_engine/MIXED_M16_ATTENTION_REAUDIT.md)
is an especially useful text profiling example. BSND removed all 36 layer
transposes, but the first unified PromptFA forward regressed 2.373→3.563 ms
because attention cost grew. Later BSH cache/layout and metadata-hoisting
work improved the candidate to 2.231–2.255 ms. This demonstrates why attention
layout, cache writes and surrounding preparation must be judged together.
The mixed verifier/draft workload itself does not belong in ColQwen retrieval.

The B1 custom vector AddRMSNorm lab also failed its adoption test:
stock CANN full step 1.1458 ms versus custom 1.1530 ms. The custom SwiGLU,
QKV-split, token-embedding and KV operators specialize tiny decode shapes;
their existence does not establish useful S1274 kernels.

### Separate GLM-OCR roadmap: measured native rotary and norm path

The owned server project `aoe_speedup_try` contains a self-contained TorchAir
decode ladder, outside this repository's numbered experiments. The retained
`03c_half_layout_rotary/manifest.json` selects `half_layout_npu_half`:
packed QKV, native RMSNorm, once-per-forward MRoPE preparation, and
`npu_rotary_mul(..., rotary_mode="half")`. It calls native IncreFA. Its
loader permutes Q/K projection rows and its fixture converts cached keys
to preserve the original interleaved model's attention semantics.

At B1/KV1024, 32 decode steps and three measured repeats, native half rotary
reports 1.832 ms per step from NPU events, versus 1.905 ms for the packed-QKV,
native-norm, manual-interleaved control (about 3.84% lower). Output token IDs
match exactly across all three variants. These timings include decode control,
LM head, argmax and cache updates; they are not ColQwen forward predictions.
The eight-step profile confirms 32 RotaryMul calls, 33 RMSNorm and 32
InplaceAddRMSNorm calls per forward. StridedSliceD calls drop from 528 to 16
across eight steps, and the 256 Pack calls disappear. Remaining slices belong
to preparation/control rather than the removed per-layer interleaved rotation.

This is an additional component reference, not the closest whole transformer:
GLM-OCR has four norm sites per layer, including branch-output norms.
ColQwen already uses half rotation, so GLM's checkpoint row permutation
should not be copied. Its prepared-factor/native-rotary approach is relevant.
The roadmap README describes an older torch-npu2.6 environment; the retained
July manifests and separate `qwen_vllm_research` investigation identify the
later server measurements. Treat version differences explicitly.

The separate investigation also profiles stock vLLM and dense-cache bridges.
Gathering paged caches introduced about 15.4 ms of Index and 5.5 ms of
Transpose across eight decode steps. Persistent dense mirrors removed those
families, but still did not match the standalone TorchAir decode cadence.
Those serving/cache transformations are unnecessary for ColQwen retrieval.

### Qwen3 reranker / dense decode / MoE / GLM

Qwen3-Reranker-4B matches all main ColQwen block dimensions: 36/H2560/I9728,
Q32/KV8/D128, epsilon1e-6, SiLU, bias-free projections and learned Q/K norms.
Keep ColQwen's rotary frequency construction/interleaved MRoPE and DeepStack.

The retrieved B4/realQ128/physicalQKV256 native-weight profile has 145 native
norm kernels (73 RMSNorm +72 AddRMSNorm), 36 native Q/K rotary calls, no layer
transposes, and 730 total kernels. ColQwen has 145 logical norms expanded to
653 kernels. This makes native normalization a concrete first candidate.
Absolute kernel times differ in token shape and cannot be subtracted to
predict a ColQwen gain.

Reranker-0.6B's older compiled prefill optimization reduced median
14.863→10.233 ms; separate profile controls report 32.5% lower clean latency
and 1582→682 kernels. BSND then removed 112 transposes (682→570 kernels);
its measured gain was small and order-sensitive, 0.3–4% across controls.

The 4B NZ profile reduced MatMul 19.562→17.353 ms, with clean median
30.167→27.863 ms. However, its older graph still has seven separate projections
per layer: 252 matmuls versus ColQwen's 144. Preserve ColQwen's packed
projections and native GQA instead of copying that whole historical graph.
The current reranker implementation has evolved beyond the saved profile's
explicit KV expansion, so source and historical trace contracts differ.

Experiment 10's selected Qwen3-0.6B B1 decode path is heavily optimized:
151.74→450.67 tok/s, with measured native/fused norms, packed QKV and rotary.
Its profile is 43.5% matmul, 29.8% IncreFA, 12% AddRMSNorm. Token prefetch,
fresh zero-residual Q/K banks and the decision to retain separate gate/up
matmuls are specific to single-token execution. In-place AddRMSNorm requires
careful residual aliasing; persistent zero banks previously corrupted results.

Experiment 14 reuses those dense decoder operations at Qwen3-32B TP2 and
measures 121.80 ms raw eager versus 31.73 ms compiled. Its TP communication
and sharding are unnecessary for this one-card ColQwen workload.

Experiment 15's Qwen3-MoE full TP2 profile retains native norms/rotary and
packed projections, but expert GMM, routing and HCCL occupy substantial time.
Those FFNs cannot replace a dense ColQwen MLP. Experiment 16's GLM dense
profile likewise confirms compact RMSNorm/AddRMSNorm/SwiGLU kernels, but
uses MLA/DSA, quantized projections and cache/indexer work. Its W8A8 NZ gain
is not evidence for FP16 ColQwen NZ weights.

Experiments 18/20 orchestrate the existing Paddle/MinerU/UniRec cores and
provide no independent replacement transformer. Experiment 22 measures vLLM
embedding/reranking serving. Decision2 (23) and Clef (25) use Qwen3.5 hybrid
Gated DeltaNet/full-attention blocks; Clef is an eager baseline with optimization
explicitly out of scope. KaLM Nano-R2 (24) is a BF16 encoder-decoder smoke,
not a validated TorchAir donor. RWKV (26) is a different recurrent architecture
and has no validated inference in the current orientation.

## Official Ascend cross-check

The official TorchAir tuned-model tree was inspected at retained revision
`4e6e491d85bb8c2a6c9c3f97c965c822b34645cd`; it has been removed from current
master by `cedb5b2054423eecc2192e5aa274bc1e890fef8b`.

- [Qwen2 tuned-model guidance](https://github.com/Ascend/torchair/blob/4e6e491d85bb8c2a6c9c3f97c965c822b34645cd/npu_tuned_model/llm/qwen/README.md)
  explicitly recommends native RMSNorm/AddRMSNorm, native rotary and BSND
  attention to avoid transposes. Its older environment and generation-shaped
  examples are component references, not matched ColQwen profiling results.
- [Qwen2-VL model](https://github.com/Ascend/torchair/blob/4e6e491d85bb8c2a6c9c3f97c965c822b34645cd/npu_tuned_model/mm/qwen2-vl/model/modeling_qwen2_vl.py)
  uses native norms; its text path retains BNSD/manual multimodal rotary. It
  is not a universally more optimized model than our owned implementations.
- [Qwen3-MoE model](https://github.com/Ascend/torchair/blob/4e6e491d85bb8c2a6c9c3f97c965c822b34645cd/npu_tuned_model/llm/qwen3_moe/scripts/models/modeling_qwen3_moe.py)
  uses learned native Q/K norms, native rotary and grouped-matmul/SwiGLU for
  experts. Its distributed MoE implementation is not a dense-model donor.
- Current [RMSNorm FX replacement source](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_utils/npu_fx_passes/joint_graph_passes/npu_rms_norm.py)
  recognizes a specific FP32-statistics/cast-before-affine pattern. The presence
  of a pass in current master does not show that the installed version enables
  it or that our graph matches it. Our actual trace still shows expanded norms.
- [QKV/RMSNorm/RoPE/cache converter](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/ge_converter/custom/npu_qkv_rms_norm_rope_cache.py)
  is interesting for future fusion, but includes output buffers, indices and
  KV-cache mutation. ColQwen's isolated retrieval forward has no decode cache;
  operator availability, hardware support and MRoPE contract need validation.

No inspected external report establishes a faster identical FP16 B1/S1274
ColQwen text-forward implementation on this CANN9/torch-npu2.10 910B2 system.
Public serving throughput or multi-card decode results cannot certify one.

## Current candidate order

The user's selected next experiment leaves norms unchanged and prioritizes
the existing rotary, attention-layout and activation implementations:

1. Native joint Q/K half-rotary using ColQwen's existing prepared MRoPE factors,
   plus BSND attention throughout the layer. Validate these separately and
   together so layout preparation costs remain visible.
2. Native SwiGLU on the existing packed gate/up output: test whether it removes
   the split as well as activation work. Keep four dense matmuls per layer.
3. Investigate the actual dense projection shapes and tiling. Native weights
   remain the control: ColQwen's earlier same-commit B1/S1254 trial measured
   61.931 ms native/internal-on versus 63.918 ms NZ; NZ also failed the recorded
   compiled-versus-own-eager tolerance gate. Do not promote NZ from another
   model's result.

Native normalization remains a potential later experiment. Its kernel count
alone does not establish a correct or faster replacement for this FP16 graph.

Every candidate needs a separate compile cache, identical frozen inputs,
clean warmed forward timing outside the profiler, an explicit numerical gate,
and separate eager/compiled kernel attribution. Preserve real interleaved
MRoPE and all three DeepStack additions. A donor's tolerated numerical drift
does not authorize widening ColQwen's gate silently.

The CANN9 `cube_utilization(%)` export is not physical occupancy or achieved
MAC/FLOP utilization; it cannot prove our matmuls are optimal. See the detailed
interpretation in experiment 09's README. Measure the complete text forward
after each change, rather than accepting a lower kernel count by itself.
