# Isolated ColQwen text transformer — 910B2

This experiment measures and profiles only the **36 text transformer layers,
three DeepStack additions and final RMSNorm**. The measured callable is
`OptimizedTextStage(*frozen_text_inputs)`. The real HR B1 sequence contains 1,274
positions: 1,260 image-feature positions and 14 prompt positions, hidden width
2,560, 32 Q heads / 8 KV heads, head width 128. This is full-sequence forward,
without generation or a KV cache.

All seven text inputs are prepared before warmup and retained on NPU: hidden
states, rotary cos/sin, causal mask and three DeepStack tensors. Eager saves them
and its hidden-state reference in `text_inputs.pt`; a separate compiled process
loads that same file. File hashes, model/checkpoint identity, anchor and source
provenance are checked. Exact compiled-versus-frozen-eager output parity is a
required gate.

The measured/profiled window excludes vision, mergers, token embedding/image
insertion, positional initialization, retrieval projection, input/output
transfers, compile/load, validation and saving. Existing optimized text math
and the checkpoint are unchanged.

Runtime source: `54f67817`. Container
`research_vllm_ascend_021_external_workspace`, host `liteserver-c001-4`,
**Ascend 910B2 physical device 2**, selected by `source npu-setup`.
Python `/workspace/venvs/colqwen3_hf_py312/bin/python`, torch/torch-npu 2.10.0,
FP16, internal formats on, native weights, HF32 off, NPU JIT off, four CPU threads.
Both lanes use native-GQA PromptFA, fused QKV and fused gate/up weights.
The compiled lane uses the existing static TorchAir text graph copied to an
isolated run-cache directory; no vision graph is compiled or called there.

Each process excludes three warmups and measures 30 clean text forwards before
and 30 after capture. It then captures separate CPU/NPU pipe and memory profiles,
three forwards each, with shapes, call stacks and the `colqwen.text.transformer`
annotation. Profiled wall times are diagnostic and must not replace clean timing.

The text analyzer checks one attention and four fused matrix projections per
layer (36 and 144 calls/forward), text attention shapes, absence of vision/merger/
image-insertion kernels, and conservation of the complete CANN counts/durations.
It exports every shape/name group, rather than only the top kernels.

The initial attempt at `d0a784da` stopped before profiling because it added a
strict full optimized-versus-manual embedding gate. The existing optimized
full-model path does not satisfy the manual reference's elementwise tolerance
(this page: max absolute 0.09155, cosine 0.99776), consistent with the previously
documented optimized attention/vision-norm differences. The manual owned model
remains bit-exact to the saved HF anchor. The corrected experiment preserves
this difference as a setup diagnostic and requires **bit-exact isolated text
compiled-versus-eager parity**, without changing math or widening the tolerance.
This run does not claim new optimized-versus-HF equivalence or retrieval quality.

## Clean warmed text latency

**77.560 ms optimized raw eager versus 62.328 ms TorchAir**, a **19.639% latency
reduction / 1.244× throughput**. These are the means of 60 clean warmed calls per
lane, on the same physical NPU 2. The lanes used the same frozen input file
(SHA-256 in `comparison.json`). Hidden states are **bit-exact**, max absolute
difference zero, shape `[1,1274,2560]`. Both runs exited 0; finite-output, initial
parity, profiler replay and final replay gates all passed.

| Clean latency, ms | Raw eager | Compiled |
|---|---:|---:|
| Mean, 60 calls | 77.560 | 62.328 |
| p50 | 77.525 | 62.347 |
| p90 | 77.657 | 62.546 |
| p99 | 78.487 | 63.799 |
| Before-profile mean, 30 calls | 77.497 | 62.117 |
| After-profile mean, 30 calls | 77.623 | 62.539 |
| NPU-event interval mean | 77.397 | 62.052 |

The small before/after shift is retained in the raw samples. Event intervals
include launch/dispatch gaps; they are not pure kernel time. Compilation/cache
loading and first use occurred before warmup. The text cache was already warm.

## Text-only kernel profile

All four captures pass the isolation checks: exactly **36 text attention calls
and 144 projection matmuls per forward**, matching sequence dimensions, and no
vision/merger/image-insertion kernels. The Chrome trace annotations additionally
contain exactly three `colqwen.text.transformer` regions each and no preparation,
vision or retrieval regions. [Annotation checks](trace_scope_checks.json).

| Pipe capture | Raw eager | Compiled |
|---|---:|---:|
| Summed device kernel durations, ms/forward | 76.940 | 61.934 |
| Device kernels/forward | 2,279 | 1,449 |
| Projection matmuls, ms | 36.452 | 35.430 |
| Attention, ms | 11.067 | 11.087 |
| Remaining kernels, ms | 29.421 | 15.417 |

Compilation saves **15.006 ms of summed device kernel work** and 830 launches.
The remaining-kernel saving is **14.004 ms**. Attention and matrix projections
are largely unchanged; partial fusion of pointwise, normalization and layout
work accounts for most of the saving.

Compiled semantic categories below are exclusive, derived from full fused kernel
names, tensor shapes/dtypes and the owned text source. Shared residual-add/norm
cast kernels are counted separately. Kernel sums are not clean wall latency or
predicted removable time.

| Compiled text work | ms/forward | Kernels/forward | Share of kernel sum |
|---|---:|---:|---:|
| Dense projections | 35.430 | 144 | 57.2% |
| PromptFlashAttention | 11.087 | 36 | 17.9% |
| RMSNorm, excluding shared residual casts | 6.528 | 653 | 10.5% |
| Transposes and QKV/gate-up projection splits | 3.546 | 216 | 5.7% |
| Rotary-position application | 2.694 | 288 | 4.4% |
| Fused SiLU(gate) × up | 1.651 | 36 | 2.7% |
| Residual/DeepStack adds and shared norm casts | 0.992 | 75 | 1.6% |
| Graph Data task | 0.006 | 1 | <0.1% |

**RMSNorm is the largest text pointwise group.** The 145 norms are 72 layer
hidden-state norms, 36 Q norms, 36 K norms and one final hidden-state norm.
They remain 653 kernels rather than one fused kernel per norm:

| Norm component | ms/forward | Calls/forward |
|---|---:|---:|
| `Pow/SquareReduceMean` (type Square) | 2.060 | 145 |
| `AddRsqrt` (AutomaticBufferFusionOp) | 0.440 | 145 |
| `MulCast` (type Mul) | 1.872 | 145 |
| FP16 affine-weight Mul | 1.401 | 145 |
| Explicit FP16-to-FP32 Cast | 0.755 | 73 |

The missing hidden-state casts are primarily fused with residual adds; they are
not counted again above. The current norm computes FP32 statistics/scaling,
rounds the scaled activation to FP16, then applies its FP16 weight. A replacement
must preserve or explicitly validate this precision boundary.

**Layout/splits:** 144 transposes total **1.686 ms**, 36 fused-QKV splits
**0.568 ms**, and 36 gate/up splits **1.292 ms**. The compiler still materializes
the projection split operations. Rotary contributes an additional 72 half-splits,
72 negations, 72 concatenations and 72 fused `MulMulAdd` kernels, **2.694 ms**.
Those half-splits are in the rotary category, so total raw `SplitVD` time is
2.442 ms and must not be added to the layout category again.

Eager has **180 AsStrided calls / 1.924 ms** and **144 Slice calls / 1.905 ms**;
neither type remains in the compiled text capture. Compiled has no MaskedScatter.
The earlier 2.2 ms image insertion belongs outside this text-forward boundary.

**Dense MLP dominates total text kernel time:** gate/up projection is **16.610 ms**
and down projection **9.398 ms**, combined **26.008 ms** (42.0% of kernel sum).
Fused QKV projection is 5.245 ms and attention output projection 4.177 ms. These
are 36 calls each. This distinguishes the largest total cost from the largest
remaining pointwise target.

The independent memory captures reproduce the totals: **76.921 ms eager** and
**61.924 ms compiled**. Compiled RMSNorm/layout/rotary subtotals are
6.530 / 3.531 / 2.698 ms. Full PMU counters are in the parsed evidence; these
observations alone do not establish chip-wide memory-bandwidth saturation.

Rich profiling changes eager host dispatch substantially: pipe-capture wall
means are **418.807 ms eager / 63.175 ms compiled**. Use the clean 77.560/62.328 ms
comparison for latency, not these diagnostic profiled wall times or trace gaps.

## Focused next investigations

The text pointwise investigation should begin with the **145 RMSNorms**, then
the **materialized projection splits/transposes** and **rotary half-split /
negate / concat chain**. The current baseline preserves all text math and
weights. Change one candidate at a time, validate the frozen-input text hidden
states, measure warmed text forward outside profiler, and capture each execution
lane separately to verify the actual emitted-kernel change. Keep matrix/attention
costs visible so a small pointwise saving is not described as the dominant total
text bottleneck.

## Evidence and reproduction

- [Paired comparison and semantic summaries](comparison.json).
- Eager [command](raw_eager/command.txt), [result](raw_eager/output/result.json),
  [run log](raw_eager/run.log), [exit code](raw_eager/exit_code.txt).
- Compiled [command](torchair/command.txt), [result](torchair/output/result.json),
  [run log](torchair/run.log), [exit code](torchair/exit_code.txt).
- Eager [pipe](raw_eager/output/profiles/pipe/parsed.md) and
  [memory](raw_eager/output/profiles/memory/parsed.md) reports.
- Compiled [pipe](torchair/output/profiles/pipe/parsed.md) and
  [memory](torchair/output/profiles/memory/parsed.md) reports.
- Each profile directory also has complete `kernel_shape_groups.json`,
  `kernel_accounting.json`, parsed JSON and parser logs.
- [Exact orchestration script](run_command.sh); initial failure
  [result](failed_setup/raw_eager/output/result.json) and
  [command](failed_setup/raw_eager/command.txt) are retained.

Raw profiles and the frozen input/hidden-state files remain on the container at:

`/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/`

The analyzer can be rerun there with:

```sh
python3 21_colqwen3_4b_inference/analyze_text_profile.py \
  --run-dir tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817
```

Raw trace/CSV data are large and kept outside Git. Four independent CPU accounting
tests passed on the server, along with the actual NPU timing/parity/profile gates.
No batching beyond B1, query-shape evaluation or 310P execution is claimed.

The raw Chrome traces and kernel/operator CSVs are also saved in this local
workspace outside Git. Load the JSON traces in a Chrome/Perfetto trace viewer:

- [Eager pipe trace](/home/luka/projects/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/profiles/raw_eager/pipe/trace_view.json).
- [Compiled pipe trace](/home/luka/projects/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/profiles/torchair/pipe/trace_view.json).
- [Eager memory trace](/home/luka/projects/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/profiles/raw_eager/memory/trace_view.json).
- [Compiled memory trace](/home/luka/projects/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/text_forward_20261007T110218_54f67817/profiles/torchair/memory/trace_view.json).

These contain the complete recorded CPU/NPU events. The larger eager trace
reflects rich per-operation recording, so its host timeline is diagnostic.
