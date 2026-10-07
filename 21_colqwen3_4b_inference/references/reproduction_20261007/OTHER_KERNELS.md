# Compiled ColQwen forward: attribution of the remaining kernels

Investigation of the **2026-10-07 Ascend 910B2 B1** capture, runtime `701ea1cd`:
5,040 vision tokens, 1,274 text tokens, FP16, optimized native GQA PromptFA,
compiled vision/text transformer stages and eager preparation/mergers/finish.
The clean warmed full forward is **137.209 ms**. All numbers below are **summed
device kernel durations averaged across three captured forwards**, not wall time
or predicted optimization savings. No new inference or optimization was run for
this posthoc investigation.

## Semantic breakdown

The non-matrix/non-attention bucket is **33.627 ms**, 27.40% of the 122.707 ms
summed kernel time. Attribution uses full CANN fused names, tensor shapes/dtypes
and the owned model source; it is not inferred from kernel type alone. Categories
are exclusive and sum to the complete capture. Mixed residual-add/FP32-cast
kernels are kept separate rather than assigned entirely to normalization.

| Work | ms/forward | Device kernels/forward |
|---|---:|---:|
| Rotary-position application in vision/text | 7.258 | 624 |
| Text RMSNorm | 7.160 | 653 |
| Layout conversion, QKV unpacking and projection splits | 5.965 | 364 |
| Manual vision LayerNorm | 4.007 | 292 |
| MLP activations, including merger GELU | 3.710 | 64 |
| Image-embedding MaskedScatter | 2.207 | 1 |
| Residual adds, shared norm casts and preparation adds | 1.844 | 127 |
| Other preparation, merger norms and retrieval finish | 1.475 | 80 |
| **Total other** | **33.627** | **2,205** |

### RMSNorm is decomposed, with partial fusion

There is no standalone `RMSNorm` kernel in this capture. The 36 text layers
contain 72 hidden-state norms, 36 Q norms and 36 K norms, followed by one final
hidden-state norm: **145 norms**. Their device work maps to:

- `Pow/SquareReduceMean`, reported as type `Square`: **2.226 ms**, 145 kernels.
  Squaring and reduction are already fused; counting another reduction would
  double-count this work.
- `AddRsqrt`, reported as `AutomaticBufferFusionOp`: **0.485 ms**, 145 kernels.
- `MulCast`, reported as `Mul`: **2.128 ms**, 145 kernels; scaling in FP32 and
  conversion to FP16.
- Separate affine-weight `Mul`: **1.476 ms**, 145 kernels.
- Explicit FP16-to-FP32 casts: **0.845 ms**, 73 kernels. Most hidden-state casts
  are fused with the preceding residual add and are outside this norm subtotal.

The norms total **7.160 ms**: hidden-state norms 3.132 ms, Q norms 2.493 ms,
K norms 1.536 ms. Source: `local_modeling_colqwen3.py:44–47` and
`optimized_prefill.py:171–189`. Preserve the existing FP32 statistics and the
FP16 rounding boundary before the affine weight when probing alternatives.

The 48 vision-block LayerNorms use a manual path (`optimized_prefill.py:75–85`),
totaling **4.007 ms**. Its kernels include 96 `ReduceMean`, 48 `SubPow/Square`,
48 `AddRsqrt`, 48 `MulCast`, 48 affine `MulAdd`, and four explicit initial casts.
Residual-add casts are again accounted separately. The four eager merger
`LayerNormV3` calls add only **0.262 ms**, outside this block-norm subtotal.

### Rotary and layout work are separate, substantial targets

`rotate_half` explicitly splits, negates and concatenates. Vision also performs
FP32 Q/K casts, separate cosine/sine multiplies, and the final add/cast. Text has
`MulMulAdd` fusion for the last arithmetic, but retains the split/negate/concat
operations. Vision rotary application totals **4.298 ms**, text **2.960 ms**.
Generating position cosines/sines in preparation is outside these totals.

Layout/projection work consists of **3.494 ms / 268 Transpose kernels**,
**0.602 ms / 24 Unpack kernels**, and **1.870 ms / 72 projection SplitVD kernels**.
The splits are 36 fused QKV splits (0.567 ms) and 36 fused gate/up splits
(1.303 ms). Transposes include attention input/output head layouts and the vision
QKV permutation. Thus the setup-fused projections still produce materialized
device split kernels here. Rotary splits are in the rotary category above.

### The expensive scatter is image insertion

The exact kernel is
`aclnnInplaceMaskedScatter_MaskedScatterAiCore_MaskedScatter`:
**2.207 ms for one call**, shapes
`[1,1274,2560]; [1,1274,2560]; [1260,2560]`.
It inserts image rows into token embeddings at `prepared_prefill.py:156`.
This remains eager in the compiled lane. Mask expansion/cast helpers add another
0.031 ms, counted in other preparation.

The same preparation function uses row-index assignment for the three DeepStack
feature tensors (`prepared_prefill.py:159`). Their **three IndexPutV2 kernels
total 0.055 ms**, with the same feature-row dimensions. That is a promising
comparison, not proof of interchangeable total forward cost: mask preparation,
destination semantics and output parity must also be checked.

The pipe capture's duration-weighted MaskedScatter vector-core counters report
**0.925 scalar activity**, **0.045 vector activity**, **0.028 MTE2**, and
**0.001 MTE3**. The separate memory capture reproduces its 2.207 ms duration.
This suggests scalar/control work in this kernel, not evidence of HBM saturation.
Counter ratios may overlap; they are not fractions of clean forward wall time.

No `StridedSlice` or `ScatteredSlide` name appears in the exported kernel names.
Ordinary **Slice is just 0.0086 ms across two calls**. The significant scatter is
MaskedScatter. Two `ViewCopy` kernels additionally total 0.330 ms, in preparation.

### What the generic fusion label actually contains

The 349 `AutomaticBufferFusionOp` kernels total **4.010 ms**:

| Fused name | Meaning | ms/forward | Calls/forward |
|---|---|---:|---:|
| `SwishMul` | Text MLP SiLU(gate) × up | 1.727 | 36 |
| `MulMulAdd` | Text Q/K rotary arithmetic | 1.020 | 72 |
| `MulAdd` | Vision LayerNorm FP16 affine | 0.689 | 48 |
| `AddRsqrt` | Text RMSNorm epsilon + inverse RMS | 0.485 | 145 |
| `AddRsqrt` | Vision LayerNorm epsilon + inverse std | 0.088 | 48 |

These rows are already included in the semantic categories. They must not be
added again as another 4 ms cost.

## Concrete follow-up probes

1. Replace only image insertion with a row-index write and compare exact outputs
   and clean warmed full-forward timing. This isolates the anomalous 2.2 ms
   scatter without changing transformer compilation.
2. Probe a fused RMSNorm against each observed hidden/Q/K shape, preserving the
   current precision contract. Validate stage and final embeddings, then measure
   the complete warmed eager and compiled forward separately. The 7.16 ms
   subtotal is an upper bound on removable work, not a predicted saving.
3. Probe combined rotary/layout preparation and view-preserving QKV/gate-up
   handling. Inspect emitted kernels to ensure materialized split/transpose
   work disappears; merely changing Python views does not establish this.
4. Probe vision LayerNorm fusion under the same FP32-statistics/FP16-affine
   contract. Keep it separate from the text RMSNorm experiment.

Each probe should change one candidate, retain the existing parity gates, measure
clean warmed full-forward latency outside profiler, and then capture eager and
compiled profiles separately to verify which kernels changed.

## Evidence and reproduction

- [Kernel shape/name export](warm_forward_701ea1cd/kernel_shape_groups.json)
  includes every kernel group for both existing eager and compiled pipe captures.
  Repeated numeric node suffixes are removed from the saved names; shapes, dtypes,
  counts and durations are preserved. Absolute raw CSV provenance is recorded.
- [Exclusive attribution](warm_forward_701ea1cd/other_kernel_attribution.json)
  preserves each contributing group. Reproduce with
  `python3 warm_forward_701ea1cd/attribute_other_kernels.py` from this directory.
  The script checks 2,515 total kernels, 145 norm reductions, 96 vision means,
  and conservation of the complete and other kernel totals.
- [Pipe/memory counter evidence](warm_forward_701ea1cd/other_kernel_metrics.json).
- [Original timing, parity and profiling report](README.md).

The generic-type ranking remains useful for checking the raw capture: Mul 5.633,
AutomaticBufferFusionOp 4.010, Transpose 3.494, SplitVD 3.213, Add 2.543,
Square 2.226, MaskedScatter 2.207, GeluV2 1.885, Cast 1.498, ReduceMeanD 1.445
ms/forward. It should not be mistaken for a source-level bottleneck breakdown.
