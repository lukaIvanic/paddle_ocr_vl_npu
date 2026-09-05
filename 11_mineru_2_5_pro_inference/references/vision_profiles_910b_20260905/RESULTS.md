# MinerU production vision profiles — 910B2, 2026-09-05

This committed reference bundle contains this report and `analysis.json` only.
The per-route timing/parsed/raw files described below remain in the author's
local evidence directory and on the 910B host; they are not required or assumed
accessible by the pull-only 310P agent.

## Result

All six requested routes were captured successfully. Each route has 60 unprofiled event/wall timing samples (30 before and 30 after profiling), plus independent pipe and memory captures of three executions each. Every capture contains exactly 96 PromptFlashAttention calls: 32 layers × 3 executions. Every replay produced bit-identical transformer output, with no nonfinite values.

The dominant cost changes with shape. Attention grows from 11.9% of summed kernel duration at S384 to 52.3% at S5632. At small shapes, rotary, normalization, layout operations and matmuls collectively dominate. These are 910B measurements, not a diagnosis of the 310P kernel yet.

## Method and scope

- Driver: `11_mineru_2_5_pro_inference/profile_production_vision_routes.py`, introduced in `6cf8bc2c`; explicit profiler schedule added in `9eb55dbe`.
- Exact existing page/crop processing and packing code is used. The diagnostic wraps the existing `PrefillDeviceTimeline.measure` boundary, completes its first real invocation, warms the same callable three more times, then replays it while its original tensors and closure are still live.
- No resized synthetic warmup tensors; no model changes, attention changes, new shape buckets or cache identities.
- Reference command: `11_mineru_2_5_pro_inference/references/vision_timing_384_910b/command.sh`. FP16, manual-FP32 LayerNorm, ordinary linear projections, native D80 PromptFA, sparse mode 1, pack target 768. Existing vision/text/decode caches reused. Decode retains `pse_sentinel_310p`.
- Resolution is deliberately the prior matched 384-page reference: min_pixels 25088, checkpoint max_pixels 1605632. This profiling run does not introduce the newer recognition cap or measure its effects.
- Profile boundary: the production `vision_transformer_blocks` region, i.e. all 32 transformer blocks. Direct routes include padding/mask construction and slicing; packed routes receive already-built packed tensors/mask. Packing construction is outside the packed block region, exactly as in the production timing metric.
- Patch embedding, initial position preparation, and merger are separately event-timed, not included in the block kernel captures. CPU image preprocessing, H2D, text prefill and decode are excluded.
- The endpoint run used physical NPU 6 at checkout `02584647f34b0fc5cc0c231d2e0c047f041a956b`; intermediate routes used physical NPU 4 at `9eb55dbee314389eba43383be1185a7a02dcc0b3`. Both are 910B2 devices, automatically selected free by `npu-setup`. This is not a same-physical-device sweep.
- The endpoint diagnostic processed 16 pages; the intermediate diagnostic processed 32. Both exited 0. Replaying stages intentionally distorts pipeline counters/times: **nested page-run summaries are NOT valid E2E throughput results**.
- Repeated-output equality checks deterministic replay, not independent model accuracy or token parity against another implementation.

## Warm block-region latency and throughput

Percentiles use all 60 unprofiled samples, linear interpolation. Tokens/s uses mean event duration. Each row is one real production sample repeated; it is not the distribution of all crops in the bucket. In particular S384's low real-token throughput reflects only 152 useful tokens in a 384-token execution.

| Route | Real tokens / members | Physical tokens | Mean ms | p50 ms | p99 ms | Max ms | Real tok/s | Physical tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct S384 | 152 | 384 | 14.525 | 14.504 | 15.083 | 15.451 | 10,465 | 26,437 |
| Direct S768 | 640 | 768 | 17.890 | 17.863 | 18.439 | 19.046 | 35,775 | 42,930 |
| Packed S768 | 480 + 192 | 768 | 17.605 | 17.589 | 17.999 | 18.357 | 38,170 | 43,623 |
| Direct S1536 | 1088 | 1536 | 26.012 | 25.984 | 26.632 | 27.248 | 41,827 | 59,050 |
| Direct S3072 | 2160 | 3072 | 48.654 | 48.479 | 50.647 | 51.702 | 44,395 | 63,139 |
| Direct S5632 | 5476 | 5632 | 105.007 | 104.903 | 106.276 | 106.776 | 52,149 | 53,634 |

Masks, real lengths and padding densities differ between rows. Do not fit a pure sequence-length complexity curve to these samples or equate physical tokens/s with useful throughput.

## Kernel breakdown

Values below are summed pipe-profile kernel durations divided by three forwards, not independently timed Python subregions. PromptFA and matmul totals come directly from kernel types. Rotary and LayerNorm attribution is derived from source operations and shape signatures; raw signatures remain available for review. Casts/fusions can cross semantic boundaries, so those semantic groups are approximate.

| Route | Total kernel ms | PromptFA ms | PromptFA share | Four linears ms | Rotary slice/math/casts ms | Manual LayerNorm ms | QKV split + attention layout ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct S384 | 14.478 | 1.719 | 11.9% | 3.226 | 4.483 | 2.607 | 1.313 |
| Direct S768 | 17.804 | 2.662 | 15.0% | 4.556 | 5.171 | 2.575 | 1.515 |
| Packed S768 | 17.663 | 2.655 | 15.0% | 4.526 | 5.195 | 2.570 | 1.505 |
| Direct S1536 | 25.901 | 5.683 | 21.9% | 7.290 | 6.160 | 3.456 | 1.791 |
| Direct S3072 | 48.320 | 16.726 | 34.6% | 13.649 | 9.031 | 4.348 | 2.313 |
| Direct S5632 | 104.832 | 54.781 | 52.3% | 23.061 | 13.930 | 6.326 | 2.981 |

The remaining time comprises MLP activation, residual adds, and small casts/padding/mask operations.

At S5632, linears split as follows:

- QKV projection: 5.685 ms.
- Attention output projection: 2.119 ms.
- MLP FC1: 7.459 ms.
- MLP FC2: 7.797 ms.

### Findings

1. **Large-shape attention is the leading comparison target for 310P.** At S5632 each PromptFA averages about 1.712 ms per layer (54.781 ms across 32). At S384 it averages 53.7 microseconds. This establishes which operator to compare next, not why its 310P implementation is slower.
2. **Rotary processing is a material second target.** The FP32 rotate-half path leaves slice/negate/concat/multiply/add/cast kernels in the compiled graph. At S5632 rotary-attributed work totals about 13.93 ms; the four half-head slices per layer alone total 7.52 ms across the model. At S384 rotary is about 31% of kernel time, more than attention's 12%.
3. **Compiled does not mean fully fused.** There are approximately 1,322 kernel records per direct forward and 1,313 per packed forward (plus a one-time profiler Data record). The graph removes Python-per-layer execution but still contains many device kernels.
4. **Mask construction itself is tiny on 910B.** At S5632 the NotEqual mask-construction kernel totals 0.087 ms per forward. This does not establish whether the mask changes the efficiency of PromptFA internally; it only measures building the mask.
5. **Packed and direct S768 execute nearly identical attention time.** Here it is 2.655 versus 2.662 ms, despite two packed members versus one direct sequence. Packing improves useful-token occupancy; these two samples do not show a large kernel-time reduction from having shorter isolated members. Do not generalize this into a statement about unpadded attention.
6. **Patch embedding and merger are small at the layout shape.** For the same 5476-token layout sample: patch embed 0.285 ms, initial position preparation 1.214 ms, merger 0.468 ms, versus blocks 105.007 ms. Initial position preparation is separate from rotary arithmetic repeated inside every layer.
7. **No large unexplained warm host gap appears here.** Unprofiled wall means exceed NPU-event means by about 0.11–0.23 ms. Event intervals can include host-induced idle gaps; they are not pure summed kernel time. Profiled kernel sums closely track unprofiled event timings, while profiler wall measurements are slower.

## Profiler validity and overhead

- Every pipe and memory capture has 96 PromptFA calls and 384 linear calls, matching three complete 32-layer forwards. The full kernel-type and shape-signature lists account for all kernel rows/duration; the 60-row parser limit did not truncate these aggregates.
- Pipe and memory captures agree closely: e.g. S5632 kernel sums are 104.832 vs 104.855 ms, and PromptFA 54.781 vs 54.797 ms.
- Profiler wall means are about 18.0–18.2 ms at S384 versus unprofiled wall 14.63 ms; S5632 about 108.6–109.0 ms versus 105.15 ms. Use unprofiled samples for speed, kernel durations for attribution.
- The initial endpoint captures emitted a default-schedule shutdown warning. Their complete kernel counts were verified. The intermediate captures use an explicit three-step schedule.
- The packed S768 pipe `step_trace_time` totals describe only one forward despite kernel_details containing all three. Do not use that step-trace total as a three-forward denominator or infer a gap from it. Kernel counts/durations and independent timings are the analysis basis.
- Raw pipe and memory PMU fields are retained. Their per-core/aggregation semantics are not assumed to represent whole-device HBM bandwidth, and busy percentages are not proof of the limiting resource.

## Next measurement

Repeat these production-route profiles on 310P and compare PromptFA, rotary, LayerNorm and matmul costs separately, with real lengths/member lists recorded. Only then attribute the cross-chip gap to a specific kernel path. No D128 padding or `_unpad` replacement has been tested here. No fresh Paddle profile was run, so this result does not quantify the Paddle/MinerU cross-chip difference.

## Evidence

- `analysis.json`: merged numerical table and semantic kernel grouping.
- `profile_endpoints_6cf8bc2c/result.json` and `profile_middle_9eb55dbe/result.json`: all unprofiled samples, tags, parity, auxiliary timings, and profiler wall samples.
- Each route's `pipe/parsed_profile_summary.json` and `memory/parsed_profile_summary.json`: kernel names, types, shapes, formats, dtypes, counts, durations, PMU fields and API summaries. Companion Markdown summaries are adjacent.
- Exact commands, checkout commits, visible-device IDs and exit codes are retained in each run directory.
- Full raw traces and diagnostic page artifacts remain on the 910B server beneath `/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/profile_endpoints_6cf8bc2c/` and `profile_middle_9eb55dbe/`.
- Parent run logs are the corresponding `.log` files in the same remote parent directory.
