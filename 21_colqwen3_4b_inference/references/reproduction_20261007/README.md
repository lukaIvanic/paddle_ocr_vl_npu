# Warm ColQwen3-4B forward and separate profiles — 910B2

The **warmed complete page forward takes 163.353 ms in optimized raw eager and
137.209 ms with TorchAir transformer graphs**: **1.191× throughput / 16.005% lower
latency**. Separate processes produced bit-exact embeddings. These are forward
latencies with device-resident inputs; page loading and preprocessing are outside
the measurement. The primary evidence is [comparison.json](warm_forward_701ea1cd/comparison.json).

## Execution and timing boundary

- Date: 2026-10-07; runtime source `701ea1cd`.
- Host `liteserver-c001-4`, container `research_vllm_ascend_021_external_workspace`.
  One healthy free **Ascend 910B2, physical device 3**, selected by `source npu-setup`.
- Python `/workspace/venvs/colqwen3_hf_py312/bin/python`, torch/torch-npu 2.10.0,
  FP16, native weights, internal formats enabled, NPU JIT off, HF32 off, four CPU threads.
- Both lanes use the same optimized model: native GQA PromptFA, fused text QKV
  and gate/up weights, manual vision FP32 LayerNorm statistics, Linear patch projection.
- B1 real HR corpus page `corpus-test-5`, 5,040 vision / 1,274 text tokens;
  2,560-dimensional output. [Input provenance](hr_inputs/manifest.json).
- The full forward includes patch/position preparation, vision transformer,
  mergers/text preparation, text transformer, retrieval projection and normalization.
  Existing shape checks and metadata synchronization within preparation are included.
- `raw_eager` is uncompiled execution. `torchair` compiles the existing static vision
  and text transformer graphs; preparation, mergers and output projection remain eager.
- Inputs are processed/transferred once before timing. Model load, graph compile/load,
  three warmups, output transfer/validation and saving are outside clean measurements.
- Each process measures 30 clean forwards before and 30 after profiler capture.
  Each call is synchronized and has an NPU event interval. No subsection timing
  synchronization is added to the forward. Event intervals include host launch gaps.

## Clean forward latency

| Measure, ms | Optimized raw eager | TorchAir |
|---|---:|---:|
| Mean, 60 clean calls | 163.353 | 137.209 |
| Median | 163.309 | 136.606 |
| p90 | 164.303 | 137.895 |
| p99 | 165.816 | 144.890 |
| Before profiler mean, 30 calls | 163.742 | 136.860 |
| After profiler mean, 30 calls | 162.964 | 137.558 |
| NPU event interval mean | 163.169 | 137.001 |

All reference-vs-HF checks, compiled-vs-optimized-eager checks, profiler replays,
finite/unit-norm checks and final cross-process parity passed. Final embeddings
are bit-exact, shape `[1,1274,2560]`, maximum difference zero. This checks one real
page forward; it is not a new full-corpus quality evaluation.

## What the separate profiles show

The follow-up [compiled other-kernel investigation](OTHER_KERNELS.md) maps the
complete remaining 33.627 ms to normalization, rotary, layout/split, activation,
scatter and preparation work using full fused names and shapes, with counter
evidence and concrete controlled probes.

Each lane has separate CPU/NPU pipe and memory captures, with three warmed
forwards per capture, input shapes and CPU call stacks, named forward sections,
CANN CSVs and Chrome traces. The runner follows the established patterns in
`11_mineru_2_5_pro_inference/profile_production_vision_routes.py` and
`13_qwen3_reranker/profile_prefix_cache_forward.py`, and reuses the MinerU
model-independent CANN parser.

The following are **summed profiled kernel durations per forward**, not clean
wall latency. Matrix/attention attribution uses the recorded token shapes.

| Kernel work, ms/forward, pipe capture | Eager | Compiled |
|---|---:|---:|
| Vision attention, 24 calls | 31.273 | 31.103 |
| Text attention, 36 calls | 10.961 | 10.875 |
| Vision matrix multiplications | 10.810 | 10.324 |
| Text matrix multiplications | 36.320 | 35.618 |
| Other matrix multiplications, primarily mergers | 1.163 | 1.159 |
| Remaining kernel work | 56.851 | 33.627 |
| Total summed kernel durations | 147.377 | 122.707 |
| Kernel launches per forward | 3,772 | 2,515 |

**Eager:** attention and matrix work are the largest groups, with substantial
additional normalization, elementwise and layout work. Multiplication kernels
consume 12.258 ms and casts 6.500 ms. Compilation reduces these to 5.633 ms and
1.498 ms respectively; cast count falls from 549 to 144 per forward. The remaining
non-attention/non-matrix work falls by 23.223 ms, accounting for most of the kernel
saving. This comparison already uses fused weights and PromptFA in both lanes.

**Compiled:** attention plus matrix multiplications account for about **72.6%**
of summed kernel duration. The main remaining targets are vision attention
(31.103 ms) and dense text MLP projections. Text fused gate/up, shape
`[1274,2560] × [19456,2560]`, takes 16.592 ms across 36 layers; down projection
`[1274,9728] × [2560,9728]` takes 9.447 ms. Their combined 26.039 ms is the major
text matrix cost. The gate/up kernel reports approximately 97.3% cube utilization
in the pipe capture; this is per-kernel evidence, not a chip-wide utilization claim.

Memory captures reproduce the kernel ranking and totals closely: 147.572 ms
eager, 122.735 ms compiled. Memory counters are preserved in the parsed reports.
These data do not by themselves establish chip-wide HBM saturation.

Rich profiler recording significantly distorts eager host dispatch: profiled
pipe-forward wall means are 740.294 ms eager and 164.131 ms compiled. These values
must not replace the clean 163.353/137.209 ms comparison. Trace gaps also include
profiler overhead and boundaries between captured forwards. Kernel sums and
overlapping task interval coverage are reported separately.

## Evidence and replay

- [Eager command](warm_forward_701ea1cd/raw_eager/command.txt),
  [result](warm_forward_701ea1cd/raw_eager/output/result.json),
  [log](warm_forward_701ea1cd/raw_eager/run.log).
- [Compiled command](warm_forward_701ea1cd/torchair/command.txt),
  [result](warm_forward_701ea1cd/torchair/output/result.json),
  [log](warm_forward_701ea1cd/torchair/run.log).
- Eager [pipe report](warm_forward_701ea1cd/raw_eager/output/profiles/pipe/parsed.md)
  and [memory report](warm_forward_701ea1cd/raw_eager/output/profiles/memory/parsed.md).
- Compiled [pipe report](warm_forward_701ea1cd/torchair/output/profiles/pipe/parsed.md)
  and [memory report](warm_forward_701ea1cd/torchair/output/profiles/memory/parsed.md).
- Four CPU accounting tests cover overlapping intervals, invalid timings and
  empty/invalid profile data. Three passed at runtime `701ea1cd`; the additional
  empty-profile rejection check was added after capture. These are accounting
  tests, separate from real NPU forward/parity validation.

Full raw traces, kernel CSVs and output embeddings remain on the server beneath:

`/workspace/repos/paddle_ocr_vl_npu/tmp/21_colqwen3_4b_inference/replicate_20261007T102343Z_f9bb6837/warm_forward_701ea1cd/`

Each `output/profiles/{pipe,memory}/raw/*/ASCEND_PROFILER_OUTPUT/` contains
`trace_view.json` and `kernel_details.csv`. No original caches were cleared.
No batches larger than one and no 310P execution are claimed.

The same day's preliminary stage replication on physical NPU 2 reproduced the
historical 4,960/1,254-token benchmark: 60.650/75.898 ms eager vision/text versus
51.769/62.985 ms compiled. [Stage result](output/result.json).
The actual HR 5,040/1,274-token stages measured 64.291/78.159 ms eager versus
54.875/62.361 ms compiled. [HR stage result](hr_stages/output/result.json).
Earlier page-pipeline measurements in this directory include preprocessing and
first-use graph loading and are supplemental; they are outside the requested
warmed-forward comparison.
