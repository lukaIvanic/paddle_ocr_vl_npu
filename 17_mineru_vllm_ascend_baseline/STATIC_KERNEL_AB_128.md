# `enable_static_kernel` A/B on 128 OmniDocBench pages

## Result

`enable_static_kernel` did not produce a meaningful hot-throughput gain in this
MinerU workload on one 910B2. The observed inference-only result was 0.46075
pages/s with the option off and 0.46443 pages/s with it on, a 0.80% difference
from one matched pair. That is too small to separate from run-to-run variance.

The startup effect was large. The static-kernel run spent 367 seconds in graph
capture and 528.99 seconds in engine setup. The run without static kernels spent
3 seconds in graph capture and 85.33 seconds in engine setup. This is 443.65
seconds of additional setup for the static-kernel lane.

The official evaluator found no material quality change. The overall score was
90.45365 with static kernels off and 90.44971 with them on, a difference of
-0.00394 points.

## Controlled setup

Both measured runs used:

- source commit `a2397dff411999610e1327416a1575979266c41b`;
- physical Ascend 910B2 device 6;
- stock vLLM 0.21.0 and vLLM-Ascend 0.21.0rc1;
- MinerU2.5-Pro-2605-1.2B in float16, TP=1;
- the first 128 pages of OmniDocBench `v1.6_full`;
- `AsyncLLM`, `max_model_len=8192`, `max_num_seqs=512`, and
  `max_num_batched_tokens=16384`;
- prefix caching, chunked prefill, and `npugraph_ex` enabled;
- the same 14 full-decode graph capture sizes;
- input-manifest SHA-256
  `77ca4566d1454cb133e099dc4233a9efd1323d1444e140bf0b3307a9c589b80e`.

Only `enable_static_kernel` and its isolated compile-cache directory differed.
The matrix selected device 6 once, then reused it for every process.

Before the static-off measurement, a one-page cold run populated its isolated
compile cache. That cold run took 43.42 seconds in `torch.compile`. The measured
static-off run loaded that cache in 4.43 seconds. The static-on measurement also
loaded its existing AOT compile cache in 4.55 seconds. It nevertheless compiled
static-kernel packages for the capture shapes again and reported 367 seconds of
graph capture.

## Performance

| Metric | Static kernel off | Static kernel on | On minus off |
| --- | ---: | ---: | ---: |
| Completed pages | 128 | 128 | 0 |
| Failed pages | 0 | 0 | 0 |
| Engine setup | 85.3328 s | 528.9862 s | +443.6534 s |
| Graph capture | 3 s | 367 s | +364 s |
| Image load | 10.8069 s | 10.9009 s | +0.0940 s |
| Inference | 277.8091 s | 275.6063 s | -2.2028 s |
| Output write | 0.1103 s | 0.1104 s | +0.0001 s |
| Post-startup benchmark wall | 288.7529 s | 286.6457 s | -2.1073 s |
| Inference-only throughput | 0.46075 pages/s | 0.46443 pages/s | +0.80% |
| Post-startup end-to-end throughput | 0.44329 pages/s | 0.44654 pages/s | +0.74% |

At the observed 0.01721-second saving per page, the 443.65-second setup penalty
would need about 25,780 pages to break even. This is only a naive extrapolation:
the measured hot delta is below 1% and is not established as a repeatable gain.

## Output and quality parity

Greedy output was not byte-identical. Static kernels changed 18 of 128 Markdown
files and 24 of 128 content-list JSON files. The differences inspected were
small token-level alternatives, such as one Chinese character or equivalent
LaTeX spellings. Both unedited prediction sets were therefore scored instead of
assuming parity.

Evaluator commit: `2b161d010d2e3aff77a0edef359ea3a6411d23cd`.

| Official metric | Static kernel off | Static kernel on | On minus off |
| --- | ---: | ---: | ---: |
| OmniDocBench overall | 90.45365 | 90.44971 | -0.00394 points |
| Text edit distance | 0.0837432 | 0.0838107 | +0.0000675 |
| Formula CDM | 98.33018 | 98.32511 | -0.00507 points |
| Table TEDS | 81.40510 | 81.40510 | 0 |
| Table structure TEDS | 92.19621 | 92.19621 | 0 |
| Reading-order edit distance | 0.0355621 | 0.0350541 | -0.0005080 |

Both evaluator runs matched all 128 pages. Each scored 908 formula samples and
four tables. Neither run had a page-match fallback, timeout, metric error, or
exception. The table denominator is small, but the 908-formula CDM result gives
a useful numerical-parity check.

## Interpretation

For this stock MinerU page pipeline, `enable_static_kernel=True` changes the
decode graph implementation but does not remove layout, image preprocessing,
vision/prefill, scheduling, or output costs. This run found no useful steady
throughput effect at the pipeline level. It did observe a large startup cost
because the static-kernel packages were rebuilt during graph capture despite a
warm ordinary vLLM compile cache.

Keep the photographed static-kernel configuration when reproducing the 310P
source contract. For normal 910B2 MinerU runs with this software stack, use
`enable_static_kernel=False` unless a repeated serving-shaped benchmark shows a
larger gain or static-kernel package reuse is separately verified.

Compact evidence is in `references/static_kernel_ab_128_a2397df/`. The full
predictions and logs remain in the ignored run directories on the 910B
container.
