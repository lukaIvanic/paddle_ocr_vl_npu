# NPU bucketed-vision result

Validated on an Atlas 800I A2 / 910B NPU with the PaddleOCR-VL-1.6 model. The
compiled boundary is the 27 vision encoder layers plus final LayerNorm. Patch
embedding, absolute-position interpolation, projector, text prefill, and the
continuous decode system remain outside that boundary.

## Static graph coverage

TorchAir successfully built and executed independent B=1 graphs for all
configured physical sequence lengths:

```text
16, 32, 64, 128, 256, 512, 1024, 2048
```

Each shape has a distinct compiler entrypoint and GE cache directory. This is
necessary because using one Python `forward` code object for every shape made
TorchDynamo classify later shapes as recompilations and TorchAir skip their
persistent caches. On the subsequent cache-reuse run, executing all eight
graphs took 5.71 seconds total (0.68--0.75 seconds per graph). Cold creation is
much slower and belongs to setup, not page throughput.

## Exact parity control

The one-region compiled and eager controls used the same page, layout result,
crop, `min_pixels=6272`, fp16 model, eager decode, and 64-token generation cap.
The generated token ID lists were exactly equal.

The crop produced 608 real vision tokens and selected the 1024 bucket:

| Path | Encoder + post-LN | Compiled input prep | Page wall |
| --- | ---: | ---: | ---: |
| eager manual attention | 52.65 ms | n/a | 0.989 s |
| compiled manual attention | 23.70 ms | 7.29 ms | 0.936 s |

The pure compiled boundary was 2.22x faster. Including mask, padding, and RoPE
preparation, the vision-encoder portion was 1.70x faster for this crop. The
single-crop page-wall difference is only a smoke result and should not be
treated as a stable E2E benchmark.

## Full-page integration

The final run used real PP-DocLayoutV3 layout inference, compiled vision,
compiled static B=4 decode, cache length 2048, maximum generation length 768,
and `min_pixels=6272`.

- Five of five regions completed; no page was partial.
- Four crops used compiled vision: bucket 64 once, 1024 once, and 2048 twice.
- One 3528-token crop exceeded the configured maximum and correctly used the
  eager unpadded overflow path.
- Compiled crops contained 3264 real rows in 5184 physical rows (62.96% useful).
  Including the unpadded overflow crop, the run-level useful fraction was
  6792 / 8712 = 77.96%.
- Page/run wall was 1.705 / 1.709 seconds.
- Decode produced 81 output tokens at 534.55 effective tok/s and 1181.63 raw
  fixed-arena tok/s.

The complete evidence is under
`tmp/08_offline_e2e_b1/compiled_vision_validation/`. The decisive files are the
paired `compiled_smoke_v3/run.json` and `eager_smoke/run.json`, plus
`compiled_full_page_b4/run.json`.

## Five-page continuous-B4 comparison

The larger comparison reused the established five-page, 179-layout-region,
160-recognition-region workload. All runs used `min_pixels=56448`, fp16,
compiled B=4 continuous decode, cache length 2048, and a 768-token generation
cap. Setup is excluded from the run-wall and throughput metrics, matching the
historical reports.

| Run | Physical NPU | Run wall | Effective decode tok/s | Raw decode tok/s | E2E output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| current eager control | 5 | 22.978 s | 1204.10 | 1285.61 | 313.00 |
| compiled vision, all 8 buckets | 5 | 22.751 s | 1115.97 | 1191.69 | 315.41 |
| current eager control repeat | 7 | 23.405 s | 1209.52 | 1291.39 | 307.28 |
| compiled vision, all 8 buckets repeat | 7 | **20.423 s** | 1091.63 | 1165.69 | **351.36** |
| compiled vision, only used buckets 512/1024/2048 | 5 | 21.488 s | **1226.45** | **1309.66** | 333.95 |
| previous explicit manual-eager baseline | unspecified | 22.965 s | 1224.37 | 1307.25 | 313.18 |

The all-bucket same-device E2E reduction ranged from 0.99% on NPU 5 to 12.74%
on NPU 7. Its fastest result was 11.07% below the previous explicit baseline
and 10.68% below the older 22.866-second historical minimum. The wide paired
range means the isolated stage totals are more reliable than a single E2E
percentage.

The NPU 7 repeat had this approximate work split. These are mixed wall and
device-event sums from an interleaved continuous pipeline, so percentages are
diagnostic shares of run wall, not mutually exclusive critical-path accounting.

| Section | Compiled, all 8 | Eager control |
| --- | ---: | ---: |
| layout total | 1.461 s (7.2%) | 2.303 s (9.8%) |
| image load + crop extraction | 0.268 s (1.3%) | 0.281 s (1.2%) |
| crop/prompt CPU preprocess + mRoPE + H2D | 1.694 s (8.3%) | 1.745 s (7.5%) |
| complete vision work | **4.235 s (20.7%)** | **6.331 s (27.0%)** |
| multimodal projector | 0.026 s (0.1%) | 0.113 s (0.5%) |
| text embedding/scatter/cache/prefill/head | 6.088 s (29.8%) | 6.534 s (27.9%) |
| continuous decode wall | 6.427 s (31.5%) | 5.814 s (24.8%) |

For the 157 requests that actually selected a graph, padding preparation plus
encoder/post-LN fell from 5.364 to 3.264 seconds: a 39.15% reduction, or 1.64x
speedup. Including eager overflow and vision embeddings, total vision work fell
from 6.331 to 4.235 seconds, a 33.10% reduction. Summed per-request prefill wall
fell by 18.37%.

Routing was identical in every compiled run:

- 118 requests selected bucket 512, 30 selected 1024, and 9 selected 2048.
- 157/160 requests compiled; three crops totaling 12,000 real tokens exceeded
  2048 and used eager overflow.
- Compiled requests carried 72,732 real rows in 109,568 physical rows: 66.38%
  useful. Including the unpadded overflow requests, the run-level fraction was
  69.70% (84,732 / 121,568).

### Two important findings

Strict greedy parity did not fully pass. Both compiled repeats differed from
both eager controls on the same one request,
`newspaper_..._region_071`: a 1628-row crop routed to bucket 2048. The compiled
text preserved the recognized content but omitted repeated `★` bullet markers,
producing 131 tokens instead of 147. The other 159 requests were token-exact.
This deterministic 159/160 result is not sufficient for the project's strict
token-parity gate.

The three-used-bucket run had about 9--10% higher interleaved decode throughput
than the paired all-eight-bucket runs. That result showed correlation, not that
resident vision graphs caused the difference. The denser 32-graph experiment
below does not reproduce a graph-count-dependent decode slowdown. Eagerly
loading every graph is still not free because warm startup time and graph
memory scale with the bucket count; lazy loading remains worth considering for
those reasons.

Warm-cache setup also matters for short offline jobs. All eight vision graphs
added about 6.5 seconds to setup (35.15 seconds total versus 28.58 eager), while
the three-used-bucket configuration set up in 31.26 seconds. Including setup,
even the fastest five-page all-bucket run was slower than eager; the page-wall
gain is intended to amortize in a persistent process.

The raw evidence is under
`tmp/08_offline_e2e_b1/five_pages_compiled_vision/`.

## Dense arbitrary-bucket experiment

The follow-up used this 32-shape policy:

```text
32..512 by 32, 576..1024 by 64, 1152..2048 by 128; larger crops eager
```

It repeated the same five pages with model-default `min_pixels` and divisors
2, 4, and 8. All runs used fp16, compiled B=4 continuous decode, cache length
2048, and a 768-token generation cap. The complete-vision time below sums patch
and position embedding, compiled-input preparation where applicable, the
encoder plus post-LayerNorm, and the projector. Setup is excluded.

| Minimum pixels | Real / physical vision rows | Useful | Complete vision | Useful vision rows/s | Text-prefill core | Text-prefill tok/s | Effective decode tok/s | Run wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default / 1 | 120736 / 125856 | 95.93% | 4.300 s | 28075 | 5.892 s | 5476 | 1121.02 | 20.817 s |
| default / 2 | 84732 / 87648 | **96.67%** | 3.620 s | 23406 | 5.900 s | 3943 | **1125.85** | 22.256 s |
| default / 4 | 69624 / 72352 | 96.23% | 3.425 s | 20329 | 6.577 s | 2963 | 1079.25 | 20.603 s |
| default / 8 | 62432 / 65440 | 95.40% | **3.219 s** | 19397 | 6.071 s | 2913 | 1112.54 | **19.640 s** |

Compared with matching eager controls, dense bucketing reduced complete vision
time by 29.97%, 43.82%, 49.36%, and 49.68% for divisors 1, 2, 4, and 8. Against
the old power-of-two graphs, the reductions were 19.48%, 15.79%, 10.67%, and
14.21%. Run wall improved by 4.91--13.58% versus eager and by 0.68--8.29%
against the old graphs. The smallest observed wall was 19.640 seconds at
default / 8.

The dense policy reduced padding from the old 66.34--72.39% useful range to
95.40--96.67%. Its effective decode throughput was 1079--1126 tok/s versus
1027--1121 tok/s for the old eight graphs: sometimes higher and sometimes
lower, with no monotonic penalty from keeping 32 vision graphs resident. A
second default / 4 run had the same generated outputs but a 24.056-second wall,
which reinforces that small E2E differences are noisier than the isolated
vision-stage totals.

Warm caches avoid recompilation but not graph initialization. Loading and
first-running all 32 cached shapes took 23.6--27.1 seconds of vision setup,
versus roughly 6.3--7.1 seconds for the old eight-shape set. The first run after
the source hash changed rebuilt the set and spent 1496.4 seconds in vision
setup. Dense buckets therefore make sense in a persistent process, but not as
free startup for short-lived jobs.

Strict greedy parity passed on all 160 requests at default / 1, default / 2,
and default / 8. At default / 4, 159/160 requests matched eager. The sole
compiled difference was deterministic across two runs and occurred at an exact
192-row bucket with no padding: compiled produced `R_{kk}` while eager produced
`R_{k k}`. Dense routing fixes the earlier 1628-to-2048 failure and greatly
improves the gate, but the default / 4 result proves that zero padding alone
does not guarantee token-exact compiled/eager output.

The raw evidence is under
`tmp/08_offline_e2e_b1/five_pages_dense_vision_buckets/`.
