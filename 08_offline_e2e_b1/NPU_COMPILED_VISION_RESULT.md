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

Keeping all eight graphs resident also reduced interleaved decode throughput by
about 9--10%. Restricting residency to the three buckets used by this workload
restored decode to 1226.45 effective tok/s and reduced run wall to 21.488
seconds. Therefore all bucket *paths* should remain available, but eagerly
loading every graph is not free. A production design should lazily load/warm
the reachable buckets or otherwise release unused graph state.

Warm-cache setup also matters for short offline jobs. All eight vision graphs
added about 6.5 seconds to setup (35.15 seconds total versus 28.58 eager), while
the three-used-bucket configuration set up in 31.26 seconds. Including setup,
even the fastest five-page all-bucket run was slower than eager; the page-wall
gain is intended to amortize in a persistent process.

The raw evidence is under
`tmp/08_offline_e2e_b1/five_pages_compiled_vision/`.
