# Serving vision D80 weight padding + joint FP32 RoPE

Source `3b222363`, Ascend 910B2 physical NPU6. Ordinary B2 decoder, client
concurrency two throughout the pipeline. No speculative/oracle routing.
The same frozen development-100 seed1 and complete-request warmup are used.
Neither validation-100 nor confirmation-1000 was run: throughput still fails.

## Implementation

Opt-in `serve_crop_ocr_api.py --vision-attention-weight-padding` moves the
existing lab formulation into `VisionPrefillStage`:

- Zero-extend Q/K/V projection rows and output-projection columns from D72 to
  D80 at load time, before NZ conversion. Preserve both 36-coordinate RoPE
  halves in two 40-coordinate halves, with four zeros in each half.
- Keep original D72 attention scaling, original frequency table, model
  checkpoint values, image pixels/resampling and real vision tokens.
- Apply the same FP32 half-RoPE formula once to adjacent Q/K. Extend factors
  with neutral cos=1/sin=0 once per encoder forward, not in each layer.
- Use existing stock PromptFA; remove per-layer Q/K/V runtime padding and the
  attention-output 80-to-72 slice. No custom operator or precision reduction.
- Give the vision variant an explicit cache identity. Other inference settings
  and all decode cache/configuration fields are unchanged.

Two CPU tests verify projection/bias preservation, zero coordinates, original
scale, masked two-layer attention for B1/B2, RoPE halves, and a static fullgraph
`backend=eager` compile boundary. Those tests are not NPU performance evidence.
Nine scheduling tests and four preprocessing tests also pass. Real NPU serving
is the evidence below. The option remains **off by default**.

## Results — retain both attempts

| Development-100 | Completed tables/s | P95 wall latency | Measured wall |
|---|---:|---:|---:|
| Saved CPU-optimized B2 control (`f0f9df06`) | 2.914567462409707 | 1.9137373317964366 s | 34.31040841899812 s |
| Candidate, fresh compilation | 2.4664988550188705 | 2.5452873182191977 s | 40.54329877207056 s |
| Identical candidate, cache-loaded restart | 2.8926129443484783 | 1.9431582124438127 s | 34.57081950607244 s |

The first candidate is a recorded regression, not a discarded timing. A single
restart tested a concrete discrepancy: decoder initialization recompiled in
13.309894 s versus 0.243839 s in the saved control. No inference setting or
sampling changed for the cache-loaded run. Reload recovers most of the slowdown,
consistent with a compile/load execution-path effect, but does not establish
its exact cause. The remaining difference needs investigation, not attribution
to a foreign NPU process or an unsupported hardware explanation.

| Diagnostic | Control | Fresh compile | Cache reload |
|---|---:|---:|---:|
| Development-100 vision-device total | 4.240170 s | 3.514195 s | 3.505439 s |
| Development-100 vision/text-prefill wall total | 6.247244 s | 5.507457 s | 5.496875 s |
| Decode calls, service lifetime incl. warmup | 24,119 | 23,810 | 24,170 |
| Decode event-region total, incl. warmup | 27.590705 s | 34.764998 s | 28.802046 s |
| Event-region average per decode call | 1.143941 ms | 1.460101 ms | 1.191644 ms |

Vision improves about 17%, but the **measured complete serving result is not
better than the saved control**. Do not substitute a projected result with
control decode speed. Event regions are not a kernel-only profile: host
submission gaps can appear between device events. Service-lifetime statistics
include the sacrificial warmup; request percentiles do not. CPU spans and
decode residency can overlap across requests and must not be summed or
subtracted from wall latency. Initial fill/final drain remain timed.

## Outputs and input contract

Both candidate runs finish all 100 requests with EOS, no failures/rejections/
truncations, and 100/100 native-token parity with each other. Each is 99/100
identical to the saved B2 control. Image dimensions, real vision/text tokens,
source membership/order, and client outstanding count are checked by
`analyze.py` (pass `cache_reload` for the second run).

Only `page_001417_table_2` changes. Native edits:

- Offset 38: `[94964]` -> `[94489]`; label `电压初赛` -> `电压初转`.
  Ground truth is `电压切断`: both outputs are wrong there.
- Offset 45: `[94176,95172]` -> `[11512]`; `电票波形一个周期` ->
  `电源波形一个周期`, matching the ground-truth cell text.

These are real content changes, not HTML-only equivalence. They arise with a
mathematically equivalent zero-extension/FP32-RoPE path, reproduced after reload.
There is no blanket quality-regression allowance and no claim that logits were
captured. Table structure and other text are unchanged. Total generated tokens
are 41,119 versus 41,120, so this cannot explain the performance regression.
`changed_table_ground_truth.json` is evaluation-only evidence, never a serving
input; no source-ID-based routing/orientation is used by this ordinary API.

## Ownership, setup, and reproduction

Exact server arrays/configuration, timestamps and client settings are preserved
in command logs, summaries and results. Vision buckets are
256/384/512/640/768/1408/1920/2048/2944/4096; text buckets are
128/256/512/1024/1152; decoder KV4096, selected native vocabulary unchanged.
The existing runtime initializes its own graphs; no separate graph builder was
introduced. Initial setup took 332.318533 s (vision 290.518241 s); reload setup
took 34.749875 s. Each server receives the same one-request warmup outside the
development timing before random100 seed1, max-in-flight two.

Direct-host checks map fresh worker PID1782757 to container2472013 and reload
worker PID1794005 to container2479328, with full API parent commands. Saved
approximately five-second NPU6 process snapshots show only that run's worker
during measurement (7 and 6 samples respectively). This is sampled monitoring,
not continuous kernel tracing. Logs cover both sides of each measured window.

Both owned servers exit zero after draining and saving service summaries.
Host checks at 02:46:51 and 02:52:54 CST on 2026-09-06 confirm respective
workers/parents gone and NPU6 free. Owned monitor PIDs1782635 and1793974 were
stopped afterward. No other user's process was signaled. No other card was used.

Next diagnostic: explain the residual ~48 microseconds per decode call in the
cache-loaded production path against the established control, before treating
the faster vision stage as an end-to-end optimization or adding another change.
