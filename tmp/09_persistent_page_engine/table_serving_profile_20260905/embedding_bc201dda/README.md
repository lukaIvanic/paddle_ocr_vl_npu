# Real vision-embedding profile: convolution dominates

Source `bc201ddaf8a9bddbda85e526391c93c6d25211e1`, physical NPU6, 910B2.
Unchanged ordinary B2 serving model, client C1; no competing request needed
for this component capture. Five complete requests warm the exact path, then
three separate real embedding forwards are captured. Every request independently
recomputes its inputs, visual features, prefills and output; only weights and
runtime resources are reused. The repeated crop is diagnostic, not a serving
acceptance sequence or a corpus-specific inference rule.

Each capture sees pixels `[1,3840,3,14,14]`, grid `[1,48,80]`, and15 kernels.
Conv2D durations are8225.660,8205.400,8214.400us. Complete kernel-duration
sums are8555.700,8537.900,8549.180us. Convolution is approximately96% of the
embedding kernel cost. Position interpolation is approximately115us; layout
operations are not the dominant issue in this stage.

The observed Conv2D input is converted from NCHW `[3840,3,14,14]` to
NC1HWC0 `[3840,1,14,14,16]`; weights become FRACTAL_Z `[196,72,16,16]`.
There is one full-patch output per batch element. Flattening each patch and
the corresponding weights expresses the same real-valued projection as
`[3840,588] @ [588,1152] + bias`. This is a content-independent algebraic
candidate, **not yet implemented or measured here**. Finite-precision output
agreement and full-serving benefit need testing before promotion.

All8 requests end at EOS with identical native output streams,973 input tokens,
and unchanged3840 real vision tokens. No generated text was re-encoded.
The first request was cold; profiler analysis additionally blocks the last
three requests. None of this run's HTTP throughput/percentiles is a performance
claim. `analyze.py` verifies capture geometry, kernel counts and outputs.

Setup29.565605s. Both client and server exit zero. Direct host inspection at
2026-09-06T06:40:33+08:00 recorded worker1931144 with PPid1931142 and
NSpid1931144/2529224. Container parent2529222 matches the profile adapter.
The owned five-second monitor PID1930320 covers the run; sampled monitoring
is not continuous tracing. Parent, worker and monitor were stopped after the
capture, and NPU6 release was checked. Raw binary captures remain remotely;
CSV exports, configuration, logs and native outputs are preserved locally.
