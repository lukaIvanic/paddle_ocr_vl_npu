# Loop-boundary prefetch diagnostic (not a serving performance result)

Source `daa3dde1`, physical NPU6, 910B2. The preceding real B2 profile found
first-layer matmuls totaling 43.302 us versus roughly 27 us for later layers.
The existing complete-layer-ahead schedule did not prefetch layer zero for
the next decode iteration. The opt-in `*_loop_prefetch` preset adds those five
weight hints alongside the final layer's existing LM-head prefetch. Two CPU
tests verify exact weight references and that no other preset setting changes.

The existing real-serving profiling adapter was reused: full request warmup,
two actual random-seed1 requests, 32 two-active-slot decode steps before
profiling, then five profiler warmups and twenty captured iterations.
This invocation compiles through the ordinary API path. Its client latencies
are perturbed by profiling and are not a development/acceptance measurement.

The user then asked to examine image-derived length prediction and scheduling;
the planned unprofiled random100 test has **not** run. No performance promotion
or gate success is claimed. Raw profiling exports remain on the container;
selected CSVs and the losslessly compressed trace are retained here.

Owned host worker1827050 mapped to container2489734; its parent was
1827048/container2489732. The server exited zero after both diagnostic requests
finished. At 2026-09-06 04:13:49 CST the host showed NPU6 free and both processes
gone. Owned monitor1826784 was stopped. No other device or process was affected.

## Captured result

After removing the prior partial async graph before the first
`UpdateModelParam_static_bin`, both captures contain twenty complete graphs,
each with91 MatMul calls. First-layer matmuls improve43.302 ->31.480us;
all matmuls517.259 ->507.450us. However, total model kernel-duration sum is
essentially unchanged:1215.508 ->1213.340us. The first-layer cache hint helped
its intended local target but this capture provides no material full-model
win. Both diagnostic outputs are native-token-identical to the saved B2
control. Do not infer a throughput gain from the first-layer result alone.
