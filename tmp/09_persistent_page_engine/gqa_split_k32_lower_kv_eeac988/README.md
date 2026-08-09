# Lower-KV split-K boundary and current bottleneck

This retained run answers two questions for the B1 FP16 16Q/2KV/D128 custom
GQA AIV operator:

1. Can the forced 32-block topology improve cache lengths below KV1024?
2. Why is a real 32-block attention task longer than its 6 us vector counter?

All device work ran on Ascend 910B2. The lower-KV direct profiles ran on
physical NPU4. The matched real-graph KV1024 profiles and lower-KV forward
passes ran on physical NPU6. Every direct profile passed stock FP16 tolerance
and the independent CPU FP32 attention reference. Every row had zero AIC time
and zero cube utilization.

## Lower-KV topology

The c32 package retains the upstream minimum of 512 KV tokens per split. A
two-way split therefore starts only at physical KV1024. The profiler, rather
than the requested resource attribute, proves the result:

| Physical KV | Requested | Actual blocks | Task | Vector | Scalar | MTE2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 16 | 15.56 us | 2.31 us | 6.15 us | 1.71 us |
| 256 | 32 | 16 | 17.41 us | 3.64 us | 7.28 us | 2.35 us |
| 512 | 32 | 16 | 19.38 us | 6.25 us | 7.53 us | 3.24 us |
| 768 | 32 | 16 | 22.54 us | 8.95 us | 8.34 us | 4.34 us |
| 1024 | 32 | 32 | 22.63 us | 6.18 us | 8.56 us | 4.19 us |

Pipeline counters overlap and must not be added. The lower-KV result is a safe
16-block fallback, not a hidden 32-block launch. Lowering the 512-token floor
is unsafe with the recovered kernel: the independent 48-block control stalled
when three KV1024 partitions were about 341 tokens each.

## Real forward behavior below KV1024

The requested-32 preset measured 766.9, 778.0, 771.4, and 758.1 tok/s at
KV128/256/512/768. These are separate-process measurements and are not a
monotonic scaling law. At KV512, the matched 16-block lane measured 781.9 tok/s
versus 771.4 for requested-32. At KV768, it measured 752.5 versus 758.1. The
opposite small differences, together with the identical 16-block topology,
show process cadence noise rather than a lower-KV 32-core benefit.

## Matched real-graph KV1024 profile

Three compiled forward steps produced 54 attention calls in each lane on the
same physical NPU6:

| Lane | Blocks | Task | AIV total | Vector | Scalar | MTE2 | MTE3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 16 | 16.21 us | 14.85 us | 11.25 us | 4.65 us | 5.31 us | 0.12 us |
| Split-K | 32 | 13.38 us | 12.20 us | 6.03 us | 4.87 us | 3.25 us | 0.38 us |

The 32-block task is 17.46% shorter and its vector lane is 46.36% shorter. It
saves about 50.96 us across the 18 attention layers in one decoder step. Only
about half of the vector-pipe saving reaches task duration.

The exact recovered source explains the rest of the time:

1. Two AIV blocks compute disjoint 512-token partial attention results for
   each query head.
2. Each block writes its partial output and log-sum-exp state to GM workspace.
3. All 32 blocks execute a global `SyncAll()`.
4. `FlashDecodeCompute()` returns immediately on blocks 16-31.
5. Blocks 0-15 load both partials, recompute stable log-sum-exp weights using
   max/subtract/exp/add/log operations and several vector-pipeline barriers,
   scale and sum the two D128 outputs, cast, and write the final result.

The profiler's vector field reports active vector-pipeline issue time. It is
not kernel wall time. Task duration also contains scalar/control instructions,
MTE transfers, dependencies and pipeline barriers, core imbalance at the
global barrier, the GM workspace round trip, the final reduction, and about
1.18 us between reported AIV time and task completion.

The next useful control is reduction-focused: replace the all-32 `SyncAll`
with a safe pairwise producer/reducer protocol, then reduce the two partial
states without a full global barrier. Huawei's A2 APIs expose `IBSet`/`IBWait`
for inter-core dependency signaling, but that design must be built as a
separately named package and proved against deadlock before any speed claim.

Raw JSON and matched `kernel_details.csv` files are in [raw](raw/). The compact
machine-readable result is [summary.json](summary.json).
