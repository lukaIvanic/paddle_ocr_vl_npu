# Split-K32 reduction optimization

## Result

Three separately packaged reduction controls were tested for the B1 FP16
16Q/2KV/D128 GQA AIV operator at physical KV1024 and valid length 769. The
strongest control keeps partition 0 in the even reducer core's UB and loads
only partition 1 from GM. It reduced the isolated profiled task from 22.63 us
to 20.90 us, or 7.64%.

That improvement did not survive the production-shaped TorchAir graph. Across
54 attention calls, the current split-K32 task averaged 13.380 us and the
local-partial control averaged 13.319 us, only 0.46% lower. This is about 1.10
us saved across all 18 attention layers in one token step.

The warm-cache B1 reverse-order sequence measured 854.81 tok/s for the current
kernel and 843.82 tok/s for the control. The control was 1.28% slower. Keep the
existing split-K32 package as the best B1 implementation. Retain the new
package as research evidence, not as a promoted default.

All device work used physical Ascend 910B2 NPU6, CANN 9.0.0, torch
2.10.0+cpu, torch-npu 2.10.0, FP16, BNSD, B1, Q `[1,16,1,128]`, and K/V
`[1,2,1024,128]`. Every completed control passed stock FP16 tolerance and the
independent CPU FP32 reference. Every profile proved 32 blocks, zero AIC time
and cycles, and zero cube utilization.

## Controls

| Control | Synchronization/reduction change | Task | Vector | Scalar | MTE2 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Current split-K32 | global barrier, generic N-way combine | 22.63 us | 6.18 us | 8.56 us | 4.19 us | retained |
| Pairwise signal | initial global barrier plus `IBSet`/`IBWait` at the tail | 26.14 us | 6.24 us | 7.68 us | 5.01 us | reject |
| Two-way algebra | direct two-part weights, no unused output LSE | 22.51 us | 6.13 us | 6.11 us | 4.99 us | no material gain |
| Local partial | even reducer retains partition 0 in UB | 20.90 us | 6.11 us | 6.84 us | 3.27 us | direct win only |

Pipeline counters overlap and must not be added.

### Pairwise synchronization

Worker blocks are paired `(0,1)`, `(2,3)`, and so on. Arbitrary operator
workspace is not guaranteed to start at zero, so the safe control first zeroed
one 32-byte inter-core flag slot per block and used a balanced `SyncAll()`
before compute. Odd workers then called `IBSet`; even workers called `IBWait`
only for their partner and reduced the pair.

This proved the protocol without deadlock, but it moved rather than removed
the global rendezvous. Task time increased by 15.5%. A pairwise protocol is not
useful unless flag initialization can be moved outside the hot call without a
stale-flag race.

### Specialized two-way algebra

The forced 32-block topology has exactly two partitions per query head. The
control replaced the generic max/sum/log/re-exponentiation path with one max,
two subtracts, one 16-float exponential, one denominator add, and two divides.
It removed about 2.45 us of scalar-pipeline work in the direct profile, but
MTE2 increased by about 0.80 us and task time changed by only 0.12 us.

### UB-resident local partial

The original reducer mapping used blocks 0-15, although the worker mapping is
`(q0p0, q0p1, q1p0, q1p1, ...)`. The control remapped reduction to the even
workers. Each even worker therefore still owns the correct partition-0 D128
result in `bmm2ResUb` after `SyncAll()`. It scales that resident vector in
place and fetches only the partition-1 vector from GM. This removes one
512-byte partial reload and the generic two-row accumulation path per query
head.

The isolated MTE2 counter fell from 4.19 to 3.27 us, and the task fell to
20.90 us. In the static compiled decoder, however, the current workspace
traffic is already hot or overlapped. The 54-call average changed by only
0.061 us per attention task.

## Correctness

The candidate-first eager gate returned without timeout. The custom output
hash matched the earlier correct two-way control. Maximum absolute difference
from stock was `1.220703125e-4`. Maximum absolute difference from the
independent CPU FP32 attention reference was `6.432831e-5`. TorchAir passed the
same required checks.

## Real B1 measurements

Each lane used the full 18-layer model and 103,424-output LM head, B1,
KV1024, initial position 768, 20 warmups, and 200 measured steps.

The first current/candidate/candidate/current sequence included one cold graph
load for each package:

| Lane | Kernel | First call | Mean step | Throughput |
| --- | --- | ---: | ---: | ---: |
| A | current | 12.834 s | 1.3705 ms | 729.67 tok/s |
| B | local partial | 12.867 s | 1.3605 ms | 735.01 tok/s |
| C | local partial | 0.222 s | 1.2150 ms | 823.02 tok/s |
| D | current | 0.251 s | 1.1831 ms | 845.26 tok/s |
| Current mean | - | - | 1.2768 ms | 787.47 tok/s |
| Candidate mean | - | - | 1.2878 ms | 779.02 tok/s |

Because cadence drift was large, a second warm-cache
candidate/current/current/candidate sequence was run:

| Lane | Kernel | First call | Mean step | Throughput |
| --- | --- | ---: | ---: | ---: |
| B | local partial | 0.223 s | 1.1848 ms | 844.04 tok/s |
| A | current | 0.219 s | 1.1656 ms | 857.91 tok/s |
| A | current | 0.218 s | 1.1741 ms | 851.70 tok/s |
| B | local partial | 0.223 s | 1.1854 ms | 843.61 tok/s |
| Current mean | - | - | 1.1699 ms | 854.81 tok/s |
| Candidate mean | - | - | 1.1851 ms | 843.82 tok/s |

The candidate has no end-to-end speed claim. Peak allocation delta was the
same 624,128 bytes in every lane.

## Bottleneck conclusion

The direct ACLNN profile overstated the value of reducing workspace traffic.
In the real compiled graph, the 32-core attention task is already about 13.3
us and its vector lane remains about 6.0 us. Attention totals about 0.240 ms
per token across 18 layers, near one fifth of the 1.17-1.19 ms warm token step.

Small copy/combine edits now have an end-to-end ceiling of only a few
microseconds per token. A material next gain needs one of these:

1. reduce the 6 us main vector arithmetic with a new safe topology, not a
   fourth sub-512 split of the recovered kernel;
2. remove the global synchronization without per-call flag initialization or
   a stale-workspace race;
3. fuse attention with adjacent graph work; or
4. optimize the larger non-attention part of the decoder.

## Remote artifacts

```text
.runtime_cache/paddle_gqa_split_k32_pairwise_sync_increfa_aiv/
.runtime_cache/paddle_gqa_split_k32_two_way_reduce_increfa_aiv/
.runtime_cache/paddle_gqa_split_k32_local_partial_reduce_increfa_aiv/
```

Exact hashes, measurements, and artifact paths are in [summary.json](summary.json).
