# Mixed M16 pairwise-split layout result

This experiment changes only how the mixed M16 graph separates its first eight
verifier tokens from its last eight draft tokens. Six direct token slices per
transformer layer were replaced by three two-output `SplitV` operations.
Attention, prefetching, matmuls, cache shapes, positions, precision, and the
compact vocabulary stayed unchanged.

All runs used physical Ascend 910B2 NPU 6, 10 warmups, and 50 measured calls.
All 16 native output token IDs exactly matched the saved B8Q1 and B1Q8 anchors.

## Ordinary timing

| Run | Graph cache before setup | Median | P95 |
|---|---|---:|---:|
| Direct-slice control | warm | 2.354 ms | 2.363 ms |
| Pairwise split, first process | cold | 2.437 ms | 2.443 ms |
| Pairwise split, warm process 1 | warm | 2.320 ms | 2.326 ms |
| Pairwise split, warm process 2 | warm | 2.339 ms | 2.347 ms |

The two warm pairwise-split medians average 2.329 ms. This is 24.4 us, or
1.04%, faster than the fresh warm-cache control. The first process loaded a new
compiled graph and did not reproduce that steady result even after its ten
warmup calls, so it remains recorded rather than hidden.

## Profiler result

The prior five-call direct-slice profile and the new five-call pairwise-split
profile used the same NPU and unchanged mixed-transformer sources except for
this layout edit.

| Layout operation | Direct slices | Pairwise split | Saving |
|---|---:|---:|---:|
| StridedSlice | 108 calls, 96.6 us | 0 calls, 0 us | 96.6 us |
| SplitV | 20 calls, 38.2 us | 20 calls, 36.0 us | 2.1 us |
| Transpose | 19 calls, 148.4 us | 19 calls, 147.0 us | 1.5 us |
| Concat | 21 calls, 45.0 us | 21 calls, 46.8 us | -1.9 us |
| ConfusionTranspose | 18 calls, 111.3 us | 18 calls, 131.9 us | -20.7 us |
| **Selected layout total** | **439.4 us** | **361.8 us** | **77.6 us** |

The complete profiled device call fell from 2,470.8 us to 2,324.7 us. Some of
that 146.2 us difference is contextual execution variance outside the changed
layout operators, so the ordinary warm-cache timing is the conservative result.

The change removes exactly the intended 108 kernels and gives a small steady
latency improvement. It is retained in commit `217ed687`.
