# Forced 48-AIV-block split-K control

This separately vendored package tested a three-way sequence split for the B1
FP16 16Q/2KV/D128 GQA AIV operator.

At KV1024, forcing 48 blocks creates partitions of about 342/341/341 tokens.
The first custom task did not complete and the runtime reported its first
three-minute synchronization timeout. This is why the upstream tiler refuses
partitions shorter than 512 tokens. No performance or correctness claim is
made for that run.

At KV1536, all three partitions contain 512 tokens. The operator completed on
physical Ascend910B2 NPU6, passed stock tolerance and the independent FP32 CPU
reference, and the profile proved `Block Num=48`, nonzero AIV counters, zero
AIC execution, and zero cube utilization.

The matched KV1536 results do not justify 48 blocks:

| Lane | Blocks | Task | Vector | Scalar | MTE2 | TorchAir boundary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Split-K32 | 32 | 24.79 us | 8.90 us | 8.39 us | 5.29 us | 243.93 us |
| Split-K48 | 48 | 33.68 us | 6.17 us | 9.78 us | 9.35 us | 251.92 us |

The third split reduces vector time by 30.7%, but task time grows by 35.8% and
the TorchAir boundary grows by 3.28%. The extra global synchronization,
workspace traffic, and three-part stable-softmax/output reduction dominate.

The full B1/KV1536 TorchAir ABBA sequence was effectively tied. The arithmetic
means were 801.12 tok/s for 32 blocks and 801.83 tok/s for 48 blocks, a 0.09%
difference. This is below the observed process cadence variation and is not an
improvement.

The Python graph wrapper and direct probe now reject 48 cores below KV1536 so
the known-stalling topology cannot be launched accidentally. Raw result JSONs
are in [raw](raw/), and [summary.json](summary.json) contains the compact result.
