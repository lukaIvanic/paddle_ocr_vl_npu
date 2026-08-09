# Four-AIV-block GQA topology control

## Result

Doubling the grouped control from two to four AIV blocks works and is faster,
but it does not beat the current 16-block production package.

All measurements used physical Ascend 910B2 NPU6, FP16 BNSD attention,
Q `[1,16,1,128]`, K/V `[1,2,KV,128]`, a bool mask, CANN 9.0.0,
torch 2.10.0+cpu, and torch-npu 2.10.0. Production code was unchanged.

| B1/KV1024 TorchAir package | Mean step | Median | P95 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| Current 16-block | 1.3636 ms | 1.3570 ms | 1.3946 ms | 733.35 tok/s |
| Four-block half-group | 1.8382 ms | 1.8343 ms | 1.8382 ms | 544.00 tok/s |
| Two-block grouped | 2.6582 ms | 2.6567 ms | 2.6606 ms | 376.20 tok/s |

Four blocks reduced latency by 30.85% and increased throughput by 44.60% versus
two blocks. They remained 34.81% slower in latency and 25.82% lower in
throughput than 16 blocks.

## What the control does

The separate `grouped_half_control` package applies patches 0007 and 0008. Its
host tiler creates two four-query-head slices for each of the two KV groups.
The supported public resource attribute remains 16, while the device profile
proves the actual launch is `Block Num=4`.

Each block still invokes the existing per-head vector algorithm four times. It
does not load one K/V tile and reuse it for all four heads. This isolates the
parallelism question before a more invasive shared-UB rewrite.

## Correctness

KV128, KV512, KV1024, KV1536, and KV2048 all passed stock FP16 tolerance and an
independent CPU FP32 matmul-softmax-matmul reference. Maximum absolute
stock-versus-custom differences were 2.4414e-4, 3.0518e-5, 6.1035e-5,
6.1035e-5, and 6.1035e-5, respectively. Every custom head mapped to the same
stock query-head index.

## Kernel profile

The bounded KV1024 custom-only pipe profiles contain three tasks per package:

| Package | Blocks | Task | AIV | Vector | Scalar | MTE2 | AIC | Cube |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current 16-block | 16 | 21.78 us | 20.09 us | 11.40 us | 7.98 us | 4.74 us | 0 | 0% |
| Four-block half-group | 4 | 54.83 us | 54.02 us | 45.45 us | 18.10 us | 17.93 us | 0 | 0% |
| Two-block grouped | 2 | 99.96 us | 99.24 us | 90.83 us | 31.06 us | 34.78 us | 0 | 0% |

Four blocks reduce task time by 45.14% versus two blocks. This is a real
kernel-level improvement, although it is only a 1.82x speedup from 2x as many
blocks.

The MemoryAccess profile shows why this is not K/V reuse:

| Package | GM-to-UB requested | Main-memory read requested |
| --- | ---: | ---: |
| Current 16-block | 8,236 KiB | 8,442.46 KiB |
| Four-block half-group | 8,218 KiB | 8,296.33 KiB |
| Two-block grouped | 8,215 KiB | 8,255.67 KiB |

Unique direct Q/K/V/mask input is 1,029 KiB. All variants request about eight
times those unique bytes because K/V is still loaded once per query head.

Even if true sharing removed every microsecond above the four-block 45.45 us
vector lane, subtracting that best-case gap across 18 layers gives 1.6692 ms,
or at most 599.08 tok/s. This intentionally overstates the benefit because
scalar and transfer pipelines overlap. The bound is still 18.31% below the
current 16-block throughput.

The clean eager ACLNN call boundary measured 173.30, 171.55, and 170.32 us for
2, 4, and 16 blocks. Fixed call overhead hides the task-level difference. The
compiled 18-layer B1 decoder exposes it.

## Build and retained artifacts

The independent build was produced from pinned `ops-transformer` commit
`afe72144f9f2ac8441929035795db88a111b30c5` and local experiment commit
`ca152b5`. The fresh wrapper took about 4 minutes 55 seconds; the actual device
kernel generation completed much earlier. Package, object, and metadata hashes:

```text
package cdd138c51adc0ddb9c6f2b8bec1c411159b17351c18c8b8ff3bbc2e3eeb73edd
object  13222ded0fb0c6ae844e306b7c4306e8662fbda18a633ca76ba37be7a82fda9c
json    9423e651d000f0556efd3839ff3be69e8bf832eb55d887074680da654b0e7870
```

The first TorchAir call took 24.03 seconds and is excluded from all steady
timing. Remote raw artifacts remain under:

```text
.runtime_cache/paddle_gqa_grouped_half_increfa_aiv/builds/grouped_half_v2/
.runtime_cache/paddle_gqa_grouped_half_increfa_aiv/results/grouped_half_v2/
.runtime_cache/paddle_gqa_core_sweep/clean_kv1024/
.runtime_cache/paddle_gqa_core_sweep/profiles/
```

An attempted extra warm-cache rerun selected NPU5 while another process already
owned memory there. It did not reach the measured window within five minutes,
was stopped, and contributes no timing to this report.

## Decision

Keep the current 16-block query-head-parallel package. Four blocks validate the
expected direction and recover much of the two-block loss, but they remain
materially slower in the real B1 forward pass. A future grouped design must
change vector arithmetic efficiency or provide true cross-head data reuse
without sacrificing this much parallelism.
