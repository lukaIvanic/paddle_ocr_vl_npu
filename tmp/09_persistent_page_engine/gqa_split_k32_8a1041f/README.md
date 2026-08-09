# Forced 32-AIV-block GQA split-K experiment

## Result

The separate `split_k32_control` package launches 32 actual AIV blocks at the
real B1/KV1024 decode shape. It passed eager and TorchAir correctness. In one
same-device `16 -> 32 -> 32 -> 16` full-decoder sequence, 32 blocks averaged
829.59 tok/s versus 778.38 tok/s for 16 blocks: 6.58% higher throughput and
6.32% lower mean latency.

All measurements used physical Ascend 910B2 NPU6, CANN 9.0.0,
torch 2.10.0+cpu, torch-npu 2.10.0, FP16 BNSD Q `[1,16,1,128]`, K/V
`[1,2,1024,128]`, and a bool mask. The full-decoder run used the real 18-layer
model and 103,424-output LM head, not a synthetic LM head.

## Why a separate tiler was required

The normal GQA FlashDecode heuristic only splits the sequence at KV2048 or
longer. Requesting 32 cores at KV1024 therefore still launched 16 blocks.
Patch `0009-force-split-k32-control.patch` is packaged under the independent
`paddle_gqa_split_k32_increfa_aiv` vendor. It forces two disjoint 512-token
partitions for each of the 16 query heads only when the requested core count is
exactly 32. The retained 16-core package is unchanged.

The independent package was built from pinned upstream `ops-transformer`
commit `afe72144f9f2ac8441929035795db88a111b30c5` and repository commit
`8a1041f`.

```text
package 3302c24ffa5cf1490db9ecfe39cdb99e41fa3ccd0051c53288962729bff359ca
object  ecbeca46a7cf39c01b6f9ecf1dacc451d3d475395c0c8db927dea00b8822aa9a
json    2e47b39b25a2555fcc0265e8bac155519be249c8052359f512021632cca31cad
```

The device object and metadata hashes match the retained package because patch
0009 changes only host tiling. The package hash differs because it uses a
separate vendor and host library.

## Correctness and AIV proof

At KV1024 with valid length 769:

- eager and TorchAir passed stock FP16 tolerance;
- maximum absolute stock/custom difference was `1.220703125e-4`;
- the independent CPU FP32 comparison passed with custom maximum absolute
  difference `6.4328e-5`;
- the profile reported `Block Num=32`;
- AIC time, cycles, MAC time, and cube utilization were zero;
- AIV time and cycles were nonzero.

The package metadata contains the non-FlashDecode and FlashDecode MIX kernels
with `taskRation: 0:1`, no `_mix_aic` function, and inter-core synchronization
enabled. `Task Type` alone is not used as proof.

## Kernel and transfer profile

The bounded pipe profile captured three 32-block tasks. Their task durations
were 24.36, 21.72, and 21.80 us. Mean pipeline counters were:

| Package | Blocks | Task | Vector | Scalar | MTE2 | MTE3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained 16-block | 16 | 21.78 us | 11.40 us | 7.98 us | 4.74 us | - |
| Forced split-K | 32 | 22.63 us | 6.18 us | 8.56 us | 4.19 us | 0.37 us |

Vector time fell by 45.8%, but the bounded total task did not fall. Scalar
work, inter-core synchronization, and partial-softmax/output combination become
the critical path. Pipeline counters overlap and are not additive.

MemoryAccess measured 8,282 KiB of GM-to-UB requests for the 32-block call and
8,236 KiB for the matched 16-block call. The increase is 46 KiB, or 0.56%, not
2x. Each paired core reads one non-overlapping sequence half. Unique direct
Q/K/V/mask input is 1,029 KiB; the remaining repeated traffic comes mostly from
reading each KV head for its eight query heads.

The all-vector kernel already uses tile-level UB prefetch. Its main VECIN queue
calls `InitBuffer(inputQue2, 2, 32_KiB)`, enabling double buffering, and starts
copying V into UB before vector softmax. A whole 512-token K/V half is about
256 KiB per core before other working storage, so the kernel streams tiles
instead of keeping the whole half resident in UB. The installed
`torch_npu.npu_prefetch` API is an L2-cache prefetch and cannot place data into
a selected core's UB.

## Real B1 TorchAir result

One physical NPU was selected before the four independent processes. Each lane
used B1, physical KV1024, initial position 768, 20 warmups, 200 measured steps,
FP16, NZ decoder/LM-head weights, the full model, and the same package.

| Lane | Preset | Blocks | First call | Mean | Median | P95 | Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | retained | 16 | 12.999 s | 1.3509 ms | 1.3261 ms | 1.4046 ms | 740.23 tok/s |
| B | split-K | 32 | 12.359 s | 1.2411 ms | 1.2350 ms | 1.2755 ms | 805.76 tok/s |
| C | split-K | 32 | 0.228 s | 1.1718 ms | 1.1704 ms | 1.1737 ms | 853.41 tok/s |
| D | retained | 16 | 0.252 s | 1.2247 ms | 1.2231 ms | 1.2270 ms | 816.53 tok/s |
| 16 mean | - | 16 | - | 1.2878 ms | - | - | 778.38 tok/s |
| 32 mean | - | 32 | - | 1.2064 ms | - | - | 829.59 tok/s |

Both pairwise comparisons favor 32 blocks. Cadence drift is visible across the
sequence, so the full ABBA mean is the result; 853.41 tok/s is not presented as
a new baseline. Peak measured allocation delta was the same 624,128 bytes for
all four lanes.

The full TorchAir pipe profile captured 54 custom attention task executions
across three decoder steps. `op_statistic` reported 13.424 us average. The three
detailed exported rows all had `Block Num=32`, nonzero AIV counters, and zero
AIC/cube counters. There is no matched 16-block full-graph pipe capture in this
experiment, so do not use that 13.424 us row as a direct 16/32 comparison.

## Reproduction and raw artifacts

Build the independent package:

```sh
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup
PADDLE_GQA_EXPERIMENT_VARIANT=split_k32_control \
RUN_ID=20260809T185123Z \
bash 09_persistent_page_engine/custom_ops/paddle_gqa_increfa_aiv/build.sh
```

Run the candidate-first eager profile after sourcing the printed package
environment:

```sh
cd /workspace/repos/paddle_ocr_vl_npu/09_persistent_page_engine
/usr/local/python3.12.13/bin/python3 \
  scripts/probes/compare_paddle_gqa_increfa_aiv.py \
  --backend eager --kv-length 1024 --valid-kv-length 769 \
  --vector-core-count 32 --experimental-split-k32-control \
  --profile-metric pipe --profile-calls 3 \
  --profile-dir ../.runtime_cache/paddle_gqa_split_k32_increfa_aiv/profiles/k32_kv1024_v768_pipe \
  --output ../.runtime_cache/paddle_gqa_split_k32_increfa_aiv/results/k32_kv1024_v768_pipe.json
```

Remote raw artifacts remain under:

```text
.runtime_cache/paddle_gqa_split_k32_increfa_aiv/builds/20260809T185123Z/
.runtime_cache/paddle_gqa_split_k32_increfa_aiv/results/
.runtime_cache/paddle_gqa_split_k32_increfa_aiv/profiles/
.runtime_cache/paddle_gqa_split_k32_increfa_aiv/real_b1_8a1041f/
```

The compact machine-readable values and exact remote file names are in
[summary.json](summary.json). Large profiler exports are not committed.

## Decision

The core-splitting idea works. It nearly halves the vector lane without
doubling KV transfer and improves the full B1 decoder in this controlled
sequence. Keep it as an independent experiment until additional same-device
16/32 sequences confirm the effect. The next kernel target is the split-K
partial-softmax/output reduction and its synchronization, not more KV copying.
