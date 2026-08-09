# Independent B1 GQA IncreFA AIV operator

This directory builds the independently named AIV implementation used by the
Experiment 09 B1 TorchAir decode lane. It does not replace or overload stock
`IncreFlashAttention`.

## Status

Validated on 2026-08-09 on a physical Ascend 910B2 with CANN 9.0.0,
torch 2.10.0, and torch-npu 2.10.0.

- Direct eager and TorchAir both run the separate operator.
- KV128, KV512, masked KV1536, and KV2048 pass stock FP16-tolerance parity and
  an independent CPU FP32 GQA reference.
- Real 374-token B1 OCR generation is token-, text-, and EOS-exact against the
  stock path.
- Static ELF and metadata gates reject a cube function.
- The runtime profile reports zero AIC time, cycles, MAC time, and cube
  utilization, with nonzero AIV time and cycles.
- A same-device ABBA B1/KV1024 full-decoder benchmark improved mean throughput
  from 795.31 to 811.14 tok/s, or 1.99%. The gain is small, so this operator
  remains an explicit experimental preset rather than the production default.

## Narrow contract

| Field | Value |
| --- | --- |
| SoC | Ascend 910B2 |
| Batch | 1 |
| Q | FP16 BNSD `[1, 16, 1, 128]` |
| K/V | FP16 BNSD `[1, 2, KV, 128]` |
| Mask | bool `[1, 1, 1, KV]` |
| Output | FP16, same shape as Q |
| Actual lengths and PSE | none |
| Attention | GQA, 16 query heads and 2 KV heads |
| Precision | `inner_precise=1` |
| Requested AIV cores | 16 in the real B1 preset |

The all-vector pipeline assigns one query-head work item to each AIV core. A
request below 16 is rejected because it deadlocks the recovered pipeline. At
the production KV1536 shape, requests of 16, 32, and 48 all reduce to an actual
16-block non-split launch. At KV2048, 32 and 48 enable extra FlashDecode splits,
but the clean TorchAir sweep did not beat 16 cores.

## Identity ledger

```text
PyTorch graph: paddleocr_vl::gqa_incre_flash_attention_aiv
PyTorch eager: paddleocr_vl_npu::paddle_gqa_incre_flash_attention_aiv_eager
GE/CANN:       PaddleGqaIncreFlashAttentionAiv
public ACLNN:  aclnnPaddleGqaIncreFlashAttentionAiv
kernel:        paddle_gqa_incre_flash_attention_aiv
vendor:        paddle_gqa_increfa_aiv
```

The real decoder selects it through the B1-only
`combined_apply_gqa_aiv_b1` preset. The preset is TorchAir-only and fails
closed for another batch size or backend.

## Build

The builder pins upstream `ops-transformer` commit
`afe72144f9f2ac8441929035795db88a111b30c5`, validates critical source hashes,
creates a detached build worktree, applies the six patches, compiles both
required tiling keys, and installs a private vendor package.

The production build applies six patches. Setting
`PADDLE_GQA_EXPERIMENT_VARIANT=grouped_serial_control` adds patch 0007, uses a
separate cache namespace and vendor, disables FlashDecode for that package, and
launches one unsplit AIV block per KV group. Never source the production and
grouped packages in the same process.

Setting `PADDLE_GQA_EXPERIMENT_VARIANT=grouped_half_control` applies patches
0007 and 0008 in another private vendor. It emits two four-query-head work
items per KV group, for four actual AIV blocks. This is a retained topology
control, not a production preset.

```sh
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup
RUN_ID=gqa_aiv_release \
PADDLE_GQA_BUILD_SOURCE_ROOT="$PWD/.runtime_cache/paddle_gqa_increfa_aiv/sources/gqa_aiv_release" \
bash 09_persistent_page_engine/custom_ops/paddle_gqa_increfa_aiv/build.sh
```

Source the printed `PADDLE_GQA_INCREFA_AIV_SET_ENV` path before every eager or
TorchAir run.

The retained v11 package has these hashes:

```text
package a59f8ee5848aea6604262677ebd24e5dc0b41b9fdfc8e66c9bb6884548d6c6
object  ecbeca46a7cf39c01b6f9ecf1dacc451d3d475395c0c8db927dea00b8822aa9a
json    2e47b39b25a2555fcc0265e8bac155519be249c8052359f512021632cca31cad
```

The AscendC compiler generated the two-key object in about 9 seconds. The full
fresh package command took about 4 minutes 53 seconds because the upstream
wrapper rebuilt Abseil, protobuf, libprotoc, ONNX plugins, and package
scaffolding. Do not report that interval as TorchAir compilation. Isolated
TorchAir first calls were 4.7 to 17.0 seconds; full B1 graph setup was 17 to 42
seconds in the retained runs.

## Validation

Run direct eager first, then TorchAir:

```sh
cd /workspace/repos/paddle_ocr_vl_npu/09_persistent_page_engine

/usr/local/python3.12.13/bin/python3 \
  scripts/probes/compare_paddle_gqa_increfa_aiv.py \
  --backend eager --kv-length 2048 --vector-core-count 16 \
  --warmup 20 --blocks 7 --repeats-per-block 200 \
  --output ../.runtime_cache/paddle_gqa_increfa_aiv/results/eager_kv2048.json

/usr/local/python3.12.13/bin/python3 \
  scripts/probes/compare_paddle_gqa_increfa_aiv.py \
  --backend torchair --kv-length 1536 --valid-kv-length 512 \
  --vector-core-count 16 \
  --cache-root ../.runtime_cache/paddle_gqa_increfa_aiv/torchair_kv1536_c16 \
  --warmup 20 --blocks 7 --repeats-per-block 200 \
  --output ../.runtime_cache/paddle_gqa_increfa_aiv/results/torchair_kv1536_c16.json
```

The probe runs the custom operator first to prevent allocator reuse from hiding
unwritten output. It requires both stock/custom tolerance parity and an
independent CPU FP32 matmul-softmax-matmul GQA reference.

## Measured results

All operator rows below passed the independent reference.

| TorchAir shape | Requested cores | Custom mean (us) | Stock mean (us) |
| --- | ---: | ---: | ---: |
| KV128 | 16 | 253.07 | 242.83 |
| KV512 | 16 | 258.61 | 272.66 |
| KV2048 | 16 | 246.32 | 255.02 |
| KV2048 | 32 | 264.45 | 243.28 |
| KV2048 | 48 | 248.93 | 236.80 |

The stock values vary between processes. Use the custom rows for the core
sweep and the full-decoder benchmark for the production decision.

One 200-step pair on physical NPU3 measured 776.90 tok/s stock and 740.12 tok/s
custom. A reverse-order pair on physical NPU6 measured 793.38 tok/s stock and
814.90 tok/s custom. Those two-device pairs disagree, so neither is the primary
result.

The decisive ABBA sequence ran `stock -> custom -> custom -> stock` after one
physical NPU6 selection. Every lane used 20 warmups and 200 measured B1/KV1024
full production steps:

| Lane | Mean step (ms) | Throughput (tok/s) |
| --- | ---: | ---: |
| Stock A | 1.2551 | 796.74 |
| Custom B | 1.2379 | 807.84 |
| Custom C | 1.2278 | 814.43 |
| Stock D | 1.2597 | 793.87 |
| Stock mean | 1.2574 | 795.31 |
| Custom mean | 1.2329 | 811.14 |

The custom mean is 1.99% faster in throughput and 1.95% lower in latency on
this same-device run. Preserve the physical-device label: the cross-device
pair disagreement shows that a two-percent effect is not portable evidence by
itself.

The matched profiler showed why kernel-only timing is insufficient. Across
three full steps, the 54 custom attention tasks averaged 16.282 us versus
18.071 us for stock and saved 96.6 us in total. That kernel saving is consistent
with the small same-device ABBA gain. Graph-level cadence remains large enough
to dilute the kernel improvement and make whole-process scheduler timing noisy.

## Grouped-core topology sweep

We tested the proposal to give one AIV block to each of the two KV groups. The
separate `grouped_serial_control` package keeps the supported resource attribute
at 16, but its host tiler emits `Block Num=2`. Each block runs its eight query
heads serially. FlashDecode is disabled only in this experimental package so
KV2048 preserves the same two-block topology.

We then tested the requested doubled-core control. The separate
`grouped_half_control` package emits two work items per KV group. Its profile
shows `Block Num=4`, and each block runs one contiguous four-head slice. It is
still the existing per-head algorithm: it does not share one K/V load across
the four heads.

The control passed stock tolerance and the independent FP32 reference at
KV128, KV512, KV1024, KV1536, and KV2048. The KV1024 pipe profile measured:

| Package | Blocks | Task | Vector | Scalar | MTE2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current query-head parallel | 16 | 21.78 us | 11.40 us | 7.98 us | 4.74 us |
| Half-group control | 4 | 54.83 us | 45.45 us | 18.10 us | 17.93 us |
| Grouped serial control | 2 | 99.96 us | 90.83 us | 31.06 us | 34.78 us |

All rows had zero AIC time and zero cube utilization. Doubling from two to four
blocks reduced task time by 45.14%, but four blocks remained 2.52 times slower
than the current 16-block kernel.

These controls deliberately preserve the per-head K/V load functions; they do
not claim that K/V was loaded once. The matched MemoryAccess profile
measured 8,215 KiB GM-to-UB for grouped and 8,236 KiB for current-16, versus
1,029 KiB of unique direct input. The four-block control measured 8,218 KiB.
All three therefore still issue about eight times the unique bytes.

That profile gives the decision bound for a copy-only rewrite. Pipeline times
overlap and must not be summed. With the current grouped vector algorithm, even
free K/V copies cannot reduce a 99.96 us task below its 90.83 us vector lane.
Across the 18 decoder layers, subtracting the entire remaining 9.13 us per task
from the measured grouped full step gives this optimistic result:

| B1/KV1024 TorchAir package | Mean step | Throughput |
| --- | ---: | ---: |
| Current 16-block | 1.3636 ms | 733.35 tok/s |
| Half-group four-block | 1.8382 ms | 544.00 tok/s |
| Grouped two-block | 2.6582 ms | 376.20 tok/s |
| Two-block copy-only ideal bound | at least 2.4939 ms | at most 400.98 tok/s |

Four blocks improved throughput by 44.60% over two blocks, but remained 25.82%
below current. The matched eager call boundary was almost flat at 173.30,
171.55, and 170.32 us for 2, 4, and 16 blocks because ACLNN/eager overhead hid
the task-level difference. The real compiled decoder exposed the accumulated
18-layer cost.

We therefore keep 16-way query-head parallelism. A genuinely different
batched-head vector algorithm is a separate research question; this sweep does
not bound that different algorithm.

Retained evidence: [two-AIV-block GQA experiment](../../../tmp/09_persistent_page_engine/gqa_grouped_two_block_994dc8f/README.md)
and [four-AIV-block GQA experiment](../../../tmp/09_persistent_page_engine/gqa_grouped_four_block_ca152b5/README.md).

## AIV-only proof

The package reports two `MIX` kernels with `taskRation: "0:1"` and no
`_mix_aic` ELF function. The retained masked-KV1536 profile is under:

```text
.runtime_cache/paddle_gqa_increfa_aiv/profiles/eager_kv1536_c32_v11/
```

For every custom task in that capture:

```text
Block Num: 16
aicore_time(us): 0
aic_total_cycles: 0
aic_mac_time(us): 0
aiv_time(us): > 0
aiv_total_cycles: > 0
cube_utilization(%): 0.000
```

`Task Type` alone is not proof. This CANN export labels the task `MIX_AIC`
despite the zero-AIC counters. Use the counters, ELF symbols, and package
metadata together.

## Current limits and next optimization question

- B1, FP16, BNSD, 16Q/2KV, D128 only.
- Mask required; PSE and actual-length tensors are not supported.
- Fewer than 16 AIV cores are unsafe for the recovered one-work-item-per-core
  pipeline and are rejected.
- The custom path remains opt-in because the measured full-decoder gain is only
  about two percent and was sensitive to physical-device/run conditions.
- The next useful experiment is to explain or remove the per-layer graph-level
  cadence around the hard-sync `MIX_AIV_1_0` envelope. More vector cores do not
  address that bottleneck at KV1024/KV1536.

See [the repository custom-operator handbook](../ASCEND_CUSTOM_OPERATOR_HANDBOOK.md)
for the full build, eager, TorchAir, profiling, and evidence workflow.
