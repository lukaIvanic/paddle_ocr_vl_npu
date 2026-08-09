# B1 GQA AIV TorchAir evidence

This bundle retains the small machine-readable summaries for the independent
`PaddleGqaIncreFlashAttentionAiv` operator and its Experiment 09 B1 integration.
The implementation and integration base is commit `1d16f33`. The captured JSON
uses the original preset name `combined_apply_gqa_aiv_b1_c16`; the final source
renames the same 16-core configuration to `combined_apply_gqa_aiv_b1`.

## Environment and package

- Date: 2026-08-09
- Device: physical Ascend 910B2
- CANN: 9.0.0
- PyTorch: 2.10.0+cpu
- torch-npu: 2.10.0
- Upstream `ops-transformer`: `afe72144f9f2ac8441929035795db88a111b30c5`
- Package SHA256: `a59f8ee5848aea6604262677ebd24e5dc0c2b41b9fdfc8e66c9bb6884548d6c6`
- Kernel object SHA256: `ecbeca46a7cf39c01b6f9ecf1dacc451d3d475395c0c8db927dea00b8822aa9a`
- Kernel JSON SHA256: `2e47b39b25a2555fcc0265e8bac155519be249c8052359f512021632cca31cad`

The AscendC two-key object compiled in about 9 seconds. A clean full package
build took 4 minutes 53 seconds because the wrapper also rebuilt host
dependencies and package scaffolding. Isolated TorchAir first calls took 4.7 to
17.0 seconds. These are different build phases.

## Operator correctness and core-count sweep

The JSON under `operator/` comes from one-shape-per-process TorchAir runs. Every
row passed FP16-tolerance parity with stock IncreFA and an independent CPU FP32
matmul-softmax-matmul GQA reference.

| Shape | Requested cores | Custom mean (us) | Stock mean (us) |
| --- | ---: | ---: | ---: |
| KV128 | 16 | 253.07 | 242.83 |
| KV512 | 16 | 258.61 | 272.66 |
| KV1536, first 512 valid | 16 | 263.59 | 235.13 |
| KV1536, first 512 valid | 32 | 246.52 | 257.17 |
| KV1536, first 512 valid | 48 | 253.50 | 241.11 |
| KV2048 | 16 | 246.32 | 255.02 |
| KV2048 | 32 | 264.45 | 243.28 |
| KV2048 | 48 | 248.93 | 236.80 |

The KV1536 profile proves that requests for 16, 32, and 48 cores all resolve to
the same 16-block non-split launch. Their timing differences are process
variance, not a useful core sweep. KV2048 activates real split work for larger
requests; the 16-core custom row was still fastest. Requests below 16 are
rejected because the recovered pipeline needs one work item for each of 16
query heads.

The profile summary in `operator/profile_eager_kv1536_c32_v11.json` points to
the raw remote capture. Its custom tasks have `Block Num=16`, zero AIC time,
cycles, MAC time, and cube utilization, with nonzero AIV time and cycles. The
installed object also has no `_mix_aic` function and its metadata uses
`taskRation: "0:1"`. The exported `Task Type=MIX_AIC` label is therefore not a
reliable core-execution signal.

## Controlled B1 full-decoder result

The decisive throughput run was one serial ABBA sequence on physical NPU6. It
used the real full decoder and LM head, B1, KV1024, position 768, 20 warmups, and
200 measured steps per lane.

```sh
/usr/local/python3.12.13/bin/python3 scripts/text_decode_lab.py \
  --mode profile --backend torchair \
  --batch-size 1 --active-slots 1 \
  --cache-length 1024 --profile-position 768 \
  --decode-optimization combined_apply_gqa_aiv_b1 \
  --warmup 20 --repeats 200 --allow-compile
```

The stock lanes substituted `--decode-optimization combined_apply`. The four
JSON files are under `fixed_b1_k1024/` with `a_` through `d_` prefixes.

| Lane | Mean step (ms) | Median (ms) | p95 (ms) | Throughput (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| A stock | 1.2551 | 1.2539 | 1.2575 | 796.74 |
| B custom | 1.2379 | 1.2363 | 1.2404 | 807.84 |
| C custom | 1.2278 | 1.2263 | 1.2297 | 814.43 |
| D stock | 1.2597 | 1.2580 | 1.2618 | 793.87 |
| Stock mean | 1.2574 | - | - | 795.31 |
| Custom mean | 1.2329 | - | - | 811.14 |

The same-device custom mean is 1.99% higher in throughput and 1.95% lower in
latency. A separate physical NPU3 pair had the opposite sign. Keep this path
opt-in until the small improvement repeats across devices and runs.

The two non-throughput profiler manifests are also under `fixed_b1_k1024/`.
Their raw remote exports showed 54 stock attention tasks at 975.88 us total
(18.071 us average) and 54 custom tasks at 879.28 us total (16.282 us average)
across three steps. The custom attention saved 96.6 us in that capture, or 32.2
us per full step. Profiler wall time is not a throughput measurement.

## Real generation gate

The files under `real_b1/` use a real OmniDocBench table crop, cache length
1536, an input length of 1021, and up to 512 new tokens. The custom TorchAir run
matched the stock path for all 374 generated tokens, decoded text, EOS stop
reason, and decode-call count. Use this as the multi-step correctness gate, not
as the stable latency comparison; whole-request scheduler rates varied strongly
between fresh processes.

## AICPU research result

An independent direct `.aicpu` prototype executed the entire B1 16Q/2KV/D128
attention calculation on physical 910B2 NPU6 with FP16 storage and FP32
accumulation. It matched stock and the CPU reference, but was not competitive:

| KV | AICPU | Stock IncreFA | Slowdown |
| ---: | ---: | ---: | ---: |
| 128 | 1.541 ms | 51.486 us | 29.9x |
| 512 | 6.889 ms | 51.915 us | 132.7x |
| 2048 | 28.535 ms | 51.596 us | 553.1x |

At KV2048, the AI CPU task averaged 28.761 ms. Host `LaunchKernelV2` averaged
7.283 us, so device execution dominated. The measured algorithmic rate was
0.588 GFLOP/s and the minimum tensor-byte rate was 0.074 GB/s. The sampled
run-level HBM counters were 26.983 MB/s read and 14.768 MB/s write. Do not use
AI CPU for QK, softmax, or AV. Its plausible role is small branch-heavy
metadata or scheduler work that removes a host synchronization and can overlap.

The direct research artifacts remain on the Blue Zone host at:

```text
/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/aicpu_attention_research/direct/
```

No generated AICPU binary or raw profiler trace is committed.
