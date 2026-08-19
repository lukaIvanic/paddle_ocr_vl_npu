# Experiment 16: GLM-5.2 W4A8C8 layer lab

This experiment owns a standalone GLM-5.2 decoder-layer implementation. It
uses vLLM-Ascend source only as an architecture and operator-contract reference;
the runtime model imports only PyTorch, torch-npu, and safetensors.

The first rung is layer 3 because it is the first MoE layer and its configured
DSA top-k indexer is skipped. The owned path implements dense, unabsorbed MLA
attention with a static full K/V cache, W8A8 attention and shared-expert
linears, and W4A8 routed experts. This is a correctness and compileability rung,
not yet the final absorbed-MLA performance path.

Eager and compiled timing use independent empty KV caches. Each lane runs the
same ordinary decode loop for warmup and measurement. Warmup includes cold
first use (and compile/cache load for TorchAir) and is excluded from measured
latency. The normal static result is one graph captured during warmup. The
runner reports the graph count after warmup and fails if another graph appears
inside the measured window.

The standalone harness supplies a fresh activation tensor to every layer call,
as an upstream layer would in a real model. The small activation clone is
included in both eager and compiled timing. Reusing one consumed standalone
activation across calls gives invalid multi-token KV state.

## Verified 910B2 result

Commit `e04f03a` ran on physical Ascend 910B2 NPU 7 with B1, BF16 activations,
KV4096, 10 excluded warmup calls, and 1,000 measured layer calls:

| Lane | Mean layer time | Layer calls/s |
| --- | ---: | ---: |
| Owned eager | 2.906 ms | 344.1 |
| Owned TorchAir static compile | 2.014 ms | 496.6 |

The compiled output and all 1,010 written key/value cache rows matched eager
exactly (`max_abs=0`). Dynamo reported one unique graph after warmup and zero
new graphs during measurement. The full summary is in
`references/layer3_steady1000_910b2_e04f03a.json`.

This single-layer result does not establish vLLM-Ascend reference parity or
whole-model token throughput. It decompresses MLA into a full K/V cache, and
the standalone layer 3 has no upstream shared DSA selection. Stack-level DSA,
absorbed MLA, real upstream activations, and final logits require later rungs.

## Verified layers 0-6 stack

The owned seven-layer stack covers the decoder-layer variants present at the
front of GLM-5.2:

- layers 0-2: W8A8 dense MLPs with full DSA indexers;
- layers 3-5: W4A8 MoE layers reusing the last DSA selection;
- layer 6: W4A8 MoE with a refreshed DSA selection.

The current DSA implementation owns the index projections, interleaved RoPE,
index cache, top-2048 selection, shared-index propagation, and sparse attention.
It uses ordinary PyTorch/torch-npu operations rather than vLLM-Ascend runtime
classes or its fused sparse-attention extension.

Commit `107dca3` ran layers 0-6 on physical Ascend 910B2 NPU 7 with B1,
KV4096, 10 ordinary warmup positions, and 2,200 measured positions. The final
162 positions had more than 2,048 valid cache entries, so DSA pruned tokens
rather than selecting the complete valid context.

| Lane | Mean seven-layer time | Stack calls/s |
| --- | ---: | ---: |
| Owned eager | 23.027 ms | 43.43 |
| Owned TorchAir static compile | 9.115 ms | 109.71 |

Measured HBM allocation was 20.13 GiB for weights, 21.89 GiB after the eager
KV/index caches, and 23.97 GiB after the compiled parity lane. The compiled
run captured one graph during warmup and no new graph during measurement.
Compiled-versus-eager output max/mean absolute differences were 0.015625 and
0.000265. Maximum K/V/index cache differences were 0.007813, 0.001953, and
0.023438.

This establishes fit, eager execution, static compilation, state propagation,
and internal eager/compiled parity. It does not yet establish independent
vLLM-Ascend parity, final model logits, or production performance. The full K/V
cache and manual sparse attention must be replaced by absorbed MLA and the
target fused DSA kernels before extrapolating this timing to the full model.
The full result is in
`references/layers0_6_dsa2200_910b2_107dca3.json`.

## Verified dense-layer TP2

The owned TP implementation shards layers 0-2 as follows:

- Q/latent A projections and all DSA indexer projections are replicated;
- Q-B and KV-B outputs are column-sharded by attention head;
- decompressed K/V caches contain 32 of 64 heads per TP2 rank;
- attention output projection is row-sharded and all-reduced;
- dense gate/up outputs are column-sharded across the intermediate dimension;
- dense down projection is row-sharded and all-reduced.

Each layer therefore has two HCCL all-reduces. The full three-layer graph has
six. Commit `deb120b` ran TP1 and TP2 on Ascend 910B2 with B1, KV4096, 10
ordinary warmup positions, and 2,200 measured positions. TP2 used physical
NPUs 6 and 7; timing uses the slower rank.

| Lane | Mean layers 0-2 time | Stack calls/s | Relative to TP1 |
| --- | ---: | ---: | ---: |
| TP1 raw eager, 200 steps | 10.171 ms | 98.32 | 1.000x |
| TP2 raw eager, 200 steps | 11.508 ms | 86.89 | 0.884x |
| TP1 TorchAir, 2,200 steps | 4.038 ms | 247.67 | 1.000x |
| TP2 TorchAir, 2,200 steps | 2.711 ms | 368.93 | **1.490x** |

TP2 eager regressed 13.1% because B1 communication was exposed between small
kernel calls. Static compilation made the reduced matrix/cache work outweigh
the six all-reduces and improved stack throughput by 49.0%. Per-NPU allocated
HBM in the compiled parity process fell from 2.93 GiB at TP1 to 1.51 GiB at
TP2. The TP2 output max/mean difference from the TP1 eager reference was
0.015625/0.0000747. Maximum local K/V/index-cache differences were
0.006836/0.000198/0.023438. Both ranks captured one graph and captured no new
graph during measurement.

These timings still use the owned manual DSA attention and decompressed K/V
cache. They demonstrate that TP2 is technically correct and beneficial for the
dense block under TorchAir; they are not full-model TPOT projections. The
comparison is saved in
`references/dense_layers0_2_tp1_tp2_910b2_deb120b.json`.

The matching Torch NPU profile explains why the end-to-end speedup is below
2x. In this model, "dense layer" describes the FFN. Each layer still has DSA
attention, including a full top-2048 attention-position indexer. The Q/KV-A
projection and that indexer are replicated on both TP ranks. Norms, RoPE,
masks, and small cache/control kernels are also not halved.

The explicitly sharded major attention kernels fell from 2.043 ms to 0.839 ms
(2.435x), while the explicitly sharded dense gate/up and down matmuls fell from
0.748 ms to 0.332 ms (2.249x). The remaining replicated, fixed-shape, and small
compute kernels barely moved: 1.014 ms to 0.968 ms. TP2 then added 0.147 ms of
communication on the critical rank. Six all-reduces are present; the first
accounted for 0.100 ms, mostly synchronization, while the other five were about
0.0095 ms each. The profiler perturbed timing, so its device-stage ratio
(4.410/2.663 ms, 1.656x) is diagnostic rather than the reported throughput
result. The complete profile accounting is saved in
`references/dense_layers0_2_tp1_tp2_profile_910b2_337074b.json`.

The warmed TP1 memory profile uses 20 ordinary forward calls on the same cache,
followed by three contiguous profiled graph calls. The clean 200-call lane was
3.940 ms per three-layer stack. Resident weights plus the minimum selected-K/V
reads account for about 1.66 GB per stack, whose 1.6 TB/s HBM roof is 1.04 ms.
The current stack therefore realizes about 26% of that minimum-traffic roof.
The compiled graph requested about 3.38 GB of logical global-memory traffic per
stack and sustained 0.87-0.91 TB/s across the complete graph. The roughly 2x
traffic amplification is mainly the manual sparse-attention gather, cast, and
materialization route. Per-kernel memory and L2 results are saved in
`references/dense_layers0_2_tp1_memory_profile_910b2_d9099a6.json`.

## Verified TP1 absorbed-MLA rung

The first vLLM-Ascend-derived optimization kept the owned DSA indexer and
manual sparse softmax, but absorbed the BF16 KV-B projection into per-head
`W_UK_T` and `W_UV` weights. It stores one latent-512 row and one RoPE-64 row
per token and layer instead of decompressed per-head K/V. The decompressed and
manual absorbed implementations now exist only in their named historical
commits and evidence files; the live dense path always uses absorption.

Commit `e9767ad` ran both paths sequentially on physical Ascend 910B2 NPU 7
with B1, KV4096, 10 ordinary warmup positions, and 1,000 measured positions.
Both paths captured one static graph during warmup and no graph during the
measurement window.

| TP1 TorchAir lane | Mean layers 0-2 time | Stack calls/s | Allocated HBM |
| --- | ---: | ---: | ---: |
| Decompressed manual | 4.028 ms | 248.29 | 2.931 GiB |
| Absorbed manual | **2.604 ms** | **384.07** | **1.227 GiB** |

Absorption reduced latency by 35.4%, increased stack throughput by 54.7%
(`1.547x`), and reduced allocated HBM by 1.704 GiB. In raw eager execution,
eight sequential positions had exact output, reconstructed K/V-cache, and
index-cache parity. The compiled absorbed and decompressed paths had identical
comparison errors against the raw reference, so this rung added no observed
compiled-path drift.

A short PipeUtilization profile found that quantized linears now account for
61.9% of recorded kernel time. The remaining manual absorbed attention still
materializes selected latent/RoPE rows and casts them for FP32 scoring, making
the fused sparse-attention operator the next isolated rung. Full evidence is in
`references/dense_layers0_2_tp1_absorbed_mla_910b2_e9767ad.json`.

## Verified contiguous-cache SparseFA rung

The contiguous SparseFA rung replaced only the selected-attention core with
`torch_npu.npu_sparse_flash_attention`. It deliberately does not use paged
attention: query and KV use `BSND`, the compressed KV cache remains one static
KV4096 tensor, `block_table=None`, and the valid KV length is the on-device
`position + 1` tensor. Invalid future entries in the fixed top-2048 result are
converted to an on-device `-1` tail. No Python sequence-length list or graph-task
parameter update is part of decode.

Commit `07ce6df` passed raw-eager parity for eight sequential positions on
physical Ascend 910B2 NPU 7. The output maximum absolute difference versus the
manual reference was `0.00390625`; reconstructed latent/RoPE cache differences
were zero. Static TorchAir captured one graph during ordinary warmup and no new
graph during any 1,000-call measured interval.

Three interleaved 1,000-call runs per lane gave:

| TP1 TorchAir lane | Three stack times | Median | Median calls/s |
| --- | --- | ---: | ---: |
| Absorbed manual | 2.515, 2.546, 2.578 ms | 2.546 ms | 392.72 |
| Contiguous SparseFA | 2.413, 2.467, 2.463 ms | **2.463 ms** | **406.08** |

SparseFA reduced median stack latency by 3.29% and increased median throughput
by `1.034x`. Allocated HBM increased by 0.064 GiB. In the matched pipe profile,
recorded kernel time fell from 8.913 ms to 8.142 ms, or 8.65%. The former manual
gather/cast/score/softmax/aggregation chain averaged 94.87 us per layer; the
SparseFlashAttention kernel averaged 20.47 us. Quantized linears now account for
67.9% of recorded kernel time. Full evidence is in
`references/dense_layers0_2_tp1_sparsefa_bsnd_910b2_07ce6df.json`.

## Verified contiguous LightningIndexer rung

The LightningIndexer rung replaced manual KV4096 scoring, head weighting and
reduction, causal-range construction, and TopK with
`torch_npu.npu_lightning_indexer`. Like the SparseFA rung, it uses ordinary
contiguous `BSND`: `block_table=None`, an on-device `position + 1` length, and
no Python sequence-length list or graph-task parameter update.

This rung also corrected an older semantic error. The authoritative GLM-5.2
indexer applies ReLU to each head's Q/K score before weighted head reduction.
The historical owned path omitted ReLU. The corrected manual ReLU control and
the incorrect legacy implementation remain reproducible at commit `10a3e91`,
but both have been removed from the live source.
On physical 910B2 NPU 7, a real 32-head/top-2048 operator probe matched the
corrected manual TopK in exact order, and eight sequential raw-eager model
positions had exact output and cache parity.

Three 1,000-call TP1 runs per lane gave:

| TP1 TorchAir indexer | Three stack times | Median | Median calls/s |
| --- | --- | ---: | ---: |
| Corrected manual ReLU | 2.435, 2.388, 2.437 ms | 2.435 ms | 410.76 |
| Contiguous LightningIndexer | 2.306, 2.327, 2.302 ms | **2.306 ms** | **433.63** |

LightningIndexer reduced median stack latency by 5.28% and increased throughput
by `1.056x`. The matched profile measured 50.7 us/layer for the former manual
indexer chain and 16.0 us/layer for LightningIndexer, a direct saving of about
104 us per three-layer stack. Complete recorded kernel time fell by 7.7%.

A single TP2 pair on physical NPUs 0 and 1 measured 1.977 ms for corrected
manual and 1.920 ms for LightningIndexer: 2.86% lower latency and `1.029x`
throughput. Both lanes passed TP1-reference parity, captured one static graph,
and captured no new graph during measurement. Full evidence is in
`references/dense_layers0_2_lightning_indexer_910b2_10a3e91.json`.

## Current dense path

The layers 0-2 TP benchmark now exposes one runtime path only:

- absorbed KV-B weights and compressed latent/RoPE KV caches;
- contiguous-BSND `npu_sparse_flash_attention`;
- contiguous-BSND `npu_lightning_indexer`;
- block-layout `npu_interleave_rope` for the attention and indexer Q/K paths;
- one static TorchAir graph after ordinary inference warmup.

There are no attention or indexer selection flags. Historical JSON files retain
the exact commands and results for the removed baselines; replay those commands
from their recorded commit, not from the current head.

## Verified InterleaveRope rung

The final replicated-kernel sweep compared the historical manual interleaved
RoPE chain, `npu_interleave_rope`, and
`npu_kv_rmsnorm_rope_cache_v2`. All lanes used B1, KV4096, FRACTAL_NZ W8A8
weights, ordinary inference warmup, and one static TorchAir graph on physical
Ascend 910B2 NPU 7.

| TP1 TorchAir lane | Two 1,000-call stack times | Mean |
| --- | --- | ---: |
| Manual interleaved RoPE | 1.828, 1.845 ms | 1.836 ms |
| `npu_interleave_rope` | 1.701, 1.704 ms | **1.703 ms** |
| `npu_kv_rmsnorm_rope_cache_v2` | 1.725, 1.713 ms | 1.719 ms |

Plain `npu_interleave_rope` reduced latency by 7.27% and increased throughput
by `1.078x` relative to the manual path. It was also 0.96% faster than the
fused KV RMSNorm/RoPE/cache-write lane. The fused KV operator was correct, but
its extra fusion did not reduce full-graph latency further.

The matched five-stack profile recorded 1,408.2 us of kernel time per manual
stack and 1,276.8 us per InterleaveRope stack. InterleaveRope removed most of
the manual Gather, buffer-fusion, Mul, Pack, and Neg chain. Its block-layout
RoPE result is a pure permutation of the historical interleaved layout. The
runtime stores that block layout directly; reference validation converts it
back only outside the timed graph.

The broader operator sweep also rejected three tempting alternatives:

- `npu_rotary_mul` matched eagerly but produced wrong compiled cache state;
- `npu_qkv_rms_norm_rope_cache` uses a standard-MHA fused-QKV/cache contract
  that does not map to GLM-5.2 compressed MLA and its split 192+64 Q heads;
- `npu_mla_prolog_v3` maps conceptually, but the installed kernel requires a
  Q-LoRA width of 1,536 and rejected GLM-5.2's width of 2,048.

The live benchmark therefore has no RoPE/cache-fusion selection flag. Full
measurements, profile rows, parity details, and rejection reasons are in
`references/dense_layers0_2_rope_fusion_910b2_5194a83.json`.

The cleaned single-path runtime at commit `b1454c7` then passed the same
eight-position raw-eager reference with exact output. Its independent static
TorchAir run used 20 excluded ordinary warmup calls and 1,000 measured calls.
It measured 1.731 ms per three-layer stack, or 577.6 stack calls/s. Dynamo
reported one graph after warmup and the same one graph after measurement.

The optimized-only TP1/TP2 validation and exact five-call PipeUtilization
profile are saved in
`references/dense_layers0_2_optimized_cleanup_profile_910b2_ff44430.json`.

## Verified TP1 FRACTAL_NZ W8A8-weight rung

The FRACTAL_NZ experiment changes only the storage format of all 18 W8A8
weights in dense layers 0-2. It enables torch-npu internal formats before the
first NPU allocation, then converts each already-logical `[K,N]` INT8 weight
from format code 2 (ND) to format code 29 (FRACTAL_NZ) once before graph
compilation. The six affected projection families in each layer are fused
Q/KV-A, Q-B, attention O, dense gate/up, dense down, and indexer WQ-B. BF16
linears are unchanged.

On physical Ascend 910B2 NPU 7, B1/KV4096, three fresh-process 1,000-call runs
per lane gave:

| TP1 TorchAir weight lane | Three stack times | Median | Median calls/s |
| --- | --- | ---: | ---: |
| Native ND, internal formats disabled | 2.328, 2.206, 2.230 ms | 2.230 ms | 448.49 |
| Native ND, internal formats enabled | 2.302, 2.189, 2.222 ms | 2.222 ms | 450.12 |
| FRACTAL_NZ, internal formats enabled | 1.833, 1.700, 1.725 ms | **1.725 ms** | **579.57** |

The native controls differ by only 0.36%. Against the matched internal-format
control, FRACTAL_NZ reduced median stack latency by 22.34% and increased stack
throughput by 28.76%. Raw eager output and all three cache families matched the
saved reference exactly for eight positions. Compiled parity was unchanged
from native. The first process captured one static graph; warm disk-cache
processes loaded it without capturing a new graph, and no process created a
graph during its measured window.

Matched five-call profiles attribute the result to the W8A8 matmuls. Their
combined time fell from 1.343 to 0.826 ms per stack, while complete recorded
kernel time fell from 1.919 to 1.412 ms. Every projection family improved:

| W8A8 projection family | Native | FRACTAL_NZ | Reduction |
| --- | ---: | ---: | ---: |
| Fused Q/KV-A | 203.43 us | 54.87 us | 73.03% |
| Dense gate/up | 440.87 us | 290.87 us | 34.02% |
| Attention O | 319.89 us | 209.29 us | 34.57% |
| Dense down | 215.32 us | 152.89 us | 28.99% |
| Q-B | 116.00 us | 83.79 us | 27.77% |
| Indexer WQ-B | 47.10 us | 34.55 us | 26.63% |

FRACTAL_NZ adds about 6.3 seconds to one-time three-layer weight setup in these
fresh processes; that setup is outside decode timing. This result establishes
the TP1 dense layers 0-2 point only. TP2 and the MoE linears need separate
validation. Full evidence and exact commands are in
`references/dense_layers0_2_fractal_nz_910b2_1ba4a0e.json`.

Run on one Ascend 910B2 after sourcing the NPU environment:

```bash
python3 16_glm52_w4a8c8_layer_lab/run_layer3.py \
  --model-dir /workspace/models/GLM-5.2-w4a8c8 \
  --cache-length 4096 \
  --warmup-steps 10 \
  --decode-steps 1000 \
  --summary-out /tmp/glm52_layer3.json
```
