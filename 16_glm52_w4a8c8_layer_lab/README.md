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

The first vLLM-Ascend-derived optimization keeps the owned DSA indexer and
manual sparse softmax, but absorbs the BF16 KV-B projection into per-head
`W_UK_T` and `W_UV` weights. It stores one latent-512 row and one RoPE-64 row
per token and layer instead of decompressed per-head K/V. The old
`decompressed_manual` path remains available as an explicit baseline;
`absorbed_manual` selects the new path.

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

The `absorbed_sparse_flash` lane replaces only the selected-attention core with
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

Run on one Ascend 910B2 after sourcing the NPU environment:

```bash
python3 16_glm52_w4a8c8_layer_lab/run_layer3.py \
  --model-dir /workspace/models/GLM-5.2-w4a8c8 \
  --cache-length 4096 \
  --warmup-steps 10 \
  --decode-steps 1000 \
  --summary-out /tmp/glm52_layer3.json
```
