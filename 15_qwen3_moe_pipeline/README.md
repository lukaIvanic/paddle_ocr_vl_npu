# Experiment 15: Qwen3-30B-A3B PP2, TP2, and stage replay

This experiment establishes and optimizes a correctness-first custom
Qwen3-30B-A3B decode lane.

The full model runs as a two-stage sequential pipeline:

- stage 0 on one 910B2: token embedding and layers 0 through 23;
- stage 1 on a second 910B2: layers 24 through 47, final RMSNorm, LM head, and
  greedy token selection.

The official BF16 checkpoint contains 61,064,245,248 bytes of tensors. A
24-layer split therefore fits comfortably on two 64 GB 910B2 devices. The
initial parity lane uses one host process and a direct NPU-to-NPU boundary copy.
The compiled throughput lane uses one process per NPU and HCCL send/receive.
B1 does not provide useful inter-stage pipeline overlap, so stage latencies
remain additive.

## MoE parity control

The selected-expert control follows the Transformers Qwen3-MoE contract:

1. project the token to 128 router logits;
2. softmax in FP32;
3. select top 8 experts;
4. renormalize the selected probabilities;
5. gather the selected packed gate/up and down weights;
6. run two batched matrix multiplies;
7. apply the weighted reduction.

This is deliberately separate from future grouped-matmul/fused-MoE paths.

## Verified 910B2 result

Verified on 2026-08-17 with the real BF16 `Qwen3-30B-A3B` checkpoint at commit
`14bd67f`:

- the custom two-NPU pipeline matched all eight greedy token IDs from an
  independent vLLM-Ascend TP2 run;
- both paths generated
  ` Paris. The capital of the United Kingdom` with token IDs
  `[12095, 13, 576, 6722, 315, 279, 3639, 15072]`;
- the uncompiled custom pipeline averaged 89.72 ms per generated token after
  model load, or 11.15 tokens/s;
- the captured second half reproduced every token, every top-10 token set, and
  all 24 layers of selected expert IDs;
- replayed logits were bit-identical to the full custom pipeline, with maximum
  absolute difference 0.0;
- after its first-use step, the uncompiled second-half replay averaged 34.63 ms
  per token, or 28.87 tokens/s.

The speed numbers are raw-eager development measurements. They are not a
compiled serving result. The checked JSON artifacts are in `references/`.

## Verified static TorchAir result

Verified on 2026-08-17 on two Ascend 910B2 devices:

- both 24-layer stages compile with `fullgraph=True`, `dynamic=False`, and
  `ge_cache=True`;
- each process owns one fixed NPU context and one static graph;
- no additional graph appeared after prompt setup on either rank;
- the independently replayed second half improved from 21.12 to 153.43
  tokens/s in the clean cold-cache run, a 7.27x speedup;
- the complete two-process HCCL pipeline sustained 86.07 tokens/s over 64
  measured tokens at the capture's KV256 capacity, or 11.62 ms TPOT;
- the first eight greedy tokens still matched the captured eager pipeline and
  vLLM reference exactly;
- both ranks loaded the second run from the persistent disk graph cache.

The complete compiled result is 7.72x faster than the original 11.15-token/s
raw-eager pipeline. Compiled BF16 logits are not bit-identical to eager logits:
the observed eight-step maximum absolute difference was 0.8125, and top-10
ordering changed on seven steps. The greedy token remained identical on all
eight checked steps. See [NPU_MOE_OPERATOR_NOTES.md](NPU_MOE_OPERATOR_NOTES.md)
for the next operator optimization lane and its constraints.

## Verified PP2 versus TP2 at B1/KV4096

The complete tensor-parallel path shards every layer across both NPUs:

- 16 query heads and 2 KV heads per rank;
- 384 intermediate dimensions from every expert per rank;
- row-parallel attention output and expert down projections;
- vocabulary-parallel token embedding and LM head;
- replicated FP32 router softmax and top-8 selection.

The static graph contains one embedding all-reduce and two all-reduces per
layer, for 97 in-graph collectives per token. The LM head exchanges only the
local maximum value and global token ID outside the graph during the measured
greedy loop.

Verified on two Ascend 910B2 devices with BF16 weights, B1, KV4096, two warmup
tokens, and 200 measured tokens:

| mode | tokens/s | mean TPOT |
| --- | ---: | ---: |
| PP2, two serial 24-layer stages | 76.87 | 13.01 ms |
| TP2, all 48 layers sharded | 94.74 | 10.55 ms |

TP2 was 23.25% faster and reduced TPOT by 18.86%. Both paths matched all eight
captured greedy token IDs, used one static graph per rank, and produced no
recompilations after setup. TP2 keeps both NPUs active while each layer reads
and computes its weight shard. PP2 executes its two model halves serially for
B1, so it cannot overlap the two stages.

TP arithmetic is not bit-identical to the unsharded pipeline. On the first
captured stage-2 token, 190 of 192 selected-expert memberships matched. Seven
layers selected the same eight experts in a different score order. Layers 38
and 39 each changed only the eighth expert at the top-8 cutoff. Both TP ranks
made identical routing decisions, and the final greedy token remained exact.

The exact evidence is in:

- `references/qwen3_moe_full_tp2_k4096.json`;
- `references/qwen3_moe_pp2_k4096.json`;
- `references/qwen3_moe_stage2_tp2_router_membership.json`.

## Verified single-NPU MoE optimization ladder

MoE kernel changes are timed on the complete second half of the model, not on
one layer. The representative gate is layers 24 through 47 plus final RMSNorm
and LM head, B1, KV4096, BF16, two warmup tokens, 200 measured tokens, and one
static TorchAir graph with `fullgraph=True` and `dynamic=False`.

| expert path | tokens/s | mean TPOT | change from selected BMM |
| --- | ---: | ---: | ---: |
| selected expert-weight gather and BMM | 146.50 | 6.826 ms | baseline |
| persistent BF16 grouped matmul | 161.48 | 6.193 ms | +10.22% |
| grouped matmul plus fused output finalization | 178.43 | 5.604 ms | +21.79% |
| InitRoutingV2 counts plus fused finalization | 195.33 | 5.120 ms | +33.33% |
| compile-compatible fused gating, InitRoutingV2, and fused finalization | 196.83 | 5.080 ms | +34.36% |

All rows use all 24 layers and include the LM head. Every measured path used
one static graph, had no recompilation after capture, and matched all eight
captured greedy token IDs. The first run of the final path used a cold graph
cache; the 200-token timing excludes that first compile/load call.

The large gains came from three changes:

1. Keep all 128 expert weights in the persistent `[expert, K, N]` layout and
   let BF16 `npu_grouped_matmul` read only the routed experts. This removes the
   complete expert-weight gathers.
2. Replace output reorder, weighting, and reduction with
   `npu_moe_finalize_routing`.
3. Use `npu_moe_init_routing_v2` to produce the expert group counts directly.
   This removes the separate `npu_moe_compute_expert_tokens` call.

`npu_moe_gating_top_k_softmax_v2(..., renorm=1)` is exposed by the installed
PyTorch API but does not compile in this TorchAir build: its GE converter
rejects the `renorm` attribute. The final path therefore uses the older
compile-compatible fused gating operator and keeps the top-k renormalization
explicit. That last change contributed only 0.77% over the preceding full-stage
run.

The corresponding JSON files are under `references/` with
`bmm_l24_k4096`, `gmm_manual_l24_k4096`, `gmm_finalize_l24_k4096`,
`gmm_v2_finalize_l24_k4096`, and `gmm_v2_gating_finalize_l24_k4096` in their
names.

## Development replay package

The full pipeline processes the prompt token-by-token, avoiding a separate
prefill implementation. Immediately before processing the final prompt token,
it snapshots only the valid stage-2 KV prefix. For every captured generation
step it stores:

- the exact input token ID and cache position;
- the layer-24 boundary hidden state;
- the expected full logits, top-10 logits, and next-token ID;
- selected expert IDs and routing weights for every stage-2 layer.

`replay_stage2.py` loads only layers 24 through 47 on one NPU, restores the
prefix KV state, feeds the captured boundary states, and checks logits, top-10
IDs, router IDs, and generated tokens. It never reconstructs token IDs by
encoding generated text.

## Intended run order

Generate the external greedy token-ID reference with vLLM-Ascend TP2:

```sh
python3 generate_vllm_reference.py \
  --model-dir /models/Qwen3-30B-A3B \
  --prompt "The capital of France is" \
  --max-new-tokens 8 \
  --output /results/qwen3_moe_vllm_reference.json
```

Run the custom full pipeline and capture the stage boundary:

```sh
python3 run_pipeline_capture.py \
  --model-dir /models/Qwen3-30B-A3B \
  --prompt "The capital of France is" \
  --max-new-tokens 8 \
  --cache-length 256 \
  --reference-json /results/qwen3_moe_vllm_reference.json \
  --capture-out /results/qwen3_moe_stage2_capture.pt \
  --summary-out /results/qwen3_moe_pipeline_summary.json
```

Replay only the second half on one NPU:

```sh
python3 replay_stage2.py \
  --model-dir /models/Qwen3-30B-A3B \
  --capture /results/qwen3_moe_stage2_capture.pt \
  --summary-out /results/qwen3_moe_stage2_replay_summary.json
```

Compile and benchmark the complete two-process pipeline:

```sh
torchrun --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29572 \
  benchmark_pipeline_compile_distributed.py \
  --model-dir /models/Qwen3-30B-A3B \
  --capture /results/qwen3_moe_stage2_capture.pt \
  --cache-length 4096 \
  --warmup-steps 2 \
  --decode-steps 200 \
  --compile-cache-dir /results/qwen3_moe_compile_cache \
  --summary-out /results/qwen3_moe_pipeline_compile_distributed_warm.json
```

Compile and benchmark the complete TP2 model:

```sh
torchrun --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29580 \
  benchmark_full_tp2.py \
  --model-dir /models/Qwen3-30B-A3B \
  --capture references/qwen3_moe_stage2_capture.pt \
  --cache-length 4096 \
  --warmup-steps 2 \
  --decode-steps 200 \
  --summary-out /results/qwen3_moe_full_tp2_k4096.json
```

Do not run both cached stage executors in one process by switching the current
NPU. TorchAir cache executors own device-context streams. That experiment either
serialized the second stage into the following token handoff or failed with
`stream is not in current ctx`. The two-process runner is the supported lane.

The verified development capture is stored at
`references/qwen3_moe_stage2_capture.pt`. It contains no model weights. It
contains only the valid prefix KV state, layer-24 boundary states, router
decisions, and expected logits needed by `replay_stage2.py`.

The first remote run must proceed in rungs: CPU unit test, one real MoE layer,
one complete stage, full pipeline without an external reference, then vLLM
token parity and stage-2 replay. Do not jump directly to the full pipeline if a
smaller rung fails. The one-layer rung is only an operator-contract and
correctness smoke. Never use it to accept or reject a performance change. All
performance decisions require the complete compiled 24-layer stage with final
RMSNorm and LM head.

The one-layer rung is:

```sh
python3 probe_real_layer.py \
  --model-dir /models/Qwen3-30B-A3B \
  --layer 24 \
  --cache-length 256
```

After that passes, load the complete second stage without the LM head:

```sh
python3 probe_real_layer.py \
  --model-dir /models/Qwen3-30B-A3B \
  --layer 24 \
  --num-layers 24 \
  --cache-length 256
```
