# Experiment 15: Qwen3-30B-A3B pipeline and stage replay

This experiment establishes a correctness-first custom Qwen3-30B-A3B decode
lane before optimizing MoE execution.

The full model runs as a two-stage sequential pipeline:

- stage 0 on one 910B2: token embedding and layers 0 through 23;
- stage 1 on a second 910B2: layers 24 through 47, final RMSNorm, LM head, and
  greedy token selection.

The official BF16 checkpoint contains 61,064,245,248 bytes of tensors. A
24-layer split therefore fits comfortably on two 64 GB 910B2 devices. B1 does
not provide useful inter-stage pipeline overlap, so this first lane uses a
single host process and a direct NPU-to-NPU boundary copy. The goal is exact
state ownership and token parity, not pipeline throughput.

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

The verified development capture is stored at
`references/qwen3_moe_stage2_capture.pt`. It contains no model weights. It
contains only the valid prefix KV state, layer-24 boundary states, router
decisions, and expected logits needed by `replay_stage2.py`.

The first remote run must proceed in rungs: CPU unit test, one real MoE layer,
one complete stage, full pipeline without an external reference, then vLLM
token parity and stage-2 replay. Do not jump directly to the full pipeline if a
smaller rung fails.

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
