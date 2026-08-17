# Experiment 14: Qwen3-32B offline TP2 decode

This experiment is a small, non-serving starting point for Qwen3-32B decode on
two Ascend 910B2 NPUs. It derives from the selected Experiment 10 B1 decode
path, but it deliberately omits the 0.6B-specific prefetch and zero-bank
micro-optimizations.

The first target is simple: load the official BF16 checkpoint, compile one
fixed-shape B1 decode graph with TorchAir, run a contiguous static KV cache, and
measure full decode iterations. The timed step includes the transformer, final
RMSNorm, vocabulary-sharded LM head, distributed greedy selection, and KV
updates. It does not use paged attention or a serving scheduler.

## TP2 contract

The layout follows vLLM's Qwen3 tensor-parallel contract:

- Qwen3-32B has 64 query heads and 8 KV heads. TP2 owns 32 query heads and 4 KV
  heads per rank. KV heads are partitioned, not replicated.
- Q, K, and V are packed after each checkpoint tensor is independently sharded
  on its output dimension.
- Gate and up projections are packed after independent output sharding.
- O and down projections shard their input dimension and all-reduce their local
  outputs.
- Embeddings and the LM head shard the vocabulary dimension.
- Greedy selection all-gathers one local `(max logit, global token id)` pair per
  rank instead of gathering the full vocabulary logits.

At TP2 the model owns about 33 GiB of BF16 weights per rank. A B1, length-4096
KV cache adds about 0.5 GiB per rank before compiler workspaces.

Primary source references:

- [vLLM Qwen3 model](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py)
- [vLLM tensor-parallel linear layers](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py)
- [vLLM vocabulary-parallel embedding and LM head](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py)
- [vLLM local-argmax logits path](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py)
- [TorchAir compiled HCCL example](https://gitee.com/ascend/torchair/blob/master/examples/example_export_allreduce.py)

## Run order

Use two free 910B2 devices. Do not include physical device 5. The visible
device list must contain exactly the two allocated devices.

```sh
source npu-setup
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MODEL_DIR=/models/Qwen3-32B
cd 14_qwen3_32b_tp2
```

First prove that TorchAir captures the selected HCCL all-reduce and all-gather
forms:

```sh
/usr/local/python3.12.13/bin/torchrun \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29514 \
  probe_torchair_tp.py
```

Then load one layer. This checks real BF16 weight sharding, GQA IncreFA, static
KV updates, both row-parallel reductions, the sharded LM head, and greedy token
selection without paying for all 64 layers:

```sh
LAYERS=1 DECODE_STEPS=8 bash run_910b2_tp2.sh
```

Run the complete model only after the one-layer rung passes:

```sh
LAYERS=64 DECODE_STEPS=32 \
JSON_OUT=/results/qwen3_32b_tp2_b1_kv4096.json \
bash run_910b2_tp2.sh
```

`cache_length=4096` is the static attention capacity. The default synthetic
prefix position is 512, as in the selected Experiment 10 B1 benchmark. The
cache starts as zeros because prefill and text correctness are intentionally
outside this bring-up rung.

## Verified 910B2 result

Verified on 2026-08-17 at commit `7cd0f82`, with two Ascend 910B2 NPUs and the
official Qwen3-32B BF16 checkpoint:

| Lane | Shape | TPOT | Decode rate |
|---|---:|---:|---:|
| Custom raw eager | B1, static KV capacity 4096, position 512 | 121.80 ms | 8.21 tok/s |
| Custom TorchAir fullgraph | B1, static KV capacity 4096, position 512 | 31.73 ms | 31.52 tok/s |
| vLLM-Ascend v0.23 | B1, 20 sequential 128-in/128-out requests | 32.49 ms median; 34.81 ms mean | 26.21 output tok/s |

The custom graph's compile plus first call took 56.30 seconds. After compile it
owned 31.10 GiB per rank and left about 28.27 GiB free per rank. The vLLM lane
is a serving-shaped anchor, not an identical attention-layout comparison:
vLLM uses paged KV and includes HTTP/scheduler work, while the custom lane uses
one offline contiguous static cache. The close TPOT values show that the custom
baseline is in the expected performance range; they are not an accuracy or
strict apples-to-apples speed claim.

Raw evidence is in `references/`:

- `qwen3_32b_tp2_b1_kv4096_raw_eager_910b2.json`
- `qwen3_32b_tp2_b1_kv4096_torchair_910b2.json`
- `qwen3_32b_vllm_ascend_tp2_b1_i128_o128_n20_910b2.json`
