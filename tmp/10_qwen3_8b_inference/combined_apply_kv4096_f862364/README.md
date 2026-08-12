# Qwen3-8B B1/KV4096 decode optimization matrix

Run on 2026-08-12 on physical Ascend 910B2 NPU 7, exposed as logical
`npu:0`, from commit `f862364`. All lanes used FP16, batch size 1, a
512-token prefix, 64 decode steps, KV capacity 4096, dynamic TorchAir compile,
and IncreFA `actual_seq_lengths`.

The common benchmark command was:

```bash
/usr/local/python3.12.13/bin/python3 \
  10_qwen3_8b_inference/benchmark_local_qwen3_0.py \
  --model-dir /workspace/models/Qwen3-8B \
  --dtype float16 \
  --device npu:0 \
  --compile-decode \
  --compile-decode-dynamic \
  --decode-increfa-mode actual_seq_lengths \
  --prefill-tokens 512 \
  --decode-steps 64 \
  --static-kv-cache-len 4096 \
  --prefill-warmups 1 \
  --prefill-repeats 1 \
  --decode-warmups 1 \
  --decode-repeats 2
```

Each JSON records its explicit `decode_optimization` and
`decode_linear_weight_format`. The selected lane is `combined_apply` with
unchanged weights. Its warmed result was 79.11 tokens/s. It matched the
baseline greedy tokens for 64/64 steps and matched optimized eager versus
compiled tokens and KV state exactly.

The profile JSON is from a separate one-repeat run of the same selected lane;
its unprofiled decode result was 79.20 tokens/s. The profiler window itself is
not used as the throughput result.
