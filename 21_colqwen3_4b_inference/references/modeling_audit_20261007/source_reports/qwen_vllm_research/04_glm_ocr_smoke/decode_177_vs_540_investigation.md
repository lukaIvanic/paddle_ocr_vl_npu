# GLM-OCR vLLM-Ascend 177 tok/s vs speedup_roadmap 540 tok/s

Date: 2026-07-07

## Summary

The 177-180 tok/s dense-KV bridge result is not caused by missing
`FULL_DECODE_ONLY` graph capture. The run was re-tested with explicit
`FULL_DECODE_ONLY`, decode `FULL` ACL graph dispatch, and ACL graph replay.
Throughput stayed ~180 tok/s.

The gap to `speedup_roadmap` is caused by implementation boundary and cache
layout differences:

- vLLM-Ascend still uses the normal vLLM request/scheduler/paged-cache path.
- The dense bridge gathers/transposes a full dense KV window out of vLLM's paged
  cache inside every attention layer and decode step.
- `speedup_roadmap` uses a custom decode-only TorchAir fullgraph module with a
  native static dense KV cache, static mask/control tensors, and no vLLM request
  machinery in the timed decode section.
- `speedup_roadmap` also includes GLM-OCR-specific decode optimizations such as
  half-layout Q/K cache, `npu_rotary_mul(..., rotary_mode="half")`, precomputed
  masks, and a direct one-token decode loop.

## Measured Results On `blue_zone_npu_container`

Common setup:

```sh
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib:${LD_LIBRARY_PATH:-}
```

All vLLM cases used GLM-OCR with `--max-model-len 2048`, batch size 1,
`--ignore-eos`, and `--max-tokens 256` unless otherwise noted.

| Case | Attention path | Graph mode | Mean output tok/s |
| --- | --- | --- | ---: |
| Stock normal | `npu_fused_infer_attention_score` | explicit `FULL_DECODE_ONLY` | 128.0 |
| Stock forced paged | `_npu_paged_attention` | explicit `FULL_DECODE_ONLY` | 108.1 |
| Dense KV bridge, 2048 window | paged cache -> dense temp -> `npu_incre_flash_attention` | default `FULL_AND_PIECEWISE` | 176.0 |
| Dense KV bridge, 2048 window | paged cache -> dense temp -> `npu_incre_flash_attention` | explicit `FULL_DECODE_ONLY` | 178.7 |
| Dense KV bridge, 2048 window, 1024 generated tokens | same as above | explicit `FULL_DECODE_ONLY` | 179.8 |
| Dense KV bridge, 1024 window, 512 generated tokens | same as above | explicit `FULL_DECODE_ONLY` | 206.7 |
| Persistent dense KV mirror, 2048 window, 128 generated tokens | per-layer dense mirror -> `npu_incre_flash_attention` | explicit `FULL_DECODE_ONLY` | 190.8 |
| Persistent dense KV mirror, 2048 window, 256 generated tokens | same as above | explicit `FULL_DECODE_ONLY` | 198.4 |

The 1024-token run staying at ~180 tok/s shows this is not just short-run fixed
overhead. The 1024-window run improving to ~207 tok/s shows the temporary dense
KV window size matters, but does not explain the full gap.

## Graph Capture Proof

Explicit `FULL_DECODE_ONLY` was passed through:

```sh
--compilation-config-json '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
```

The engine config showed:

```text
cudagraph_mode: <CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>
splitting_ops: []
cudagraph_capture_sizes: [1]
```

The logs showed only decode FULL capture:

```text
Capturing CUDA graphs (decode, FULL)
Graph capturing finished
```

The runtime trace showed prefill outside graph and decode in full replay:

```text
VLLM_CG_DISPATCH result=NONE requested_tokens=308 ... uniform_decode=False
VLLM_CG_REQUEST cg_mode=NONE will_run_fullgraph=False actual_tokens=308
VLLM_CG_DISPATCH result=FULL requested_tokens=1 ... uniform_decode=True
VLLM_CG_REQUEST cg_mode=FULL will_run_fullgraph=True actual_tokens=1
acl_graph.py:257 Replaying aclgraph
```

So the vLLM-Ascend path is using a real decode FULL ACL graph. That alone is
not sufficient to reach the custom roadmap speed.

## Profiler Attribution

This was profiled after the first comparison because graph-mode evidence alone
was not enough to explain the speed gap. The profiler confirms the blocker is
different for stock vLLM-Ascend and for the dense-KV bridge.

Profile command for the dense bridge:

```sh
cd /workspace/repos/qwen_vllm_research
rm -rf outputs/prof_vllm_dense_bridge_decode8
export VLLM_ASCEND_RESEARCH_DENSE_KV_BRIDGE=1
export VLLM_ASCEND_RESEARCH_DENSE_KV_MAX_LEN=2048
export VLLM_ASCEND_FORCE_REQUESTED_CUDAGRAPH_MODE=1
unset VLLM_ASCEND_RESEARCH_CG_TRACE
unset VLLM_ASCEND_RESEARCH_ATTENTION_TRACE
unset VLLM_ASCEND_RESEARCH_DENSE_KV_VALIDATE
unset VLLM_ASCEND_FORCE_PAGED_ATTENTION
/usr/local/python3.12.13/bin/python3 04_glm_ocr_smoke/profile_glm_ocr_vllm_ascend_decode.py \
  --profile-dir outputs/prof_vllm_dense_bridge_decode8 \
  --metadata-json outputs/prof_vllm_dense_bridge_decode8_metadata.json \
  --max-model-len 2048 \
  --max-tokens 20 \
  --warmup-generates 2 \
  --delay-iterations 2 \
  --max-iterations 8 \
  --full-decode-only \
  > outputs/prof_vllm_dense_bridge.log 2>&1
```

Generated profile:

```text
/workspace/repos/qwen_vllm_research/outputs/prof_vllm_dense_bridge_decode8/rank0_480656_20260707183625295_ascend_pt/ASCEND_PROFILER_OUTPUT/
```

Previously collected comparison profiles:

```text
stock normal:
/workspace/repos/qwen_vllm_research/outputs/prof_vllm_decode8/rank0_396054_20260706125258452_ascend_pt/ASCEND_PROFILER_OUTPUT/

forced paged:
/workspace/repos/qwen_vllm_research/outputs/prof_vllm_decode8/rank0_425672_20260707114609938_ascend_pt/ASCEND_PROFILER_OUTPUT/
```

The profiler wall time is not representative throughput because Torch profiler
adds large overhead. Use the CSVs for attribution.

### Step Trace

All values below are totals over the 8 profiled decode steps.

| Case | Computing | Free | Stage | Preparing |
| --- | ---: | ---: | ---: | ---: |
| Stock normal FIA | 17.299 ms | 58.306 ms | 75.605 ms | 1.028 ms |
| Forced paged attention | 18.796 ms | 78.199 ms | 96.995 ms | 0.992 ms |
| Dense KV bridge | 41.316 ms | 6.197 ms | 47.513 ms | 0.920 ms |

The dense bridge removes most of the stock vLLM free/bubble time, but it does so
by making device compute much heavier.

### Stock Normal FIA Top Device Ops

| Op | Total device time | Count | Share |
| --- | ---: | ---: | ---: |
| `MatMulV2` | 9.145 ms | 520 | 52.9% |
| `FusedInferAttentionScore` | 3.265 ms | 128 | 18.9% |
| `RopeWithSinCosCache` | 1.065 ms | 128 | 6.2% |
| `RmsNorm` | 1.014 ms | 264 | 5.9% |
| `_compute_slot_mapping_kernel` | 0.963 ms | 8 | 5.6% |

Host/API attribution shows repeated attention setup work:

| API/self event | Total | Count |
| --- | ---: | ---: |
| `npu::npu_fused_infer_attention_score` host self | 6.666 ms | 128 |
| `aclnnInnerFusedInferAttentionScore` | 4.088 ms | 128 |
| `aclnnInnerFusedInferAttentionScoreGetWorkspaceSize` | 1.861 ms | 128 |
| `aclrtSynchronizeStreamWithTimeout` | 0.144 ms | 8 |

So stock decode is not mainly blocked on explicit stream synchronization. It is
mostly framework/attention setup free time plus normal graph compute.

### Forced Paged Top Device Ops

| Op | Total device time | Count | Share |
| --- | ---: | ---: | ---: |
| `MatMulV2` | 10.101 ms | 520 | 53.7% |
| `paged_attention_mask_16_mix_aic` | 3.817 ms | 128 | 20.3% |
| `RopeWithSinCosCache` | 1.116 ms | 128 | 5.9% |
| `RmsNorm` | 1.016 ms | 264 | 5.4% |
| `_compute_slot_mapping_kernel` | 0.962 ms | 8 | 5.1% |

Paged attention has even more host-side setup:

| API/self event | Total | Count |
| --- | ---: | ---: |
| `PagedAttentionOperation::Setup` | 9.925 ms | 256 |
| `PagedAttentionOperation::Execute` | 5.129 ms | 128 |
| `atb::_npu_paged_attention_get_workspace` host self | 8.897 ms | 128 |
| `atb::_npu_paged_attention` host self | 3.310 ms | 128 |

This explains why forcing `_npu_paged_attention` was slower than the stock FIA
path in the batch-size-1 GLM-OCR decode test.

### Dense KV Bridge Top Device Ops

| Op | Total device time | Count | Share |
| --- | ---: | ---: | ---: |
| `Index` | 15.448 ms | 272 | 37.4% |
| `MatMulV2` | 10.087 ms | 520 | 24.4% |
| `Transpose` | 5.456 ms | 256 | 13.2% |
| `IncreFlashAttention` | 4.318 ms | 128 | 10.5% |
| `RopeWithSinCosCache` | 1.080 ms | 128 | 2.6% |
| `RmsNorm` | 1.021 ms | 264 | 2.5% |
| `_compute_slot_mapping_kernel` | 0.963 ms | 8 | 2.3% |

The dense bridge changes the limiting factor from stock host/framework idle
time to dense-cache reconstruction work. The bridge-specific cost is visible as
`Index` plus `Transpose`, and the attention kernel itself is not the largest
piece.

Per profiled decode step, the dense path spends roughly:

| Dense bridge component | Approx. per-step time |
| --- | ---: |
| `Index` gather from paged KV | 1.93 ms |
| `Transpose` / layout work | 0.68 ms |
| `IncreFlashAttention` | 0.54 ms |
| Combined gather/layout/attention | 3.15 ms |

This is the measured reason it lands around 180-207 tok/s instead of 540 tok/s.
It is paying every layer and every token to reconstruct a dense static cache
view from vLLM's paged cache.

### Exact Source Of `Index`

The dense bridge lives in the local vLLM-Ascend fork:

```text
/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py
```

The expensive `Index` profiler rows come from `_forward_dense_working_kv_attention`:

```python
block_table = attn_metadata.block_tables[:num_tokens, :blocks_per_seq]
dense_key = self.key_cache[block_table.long()].reshape(
    num_tokens,
    physical_len,
    self.num_kv_heads,
    self.head_size,
)
dense_value = self.value_cache[block_table.long()].reshape(
    num_tokens,
    physical_len,
    self.num_kv_heads,
    self.head_size,
)
dense_key = dense_key.transpose(1, 2).contiguous()
dense_value = dense_value.transpose(1, 2).contiguous()
```

That is ordinary PyTorch advanced indexing on NPU tensors. For GLM-OCR
batch-size 1 with a 2048-token dense window and block size 128:

```text
self.key_cache       [num_blocks, 128, 8, 128]
block_table slice    [1, 16]
indexed result       [1, 16, 128, 8, 128]
reshape result       [1, 2048, 8, 128]
transpose result     [1, 8, 2048, 128]
```

It captures because the graph sees a fixed-shape tensor indexing operation:

- `block_table` is a tensor input with stable shape.
- `block_table.long()` is a tensor cast, not a Python value escape.
- The indexed output shape is static for a fixed dense window.
- CANN lowers the operation to an `Index` device kernel.

So this is graph-compatible, but expensive. The graph is not broken; it is just
doing a large paged-cache gather inside every attention layer.

### Can We Rebuild Dense KV Once Per Step?

There are two different interpretations:

1. Rebuild one dense KV tensor total and share it across layers.
2. Rebuild all layer-specific dense KV tensors once at the beginning of the
   decode step, then each layer reads its own dense slice.

The first is not valid. Every transformer layer has different K/V cache
contents. Layer 0 dense K/V cannot be reused by layer 1.

The second is possible, but it does not remove the fundamental memory movement.
It still has to gather K and V for every layer:

```text
num_layers * 2 * physical_len * num_kv_heads * head_dim
```

For GLM-OCR text decode at `physical_len=2048`, that is still about 128 MiB of
K/V data per generated token across 16 layers, before layout conversions. It
may reduce duplicated block-table casting, mask creation, and Python/attention
method overhead, but it will not eliminate the profiler's dominant `Index`
cost.

The actual route toward `speedup_roadmap` performance is to avoid rebuilding
from the paged cache:

- Keep the scheduler-owned paged cache for vLLM correctness and block
  ownership.
- Add a persistent dense mirror cache for the experimental GLM-OCR decode path.
- During prefill or first decode entry, initialize the dense mirror once from
  the paged cache.
- On each decode step, update only the current token's dense K/V for each layer
  while vLLM still writes the paged cache through its existing
  `_npu_reshape_and_cache` path.
- Run `npu_incre_flash_attention` directly over the dense mirror.

That turns the current per-token, per-layer dense-window gather into a
per-token, per-layer one-position dense update, matching the shape of the
`speedup_roadmap` static-cache design.

## Persistent Dense Mirror Result

Commit `af7f4974` in the local vLLM-Ascend fork adds the experimental persistent
dense KV mirror:

```text
/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py
```

Runtime controls used for the verified run:

```sh
export VLLM_ASCEND_RESEARCH_DENSE_KV_BRIDGE=1
export VLLM_ASCEND_RESEARCH_DENSE_KV_PERSISTENT=1
export VLLM_ASCEND_RESEARCH_DENSE_KV_MAX_LEN=2048
export VLLM_ASCEND_RESEARCH_DENSE_KV_UPDATE_MODE=scatter_nd_direct
export VLLM_ASCEND_FORCE_REQUESTED_CUDAGRAPH_MODE=1
```

The implementation keeps one dense mirror per attention layer in attention-ready
layout:

```text
[batch, num_kv_heads, physical_len, head_dim] = [1, 8, 2048, 128]
```

Each layer still writes the normal vLLM paged cache through
`DeviceOperator.reshape_and_cache`. The research path additionally updates only
the current token positions in the dense mirror. The first attempted updater used
`torch_npu.scatter_update_`, but ACL graph replay failed in
`aclnnInplaceScatterUpdate`. The working updater uses direct 4D
`torch_npu.npu_scatter_nd_update_` indices:

```text
[batch_row, kv_head, position] -> head_dim row
```

Verified smoke:

```text
outputs/glm_ocr_persistent_dense_smoke.json
status=ok
warm repeat 0: 128 tokens / 0.6718 s = 190.5 tok/s
warm repeat 1: 128 tokens / 0.6698 s = 191.1 tok/s
text starts with the expected BLUE ZONE CAFE receipt content
```

The stronger 256-token comparison also passed:

```text
outputs/glm_ocr_persistent_dense_256_compare.json
status=ok
256 tokens / 1.2905 s = 198.4 tok/s
text_sha256=932cd1f2bcfd4207d45bd0bda7dd78772568ce13abe4b41c83aae5042c980d72
matches known stock/dense-bridge 256-token hash: true
```

The trace confirms persistent updates are active for prefill and decode:

```text
[research dense kv persistent] allocated shape=(1, 8, 2048, 128)
[research dense kv persistent] update ... capturing=True mode=scatter_nd_direct
[research dense kv persistent] update num_tokens=308 ... capturing=False mode=scatter_nd_direct
```

Profiler output:

```text
outputs/prof_vllm_persistent_dense_decode8/rank0_485536_20260707190457006_ascend_pt/ASCEND_PROFILER_OUTPUT/
```

Top device ops over 8 profiled decode steps:

| Op | Total time | Count | Share |
| --- | ---: | ---: | ---: |
| `MatMulV2` | 10.204 ms | 520 | 43.1% |
| `IncreFlashAttention` | 4.382 ms | 128 | 18.5% |
| `ScatterNdUpdate` | 1.684 ms | 256 | 7.1% |
| `RopeWithSinCosCache` | 1.117 ms | 128 | 4.7% |
| `RmsNorm` | 1.056 ms | 264 | 4.5% |
| `_compute_slot_mapping_kernel` | 0.964 ms | 8 | 4.1% |
| `Index` | 0.155 ms | 16 | 0.7% |

This proves the intended expensive block-table gather was removed from the decode
critical path. In the old dense bridge, `Index` was 15.433 ms and `Transpose` was
5.456 ms over the same 8-step profiling window. In the persistent path, `Index`
is only 0.155 ms and `Transpose` is absent; the replacement cost is
`ScatterNdUpdate` at 1.684 ms.

The persistent mirror is therefore directionally correct, but not sufficient by
itself to reach the standalone `speedup_roadmap` 540 tok/s result. Remaining
major differences include vLLM scheduler/output/input-prep overhead, slot mapping
work, per-layer dense mirror scatter updates, and the fact that this vLLM path
still uses the stock GLM-OCR model structure rather than the roadmap's direct
static decode loop and GLM-specific layout/rotary optimizations.

## speedup_roadmap Reference

Saved artifacts on the same container:

```text
/workspace/repos/aoe_speedup_try/artifacts/speedup_roadmap/03c_half_layout_rotary/manifest.json
```

Relevant saved results:

| Variant | cache_len | decode tok/s | NPU-event tok/s |
| --- | ---: | ---: | ---: |
| current_interleave_manual | 1024 | 521.7 | 524.9 |
| half_layout_manual | 1024 | 514.1 | 515.6 |
| half_layout_npu_half | 1024 | 544.4 | 545.9 |

Batch-size-1 saved artifact:

```text
/workspace/repos/aoe_speedup_try/artifacts/speedup_roadmap/03d_batched_decode/manifest_bs1.json
```

It reports `half_layout_npu_half` at 541.5 tok/s, 545.0 NPU-event tok/s.

The roadmap benchmark starts timing after prefill has completed and synced, then
times only the decode loop. It does not include vLLM request scheduling,
multimodal prompt processing, block-table metadata construction, output object
assembly, or prefix-cache lookup.

## Main Technical Difference

Current dense bridge in vLLM-Ascend:

```text
vLLM paged KV cache
  -> block_table gather
  -> reshape to dense [B, S, kv_heads, head_dim]
  -> transpose + contiguous
  -> build dense padding mask
  -> npu_incre_flash_attention
```

That happens in each attention layer. For GLM-OCR text config:

```text
num_hidden_layers = 16
num_attention_heads = 16
num_key_value_heads = 8
head_dim = 128
hidden_size = 1536
```

For a 2048-token bridge window, one layer materializes:

```text
2048 * 8 * 128 fp16 values for K = ~4 MiB
2048 * 8 * 128 fp16 values for V = ~4 MiB
~8 MiB per layer, before transpose/mask/attention
~128 MiB per decode token across 16 layers
```

`speedup_roadmap` instead keeps the static dense KV cache as the real cache and
updates it directly with `torch_npu.scatter_update_`. There is no paged-cache to
dense-window gather on every layer and token.

## Conclusion

The original dense bridge was a useful correctness probe, but its per-layer
paged-cache gather was the wrong performance boundary. The persistent dense
mirror fixes that specific blocker and improves the verified batch-size-1 warm
decode result to about 191 tok/s with `FULL_DECODE_ONLY`.

It is still not equivalent to `speedup_roadmap`. To approach 540 tok/s inside
vLLM-Ascend, the remaining work is to reduce framework/scheduler overhead,
metadata/slot-mapping work, per-layer dense mirror update cost, and model-kernel
differences such as the roadmap's half-layout Q/K cache and rotary path.
