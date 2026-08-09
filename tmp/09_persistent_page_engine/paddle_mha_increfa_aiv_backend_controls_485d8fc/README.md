# Separate Paddle MHA AIV backend controls

Running without TorchAir is the correct first control, but it answers two
different questions depending on scope:

1. The real B1 decoder in raw eager mode measures whether TorchAir can be
   removed from the production path. It cannot: throughput falls to about
   one tenth of the compiled B1 range.
2. The isolated raw-eager operator measures the PyTorch/ACLNN dispatch path and
   the kernel without graph integration. It shows that the custom AIV kernel
   body is faster than the stock mixed kernel at KV128, while the first custom
   eager bridge has much higher launch cadence overhead.

All valid measurements below ran on physical Ascend 910B2 NPU 6, exposed as
logical `npu:0`. Physical NPU 5 was excluded: both stock and custom attention
calls timed out there in the same session, so that device could not distinguish
an operator failure from a device/runtime failure.

## Real B1 raw-eager GQA

The real-forward control used the production-relevant B1 GQA contract: 16 query
heads, 2 stored KV heads, FP16, static KV4096, `combined_apply`, and stock
IncreFA. Vision, text prefill, and decode all used `raw_eager`; no TorchAir
compile API was active.

- Setup: `21.317426 s`
- Run wall: `5.928648 s`
- Input tokens: `1021`
- Generated tokens: `374`, stopped by EOS
- Continuous decode wall: `5.400518 s`
- Decode model plus argmax device time: `5.165274 s`
- Raw decode throughput: `69.2526 tok/s`
- Effective decode throughput: `69.0674 tok/s`
- Output token hash:
  `6dfe54351979fbfa7903999f941b761a32c9a00c29128b0f75d0270da3c93f56`

The user's measured compiled B1 range is about `700-800 tok/s`. The prior
direct compiled GQA control in this repository reached `678.7` effective
device tok/s. Therefore raw eager is useful as a diagnostic control, but it is
not a viable replacement for TorchAir in the complete decoder.

Remote artifact:

```text
.runtime_cache/paddle_mha_increfa_aiv/real_b1_raw_eager_gqa.json
```

Command after `source npu-setup`:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_real_generation.py \
  --decode-backend raw_eager \
  --decode-optimization combined_apply \
  --batch-size 1 \
  --replicas 1 \
  --cache-length 4096 \
  --max-new-tokens 512 \
  --target-effective-length 1280 \
  --output \
    .runtime_cache/paddle_mha_increfa_aiv/real_b1_raw_eager_gqa.json
```

## Isolated raw-eager MHA operator

This control used B1 FP16 BNSD MHA with 16 query heads, 16 KV heads, head
dimension 128, a bool attention mask, and no TorchAir. Stock and custom lanes
ran in separate processes. Outputs matched bit-for-bit at every KV length.

| KV length | Stock median | Custom median | Custom change |
| ---: | ---: | ---: | ---: |
| 128 | 51.3728 us | 162.6420 us | +216.6% latency |
| 512 | 52.8003 us | 166.9864 us | +216.3% latency |
| 2048 | 53.0126 us | 171.8772 us | +224.2% latency |

This is not evidence that the custom kernel body is slower. The KV128 pipe
profiles contain exactly one attention kernel per call and no helper or
TransData kernels. Across five calls:

- Stock kernel compute: `119.460 us` total, or `23.892 us/call`.
- Custom kernel compute: `104.020 us` total, or `20.804 us/call`.
- Kernel-body reduction: `12.9%`.

The raw-eager regression is outside the kernel body. The first separate ACLNN
bridge does not use the mature stock op-plugin launch/cache path, and the
profile attributes substantially more stage-free/preparing time to the custom
lane. This makes direct eager valuable for finding the bridge overhead, but it
does not predict compiled graph latency.

Remote artifacts:

```text
.runtime_cache/paddle_mha_increfa_aiv/stock_only_eager_matrix_f4e34ba.json
.runtime_cache/paddle_mha_increfa_aiv/custom_only_eager_matrix_f4e34ba.json
.runtime_cache/paddle_mha_increfa_aiv/profiles_cae753d/stock/profile_parse_summary.md
.runtime_cache/paddle_mha_increfa_aiv/profiles_cae753d/custom/profile_parse_summary.md
```

The later package changes restored nested composite tiling schema names but did
not change the kernel object. A current-package KV128 control on the same NPU
again matched the stock hash exactly and reproduced the slower eager cadence.

## Isolated TorchAir graph control

Each KV length ran in a fresh process. First-call graph compile/cache-load time
was recorded separately and excluded. This is important: a discarded
multi-length-in-one-process attempt emitted repeated `recompiled` cache
warnings, so its timings are not used here.

| KV length | Stock median | Custom median | Custom change | Exact output |
| ---: | ---: | ---: | ---: | :---: |
| 128 | 246.4670 us | 136.5778 us | -44.6% latency | yes |
| 512 | 245.3936 us | 147.8388 us | -39.8% latency | yes |
| 2048 | 240.6119 us | 245.2068 us | +1.9% latency | yes |

The graph result is shape-dependent. The separate AIV operator wins strongly
at KV128 and KV512, then reaches parity at KV2048. A single-op graph also has
large fixed graph-launch overhead, so these absolute values must not be treated
as complete-decoder token latency.

Remote artifacts:

```text
.runtime_cache/paddle_mha_increfa_aiv/compiled_custom_kv128_schema_e892634_npu6_control.json
.runtime_cache/paddle_mha_increfa_aiv/compiled_stock_kv128_e2a152f_npu6_control.json
.runtime_cache/paddle_mha_increfa_aiv/compiled_custom_kv512_isolated_schema_e892634.json
.runtime_cache/paddle_mha_increfa_aiv/compiled_stock_kv512_isolated_e2a152f.json
.runtime_cache/paddle_mha_increfa_aiv/compiled_custom_kv2048_isolated_schema_e892634.json
.runtime_cache/paddle_mha_increfa_aiv/compiled_stock_kv2048_isolated_e2a152f.json
```

## Boundary and next step

The independent custom operator currently implements the MHA contract
`16Q/16KV`. The production decoder uses GQA `16Q/2KV`. Expanding the cache from
2 to 16 KV heads already regressed the real compiled forward by about 2%, and
the earlier zero-cube GQA package hung after successful graph compilation.

The next production-relevant experiment is a second, independently named GQA
AIV operator. It must preserve 2 stored KV heads. It must first pass raw-eager
parity and profiling, then isolated TorchAir integration, and only then replace
stock IncreFA in one real compiled B1 forward lane.
