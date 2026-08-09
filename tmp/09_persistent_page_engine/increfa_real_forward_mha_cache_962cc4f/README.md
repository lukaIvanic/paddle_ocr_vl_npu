# IncreFA zero-cube real-forward result

The zero-cube `0:1` hard-sync package is correct in a real B16 PaddleOCR-VL
decode, but it is slower than the validated mixed `1:2` control. It should not
replace the mixed package on performance grounds.

## Scope

- Project commit: `962cc4f`
- Hardware: physical Ascend 910B2 NPU 6, exposed as `npu:0`
- Dtype/layout: FP16/BNSD
- Decode optimization: `combined_apply_mha_cache`
- Shape: B16, KV4096, 16 query heads, 16 stored KV heads, head dimension 128
- Control: validated mixed package with `taskRation=1:2`
- Candidate: hard-sync package from source commit `a63741c`, with
  `KERNEL_TYPE_MIX_AIV_1_0` and `taskRation=0:1`

The fixed custom package covers the FP16 MHA key. Experiment 09 production uses
the GQA `combined_apply` key, so the default production path would not exercise
this package. `combined_apply_mha_cache` is the closest valid full-model lane:
it performs the real 18-layer decoder forward while keeping an expanded MHA KV
arena, and therefore selects the fixed MHA key without per-layer full-cache
repetition.

After `source npu-setup`, each real-generation process used this command shape:

```sh
env \
  ASCEND_CUSTOM_OPP_PATH="<selected-package>:${ASCEND_CUSTOM_OPP_PATH}" \
  LD_LIBRARY_PATH="<selected-package>/op_api/lib/:${LD_LIBRARY_PATH}" \
  /workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_real_generation.py \
  --decode-cache-dir <implementation-specific-cache> \
  --decode-optimization combined_apply_mha_cache \
  --reference <mixed-warm-result> \
  --output <per-process-result>
```

The steady profile used the same package and cache selection with:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_lab.py \
  --mode profile \
  --batch-size 16 \
  --cache-length 4096 \
  --profile-position 1279 \
  --decode-optimization combined_apply_mha_cache \
  --cache-dir <implementation-specific-cache> \
  --warmup 20 \
  --repeats 100 \
  --allow-compile \
  --output <per-process-result>
```

The selected package roots were
`.runtime_cache/increfa_mixed_opp_20260809/vendors/paddle_increfa_mixed_transformer`
and
`.runtime_cache/increfa_aiv_only_hardsync_opp_a63741c/vendors/paddle_increfa_aiv_only_transformer`.
The cache roots were `.runtime_cache/09_increfa_real_forward_mixed_962cc4f`
and `.runtime_cache/09_increfa_real_forward_hardsync_962cc4f`.

## Real generation

`text_decode_real_generation.py` selected block 3 from the fixed OmniDocBench
page, ran real vision and text prefill, admitted 16 identical requests, and
decoded each request for 374 generated tokens through EOS. This executed 374
complete decoder graph calls and produced 5,968 effective tokens per process.

Each implementation received one cache-warmup process. The measured order was
ABBA/BABA with four fresh processes per implementation. Each implementation had
its own TorchAir cache root. All candidate and measured-control outputs matched
the warm control exactly at token, text, and stop-reason level. Every request
had token hash
`6dfe54351979fbfa7903999f941b761a32c9a00c29128b0f75d0270da3c93f56`.

| Metric | Mixed `1:2` median | Zero-cube `0:1` median | Candidate change |
| --- | ---: | ---: | ---: |
| Model plus argmax device time | 3.0448 s | 3.1028 s | +1.90% latency |
| Continuous decode wall | 3.0862 s | 3.1405 s | +1.76% latency |
| Full run wall | 5.4219 s | 5.5033 s | +1.50% latency |
| Effective decode throughput | 1,933.8 tok/s | 1,900.4 tok/s | -1.73% throughput |
| Device-only effective throughput | 1,960.1 tok/s | 1,923.5 tok/s | -1.87% throughput |

The per-process model-plus-argmax device times were:

- Mixed: `3.032158`, `3.057429`, `3.074447`, `3.030324` seconds.
- Zero-cube: `3.113054`, `3.091862`, `3.092474`, `3.124045` seconds.

## Steady full-decoder profile

`text_decode_lab.py --mode profile` then measured the complete real-weight
decoder step at cache position 1279. Each process used 20 warmups followed by
100 measured B16 steps. Two fresh processes were run per implementation in
ABBA order.

| Implementation | Process mean latencies | Mean of process means | Physical throughput |
| --- | --- | ---: | ---: |
| Mixed `1:2` | 8.1177, 8.0480 ms | 8.0828 ms | 1,979.5 tok/s |
| Zero-cube `0:1` | 8.2214, 8.2636 ms | 8.2425 ms | 1,941.2 tok/s |

The steady profile therefore measured a `+1.98%` candidate latency change and
a `-1.94%` throughput change.

## Interpretation

The B1 operator microbenchmark found a small fixed launch saving at KV128 and
KV512, but that saving does not carry into the real B16 full-decoder graph. The
full-model measurements consistently favor the mixed schedule. A likely class
of explanations is that the `0:1` hard-sync runtime shape changes vector-core
scheduling or synchronization behavior at B16, but this run does not identify
the causal hardware counter. A focused Torch-NPU profile is required before
assigning the regression to occupancy, barriers, or another runtime effect.

The actionable result is already clear: do not adopt the current zero-cube
package as a real-forward performance optimization, and do not describe it as
a production GQA result.
