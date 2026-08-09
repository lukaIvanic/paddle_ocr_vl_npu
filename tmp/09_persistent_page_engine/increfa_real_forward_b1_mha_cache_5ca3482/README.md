# IncreFA zero-cube B1 real-forward result

The corrected B1 test does not show an end-to-end latency improvement. The
zero-cube `0:1` hard-sync package makes the isolated IncreFA call slightly
faster at the real KV lengths, but the complete compiled 18-layer forward is
consistently about 2% slower than the validated mixed `1:2` control.

## Scope and production boundary

- Project commit: `5ca3482`
- Hardware: physical Ascend 910B2 NPU 6, exposed as `npu:0`
- Dtype/layout: FP16/BNSD
- Shape: B1, KV4096, 16 query heads, 16 stored KV heads, head dimension 128
- Decode optimization: `combined_apply_mha_cache`
- Control package: mixed `taskRation=1:2`
- Candidate package: zero-cube hard-sync `taskRation=0:1`, source commit
  `a63741c`

The fixed binary was reconstructed and validated with FP16 MHA inputs. The best
production B1 lane uses `combined_apply` GQA. A later direct probe established
that the mixed fixed-key package can execute that GQA call exactly, but the
zero-cube package hangs after graph compilation. Therefore the MHA-cache matrix
below is the completed candidate/control comparison, while the direct GQA probe
is a compatibility failure rather than a timing result.

## Real B1 generation

`text_decode_real_generation.py` selected the fixed real OmniDocBench table
crop, performed real vision and text prefill, and decoded one request for 374
tokens through EOS. Each implementation received one cache-warmup process. The
measured order was ABBA/BABA with four fresh processes per implementation and
separate TorchAir cache roots.

All candidate outputs matched the mixed reference exactly at token, text, and
stop-reason level. The generated-token hash was
`6dfe54351979fbfa7903999f941b761a32c9a00c29128b0f75d0270da3c93f56`.

| Metric | Mixed `1:2` median | Zero-cube `0:1` median | Candidate change |
| --- | ---: | ---: | ---: |
| Model plus argmax device time | 0.6408 s | 0.6542 s | +2.10% latency |
| Continuous decode wall | 0.6570 s | 0.6700 s | +1.99% latency |
| Full run wall | 1.1840 s | 1.1966 s | +1.06% latency |
| Effective decode throughput | 567.8 tok/s | 556.7 tok/s | -1.94% throughput |
| Device-only effective throughput | 582.1 tok/s | 570.2 tok/s | -2.06% throughput |

The per-process model-plus-argmax device times were:

- Mixed: `0.645971`, `0.635577`, `0.634091`, `0.647571` seconds.
- Zero-cube: `0.648858`, `0.663605`, `0.659629`, `0.648815` seconds.

## Steady B1 full-decoder profile

`text_decode_lab.py --mode profile` measured the complete real-weight decoder
step at cache position 1279. Every fresh process used 200 warmups and 2,000
measured steps. Six processes per implementation were run in interleaved order.

| Implementation | Process mean latencies, ms | Median | Median throughput |
| --- | --- | ---: | ---: |
| Mixed `1:2` | 1.6331, 1.5982, 1.5970, 1.6209, 1.6192, 1.6040 | 1.6116 ms | 620.5 tok/s |
| Zero-cube `0:1` | 1.6343, 1.6637, 1.6457, 1.6336, 1.6390, 1.6409 | 1.6399 ms | 609.8 tok/s |

The candidate changed median latency by `+1.76%` and median throughput by
`-1.73%`. Comparing the arithmetic mean of all six process means gives a
`+1.91%` latency change.

## Isolated operator at the real KV lengths

The operator probe used four fresh processes per implementation. Every process
used 5,000 warmups followed by nine blocks of 2,000 calls at each KV length.
The table reports the median of process block medians.

| KV length | Mixed `1:2` | Zero-cube `0:1` | Candidate latency change |
| ---: | ---: | ---: | ---: |
| 1024 | 50.4641 us | 50.2393 us | -0.45% |
| 1280 | 50.8631 us | 50.3331 us | -1.04% |
| 1408 | 51.2370 us | 50.7046 us | -1.04% |

The custom op therefore remains locally faster. At KV1280, the saving is about
0.53 us per IncreFA call. Eighteen calls predict only about 9.5 us of saving in
a roughly 1.6 ms decoder step. The compiled full graph instead regresses by
about 28--31 us. This indicates a graph/runtime scheduling interaction rather
than slower IncreFA arithmetic. This run does not identify whether that
interaction is occupancy, hard-sync behavior, or another stream-scheduling
effect.

## Direct production-GQA probe

The mixed custom package completed the real B1 `combined_apply` GQA generation
with exact output. Its 374-step model-plus-argmax device time was `0.549554 s`,
or `678.7` effective device tok/s. This is the metric corresponding to the
roughly 700--800 tok/s production-latency range. The first cold run also spent
`0.429165 s` in slot admission, so its total scheduler-window throughput of
`376.6` tok/s must not be used as steady decode throughput.

The zero-cube GQA graph compiled successfully. The `.om`, compiled module, and
index files were complete by `09:20:23`; they did not change during the
following ten minutes. Execution did not return. At inspection time, the main
Python thread used about 96.5% CPU and the `RT_RECYCLE_6` runtime thread used
about 54.4% CPU. The process was interrupted after more than ten minutes, all
workers exited, and physical NPU 6 returned to 0% AIC/AIV utilization.

This was not a 13-minute compilation. It was a post-compile runtime hang. There
is no valid zero-cube latency number for the production GQA path.

## Commands

After `source npu-setup`, real generation used:

```sh
env ASCEND_CUSTOM_OPP_PATH="<selected-package>:${ASCEND_CUSTOM_OPP_PATH}" \
  LD_LIBRARY_PATH="<selected-package>/op_api/lib/:${LD_LIBRARY_PATH}" \
  /workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_real_generation.py \
  --decode-cache-dir <implementation-cache> \
  --decode-optimization combined_apply_mha_cache \
  --batch-size 1 \
  --replicas 1 \
  --reference <mixed-warm-result> \
  --output <per-process-result>
```

The steady profile used the same package and cache selection with:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/text_decode_lab.py \
  --mode profile \
  --batch-size 1 \
  --cache-length 4096 \
  --profile-position 1279 \
  --decode-optimization combined_apply_mha_cache \
  --cache-dir <implementation-cache> \
  --warmup 200 \
  --repeats 2000 \
  --allow-compile \
  --output <per-process-result>
```

## Conclusion

Do not adopt the current zero-cube package as a B1 latency optimization. It
regresses the completed MHA full-forward lane and hangs in the direct production
GQA lane. The next production-relevant step is to capture the stock B1 GQA
tiling/runtime contract and build a GQA-specific zero-cube package rather than
forcing the MHA-validated fixed-key change onto that call. A focused graph
profile is also needed to explain why the MHA full graph loses the isolated
operator saving.
