# MinerU optimization rung 3: CPU-side MRoPE preparation

## Result

Accepted. The existing CPU preparation worker now constructs MRoPE position
IDs and deltas. Text prefill consumes those tensors directly instead of running
the Python-driven MRoPE procedure on the NPU inference thread.

| Metric | Global stream baseline | CPU MRoPE | Change |
|---|---:|---:|---:|
| Pipeline wall | 199.833 s | 184.545 s | -15.288 s (-7.7%) |
| Pages/s | 0.6405 | 0.6936 | +8.3% |
| Generation wall | 174.582 s | 159.812 s | -14.771 s |
| Prefill wall | 88.400 s | 78.585 s | -9.815 s |
| Decode wall | 40.122 s | 40.010 s | -0.112 s |
| NPU MRoPE stage | 6.913 s | 0.000 s | -6.913 s |
| CPU MRoPE work | not recorded | 2.735 s | background worker |
| CPU prepare wait | 1.393 s | 1.511 s | +0.118 s |

Both runs used the first 128 OmniDocBench pages, one global request stream,
B32, KV4096, compiled PromptFA vision prefill, packed compiled text prefill,
compiled IncreFA decode, NZ decode weights, and warm graph caches.

## Accuracy

All 128 generated Markdown files were byte-identical to the accepted global
request-stream baseline. The 32-page gate was also byte-identical for all 32
pages.

The CPU implementation calls the same `get_rope_index` method on the same token
IDs, attention mask, and image-grid metadata before the request is transferred.
The NPU prefill path retains its prior calculation only as a compatibility
fallback for callers that do not supply the precomputed tensors.

## Artifacts

- 32-page gate: `tmp/11_mineru_2_5_pro_inference/opt3_cpu_mrope_n32_501cc02/`
- 128-page run: `tmp/11_mineru_2_5_pro_inference/opt3_cpu_mrope_n128_501cc02/`
- Baseline: `tmp/11_mineru_2_5_pro_inference/opt2_global_stream_n128_1da43c9/`
