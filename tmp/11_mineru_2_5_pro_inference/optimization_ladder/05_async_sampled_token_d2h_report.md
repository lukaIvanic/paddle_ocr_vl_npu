# MinerU optimization rung 5: asynchronous sampled-token D2H

## Result

Accepted. Decode now copies sampled token IDs through a dedicated NPU stream
and a two-entry pinned CPU ring. The scheduler launches iteration N+1 before it
waits for iteration N's token IDs. Slot generations discard the speculative
token if a request finishes and its slot is reused.

| Metric | CPU-MRoPE baseline | Async token D2H | Change |
|---|---:|---:|---:|
| Pipeline wall | 184.545 s | 170.948 s | -13.597 s (-7.4%) |
| Pages/s | 0.6944 | 0.7496 | +8.0% |
| Generation wall | 159.812 s | 146.426 s | -13.386 s |
| Prefill wall | 78.585 s | 82.056 s | +3.471 s |
| Compiled decode plus argmax device time | not directly comparable | 38.777 s | event-scoped metric |
| Token-copy submission | not recorded | 2.173 s | host submission |
| Sampled-token D2H wait | blocking path included elsewhere | 5.659 s | one-iteration pipeline |
| Hot-swap safety synchronization | not recorded | 4.638 s | only before refilling completed slots |

Both runs used the first 128 OmniDocBench pages, one global request stream,
B32, KV4096, CPU MRoPE, compiled PromptFA vision prefill, packed compiled text
prefill, compiled IncreFA decode, NZ decode weights, and warm graph caches.

The new `decode_s` metric is device-event time for compiled decode plus argmax.
The old metric included different host synchronization semantics, so pipeline
and generation wall time are the authoritative paired speed measurements.

## Accuracy

- 32-page gate: 32/32 Markdown files byte-identical.
- 128-page run: 126/128 Markdown files byte-identical.
- One changed page reorders two adjacent Chinese annotation fragments.
- One changed page changes only LaTeX array-column syntax and spacing.
- The decode-token count changed by 5 tokens out of about 249,000.
- No content was missing and no output degenerated.

## Artifacts

- 32-page gate: `tmp/11_mineru_2_5_pro_inference/opt5_async_token_d2h_n32_d9c9c82/`
- 128-page run: `tmp/11_mineru_2_5_pro_inference/opt5_async_token_d2h_n128_d9c9c82/`
- Baseline: `tmp/11_mineru_2_5_pro_inference/opt3_cpu_mrope_n128_501cc02/`
