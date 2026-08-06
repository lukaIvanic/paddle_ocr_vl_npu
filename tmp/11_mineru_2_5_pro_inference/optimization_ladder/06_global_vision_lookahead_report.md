# MinerU optimization rung 6: global vision lookahead packing

## Result

Accepted. The continuous engine prepares vision embeddings in FIFO windows of
32 requests before text prefill and decode-slot admission. The vision router can
therefore pack across the prepared-request stream instead of seeing only the
slots that finish in one decode iteration. A CPU prefetch depth of 64 keeps two
vision windows available and prevents producer starvation.

| Metric | Async-D2H baseline | Global vision lookahead | Change |
|---|---:|---:|---:|
| Pipeline wall | 170.948 s | 158.410 s | -12.539 s (-7.3%) |
| Pages/s | 0.7496 | 0.8090 | +7.9% |
| Generation wall | 146.426 s | 133.567 s | -12.859 s |
| Prefill wall | 82.056 s | 75.546 s | -6.510 s |
| Vision transformer | 45.552 s | 33.423 s | -12.129 s (-26.6%) |
| Real vision tokens | 1,689,440 | 1,689,440 | unchanged |
| Physical vision tokens | 2,117,816 | 1,842,872 | -274,944 (-13.0%) |
| Useful vision-token fraction | 79.77% | 91.67% | +11.90 points |
| CPU prepare wait | 1.392 s | 1.705 s | +0.312 s |

Both runs used the first 128 OmniDocBench pages, one global request stream,
B32, KV4096, CPU MRoPE, asynchronous sampled-token D2H, compiled PromptFA
vision prefill, packed compiled text prefill, compiled IncreFA decode, NZ
decode weights, and warm graph caches.

## Producer-size gate

The first 32-page test retained the old CPU prefetch depth of 16. Vision
transformer time fell by 24%, but CPU wait rose from 1.48 s to 5.33 s and the
pipeline became 4.1% slower. Increasing CPU prefetch depth to 64 retained the
same 91.3% packing efficiency and reduced the 32-page pipeline from 39.67 s to
38.22 s. The default is therefore 64.

## Accuracy

- 32-page gate: 31/32 Markdown files byte-identical.
- 128-page run: 125/128 Markdown files byte-identical.
- Two changes are local LaTeX grouping or line-layout alternatives.
- One change substitutes one Chinese character in a short OCR line.
- No page lost content and no output degenerated.

## Artifacts

- Starved 32-page gate: `tmp/11_mineru_2_5_pro_inference/opt6_global_vision_n32_c6a82a1/`
- Corrected 32-page gate: `tmp/11_mineru_2_5_pro_inference/opt6_global_vision_n32_prefetch64_c6a82a1/`
- 128-page run: `tmp/11_mineru_2_5_pro_inference/opt6_global_vision_n128_b4cf510/`
- Baseline: `tmp/11_mineru_2_5_pro_inference/opt5_async_token_d2h_n128_d9c9c82/`
