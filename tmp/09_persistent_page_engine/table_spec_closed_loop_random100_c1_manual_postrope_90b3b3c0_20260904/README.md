# Random-100 speculative API with the manual verifier

This Ascend 910B2 run repeats the random-100, seed-1, concurrency-one workload
from `table_spec_closed_loop_random100_c1_bf8f7673_20260904`. The request order
and height routing are identical. The server command explicitly selects:

```text
SPEC_VERIFY_ATTENTION=manual_grouped_legal_scaled_masked_softmax_fp16_combined_qkv_post_rope
```

The server also uses `--allow-compile`. Missing graphs compile before `/ready`
becomes available, so compilation is outside the measured request window.

| Metric | PromptFA default | Manual combined post-RoPE | Change |
|---|---:|---:|---:|
| QPS | 1.5744 | 1.6865 | +7.12% |
| Mean latency | 0.6343 s | 0.5920 s | -6.66% |
| P50 | 0.5315 s | 0.5307 s | -0.13% |
| P90 | 1.2301 s | 1.1143 s | -9.41% |
| P95 | 1.4240 s | 1.2819 s | -9.98% |
| P99 | 1.6193 s | 1.5227 s | -5.96% |
| Maximum | 3.3644 s | 2.5227 s | -25.02% |

The verifier wall total fell from 14.1092 s to 9.7391 s. Its physical verifier
throughput rose from 4,597.3 to 6,648.8 query positions per verifier-wall
second. The verifier wall cost per target call fell from 2.7983 ms to 1.9366 ms.

Both runs have 99 of 100 tables token-identical to the saved B1 reference. The
same table, `page_000263_table_box_id_7`, differs from the saved reference in
both runs. Its PromptFA and manual outputs also differ from each other, starting
at native token position 1,000. No generated text was encoded for this check.
