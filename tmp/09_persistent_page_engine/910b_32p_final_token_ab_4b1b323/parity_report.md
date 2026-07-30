# 910B2 32-page packed final-token selection parity

Commit: `4b1b323`

Workload:

- OmniDocBench pages 0-31
- 32 pages, 510 recognition requests
- layout-first frontend
- decode batch size 32
- KV length 4096
- reduced `min_pixels=28224`
- compiled PromptFA vision prefill with greedy packing
- packed compiled text prefill
- production compiled continuous decode

The production lane used slice-and-concatenate final-token selection. The
reference lane used the former `torch.index_select` implementation after the
same packed compiled text-prefill graph. No other model, graph, routing,
packing, cache, admission, or decode code differed.

## Exact comparisons

| Comparison | Result |
| --- | --- |
| 510 recognition records, sorted by `request_id` | exact |
| token IDs | exact |
| decoded text | exact |
| stop reasons | exact |
| generated/decode/input/vision token counts per request | exact |
| normalized recognition SHA-256 | `0a8e24f1aec10212036214bd24a355676311a4e3772efb308436b5ea9e0f5fc3` for both lanes |
| 32 prediction files | exact |
| page-region JSONL | byte-identical |
| vision route plan | byte-identical |
| page completion order | exact |

## Runtime and scheduler accounting

| Metric | Slice-concat | Index-select |
| --- | ---: | ---: |
| Pipeline E2E | 19.628137 s | 19.631000 s |
| Pages/s | 1.630313 | 1.630075 |
| Requests | 510 | 510 |
| Generated tokens including EOS | 26,766 | 26,766 |
| Effective decode tokens | 26,256 | 26,256 |
| Decode graph calls | 1,341 | 1,341 |
| Raw decode slots | 42,912 | 42,912 |

The B32 arena processed 510 requests, so the run necessarily admitted 478
requests after the initial 32 slots. This validates the replacement under
repeated decode-slot reuse rather than only a single static cohort.

Both lanes' private prefill-cache pools also matched:

- capacity: 160
- acquisitions: 510
- reuses: 350
- releases: 510
- high-water active caches: 132
- final active caches: 0

Conclusion: slice-and-concatenate is exactly output-equivalent to the former
final-token `index_select` on this 32-page, slot-reusing 910B2 E2E workload.
