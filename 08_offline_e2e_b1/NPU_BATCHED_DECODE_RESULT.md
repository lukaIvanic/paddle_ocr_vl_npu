# 910B padded B=2 and B=4 decode validation

Observed on 2026-07-16 from commit `e12ac61`, using one logical `npu:0` on a
910B server. The input and model configuration match
[`NPU_FULL_PAGE_RESULT.md`](NPU_FULL_PAGE_RESULT.md), except that compiled
decode used fixed batch sizes of two and four.

The page has five real layout regions. At B=2 they formed `2 + 2 + (1 real + 1
dummy)` decode cohorts. At B=4 they formed `4 + (1 real + 3 dummy)` cohorts.
Vision and text prefill remained eager, native-resolution, and sequential B=1.
Only decode was padded and batched.

## Correctness

Both runs completed successfully, and all five requests stopped at EOS. Their
recognized strings and token-ID sequences were exactly equal to the earlier
B=1 full-page result: 7, 14, 42, 15, and 3 generated tokens including EOS.

## Decode accounting

| Decode batch | Graph calls | Raw slots | Effective slots | Finished-row padding | Final-cohort padding | Utilization | Decode wall | Raw tok/s | Effective tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 59 | 118 | 76 | 39 | 3 | 64.407% | 0.187081 s | 630.742 | 406.241 |
| 4 | 45 | 180 | 76 | 95 | 9 | 42.222% | 0.146848 s | 1225.754 | 517.541 |

Raw tok/s counts every fixed-batch slot executed by the compiled graph.
Effective tok/s counts only real post-prefill generated tokens, including EOS.
Both use the same aggregate compiled-decode wall time. Finished-row padding is
the work done after a shorter real row has emitted EOS; final-cohort padding is
the work done for dummy rows added to complete the fixed batch.

## B=1 comparison

The original B=1 page took 0.208081 s in compiled decode and sustained 365.242
effective tok/s. B=2 reduced decode wall time by 10.1% and raised effective
throughput by 11.2%. B=4 reduced decode wall time by 29.4% and raised effective
throughput by 41.7%, reaching 517.541 effective tok/s despite executing more
padded than useful slots.

Full page inference was 1.872002 s at B=1, 1.823523 s at B=2, and 1.773630 s at
B=4. E2E output throughput was respectively 43.269, 44.420, and 45.669 tok/s.
The page-level gain is smaller because real layout and sequential prefill are
unchanged.

These are functional validations, not yet stable throughput benchmarks: there
is only one page, the region lengths are heterogeneous, layout timing varied
slightly between runs, and cohorts currently follow reading order rather than
length-aware scheduling.

The complete results remain in the Blue Zone checkout:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/08_offline_e2e_b1/full_page_b2/run.json
/workspace/repos/paddle_ocr_vl_npu/tmp/08_offline_e2e_b1/full_page_b4/run.json
```
