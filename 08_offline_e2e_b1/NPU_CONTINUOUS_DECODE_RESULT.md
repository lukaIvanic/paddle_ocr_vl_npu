# 910B continuous decode validation

Observed on 2026-07-16 from commit `d144ae3`, using one logical `npu:0` on a
910B server. The input, models, cache length, output limit, and compiled decode
graph match the historical B=2/B=4 fixed-cohort runs in
[`NPU_BATCHED_DECODE_RESULT.md`](NPU_BATCHED_DECODE_RESULT.md).

Sequential B=1 vision/text prefill produced five ready KV states before decode.
The continuous scheduler then kept one persistent compiled arena, retired
requests from individual slots, and copied the next ready request's valid KV
prefix into each freed slot. It never rebuilt the arena or moved another active
request.

## Correctness

Both B=2 and B=4 passed the full-page smoke validator. All five requests stopped
at EOS. Their strings and complete token-ID sequences were exactly equal to the
historical fixed-cohort and B=1 results: 7, 14, 42, 15, and 3 generated tokens
including EOS.

B=2 exercised three real replacements:

```text
slot 0: region 1 epoch 1 -> region 3 epoch 2
slot 1: region 2 epoch 1 -> region 4 epoch 2 -> region 5 epoch 3
```

B=4 exercised one replacement while the 42-token region remained active:

```text
slot 0: region 1 epoch 1 -> region 5 epoch 2
```

Every request launched exactly one delayed look-ahead iteration. Epoch checks
discarded its stale sampled token when the slot had already been reused.

## Results

| Schedule | B | Graph calls | Raw slots | Effective | Idle | Look-ahead | Decode wall | Raw tok/s | Effective tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed cohorts | 2 | 59 | 118 | 76 | n/a | n/a | 0.187081 s | 630.742 | 406.241 |
| Continuous | 2 | 49 | 98 | 76 | 17 | 5 | 0.159275 s | 615.288 | 477.162 |
| Fixed cohorts | 4 | 45 | 180 | 76 | n/a | n/a | 0.146848 s | 1225.754 | 517.541 |
| Continuous | 4 | 42 | 168 | 76 | 87 | 5 | 0.145588 s | 1153.938 | 522.019 |

At B=2, iteration-level replacement removed ten graph calls, reduced the
decode wall by 14.9%, and increased effective throughput by 17.5%. At B=4 it
removed the separate three-call final cohort; the long request already
determined almost the entire decode window, so effective throughput improved by
0.9%.

The lower B=4 raw tok/s is not a useful-token regression. Continuous decode
executes fewer raw slots, while its denominator honestly includes initial and
hot-swap KV admission, D2H waits, and host retirement. Effective tok/s is the
relevant comparison.

Active-slot utilization was 82.653% at B=2 and 48.214% at B=4. Effective-slot
utilization after excluding one look-ahead per request was respectively 77.551%
and 45.238%. B=4 remains faster because its compiled model iterations have
higher device throughput despite more idle slots.

## Timing and traffic

All five valid prompt prefixes totalled 35,112,960 copied KV bytes. At B=2,
8,441,856 bytes entered the two initial slots and 26,671,104 bytes were copied
during three hot swaps. At B=4, 32,034,816 bytes entered initially and only
3,078,144 bytes belonged to the one hot swap.

The B=2 scheduler reported 0.147848 s of decode-model-plus-argmax device time,
0.006481 s of slot-admission device time, and 0.006637 s in host retirement and
refill. B=4 reported 0.129444 s, 0.009828 s, and 0.003715 s respectively. These
event and phase measurements overlap, so they must not be summed into wall time.

Full page time varied between runs because layout and eager prefill are still
sequential and dominate more of the page. The continuous scheduler currently
operates within one prepared page; it does not yet overlap layout, prefill, and
decode or batch requests across page boundaries.

The complete results remain in the Blue Zone checkout:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/08_offline_e2e_b1/full_page_continuous_b2/run.json
/workspace/repos/paddle_ocr_vl_npu/tmp/08_offline_e2e_b1/full_page_continuous_b4/run.json
```
