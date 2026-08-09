# B1 greedy suppression of `\(` token ID 47536

Date: 2026-08-09  
Device: Ascend 910B2, physical NPU 6  
Candidate commit: `d3afb95`  
Cohort: all 665 OmniDocBench tables on 458 pages  
Dataset fingerprint: matched the saved 1,651-image benchmark exactly

## Result

Completely suppressing token ID 47536 (`\(`) makes the full-benchmark score
slightly worse.

| B1 policy | Page-TEDS | Sample TEDS | Structure-only Page-TEDS |
|---|---:|---:|---:|
| Saved ordinary greedy B1 (`04fbc8e`) | 0.954554208 | 0.949733383 | 0.978143202 |
| Greedy except token 47536 (`d3afb95`) | 0.953596303 | 0.949439302 | 0.977985807 |
| Candidate minus saved B1 | **-0.000957905** | **-0.000294081** | **-0.000157395** |

There were zero TEDS timeouts and zero scoring errors.

The saved baseline is the established run from commit `04fbc8e`, not a new
same-commit control. The candidate's absolute score is definitive. The delta
assumes no relevant ordinary-B1 drift between those commits. That assumption is
supported, but not proven, by 577/665 normalized prediction strings remaining
identical.

## Policy

`suppress_math_open_greedy` operates directly on native logits and token IDs.
For table prompts only:

1. Compute ordinary greedy argmax.
2. If argmax is not token ID 47536, retain it unchanged.
3. If argmax is token ID 47536, select the highest-scoring other token from
   the same logits.

The policy runs on the first token produced by prefill and every later decode
step. It does not decode and re-encode generated text.

The direct-logit unit suite passed 16/16 tests. A formula-heavy native-ID smoke
on `page_000271_table_box_id_1` generated 3,094 tokens and contained zero
instances of token ID 47536.

The established all-table HTTP client does not persist `token_ids` in its
benchmark JSONL, even though the server response contains them. Therefore the
all-665 native-ID count cannot be reconstructed after this run. The same tested
selection path was active for every table request.

## Per-table effects

- Prediction text changed on 88/665 tables and was identical on 577/665.
- TEDS improved on 37 tables, was equal on 592, and regressed on 36.
- Structure-only TEDS improved on 4 tables, was equal on 657, and regressed on 4.
- Mean per-table TEDS delta was -0.000294081; median delta was 0.

Largest improvements:

| Request | Saved B1 TEDS | Suppressed TEDS | Delta |
|---|---:|---:|---:|
| `page_000292_table_box_id_2` | 0.583030 | 0.921137 | +0.338107 |
| `page_000288_table_box_id_1` | 0.734144 | 0.984235 | +0.250090 |
| `page_000785_table_7` | 0.752066 | 0.979339 | +0.227273 |
| `page_000626_table_box_id_4` | 0.861364 | 0.996622 | +0.135258 |
| `page_000271_table_box_id_1` | 0.774445 | 0.874224 | +0.099779 |

Largest regressions:

| Request | Saved B1 TEDS | Suppressed TEDS | Delta |
|---|---:|---:|---:|
| `page_000206_table_box_id_9` | 0.996478 | 0.594635 | -0.401843 |
| `page_000631_table_0` | 0.929711 | 0.560838 | -0.368873 |
| `page_000259_table_box_id_1` | 1.000000 | 0.725158 | -0.274842 |
| `page_000273_table_box_id_1` | 0.987302 | 0.829412 | -0.157890 |
| `page_001215_table_5` | 0.992804 | 0.859177 | -0.133627 |

## Generation behavior

| Metric | Saved ordinary B1 | Suppressed 47536 |
|---|---:|---:|
| Generation wall | 592.684 s | 672.829 s |
| Mean HTTP latency | 0.737 s | 0.858 s |
| P90 HTTP latency | 1.612 s | 1.958 s |
| P99 HTTP latency | 4.786 s | 5.537 s |
| Output tokens including EOS | 268,409 | 266,706 |
| EOS / KV-full stops | 657 / 8 | 658 / 7 |

The latency result is not a production-quality cost measurement of token
suppression. This experimental policy uses top-2 selection at every step to
identify the best non-47536 token. It adds selection overhead even when 47536
would not have won.

## Decision

Do not globally suppress token ID 47536 in B1. The policy helps several of the
known delimiter-heavy tables, including `page_000271`, but creates comparably
large new failures elsewhere and slightly reduces aggregate Page-TEDS.
