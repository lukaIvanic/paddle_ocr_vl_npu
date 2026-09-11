# Actual generated-ID inventory, 2026-09-11

This supersedes reference-retokenization counts as evidence of observed model
generation. No tokenizer encode/decode operation was used to count these IDs.
The active vocabulary is unchanged; no NPU work was needed.

## Recovered original 16k provenance

`table_decode_vocab_topfreq_16384_20260812.json` identifies
`table_spec_full_d1e6d00/whole/row_ocr_records.jsonl` as its source. Recounting that
file gives 665 whole-table records, 268,080 native token occurrences and 13,092
unique IDs. All are inside the current 16,384 selection (digest
`9c48e5c3b92776ba250f75359fccb407448c4da8419fe927f5ea381d345712c3`).
658 records stopped at EOS; seven at the KV limit.

The separate U8-draft-inclusive metadata reports 14,339 unique IDs from 1,330
table records (whole tables plus their split-draft records). The original active
16k selection was therefore table-specific, not an all-page-text inventory.

## Two complete page-pipeline runs

| Source commit | Text crops / unique IDs | Formula crops / unique IDs | Table crops / unique IDs | Union | Generated occurrences |
|---|---:|---:|---:|---:|---:|
| 8634d3a | 28,125 / 46,452 | 1,681 / 1,388 | 751 / 13,431 | 48,158 | 1,686,742 |
| d9197ea | 28,125 / 46,336 | 1,681 / 1,397 | 751 / 13,358 | 48,047 | 1,642,116 |

Category vocabularies overlap; do not sum their unique counts.
Both summaries report 1,651 page results/predictions and 30,557 recognition
requests. Recognition IDs are present on 1,649 page indices; two pages have no
recognition requests: page index 1478 has only an image region and 1483 has no
detected regions, verified by matching source image names. These are layout-detected crops, not the exact annotated
665-table set or every ground-truth region. No record is missing its token_ids.

The first run has 30,534 EOS and 23 KV-cap completions. The second has 30,475 EOS,
69 repetition stops and 13 KV-cap completions. Counts include saved tokens up to
those stops, not hypothetical continuations. All records are retained.

The first run's committed command requests offset 0, limit 1651, the exact
PaddleOCR-VL-1.6 checkpoint, FP16, TorchAir and KV4096. Both historical
`text_decode.py` implementations directly project through `model.lm_head` at
line 1015, without compact selection. The compact-head implementation was added
later in 13bfb740 (2026-08-12). The completion path passes
`completion.token_ids` through to the recorded result, not re-encoded text.

These older runs use different preprocessing/scheduling than current experiment
19. Their IDs are evidence of full-head historical generation, not proof that a
trimmed current engine reproduces all outputs or every future crop.

## Observed unions and budget implications

- Two full-page runs: **48,764** distinct generated IDs.
- Plus the original 665-table source: **48,784**.
- Plus every current 16k selection ID: **49,885** (1,315 slots left at 51,200).
- Plus every non-special Han-containing vocabulary entry: **56,160**.
- Plus the previously defined character/grapheme protections: **59,537**.

Thus preserving the inspected raw generation IDs together with the current
selection fits 51,200. Also preserving every Han and core character entry does
not. The latter is an explicit broader-coverage policy, not a claim that those
extra tokens were emitted in these runs.

## Files

- `native_generation_counts.json`: source SHA256s, per-category native-ID
  frequencies, sorted unique IDs, completion counts and run-summary metadata.
- `collect_native.py`: reproduces the read-only remote aggregation. No model,
  tokenizer, annotations or OCR text are inputs to the counting operation.
- Full remote trace locations are recorded in the JSON. Counts were transferred
  instead of duplicating the roughly 100 MB trace per run locally.

For context inspection the user subsequently authorized retrieving full traces
including OCR text; that is not required to reproduce these counts.
