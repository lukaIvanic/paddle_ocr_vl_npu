# OCR resolution ablation

- Reference: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_full_maxpixels401408_c41c3d8_r2/output`
- Candidate: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_textscale_half_full_844c4e7/output`
- Shared crops: **30557**
- Affected crops: **28125 across 1645 pages**
- Affected because the vision grid changed: **17186**
- Affected because the text-crop pixel input changed: **28125**
- Real vision tokens: **13,571,704 -> 6,977,788** (6,593,916 saved)
- Physical vision tokens: **15,050,588 -> 7,593,688** (7,456,900 saved)
- Token streams exact among affected crops: **21073/28125**
- Whitespace-insensitive text exact: **22630/28125**
- Automatically flagged for manual review: **268**
- Unaffected crops whose generation changed: **13**

## Runtime

- Reference pipeline: **956.130s**
- Candidate pipeline: **879.925s**
- Delta: **-76.205s (-7.97%)**

## Affected crops by recognizer route

| Route | Crops | Real vision tokens | Physical vision tokens | Token exact | Compact-text exact | Flagged | Mean normalized character edit |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 28125 | 13,571,704 -> 6,977,788 | 15,050,588 -> 7,593,688 | 21073 | 22630 | 268 | 0.0212 |

## All crops by recognizer route

| Route | Crops | Real vision tokens | Physical vision tokens |
|---|---:|---:|---:|
| formula | 1681 | 632,228 -> 632,228 | 684,680 -> 654,660 |
| table | 751 | 1,041,308 -> 1,041,308 | 1,073,692 -> 1,070,436 |
| text | 28125 | 13,571,704 -> 6,977,788 | 15,050,588 -> 7,593,688 |

## Automatic review flags

- `candidate_collapsed_length`: 7
- `candidate_lost_eos`: 57
- `candidate_repetition_regression`: 59
- `candidate_runaway_length`: 57
- `large_compact_text_edit`: 268

## Worst affected-crop differences

| Request | Label | Vision tokens | Output tokens | Character edit | Flags |
|---|---|---:|---:|---:|---|
| `page_000179_block_000007` | text | 216 -> 216 | 24 -> 4030 | 1.0000 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000609_block_000008` | text | 160 -> 160 | 46 -> 4044 | 1.0000 | candidate_runaway_length, candidate_lost_eos, large_compact_text_edit |
| `page_000146_block_000000` | text | 1920 -> 588 | 4 -> 3937 | 1.0000 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001518_block_000001` | text | 1640 -> 492 | 3674 -> 26 | 1.0000 | candidate_collapsed_length, large_compact_text_edit |
| `page_001518_block_000007` | text | 1200 -> 360 | 28 -> 2264 | 1.0000 | candidate_runaway_length, candidate_repetition_regression, large_compact_text_edit |
| `page_000257_block_000045` | text | 760 -> 188 | 218 -> 16 | 1.0000 | large_compact_text_edit |
| `page_000363_block_000006` | text | 208 -> 196 | 54 -> 1 | 1.0000 | large_compact_text_edit |
| `page_000363_block_000003` | text | 204 -> 204 | 52 -> 1 | 1.0000 | large_compact_text_edit |
| `page_000363_block_000011` | text | 280 -> 312 | 45 -> 5 | 1.0000 | large_compact_text_edit |
| `page_000179_block_000000` | text | 144 -> 192 | 4 -> 37 | 1.0000 | large_compact_text_edit |
| `page_000363_block_000016` | text | 312 -> 352 | 32 -> 1 | 1.0000 | large_compact_text_edit |
| `page_000377_block_000001` | text | 720 -> 160 | 2 -> 22 | 1.0000 | large_compact_text_edit |
| `page_000399_block_000001` | text | 860 -> 252 | 11 -> 29 | 1.0000 | large_compact_text_edit |
| `page_000151_block_000000` | text | 168 -> 168 | 2 -> 9 | 1.0000 | large_compact_text_edit |
| `page_000111_block_000008` | text | 168 -> 168 | 5 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000326_block_000003` | text | 168 -> 168 | 2 -> 5 | 1.0000 | large_compact_text_edit |
| `page_000361_block_000000` | text | 576 -> 144 | 9 -> 6 | 1.0000 | large_compact_text_edit |
| `page_000363_block_000005` | text | 160 -> 152 | 8 -> 5 | 1.0000 | large_compact_text_edit |
| `page_000889_block_000018` | text | 168 -> 168 | 3 -> 6 | 1.0000 | large_compact_text_edit |
| `page_000110_block_000021` | text | 204 -> 204 | 2 -> 4 | 1.0000 | large_compact_text_edit |
| `page_000772_block_000003` | text | 180 -> 180 | 6 -> 8 | 1.0000 | large_compact_text_edit |
| `page_000807_block_000038` | text | 144 -> 144 | 3 -> 5 | 1.0000 | large_compact_text_edit |
| `page_000863_block_000019` | text | 192 -> 192 | 2 -> 4 | 1.0000 | large_compact_text_edit |
| `page_000863_block_000017` | text | 192 -> 192 | 4 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000330_block_000014` | text | 264 -> 272 | 1 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000363_block_000013` | text | 192 -> 192 | 8 -> 7 | 1.0000 | large_compact_text_edit |
| `page_000400_block_000000` | text | 576 -> 144 | 3 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000714_block_000012` | text | 160 -> 160 | 3 -> 4 | 1.0000 | large_compact_text_edit |
| `page_000807_block_000019` | text | 168 -> 168 | 3 -> 2 | 1.0000 | large_compact_text_edit |
| `page_001106_block_000078` | text | 160 -> 160 | 2 -> 3 | 1.0000 | large_compact_text_edit |
| `page_001363_block_000000` | text | 168 -> 168 | 5 -> 4 | 1.0000 | large_compact_text_edit |
| `page_000151_block_000001` | text | 152 -> 152 | 11 -> 11 | 1.0000 | large_compact_text_edit |
| `page_000193_block_000001` | text | 192 -> 168 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000267_block_000002` | text | 168 -> 168 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000595_block_000010` | text | 168 -> 168 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000656_block_000009` | text | 168 -> 168 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000661_block_000009` | text | 168 -> 168 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000665_block_000008` | text | 168 -> 168 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000714_block_000010` | text | 168 -> 168 | 3 -> 3 | 1.0000 | large_compact_text_edit |
| `page_000714_block_000031` | text | 144 -> 144 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_001269_block_000003` | text | 168 -> 168 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_001269_block_000007` | text | 168 -> 168 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_001407_block_000023` | text | 168 -> 144 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_001414_block_000039` | text | 168 -> 144 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_001417_block_000005` | text | 192 -> 192 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_001417_block_000016` | text | 168 -> 168 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_001418_block_000027` | text | 168 -> 144 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_001419_block_000028` | text | 168 -> 144 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000363_block_000008` | text | 204 -> 192 | 51 -> 4036 | 0.9998 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000179_block_000002` | text | 320 -> 192 | 4004 -> 4036 | 0.9993 | large_compact_text_edit |

Full texts and exact per-crop metrics are in `per_crop.jsonl` and `manual_review.csv`.
Official OmniDocBench metrics are intentionally reported separately because they operate on matched page elements, not raw recognition crops.
