# OCR resolution ablation

- Reference: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_full_maxpixels401408_c41c3d8_r2/output`
- Candidate: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_textcap384_full_844c4e7/output`
- Shared crops: **30557**
- Affected crops: **9789 across 1413 pages**
- Affected because the vision grid changed: **9789**
- Affected because the text-crop pixel input changed: **0**
- Real vision tokens: **9,795,840 -> 3,235,620** (6,560,220 saved)
- Physical vision tokens: **10,564,168 -> 3,489,480** (7,074,688 saved)
- Token streams exact among affected crops: **6226/9789**
- Whitespace-insensitive text exact: **6949/9789**
- Automatically flagged for manual review: **154**
- Unaffected crops whose generation changed: **50**

## Runtime

- Reference pipeline: **956.130s**
- Candidate pipeline: **814.272s**
- Delta: **-141.858s (-14.84%)**

## Affected crops by recognizer route

| Route | Crops | Real vision tokens | Physical vision tokens | Token exact | Compact-text exact | Flagged | Mean normalized character edit |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 9789 | 9,795,840 -> 3,235,620 | 10,564,168 -> 3,489,480 | 6226 | 6949 | 154 | 0.0292 |

## All crops by recognizer route

| Route | Crops | Real vision tokens | Physical vision tokens |
|---|---:|---:|---:|
| formula | 1681 | 632,228 -> 632,228 | 684,680 -> 657,596 |
| table | 751 | 1,041,308 -> 1,041,308 | 1,073,692 -> 1,071,436 |
| text | 28125 | 13,571,704 -> 7,011,484 | 15,050,588 -> 7,660,024 |

## Automatic review flags

- `candidate_collapsed_length`: 5
- `candidate_lost_eos`: 99
- `candidate_repetition_regression`: 102
- `candidate_runaway_length`: 98
- `large_compact_text_edit`: 153

## Worst affected-crop differences

| Request | Label | Vision tokens | Output tokens | Character edit | Flags |
|---|---|---:|---:|---:|---|
| `page_000146_block_000000` | text | 1920 -> 340 | 4 -> 3999 | 1.0000 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000983_block_000001` | text | 1900 -> 352 | 3609 -> 1 | 1.0000 | candidate_collapsed_length, large_compact_text_edit |
| `page_000257_block_000045` | text | 760 -> 260 | 218 -> 16 | 1.0000 | large_compact_text_edit |
| `page_001521_block_000016` | text | 1240 -> 264 | 12 -> 21 | 1.0000 | large_compact_text_edit |
| `page_000361_block_000000` | text | 576 -> 304 | 9 -> 6 | 1.0000 | large_compact_text_edit |
| `page_000377_block_000001` | text | 720 -> 336 | 2 -> 4 | 1.0000 | large_compact_text_edit |
| `page_001518_block_000001` | text | 1640 -> 304 | 3674 -> 14 | 0.9997 | candidate_collapsed_length, large_compact_text_edit |
| `page_000810_block_000001` | text | 1924 -> 320 | 507 -> 4004 | 0.9992 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000196_block_000009` | text | 1500 -> 336 | 3709 -> 4 | 0.9976 | candidate_collapsed_length, large_compact_text_edit |
| `page_000014_block_000007` | text | 888 -> 288 | 3862 -> 125 | 0.9974 | candidate_collapsed_length, large_compact_text_edit |
| `page_000452_block_000009` | text | 1968 -> 360 | 767 -> 3994 | 0.9922 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001081_block_000070` | text | 1976 -> 320 | 611 -> 4004 | 0.9897 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000842_block_000005` | text | 1972 -> 336 | 380 -> 4000 | 0.9888 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001169_block_000007` | text | 1976 -> 352 | 479 -> 3996 | 0.9881 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001227_block_000008` | text | 1976 -> 352 | 278 -> 3996 | 0.9863 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001013_block_000061` | text | 1980 -> 336 | 592 -> 4000 | 0.9860 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000530_block_000006` | text | 460 -> 320 | 3 -> 41 | 0.9848 | large_compact_text_edit |
| `page_000480_block_000003` | text | 1932 -> 360 | 227 -> 3994 | 0.9828 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000005` | text | 1716 -> 360 | 869 -> 3994 | 0.9826 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001095_block_000080` | text | 1904 -> 360 | 481 -> 3994 | 0.9805 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001155_block_000009` | text | 1924 -> 320 | 342 -> 4004 | 0.9766 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000156_block_000005` | text | 1904 -> 336 | 248 -> 4000 | 0.9758 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001595_block_000005` | text | 2000 -> 216 | 121 -> 4030 | 0.9737 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000646_block_000003` | text | 1944 -> 352 | 378 -> 3996 | 0.9714 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001031_block_000011` | text | 1980 -> 336 | 291 -> 4000 | 0.9698 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001170_block_000005` | text | 1984 -> 364 | 389 -> 3993 | 0.9697 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001014_block_000026` | text | 1920 -> 312 | 301 -> 4006 | 0.9666 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001043_block_000015` | text | 1700 -> 352 | 234 -> 3996 | 0.9663 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000174_block_000004` | text | 2024 -> 324 | 557 -> 4003 | 0.9661 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000464_block_000007` | text | 1936 -> 324 | 344 -> 4003 | 0.9656 | candidate_runaway_length, candidate_lost_eos, large_compact_text_edit |
| `page_000175_block_000000` | text | 1920 -> 320 | 177 -> 4004 | 0.9635 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001079_block_000022` | text | 1932 -> 360 | 374 -> 3994 | 0.9613 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000469_block_000009` | text | 1936 -> 324 | 286 -> 4003 | 0.9611 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000183_block_000000` | text | 1944 -> 352 | 400 -> 3996 | 0.9611 | candidate_runaway_length, candidate_lost_eos, large_compact_text_edit |
| `page_001064_block_000016` | text | 2016 -> 360 | 307 -> 3994 | 0.9610 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000552_block_000002` | text | 1936 -> 304 | 239 -> 4008 | 0.9607 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001027_block_000015` | text | 1984 -> 364 | 440 -> 3993 | 0.9604 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000858_block_000003` | text | 488 -> 236 | 9 -> 46 | 0.9583 | large_compact_text_edit |
| `page_001156_block_000009` | text | 1924 -> 320 | 374 -> 4004 | 0.9577 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000450_block_000003` | text | 1536 -> 352 | 368 -> 3996 | 0.9561 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001216_block_000004` | text | 1920 -> 364 | 438 -> 3993 | 0.9559 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000265_block_000004` | text | 1924 -> 320 | 381 -> 4004 | 0.9542 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001212_block_000023` | text | 1976 -> 352 | 305 -> 3996 | 0.9533 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000493_block_000005` | text | 1900 -> 352 | 290 -> 3996 | 0.9500 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000717_block_000009` | text | 1904 -> 336 | 821 -> 4000 | 0.9476 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001033_block_000005` | text | 2028 -> 320 | 347 -> 4004 | 0.9456 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001029_block_000014` | text | 1972 -> 336 | 219 -> 4000 | 0.9452 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001218_block_000015` | text | 2040 -> 364 | 477 -> 3993 | 0.9449 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001117_block_000005` | text | 1904 -> 360 | 351 -> 3994 | 0.9439 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001208_block_000014` | text | 768 -> 340 | 1952 -> 118 | 0.9423 | candidate_collapsed_length, large_compact_text_edit |

Full texts and exact per-crop metrics are in `per_crop.jsonl` and `manual_review.csv`.
Official OmniDocBench metrics are intentionally reported separately because they operate on matched page elements, not raw recognition crops.
