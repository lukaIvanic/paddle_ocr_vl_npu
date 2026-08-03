# OCR resolution ablation

- Reference: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_full_maxpixels401408_c41c3d8_r2/output`
- Candidate: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_textcap512_full_844c4e7/output`
- Shared crops: **30557**
- Affected crops: **7888 across 1343 pages**
- Affected because the vision grid changed: **7888**
- Affected because the text-crop pixel input changed: **0**
- Real vision tokens: **8,941,000 -> 3,564,036** (5,376,964 saved)
- Physical vision tokens: **9,604,364 -> 3,862,876** (5,741,488 saved)
- Token streams exact among affected crops: **5311/7888**
- Whitespace-insensitive text exact: **5883/7888**
- Automatically flagged for manual review: **89**
- Unaffected crops whose generation changed: **52**

## Runtime

- Reference pipeline: **956.130s**
- Candidate pipeline: **833.612s**
- Delta: **-122.518s (-12.81%)**

## Affected crops by recognizer route

| Route | Crops | Real vision tokens | Physical vision tokens | Token exact | Compact-text exact | Flagged | Mean normalized character edit |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 7888 | 8,941,000 -> 3,564,036 | 9,604,364 -> 3,862,876 | 5311 | 5883 | 89 | 0.0222 |

## All crops by recognizer route

| Route | Crops | Real vision tokens | Physical vision tokens |
|---|---:|---:|---:|
| formula | 1681 | 632,228 -> 632,228 | 684,680 -> 665,716 |
| table | 751 | 1,041,308 -> 1,041,308 | 1,073,692 -> 1,074,564 |
| text | 28125 | 13,571,704 -> 8,194,740 | 15,050,588 -> 9,068,808 |

## Automatic review flags

- `candidate_collapsed_length`: 6
- `candidate_lost_eos`: 54
- `candidate_repetition_regression`: 55
- `candidate_runaway_length`: 51
- `large_compact_text_edit`: 87

## Worst affected-crop differences

| Request | Label | Vision tokens | Output tokens | Character edit | Flags |
|---|---|---:|---:|---:|---|
| `page_000257_block_000045` | text | 760 -> 300 | 218 -> 13 | 1.0000 | large_compact_text_edit |
| `page_000146_block_000000` | text | 1920 -> 480 | 4 -> 1 | 1.0000 | large_compact_text_edit |
| `page_000361_block_000000` | text | 576 -> 440 | 9 -> 6 | 1.0000 | large_compact_text_edit |
| `page_000377_block_000001` | text | 720 -> 448 | 2 -> 2 | 1.0000 | large_compact_text_edit |
| `page_000983_block_000001` | text | 1900 -> 432 | 3609 -> 146 | 0.9995 | candidate_collapsed_length, large_compact_text_edit |
| `page_000196_block_000009` | text | 1500 -> 448 | 3709 -> 5 | 0.9984 | candidate_collapsed_length, large_compact_text_edit |
| `page_001518_block_000001` | text | 1640 -> 352 | 3674 -> 16 | 0.9984 | candidate_collapsed_length, large_compact_text_edit |
| `page_000014_block_000007` | text | 888 -> 448 | 3862 -> 124 | 0.9974 | candidate_collapsed_length, large_compact_text_edit |
| `page_000754_block_000036` | text | 1240 -> 468 | 35 -> 3967 | 0.9887 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000005` | text | 1716 -> 420 | 869 -> 3979 | 0.9753 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000023` | text | 1232 -> 476 | 679 -> 3965 | 0.9689 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000022` | text | 1232 -> 504 | 676 -> 3958 | 0.9671 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000638_block_000012` | text | 1932 -> 440 | 396 -> 3974 | 0.9643 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000634_block_000014` | text | 1972 -> 448 | 492 -> 3972 | 0.9641 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000414_block_000004` | text | 580 -> 432 | 18 -> 8 | 0.9615 | large_compact_text_edit |
| `page_001370_block_000018` | text | 1364 -> 432 | 713 -> 3976 | 0.9582 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000183_block_000000` | text | 1944 -> 468 | 400 -> 3967 | 0.9492 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000017` | text | 1320 -> 432 | 718 -> 3976 | 0.9479 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001217_block_000009` | text | 1872 -> 432 | 312 -> 3976 | 0.9436 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001208_block_000014` | text | 768 -> 480 | 1952 -> 112 | 0.9428 | candidate_collapsed_length, large_compact_text_edit |
| `page_001370_block_000019` | text | 1540 -> 480 | 774 -> 3964 | 0.9416 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001096_block_000065` | text | 560 -> 416 | 23 -> 22 | 0.9333 | large_compact_text_edit |
| `page_001370_block_000012` | text | 1232 -> 476 | 649 -> 3965 | 0.9310 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001203_block_000004` | text | 1904 -> 448 | 572 -> 3972 | 0.9281 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000004` | text | 1628 -> 480 | 796 -> 3964 | 0.9276 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001063_block_000015` | text | 2016 -> 504 | 411 -> 3958 | 0.9249 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000493_block_000005` | text | 1900 -> 432 | 290 -> 3976 | 0.9236 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001220_block_000004` | text | 1944 -> 468 | 555 -> 3967 | 0.9208 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001226_block_000007` | text | 1976 -> 468 | 662 -> 3967 | 0.9171 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001233_block_000005` | text | 1976 -> 468 | 484 -> 3967 | 0.9147 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000429_block_000008` | text | 1924 -> 432 | 483 -> 3976 | 0.9116 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000003` | text | 1540 -> 480 | 772 -> 3964 | 0.9083 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000014` | text | 1188 -> 476 | 633 -> 3965 | 0.9083 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000010` | text | 1496 -> 456 | 754 -> 3970 | 0.9083 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001041_block_000002` | text | 2016 -> 504 | 406 -> 3958 | 0.9053 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001234_block_000008` | text | 1980 -> 448 | 405 -> 3972 | 0.9038 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001385_block_000056` | text | 1980 -> 448 | 642 -> 3972 | 0.8982 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000492_block_000003` | text | 1920 -> 448 | 594 -> 3972 | 0.8971 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001385_block_000031` | text | 1980 -> 448 | 642 -> 3972 | 0.8963 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001385_block_000104` | text | 1980 -> 448 | 644 -> 3972 | 0.8963 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000489_block_000004` | text | 1944 -> 468 | 375 -> 48 | 0.8959 | candidate_collapsed_length, large_compact_text_edit |
| `page_001370_block_000015` | text | 1232 -> 476 | 662 -> 3965 | 0.8945 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000646_block_000003` | text | 1944 -> 468 | 378 -> 3967 | 0.8926 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000779_block_000005` | text | 2040 -> 480 | 672 -> 3964 | 0.8924 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000452_block_000009` | text | 1968 -> 480 | 767 -> 3964 | 0.8918 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001385_block_000008` | text | 1960 -> 476 | 666 -> 3965 | 0.8907 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001385_block_000042` | text | 1980 -> 448 | 641 -> 3972 | 0.8897 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001216_block_000004` | text | 1920 -> 480 | 438 -> 3964 | 0.8856 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000488_block_000007` | text | 1904 -> 476 | 659 -> 3965 | 0.8842 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000020` | text | 1716 -> 504 | 909 -> 3958 | 0.8791 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |

Full texts and exact per-crop metrics are in `per_crop.jsonl` and `manual_review.csv`.
Official OmniDocBench metrics are intentionally reported separately because they operate on matched page elements, not raw recognition crops.
