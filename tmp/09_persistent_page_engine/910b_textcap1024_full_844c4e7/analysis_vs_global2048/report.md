# OCR resolution ablation

- Reference: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_full_maxpixels401408_c41c3d8_r2/output`
- Candidate: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_textcap1024_full_844c4e7/output`
- Shared crops: **30557**
- Affected crops: **3617 across 995 pages**
- Affected because the vision grid changed: **3617**
- Affected because the text-crop pixel input changed: **0**
- Real vision tokens: **5,841,940 -> 3,414,900** (2,427,040 saved)
- Physical vision tokens: **6,177,832 -> 3,733,132** (2,444,700 saved)
- Token streams exact among affected crops: **2619/3617**
- Whitespace-insensitive text exact: **2864/3617**
- Automatically flagged for manual review: **15**
- Unaffected crops whose generation changed: **56**

## Runtime

- Reference pipeline: **956.130s**
- Candidate pipeline: **891.604s**
- Delta: **-64.525s (-6.75%)**

## Affected crops by recognizer route

| Route | Crops | Real vision tokens | Physical vision tokens | Token exact | Compact-text exact | Flagged | Mean normalized character edit |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 3617 | 5,841,940 -> 3,414,900 | 6,177,832 -> 3,733,132 | 2619 | 2864 | 15 | 0.0123 |

## All crops by recognizer route

| Route | Crops | Real vision tokens | Physical vision tokens |
|---|---:|---:|---:|
| formula | 1681 | 632,228 -> 632,228 | 684,680 -> 680,796 |
| table | 751 | 1,041,308 -> 1,041,308 | 1,073,692 -> 1,074,256 |
| text | 28125 | 13,571,704 -> 11,144,664 | 15,050,588 -> 12,527,572 |

## Automatic review flags

- `candidate_collapsed_length`: 2
- `candidate_lost_eos`: 7
- `candidate_repetition_regression`: 7
- `candidate_runaway_length`: 6
- `large_compact_text_edit`: 15

## Worst affected-crop differences

| Request | Label | Vision tokens | Output tokens | Character edit | Flags |
|---|---|---:|---:|---:|---|
| `page_000146_block_000000` | text | 1920 -> 896 | 4 -> 3860 | 1.0000 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000983_block_000001` | text | 1900 -> 1008 | 3609 -> 3832 | 1.0000 | large_compact_text_edit |
| `page_000196_block_000009` | text | 1500 -> 960 | 3709 -> 3844 | 0.9990 | large_compact_text_edit |
| `page_001518_block_000001` | text | 1640 -> 992 | 3674 -> 16 | 0.9976 | candidate_collapsed_length, large_compact_text_edit |
| `page_001579_block_000019` | text | 1232 -> 960 | 261 -> 15 | 0.9712 | candidate_collapsed_length, large_compact_text_edit |
| `page_000950_block_000002` | text | 2016 -> 912 | 488 -> 3856 | 0.9089 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001385_block_000031` | text | 1980 -> 920 | 642 -> 3854 | 0.8906 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000019` | text | 1540 -> 1008 | 774 -> 3832 | 0.8661 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001370_block_000020` | text | 1716 -> 960 | 909 -> 3844 | 0.8306 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000438_block_000008` | text | 1904 -> 960 | 1923 -> 3844 | 0.8132 | candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_001521_block_000016` | text | 1240 -> 864 | 12 -> 17 | 0.8000 | large_compact_text_edit |
| `page_001599_block_000003` | text | 1980 -> 920 | 6 -> 16 | 0.6667 | large_compact_text_edit |
| `page_001518_block_000007` | text | 1200 -> 864 | 28 -> 15 | 0.6452 | large_compact_text_edit |
| `page_001370_block_000005` | text | 1716 -> 960 | 869 -> 3844 | 0.6439 | candidate_runaway_length, candidate_repetition_regression, candidate_lost_eos, large_compact_text_edit |
| `page_000954_block_000005` | text | 1932 -> 960 | 296 -> 350 | 0.5109 | large_compact_text_edit |
| `page_001346_block_000002` | text | 1920 -> 924 | 147 -> 107 | 0.4809 | - |
| `page_000761_block_000039` | text | 1936 -> 992 | 85 -> 77 | 0.4029 | - |
| `page_000802_block_000002` | text | 1760 -> 992 | 30 -> 22 | 0.4000 | - |
| `page_000948_block_000003` | text | 1936 -> 960 | 194 -> 192 | 0.3830 | - |
| `page_000351_block_000007` | text | 2016 -> 936 | 75 -> 65 | 0.3817 | - |
| `page_000174_block_000004` | text | 2024 -> 960 | 557 -> 604 | 0.3792 | - |
| `page_001521_block_000018` | text | 1728 -> 924 | 35 -> 16 | 0.3506 | - |
| `page_000014_block_000005` | text | 1960 -> 960 | 293 -> 343 | 0.3392 | - |
| `page_001335_block_000002` | text | 1908 -> 888 | 91 -> 123 | 0.3392 | - |
| `page_000658_block_000020` | text | 1728 -> 864 | 132 -> 160 | 0.3217 | - |
| `page_001518_block_000005` | text | 1952 -> 688 | 43 -> 46 | 0.3158 | - |
| `page_000774_block_000005` | text | 1344 -> 980 | 186 -> 186 | 0.2996 | - |
| `page_000812_block_000007` | text | 1728 -> 936 | 588 -> 533 | 0.2991 | - |
| `page_000464_block_000003` | text | 1932 -> 960 | 552 -> 443 | 0.2899 | - |
| `page_000861_block_000001` | text | 1980 -> 936 | 135 -> 107 | 0.2892 | - |
| `page_000718_block_000003` | text | 1632 -> 936 | 138 -> 138 | 0.2778 | - |
| `page_000842_block_000000` | text | 1920 -> 952 | 572 -> 567 | 0.2589 | - |
| `page_001406_block_000010` | text | 1340 -> 896 | 92 -> 81 | 0.2586 | - |
| `page_000442_block_000004` | text | 1560 -> 960 | 505 -> 483 | 0.2576 | - |
| `page_000075_block_000011` | text | 1056 -> 900 | 191 -> 178 | 0.2379 | - |
| `page_000806_block_000025` | text | 1992 -> 928 | 13 -> 12 | 0.2353 | - |
| `page_001355_block_000001` | text | 1488 -> 832 | 29 -> 41 | 0.2302 | - |
| `page_000059_block_000006` | text | 1104 -> 900 | 150 -> 173 | 0.2286 | - |
| `page_000754_block_000036` | text | 1240 -> 880 | 35 -> 34 | 0.2254 | - |
| `page_001627_block_000001` | text | 1380 -> 928 | 7 -> 11 | 0.2188 | - |
| `page_000761_block_000011` | text | 1984 -> 880 | 33 -> 28 | 0.2143 | - |
| `page_000183_block_000000` | text | 1944 -> 988 | 400 -> 288 | 0.2139 | - |
| `page_000861_block_000004` | text | 1976 -> 972 | 94 -> 113 | 0.2096 | - |
| `page_001347_block_000001` | text | 1920 -> 1012 | 114 -> 78 | 0.2081 | - |
| `page_000089_block_000016` | text | 1200 -> 920 | 188 -> 192 | 0.2078 | - |
| `page_001282_block_000009` | text | 1220 -> 848 | 66 -> 80 | 0.2072 | - |
| `page_000005_block_000005` | text | 1176 -> 936 | 127 -> 130 | 0.2070 | - |
| `page_000030_block_000011` | text | 1536 -> 960 | 151 -> 138 | 0.2054 | - |
| `page_000761_block_000006` | text | 1920 -> 896 | 43 -> 39 | 0.2041 | - |
| `page_001359_block_000004` | text | 1368 -> 960 | 24 -> 27 | 0.2000 | - |

Full texts and exact per-crop metrics are in `per_crop.jsonl` and `manual_review.csv`.
Official OmniDocBench metrics are intentionally reported separately because they operate on matched page elements, not raw recognition crops.
