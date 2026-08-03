# Max-pixels OCR ablation

- Reference: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/max_pixels_ablation_o1373_n256_normal_da274e2/output`
- Candidate: `/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/max_pixels_ablation_o1373_n256_half_da274e2/output`
- Shared crops: **6130**
- Vision-grid-affected crops: **608 across 68 pages**
- Real vision tokens: **2,720,284 -> 1,481,696** (1,238,588 saved)
- Token streams exact among affected crops: **463/608**
- Whitespace-insensitive text exact: **489/608**
- Automatically flagged for manual review: **1**
- Unaffected-grid crops whose generation changed: **6**

## Runtime

- Reference pipeline: **226.427s**
- Candidate pipeline: **198.994s**
- Delta: **-27.433s (-12.12%)**

## Affected crops by label

| Label | Crops | Token exact | Compact-text exact | Flagged | Mean normalized character edit |
|---|---:|---:|---:|---:|---:|
| table | 16 | 9 | 9 | 0 | 0.0642 |
| text | 592 | 454 | 480 | 1 | 0.0046 |

## Automatic review flags

- `large_compact_text_edit`: 1

## Worst affected-crop differences

| Request | Label | Vision tokens | Output tokens | Character edit | Flags |
|---|---|---:|---:|---:|---|
| `page_000226_block_000003` | text | 4876 -> 2368 | 18 -> 10 | 0.6000 | large_compact_text_edit |
| `page_000003_block_000032` | table | 4992 -> 2448 | 326 -> 435 | 0.4225 | - |
| `page_000180_block_000009` | table | 4704 -> 2400 | 157 -> 160 | 0.4000 | - |
| `page_000148_block_000018` | text | 4760 -> 2420 | 45 -> 17 | 0.3544 | - |
| `page_000145_block_000005` | text | 4176 -> 2176 | 43 -> 42 | 0.3544 | - |
| `page_000148_block_000010` | text | 4788 -> 2500 | 31 -> 21 | 0.3108 | - |
| `page_000145_block_000021` | text | 4752 -> 2432 | 55 -> 65 | 0.1788 | - |
| `page_000222_block_000004` | table | 4960 -> 2464 | 2844 -> 2773 | 0.1731 | - |
| `page_000222_block_000005` | text | 2820 -> 2240 | 137 -> 121 | 0.1196 | - |
| `page_000011_block_000003` | text | 4752 -> 2432 | 23 -> 24 | 0.0685 | - |
| `page_000223_block_000010` | text | 5040 -> 2408 | 138 -> 143 | 0.0397 | - |
| `page_000010_block_000021` | text | 4104 -> 2436 | 87 -> 86 | 0.0244 | - |
| `page_000237_block_000013` | text | 4900 -> 2380 | 62 -> 64 | 0.0230 | - |
| `page_000248_block_000009` | text | 5016 -> 2484 | 169 -> 168 | 0.0228 | - |
| `page_000033_block_000009` | table | 3808 -> 2464 | 143 -> 146 | 0.0216 | - |
| `page_000006_block_000017` | text | 5040 -> 2480 | 92 -> 92 | 0.0215 | - |
| `page_000016_block_000013` | text | 3444 -> 2520 | 73 -> 74 | 0.0212 | - |
| `page_000011_block_000048` | text | 5040 -> 2400 | 104 -> 104 | 0.0186 | - |
| `page_000012_block_000017` | text | 2788 -> 2496 | 3387 -> 3460 | 0.0173 | - |
| `page_000217_block_000010` | text | 4900 -> 2520 | 66 -> 66 | 0.0162 | - |
| `page_000011_block_000025` | text | 4560 -> 2464 | 91 -> 91 | 0.0158 | - |
| `page_000146_block_000010` | text | 3336 -> 2340 | 21 -> 21 | 0.0154 | - |
| `page_000011_block_000009` | text | 4440 -> 2464 | 73 -> 73 | 0.0141 | - |
| `page_000239_block_000006` | text | 4968 -> 2432 | 173 -> 177 | 0.0139 | - |
| `page_000221_block_000011` | text | 4960 -> 2464 | 105 -> 104 | 0.0137 | - |
| `page_000011_block_000007` | text | 4104 -> 2436 | 92 -> 92 | 0.0137 | - |
| `page_000011_block_000033` | text | 3192 -> 2376 | 72 -> 72 | 0.0132 | - |
| `page_000239_block_000008` | text | 4928 -> 2400 | 147 -> 147 | 0.0132 | - |
| `page_000221_block_000015` | text | 2640 -> 2432 | 41 -> 41 | 0.0125 | - |
| `page_000251_block_000022` | text | 4324 -> 2448 | 69 -> 68 | 0.0111 | - |
| `page_000225_block_000012` | text | 4324 -> 2448 | 72 -> 74 | 0.0110 | - |
| `page_000243_block_000008` | text | 5040 -> 2436 | 198 -> 198 | 0.0104 | - |
| `page_000255_block_000016` | text | 4828 -> 2448 | 103 -> 102 | 0.0096 | - |
| `page_000253_block_000011` | text | 4928 -> 2480 | 81 -> 81 | 0.0095 | - |
| `page_000220_block_000007` | text | 5040 -> 2400 | 135 -> 135 | 0.0095 | - |
| `page_000233_block_000006` | text | 4956 -> 2520 | 143 -> 144 | 0.0085 | - |
| `page_000250_block_000007` | text | 4988 -> 2480 | 93 -> 92 | 0.0082 | - |
| `page_000225_block_000005` | text | 5032 -> 2496 | 137 -> 137 | 0.0081 | - |
| `page_000011_block_000038` | text | 3648 -> 2480 | 63 -> 63 | 0.0080 | - |
| `page_000253_block_000008` | text | 5100 -> 2448 | 63 -> 63 | 0.0075 | - |
| `page_000011_block_000008` | text | 4104 -> 2436 | 68 -> 68 | 0.0074 | - |
| `page_000251_block_000023` | text | 5040 -> 2400 | 140 -> 140 | 0.0073 | - |
| `page_000012_block_000008` | text | 2788 -> 2496 | 667 -> 667 | 0.0073 | - |
| `page_000006_block_000020` | text | 4992 -> 2448 | 102 -> 102 | 0.0073 | - |
| `page_000231_block_000012` | text | 4920 -> 2436 | 100 -> 100 | 0.0072 | - |
| `page_000221_block_000004` | text | 4900 -> 2520 | 144 -> 144 | 0.0072 | - |
| `page_000226_block_000009` | text | 5040 -> 2400 | 157 -> 157 | 0.0072 | - |
| `page_000227_block_000007` | text | 4232 -> 2448 | 81 -> 81 | 0.0072 | - |
| `page_000220_block_000009` | text | 4960 -> 2464 | 136 -> 136 | 0.0070 | - |
| `page_000101_block_000003` | text | 2816 -> 2400 | 199 -> 200 | 0.0069 | - |
| `page_000013_block_000044` | text | 5016 -> 2392 | 82 -> 82 | 0.0062 | - |
| `page_000012_block_000066` | text | 2788 -> 2496 | 665 -> 665 | 0.0061 | - |
| `page_000006_block_000016` | text | 4872 -> 2460 | 90 -> 90 | 0.0061 | - |
| `page_000219_block_000010` | text | 5040 -> 2432 | 100 -> 100 | 0.0058 | - |
| `page_000253_block_000015` | text | 5040 -> 2480 | 238 -> 237 | 0.0058 | - |
| `page_000217_block_000011` | text | 4872 -> 2520 | 93 -> 93 | 0.0058 | - |
| `page_000245_block_000014` | text | 4988 -> 2400 | 116 -> 116 | 0.0054 | - |
| `page_000249_block_000009` | text | 4988 -> 2480 | 109 -> 109 | 0.0054 | - |
| `page_000009_block_000036` | text | 3224 -> 2484 | 102 -> 102 | 0.0053 | - |
| `page_000123_block_000004` | table | 5016 -> 2484 | 280 -> 285 | 0.0053 | - |
| `page_000254_block_000003` | text | 3128 -> 2400 | 62 -> 64 | 0.0052 | - |
| `page_000215_block_000019` | text | 4920 -> 2436 | 107 -> 107 | 0.0050 | - |
| `page_000251_block_000005` | text | 3128 -> 2460 | 56 -> 56 | 0.0049 | - |
| `page_000231_block_000020` | text | 4960 -> 2464 | 122 -> 122 | 0.0047 | - |
| `page_000246_block_000008` | text | 4960 -> 2464 | 141 -> 141 | 0.0047 | - |
| `page_000254_block_000021` | text | 5040 -> 2432 | 136 -> 136 | 0.0046 | - |
| `page_000006_block_000009` | text | 5104 -> 2460 | 63 -> 63 | 0.0042 | - |
| `page_000016_block_000008` | text | 4992 -> 2464 | 144 -> 144 | 0.0040 | - |
| `page_000016_block_000019` | text | 4644 -> 2480 | 68 -> 68 | 0.0040 | - |
| `page_000009_block_000058` | text | 2652 -> 2432 | 63 -> 63 | 0.0039 | - |
| `page_000221_block_000007` | text | 4352 -> 2400 | 58 -> 58 | 0.0039 | - |
| `page_000249_block_000008` | text | 5032 -> 2496 | 169 -> 170 | 0.0038 | - |
| `page_000225_block_000015` | text | 4232 -> 2448 | 68 -> 67 | 0.0037 | - |
| `page_000237_block_000012` | text | 4968 -> 2432 | 74 -> 72 | 0.0036 | - |
| `page_000253_block_000013` | text | 5076 -> 2508 | 80 -> 79 | 0.0036 | - |
| `page_000244_block_000017` | text | 4888 -> 2376 | 71 -> 70 | 0.0035 | - |
| `page_000011_block_000039` | text | 4104 -> 2436 | 68 -> 68 | 0.0035 | - |
| `page_000016_block_000026` | text | 4960 -> 2464 | 71 -> 72 | 0.0033 | - |
| `page_000009_block_000042` | text | 2728 -> 2520 | 85 -> 85 | 0.0033 | - |
| `page_000254_block_000017` | text | 5032 -> 2496 | 107 -> 106 | 0.0033 | - |

Full texts and exact per-crop metrics are in `per_crop.jsonl` and `manual_review.csv`.
Official OmniDocBench metrics are intentionally reported separately because they operate on matched page elements, not raw recognition crops.
