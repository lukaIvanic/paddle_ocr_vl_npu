# Ascend 910B2 OmniDocBench prefix reference

Commit `3a9244b`; physical NPU 5; layout first; two timing repeats per prefix; official OmniDocBench quick-match evaluator at `2b161d0`; CDM skipped.

| pages | e2e mean s (range) | pages/s mean (range) | text edit | formula edit | table TEDS | table edit | reading edit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 19.481 (19.388-19.573) | 1.6427 (1.6349-1.6505) | 0.138586 | 0.157549 | 0.967730 | 0.027912 | 0.042214 |
| 64 | 45.747 (45.141-46.352) | 1.3993 (1.3807-1.4178) | 0.098753 | 0.144277 | 0.967730 | 0.027912 | 0.032064 |
| 128 | 71.987 (71.911-72.063) | 1.7781 (1.7762-1.7800) | 0.094695 | 0.136436 | 0.885779 | 0.095465 | 0.052814 |
| 256 | 125.214 (124.579-125.849) | 2.0446 (2.0342-2.0549) | 0.065937 | 0.137559 | 0.912517 | 0.090540 | 0.125747 |

Both timing repeats had exact recognition semantics and byte-identical prediction Markdown for every prefix. Lower edit distance is better; higher TEDS is better. No CDM means no leaderboard Overall score.
