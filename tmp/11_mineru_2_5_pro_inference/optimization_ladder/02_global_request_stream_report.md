# MinerU optimization rung 2: one global request stream

## Result

Accepted. `--global-request-stream` removes the drains between 32-page groups.
The first 128 OmniDocBench pages become one layout call and one recognition
call. No model or graph implementation changed.

| Metric | Four 32-page groups | One 128-page stream | Change |
|---|---:|---:|---:|
| Pipeline wall | 226.077 s | 199.833 s | -26.244 s (-11.6%) |
| Pages/s | 0.5667 | 0.6405 | +13.0% |
| Generation wall | 201.290 s | 174.582 s | -26.708 s |
| Prefill wall | 89.630 s | 88.400 s | -1.230 s |
| Decode wall | 64.587 s | 40.122 s | -24.465 s |
| Decode active-slot fraction | 53.21% | 83.85% | +30.64 points |
| Raw decode slots | 468,064 | 297,024 | -36.5% |

Both runs used B32, KV4096, PromptFA compiled vision prefill, packed compiled
text prefill, IncreFA compiled decode, NZ decode weights, and warm graph caches.

## Accuracy

- 125 of 128 page content lists were byte-exact.
- Three pages changed in one block each.
- One difference was only Chinese and LaTeX whitespace.
- Two differences were equivalent formula formatting. One added braces around
  an existing term; one changed an array column separator.
- No block count, block type, reading order, or substantive text changed.

The output differences are normal argmax sensitivity from changing continuous
decode cohort history. They do not show an accuracy regression.
