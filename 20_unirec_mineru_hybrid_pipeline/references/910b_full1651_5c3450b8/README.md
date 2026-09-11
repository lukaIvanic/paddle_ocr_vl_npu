# Full 1,651-image inference on 910B

2026-09-11, one Ascend 910B2 (physical NPU 7, logical npu:0).
Inference source: `5c3450b8376caa5119d3b97673e25fe0b01cadf9`, main branch,
clean tracked source on the pull-only 910B container. No code or configuration
changes during this run. Exact invocation is in `command.txt`.

## Completed result

| Measurement | Result |
|---|---:|
| Images completed | 1,651 / 1,651 |
| Crops completed | 30,557 |
| Pipeline processing | 641.591 s |
| Pipeline throughput | 2.573290 images/s |
| Separately measured model/setup stage | 38.152 s |
| External process lifetime | 721.992 s |
| Whole-process throughput | 2.286730 images/s |
| Whole-device sampled HBM peak | 22,952 MiB = 22.414 GiB |
| Whole-device HBM baseline | 3,396 MiB = 3.316 GiB |
| Peak increase over baseline | 19,556 MiB = 19.098 GiB |

Pipeline wall includes layout, recognition, page assembly/output and final
drain; it excludes model setup and process startup/shutdown. The external
sampler includes the entire child process lifetime, including imports, setup,
diagnostic trace writing and teardown. Evaluation time is separate from both.
HBM is **total device usage**, not an additional 22.414 GiB for the program.
Sampling is once per second; a higher instantaneous peak is possible.

All completion checks passed, exit 0. UniRec and MinerU ready pools ended at
zero. Owner timing partition error was -1.14e-13 s (floating-point roundoff).

## Settings and workload

Validated defaults unchanged: PP-DocLayoutV3 only; UniRec text; MinerU tables
and formulas. No MinerU layout or chart/image recognition. UniRec streamed
stages use four vision lanes, four CPU processes/eight resize threads each,
B128 with ready capacity 64. MinerU uses the experiment-11 production kernels,
FP16, PSE-sentinel IncreFA, B32/KV4096 and ready capacity 32/S4096. Recognition
min/max pixels remain 25,088/602,112. No output-length reduction was added.

- UniRec: 28,125 text crops, 14,932 graph calls; 65 length stops.
- MinerU: 1,681 formulas and 751 tables, 18,704 graph calls; 13 formula and
  4 table length stops. All other completions stopped at EOS.
- MinerU active-slot fraction: 86.113%; idle slots with ready work: 0.
- MinerU ready-KV admission: 6.372 s; active and ready arenas each 1.5 GiB.
- MinerU vision: 2,027,436 real tokens; text prefill: 567,659 useful /
  758,016 physical tokens. Device-event vision blocks: 49.205 s; text
  transformer prefill: 17.327 s; decode: 78.150 s. These event envelopes can
  include launch gaps and are not pure kernel-active time.

Coordinator action envelopes: UniRec streamed work 220.172 s, layout 166.467 s,
MinerU prefill 96.714 s, MinerU decode/admission/completion/fence 92.635 s,
wait 62.190 s. Control/orchestration accounts for the remainder. Do not add
nested device times to these mutually exclusive owner envelopes.

The full run is more representative than the formula-heavy first-64 subset.
Compared with the earlier UniRec+Paddle full run's 2.6123 pages/s, this is about
1.5% lower throughput. That is historical context, not a simultaneous controlled
A/B: crop counts also differ slightly (30,557 here vs 30,570 previously).

[Accuracy evaluation](../910b_accuracy_5c3450b8/README.md) includes every page
and all length-stopped predictions. Full raw artifacts remain at:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/20_unirec_mineru_hybrid_pipeline/full1651_5c3450b8/
```

`run_summary.json` is the unmodified inference report. `memory_summary.json`
is a compact extraction from the original external sampler; all raw samples
remain in that run directory. No 310P result is implied.
