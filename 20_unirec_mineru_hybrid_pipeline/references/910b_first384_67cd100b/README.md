# Initial 910B continuous hybrid validation

Date: 2026-09-11. One Ascend 910B2, physical NPU7/logical npu:0.
Inference commit: `67cd100b`, main branch. Source was authored locally,
pushed and pulled into `/workspace/repos/paddle_ocr_vl_npu`; no remote source
edits. Environment: `source npu-setup`,
`/workspace/venvs/mineru_pro_vllm_py312/bin/python`.

Remote run directory:
`tmp/20_unirec_mineru_hybrid_pipeline/first384_streamed_67cd100b/`.
The complete log, predictions, recognition trace, timing trace and raw memory
sampler remain there. Here, `command.txt` is the exact invocation and
`run_summary.json` is the original summary. `memory_summary.json` preserves
sampled device HBM but omits unrelated process listings/raw all-device tables.

## Result and timing boundaries

- All384 pages and4,346 crops completed; exit0 and completion checks passed.
- Pipeline processing192.495s =1.994854 pages/s. Includes layout, recognition,
  page output and first-use work inside the pipeline; excludes model setup.
- Separately timed setup31.094s.
- External sampler observed260.564s process lifetime (~1.474 pages/s), including
  interpreter/import, setup and teardown costs. Do not label pipeline rate as
  whole-process cold-start rate.
- Owner timing partition error exactly0.0s.

Coordinator action envelopes (mutually exclusive; not pure kernel times):

| Action | Seconds |
|---|---:|
| UniRec streamed work |54.053|
| MinerU decode/admission/completion/fence |45.833|
| Layout advancement/detection/publication |43.274|
| MinerU prefill |39.773|
| Explicit wait |8.643|

Control/orchestration accounts for the remaining owner time. Nested device
timings must not be added to these envelopes. UniRec stage workers overlap
inside its exclusive model turns; MinerU and layout remain fenced/exclusive.

## Work and memory

UniRec:3,657 text crops (13 length stops),3,076 graph calls.
MinerU:531 formulas (9 length stops),158 tables (3 length stops),9,833 graph calls.
Other crop completions stopped at EOS. Length stops are not crashes, but these
outputs still require quality evaluation; successful drain is not an accuracy
claim. PPv3 is the only layout model. Images/charts are skipped.

MinerU active decode-slot fraction82.149%; idle slots with ready work0;
689 admissions. Ready-KV admission total1.268s, separate from model prefill.
Vision751,840 real tokens; text prefill205,185 useful/262,656 physical tokens.
MinerU decode device-event envelope41.312s; these envelopes include submission
gaps and are not pure kernel-active time.

Whole-device sampled HBM baseline3,398MiB, peak22,400MiB (21.875GiB),
increase19,002MiB (18.557GiB). This is **total device usage**, not an additional
22,400MiB process allocation. Sampling interval1s; instantaneous peaks may be
higher. No310P result follows from this910B measurement.

MinerU ready arena32/S4096 and active arenaB32/S4096 each1.5GiB.
Ready high-water32, final0. UniRec compact ready-KV high-water64 rows,
final0. UniRec uses four vision lanes, four persistent CPU processes/eight
resize threads each, B128, ready capacity64.

## Validation ladder

- Initial2-page run at`cc0859b3`: completed in14.737s processing.
- First64 streamed at`cc0859b3`:1,015 crops,76.054s processing, all drained.
  Whole-device sampled peak21,076MiB. This subset is formula-heavy; it is not
  representative of the full corpus.
- Nine experiment20 tests passed on the actual NPU at`67cd100b`, including
  prefix-copy isolation, import coexistence, cooperative pause/refill/drain,
  and timing-counter units. Experiment18's48 regression tests also passed on
  the NPU during initial bring-up.
- [Seven real crop token comparisons](../910b_parity_67cd100b/parity.json):
  standalone stream vs hybrid ready-KV path, exact raw-token match7/7,
  154 graph calls in each. Contains text, table and formula requests.

The initial failure was a same-name Python timing-module import collision;
MinerU now imports uniquely named`mineru_prefill_timing`. No model/kernel
fallback was introduced. Later commits make the explicitly tested UniRec
streamed/4-lane/ready64 arguments the experiment20 defaults; experiment18
defaults remain unchanged.

Not yet done: full1,651-page inference/evaluation, complete output-quality
comparison, or any310P validation/memory-fit claim.
