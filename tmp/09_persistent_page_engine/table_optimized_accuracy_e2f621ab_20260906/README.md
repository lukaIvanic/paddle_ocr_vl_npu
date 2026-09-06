# Optimized ordinary C1/C2 accuracy — 665 unique tables

Scored with the same OmniDocBench evaluator and the existing parent-managed,
12-process TEDS worker implementation. No errors or timeouts in any lane.
Page-TEDS averages table scores within each page, then averages458 pages.
Sample/Table-TEDS averages665 tables directly. All8 unique KV4096-cap cases
remain included; the335 repeated load-test requests are not double-weighted.

| Route | Page-TEDS | Table-TEDS | Page structure-only TEDS |
|---|---:|---:|---:|
| Previous optimized C1 reference |95.46253076189666%|95.00127232929343%|97.81178168042242%|
| New optimized C1 |95.45444093340294%|94.97180140714785%|97.81178168042242%|
| New optimized C2 |95.45444093340294%|94.97180140714785%|97.81178168042242%|

New C1 and C2 match665/665 native token streams and every per-table score.
Relative to the previous reference, Page-TEDS changes by-0.008089828493718088
percentage points. Three table scores improve and six regress. The largest
regression is `page_001398_table_box-7mwax454`, from0.5792616240676276 to
0.3924402975617438, or-18.68213265058838 points. The aggregate's small change
must not obscure that individual regression. Numerical/content differences
were documented in the serving-validation reports; no scoring normalization
is fed back into inference.

## Inputs and provenance

- Reference: `../table_1000_matrix_02fe5645_20260905/b1/measured/results.jsonl`.
- New C2: `../table_packed_noevents_23d5518c_20260906/validation1000_a/results.jsonl`.
- New C1: `c1/results.jsonl`, freshly generated665 requests, seed3, C1/B1,
  one full warmup, same packed-MLP/linear-patch/GC-freeze/no-events stack as C2.
  Exact command and source commit are in `c1_command.txt`.
- Ground-truth join: `../table_b1_latency_full_04fbc8e/client/tables.jsonl`;
  used only in CPU evaluation, never by the ordinary API model worker.
- `score_saved.py` checks665 unique coverage and duplicate native-ID parity,
  then calls the existing `_score` implementation without changing TEDS.
- Evaluator revision2b161d010d2e3aff77a0edef359ea3a6411d23cd, Python3.10.12.
  Source hashes and evaluator worktree status are preserved. Only old result
  files, not evaluator source, were dirty.
- `reference_scores/`, `c1_scores/`, `c2_scores/` contain full predictions/GT,
  per-table/per-page scores and aggregate scores. Logs/exit codes are retained.

C1 completed665 requests (657 EOS,8 unchanged cache-cap stops), no HTTP errors.
Its server summary includes666 requests including warmup. CPU scoring partially
overlapped the accuracy run; do not treat its latency as a new controlled
performance benchmark. Only physical NPU6 was used; PID/namespace checks and
monitor evidence are saved. The server, client, all score workers and owned
monitor finished/stopped; `final_cleanup.log` confirms NPU6 empty.

The Mac initially refused writes for lack of disk space. Artifacts remained
intact on the remote server and were copied locally after space became available.
