# Real C2 serving: frozen output-length oracle thresholds

Ascend 910B2, physical NPU 6; 2026-09-05. Inference source commit `a15a5b6d`.
Same random-100 sample and dispatch order, one server, at most two complete
table requests anywhere in the pipeline. These are measured client HTTP
wall latencies, not simulations or sums of stage timings.

## Results

| Metric | Fresh current C2 control | Oracle >=512 | Oracle >=1,024 |
|---|---:|---:|---:|
| Speculative / ordinary requests | 61 / 39 | 25 / 75 | 8 / 92 |
| Completed tables/s | 2.046 | 2.338 | 2.470 |
| Mean | 0.976 s | 0.854 s | 0.809 s |
| P50 | 0.792 s | 0.685 s | 0.615 s |
| P90 | 1.969 s | 1.588 s | 1.630 s |
| P95 | 2.420 s | 2.064 s | 1.995 s |
| P99 | 2.826 s | 2.479 s | 2.356 s |
| Maximum | 5.067 s | 4.638 s | 4.707 s |
| Requests >2 s | 10 | 7 | 5 |
| Successful / EOS-terminated | 100 / 100 | 100 / 100 | 100 / 100 |
| Native outputs identical to current C2 reference | 100 / 100 | 100 / 100 | 100 / 100 |

Relative to the fresh control, >=1,024 raises completion throughput by 20.7%
and reduces P95 by 17.6%. Its P95 is 1.995261531 seconds, only 4.74 ms below
2 seconds. This single run is promising, not a repeatability/SLA guarantee;
it does not reach 3 tables/s. The 512 cutoff has better P90 and maximum here,
while 1,024 has better throughput, mean, median, P95 and P99.

The previous saved C2 run was 2.033 tables/s, P95 2.514 s, max 5.016 s, with
8 requests >2 s. We keep that historical run rather than replace it; fresh
control differences show normal ordering/timing sensitivity near thresholds.

The historical regular-B2 run on the exact same sample/order was 2.483 tables/s,
mean 0.793 s, P95 2.151 s, P99 4.915 s, max 5.081 s. Thus >=1,024 now approaches
that saved ordinary throughput with a substantially lower P99. Regular B2 was
not rerun in this experiment, so this is a labeled historical comparison.

## Per-table changes

Compared to the fresh control, 512 made 75/100 requests faster (53 by >50 ms)
and 15 slower by >50 ms. At 1,024, 76/100 were faster (62 by >50 ms) and 17
slower by >50 ms. Scheduling changes can hurt individual companions even when
the aggregate improves. `comparison.json` retains all 100 matched changes.

Illustrative changes with >=1,024:

| Table | Oracle B1 tokens | Route | Control | >=1,024 |
|---|---:|---|---:|---:|
| page_000921_table_9 | 90 | ordinary | 2.765 s | 0.363 s |
| page_000217_table_box_id_0 | 138 | ordinary | 2.414 s | 0.354 s |
| page_000261_table_box_id_3 | 1,106 | spec retained | 2.531 s | 1.987 s |

Five remaining >2 s requests at >=1,024:

| Table | Route | Latency |
|---|---|---:|
| page_000263_table_box_id_7 | spec | 4.707 s |
| page_000272_table_box_id_1 | spec | 2.332 s |
| page_000626_table_box_id_1 | spec | 2.281 s |
| page_001270_table_box_id_1 | ordinary | 2.163 s |
| page_000275_table_box_id_1 | spec | 2.152 s |

The 789-token ordinary table is slower than its 2.020 s fresh speculative
control; filtering does not promise an improvement for every request.
The 1,532-token chemistry table remains the maximum: output length alone does
not predict draft acceptance or guarantee that speculation helps.

## Exact experimental contract

- Existing height eligibility AND frozen ordinary-B1 output count >= threshold.
  Count includes EOS. No arrival position, companion, latency or acceptance
  enters the decision. No per-table routing exceptions.
- Oracle map is `../table_token_oracle_screen_20260905/oracle_counts.json`.
  The 100 measured counts come from native IDs in the saved ordinary-B1 run;
  other warmup candidates use the corpus's saved B1 output-count metadata.
  This is a zero-inference-cost experimental oracle, not an image estimator.
- The gate runs before row-crop preparation. Ordinary-routed responses were
  checked to contain neither draft rows nor row preparation. Unknown oracle
  IDs fail rather than silently falling back. The server defaults are unchanged.
- All model graph contracts are identical across runs, excluding recorded
  startup durations. Optimized Q1 decode, B8/B16 drafting, manual verifier,
  compact vocabulary/native-ID mapping, cache sizes, greedy decoding,
  preprocessing and adaptive K remain unchanged.
- Saved token sequences are not used to generate outputs. The map stores
  integers only; the existing post-generation reference comparison remains
  diagnostic. No generated text was re-encoded.
- Full request latency includes server decoding of crop PNG, preparation,
  waiting, prefill, inference, postprocessing and response. The unchanged
  client preloads crop payloads before timing. No interruptions, overlapped
  CPU spans or outliers are subtracted. Results flush after each response.

## Warmup and execution

Three fresh server processes, run sequentially: control, 512, 1,024. Each
loads the same cached production graphs and receives the same two external
full-request warmups at C2 (`--set a --count 2`):
`page_000279_table_box_id_0` and `page_000283_table_box_id_1`. Both remain
speculative under all three rules. This differs from the earlier control's
warmup sequence, but is identical among these three runs; the new fresh
control is therefore the primary comparison. Warmups and setup are outside
measurement. No extra untimed preparation is done per measured table.

The unchanged measured client uses `--set random --count 100 --shuffle-seed 1
--max-in-flight 2`. Every selection file has SHA-256
`1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85`.
Exact commands are in each measured directory's `command.txt`.

Client interval reconstruction and 100 measured server admissions per run
independently confirm the two-table cap, including background preparation.
Both clients and owned servers exited with status 0. `exit_status.json`
records statuses observed in the persistent terminal, including warmups.

## Output and ownership audit

All three configurations produce exactly the same native output lists on all
100 tables. Comparison against the *historical ordinary-B1* stream differs
for three rerouted tables at 512 and four at 1,024; these outputs already
existed identically in the current C2 control, so they are not new differences
introduced by this experiment. `comparison.json` keeps both comparison scopes.
No TEDS rescore was needed to establish unchanged output relative to control;
this is not a new independent accuracy evaluation.

Direct-host checks identified the sole NPU owner in each run:
control worker PID 1689638 (parent 1689636), 512 worker 1704968 (parent 1704966),
1,024 worker 1713251 (parent 1713249). The raw host monitor covers every
measured interval and shows no other process on NPU 6. Sampling uses a
five-second sleep plus query time; it is not continuous process tracing.
At 2026-09-05 15:44:49 UTC, direct-host NPU 6 was free after graceful shutdown.
No speculative API server remained, and the owned monitor was stopped.

CPU tests: 3 API/oracle tests, 13 scheduler tests and 5 decode-pipeline/identity
tests passed. One intermediate identity-test fixture needed the new helper
added to its isolated AST namespace; no runtime/model change was required.

`analyze.py` regenerates `comparison.json` and `audit.json` from the raw files.
No additional inference experiment was started after these three runs.
