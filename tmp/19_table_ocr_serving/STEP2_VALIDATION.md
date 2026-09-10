# Step 2: experiment-19 relocation parity

Completed 2026-09-10. Code tested: `dc755584`. One physical Ascend 910B2, NPU6.
No inference or scheduler fixes were made after the mechanical relocation.
No step-3 cleanup has begun.

## Compared with the verified step-1 control

| Metric | Step 1 | Step 2, fresh-compile process | Step 2, cache-loaded process |
|---|---:|---:|---:|
| Mean request latency (s) | 1.282710744 | 1.360154561 | 1.283549087 |
| P50 (s) | 0.863417849 | 0.944289759 | 0.862832514 |
| P90 (s) | 2.715154548 | 2.846092651 | 2.713301178 |
| P95 (s) | 3.781593808 | 3.917355244 | 3.779593972 |
| P99 (s) | 7.249392700 | 7.485729625 | 7.259671545 |
| Maximum (s) | 8.539985053 | 8.744314916 | 8.545706200 |
| Completed requests/s | 5.734012961 | 5.730879792 | 5.733948247 |
| Errors / requests | 0 / 1000 | 0 / 1000 | 0 / 1000 |
| KV4096-limit stops, retained | 10 | 10 | 10 |

Offered setting 6 QPS, actual finite arrival rate 5.763293819/s. The same 665
unique tables plus 335 repeated occurrences, globally shuffled. Schedule files
match step 1 byte-for-byte. Ordered request-ID SHA-256 is
`97a1f87dd18ace0833f6d66796ce6868845b04880292d0f632f3575c3693caa9`.

All responses are retained, with no subtraction of CPU work, interruptions or
queueing. The historical client's source-page crop/PNG preparation is outside
the arrival clock; all processing inside the API remains timed.

## Accuracy / behavior

Join `results.jsonl` by occurrence `sequence`, not completion order/table ID.
For **both** experiment-19 runs versus step 1:

- 1000/1000 native token streams identical.
- 1000/1000 raw and formatted text/HTML outputs identical.
- 1000/1000 completion reasons identical.
- 1000/1000 input-token counts, projected-image-token counts and crop dimensions identical.
- All schedule records identical.

No separate ground-truth evaluation was rerun; exact formatted-output equality
establishes parity with the historical control, not a new independent TEDS result.

## Startup control, not a runtime optimization

The first complete experiment-19 process compiled fresh graphs in its separate
cache namespace, then served the real warmup and measured workload. It took
394.168 s for recognizer setup, collecting 100327 objects and freezing 1346978.
This run had a 6.0% mean / 3.6% P95 latency increase versus step 1. It is kept.

Readiness settings matched except for relocated paths, source-hash/cache
identities, measured setup times and GC setup counts. Mean vision device time
was essentially unchanged (0.034633 s control / 0.034588 s first experiment-19
run). The added E2E time was mostly waiting and decode-slot residency.

A new process loaded the same experiment-19 cached graphs without any code or
configuration changes, then performed the same real warmup. Recognizer setup
took 34.437 s, collected 10 objects and froze 636274 (step 1 froze 636312).
Its mean/P95 returned to within +0.065% / -0.053% of step 1. These observations
support matching cache-loaded startup for the relocation control; they do not
isolate which fresh-compile process state caused the earlier gap. That is a
future startup-design discussion, not an excuse to delete the first result.

## Source and tooling

Experiment 19 was copied from `be691de190ae099d1a9b0ba80865006b122ecc00` into the
agreed main filenames, keeping temporary support modules. Internal imports,
entrypoint directory discovery and cache source paths were mechanically
adjusted. The prefill `_linear_tokenwise` helper was renamed to avoid collision
with the distinct decode helper when joining the text modules.

`19_table_ocr_serving/tests/check_relocation.py` passes 47 file/provenance checks
and compares 316 function/class bodies. The new serving runtime imports without
loading any `paddleocr_vl` module from experiment 09. The NPU-side tracked source
was checked unchanged against `dc755584` after the run.

Code was authored/committed/pushed locally. The remote checkout lacked private
GitHub authentication, so the same Git commit was transferred in a bundle and
pulled with `--ff-only`; no remote tracked source was edited or credentials
changed. No packages, drivers or container configuration were changed.

The unchanged historical benchmark harness/clients ran from the isolated
step-1 clone. Orchestration changes select B8/QPS6, the experiment-19 absolute
entrypoint and its ownership-name match, and add experiment-19 files to the
source fingerprint. Each completed run includes its exact `driver.py` receipt,
server/client commands and `relocated_runtime.json`.

## Artifacts and ownership

- `step2_b8qps6_dc755584_20260910/`: rejected **startup only**. The first launcher
  adapter missed the server filename embedded in its shell command and started
  the old endpoint. The ownership guard rejected it before warmup/measurement.
  Its exact owned PID2409000 was inspected and stopped; NPU6/port8767 were
  confirmed free. There was no external competing NPU process and no measured
  result from that attempt. The launcher, not inference code, was corrected.
- `step2_b8qps6_dc755584_20260910_retry/`: first complete relocated run,
  fresh graph compilation. Worker host PID2413736, owned server PID2412421.
- `step2_b8qps6_dc755584_20260910_cached/`: cached-process confirmation.
  Worker host PID2441863, owned server PID2440795. 96 ownership records, no
  foreign PID. Server stopped and NPU6 released at 19:02:28 CST / 13:02:28 Zagreb.

Each complete directory retains `b8/qps6/measured/results.jsonl`,
`schedule.jsonl`, `summary.json`, startup/server logs, readiness configuration,
service summary and ownership log. The source/workload control remains in
`tmp/09_persistent_page_engine/step1_b8qps6_20260910/`.

**Conclusion:** cached-start step-2 performance and complete-output parity pass.
The fresh-compile process difference is explicitly retained. Stop here before
beginning simplification.
