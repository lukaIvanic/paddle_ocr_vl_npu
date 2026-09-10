# Step 1: historical B8 / 6 QPS sanity anchor

Completed 2026-09-10 on physical NPU6, one Ascend 910B2.
This is the control before relocation into experiment 19. No inference source
was edited, no experiment-19 implementation was created, and the existing main
checkout was not changed.

## Results

| Metric | Original chart run | Step-1 rerun |
|---|---:|---:|
| Mean request latency (s) | 1.290604390 | 1.282710744 |
| P50 (s) | 0.875825615 | 0.863417849 |
| P90 (s) | 2.725499259 | 2.715154548 |
| P95 (s) | 3.776260672 | 3.781593808 |
| P99 (s) | 7.281013327 | 7.249392700 |
| Maximum (s) | 8.554178314 | 8.539985053 |
| Completed requests / second, including drain | 5.733797109 | 5.734012961 |
| Requests | 1000 | 1000 |
| Errors | 0 | 0 |
| KV4096-limit stops, retained | 10 | 10 |

The Poisson setting is 6 QPS. The exact finite schedule has actual arrival rate
5.763293819 requests/s and arrival span 173.511889437 s. Request latency includes
queueing and all API processing; source-page cropping and client payload
preparation occur before the arrival clock, as in the original experiment.

## Paired output and input checks

Joined original and new `results.jsonl` by request occurrence `sequence`, not by
completion order or table ID (the sequence intentionally repeats tables).

- Native token-ID streams: **1000/1000 identical**.
- Raw text: **1000/1000 identical**.
- Formatted table text/HTML: **1000/1000 identical**.
- Completion reasons: **1000/1000 identical**.
- Input-token counts, projected-image-token counts and crop dimensions:
  **1000/1000 identical**.
- The two `schedule.jsonl` files are byte-identical.
- Ordered request-ID SHA-256:
  `97a1f87dd18ace0833f6d66796ce6868845b04880292d0f632f3575c3693caa9`.

No new ground-truth TEDS evaluation was necessary to establish output parity:
the complete formatted outputs match the original exactly. This is not a claim
that a separate new TEDS evaluation was run.

## Execution provenance

- Source: `be691de190ae099d1a9b0ba80865006b122ecc00`.
- Historical sparse clone on host:
  `/data1/lukaiv/workspace/repos/table_step1_be691de1_20260910`.
- Same clone in the existing container:
  `/workspace/repos/table_step1_be691de1_20260910`.
- Container: `research_vllm_ascend_021_external_workspace`.
- Interpreter: `/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python`.
- Model: `/workspace/models/PaddleOCR-VL-1.6`.
- Package inventory matches the lock: torch 2.10.0+cpu, torch-npu 2.10.0,
  transformers 5.5.4, NumPy 1.26.4, Pillow 12.2.0, kornia-rs 0.1.14;
  CANN symlink resolves to 9.0.0.
- Historical tracked experiment-09 files passed `git diff --exit-code be691de1`.
- Existing compilation caches reused through the isolated clone's
  `.runtime_cache` symlink; original current checkout left untouched.
- The original `table_poisson_frontier.py` harness was loaded through `runpy`.
  Only its orchestration globals were selected: `MATRIX={8:[6]}` and
  `CONTAINER_REPO=/workspace/repos/table_step1_be691de1_20260910`.
  Its unchanged `Sweep` ran with count 1000, NPU6 and this output directory.
  Server/client source, scheduling, request generation, timing and cleanup
  implementations were unchanged.
- Actual server and client commands are saved in `b8/server_command.txt`,
  `b8/warm/command.txt`, and `b8/qps6/command.txt`.
- Synthetic constructor compilation, one complete real-request warmup,
  setup GC freeze and original instrumentation were preserved.
- Resolved readiness configuration matches the original except for the
  relocated vocabulary file path, measured setup times and a one-object
  difference in the GC frozen-object count (636312 vs 636311). Vocabulary
  content hash and execution settings match.

## Ownership and artifacts

`ownership.jsonl` contains 99 snapshots: no foreign NPU6 PID. Direct-host manual
checks agreed with the ownership monitor. Our server worker was host PID2393946,
under owned server PID2392826. The harness stopped its server and verified NPU6
free at 18:27:41 CST (12:27:41 Europe/Zagreb). No other user's process was touched.

- `b8/qps6/measured/results.jsonl`: every response, including native IDs.
- `b8/qps6/measured/schedule.jsonl`: exact request sequence and offsets.
- `b8/qps6/measured/summary.json`: latency and stage metrics.
- `b8/ready.json`, `b8/service.json`, `b8/server.log`: resolved runtime and setup.
- `results.json`, `status.json`, `ownership.jsonl`: original harness evidence.

Original comparison artifacts are under
`tmp/09_persistent_page_engine/table_poisson_frontier_screen1000_be691de1_20260907/b8/qps6/measured/`.

Conclusion: **Step 1 passes.** Mean differs by about -0.61%, P95 by +0.14%,
and all outputs are identical. Step 2 has not started.
