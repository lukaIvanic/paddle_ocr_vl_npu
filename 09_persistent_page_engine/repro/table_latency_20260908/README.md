# Locked table-latency comparison — 2026-09-08

This is the reproduction authority for the product-facing **Optimized Ascend
pipeline vs vLLM-Ascend** chart. It freezes the measured configurations, not
today's CLI defaults. **No model run was performed to create this lock.**
The command expansion and data/source checks were validated on CPU.

## Start here

From the repository root, verify the saved evidence:

```bash
python3 09_persistent_page_engine/repro/table_latency_20260908/verify.py
python3 09_persistent_page_engine/repro/table_latency_20260908/test_lock.py
```

Print the complete host-side commands for a selected chart point:

```bash
python3 09_persistent_page_engine/repro/table_latency_20260908/verify.py \
  --commands optimized --qps 6 --npu 6 --output-dir tmp/repro_optimized_qps6_NEW

python3 09_persistent_page_engine/repro/table_latency_20260908/verify.py \
  --commands vllm --qps 6 --npu 4 --output-dir tmp/repro_vllm_qps6_NEW
```

These **only print commands**; they do not SSH, load models, reserve a card,
change source, or start/stop a service. Choose a fresh output directory and a
manually verified free card. On the host, run the printed server in its own
terminal, await readiness, run each warmup to completion, then run the client.
Retain server/client logs. When finished, stop only your own server and verify
the card is released. Never adopt or terminate another user's service.

The commands assume the two existing containers and their recorded filesystems.
The source revision and environment requirements below remain mandatory;
printing a command does not certify that the container currently satisfies them.

## Exactly which results?

All six integer-QPS pairs contain **1,000 requests**, covering all **665 tables**,
plus 335 repeated occurrences, globally shuffled. Mean and P95 use the **same
custom run at each rate**, selected by lowest measured P95, not separate winners.

| Incoming QPS | Optimized physical decode B | Optimized mean / P95 s | vLLM mean / P95 s |
|---:|---:|---:|---:|
| 1 | 2 | 0.6165 / 1.9079 | 4.3399 / 14.9175 |
| 2 | 3 | 0.6985 / 2.1785 | 5.3396 / 18.6224 |
| 3 | 4 | 0.8014 / 2.4787 | 7.1294 / 25.8291 |
| 4 | 6 | 0.8850 / 2.8346 | 10.7403 / 38.0586 |
| 5 | 5 | 1.1920 / 3.0516 | 18.1820 / 51.5708 |
| 6 | 8 | 1.2906 / 3.7763 | 30.3438 / 67.0242 |

This is **ordinary continuous decoding**, not speculative decoding. B is a
fixed physical decode shape, not a client concurrency limit. Empty decode slots
are padded; requests finish and refill independently. Batch choice here is an
offline per-load experiment setting, not an image-ID or per-request oracle.

vLLM has one fixed **max-sequences 64 / forward-budget 16,384** configuration
at every rate. Its active batch is dynamic, not forced B64. Its complete sweep
also includes 1.5/2.5/3.5/4.5/5.5/6.5/7/8 QPS. The original optimized 30-point
matrix and all server/client commands and resolved readiness records are saved.

Chart/source metrics:

- `outputs/01a0735d-f277-7262-b1d0-b87d6db95456/latency-bars-comparison.png`
- Same folder: `latency-bars-comparison.svg`, `latency-bars-data.json`
- Full optimized metrics: `outputs/poisson-frontier-20260907/results.json`
- Full vLLM report: `tmp/09_persistent_page_engine/table_vllm_poisson1000_seq64_14qps_npu4_retry_85a90862_20260908/README.md`

## Source and evidence pins

| Lane | Original source commit | Original artifact root under `tmp/09_persistent_page_engine/` |
|---|---|---|
| Optimized | `be691de190ae099d1a9b0ba80865006b122ecc00` | `table_poisson_frontier_screen1000_be691de1_20260907` |
| vLLM orchestration/client | `85a9086245595fa75997f4ef574c7543fc58408e` | `table_vllm_poisson1000_seq64_14qps_npu4_retry_85a90862_20260908` |

`lock.json` retains full source Git blob IDs, evidence SHA-256 hashes, original
matrix/rates, chart selection, expanded optimized server argv, and the vLLM
launcher hash. Original `server_command.txt`, client `command.txt`, `ready.json`,
warmup commands and summaries are the authority; do not replace them with a
recollection of a similarly named experiment.

For strict historical reproduction, use the corresponding recorded source
revision in a clean reproduction environment. Do **not** reset a dirty main
checkout or change the shared working tree to accomplish this. The source and
newly archived evidence live at different commits: preserve the locked evidence
and schedule when preparing the historical environment. The default checker
validates historical Git objects and artifact hashes, not installed NPU binaries.

Optional working-tree check:

```bash
python3 09_persistent_page_engine/repro/table_latency_20260908/verify.py --worktree vllm
python3 09_persistent_page_engine/repro/table_latency_20260908/verify.py --worktree optimized
```

At lock creation, the vLLM source check passed. The optimized historical check
correctly flagged four newer files: the two layout frontend/postprocess files,
Poisson orchestration script, and Poisson client (which later gained the vLLM
adapter/replay support). The optimized model/serving package and crop API were
unchanged. Do not silently call a newer checkout an exact historical checkout.
The expanded optimized flags are tested against the **historical parser** and
match all its original resolved arguments, including false/absent options.

## Optimized execution contract

`lock.json:optimized.expanded_server_argv` explicitly spells out all numeric,
model, graph-cache, vision/text-bucket and optimization CLI arguments. Important
resolved settings, also preserved in each batch's `ready.json`:

- FP16, greedy ordinary argmax; **no token suppression or LaTeX override**.
- `combined_apply_complete_layer_prefetch1_rope_lut_packed_mlp`.
- Frozen frequency-selected 16,384-row LM head, mapped back to native IDs;
  full checkpoint head for prefill. Native-ID list hash:
  `9c48e5c3b92776ba250f75359fccb407448c4da8419fe927f5ea381d345712c3`.
- KV4096, max-new-tokens4096; total context reaching KV4096 still stops output.
- Packed QKV and MLP, AddRMSNorm, SwiGLU, RoPE lookup, NZ weights, complete
  next-layer prefetch and post-scatter KV prefetch.
- Stock IncreFA GQA, mask-length mode, `inner_precise=None` (not forced 1);
  ordinary NPU scatter. No custom attention operator or packed-KV scatter.
- Optional per-decode timing events **off**; request/scheduling metrics **on**.
  `compact_decode_control=False`, `max_prefill_interruptions=None`.
- Setup GC collect/freeze enabled; GC stays enabled for new request objects.
- Vision: TorchAir PromptFA BNSD, D72 weights zero-padded to D80, joint FP32
  RoPE, aligned128, MLP4352, NZ weights, equivalent linear patch projection.
- Vision buckets: `256,384,512,640,768,1408,1920,2048,2944,4096`.
  Text buckets: `128,256,512,1024,1152`; manual causal FP32-softmax text prefill.
- Vision packing and text packing **off** in this serving experiment.
- Pixels min28224/max802816 unchanged; patch14, merge2, resize-factor28.
- One background CPU preparation worker, max pending2; NPU prefill admission
  only for free decode slots (`free_decode_slots_only_cpu_lookahead`).
- API host127.0.0.1, port8767, queue64, server timeout3600s, input limit64MiB.
- One complete real-request warmup after server/graph setup, outside timing.

## vLLM execution contract

Use the existing isolated container `research_vllm_ascend_023_paddleocr`.
The executed `/workspace/serve_vllm_table_reference.sh` was verified byte-identical
to `09_persistent_page_engine/scripts/serve_vllm_table_reference.sh` at the pin:
SHA-256 `1739aae950e010714fbc05c858ed5127056b2f1758f23ce766bbb011431d235a`.
Always check this copy; changing the repository file does not update it.

The locked launch overrides are **mandatory**; the generic launcher's defaults
are intentionally different:

```text
ASCEND_RT_VISIBLE_DEVICES=<one manually verified free physical card>
TABLE_VLLM_MAX_SEQS=64
TABLE_VLLM_TOKEN_BUDGET=16384
```

Complete effective CLI (capture sizes are every integer1 through64):

```text
vllm serve /workspace/models/PaddleOCR-VL-1.6
  --served-model-name PaddleOCR-VL-1.6
  --trust-remote-code --dtype float16
  --max-model-len 4096 --max-num-batched-tokens 16384 --max-num-seqs 64
  --no-enable-prefix-caching --mm-processor-cache-gb 0
  --async-scheduling --enable-chunked-prefill
  --mm-processor-kwargs '{"min_pixels":28224,"max_pixels":802816}'
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]}'
  --host 127.0.0.1 --port 18081
```

The launcher exports `TASK_QUEUE_ENABLE=1`,
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `PYTHONUNBUFFERED=1`.
Recorded resolved defaults: seed0, TP1/PP1/DP1, KV dtype auto,
GPU-memory-utilization0.92, vision-encoder compilation/capture disabled.
Two complete real-request warmups before measurement. Native server log records
the full compiler configuration. Prefix/processor caches are off; the installed
vLLM request-UUID behavior prevents cross-request encoder feature reuse.

## Environment identity, not just CLI

`environment.json` records the current post-run inventory and model/tokenizer
hashes checked in **both** containers. Preserve these environments:

- Optimized: torch2.10.0+cpu / torch-npu2.10.0, CANN9.0.0, pipeline_py312,
  Pillow12.2.0, kornia-rs0.1.14. `source npu-setup` also loads ATB and jemalloc.
- vLLM: vLLM0.23.0+empty / Ascend0.23.0rc1, torch-npu2.10.0.post2,
  installed CANN link9.0.1, Pillow12.3.0. Image digest:
  `quay.io/ascend/vllm-ascend@sha256:9fd154211d2faee7735cfe40785f863603e42ccd53bfb4bca82dc98c9446ea0a`.
- Both: transformers5.5.4, NumPy1.26.4, one910B2. Custom measured on physical6;
  vLLM on physical4. Do not assume different cards/software have identical cadence.
- Both containers use host networking, privileged mode and128GiB `/dev/shm`.

The base image digest does **not** capture later changes to container layers,
external venvs, drivers, mounted models or compiled caches. TorchAir's installed
distribution version is unavailable (`unknown` in original records). This lock
is not a complete binary environment export. Reprovisioning must validate the
recorded runtime/configuration and warm measurements before claiming parity.
No driver/container changes or binary exports were performed for this task.

Where the model files are accessible, add `--model-dir /workspace/models/PaddleOCR-VL-1.6`
to `verify.py` to check the saved weight/tokenizer/processor hashes without
importing Torch or creating an NPU context.

## Request and timing contract

- Source: `tmp/09_persistent_page_engine/table_b1_latency_full_04fbc8e/client/tables.jsonl`.
  Images: `/workspace/datasets/OmniDocBench/images`.
- Prompt `Table Recognition:`; fixed crop bounding boxes from the saved source.
  Client prepares RGB PNG table crops before the arrival clock starts. The API
  performs its own image decode, resize/normalization and model work. Page layout
  detection/cropping is not in these latency numbers. Source GT/old latency is
  benchmark metadata, not an inference routing input.
- Client `--cohort all --max-requests 1000 --seed 1 --shuffle-all`;
  timeout900s; **no client max-in-flight cap**, no sleep between responses.
  `--duration-s` is ignored when max-requests is set; `--ocr-time-s` is irrelevant
  with an HTTP endpoint. The full-corpus cohort includes the first table.
- Optimized original client generates the sequence from seed1. Tests reproduce
  it exactly. vLLM replays the saved B2/QPS1 schedule with
  `--schedule-source-qps 1`, scaling offsets by1/target-QPS.
- Sequence hash (JSON-encoded ordered request-ID list):
  `97a1f87dd18ace0833f6d66796ce6868845b04880292d0f632f3575c3693caa9`.
- vLLM HTTP payload: model `PaddleOCR-VL-1.6`, temperature0, native token IDs
  requested; one image-url plus the table prompt. No explicit max_tokens field;
  original vLLM fills the remaining4096 context budget. Do not silently add a
  shorter output cap. Optimized uses the ordinary image/png crop API.
- Latency: actual client request attempt through complete response, including
  all queueing and processing. Scheduled-arrival latency/dispatch lag retained.
- Completed throughput includes the full finite run and final drain. Offered
  QPS is a randomized load setting, not achieved throughput or proven steady
  capacity. Actual arrival rate is about96.05% of target for this fixed sample.
- Every point retains10 context-limit stops, same occurrences across systems;
  no removal of slow/capped requests. All recorded points have zero request
  errors. TEDS runs after inference; retain native output IDs, never re-encode.
- Keep the original sweep order for the closest cadence/cache reproduction.
  One-point commands are configuration-equivalent checks, not a promise of
  identical measured numbers after a different warmup history.

## Whole-sweep reproduction

The original host-side harness already monitors NPU ownership, writes every
response immediately, rejects contamination and drains/stops its owned server.
Use it at the pinned source revision, with a fresh output directory:

```bash
# Optimized, original 30-point matrix, one server lifetime per B.
python3 09_persistent_page_engine/scripts/table_poisson_frontier.py \
  --npu 6 --count 1000 --output-dir tmp/repro_optimized_full_NEW

# vLLM, fixed configuration, all 14 rates in the original order.
python3 09_persistent_page_engine/scripts/table_poisson_frontier.py \
  --api-kind vllm --npu 4 --count 1000 --vllm-max-seqs 64 \
  --vllm-token-budgets 16384 \
  --vllm-qps 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 8 \
  --schedule-jsonl tmp/09_persistent_page_engine/table_poisson_frontier_screen1000_be691de1_20260907/b2/qps1/measured/schedule.jsonl \
  --output-dir tmp/repro_vllm_full_NEW
```

Do not launch both simultaneously. Manually inspect `npu-smi info -t proc-mem`
on the selected card before launch and check its owning commands; retain
ownership monitoring throughout. Never count an overlapped run. No automatic
card hopping or benchmark retries. Compile/cache warmup belongs outside timing;
unexpected compilation in measured requests must be reported, not subtracted.

**Acceptance of a reproduction:** exact input/order/flags/environment audit,
all1000 responses retained, errors and context stops accounted for, no foreign
NPU process, and paired latency/throughput/output-quality comparison to the
saved anchors. Configuration equivalence does not guarantee exact latency or
bit-identical logits. Preserve any regression rather than adjusting the target.
