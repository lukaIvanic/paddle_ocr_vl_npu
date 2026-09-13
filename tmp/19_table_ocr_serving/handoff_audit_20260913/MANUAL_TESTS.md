# Luka's manual checks — not part of the product bundle

These commands use the existing 910B2 container and validated code already on
the server (`59392313`; local `044e830e` only adds comments/evidence). No model
code changed during this audit. Commands and referenced paths were checked;
the new mixed/Python/1,000-request tests have not been run by this audit.
Do not run two inference servers on the same device/cache at once.

## 1. Enter the container (repeat in each Mac terminal)

```sh
ssh -t -i /Users/lukaivanic/Downloads/key.pem \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=10 -o ServerAliveInterval=30 \
  -o ControlMaster=auto -o ControlPersist=600 \
  -o ControlPath=/tmp/paddle-bluezone-validation-master \
  root@116.204.40.238 \
  'docker exec -it -w /workspace/repos/paddle_ocr_vl_npu research_vllm_ascend_021_external_workspace bash'
```

All commands below run **inside that container**, not on the Mac.

## 2. Terminal A: start HTTP and watch live server logs

```sh
source npu-setup
npu-smi info
```

Check that physical NPU6 has no running processes. It was free during this
audit, but check again before starting. If occupied, stop here and choose a
free card; never stop someone else's process. The explicit choice below uses
our previously validated card; `npu-setup` may have selected another one.

```sh
export ASCEND_RT_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1
server_logs=$(mktemp -d /workspace/repos/paddle_ocr_vl_npu/tmp/manual-ocr-server.XXXXXX)
printf 'Server logs: %s\n' "$server_logs"

/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -u \
  19_table_ocr_serving/p01_serve.py \
  --model-path /workspace/models/PaddleOCR-VL-1.6 \
  --graph-cache-directory /workspace/repos/paddle_ocr_vl_npu/.runtime_cache/19_current_checkpoint_20260913 \
  --log-folder "$server_logs" \
  --host 127.0.0.1 --port 8767 \
  --device npu:0 --decode-batch-size 8 \
  --max-in-flight-requests 64 \
  --request-timeout-s 3600 \
  --metrics-level detailed
```

Leave this terminal running. Startup graph progress, request events and
15-second heartbeats print here and go to `$server_logs/events.jsonl`.
Wait for `http_ready`. The 3,600-second timeout reproduces the historical
benchmark override; normal product usage defaults to 60 seconds.
Port 8767 must be unused. The cached 60,416-row B8 graphs already exist.

## 3. Terminal B: one request, then mixed requests

```sh
curl -fsS http://127.0.0.1:8767/ready | /workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -m json.tool

curl -fsS -H 'Content-Type: image/png' \
  --data-binary @crops/crop_05_table_rwkv_dims.png \
  'http://127.0.0.1:8767/v1/ocr?crop_type=table&request_id=manual-table' \
  | /workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -m json.tool
```

Send three types concurrently; each response is saved separately:

```sh
mixed_results=$(mktemp -d /workspace/repos/paddle_ocr_vl_npu/tmp/manual-mixed.XXXXXX)
send_crop() {
  curl -sS --fail-with-body -H 'Content-Type: image/png' \
    --data-binary "@$2" \
    "http://127.0.0.1:8767/v1/ocr?crop_type=$1&request_id=manual-$1" \
    -o "$mixed_results/$1.json" \
    -w "$1: HTTP %{http_code}, %{time_total}s\n"
}
send_crop table crops/crop_05_table_rwkv_dims.png &
send_crop text crops/crop_01_text_block_en.png &
send_crop formula crops/crop_02_equation_matrix.png &
wait
printf 'Responses: %s\n' "$mixed_results"
```

This is a routing/recognition smoke, not a comprehensive quality test. Ground
truth for these crops is in `crops/manifest.json`. Inspect the saved `raw_text`,
`text`, `token_ids` and `stop_reason`; HTTP 200 alone is not an accuracy check.

## 4. Terminal B: saved Poisson100, then the exact 1,000-request schedule

For comparable timing, run without other clients. A fresh server plus the
historical one-request warmup matches our benchmark protocol. If you ran the
mixed smoke above, restart Terminal A before the acceptance benchmark.

One real-request warmup outside measurement:

The benchmark clients create their output directory themselves and reject an
existing one. Create a unique parent with `mktemp`, then give the client a new
`results` child directory. Server logs and curl outputs do not have this restriction.

```sh
warm_results="$(mktemp -d /workspace/repos/paddle_ocr_vl_npu/tmp/manual-warm.XXXXXX)/results"
cd /workspace/repos/table_step1_be691de1_20260910
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -u \
  09_persistent_page_engine/scripts/table_closed_loop_api_client.py \
  --api-url http://127.0.0.1:8767/v1/ocr \
  --set warm --count 1 --max-in-flight 1 --output-dir "$warm_results"
cd /workspace/repos/paddle_ocr_vl_npu
```

Saved 100-table / 6-QPS check (prints each completion and final latency summary):

```sh
poisson100_results="$(mktemp -d /workspace/repos/paddle_ocr_vl_npu/tmp/manual-poisson100.XXXXXX)/results"
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -u \
  09_persistent_page_engine/scripts/table_request_load_simulator.py \
  --api-url http://127.0.0.1:8767/v1/ocr --cohort all --qps 6 --max-requests 100 --seed 1 \
  --source-jsonl /workspace/repos/table_step1_be691de1_20260910/tmp/09_persistent_page_engine/table_b1_latency_full_04fbc8e/client/tables.jsonl \
  --schedule-jsonl /workspace/repos/paddle_ocr_vl_npu/tmp/19_table_ocr_serving/preprocess_poisson100_20260911/schedule.jsonl \
  --output-dir "$poisson100_results"
```

For the final 1,000-request comparison, restart Terminal A and repeat the
one-request warmup above. This keeps the starting conditions comparable rather
than adding the Poisson100 run as extra warmup. The client waits for **all**
responses; it does not leave accepted requests running after successful exit.

```sh
poisson1000_results="$(mktemp -d /workspace/repos/paddle_ocr_vl_npu/tmp/manual-poisson1000.XXXXXX)/results"
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -u \
  09_persistent_page_engine/scripts/table_request_load_simulator.py \
  --api-url http://127.0.0.1:8767/v1/ocr --cohort all --qps 6 --max-requests 1000 --seed 1 --shuffle-all \
  --source-jsonl /workspace/repos/table_step1_be691de1_20260910/tmp/09_persistent_page_engine/table_b1_latency_full_04fbc8e/client/tables.jsonl \
  --schedule-jsonl /workspace/repos/paddle_ocr_vl_npu/tmp/19_table_ocr_serving/integrated_runtime_20260913/poisson1000/cached/b8/qps6/measured/schedule.jsonl \
  --output-dir "$poisson1000_results"
```

The schedule file fixes both table order and arrival offsets, rather than
regenerating them. Allow about three minutes plus startup. Save the printed
output paths: `results.jsonl` contains per-request timings and native IDs,
`summary.json` contains mean/P95, failures and throughput. The accepted 1,000
reference is beside the source schedule. Do not run extra manual requests
during a measured test. Final acceptance still requires comparing those outputs
and timings against the reference, not merely completing this command.

## 5. Optional Terminal C: watch memory

```sh
source npu-setup
watch -n 5 npu-smi info
```

For CPU resident memory of the inference process (RSS is in KiB):

```sh
ocr_worker_pid=$(curl -fsS http://127.0.0.1:8767/ready | /workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -c 'import json,sys; print(json.load(sys.stdin)["worker_pid"])')
watch -n 5 "ps -p $ocr_worker_pid -o pid,rss,etime,args"
```

Use the PID from `/ready` inside the container; `npu-smi` may show host PIDs.
Compare idle-after-drain RAM across repeated rounds on the **same** server for
memory growth. Restarting between rounds would hide lifetime retention.

## 6. Python API, after stopping HTTP

Press Ctrl+C in Terminal A, wait for `shutdown_finished`, then verify the card
is free with `npu-smi info`. Run this in that terminal with the same initialized
Ascend environment and `ASCEND_RT_VISIBLE_DEVICES=6`.

`python -c` is intentional: a spawned worker cannot reload a `python -` stdin
script. No additional application file or bulk-submission API is needed.

```sh
PYTHONPATH=/workspace/repos/paddle_ocr_vl_npu/19_table_ocr_serving \
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -u -c '
import asyncio, json, tempfile
from pathlib import Path
from p01_serve import ServeConfig, InferenceServer

async def main():
    server = InferenceServer(ServeConfig(
        model_path=Path("/workspace/models/PaddleOCR-VL-1.6"),
        graph_cache_directory=Path("/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/19_current_checkpoint_20260913"),
        log_folder=Path(tempfile.mkdtemp(prefix="manual-python-", dir="/workspace/repos/paddle_ocr_vl_npu/tmp")),
        decode_batch_size=8, metrics_level="detailed",
    ))
    try:
        server.start()
        crops = [("text", "crop_01_text_block_en.png"),
                 ("formula", "crop_02_equation_matrix.png"),
                 ("table", "crop_05_table_rwkv_dims.png")]
        tasks = [asyncio.create_task(server.recognize_async(
            (Path("crops") / filename).read_bytes(), crop_type=kind, request_id=filename,
        )) for kind, filename in crops]
        for completed in asyncio.as_completed(tasks):
            print(json.dumps(await completed, ensure_ascii=False), flush=True)
    finally:
        server.close()

asyncio.run(main())
'
```

Each finished crop prints immediately, without waiting for the others. This
loads the same persistent engine without opening an HTTP port.

## 7. Full vocabulary and error-path checks

For full-vocabulary HTTP checks, restart Terminal A with
`--full-decode-lm-head` added. For Python set `full_decode_lm_head=True` in
`ServeConfig`. The head cannot change in a running server. The full decode
graph for this source may need compilation; shared vision/text-prefill graphs
reuse the same namespace. Keep full-head responses separately for comparison.

Invalid image (server should report a per-request error and remain usable):

```sh
curl -sS -i -X POST --data-binary 'not an image' \
  'http://127.0.0.1:8767/v1/ocr?crop_type=table&request_id=invalid-image'
```

For capacity/late-result tests, restart with `--max-in-flight-requests 1
--request-timeout-s 0.01`, then send the three concurrent requests from step 3.
Expect timeout/rejection events while inference continues. The precise mix is
scheduling-dependent: HTTP 503 means capacity rejection, 504 means the caller
stopped waiting; neither permits accepting more work until unfinished OCR ends.
Watch `request_finished` and `result_not_returned` later. Send a normal request
after restarting with the usual settings to confirm recovery.

To check busy shutdown, send a valid request and press Ctrl+C in Terminal A
while it is running. Wait for its response and `shutdown_finished`; do not
kill the process or stop the container. These are manual behavior checks, not
automatically asserted tests. Do not use their modified settings for benchmarks.
