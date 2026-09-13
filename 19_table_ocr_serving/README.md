# Experiment 19: table OCR serving

## Start the crop server

From this directory, with the Ascend environment already initialized:

```sh
python p01_serve.py \
  --model-path /models/PaddleOCR-VL-1.6 \
  --graph-cache-directory /cache/paddle-graphs \
  --log-folder /logs/paddle \
  --device npu:0 \
  --decode-batch-size 8 \
  --host 127.0.0.1 --port 8765 \
  --metrics-level basic
```

Replace the three directories with your own paths. This uses the bundled
60,416-row decode vocabulary; add `--full-decode-lm-head` for all 103,424 rows.
Startup prepares the graphs before accepting requests. Stop with Ctrl+C.
Each HTTP request sends encoded image bytes to
`/v1/ocr?crop_type=table`, `text`, or `formula`.

## Try one image

In another terminal, send a cropped table image. Replace `table.png` with your
own image path; the server returns its recognized text and table HTML as JSON.

```sh
curl -sS --fail-with-body \
  -H 'Content-Type: image/png' --data-binary @table.png \
  'http://127.0.0.1:8765/v1/ocr?crop_type=table' \
  -w '\nTotal request time: %{time_total}s\n'
```

To try several requests at once, send them independently. Each response goes
to its own file, and the terminal shows how long each request took:

```sh
curl -sS --fail-with-body --data-binary @table.png \
  'http://127.0.0.1:8765/v1/ocr?crop_type=table' \
  -o table-result.json -w 'Table: %{http_code}, %{time_total}s\n' &
curl -sS --fail-with-body --data-binary @text.png \
  'http://127.0.0.1:8765/v1/ocr?crop_type=text' \
  -o text-result.json -w 'Text: %{http_code}, %{time_total}s\n' &
curl -sS --fail-with-body --data-binary @formula.png \
  'http://127.0.0.1:8765/v1/ocr?crop_type=formula' \
  -o formula-result.json -w 'Formula: %{http_code}, %{time_total}s\n' &
wait
```

## Try 100 or 1,000 tables

`benchmark_tables.py` reads the annotated table crops from OmniDocBench and
sends them to your running server. Point it at the folder containing
`OmniDocBench.json` and `images/`. The client needs Pillow, not an NPU.

```sh
python benchmark_tables.py \
  --omnidocbench /datasets/OmniDocBench \
  --api-url http://127.0.0.1:8765/v1/ocr \
  --requests 100 --qps 6 --output-dir results-100
```

For a longer run, change the request count and use a new output folder:

```sh
python benchmark_tables.py \
  --omnidocbench /datasets/OmniDocBench \
  --api-url http://127.0.0.1:8765/v1/ocr \
  --requests 1000 --qps 6 --output-dir results-1000
```

`--qps 6` means six incoming requests per second **on average**, with random
gaps (Poisson arrivals). Requests do not wait for earlier ones to finish.
You will see `SEND` and `RECV` lines, followed by mean/P50/P95/maximum latency,
failures and maximum concurrent requests. Keep the server terminal open to
watch its logs at the same time. After crop preparation, 1,000 requests at
6 QPS takes roughly three minutes, depending on how quickly the server finishes.

The client performs one warmup request outside measurement, then waits for
every measured response. Successful-request latency includes the connection
and response, not image preparation. Failures are counted separately and make
the command exit unsuccessfully; check them before interpreting latency.

OmniDocBench v1.6 supplies 665 tables. A 1,000-request run includes all 665 plus
335 randomly selected tables, shuffled together. Other counts work the same
way. `--seed` defaults to 1 for repeatable order and arrival times. This is a
self-contained demo workload, not the historical chart's locked table order.
It tests serving performance, not ground-truth recognition accuracy.

The output folder contains `results.jsonl` (individual responses and timings),
`schedule.jsonl` (the chosen tables and arrival times), and `summary.json`.
Existing empty folders are accepted; previous run files are not overwritten.

## Live logging

`--log-folder` receives `events.jsonl` while serving and the existing
`service_summary.json` at shutdown. Every event is also printed as the same
JSON line to the terminal. One background writer formats, writes and flushes
each event immediately; no batching timer or separate logging process exists.

`--metrics-level basic` (default) logs startup/model/graph preparation, readiness,
accepted and finished requests, response writes, timeouts/rejections/errors and
shutdown. Finished requests include their ID, type, worker latency, output-token
count and stopping reason. `detailed` additionally includes existing per-request
CPU/wait/combined-prefill/decode-residency/detokenization wall timings, output
formatting duration, dimensions and input-token counts. Neither level enables
NPU profiling events or the research scheduling traces.

The inference worker sends a heartbeat every 15 seconds by default, including when idle,
plus initial/final snapshots. It reports retained output tokens (including EOS),
occupied decode slots, CPU-preparation/prefill waiting crops and prefilled crops
waiting for decode. The parent adds all unfinished requests and its admission
limit. CPU-preparation counts cover the worker's lookahead, not incoming jobs
it has not pulled yet. Snapshot values are observations, not an atomic view of
both processes. Set `--heartbeat-interval-s 5` for updates every five seconds,
or set `ServeConfig.heartbeat_interval_s` when using the Python interface.

Output tokens/s uses the change in the live retained-token count divided by
the actual elapsed snapshot interval. It includes prefill's first token and EOS,
excludes inactive slots and discarded lookahead, and includes all preparation,
prefill, queueing and idle time in its wall-clock denominator. It is not isolated
NPU decode throughput. Request event rates cover events received since the last
heartbeat. Detailed heartbeats also give mean/P95 worker latency for completions
in the last 60 seconds, bounded to the latest 2,048 completions, with sample count.


`worker_wall_s` ends after formatting and before returning the result through
IPC. `http_wall_s` is measured before writing the HTTP response. Neither includes
the client's network latency. `response_sent` records a successful server write,
not acknowledgement by the client. A caller timeout/cancellation is logged
separately from inference finishing; `result_not_returned` identifies late results.
Operational logs exclude image bytes, recognized text and native token arrays.

The writer queue is bounded to 1,024 events. A full queue drops logs instead of
blocking inference; the writer warns when it can proceed. Console/file failures
produce rate-limited warnings (at most once per heartbeat interval); one failed output
does not prevent trying the other. Flush does not mean fsync/crash-proof storage.
Shutdown gives the writer two seconds to finish queued events, so a stuck disk
or terminal cannot prevent process exit. Log loss can make event-derived counts
incomplete; runtime token counters do not depend on log delivery.


## Python API, crop types and unfinished-request limit

HTTP accepts `crop_type=table`, `text` or `formula`. Each request supplies one
encoded image and its crop type; there is no crop classification or bulk API.
The corresponding model prompts are `Table Recognition:`, `OCR:` and
`Formula Recognition:`. All results retain `raw_text` and native `token_ids`.
`text` applies math-delimiter normalization; only tables additionally convert
OTSL to HTML.

Python uses the same inference service without opening an HTTP port. Import
`p01_serve` with this experiment directory on the Python module search path:

```python
import asyncio
from pathlib import Path
from p01_serve import InferenceServer, ServeConfig

async def main():
    config = ServeConfig(
        model_path=Path("/models/PaddleOCR-VL-1.6"),
        graph_cache_directory=Path("/cache/paddle-graphs"),
        log_folder=Path("/logs/paddle"),
    )
    server = InferenceServer(config)
    try:
        server.start()
        result = await server.recognize_async(
            Path("crop.png").read_bytes(), crop_type="text",
        )
        print(result["text"])
    finally:
        server.close()

if __name__ == "__main__":
    asyncio.run(main())
```
