# Equivalent patch projection: matched development-100 audit

Source `c12b36335d3c9daf200c92b45ffc69b811dcc82f`, Ascend 910B2,
physical NPU6. Ordinary B2 decode, client C2. These are complete HTTP
request latencies, not composed or overlap-subtracted stage estimates.

Both direct API runs use the arguments in `command.txt`. The candidate adds
only `--vision-linear-patch-projection`. The opt-in replaces each full-patch
Conv2D by the equivalent flattened linear projection, using the original
weights and bias. It changes no input pixels, resampling, vision token counts,
positions, checkpoint parameters, vocabulary, KV4096 capacity, output limits,
or greedy/stopping policy. The code default remains convolution.

Each server starts separately and processes one complete warm request outside
measurement. The same client command is used for both measured runs:

```sh
python 09_persistent_page_engine/scripts/table_closed_loop_api_client.py \
  --api-url http://127.0.0.1:8767/v1/ocr \
  --set random --count 100 --shuffle-seed 1 --max-in-flight 2 \
  --output-dir <control-or-candidate-directory>
```

Warmup uses `--set warm --count 1 --max-in-flight 1`. The interpreter is
`/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python`.
Both saved manifests have the frozen development SHA256
`1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85`.
Payload preparation precedes timing; response-driven refilling never has more
than two requests outstanding. There is no cohort wait or ID-based routing.

| Measured result | Convolution control | Linear candidate |
|---|---:|---:|
| Completed tables/s | 2.837426103374047 | 2.8792158364868197 |
| P95 request latency (s) | 1.9805108521773938 | 1.960145636240485 |
| Mean embedding device time (s/request) | 0.005906608800888061 | 0.0011628769928216933 |
| Mean vision/text prefill host span (s/request) | 0.055601474842987955 | 0.05227333396440372 |
| Successful HTTP responses / EOS | 100 / 100 | 100 / 100 |

Throughput improves approximately 1.47%; P95 decreases approximately 20 ms.
**The >=3 tables/s development gate is not passed.** No seed2 or final1000
validation was launched. Stage times are diagnostics, not terms to subtract
from response latency. CPU preparation already runs in the background; its
service time overlaps decode and is not an additive latency saving.

## Output review

99/100 native streams match. The sole difference is `page_001227_table_10`:
the candidate deletes one literal `C` at raw-text offset882 in the cell
identified by `S750A_rp`. Both outputs have910 generated IDs. Character edit
distance of that cell to saved ground truth increases from1 to2. This is a
content difference, not equivalent formatting.

The same-input projection diagnostic measured small FP16 differences; the
projection is algebraically equivalent. First-divergence logits were not
captured, so this audit does not claim a token-level numerical causal proof or
a zero-quality-regression result. Ground truth is used only in this offline
review; it is not read by the serving path. Both complete outputs and native
IDs are retained in `comparison.json` and the request records. No generated
text was re-encoded.

## Ownership, reproducibility and next evidence

`analyze.py` checks the frozen input hash, all100 submissions, full C2 cap,
unchanged model/input configuration, EOS stops, native outputs, and sampled
NPU ownership. Six direct-host samples fall within each ~35-second measured
window; all show only its mapped worker. Samples bracket both windows.
`ownership_pid_mapping.txt` records manual host/container PID checks.
Both API logs finish with persisted service summaries. All owned API workers,
parents and the monitor were stopped. A new direct-host check at
2026-09-06T07:14:16+08:00 found NPU6 empty and all recorded PIDs absent.

The separate real embedding profile is in
`../table_serving_profile_20260905/embedding_linear_c12b3633/`.
Its approximately25.5x embedding-kernel improvement is **not** the serving
speedup. The existing real B2 decode profile attributes about74% of model
kernel time to matmuls and IncreFA, versus less than1% to the five ordinary
control kernels. CPU prep already overlaps execution. These observations
favor investigating substantial model execution cost over another speculative
control-loop rewrite; they do not yet establish a winning next optimization.
