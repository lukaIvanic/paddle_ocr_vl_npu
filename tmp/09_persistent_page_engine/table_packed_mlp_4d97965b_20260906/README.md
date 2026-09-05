# Packed MLP with the actual production prefetch schedule

Source `4d97965b`, Ascend 910B2, physical NPU6 only. This is an opt-in,
content-independent execution change, not a routing rule or token predictor.
The model checkpoint, selected native vocabulary, greedy policy, pixels,
vision tokens, output limits and KV4096 capacity are unchanged.

The existing production preset has separate gate/up MLP matmuls. The candidate
packs their original weights into one projection and uses stock NPU SwiGLU.
Its prefetch schedule references the packed allocation actually read by decode,
not the separate weights retained for text prefill. Full-layer-ahead prefetch,
KV prefetch, RoPE lookup, native RMSNorm/rotary, NZ formats and all other
production settings remain enabled. CPU tests check real-valued projection
equivalence and exact prefetch object references. They do not claim NPU parity.

The older saved packed-MLP experiment used native-format fallback, KV1024 and
none of the current complete-layer prefetch stack; it was not a matched test
of this combination.

## Commands and run order

`command.txt` saves the full base array and source commit. Candidate preset:
`combined_apply_complete_layer_prefetch1_rope_lut_packed_mlp`.
Control preset: `combined_apply_complete_layer_prefetch1_rope_lut`.
Both enable `--vision-linear-patch-projection` and weight-padded vision, with
exactly the previous vision/text buckets. Both retain static decode B2.
The summary flag is `--service-summary-output`; the first attempted profile
launch used an incorrect flag, exited2 before model execution, and is retained
as `launch_argument_error.log`.

Measured servers use the real `serve_crop_ocr_api.py`, each in a fresh process.
Each receives one full warm request (`--set warm --count 1 --max-in-flight 1`)
before this unchanged client command:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/table_closed_loop_api_client.py \
  --api-url http://127.0.0.1:8767/v1/ocr \
  --set random --count 100 --shuffle-seed 1 --max-in-flight 2 \
  --output-dir <candidate-or-control>
```

The frozen development manifest SHA256 remains
`1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85`.
Submission-to-response wall timing includes all serving work; payloads are
prepared before timing. No per-request preparation or interruptions are
subtracted. All100 requests are retained and the measured outstanding cap is2.

Order: candidate profile; fresh candidate100; fresh control100; separate fresh
control profile to investigate the unexpectedly slow control. Profiles use
`table_serving_profile_20260905/profile_api.py`, one full warm request and then
two real requests (`--set random --count 2 --shuffle-seed 1 --max-in-flight 2`).
After32 real two-active-slot iterations, the profiler warms5 and captures20.
No synthetic graph inputs are used. All profile clients and servers finish
successfully, and both diagnostic crops have identical native streams between
candidate/control and the older diagnostic. Profile HTTP times are not serving
results. First setup took197.58s, including146.96s of text-prefill graph setup;
later candidate/control setups took40.59s and50.08s.

## Full-serving result and important control anomaly

| Run | Completed tables/s | P95 request wall (s) |
|---|---:|---:|
| Earlier linear-projection control, c12b3633 | 2.8792158364868197 | 1.960145636240485 |
| Packed-MLP candidate | 2.959656718087691 | 1.8991991819231762 |
| Fresh same-source unpacked control | 2.4883203438012944 | 2.319046922435517 |

**The candidate does not pass >=3.000 tables/s.** No seed2 or1000-table gate
was run. The fresh control is anomalously slower than its predecessor; do not
advertise the candidate/control ratio as a clean19% end-to-end improvement.
The earlier-to-candidate gain is about2.8%, but that historical comparison is
not a substitute for a stable matched control either. Defaults remain unchanged.

The fresh control's mean whole-server decode event interval is1.4624ms/call,
versus1.2124ms in the earlier linear-projection control. Prefill host time stays
near52ms/request. Configuration comparison found unchanged production options;
the new decode-source hash changes the compiled cache identity. The follow-up
profile below does NOT reproduce a persistent kernel slowdown. Attribution of
the unprofiled control anomaly remains unresolved; do not claim a hardware
fault, compiler regression or host bottleneck as established.

## Real-loop profiles

| Mean per complete captured graph | Old control | Fresh control | Packed MLP |
|---|---:|---:|---:|
| Matmul kernels (us) | 517.259 | 520.393 | 368.173 |
| IncreFA kernels (us) | 386.151 | 385.756 | 393.368 |
| Total model kernels (us) | 1215.508 | 1216.587 | 1046.796 |
| Model device envelope (us) | 1295.475 | 1291.688 | 1124.713 |
| Matmuls per graph | 91 | 91 | 73 |

Packing removes18 gate/up matmul launches and lowers aggregate matmul time.
Its18 SwiGLU kernels total43.247us versus36.695us for the control's fused
activation kernels: the activation is slightly slower, but the projection gain
dominates. Attention is essentially unchanged. The fresh control reproduces
the historical kernel cost, so the evidence does not support blaming its
new cache identity alone for slower request timing.

Profiler host scopes/cadence are perturbed and nested, not additive latency
terms. The packed profile's host cache-compiler scope is longer than control's;
this does not prove a corresponding unprofiled overhead. Device-event intervals
can include submission gaps and are not identical to kernel-duration sums.
Both profiles report1800MHz samples. A post-run snapshot reports1800MHz and
48–49C, not continuous frequency/thermal history.

## Outputs and ownership

Both100-request runs finish with100 EOS and no HTTP errors.99/100 native
outputs match. In `page_001417_table_2`, two Chinese characters differ:
`电压初转` becomes `电压初赛`, and `电源波形一个周期` becomes
`电票波形一个周期`. Ground truth has `电压切断` for the first cell, so both
variants have errors there; the second change introduces one additional error.
This is a content difference, not formatting. Candidate generates69 tokens
versus68, so the gain is not obtained by truncating that output. Equivalent
matmul/activation paths permit FP16 drift, but first-divergence logits were not
captured. Keep this explicit caveat; do not claim exact quality parity.

Manual host/container PID mappings and the five-second ownership monitor are
retained. Sampled checks show only the relevant worker on NPU6; this is not
continuous hardware tracing. At2026-09-06T07:38:45+08:00, all four owned API
parents/workers and the monitor were absent and NPU6 was empty. No other user's
process was changed. Full raw profiler directories remain remotely; compact
CSV and compressed Chrome traces are retained here.

Reproduce the local audits with the existing analyzers, supplying this root
and, for the control profile, `control_profile_run`:

```sh
python3 tmp/09_persistent_page_engine/table_patch_linear_c12b3633_20260906/analyze.py \
  tmp/09_persistent_page_engine/table_packed_mlp_4d97965b_20260906
python3 tmp/09_persistent_page_engine/table_serving_profile_20260905/analyze.py \
  tmp/09_persistent_page_engine/table_packed_mlp_4d97965b_20260906
python3 tmp/09_persistent_page_engine/table_serving_profile_20260905/analyze.py \
  tmp/09_persistent_page_engine/table_packed_mlp_4d97965b_20260906/control_profile_run
```
