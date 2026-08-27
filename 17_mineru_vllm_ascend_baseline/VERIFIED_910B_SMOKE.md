# Verified 910B2 one-page smoke

This record covers the first accepted experiment-17 smoke. It proves that the
fixed stock vLLM-Ascend configuration starts, compiles its requested graphs,
runs MinerU two-step extraction, writes nonempty outputs, and releases the NPU.
It is not an OmniDocBench accuracy score or a 981-page throughput result.

## Run identity

```text
UTC date: 2026-08-27
git commit: 217052653a86d1923289eddfc9ab023f4c13e542
mode: compiled_async
physical device: Ascend 910B2 NPU6, exposed as logical npu:0
command: LIMIT=1 bash run_npu_reproduction.sh
remote run directory: /workspace/repos/paddle_ocr_vl_npu/tmp/17_mineru_vllm_ascend_baseline/compiled_async_n1_20260827T170449Z_2170526
```

The fresh environment was
`/workspace/venvs/mineru_vllm_ascend_exp17_py312`. Environment verification
reported vLLM 0.21.0+empty, vLLM-Ascend 0.21.0rc1, torch 2.10.0+cpu,
torch-npu 2.10.0, Transformers 5.5.4, mineru-vl-utils 1.0.5, and NumPy 1.26.4.

The selected input was OmniDocBench dataset index 0:

```text
image: page-d1561665-5359-42fe-920c-d6e3bff81953.png
bytes: 443653
sha256: 8282ad01c38a151423c205bd30a21d9d049dee0e27402cc0d0270a11dd3ed34c
```

## Compile and runtime checks

- The engine log confirmed TP1, FP16, no quantization, maximum model length
  8192, asynchronous scheduling, chunked prefill, prefix caching, and the
  experiment-owned vLLM compile cache.
- vLLM-Ascend saved the compiled graph and enabled `npugraph_ex` and static
  kernels.
- All 14 requested full-decode graphs captured. Capture took 455 seconds.
- Full engine setup took 640.544 seconds. The photographed benchmark contract
  excludes this from page throughput.
- ACL graph replay occurred during the measured inference.
- The process exited with code 0. The log had no traceback and no
  `ModuleNotFoundError`.
- Engine shutdown completed. NPU6 then reported free, 3424 MiB of 65536 MiB
  HBM in use, 0 percent AICore use, and `Health=OK`.

The first attempted smoke at commit `5282e8e` was rejected. CANN's
`op_compiler` selected `/usr/bin/python3`, which had no NumPy, so its static
kernel compiler invocations failed even though graph capture continued. Commit
`2170526` fixes this by exporting the fresh environment interpreter through
`HI_PYTHON` before engine startup. The accepted run had zero such failures.

## Measured one-page result

```text
client setup:             0.025527 s
image load:               0.114225 s
two-step inference:      22.813134 s
output writes:            0.000983 s
benchmark wall:          22.953887 s
completed:                        1
failed:                           0
pages/s:                   0.043566
inference-only pages/s:    0.043834
```

The output set contained:

```text
predictions/page-d1561665-5359-42fe-920c-d6e3bff81953.md     3447 bytes
content_lists/page-d1561665-5359-42fe-920c-d6e3bff81953.json 5608 bytes
input_manifest.json                                           321 bytes
model_manifest.json                                          1464 bytes
run_manifest.json                                            2315 bytes
run_summary.json                                             2697 bytes
```

The Markdown began with the expected textbook page heading, body text, and
matrix formulas. This is only a nondegenerate-output check. The prediction set
has not been scored against OmniDocBench ground truth.

## Next valid step

Run the documented 10-page ladder entry from a new process. Treat it as a
warm-cache smoke and label it separately. Do not call the current first 981
pages an exact reproduction of the photographed 981-page corpus; its original
ordered image list is still unavailable.
