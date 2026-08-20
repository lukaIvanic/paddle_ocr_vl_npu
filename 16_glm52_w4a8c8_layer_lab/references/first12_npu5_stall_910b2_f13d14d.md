# GLM-5.2 first-12-layer NPU 5 stall

Date: 2026-08-20

Source commit: `f13d14d`

The comparison used physical 910B2 NPU 6 as the control and physical NPU 5 as
the device under test. Both commands requested layers 0-11, B1, KV4096, eight
validation positions, 20 ordinary warmup calls, three 200-call timing windows,
an explicit cache parity limit of 0.078125, and a device-specific cold TorchAir
cache.

NPU 6 completed. Its exact summary is
`first12_npu6_910b2_f13d14d.json`.

The NPU 5 command was:

```bash
export ASCEND_RT_VISIBLE_DEVICES=5
/usr/local/python3.12.13/bin/python3 \
  16_glm52_w4a8c8_layer_lab/benchmark_optimized_tp1_stack.py \
  --model-dir /workspace/models/GLM-5.2-w4a8c8 \
  --first-layer 0 \
  --last-layer 11 \
  --cache-length 4096 \
  --validation-steps 8 \
  --cache-parity-atol 0.078125 \
  --warmup-steps 20 \
  --decode-steps 200 \
  --measurement-repeats 3 \
  --compile-cache-dir .runtime_cache/16_glm52_w4a8c8_layer_lab/device5 \
  --summary-out /tmp/glm52_tp1_first12_npu5_b969386.json
```

NPU 5 loaded layer 0, printed `loading optimized TP1 layer 1`, and made no
further progress for more than two minutes. During the stall, `npu-smi` reported
health `OK`, AICore 0%, and 4,105 MiB total HBM use versus about 3,420 MiB at
idle. The device process held 731 MiB. The container process slept in
`hrtimer_nanosleep`; its other threads were primarily blocked on futexes.

Ctrl-C did not unwind the blocked runtime call. SIGTERM was sent only to the
owned benchmark PID 3940130. The process stopped. NPU 5 returned to 3,425 MiB
HBM, 0% AICore, and no running device process.

Conclusion: NPU 5 reports healthy at the management layer but does not behave
correctly for this model-loading workload. Do not include it in a multi-NPU GLM
run until the device or runtime state is repaired and this exact smoke passes.
