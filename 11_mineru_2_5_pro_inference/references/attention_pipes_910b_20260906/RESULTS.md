# Production MinerU attention PMU investigation — 910B2, 2026-09-06

## Scope and evidence

Physical NPU 4, `Ascend910B2`, same real production captures as the preceding
attention matrix. Direct S768 has 640 real + 128 filler rows. S5632 has 5476
real + 156 filler rows. B1, N16, native D80, FP16; all 32 vision blocks execute.
The unpad-D128 variant pads the head dimension, retains the D80 scale and
slices output back. Masks, weights, resolution and production defaults are
unchanged. Native weight formatting and manual FP32 LayerNorm are retained.

Collection wrapper commit `99b3173d`; analyzer subsequently hardened through
`59259548`. The wrapper changes only the timing boundary around the original
callable; neither inference source nor graph-cache identities were changed.
The compiled baseline reused existing caches. Eager variants do not compile.
No new accuracy/E2E run is claimed.

Eight lanes: baseline, eager PromptFA, unpad D80 and unpad D128 at S768/S5632.
Each requested eight independent profiler groups, three forwards per group,
32 attention calls per forward. Ten unprofiled warm samples precede collection.
Every executed profile replay matched its unprofiled candidate output exactly.
That is repeatability, not equivalence of different attention variants or OCR
accuracy. The baseline additionally matches its saved production features.

Raw device export was absent for all eight groups in the first unpad-D80 S768
process even though the math completed. The missing data is explicitly retained
in `metric_counters.json`, not filled with zeros. The affected lane was retried
alone; its evidence is recorded separately in `retry_metric_counters.json`.
The retry at `59259548`, again on physical NPU 4, exported all eight groups
successfully. Its durations are within about 1.2% of the pipe pass, with no
negative PMU values. Thus all 64 requested combinations have device data;
the two perturbed/invalid passes described below still require caution.
Thirteen CPU contract/parser tests passed; hardware validation is the actual
910B collection and repeat-parity checks, not those CPU tests.

Files:

- `existing_pipe_counters.json`: extraction from the previous matrix, no rerun.
- `metric_counters.json`: new suite, including missing/invalid/perturbed flags.
- `retry_metric_counters.json`: the targeted export retry, not pooled silently.
- Full raw CSVs and per-call JSON remain on the server (paths below).

Statistics include mean, min, p50, p99, max, valid/missing counts, and raw units.
Actual zero counters are included in averages. Negative PMU exports are kept in
raw calls but excluded from numerical summaries and explicitly flagged.

## Main finding 1: large PromptFA has substantial vector-side work

S5632 baseline, average per attention invocation:

| Field | Microseconds |
|---|---:|
| Actual kernel elapsed | 1713.93 |
| AIC MAC | 493.24 |
| AIC MTE1 | 393.63 |
| AIC MTE2 | 566.28 |
| AIC FixPipe | 393.79 |
| AIV vector | 1329.63 |
| AIV MTE2 | 298.64 |
| AIV MTE3 | 246.31 |

The vector active ratio is about 0.801 of recorded AIV time. The independent
arithmetic pass reports FP32 vector ratio 0.4395 and FP16 ratio 0.11575. These
are the profiler's ratios, not fractions of all arithmetic instructions and
not additive wall-time stages. The vector bank-conflict and bankgroup-conflict
ratios are 0.056 and 0.039; these do not independently prove a bottleneck.

For S768, baseline elapsed is 83.21 us, vector time 25.42 us, MAC time 9.28 us.
Its performance balance differs substantially from S5632. Compiled and eager
PromptFA have nearly identical S5632 kernel counters: eager vector 1329.60 us
and MAC 493.24 us, despite different whole-encoder dispatch/fusion overhead.

This motivates profiling FP32/vector work in the 310P baseline vs approximate
mode. It does NOT establish that all vector time is softmax, or that vector
arithmetic alone is the kernel critical path. Scalar engine occupancy is also
high, but cannot distinguish useful scalar work from synchronization/control.

## Main finding 2: unpad D128 is not faster because it transfers fewer bytes

S5632, averages per invocation; both report 24 AIC blocks and 48 mix blocks:

| Measurement | Unpad D80 | Unpad D128 |
|---|---:|---:|
| Kernel elapsed, us | 2545.74 | 1744.82 |
| MAC, us | 479.97 | 742.20 |
| MTE2, us (AIC) | 2309.28 | 1146.93 |
| FixPipe, us | 1882.08 | 684.05 |
| Vector, us | 1482.03 | 1547.46 |
| GM to L1, exported KB | 2134520 | 2850860 |
| L0C to GM, exported KB | 1543834 | 1906186 |

D128 reduces elapsed time about 31.5%, AIC MTE2 about 50.3%, and FixPipe about
63.7%, while reported GM-to-L1 volume INCREASES about 33.6% and L0C-to-GM
volume increases about 23.5%. MAC and vector time also increase.

This is evidence for a difference in transfer/layout/tiling efficiency or
pipeline interaction, rather than less arithmetic or fewer transferred bytes.
It does not isolate which implementation detail is responsible. In particular,
these GM-interface counters include cache effects and must not be read as
physical HBM traffic. Do not compare their derived rate directly with the
advertised HBM bandwidth. The D128 attention kernel only reaches roughly the
current PromptFA time; its eager full encoder remains slower.

## Measurement quality matters

- S5632 compiled-baseline `MemoryAccess` changes elapsed kernel time from
  1713.93 to 3369.83 us (1.966x), and emits negative traffic values in some
  fields. Quarantine this pass for performance/traffic conclusions.
- S5632 unpad-D128 L2 collection changes elapsed from 1744.82 to 2308.27 us
  (1.323x). Its cache counters describe an instrumented, perturbed execution.
- Other available groups are much closer in duration to their same-lane pipe
  pass. Check `elapsed_ratio_to_pipe`, not only profiler exit status.
- Missing CSVs are missing device measurements, never zero device work.
- Ratios/bandwidths are not summed. PMU engine times overlap and are NOT an
  additive breakdown of elapsed kernel time. AIC/AIV counter aggregation is
  distinct from wall-time aggregation. No absolute chip-bandwidth saturation
  or theoretical-compute-efficiency claim is made here.

## Bounded deeper operator-tool trials

Installed CANN 9.0.0 `msopprof` offers TimelineDetail in its help output.
Two attempts used the existing warmed S5632 baseline path, not a synthetic op:

1. `TimelineDetail` + application replay: rejected immediately as an invalid
   combination, exit 255; no model execution.
2. Kernel replay + MSTX `attention_hot` window: model/baseline replay reached
   the timing boundary, but Python MSTX range creation failed. No valid
   instruction timeline was obtained. The tool logged a failing child and
   `Get profiling data failed` but returned shell exit 0.

Both attempts had a 240-second external timeout. No vendor kernel was rebuilt,
no fake standalone input used, and no graph caches cleared. Kernel replay's
different L2 behavior would have made any resulting latency a separate diagnostic,
not a hot production measurement. This failed integration is not proof that
the chip inherently lacks instruction-timeline support.

## Next 310P comparison

Use `WORK_SERVER_310P_MINERU_ATTENTION_PIPES.md`: same captures and warmed
production callables, adding approximate D80. Which counters disappear with
its measured attention saving? Does unpad's D80/D128 transfer behavior reverse
on 310P? Which metrics are actually supported and export valid data there?
Retain the source/model/capture checks and all failed collection evidence.

## Server evidence roots

```
/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/attention_matrix_910b_r1/
/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/attention_pipes_910b_99b3173d/
/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/attention_pipes_910b_retry_59259548/
/workspace/repos/paddle_ocr_vl_npu/tmp/11_mineru_2_5_pro_inference/attention_op_timeline_910b_5b009d43/
```
