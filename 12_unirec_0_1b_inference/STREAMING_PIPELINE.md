# Bounded single-lane streaming pipeline

`run_opendoc_batched_unirec.py` can overlap worker-owned page prefill with one
continuous B128 decode arena. This is the low-RAM alternative to
`run_two_phase_batched_unirec.py`.

The worker processes reserve shared-memory bytes before packing a page. The
decoder returns those credits immediately after the page's last cross-KV row
has entered the NPU arena. The producer blocks at the byte limit. Queue depth
is not used as a memory estimate.

The decoder keeps its B128 arena full while the producer remains open. It
blocks for replacement rows instead of decoding a partial batch. It permits a
partial batch only after producer EOF.

## Accuracy-safe production configuration

Use the same model, layout cache, K20 cache parent, and decode cache parent as
the validated two-phase run. The important arguments are:

```text
--layout-process-workers 4
--prefill-in-layout-workers
--shared-cross-kv-budget-gib 3.5
--layout-execution torchair
--layout-dtype float32
--layout-reading-order-dtype float32
--layout-batch-size 2
--layout-cpu-threads 16
--layout-threshold 0.5
--vision-page-lookahead 4
--vision-bucket-preset 310p_k20_l4
--vision-focal-depthwise-rewrite constant_grouped_all
--vision-weight-format torchair_internal
--recognition-preprocess-threads 8
--recognition-input-contract compact_uint8_hwc
--vision-prefill-mode compiled_full_buckets
--text-prefill-mode compiled_packed_s1024
--decode-scheduling continuous
--decode-mode compiled_ifa
--decode-batch-size 128
--cross-cache-length 1320
--self-cache-length 2048
--max-length 2048
--decode-weight-format nz
--decode-lm-head-rows 57344
--decode-admission-prefetch-depth 0
--decode-live-arena-warmup-passes 2
```

Set `UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE` to the validated decode
cache parent when migrating from the two-phase runner. This prevents a new OM
namespace from being created only because the runner changed.

The streaming runner also defaults `TE_PARALLEL_COMPILER=1` and
`CANN_KNOWLEDGE_BANK_PROCESS_NUM=0` before importing Torch/NPU modules. Cached
serving graphs do not need CANN's default eight TBE compiler workers or its
eight-worker knowledge-bank service. Explicit environment values still win,
so cold-cache compilation experiments can opt back into parallel compilers.

## One-worker host-memory validation

On physical 910B2 NPU 7, the first 128 OmniDocBench pages reached 18.28 GiB
peak process-group PSS and 41 processes with CANN's default process services.
Cross-KV peaked at only 67.29 MiB; it was not the host-memory problem. Each of
the coordinator and prefill worker owned a CANN forkserver with eight TBE
compiler workers, eight knowledge-bank workers, and one knowledge-bank daemon.
The knowledge-bank service also owned a separate manager process.

With the serving defaults above, the same run used:

- 8.51 GiB peak process-group PSS;
- 9.49 GiB summed RSS and 7.81 GiB private memory;
- seven processes instead of 41;
- 13,063 MiB peak HBM;
- 67.29 MiB peak live cross-KV;
- 39.444 s measured pipeline wall time, or 3.245 pages/s.

All 128 Markdown files were byte-identical to the default-process-service
control. The OM inventory was unchanged. The control took 40.085 s, or 3.193
pages/s, so reducing the compiler services did not trade throughput for RAM.

The evidence root is:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/12_unirec_0_1b_inference/
w1_hostmem_te1_kb0_first128_29139b1_20260824T185433/
```

## 910B2 validation

Commit `b511fdb` passed the distribution-matched 128-page set on physical NPU
7. The measured phase took 27.030 s, or 4.735 pages/s. All 128 Markdown files
were byte-identical to the validated full-1651 output. The prefill producer
finished in 12.670 s. The decode graph took 18.718 s at 20.33k raw token-slots/s
and 9.27k effective tokens/s.

The 3.5 GiB shared-memory limit reached a 941,359,104-byte peak and returned to
zero. All 136 reservations were released. The measured run created no new OM
files and emitted no decode-recompile warning.

The remote evidence root is:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/12_unirec_0_1b_inference/
streaming_rep128_b511fdb_20260824T173546/
```
