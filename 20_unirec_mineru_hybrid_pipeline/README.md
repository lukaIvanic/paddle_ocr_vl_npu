# Experiment 20: continuous PPv3 + UniRec/MinerU

Continuous integration validated on all 1,651 OmniDocBench images on one
910B2: **2.573 pages/s pipeline processing, 95.095 overall accuracy**.
Whole-process throughput including startup/setup/shutdown is 2.287 pages/s.
See [full-run evidence](references/910b_full1651_5c3450b8/README.md) and
[accuracy evaluation](references/910b_accuracy_5c3450b8/README.md).
Seven real crops also match the standalone MinerU stream token-for-token
through the new ready-KV path. No 310P validation or memory fit is claimed.

Experiment 18 remains the execution owner: its coordinator, PageInbox,
PageSource, persistent CPU workers, staged PPv3 layout/cropping, page assembly,
UniRec adapter and timing recorder are imported directly. There are no page
decode cohorts, batching timers or corpus-wide model phases. PPv3 is the only
layout model; no MinerU layout request is issued. Images/charts remain skipped.

Three CLI routes default to UniRec text and MinerU table/formula. All-UniRec and
all-MinerU routes are controls using the same PPv3 geometry, not reproductions
of their original full-page layout pipelines. No fallback recognizer is used.

## Engine ownership

MinerU uses experiment 11's FP16 model, manual FP32 vision LayerNorm + nn.Linear,
compiled PromptFA vision, packed text prefill, NZ decode and PSE-sentinel
IncreFA. Default decode is B32/KV4096; min/max pixels are 25088/602112 (the
3072 raw-vision-token cap). Existing compatible cache roots are reused.

The new experiment-11 `iter_decode_stream(cooperative=True)` exposes safe host
boundaries around the existing loop. The blocking `run_decode_stream` consumes
the same iterator with cooperative mode disabled, retaining standalone behavior.
Pending token copies, epochs, token history and KV controls stay in that loop.
An underfilled arena yields while submitted upstream work remains; it never
blocks the shared owner waiting for its own producer.

One persistent CPU worker reuses the existing MinerU client preparation and
CPU mRoPE. Page-local crop groups, bounded by ready storage, use the unchanged
vision-window and packed-prefill methods. A separate ready KV arena holds B32
rows at length4096 initially. Engine-owned grouped prefix admission copies into
the active arena and fences before releasing ready leases. The inactive suffix
remains masked; no generation limit is reduced. Memory tuning is deferred until
910B measurements. CPU capacity remains64 and is not an HBM limit.

MinerU outputs go through its existing single-block postprocessor before the
owned PPv3 page assembler. This boundary needs real table/formula validation,
especially for embedded table-image placeholders. Raw token IDs are retained.

UniRec's existing streamed mode is the experiment-20 default, with four vision
lanes and ready capacity64. Its internal stages can overlap; the inherited
owner joins/fences before MinerU or layout. Experiment18 defaults are unchanged.
Models are never repeatedly unloaded/reloaded.

## Reproduction

For the pull-only 310P agent, use the
[310P validation handoff](WORK_SERVER_310P_HYBRID_E2E.md): two-page smoke,
384-page memory/throughput validation, then Luka's approval for full1,651
and frozen accuracy evaluation. It preserves knowledge-bank processes=1,
explicit 310P cache paths and external device-memory sampling.

On the 910B container, pull committed source, then `source npu-setup`:

```bash
RUN_ROOT=tmp/20_unirec_mineru_hybrid_pipeline/smoke64 PAGE_LIMIT=64 \
  bash 20_unirec_mineru_hybrid_pipeline/run_910b.sh
```

The root must be new. The wrapper uses the existing process-tree/NPU-SMI
sampler, saves command/commit/log/exit status and checks complete drain. Use
PAGE_LIMIT=2 for initial bring-up, then64,384,1651 only after the preceding
stage is validated. Setup is separate; first-use work is visible in pipeline
wall. RUN_ROOT and cache ownership must not overlap another live run.

Local checks:

```bash
python -m unittest discover -s 20_unirec_mineru_hybrid_pipeline/tests -v
PYTHONPATH=11_mineru_2_5_pro_inference python -m unittest discover \
  -s 11_mineru_2_5_pro_inference -p test_streaming_decode.py -q
python -m unittest discover -s 18_unirec_paddle_hybrid_pipeline/tests -q
```

CPU tests check routing/default isolation, pause/resume token parity,
upstream-aware saturation/drain and prefix-copy isolation. They do not prove
TorchAir correctness. With the actual NPU environment, set
`READY_CACHE_TEST_DEVICE=npu:0` for the admission-copy test.

Real NPU crop-token control (uses the seven text/table/formula entries in
`crops/manifest.json`; the chart entry is outside this experiment's routes):

```bash
/workspace/venvs/mineru_pro_vllm_py312/bin/python \
  20_unirec_mineru_hybrid_pipeline/check_mineru_parity.py \
  --input crops --output-dir tmp/20_unirec_mineru_hybrid_pipeline/parity_new
```

The control executes the ordinary experiment11 stream first, then the hybrid
ready-prefill/yield path on the same B32 graph and crop group. It requires exact
raw token agreement, including EOS. It is not a quality score or speed test.

Timing uses experiment18's checked exclusive owner partition. MinerU's
generation/drain lifetime spans cooperative pauses and is labelled legacy,
not added to execution time. Decode device events and prefill events are
envelopes, not pure kernel-active time. Ready storage and active KV bytes are
reported independently alongside whole-device sampled peak memory.
