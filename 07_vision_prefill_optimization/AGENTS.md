# Experiment 07 Agent Notes

## Scope

Experiment 07 isolates PaddleOCR-VL vision-prefill behavior and performance.
Decode, continuous batching, page-level layout inference, and OmniDocBench
scoring belong to later experiments.

The current question is narrower than the original experiment: measure the
complete vision transformer stack on the small real crop shapes used by
Experiment08, comparing:

- manual attention versus NPU PromptFA;
- eager execution versus TorchAir full-graph compilation;
- effective real-token throughput versus physical padded-token throughput.

Experiment 07 must not import code from another experiment directory. It may
consume a stable data artifact, such as Experiment08's `run.json`, when the
artifact is treated only as crop geometry and provenance.

## Current Primary Benchmark

Use:

```sh
source npu-setup
cd /workspace/repos/paddle_ocr_vl_npu
bash 07_vision_prefill_optimization/run_npu_small_visual_encoder_matrix.sh
```

The runner uses `/usr/local/python3.12.13/bin/python3`, the local model at
`/workspace/models/PaddleOCR-VL-1.6`, and the committed five-page Experiment08
crop population. It writes logs and JSON under
`tmp/07_vision_prefill_optimization/` and compiler state under
`.runtime_cache/`.

The default first matrix is deliberately small:

```text
min_pixels=56448:  bucket (0, 384]
min_pixels=112896: buckets (0, 640] and (640, 768]
B=1
attention={manual,prompt_flash_attention}
backend={none,torchair}
```

Expand shapes or batch sizes only after the default matrix has produced valid
correctness and timing results.

## Measurement Contract

The headline timed boundary is exactly:

```text
encoder layers + post LayerNorm over [B, S_physical, hidden]
```

Before timing, the benchmark prepares crop pixels, patch embeddings, absolute
position embeddings, RoPE, padding masks, and all device tensors. The projector,
text prefill, and decode are outside the benchmark.

For each case:

1. Build inputs from the same real crop population.
2. Select the highest-fill crops inside the declared bucket.
3. Run the identical static wrapper eagerly as the numerical reference.
4. Measure the candidate's first call separately.
5. Run warmup forwards outside the measured region.
6. Measure repeated forward blocks with one device synchronization before and
   after each block.

Always report:

- mean forward latency and repeated-block dispersion;
- physical tokens/s: `B * S_physical / time`;
- effective tokens/s: `sum(S_real) / time`;
- useful-token fraction;
- first-call time and compile API;
- all correctness and nonfinite checks.

Do not use the NPU profiler for headline timing. Profiling is a separate
diagnostic run after a valid speed result exists.

## Fair Comparison Rule

Eager and compiled cases must use the same `BatchedStaticVisualEncoderModule`,
the same weights, the same attention implementation, the same LayerNorm and
Linear variants, the same physical shape, and the same real inputs. Compilation
must be the only intended difference.

The current primary defaults preserve the stock mathematical path:

```text
LayerNorm implementation: module
LayerNorm -> Linear mode: normal
PromptFA call head dimension: native D=72
```

Do not silently use the older `manual_fp32`, grouped-matmul, or PromptFA D=80
workarounds in a stock comparison. If the stock compiled path fails, record that
failure first, then run a clearly named workaround experiment.

TorchAir cache directories must distinguish at least attention implementation,
batch size, physical sequence length, dtype, LayerNorm/Linear mode, PromptFA
call dimension, layout, mask mode, and TorchAir mode. Never reuse a manual graph
as a PromptFA graph or vice versa.

## Correctness Gate

The benchmark compares compiled output against the same static wrapper run
eagerly. Every real row must pass `atol=rtol=0.1`, and the final physical output
must contain no nonfinite values. After timing, both outputs also pass through
the real adaptive projector and text prefill; image embeddings and prefill
logits must pass the same tolerance and the next-token argmax must match before
throughput is treated as valid.

When PromptFA uses a padded call head dimension, also compare its eager output
against native-head-dimension eager PromptFA. A D=80 workaround is not valid
merely because compiled D=80 matches eager D=80.

The old 310P/CANN 8.2 investigation found two risks that must be re-tested, not
assumed, on 910B/CANN 9:

- masked PromptFA used sparse mode 1 because mode 0 ignored the custom mask;
- compiled native D=72 PromptFA sometimes drifted or produced NaNs, while D=80
  fixed the attention-only reproduction.

Use `vision_prefill_bench.py probe-promptfa-mask` and the attention-only compile
probe only if the full encoder matrix exposes a PromptFA correctness problem.

## Padding Invariant

Static buckets append masked dummy rows. Real rows must not attend to dummy rows,
dummy rows must not attend to real rows, and dummy rows must be sliced before
downstream real-token consumers. Crops that exceed a bucket are routed to a
larger bucket; never resize, clip, or truncate them merely to make a graph fit.

## Historical Tools

The stored-baseline compare, generation checks, 310P probes, MSIT dumps, and
profiler scripts remain useful diagnostic tools. They are not the primary
small-shape speed workflow and several shell wrappers still contain old work-box
defaults. Pass current paths explicitly if running them.

The reusable implementation lives in:

- `validate_static_visual_batched_encoder.py`: static encoder module, prefix
  construction, and compile wrapper;
- `benchmark_small_visual_encoder.py`: one reliable real-crop timing case;
- `run_npu_small_visual_encoder_matrix.sh`: current matrix orchestration;
- `summarize_small_visual_encoder_matrix.py`: compiled/eager and
  PromptFA/manual comparisons.

## Anti-Cheat Rules

- Define and record the timed boundary before measuring.
- Use real crop geometry and the normal PaddleOCR-VL preprocessor.
- Keep compilation cold time separate from warm steady-state timing.
- Do enough repeated work that synchronization and timer noise do not dominate.
- Never report physical padded tokens/s as useful/effective tokens/s.
- Never report a fast result whose correctness gate failed.
- Keep case order and physical NPU visible in the run artifacts.
- Treat CUDA results as orchestration smoke only, never as NPU evidence.
- Do not let a missing model path trigger a download; fail fast.
