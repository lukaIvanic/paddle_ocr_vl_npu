# Experiment 18: continuous UniRec + Paddle page pipeline

One PP-DocLayoutV3 frontend feeds two resident recognizers on one NPU.
Experiment 09 owns Paddle layout/cropping/assembly and inference; experiment
12 owns UniRec inference. This experiment owns routing and cross-model turns.

```sh
python 18_unirec_paddle_hybrid_pipeline/run_pipeline.py \
  --input /workspace/datasets/OmniDocBench/images \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --text-model unirec --table-model paddle --formula-model paddle \
  --unirec-model-path /path/to/unirec \
  --openocr-root /path/to/OpenOCR \
  --unirec-vision-cache /existing/vision/cache \
  --unirec-decode-cache /existing/decode/cache/parent \
  --limit 2 --output-dir /new/run/output
```

Set all three model arguments to one recognizer to load only that recognizer.
No YAML, label-specific overrides, or fallback recognizer exists. Unknown
prompts fail rather than being silently assigned to text. Chart/image/seal
handling remains the owned Paddle frontend's existing skip policy.

## Scheduling

Full arenas (including ready requests reserving slots) take priority. Ties
alternate UniRec/Paddle. A turn runs up to `--decode-steps` graph calls (32
initially), stopping earlier when slots need refill. When neither arena is
full, execute existing prefill groups for pending crops, otherwise advance
layout. Partial arenas drain only when submitted upstream work for that model
has exhausted. `PageInbox` supports independent arrivals and explicit close;
the benchmark submits filenames then closes input, not the recognition arenas.

No page cohorts and no corpus-wide text/table phases. Pending crop work is
consumed before more pages are examined. There is one compute owner; NPU work
is synchronized at cross-model yield boundaries. The engines retain pending
token-copy objects, slot epochs, EOS handling and KV update ordering.

The existing blocking decode entrypoints consume the new iterator boundaries
internally. Standalone callers do not opt into cross-model synchronization.
Model definitions and their compiled cache keys are unchanged by those seams.

## First implementation scope

Paddle keeps its production greedy vision / packed text prefill, B64/KV4096,
PSE-sentinel decode, and 0.5 text-crop scaling. UniRec uses the K20 production
vision kernels, packed S1024 text prefill, NZ decode and B128/C1320/S2048.
UniRec vision runs on the shared owner, not four concurrent vision lanes.
Both consume Paddle's region geometry/merge policy. UniRec receives unscaled
crop pixels and uses its own resize/normalization and output conversion.

Setup is separate from pipeline wall. First-use compilation remains visible
inside pipeline wall; no hot-throughput claim should be made from a cold smoke.
Per-crop traces retain the model, task prompt, token IDs and content. Do not
compare token IDs across tokenizers. The initial writer is synchronous.

CPU policy tests are not inference validation. 910B and 310P validation must
be recorded separately; no 310P result is claimed by this implementation.

```sh
python -m unittest discover -s 18_unirec_paddle_hybrid_pipeline/tests -v
```
