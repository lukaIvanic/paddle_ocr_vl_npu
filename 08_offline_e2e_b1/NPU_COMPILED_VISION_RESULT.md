# NPU bucketed-vision result

Validated on an Atlas 800I A2 / 910B NPU with the PaddleOCR-VL-1.6 model. The
compiled boundary is the 27 vision encoder layers plus final LayerNorm. Patch
embedding, absolute-position interpolation, projector, text prefill, and the
continuous decode system remain outside that boundary.

## Static graph coverage

TorchAir successfully built and executed independent B=1 graphs for all
configured physical sequence lengths:

```text
16, 32, 64, 128, 256, 512, 1024, 2048
```

Each shape has a distinct compiler entrypoint and GE cache directory. This is
necessary because using one Python `forward` code object for every shape made
TorchDynamo classify later shapes as recompilations and TorchAir skip their
persistent caches. On the subsequent cache-reuse run, executing all eight
graphs took 5.71 seconds total (0.68--0.75 seconds per graph). Cold creation is
much slower and belongs to setup, not page throughput.

## Exact parity control

The one-region compiled and eager controls used the same page, layout result,
crop, `min_pixels=6272`, fp16 model, eager decode, and 64-token generation cap.
The generated token ID lists were exactly equal.

The crop produced 608 real vision tokens and selected the 1024 bucket:

| Path | Encoder + post-LN | Compiled input prep | Page wall |
| --- | ---: | ---: | ---: |
| eager manual attention | 52.65 ms | n/a | 0.989 s |
| compiled manual attention | 23.70 ms | 7.29 ms | 0.936 s |

The pure compiled boundary was 2.22x faster. Including mask, padding, and RoPE
preparation, the vision-encoder portion was 1.70x faster for this crop. The
single-crop page-wall difference is only a smoke result and should not be
treated as a stable E2E benchmark.

## Full-page integration

The final run used real PP-DocLayoutV3 layout inference, compiled vision,
compiled static B=4 decode, cache length 2048, maximum generation length 768,
and `min_pixels=6272`.

- Five of five regions completed; no page was partial.
- Four crops used compiled vision: bucket 64 once, 1024 once, and 2048 twice.
- One 3528-token crop exceeded the configured maximum and correctly used the
  eager unpadded overflow path.
- Compiled crops contained 3264 real rows in 5184 physical rows (62.96% useful).
  Including the unpadded overflow crop, the run-level useful fraction was
  6792 / 8712 = 77.96%.
- Page/run wall was 1.705 / 1.709 seconds.
- Decode produced 81 output tokens at 534.55 effective tok/s and 1181.63 raw
  fixed-arena tok/s.

The complete evidence is under
`tmp/08_offline_e2e_b1/compiled_vision_validation/`. The decisive files are the
paired `compiled_smoke_v3/run.json` and `eager_smoke/run.json`, plus
`compiled_full_page_b4/run.json`.
