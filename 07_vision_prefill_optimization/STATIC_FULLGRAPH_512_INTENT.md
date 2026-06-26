# Static Fullgraph Fixed-S Vision Intent

This note records the intended next direction for Experiment 07 so future agents do not reinterpret it into a weaker or easier benchmark.

We are trying to build and validate a real static fullgraph vision-prefill path for PaddleOCR-VL, not a toy shortcut. The target path is:

- PromptFlashAttention in the vision encoder.
- PromptFA Q/K/V head-dim padding from the real PaddleOCR-VL vision `D=72` to an aligned call dimension such as `D=80`, with `scale_value` still based on `D=72`, and output sliced back to `D=72` before the attention output projection.
- Fixed physical visual sequence length for the compiled graph, for example 512 or 1024 tokens. Real visual tokens are preserved; dummy rows are appended only as padding.
- RoPE calculated inside the compiled graph, because keeping RoPE outside is not representative for the fullgraph static path we actually need.
- Manual fp32 LayerNorm math inside the graph to avoid the CANN/TorchAir fused `LayerNormV3` failure mode while preserving LayerNorm semantics.
- The sensible existing compile-safety fixes that already proved useful, including grouped matmul for LayerNorm-fed QKV/MLP-fc1 where appropriate.
- `torch.compile(..., fullgraph=True, dynamic=False)` or `torchair.inference.cache_compile(..., dynamic=False, ge_cache=True)` on NPU/TorchAir for the actual candidate.

The invariant is that real-token math must be preserved. Padding exists to make the graph static and batch/bucket-friendly; it is not allowed to become an accuracy hack. Real rows must not attend to padded rows, padded rows must not influence real rows, and downstream consumers must receive only the real visual rows.

For a fixed-S graph, crops with more than S real visual tokens cannot be silently resized, clipped, cropped, or truncated. They must be excluded from that specific bucket or routed to a larger bucket in a later experiment. Any benchmark must report how many crops are eligible, excluded, and why.

Do not weaken this into per-crop dynamic shapes, no-padding controls, RoPE precomputed outside the graph, or a graph that only compiles the attention kernel. Those are diagnostics only. The goal here is the representative full vision tower path with fixed physical token count and correctness checked against the stored eager PromptFA baseline.

The current validation ladder must include three distinct checks:

1. Static-visual tensor/logit compare against the stored eager PromptFA truth bundle.
2. Actual OCR generation compare where stored baseline `visual_features` and candidate static visual `visual_features` both flow through the same projector, text prefill, and static decode loop.
3. A no-resize batching audit that reports true same-grid/same-pixel-shape groups before anyone claims batched vision throughput.

Cache compilation is a cold-start optimization, not a math change. When enabled, it must be the same static visual callable and must report the GE cache directory/key, first-call timing, effective visual tok/s, and physical padded tok/s.

I understand this intention and will not fuck it up by changing the target into an easier benchmark.
