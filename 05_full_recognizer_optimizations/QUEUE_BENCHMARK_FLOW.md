# Experiment 5 Queue Benchmark Flow

`bench_recognizer_queue.py` measures a recognizer-only serving shape for real
OmniDocBench crops that are already selected in `crops/hotswap_100_manifest.json`.
It does not run page layout detection. It does include crop file read/decode,
local PaddleOCR-VL preprocessing, native-resolution vision encoding, adaptive
MLP projection, text prefill, static-cache decode, output postprocess, and
correctness validation.

The queue benchmark defaults to hot-swap decode through `ACTIVE_BATCH_SIZE`.
It first builds one ready-state bank for real crops, then reuses one active
compiled decode batch and swaps finished slots from that bank. It stays
padding-free: hot-swap rejects `ACTIVE_BATCH_SIZE > NUM_ITEMS`, and fixed
cohorts remain available only through `DECODE_SCHEDULE=fixed_cohort` as an
explicit baseline.

## Data Flow

```mermaid
flowchart LR
    A["Manifest rows<br/>crop path, prompt, metadata"] --> B["CPU input build"]
    B --> B1["Image read/decode<br/>PIL RGB crop"]
    B1 --> B2["Resize + normalize + patchify<br/>pixel_values: [N, 3, 14, 14]<br/>image_grid_thw: [1, 3]"]
    B2 --> B3["Tokenizer + image-token prompt<br/>input_ids: [1, T]<br/>attention_mask: [1, T]"]
    B3 --> C["Cache preflight<br/>required = T + max_new_tokens - 1"]
    C -->|overflow| CERR["Stop with JSON error<br/>cache_length_too_small"]
    C -->|fits| D["Model setup"]
    D --> D1["Load local model fp16"]
    D1 --> D2["Decode weight format<br/>NPU: FRACTAL_NZ<br/>CUDA: skipped"]
    D2 --> D3["Compile/wrap decode<br/>NPU: TorchAir cache_compile<br/>CUDA: raw eager or torch.compile"]
    D3 --> E["Ready bank build<br/>one crop at a time"]
    E --> E1["Device transfer"]
    E1 --> E2["Vision embeddings + 27-layer encoder<br/>hidden: [N, 1152]"]
    E2 --> E3["Post LN + adaptive MLP projector<br/>image_embeds: [N/4, 1024]"]
    E3 --> E4["Text token embeddings + image embed scatter<br/>inputs_embeds: [1, T, 1024]"]
    E4 --> E5["mRoPE indices + static KV cache alloc"]
    E5 --> E6["Text prefill writes KV cache<br/>per layer K/V: [1, 2, cache_length, 128]"]
    E6 --> E7["LM head argmax<br/>ReadyItem: cache, rope_deltas, cache_position, next_token"]
    E7 --> F["Ready bank<br/>all real crop states on device"]
    F --> F1["Hot-swap decode queue<br/>one active batch of B slots"]
    F1 --> F2["Batched static decode calls<br/>next_token[B,1] -> logits -> argmax"]
    F2 --> F3["Overlap token-row copy<br/>NPU stream -> pinned CPU row"]
    F3 --> F4["CPU per-item token history<br/>EOS/length-cap bookkeeping"]
    F4 --> F5["Swap all finished slots<br/>batched index_copy_ for K/V and slot controls"]
    F5 --> G["Postprocess"]
    G --> G1["Materialize token rows<br/>hot-swap rows already CPU; fixed cohort copies device tensors"]
    G1 --> G2["Trim at EOS + tokenizer.decode"]
    G2 --> H["Validation"]
    H --> H1["Direct local static generation per item"]
    H1 --> H2["Compare trimmed token IDs<br/>check invalid IDs"]
    H2 --> I["JSON report + printed summary"]
```

## Code Flow

```mermaid
sequenceDiagram
    participant CLI as parse_args/main
    participant Inputs as build_queue_inputs
    participant Model as LocalPaddleOCRVL
    participant Ready as build_ready_item
    participant Decode as static_hotswap_decode_loop
    participant Output as materialize/validate

    CLI->>Inputs: load manifest and selected crop rows
    Inputs->>Inputs: preprocess_crop_timed for each crop
    Inputs->>Inputs: build_inputs tokenizer prompt for each crop
    Inputs-->>CLI: list[QueueInput], input timing summary
    CLI->>CLI: prompt_token_summary cache preflight
    alt any item needs more than cache_length
        CLI-->>CLI: print valid JSON error and exit
    else all items fit
        CLI->>Model: from_pretrained(dtype, device)
        CLI->>Model: cast_decode_linear_weights_to_nz
        CLI->>Model: make_flat_static_decode_module
        CLI->>Decode: compile_decode_module and one warm call
        loop selected QueueInput rows
            CLI->>Ready: build_ready_item
            Ready->>Model: vision embeddings, encoder, post LN
            Ready->>Model: adaptive MLP projector
            Ready->>Model: text embedding + image scatter
            Ready->>Model: mRoPE + static cache allocate
            Ready->>Model: text prefill + lm_head argmax
            Ready-->>CLI: ReadyItem
        end
        CLI->>Decode: make_ready_bank from all ReadyItem rows
        CLI->>Decode: static_hotswap_decode_loop
        Decode->>Decode: load first B real items into active slots
        loop until all items complete
            Decode->>Decode: call compiled/raw static decode for active slots
            Decode->>Decode: copy sampled token row to CPU
            Decode->>Decode: append tokens to per-item CPU rows
            Decode->>Decode: swap every finished slot from the ready bank
        end
        Decode-->>CLI: HotSwapDecodeResult
        CLI->>Output: materialize_hotswap_item or materialize_decoded_item
        CLI->>Output: validate_outputs against generate_ids_static
        Output-->>CLI: correctness and text samples
        CLI-->>CLI: print JSON and runner summary
    end
```

## Main Data Objects

`QueueInput` is host-side prepared data for one crop. It contains the manifest
entry, absolute crop path, prompt, `input_ids`, `attention_mask`, `pixel_values`,
`image_grid_thw`, and CPU input-build timings. The important shapes are:

- `pixel_values`: `[vision_tokens, 3, 14, 14]`
- `image_grid_thw`: `[1, 3]`, with `[grid_t, grid_h, grid_w]`
- `input_ids`: `[1, prompt_tokens]`
- `attention_mask`: `[1, prompt_tokens]`

`ReadyItem` is device-side state after vision/projector/prefill. It contains the
per-item static cache, `rope_deltas`, `next_cache_position`, first `next_token`,
stage timings, `vision_tokens`, and `projected_image_tokens`. For this model,
projected image tokens are normally `vision_tokens / 4` because the adaptive MLP
connector performs a 2x2 spatial merge.

`ReadyBank` is the device-side collection of all prefills selected for the run.
It concatenates real per-crop K/V cache rows, `rope_deltas`,
`next_cache_position`, and first `next_token` across the item dimension. This
is the source for hot-swap slot loads.

`HotSwapDecodeResult` is the hot-swap scheduler output. Token history is stored
as CPU rows copied from sampled token rows, matching the experiment-4 optimized
path. Slot control and K/V swaps use batched `index_copy_`, not slice `fill_` or
per-slot scalar writes.

`DecodedDeviceItem` is still used by the optional fixed-cohort baseline. It
keeps a list of one-token tensors so the measured fixed `decode_queue` phase
does not include tokenizer decode or bulk CPU token copies.

`ReadyCohort` is a padding-free batch of `ReadyItem` rows. It concatenates
single-item K/V cache rows, `rope_deltas`, `next_cache_position`, and
`next_token` across batch dimension. The last cohort can be smaller than
`ACTIVE_BATCH_SIZE`.

`DecodedItem` is the CPU-side output after materialization. It contains raw token
IDs, EOS-trimmed token IDs, generated text, EOS/length-cap flags, and
postprocess timing.

## Filters And Hard Checks

The benchmark deliberately fails early instead of silently changing the serving
shape:

- `num_items`, `max_new_tokens`, and `cache_length` must be positive.
- `active_batch_size` must be positive.
- Every selected crop file must exist.
- `cache_length` must cover `input_tokens + max_new_tokens - 1` for every item.
- Image-token count and projected-image-embedding count must match exactly.
- Validation is required by default (`validation_items=-1`) and compares every
  queued output against direct local static generation.
- Token IDs are checked for invalid values outside the model vocabulary.

There is no hidden text padding path in this benchmark. Different crop/prompt
lengths become different ready states. The default hot-swap decode queue loads
only real ready states into active slots; fixed cohorts group real ready states
only when `DECODE_SCHEDULE=fixed_cohort` is requested.

## Timing Buckets

`setup_timing_s` is reported but excluded from throughput. It includes model
load, device-specific decode weight format handling, compile/wrapper creation,
and the first decode call that may trigger compilation or cache load. On NPU,
weight format handling preconverts decoder linear weights to FRACTAL_NZ. On
CUDA, the same step is skipped and reported as such.

`phase_timing_s.input_build_wall` is host-side crop processing and prompt
construction for all selected crops.

`phase_timing_s.ready_bank_build` is sequential per-crop device transfer,
native-resolution vision encoder, adaptive MLP projector, text prefill, and
first-token LM head.

`phase_timing_s.decode_queue` is only text decode over already-ready states. In
the default hot-swap schedule this includes active-slot setup, the hot-swap
decode loop, token-row copies for EOS bookkeeping, and batched row swaps. It
does not include vision/projector/text prefill or tokenizer postprocess.

`pipeline_stage_timing_summary_s` is a clearer set of aliases over the existing
timers:

- `vision_prefill`: vision prepare, native-resolution visual encoder, and
  adaptive MLP projector.
- `text_prefill`: text token embedding, image embed scatter, mRoPE/static-cache
  setup, text decoder prefill, first-token LM head, and argmax.
- `text_decode`: the selected decode queue, hot-swap by default.

`phase_timing_s.decode_output_postprocess` is final token-row materialization,
EOS trimming, and tokenizer decode. In hot-swap mode, per-step sampled-token
row copies are part of `decode_queue` because they drive EOS detection and slot
replacement.

`phase_timing_s.validation` is direct static generation used for correctness.
It is not included in throughput.

## Device Differences

On NPU, the decode path uses TorchAir cache compile, IncreFlashAttention,
`torch_npu.scatter_update_` for decode KV writes, and overlapped sampled-token
row copies on a second NPU stream for hot-swap EOS bookkeeping. Decoder linear
weights are preconverted to FRACTAL_NZ before compile. Hot-swap slot loads use
batched `index_copy_` for K/V rows and slot-control tensors.

On CUDA, TorchAir and `torch_npu` operators are not used. The same local model
falls back to manual attention and per-row KV cache writes. Use `raw_eager` for
the closest correctness smoke, or a regular `torch.compile` backend such as
`inductor` for CUDA compiler smoke. CUDA numbers are useful for debugging the
Python/data flow, but they are not Ascend throughput evidence.
