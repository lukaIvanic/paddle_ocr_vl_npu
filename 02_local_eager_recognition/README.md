# Experiment 02: Local Eager Recognition

The PaddleOCR-VL-1.6 recognizer reimplemented in plain PyTorch, with no
Transformers imports. This is the correctness reference for the ladder:
every later experiment (compiled decode, batching, the persistent page
engine) derives from this implementation, and parity claims chain back to
it. It is deliberately not optimized — eager attention, a dynamic KV cache,
greedy decoding, batch size 1.

The bar it is held to: identical output token IDs to the Experiment 01
Transformers baseline on the same crop and prompt.

## Files

- `local_modeling_paddleocr_vl.py` — a replica of and drop-in replacement
  for the Transformers `modeling_paddleocr_vl.py`: the full model (vision
  encoder, adaptive projector, ERNIE decoder, mrope, greedy loop) in plain
  PyTorch. Intentionally not cleaned up or optimized — it stays close to
  the upstream source, dead branches and all, so it can be diffed against
  it and trusted as the baseline that later experiments develop from.
- `config.py` — the pinned architecture, hardcoded. Nothing is read from
  `config.json`; values are verified against the checkpoint (see the file
  docstring). Wrong checkpoints fail loudly at weight load.
- `run_local_recognition.py` — single-crop CLI. Reimplements the slow HF
  image preprocessing (`smart_resize`, patchify) and the chat-template
  prompt with hardcoded, checkpoint-verified constants. The slow processor
  is the parity target; the fast one rounds resizes differently.

## Run it

Needs an Ascend NPU. From the repo root:

```sh
python3 02_local_eager_recognition/run_local_recognition.py \
  --model /path/to/PaddleOCR-VL-1.6 \
  --crop crops/crop_01_text_block_en.png \
  --prompt "OCR:"
```

Expected on `crop_01_text_block_en.png`: the same text Experiment 01
prints, 253 input tokens, 79 new tokens.
