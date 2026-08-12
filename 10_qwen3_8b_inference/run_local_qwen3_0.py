#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch_npu
from safetensors.torch import load_file
from torch_npu.dynamo import torchair
from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig
from transformers import AutoTokenizer

from local_modeling_qwen3_0 import (
    DECODE_OPTIMIZATION_PRESETS,
    LocalQwen3Config,
    LocalQwen3ForCausalLM,
    cast_decode_linear_weights_to_nz,
    resolve_decode_optimization,
)


class LocalQwen30Runner:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        compile_decode: bool = False,
        compile_decode_dynamic: bool = True,
        decode_increfa_mode: str = "actual_seq_lengths",
        decode_optimization: str = "combined_apply",
        decode_linear_weight_format: str = "unchanged",
        static_kv_cache_len: int = 65536,
    ):
        if decode_increfa_mode not in {"mask", "actual_seq_lengths"}:
            raise ValueError(f"Unsupported decode_increfa_mode={decode_increfa_mode!r}")
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.compile_decode = compile_decode
        self.compile_decode_dynamic = compile_decode_dynamic
        self.decode_increfa_mode = decode_increfa_mode
        if decode_optimization not in DECODE_OPTIMIZATION_PRESETS:
            raise ValueError(
                f"Unsupported decode_optimization={decode_optimization!r}; "
                f"expected {tuple(DECODE_OPTIMIZATION_PRESETS)}"
            )
        if decode_linear_weight_format not in {"unchanged", "fractal_nz"}:
            raise ValueError(
                "decode_linear_weight_format must be 'unchanged' or "
                f"'fractal_nz', got {decode_linear_weight_format!r}"
            )
        self.decode_optimization = decode_optimization
        self.decode_linear_weight_format = decode_linear_weight_format
        self.static_kv_cache_len = int(static_kv_cache_len)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.config = LocalQwen3Config.from_model_dir(self.model_dir)
        self.model = self.load_model()
        self.decode_optimization_metadata = (
            self.model.prepare_decode_optimizations(
                cache_length=self.static_kv_cache_len,
            )
        )
        self.decode_linear_weight_metadata = {
            "requested": self.decode_linear_weight_format,
            "effective": "unchanged",
        }
        if self.decode_linear_weight_format == "fractal_nz":
            self.decode_linear_weight_metadata = {
                "requested": self.decode_linear_weight_format,
                "effective": "fractal_nz",
                **cast_decode_linear_weights_to_nz(self.model),
            }
        self.eager_decode = self.model.decode
        if compile_decode:
            compiler_config = CompilerConfig()
            if compile_decode_dynamic:
                compiler_config.experimental_config.tiling_schedule_optimize = True
            backend = torchair.get_npu_backend(compiler_config=compiler_config)
            self.model.decode = torch.compile(
                self.eager_decode,
                backend=backend,
                dynamic=compile_decode_dynamic,
                fullgraph=True,
            )

    def load_model(self) -> LocalQwen3ForCausalLM:
        weight_files = sorted(self.model_dir.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors weights found in {self.model_dir}")

        # Construct without allocating or initializing a second, float32 copy of
        # the model on the host. Allocate the final dtype directly on the target
        # device, then copy one checkpoint shard at a time. This matters for 8B:
        # the naive model + complete state_dict path temporarily retains roughly
        # 48 GiB of host tensors before the NPU copy even starts.
        with torch.device("meta"):
            model = LocalQwen3ForCausalLM(
                self.config,
                decode_increfa_mode=self.decode_increfa_mode,
                decode_optimization=self.decode_optimization,
            )
        model = model.to(dtype=self.dtype)
        model.to_empty(device=self.device)

        # ``inv_freq`` is a non-persistent buffer, so it is deliberately absent
        # from the checkpoint and must be restored after ``to_empty``.
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(
                    0,
                    self.config.head_dim,
                    2,
                    dtype=torch.float32,
                    device=self.device,
                )
                / self.config.head_dim
            )
        )
        model.rotary_emb.inv_freq.copy_(inv_freq)

        parameters = dict(model.named_parameters(remove_duplicate=False))
        expected = set(parameters)
        if self.config.tie_word_embeddings:
            expected.discard("lm_head.weight")
        loaded: set[str] = set()
        unexpected: list[str] = []

        for shard_index, weights_path in enumerate(weight_files, start=1):
            print(
                f"loading checkpoint shard {shard_index}/{len(weight_files)}: "
                f"{weights_path.name}",
                file=sys.stderr,
                flush=True,
            )
            shard = load_file(str(weights_path), device="cpu")
            with torch.no_grad():
                for checkpoint_key, value in shard.items():
                    if checkpoint_key.startswith("model."):
                        model_key = checkpoint_key[len("model.") :]
                    elif checkpoint_key == "lm_head.weight":
                        model_key = checkpoint_key
                    else:
                        unexpected.append(checkpoint_key)
                        continue
                    target = parameters.get(model_key)
                    if target is None:
                        unexpected.append(checkpoint_key)
                        continue
                    if target.shape != value.shape:
                        raise RuntimeError(
                            f"shape mismatch for {checkpoint_key}: "
                            f"checkpoint={tuple(value.shape)} model={tuple(target.shape)}"
                        )
                    target.copy_(value.to(device=self.device, dtype=target.dtype))
                    loaded.add(model_key)
            del shard

        if self.config.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        missing = sorted(expected - loaded)
        if missing or unexpected:
            raise RuntimeError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")

        model.eval()
        return model

    def mark_static_decode_state(
        self,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
    ) -> None:
        if not self.compile_decode_dynamic:
            return
        for cache in key_caches:
            torch._dynamo.mark_static(cache)
        for cache in value_caches:
            torch._dynamo.mark_static(cache)

    def decode_one(
        self,
        next_id: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        *,
        actual_seq_length: int | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if self.compile_decode_dynamic:
            torch._dynamo.mark_static(next_id)
            torch._dynamo.mark_static(cache_position)
        output_next_id = self.model.decode(
            next_id,
            cache_position,
            key_caches,
            value_caches,
            actual_seq_length,
        )
        return output_next_id, key_caches, value_caches

    def decode_one_eager(
        self,
        next_id: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        *,
        actual_seq_length: int | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        output_next_id = self.eager_decode(
            next_id,
            cache_position,
            key_caches,
            value_caches,
            actual_seq_length,
        )
        return output_next_id, key_caches, value_caches

    def decode_one_baseline(
        self,
        next_id: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        *,
        actual_seq_length: int | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        previous = self.model.decode_optimization
        self.model.decode_optimization = resolve_decode_optimization("baseline")
        try:
            output_next_id = self.eager_decode(
                next_id,
                cache_position,
                key_caches,
                value_caches,
                actual_seq_length,
            )
        finally:
            self.model.decode_optimization = previous
        return output_next_id, key_caches, value_caches

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        encoded = self.tokenizer(text, return_tensors="pt")
        return encoded["input_ids"].to(self.device)

    def make_initial_decode_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        # A size-1 slice can report contiguous while retaining the prompt's
        # larger row stride. Copy into the exact layout produced by argmax so
        # the first and later calls share one static graph specialization.
        decode_input = torch.empty(
            (input_ids.shape[0], 1),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        return decode_input.copy_(input_ids[:, -1:])

    def generate_ids(self, input_ids: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
        max_generated_tokens = min(max_new_tokens, max(0, self.static_kv_cache_len - input_ids.shape[1]))
        with torch.inference_mode():
            # Deliberate contract: prefill is cache-only and never supplies a
            # sampled "free" token. Re-feed the final prompt token through the
            # same decode graph used for every later step. This recomputes one
            # token, but gives static TorchAir one uniform decode contract.
            key_caches, value_caches = self.model.prefill(input_ids, static_kv_cache_len=self.static_kv_cache_len)
            self.mark_static_decode_state(key_caches, value_caches)
            decode_input_id = self.make_initial_decode_input(input_ids)
            generated = [input_ids]
            for decode_position in range(input_ids.shape[1] - 1, input_ids.shape[1] - 1 + max_generated_tokens):
                cache_position = torch.tensor([decode_position], device=input_ids.device, dtype=torch.long)
                actual_seq_length = decode_position + 1 if self.decode_increfa_mode == "actual_seq_lengths" else None
                decode_input_id, key_caches, value_caches = self.decode_one(
                    decode_input_id,
                    cache_position,
                    key_caches,
                    value_caches,
                    actual_seq_length=actual_seq_length,
                )
                generated.append(decode_input_id)
        return torch.cat(generated, dim=-1)

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        input_ids = self.encode_prompt(prompt)
        generated_ids = self.generate_ids(input_ids, max_new_tokens=max_new_tokens)
        new_tokens = generated_ids[:, input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal local Qwen 3.0 dense model.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Write a tiny Python function that adds two numbers.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--static-kv-cache-len", type=int, default=65536)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--compile-decode", action="store_true")
    parser.add_argument(
        "--compile-decode-dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--decode-increfa-mode",
        choices=("mask", "actual_seq_lengths"),
        default="actual_seq_lengths",
    )
    parser.add_argument(
        "--decode-optimization",
        choices=tuple(DECODE_OPTIMIZATION_PRESETS),
        default="combined_apply",
    )
    parser.add_argument(
        "--decode-linear-weight-format",
        choices=("unchanged", "fractal_nz"),
        default="unchanged",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    if device.type == "npu":
        if args.decode_linear_weight_format == "fractal_nz":
            torch.npu.config.allow_internal_format = True
        torch.npu.set_device(device)
        torch.npu.set_compile_mode(jit_compile=False)
    runner = LocalQwen30Runner(
        args.model_dir,
        device=device,
        dtype=dtype,
        compile_decode=args.compile_decode,
        compile_decode_dynamic=args.compile_decode_dynamic,
        decode_increfa_mode=args.decode_increfa_mode,
        decode_optimization=args.decode_optimization,
        decode_linear_weight_format=args.decode_linear_weight_format,
        static_kv_cache_len=args.static_kv_cache_len,
    )
    print(runner.generate(args.prompt, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
