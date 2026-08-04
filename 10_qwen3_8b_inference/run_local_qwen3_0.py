#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch_npu
from safetensors.torch import load_file
from torch_npu.dynamo import torchair
from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig
from transformers import AutoTokenizer

from local_modeling_qwen3_0 import LocalQwen3Config, LocalQwen3ForCausalLM


class LocalQwen30Runner:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        compile_decode: bool = False,
        compile_decode_dynamic: bool = False,
        decode_increfa_mode: str = "mask",
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
        self.static_kv_cache_len = int(static_kv_cache_len)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.config = LocalQwen3Config.from_model_dir(self.model_dir)
        self.model = self.load_model()
        if compile_decode:
            compiler_config = CompilerConfig()
            if compile_decode_dynamic:
                compiler_config.experimental_config.tiling_schedule_optimize = True
            backend = torchair.get_npu_backend(compiler_config=compiler_config)
            self.model.decode = torch.compile(
                self.model.decode,
                backend=backend,
                dynamic=compile_decode_dynamic,
                fullgraph=True,
            )

    def load_model(self) -> LocalQwen3ForCausalLM:
        model = LocalQwen3ForCausalLM(self.config, decode_increfa_mode=self.decode_increfa_mode)
        state = {}
        weight_files = sorted(self.model_dir.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors weights found in {self.model_dir}")
        for weights_path in weight_files:
            shard = load_file(str(weights_path), device="cpu")
            for key, value in shard.items():
                if key.startswith("model."):
                    state[key[len("model.") :]] = value
                elif key == "lm_head.weight":
                    state[key] = value

        missing, unexpected = model.load_state_dict(state, strict=False)
        if self.config.tie_word_embeddings and missing == ["lm_head.weight"]:
            model.lm_head.weight = model.embed_tokens.weight
            missing = []
        if missing or unexpected:
            raise RuntimeError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")

        model.to(device=self.device, dtype=self.dtype)
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
        return self.model.decode(
            next_id,
            cache_position,
            key_caches,
            value_caches,
            actual_seq_length,
        )

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        encoded = self.tokenizer(text, return_tensors="pt")
        return encoded["input_ids"].to(self.device)

    def generate_ids(self, input_ids: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
        max_generated_tokens = min(max_new_tokens, max(0, self.static_kv_cache_len - input_ids.shape[1]))
        with torch.inference_mode():
            key_caches, value_caches = self.model.prefill(input_ids, static_kv_cache_len=self.static_kv_cache_len)
            self.mark_static_decode_state(key_caches, value_caches)
            next_id = input_ids[:, -1:]
            generated = [input_ids]
            for decode_position in range(input_ids.shape[1] - 1, input_ids.shape[1] - 1 + max_generated_tokens):
                cache_position = torch.tensor([decode_position], device=input_ids.device, dtype=torch.long)
                actual_seq_length = decode_position + 1 if self.decode_increfa_mode == "actual_seq_lengths" else None
                next_id, key_caches, value_caches = self.decode_one(
                    next_id,
                    cache_position,
                    key_caches,
                    value_caches,
                    actual_seq_length=actual_seq_length,
                )
                generated.append(next_id)
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
    parser.add_argument("--compile-decode-dynamic", action="store_true")
    parser.add_argument("--decode-increfa-mode", choices=("mask", "actual_seq_lengths"), default="mask")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
        torch.npu.set_compile_mode(jit_compile=False)
    runner = LocalQwen30Runner(
        args.model_dir,
        device=device,
        dtype=dtype,
        compile_decode=args.compile_decode,
        compile_decode_dynamic=args.compile_decode_dynamic,
        decode_increfa_mode=args.decode_increfa_mode,
        static_kv_cache_len=args.static_kv_cache_len,
    )
    print(runner.generate(args.prompt, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
