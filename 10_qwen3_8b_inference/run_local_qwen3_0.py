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
    LocalQwen3Config,
    LocalQwen3ForCausalLM,
)


class LocalQwen30Runner:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: torch.device,
        static_kv_cache_len: int = 4096,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = torch.float16
        self.config = LocalQwen3Config.from_model_dir(self.model_dir)
        expected_shape = (1024, 3072, 28, 16, 8, 128, 151936)
        actual_shape = (
            self.config.hidden_size,
            self.config.intermediate_size,
            self.config.num_hidden_layers,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            self.config.vocab_size,
        )
        if actual_shape != expected_shape:
            raise ValueError(
                "Experiment 10 now supports only the optimized Qwen3-0.6B "
                f"shape. Expected {expected_shape}, got {actual_shape}."
            )
        self.static_kv_cache_len = int(static_kv_cache_len)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = self.load_model()
        self.decode_optimization_metadata = (
            self.model.prepare_decode_optimizations(
                cache_length=self.static_kv_cache_len,
            )
        )
        self.eager_decode = self.model.decode
        backend = torchair.get_npu_backend(compiler_config=CompilerConfig())
        self.model.decode = torch.compile(
            self.eager_decode,
            backend=backend,
            dynamic=False,
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
            model = LocalQwen3ForCausalLM(self.config)
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

    def decode_one(
        self,
        next_id: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        output_next_id = self.model.decode(
            next_id,
            cache_position,
            key_caches,
            value_caches,
        )
        return output_next_id, key_caches, value_caches

    def decode_one_eager(
        self,
        next_id: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        output_next_id = self.eager_decode(
            next_id,
            cache_position,
            key_caches,
            value_caches,
        )
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
            decode_input_id = self.make_initial_decode_input(input_ids)
            generated = [input_ids]
            for decode_position in range(input_ids.shape[1] - 1, input_ids.shape[1] - 1 + max_generated_tokens):
                cache_position = torch.tensor([decode_position], device=input_ids.device, dtype=torch.long)
                decode_input_id, key_caches, value_caches = self.decode_one(
                    decode_input_id,
                    cache_position,
                    key_caches,
                    value_caches,
                )
                generated.append(decode_input_id)
        return torch.cat(generated, dim=-1)

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        input_ids = self.encode_prompt(prompt)
        generated_ids = self.generate_ids(input_ids, max_new_tokens=max_new_tokens)
        new_tokens = generated_ids[:, input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the optimized compiled Qwen3-0.6B decoder."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Write a tiny Python function that adds two numbers.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--static-kv-cache-len", type=int, default=4096)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("The optimized Experiment 10 path requires an Ascend NPU")
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    runner = LocalQwen30Runner(
        args.model_dir,
        device=device,
        static_kv_cache_len=args.static_kv_cache_len,
    )
    print(runner.generate(args.prompt, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
