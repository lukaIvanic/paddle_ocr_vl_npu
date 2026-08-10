#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from local_modeling_qwen3_reranker import (
    LocalQwen3RerankerConfig,
    LocalQwen3RerankerForCausalLM,
    build_left_padded_causal_bool_mask,
    build_left_padded_causal_mask,
)
from transformers_rerank import DEFAULT_TASK, build_inputs, format_instruction


class LocalQwen3RerankerRunner:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_length: int,
        batch_size: int,
        compile_forward: bool = False,
        attention_impl: str = "eager",
        ffn_weight_mode: str = "dense",
        prefill_chunk_size: int = 0,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.compile_forward = compile_forward
        self.attention_impl = attention_impl
        self.ffn_weight_mode = ffn_weight_mode
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.compiled_chunked_hidden_states = None
        if self.attention_impl not in {"eager", "prompt_flash_attention"}:
            raise ValueError(f"Unsupported attention_impl={self.attention_impl!r}")
        if self.attention_impl == "prompt_flash_attention" and self.dtype is not torch.float16:
            raise ValueError("prompt_flash_attention requires float16")
        if self.ffn_weight_mode not in {"dense", "w8a8", "all_w8a8"}:
            raise ValueError(f"Unsupported ffn_weight_mode={self.ffn_weight_mode!r}")
        if self.ffn_weight_mode != "dense" and self.dtype is not torch.float16:
            raise ValueError("W8A8 modes require float16")
        if self.prefill_chunk_size:
            if self.attention_impl != "prompt_flash_attention":
                raise ValueError("chunked prefill requires --attention-impl prompt_flash_attention")
            if self.prefill_chunk_size % 128 != 0:
                raise ValueError("310P-compatible prefill chunk size must be a multiple of 128")
            if self.max_length % self.prefill_chunk_size != 0:
                raise ValueError("--max-length must be divisible by --prefill-chunk-size")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, padding_side="left")
        self.false_token_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        self.true_token_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        self.config = LocalQwen3RerankerConfig.from_model_dir(self.model_dir)
        self.model = self.load_model()
        if compile_forward:
            from torch_npu.dynamo import torchair
            from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

            backend = torchair.get_npu_backend(compiler_config=CompilerConfig())
            if self.prefill_chunk_size:
                self.compiled_chunked_hidden_states = torch.compile(
                    self.model.forward_hidden_states_chunked,
                    backend=backend,
                    dynamic=False,
                    fullgraph=True,
                )
            else:
                self.model.forward_prepared_yes_no = torch.compile(
                    self.model.forward_prepared_yes_no,
                    backend=backend,
                    dynamic=False,
                    fullgraph=True,
                )

    def load_model(self) -> LocalQwen3RerankerForCausalLM:
        from safetensors.torch import load_file

        model = LocalQwen3RerankerForCausalLM(self.config, attention_impl=self.attention_impl)
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
        if self.ffn_weight_mode == "w8a8":
            from local_reranker_w8a8 import quantize_reranker_ffn_inplace

            quantize_reranker_ffn_inplace(model, out_dtype=self.dtype)
        elif self.ffn_weight_mode == "all_w8a8":
            from local_reranker_w8a8 import quantize_reranker_all_linears_inplace

            quantize_reranker_all_linears_inplace(model, out_dtype=self.dtype)
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        return model

    def encode_pairs(self, query: str, documents: list[str], task: str) -> dict[str, torch.Tensor]:
        if len(documents) != self.batch_size:
            raise ValueError(f"expected exactly batch_size={self.batch_size} documents, got {len(documents)}")
        pairs = [format_instruction(task, query, document) for document in documents]
        return build_inputs(self.tokenizer, pairs, max_length=self.max_length, device=self.device)

    def prepared_forward_inputs(
        self,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        layer_attention_mask = (
            build_left_padded_causal_bool_mask(attention_mask)
            if self.attention_impl == "prompt_flash_attention"
            else build_left_padded_causal_mask(attention_mask, self.dtype)
        )
        return position_ids, layer_attention_mask

    def logits_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.compiled_chunked_hidden_states is not None:
            hidden_states = self.compiled_chunked_hidden_states(
                input_ids,
                attention_mask,
                chunk_size=self.prefill_chunk_size,
            )
            return self.model.lm_head(hidden_states[:, -1])
        if not self.compile_forward:
            hidden_states = (
                self.model.forward_hidden_states_chunked(
                    input_ids,
                    attention_mask,
                    chunk_size=self.prefill_chunk_size,
                )
                if self.prefill_chunk_size
                else self.model.forward_hidden_states(input_ids, attention_mask)
            )
            return self.model.lm_head(hidden_states[:, -1])
        position_ids, additive_mask = self.prepared_forward_inputs(attention_mask)
        return self.model.forward_prepared(input_ids, position_ids, additive_mask)

    def score_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            yes_no_ids = torch.tensor(
                [self.false_token_id, self.true_token_id],
                device=self.device,
                dtype=torch.long,
            )
            yes_no_weight = self.model.lm_head.weight[yes_no_ids]
            if self.compiled_chunked_hidden_states is not None:
                hidden_states = self.compiled_chunked_hidden_states(
                    input_ids,
                    attention_mask,
                    chunk_size=self.prefill_chunk_size,
                )
                yes_no_logits = torch.nn.functional.linear(hidden_states[:, -1], yes_no_weight)
            elif self.compile_forward:
                position_ids, additive_mask = self.prepared_forward_inputs(attention_mask)
                yes_no_logits = self.model.forward_prepared_yes_no(
                    input_ids,
                    position_ids,
                    additive_mask,
                    yes_no_weight,
                )
            else:
                hidden_states = (
                    self.model.forward_hidden_states_chunked(
                        input_ids,
                        attention_mask,
                        chunk_size=self.prefill_chunk_size,
                    )
                    if self.prefill_chunk_size
                    else self.model.forward_hidden_states(input_ids, attention_mask)
                )
                yes_no_logits = torch.nn.functional.linear(hidden_states[:, -1], yes_no_weight)
            return torch.softmax(yes_no_logits, dim=-1)[:, 1]

    def calibrate_ffn_input_scales(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if self.ffn_weight_mode == "dense":
            return
        from local_reranker_w8a8 import calibrate_w8a8_input_scales

        with torch.inference_mode():
            calibrate_w8a8_input_scales(self.model, lambda: self.model.forward_hidden_states(input_ids, attention_mask))

    def score(self, query: str, documents: list[str], task: str = DEFAULT_TASK) -> torch.Tensor:
        inputs = self.encode_pairs(query, documents, task)
        return self.score_ids(inputs["input_ids"], inputs["attention_mask"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal local Qwen3 reranker model.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--query", default="What is the capital of China?")
    parser.add_argument(
        "--documents",
        nargs="+",
        default=[
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other.",
        ],
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--compile-forward", action="store_true")
    parser.add_argument("--attention-impl", choices=("eager", "prompt_flash_attention"), default="eager")
    parser.add_argument("--ffn-weight-mode", choices=("dense", "w8a8", "all_w8a8"), default="dense")
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=0,
        help="Sequential PromptFA prefill chunk size; 0 disables chunking, 310P-safe values are multiples of 128",
    )
    return parser.parse_args()


def parse_device(device_name: str) -> torch.device:
    if device_name.startswith("npu"):
        import torch_npu  # noqa: F401

    return torch.device(device_name)


def main() -> None:
    args = parse_args()
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = parse_device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    runner = LocalQwen3RerankerRunner(
        args.model_dir,
        device=device,
        dtype=dtype,
        max_length=args.max_length,
        batch_size=args.batch_size,
        compile_forward=args.compile_forward,
        attention_impl=args.attention_impl,
        ffn_weight_mode=args.ffn_weight_mode,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    inputs = runner.encode_pairs(args.query, args.documents, args.task)
    runner.calibrate_ffn_input_scales(inputs["input_ids"], inputs["attention_mask"])
    scores = runner.score_ids(inputs["input_ids"], inputs["attention_mask"])
    ranked = sorted(enumerate(scores.detach().float().cpu().tolist()), key=lambda item: item[1], reverse=True)
    print(f"shape={tuple(scores.shape)}")
    print(f"dtype={scores.dtype}")
    print(f"device={scores.device}")
    print(f"scores={scores.detach().float().cpu().tolist()}")
    print(f"ranked={ranked}")


if __name__ == "__main__":
    main()
