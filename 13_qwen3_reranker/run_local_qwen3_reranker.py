#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch

from local_modeling_qwen3_reranker import (
    LocalQwen3RerankerConfig,
    LocalQwen3RerankerForCausalLM,
    PREFILL_OPTIMIZATION_PRESETS,
    build_310p_square_promptfa_mask,
    build_left_padded_causal_bool_mask,
    build_left_padded_causal_bool_mask_chunk,
    build_left_padded_causal_mask,
)
from transformers_rerank import DEFAULT_TASK, PREFIX, SUFFIX, build_inputs, format_instruction


def _source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "run_local_qwen3_reranker.py",
        "local_modeling_qwen3_reranker.py",
        "local_reranker_w8a8.py",
    ):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()[:12]


def _import_cache_compile():
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile
    except ImportError:
        from torchair.inference import cache_compile
    return cache_compile


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
        prefix_cache: bool = False,
        compile_cache_dir: str | Path | None = None,
        graph_warmups: int = 0,
        prefill_optimization: str = "baseline",
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
        self.prefix_cache = bool(prefix_cache)
        self.compile_cache_dir = (
            Path(compile_cache_dir).expanduser().resolve()
            if compile_cache_dir is not None
            else (Path.cwd() / ".runtime_cache" / "13_qwen3_reranker").resolve()
        )
        self.graph_warmups = int(graph_warmups)
        self.prefill_optimization = str(prefill_optimization)
        self.compiled_first_chunk = None
        self.compiled_next_chunk = None
        self.compiled_cached_suffix = None
        self._prefix_task: str | None = None
        self._prefix_input_ids: torch.Tensor | None = None
        self._prefix_attention_mask: torch.Tensor | None = None
        self._prefix_key_caches: tuple[torch.Tensor, ...] | None = None
        self._prefix_value_caches: tuple[torch.Tensor, ...] | None = None
        self._batched_prefix_key_caches: tuple[torch.Tensor, ...] | None = None
        self._batched_prefix_value_caches: tuple[torch.Tensor, ...] | None = None
        if self.attention_impl not in {"eager", "prompt_flash_attention"}:
            raise ValueError(f"Unsupported attention_impl={self.attention_impl!r}")
        if self.attention_impl == "prompt_flash_attention" and self.dtype is not torch.float16:
            raise ValueError("prompt_flash_attention requires float16")
        if self.ffn_weight_mode not in {"dense", "gate_up_w8a8", "w8a8", "all_w8a8"}:
            raise ValueError(f"Unsupported ffn_weight_mode={self.ffn_weight_mode!r}")
        if self.ffn_weight_mode != "dense" and self.dtype is not torch.float16:
            raise ValueError("W8A8 modes require float16")
        if self.graph_warmups < 0:
            raise ValueError("graph_warmups must be non-negative")
        if self.prefill_optimization not in PREFILL_OPTIMIZATION_PRESETS:
            raise ValueError(
                f"unsupported prefill_optimization={self.prefill_optimization!r}"
            )
        if self.prefix_cache:
            if not self.compile_forward:
                raise ValueError("prefix caching currently requires --compile-forward")
            if self.attention_impl != "prompt_flash_attention":
                raise ValueError("prefix caching requires prompt_flash_attention")
            if self.ffn_weight_mode not in {"dense", "gate_up_w8a8"}:
                raise ValueError("prefix caching supports dense or gate_up_w8a8 weights")
            if self.prefill_chunk_size != 128:
                raise ValueError("the initial static prefix-cache shape requires --prefill-chunk-size 128")
        if self.prefill_chunk_size:
            if self.attention_impl != "prompt_flash_attention":
                raise ValueError("chunked prefill requires --attention-impl prompt_flash_attention")
            if self.prefill_chunk_size % 128 != 0:
                raise ValueError("310P-compatible prefill chunk size must be a multiple of 128")
            if not self.prefix_cache and self.max_length % self.prefill_chunk_size != 0:
                raise ValueError("--max-length must be divisible by --prefill-chunk-size")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, padding_side="left")
        self.false_token_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        self.true_token_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        self.config = LocalQwen3RerankerConfig.from_model_dir(self.model_dir)
        self.model = self.load_model()
        self.model.set_prefill_optimization(self.prefill_optimization)
        if compile_forward:
            from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

            self.compile_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_compile = _import_cache_compile()
            config = CompilerConfig()
            source_hash = _source_hash()
            shape_key = (
                f"b{self.batch_size}_s{self.max_length}_c{self.prefill_chunk_size}_"
                f"h{self.config.hidden_size}_l{self.config.num_hidden_layers}_"
                f"opt{self.prefill_optimization}_fp16_src{source_hash}"
            )
            if self.prefix_cache:
                continuation_dir = self.compile_cache_dir / (
                    f"prefix_cached_suffix_b{self.batch_size}_q128_kv256_{shape_key}"
                )
                continuation_dir.mkdir(parents=True, exist_ok=True)
                self.compiled_cached_suffix = cache_compile(
                    self.model.forward_cached_suffix_prepared,
                    config=config,
                    dynamic=False,
                    cache_dir=str(continuation_dir),
                    ge_cache=True,
                    fullgraph=True,
                )
            elif self.prefill_chunk_size:
                prefix_dir = self.compile_cache_dir / f"prefix_b1_q128_{shape_key}"
                continuation_dir = self.compile_cache_dir / (
                    f"continuation_b{self.batch_size}_q128_kv256_{shape_key}"
                )
                prefix_dir.mkdir(parents=True, exist_ok=True)
                continuation_dir.mkdir(parents=True, exist_ok=True)
                self.compiled_first_chunk = cache_compile(
                    self.model.forward_first_chunk_prepared,
                    config=config,
                    dynamic=False,
                    cache_dir=str(prefix_dir),
                    ge_cache=True,
                    fullgraph=True,
                )
                self.compiled_next_chunk = cache_compile(
                    self.model.forward_next_chunk_prepared,
                    config=config,
                    dynamic=False,
                    cache_dir=str(continuation_dir),
                    ge_cache=True,
                    fullgraph=True,
                )
            else:
                full_dir = self.compile_cache_dir / f"full_yes_no_{shape_key}"
                full_dir.mkdir(parents=True, exist_ok=True)
                self.model.forward_prepared_yes_no = cache_compile(
                    self.model.forward_prepared_yes_no,
                    config=config,
                    dynamic=False,
                    cache_dir=str(full_dir),
                    ge_cache=True,
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
        if self.ffn_weight_mode == "gate_up_w8a8":
            from local_reranker_w8a8 import quantize_reranker_gate_up_inplace

            quantize_reranker_gate_up_inplace(model, out_dtype=self.dtype)
        elif self.ffn_weight_mode == "w8a8":
            from local_reranker_w8a8 import quantize_reranker_ffn_inplace

            quantize_reranker_ffn_inplace(model, out_dtype=self.dtype)
        elif self.ffn_weight_mode == "all_w8a8":
            from local_reranker_w8a8 import quantize_reranker_all_linears_inplace

            quantize_reranker_all_linears_inplace(model, out_dtype=self.dtype)
        model.to(device=self.device, dtype=self.dtype)
        if self.ffn_weight_mode != "dense":
            from local_reranker_w8a8 import restore_w8a8_scale_dtypes

            restore_w8a8_scale_dtypes(model)
        model.eval()
        return model

    def encode_pairs(self, query: str, documents: list[str], task: str) -> dict[str, torch.Tensor]:
        if len(documents) != self.batch_size:
            raise ValueError(f"expected exactly batch_size={self.batch_size} documents, got {len(documents)}")
        if self.prefix_cache:
            return self._encode_prefix_cached_pairs(query, documents, task)
        pairs = [format_instruction(task, query, document) for document in documents]
        return build_inputs(self.tokenizer, pairs, max_length=self.max_length, device=self.device)

    def _encode_prefix_cached_pairs(
        self,
        query: str,
        documents: list[str],
        task: str,
    ) -> dict[str, torch.Tensor]:
        block_size = self.prefill_chunk_size
        fixed_body_prefix = f"<Instruct>: {task}\n<Query>:"
        fixed_prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        fixed_body_prefix_ids = self.tokenizer.encode(fixed_body_prefix, add_special_tokens=False)
        prefix_ids = fixed_prefix_ids + fixed_body_prefix_ids
        suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        if len(prefix_ids) > block_size:
            raise ValueError(f"fixed prefix uses {len(prefix_ids)} tokens, exceeding block_size={block_size}")
        expected_max_length = len(prefix_ids) + block_size
        if self.max_length != expected_max_length:
            raise ValueError(
                "the initial static prefix-cache shape requires max_length="
                f"prefix_tokens({len(prefix_ids)}) + continuation_block({block_size}) = "
                f"{expected_max_length}, got {self.max_length}"
            )

        prefix_padding = block_size - len(prefix_ids)
        prefix_input_ids = [int(self.tokenizer.pad_token_id)] * prefix_padding + prefix_ids
        prefix_attention = [0] * prefix_padding + [1] * len(prefix_ids)
        prefix_input_tensor = torch.tensor(
            [prefix_input_ids], device=self.device, dtype=torch.long
        )
        prefix_attention_tensor = torch.tensor(
            [prefix_attention], device=self.device, dtype=torch.long
        )
        self._ensure_prefix_cache(
            task=task,
            input_ids=prefix_input_tensor,
            attention_mask=prefix_attention_tensor,
        )

        body_capacity = self.max_length - len(fixed_prefix_ids) - len(suffix_ids)
        continuation_body_capacity = block_size - len(suffix_ids)
        continuation_rows: list[list[int]] = []
        continuation_attention_rows: list[list[int]] = []
        for document in documents:
            full_body = format_instruction(task, query, document)
            full_body_ids = self.tokenizer(
                full_body,
                padding=False,
                truncation=True,
                return_attention_mask=False,
                max_length=body_capacity,
            )["input_ids"]
            continuation_text = f" {query}\n<Document>: {document}"
            continuation_body_ids = self.tokenizer(
                continuation_text,
                add_special_tokens=False,
                truncation=True,
                max_length=continuation_body_capacity,
            )["input_ids"]
            if fixed_body_prefix_ids + continuation_body_ids != full_body_ids:
                raise RuntimeError(
                    "prefix-cache token boundary changed the reference token IDs; "
                    "refusing to cache this prompt"
                )
            continuation_ids = continuation_body_ids + suffix_ids
            continuation_padding = block_size - len(continuation_ids)
            continuation_rows.append(
                [int(self.tokenizer.pad_token_id)] * continuation_padding + continuation_ids
            )
            continuation_attention_rows.append(
                [0] * continuation_padding + [1] * len(continuation_ids)
            )
        return {
            "input_ids": torch.tensor(continuation_rows, device=self.device, dtype=torch.long),
            "attention_mask": torch.tensor(
                continuation_attention_rows, device=self.device, dtype=torch.long
            ),
        }

    def _ensure_prefix_cache(
        self,
        *,
        task: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        if self._prefix_key_caches is not None:
            if task != self._prefix_task:
                raise ValueError("one runner cannot reuse a prefix cache across different tasks")
            if not torch.equal(input_ids, self._prefix_input_ids) or not torch.equal(
                attention_mask, self._prefix_attention_mask
            ):
                raise RuntimeError("fixed prefix tensors changed after prefix-cache creation")
            return
        position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        prefix_mask = build_left_padded_causal_bool_mask(attention_mask)
        started = time.perf_counter()
        selected_optimization = self.model.prefill_optimization
        self.model.set_prefill_optimization("baseline")
        try:
            with torch.inference_mode():
                key_caches, value_caches = self.model.build_prefix_cache_eager(
                    input_ids,
                    position_ids,
                    prefix_mask,
                )
        finally:
            self.model.set_prefill_optimization(selected_optimization)
        if self.device.type == "npu":
            torch.npu.synchronize()
        print(
            "PREFIX_CACHE_EAGER_BUILD "
            f"tokens={int(attention_mask.sum().item())} wall_s={time.perf_counter() - started:.3f}",
            flush=True,
        )
        self._prefix_task = task
        self._prefix_input_ids = input_ids
        self._prefix_attention_mask = attention_mask
        self._prefix_key_caches = key_caches
        self._prefix_value_caches = value_caches
        prepared_keys, prepared_values = self.model.prepare_prefix_caches(
            key_caches,
            value_caches,
        )
        self._batched_prefix_key_caches = tuple(
            cache.expand(self.batch_size, -1, -1, -1).contiguous() for cache in prepared_keys
        )
        self._batched_prefix_value_caches = tuple(
            cache.expand(self.batch_size, -1, -1, -1).contiguous() for cache in prepared_values
        )

    def forward_prefix_cached_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.compiled_cached_suffix is None:
            raise RuntimeError("compiled continuation graph is not initialized")
        if (
            self._prefix_attention_mask is None
            or self._batched_prefix_key_caches is None
            or self._batched_prefix_value_caches is None
        ):
            raise RuntimeError("prefix KV cache has not been created")
        batch = int(input_ids.shape[0])
        if batch != self.batch_size:
            raise ValueError(f"expected static batch_size={self.batch_size}, got {batch}")
        prefix_attention = self._prefix_attention_mask.expand(batch, -1)
        combined_attention = torch.cat((prefix_attention, attention_mask), dim=1)
        position_ids = combined_attention.to(dtype=torch.long).cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        continuation_mask = build_left_padded_causal_bool_mask_chunk(
            combined_attention,
            query_start=self.prefill_chunk_size,
            query_end=2 * self.prefill_chunk_size,
        )
        if self.model.prefill_optimization.prebuilt_square_mask:
            continuation_mask = build_310p_square_promptfa_mask(continuation_mask)
        hidden_states = self.compiled_cached_suffix(
            input_ids,
            position_ids[:, self.prefill_chunk_size :].contiguous(),
            continuation_mask,
            self._batched_prefix_key_caches,
            self._batched_prefix_value_caches,
        )
        return hidden_states

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
        if self.prefix_cache:
            hidden_states = self.forward_prefix_cached_hidden_states(input_ids, attention_mask)
            return self.model.lm_head(hidden_states[:, -1])
        if self.compiled_first_chunk is not None:
            hidden_states = self.forward_hidden_states_chunked_compiled(input_ids, attention_mask)
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
            if self.prefix_cache:
                hidden_states = self.forward_prefix_cached_hidden_states(input_ids, attention_mask)
                yes_no_logits = torch.nn.functional.linear(hidden_states[:, -1], yes_no_weight)
            elif self.compiled_first_chunk is not None:
                hidden_states = self.forward_hidden_states_chunked_compiled(input_ids, attention_mask)
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

    def forward_hidden_states_chunked_compiled(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.compiled_first_chunk is None or self.compiled_next_chunk is None:
            raise RuntimeError("compiled chunk-step graphs are not initialized")
        position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        sequence_length = int(input_ids.shape[1])
        key_caches = None
        value_caches = None
        hidden_states = None
        for query_start in range(0, sequence_length, self.prefill_chunk_size):
            query_end = query_start + self.prefill_chunk_size
            chunk_mask = build_left_padded_causal_bool_mask_chunk(
                attention_mask,
                query_start=query_start,
                query_end=query_end,
            )
            chunk_input_ids = input_ids[:, query_start:query_end].contiguous()
            chunk_position_ids = position_ids[:, query_start:query_end].contiguous()
            if key_caches is None:
                hidden_states, key_caches, value_caches = self.compiled_first_chunk(
                    chunk_input_ids,
                    chunk_position_ids,
                    chunk_mask,
                )
            else:
                hidden_states, key_caches, value_caches = self.compiled_next_chunk(
                    chunk_input_ids,
                    chunk_position_ids,
                    chunk_mask,
                    key_caches,
                    value_caches,
                )
        if hidden_states is None:
            raise RuntimeError("compiled chunked prefill produced no chunks")
        return hidden_states

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
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(".runtime_cache/13_qwen3_reranker"),
    )
    parser.add_argument("--graph-warmups", type=int, default=2)
    parser.add_argument(
        "--prefill-optimization",
        choices=tuple(PREFILL_OPTIMIZATION_PRESETS),
        default="baseline",
        help="Compiled prefix-cache suffix optimization preset.",
    )
    parser.add_argument("--attention-impl", choices=("eager", "prompt_flash_attention"), default="eager")
    parser.add_argument(
        "--ffn-weight-mode",
        choices=("dense", "gate_up_w8a8", "w8a8", "all_w8a8"),
        default="dense",
    )
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
        torch.npu.set_compile_mode(jit_compile=False)
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
        prefix_cache=args.prefix_cache,
        compile_cache_dir=args.compile_cache_dir,
        graph_warmups=args.graph_warmups,
        prefill_optimization=args.prefill_optimization,
    )
    inputs = runner.encode_pairs(args.query, args.documents, args.task)
    runner.calibrate_ffn_input_scales(inputs["input_ids"], inputs["attention_mask"])
    for warmup_index in range(args.graph_warmups):
        started = time.perf_counter()
        runner.score_ids(inputs["input_ids"], inputs["attention_mask"])
        if device.type == "npu":
            torch.npu.synchronize()
        print(
            "RERANKER_GRAPH_WARMUP "
            f"pass={warmup_index + 1}/{args.graph_warmups} "
            f"wall_s={time.perf_counter() - started:.3f}",
            flush=True,
        )
    scores = runner.score_ids(inputs["input_ids"], inputs["attention_mask"])
    ranked = sorted(enumerate(scores.detach().float().cpu().tolist()), key=lambda item: item[1], reverse=True)
    print(f"shape={tuple(scores.shape)}")
    print(f"dtype={scores.dtype}")
    print(f"device={scores.device}")
    print(f"scores={scores.detach().float().cpu().tolist()}")
    print(f"ranked={ranked}")


if __name__ == "__main__":
    main()
