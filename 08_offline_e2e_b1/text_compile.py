"""Bucketed static TorchAir execution for PaddleOCR-VL text prefill.

Token embedding, multimodal embedding scatter, the LM head, and greedy argmax
remain eager.  This module pads the already-built multimodal embedding prefix
to a static bucket and compiles only the text transformer plus in-place KV
cache population.  The graph returns the hidden state at the last real prompt
token, so padded query rows never become observable.
"""

from __future__ import annotations

import hashlib
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from local_modeling_paddleocr_vl import (
    LocalPaddleOCRVLForConditionalGeneration,
    LocalPaddleOCRVLStaticCache,
    get_text_softmax_dtype_mode,
)
from probe_static_compile import (
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from timing import synchronize


DEFAULT_TEXT_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)
TEXT_BACKEND_CHOICES = ("raw_eager", "torchair")


def parse_text_buckets(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if not pieces:
            raise ValueError("text compile buckets cannot be empty")
        try:
            buckets = tuple(int(piece) for piece in pieces)
        except ValueError as exc:
            raise ValueError(f"invalid text compile buckets: {value!r}") from exc
    else:
        buckets = tuple(int(item) for item in value)
    if not buckets:
        raise ValueError("text compile buckets cannot be empty")
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError("every text compile bucket must be positive")
    if tuple(sorted(set(buckets))) != buckets:
        raise ValueError("text compile buckets must be unique and strictly increasing")
    return buckets


def select_text_bucket(real_seq_len: int, buckets: Iterable[int]) -> int | None:
    real_seq_len = int(real_seq_len)
    if real_seq_len <= 0:
        raise ValueError("real text sequence length must be positive")
    for bucket in buckets:
        if real_seq_len <= int(bucket):
            return int(bucket)
    return None


class StaticTextPrefill(torch.nn.Module):
    """Text transformer prefill with flat mutable cache inputs."""

    def __init__(self, model: LocalPaddleOCRVLForConditionalGeneration):
        super().__init__()
        self.text_model = model.model
        self.num_layers = int(model.config.text_config.num_hidden_layers)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        last_token_index: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        key_caches = tuple(flat_cache_tensors[: self.num_layers])
        value_caches = tuple(flat_cache_tensors[self.num_layers :])
        cache = LocalPaddleOCRVLStaticCache(
            key_caches=key_caches,
            value_caches=value_caches,
            cache_length=int(key_caches[0].shape[2]),
        )
        hidden_states = self.text_model.forward_prefill_static(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
        )
        return torch.index_select(hidden_states, 1, last_token_index)


def unique_bucket_forward(
    module: StaticTextPrefill,
    bucket: int,
) -> Callable[..., torch.Tensor]:
    """Give each static bucket a distinct Dynamo code object."""

    original = module.forward.__func__
    name = f"text_prefill_bucket_{int(bucket)}"
    code = original.__code__.replace(co_name=name)
    function = types.FunctionType(
        code,
        original.__globals__,
        name,
        original.__defaults__,
        original.__closure__,
    )
    function.__annotations__ = dict(original.__annotations__)
    function.__kwdefaults__ = original.__kwdefaults__
    return types.MethodType(function, module)


@dataclass(frozen=True)
class PreparedTextBucket:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    last_token_index: torch.Tensor
    real_seq_len: int
    physical_seq_len: int


def prepare_text_bucket(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    physical_seq_len: int,
) -> PreparedTextBucket:
    if inputs_embeds.ndim != 3 or int(inputs_embeds.shape[0]) != 1:
        raise ValueError(
            "compiled text prefill expects B=1 embeddings shaped [1, S, H], "
            f"got {tuple(inputs_embeds.shape)}"
        )
    real_seq_len = int(inputs_embeds.shape[1])
    physical_seq_len = int(physical_seq_len)
    if real_seq_len > physical_seq_len:
        raise ValueError(
            f"real text sequence {real_seq_len} exceeds bucket {physical_seq_len}"
        )
    if tuple(attention_mask.shape) != (1, real_seq_len):
        raise ValueError(
            f"attention_mask must have shape {(1, real_seq_len)}, "
            f"got {tuple(attention_mask.shape)}"
        )
    if tuple(position_ids.shape) != (3, 1, real_seq_len):
        raise ValueError(
            f"position_ids must have shape {(3, 1, real_seq_len)}, "
            f"got {tuple(position_ids.shape)}"
        )

    pad_tokens = physical_seq_len - real_seq_len
    padded_embeds = F.pad(inputs_embeds, (0, 0, 0, pad_tokens)).contiguous()
    padded_mask = F.pad(attention_mask, (0, pad_tokens), value=0).contiguous()
    # get_rope_index uses position 1 for masked/padded rows. The padded query
    # results are discarded, but preserving that convention keeps the graph's
    # unused rows well defined.
    padded_positions = F.pad(position_ids, (0, pad_tokens), value=1).contiguous()
    last_token_index = torch.tensor(
        [real_seq_len - 1],
        device=inputs_embeds.device,
        dtype=torch.int64,
    )
    return PreparedTextBucket(
        inputs_embeds=padded_embeds,
        attention_mask=padded_mask,
        position_ids=padded_positions,
        last_token_index=last_token_index,
        real_seq_len=real_seq_len,
        physical_seq_len=physical_seq_len,
    )


def text_source_hash() -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    for name in ("local_modeling_paddleocr_vl.py", "text_compile.py"):
        digest.update(name.encode("utf-8"))
        digest.update(short_file_hash(here / name).encode("utf-8"))
    return digest.hexdigest()[:12]


def text_cache_dir_for_bucket(
    cache_root: Path,
    *,
    bucket: int,
    cache_length: int,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
    linear_weight_format: str,
) -> Path:
    key = "_".join(
        [
            "text_transformer_prefill",
            f"softmax{cache_key_part(get_text_softmax_dtype_mode())}",
            "bs1",
            f"seq{int(bucket)}",
            f"cache{int(cache_length)}",
            f"weights{cache_key_part(linear_weight_format)}",
            f"dtype{cache_key_part(dtype)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{text_source_hash()}",
        ]
    )
    return cache_root.expanduser().resolve() / key


class BucketedTextPrefillRuntime:
    """Own one static text-prefill graph per configured sequence bucket."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        backend: str,
        buckets: Iterable[int],
        cache_root: Path,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        model_dir: Path,
        linear_weight_format: str,
    ):
        self.model = model
        self.backend = str(backend)
        self.buckets = parse_text_buckets(buckets)
        self.cache_root = cache_root.expanduser().resolve()
        self.cache_length = int(cache_length)
        self.device = device
        self.dtype = dtype
        self.compiled: dict[int, Callable[..., torch.Tensor]] = {}
        self.entrypoints: dict[int, Callable[..., torch.Tensor]] = {}
        self.modules: dict[int, StaticTextPrefill] = {}
        self.compile_metadata: dict[str, Any] = {
            "backend": self.backend,
            "enabled": self.backend == "torchair",
            "boundary": "text_transformer_plus_in_place_prefill_kv_writes",
            "buckets": list(self.buckets),
            "overflow": "eager_unpadded",
        }
        if self.backend not in TEXT_BACKEND_CHOICES:
            raise ValueError(
                f"text backend must be one of {TEXT_BACKEND_CHOICES}, got {backend!r}"
            )
        if self.backend == "raw_eager":
            return
        if self.device.type != "npu":
            raise ValueError("compiled text backend torchair requires an NPU device")
        if self.buckets[-1] > self.cache_length:
            raise ValueError(
                f"largest text bucket {self.buckets[-1]} exceeds cache length "
                f"{self.cache_length}"
            )

        torchair, CompilerConfig = import_torchair()
        hidden_size = int(model.config.text_config.hidden_size)
        per_bucket: dict[str, Any] = {}
        wrapper_total_s = 0.0
        first_call_total_s = 0.0
        for bucket in self.buckets:
            module = StaticTextPrefill(model).eval()
            cache_dir = text_cache_dir_for_bucket(
                self.cache_root,
                bucket=bucket,
                cache_length=self.cache_length,
                dtype=self.dtype,
                device=self.device,
                model_dir=model_dir,
                linear_weight_format=linear_weight_format,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            config = CompilerConfig()
            entrypoint = unique_bucket_forward(module, bucket)
            synchronize(self.device)
            started = time.perf_counter()
            compiled = torchair.inference.cache_compile(
                entrypoint,
                config=config,
                dynamic=False,
                cache_dir=str(cache_dir),
                ge_cache=True,
            )
            synchronize(self.device)
            wrapper_s = time.perf_counter() - started

            warm_inputs = torch.zeros(
                (1, bucket, hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
            warm_mask = torch.ones(
                (1, bucket),
                device=self.device,
                dtype=torch.int64,
            )
            warm_positions = torch.zeros(
                (3, 1, bucket),
                device=self.device,
                dtype=torch.int64,
            )
            warm_last_index = torch.tensor(
                [bucket - 1],
                device=self.device,
                dtype=torch.int64,
            )
            warm_cache = model.allocate_static_cache(
                batch_size=1,
                cache_length=self.cache_length,
                device=self.device,
                dtype=self.dtype,
                init_mode="zeros",
            )
            synchronize(self.device)
            started = time.perf_counter()
            warm_output = compiled(
                warm_inputs,
                warm_mask,
                warm_positions,
                warm_last_index,
                *warm_cache.flat_tensors(),
            )
            synchronize(self.device)
            first_call_s = time.perf_counter() - started
            del warm_output, warm_inputs, warm_mask, warm_positions, warm_last_index, warm_cache

            self.modules[bucket] = module
            self.entrypoints[bucket] = entrypoint
            self.compiled[bucket] = compiled
            wrapper_total_s += wrapper_s
            first_call_total_s += first_call_s
            per_bucket[str(bucket)] = {
                "compile_wrapper_s": float(wrapper_s),
                "compile_first_call_s": float(first_call_s),
                "torchair_cache_dir": str(cache_dir),
            }
        self.compile_metadata.update(
            {
                "compile_api": "torchair.inference.cache_compile",
                "dynamic": False,
                "fullgraph": True,
                "torchair_ge_cache": True,
                "compile_wrapper_total_s": float(wrapper_total_s),
                "compile_first_call_total_s": float(first_call_total_s),
                "per_bucket": per_bucket,
                "cache_key_fields": {
                    "cache_length": self.cache_length,
                    "dtype": str(dtype),
                    "linear_weight_format": linear_weight_format,
                    "model_config_hash": short_file_hash(model_dir / "config.json"),
                    "torch": str(torch.__version__),
                    "torch_npu": torch_npu_version_label(device),
                    "torchair": torchair_version_label(device),
                    "text_source_hash": text_source_hash(),
                    "attention": "manual_causal",
                    "softmax_dtype": get_text_softmax_dtype_mode(),
                },
            }
        )

    def route(self, real_seq_len: int) -> dict[str, Any]:
        real_seq_len = int(real_seq_len)
        if self.backend == "raw_eager":
            return {
                "execution": "eager",
                "real_text_tokens": real_seq_len,
                "physical_text_tokens": real_seq_len,
                "padding_text_tokens": 0,
                "useful_token_fraction": 1.0,
                "bucket": None,
            }
        bucket = select_text_bucket(real_seq_len, self.buckets)
        if bucket is None:
            return {
                "execution": "eager_overflow",
                "real_text_tokens": real_seq_len,
                "physical_text_tokens": real_seq_len,
                "padding_text_tokens": 0,
                "useful_token_fraction": 1.0,
                "bucket": None,
            }
        return {
            "execution": "compiled",
            "real_text_tokens": real_seq_len,
            "physical_text_tokens": bucket,
            "padding_text_tokens": bucket - real_seq_len,
            "useful_token_fraction": float(real_seq_len) / float(bucket),
            "bucket": bucket,
        }

    def prepare(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        bucket: int,
    ) -> PreparedTextBucket:
        return prepare_text_bucket(
            inputs_embeds,
            attention_mask,
            position_ids,
            physical_seq_len=int(bucket),
        )

    def run_prepared(
        self,
        prepared: PreparedTextBucket,
        cache: LocalPaddleOCRVLStaticCache,
    ) -> torch.Tensor:
        return self.compiled[prepared.physical_seq_len](
            prepared.inputs_embeds,
            prepared.attention_mask,
            prepared.position_ids,
            prepared.last_token_index,
            *cache.flat_tensors(),
        )
