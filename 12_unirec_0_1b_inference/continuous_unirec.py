"""Fixed-arena continuous decode for the local UniRec runtime."""

from __future__ import annotations

import hashlib
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event as ThreadEvent
from threading import Thread
from typing import Any, Callable, Iterable

import torch

from modeling_optimized_unirec import (
    LOCAL_UNIREC_STATIC_CACHE_LEN,
    LocalUniRecStaticCache,
    OptimizedUniRecRunner,
    UniRecPrefilledItem,
)


def production_decode_cache_parent(base: str | Path) -> Path:
    """Return a stable cache namespace for the compiled decode graph.

    Continuous scheduler changes do not change the compiled
    LocalUniRecCachedDecodeStepModule graph and must not force a large B128 OM
    rebuild on memory-limited 310P devices.  Hash only the file that defines
    the compiled graph plus the runtime.  An explicit override supports reuse
    of a previously validated complete cache during migration.
    """
    override = os.environ.get("UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                "UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE is not a "
                f"directory: {path}"
            )
        return path

    digest = hashlib.sha256(b"unirec-production-decode-graph-v1\0")
    directory = Path(__file__).resolve().parent
    source = directory / "modeling_optimized_unirec.py"
    digest.update(source.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(source.read_bytes())
    digest.update(b"\0")
    digest.update(str(torch.__version__).encode("utf-8"))
    try:
        import torch_npu

        digest.update(str(torch_npu.__version__).encode("utf-8"))
    except Exception:
        digest.update(b"torch_npu_unavailable")
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        version_file = Path(ascend_home) / "opp" / "version.info"
        if version_file.is_file():
            digest.update(version_file.read_bytes())
    return Path(base).expanduser().resolve() / (
        f"production_decode_graph_{digest.hexdigest()[:16]}"
    )


@dataclass
class ContinuousReadyItem:
    request_id: str
    payload: Any
    prefilled: UniRecPrefilledItem | "ContinuousWorkerPrefilledItem"
    on_admitted: Callable[[], None] | None = None

    def release_source_after_admission(self) -> None:
        callback = self.on_admitted
        if callback is None:
            return
        self.on_admitted = None
        callback()


@dataclass
class ContinuousWorkerPrefilledItem:
    """Compact CPU cross-K/V prefix ready for direct arena admission."""

    packed_cross_kv: Any
    prep: dict[str, Any]
    prefill_s: float
    actual_cross_attention_length: int
    prefill_device_stage_s: dict[str, float] | None = None
    text_prefill_execution: str = "eager"
    text_prefill_real_source_tokens: int | None = None
    text_prefill_physical_source_tokens: int | None = None


@dataclass
class ContinuousCompletedItem:
    request_id: str
    payload: Any
    result: dict[str, Any]
    slot: int
    admission_index: int
    completion_index: int


@dataclass
class _StagedWorkerAdmission:
    ready: ContinuousReadyItem
    buffer_index: int
    device_flat: torch.Tensor
    source_len: int
    elements: int
    ready_event: Any


class _WorkerAdmissionPrefetcher:
    """Move future pageable CPU cross-K/V into a small NPU staging ring."""

    def __init__(
        self,
        *,
        next_ready: Callable[[], ContinuousReadyItem | None],
        device: str,
        dtype: torch.dtype,
        max_elements: int,
        depth: int,
    ) -> None:
        if depth < 1:
            raise ValueError("admission prefetch depth must be positive")
        self.next_ready = next_ready
        self.device = device
        self.buffers = tuple(
            torch.empty(max_elements, dtype=dtype, device=device)
            for _ in range(depth)
        )
        self.transfer_stream = torch.npu.Stream(device=torch.device(device))
        self.free: Queue[int | None] = Queue()
        self.ready: Queue[_StagedWorkerAdmission | BaseException | None] = Queue()
        self.release_events: list[Any | None] = [None for _ in range(depth)]
        self.stop_requested = ThreadEvent()
        self.h2d_host_enqueue_s = 0.0
        self.h2d_items = 0
        self.h2d_bytes = 0
        self.max_ready_depth = 0
        for index in range(depth):
            self.free.put(index)
        self.thread = Thread(
            target=self._run,
            name="unirec-cross-kv-prefetch",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        try:
            import torch_npu

            torch_npu.npu.set_device(self.device)
            while not self.stop_requested.is_set():
                buffer_index = self.free.get()
                if buffer_index is None or self.stop_requested.is_set():
                    return
                ready = self.next_ready()
                if ready is None:
                    self.ready.put(None)
                    return
                if not isinstance(ready.prefilled, ContinuousWorkerPrefilledItem):
                    raise RuntimeError(
                        "admission prefetch requires worker-prefilled items"
                    )
                packed_host = ready.prefilled.packed_cross_kv
                source = torch.from_numpy(packed_host).reshape(-1)
                elements = source.numel()
                if elements > self.buffers[buffer_index].numel():
                    raise RuntimeError(
                        "worker cross-K/V exceeds admission staging buffer: "
                        f"{elements} > {self.buffers[buffer_index].numel()}"
                    )
                enqueue_started = time.perf_counter()
                with torch.npu.stream(self.transfer_stream):
                    release_event = self.release_events[buffer_index]
                    if release_event is not None:
                        self.transfer_stream.wait_event(release_event)
                    self.buffers[buffer_index][:elements].copy_(
                        source,
                        non_blocking=True,
                    )
                    ready_event = torch.npu.Event()
                    ready_event.record(self.transfer_stream)
                self.h2d_host_enqueue_s += time.perf_counter() - enqueue_started
                self.h2d_items += 1
                self.h2d_bytes += int(packed_host.nbytes)
                self.ready.put(
                    _StagedWorkerAdmission(
                        ready=ready,
                        buffer_index=buffer_index,
                        device_flat=self.buffers[buffer_index],
                        source_len=int(packed_host.shape[-2]),
                        elements=elements,
                        ready_event=ready_event,
                    )
                )
                self.max_ready_depth = max(
                    self.max_ready_depth,
                    self.ready.qsize(),
                )
        except BaseException as exception:
            self.ready.put(exception)

    def take(self) -> _StagedWorkerAdmission | None:
        value = self.ready.get()
        if isinstance(value, BaseException):
            raise RuntimeError("cross-K/V admission prefetch failed") from value
        return value

    def release(self, staged: _StagedWorkerAdmission) -> None:
        release_event = torch.npu.Event()
        release_event.record(torch.npu.current_stream(torch.device(self.device)))
        self.release_events[staged.buffer_index] = release_event
        self.free.put(staged.buffer_index)

    def close(self) -> None:
        if self.thread.is_alive():
            self.stop_requested.set()
            self.free.put(None)
            self.thread.join(timeout=30.0)
        if self.thread.is_alive():
            raise RuntimeError("cross-K/V admission prefetch thread did not stop")

    def summary(self) -> dict[str, Any]:
        return {
            "depth": len(self.buffers),
            "buffer_bytes": sum(
                int(buffer.numel() * buffer.element_size())
                for buffer in self.buffers
            ),
            "h2d_host_enqueue_s": self.h2d_host_enqueue_s,
            "h2d_items": self.h2d_items,
            "h2d_bytes": self.h2d_bytes,
            "max_ready_depth": self.max_ready_depth,
        }


class ContinuousUniRecDecoder:
    """Hot-swap B1-prefilled requests in a fixed physical decode batch."""

    def __init__(
        self,
        *,
        runner: OptimizedUniRecRunner,
        batch_size: int,
        max_length: int,
        decode_mode: str,
        compile_backend: str,
        compile_dynamic: bool = False,
        admission_prefetch_depth: int = 0,
        self_cache_length: int | None = None,
        cross_cache_length: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Continuous decode batch_size must be >= 1")
        self.self_cache_length = int(
            LOCAL_UNIREC_STATIC_CACHE_LEN
            if self_cache_length is None
            else self_cache_length
        )
        self.cross_cache_length = (
            None if cross_cache_length is None else int(cross_cache_length)
        )
        if self.self_cache_length < 1:
            raise ValueError("self_cache_length must be positive")
        if self.cross_cache_length is not None and self.cross_cache_length < 1:
            raise ValueError("cross_cache_length must be positive")
        if max_length > self.self_cache_length:
            raise ValueError(
                f"max_length must be <= {self.self_cache_length}, got {max_length}"
            )
        if decode_mode not in {"eager", "compiled", "compiled_ifa"}:
            raise ValueError(f"Unsupported decode_mode: {decode_mode}")
        self.runner = runner
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.decode_mode = decode_mode
        self.compile_backend = compile_backend
        self.compile_dynamic = bool(compile_dynamic)
        self.admission_prefetch_depth = int(admission_prefetch_depth)
        if self.admission_prefetch_depth < 0:
            raise ValueError("admission_prefetch_depth must be non-negative")
        self._cross_attention_mask_templates: torch.Tensor | None = None

    @staticmethod
    def _allocate_decode_device_inputs(
        batch_size: int,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate the exact inference-tensor contract used by decode.

        TorchDynamo guards whether a tensor was created under inference mode.
        Graph warmup already creates its inputs under inference mode.  Keep the
        long-lived production input buffers under the same contract so the
        first real decode step cannot specialize a second graph.
        """
        with torch.inference_mode():
            next_token_tensor = torch.empty(
                (int(batch_size), 1),
                dtype=torch.long,
                device=device,
            )
            cache_position_tensor = torch.empty(
                int(batch_size),
                dtype=torch.int64,
                device=device,
            )
        if not (
            next_token_tensor.is_inference()
            and cache_position_tensor.is_inference()
        ):
            raise RuntimeError(
                "continuous decode device inputs must be inference tensors"
            )
        return next_token_tensor, cache_position_tensor

    @staticmethod
    def _copy_cache_row(
        destination: LocalUniRecStaticCache,
        slot: int,
        source: LocalUniRecStaticCache,
    ) -> None:
        if (
            destination.cross_key_cache is None
            or destination.cross_value_cache is None
            or destination.cross_attention_mask is None
            or source.cross_key_cache is None
            or source.cross_value_cache is None
            or source.cross_attention_mask is None
        ):
            raise RuntimeError("Continuous decode requires static cross-attention caches")
        for layer in range(len(destination.key_cache)):
            destination.key_cache[layer][slot : slot + 1].copy_(
                source.key_cache[layer]
            )
            destination.value_cache[layer][slot : slot + 1].copy_(
                source.value_cache[layer]
            )
            destination.cross_key_cache[layer][slot : slot + 1].copy_(
                source.cross_key_cache[layer]
            )
            destination.cross_value_cache[layer][slot : slot + 1].copy_(
                source.cross_value_cache[layer]
            )
        destination.cross_attention_mask[slot : slot + 1].copy_(
            source.cross_attention_mask
        )

    def _allocate_empty_arena(self) -> LocalUniRecStaticCache:
        """Allocate the fixed physical decode cache without temporary B1 rows."""
        layer_count = len(self.runner.model.decoder.layers)
        num_heads = int(self.runner.config.decoder_attention_heads)
        head_dim = int(self.runner.config.d_model) // num_heads
        cross_cache_len = int(
            self.runner._get_static_cross_cache_len()
            if self.cross_cache_length is None
            else self.cross_cache_length
        )
        cache_shape = (
            self.batch_size,
            num_heads,
            self.self_cache_length,
            head_dim,
        )
        cross_shape = (
            self.batch_size,
            num_heads,
            cross_cache_len,
            head_dim,
        )
        device = self.runner.device
        dtype = self.runner.dtype
        negative_inf = torch.finfo(torch.float32).min
        with torch.inference_mode():
            key_cache = tuple(
                torch.zeros(cache_shape, dtype=dtype, device=device)
                for _ in range(layer_count)
            )
            value_cache = tuple(
                torch.zeros(cache_shape, dtype=dtype, device=device)
                for _ in range(layer_count)
            )
            packed_cross_kv = torch.zeros(
                (2 * layer_count, *cross_shape),
                dtype=dtype,
                device=device,
            )
            cross_key_cache = tuple(
                packed_cross_kv[layer] for layer in range(layer_count)
            )
            cross_value_cache = tuple(
                packed_cross_kv[layer_count + layer]
                for layer in range(layer_count)
            )
            cross_attention_mask = torch.full(
                (self.batch_size, 1, 1, cross_cache_len),
                negative_inf,
                dtype=torch.float32,
                device=device,
            )
            positions = torch.arange(
                cross_cache_len,
                dtype=torch.int64,
                device=device,
            ).view(1, 1, 1, cross_cache_len)
            lengths = torch.arange(
                cross_cache_len + 1,
                dtype=torch.int64,
                device=device,
            ).view(cross_cache_len + 1, 1, 1, 1)
            valid = positions < lengths
            self._cross_attention_mask_templates = torch.where(
                valid,
                torch.zeros((), dtype=torch.float32, device=device),
                torch.full(
                    (),
                    negative_inf,
                    dtype=torch.float32,
                    device=device,
                ),
            )
        return LocalUniRecStaticCache(
            key_cache=key_cache,
            value_cache=value_cache,
            cache_len=self.self_cache_length,
            cross_key_cache=cross_key_cache,
            cross_value_cache=cross_value_cache,
            cross_attention_mask=cross_attention_mask,
            actual_cross_attention_length=None,
            packed_cross_kv=packed_cross_kv,
        )

    def _write_cross_attention_mask(
        self,
        destination: LocalUniRecStaticCache,
        slot: int,
        source_len: int,
    ) -> None:
        templates = self._cross_attention_mask_templates
        if templates is None:
            raise RuntimeError("cross-attention mask templates are not allocated")
        destination.cross_attention_mask[slot : slot + 1].copy_(
            templates[source_len : source_len + 1]
        )

    def _admit_worker_row(
        self,
        destination: LocalUniRecStaticCache,
        slot: int,
        source: ContinuousWorkerPrefilledItem,
        *,
        reset_reused_row: bool,
    ) -> tuple[float, int, int]:
        """Write compact CPU cross-K/V directly into one existing arena row."""
        if (
            destination.cross_key_cache is None
            or destination.cross_value_cache is None
            or destination.cross_attention_mask is None
        ):
            raise RuntimeError("Continuous decode requires static cross-attention caches")
        packed_host = source.packed_cross_kv
        layer_count = len(destination.key_cache)
        if packed_host.ndim != 5 or int(packed_host.shape[0]) != 2 * layer_count:
            raise RuntimeError(
                "unexpected worker cross-K/V shape: "
                f"{packed_host.shape}; decoder_layers={layer_count}"
            )
        if int(packed_host.shape[1]) != 1:
            raise RuntimeError(
                "worker cross-K/V must contain exactly one batch row, got "
                f"shape={packed_host.shape}"
            )
        source_len = int(packed_host.shape[-2])
        if source_len != int(source.actual_cross_attention_length):
            raise RuntimeError(
                "worker cross-K/V length mismatch: "
                f"tensor={source_len} metadata={source.actual_cross_attention_length}"
            )
        cross_cache_len = int(destination.cross_attention_mask.shape[-1])
        if source_len > cross_cache_len:
            raise RuntimeError(
                "worker cross-K/V exceeds the decode arena: "
                f"source={source_len} capacity={cross_cache_len}"
            )
        expected_tail = (
            int(destination.cross_key_cache[0].shape[1]),
            source_len,
            int(destination.cross_key_cache[0].shape[3]),
        )
        if tuple(int(value) for value in packed_host.shape[2:]) != expected_tail:
            raise RuntimeError(
                "worker cross-K/V head shape mismatch: "
                f"tensor={tuple(packed_host.shape[2:])} expected={expected_tail}"
            )

        started = time.perf_counter()
        with torch.inference_mode():
            self._write_cross_attention_mask(destination, slot, source_len)
            # Reused self-K/V is safe without clearing: every decode layer
            # overwrites cache_position before attention, and the static
            # self-attention mask excludes every later position. Likewise, the
            # new prefix overwrites the live cross-K/V range while the cross-
            # attention mask excludes the stale tail.
            if destination.packed_cross_kv is not None:
                destination.packed_cross_kv[
                    :, slot : slot + 1, :, :source_len, :
                ].copy_(torch.from_numpy(packed_host))
            else:
                for layer in range(layer_count):
                    destination.cross_key_cache[layer][
                        slot : slot + 1, :, :source_len, :
                    ].copy_(torch.from_numpy(packed_host[layer]))
                    destination.cross_value_cache[layer][
                        slot : slot + 1, :, :source_len, :
                    ].copy_(
                        torch.from_numpy(packed_host[layer_count + layer])
                    )
        enqueue_s = time.perf_counter() - started
        source.prefill_s += enqueue_s
        stages = dict(source.prefill_device_stage_s or {})
        stages["coordinator_direct_arena_admission_enqueue"] = enqueue_s
        source.prefill_device_stage_s = stages
        transferred_bytes = int(packed_host.nbytes)
        reset_bytes = 0
        if reset_reused_row:
            reset_bytes = int(
                destination.cross_attention_mask[slot : slot + 1].numel()
                * destination.cross_attention_mask.element_size()
            )
        return enqueue_s, transferred_bytes, reset_bytes

    def _admit_staged_worker_row(
        self,
        destination: LocalUniRecStaticCache,
        slot: int,
        staged: _StagedWorkerAdmission,
    ) -> tuple[float, int, int]:
        """Place one already-device-resident cross-K/V row into the arena."""
        source = staged.ready.prefilled
        if not isinstance(source, ContinuousWorkerPrefilledItem):
            raise RuntimeError("staged admission requires a worker-prefilled item")
        packed_host = source.packed_cross_kv
        source_len = staged.source_len
        if source_len != int(source.actual_cross_attention_length):
            raise RuntimeError(
                "staged cross-K/V length mismatch: "
                f"staged={source_len} metadata={source.actual_cross_attention_length}"
            )
        if destination.packed_cross_kv is None:
            raise RuntimeError("staged admission requires a packed decode arena")
        expected_elements = int(packed_host.size)
        if staged.elements != expected_elements:
            raise RuntimeError(
                "staged cross-K/V element mismatch: "
                f"{staged.elements} != {expected_elements}"
            )
        started = time.perf_counter()
        current_stream = torch.npu.current_stream(torch.device(self.runner.device))
        current_stream.wait_event(staged.ready_event)
        with torch.inference_mode():
            self._write_cross_attention_mask(destination, slot, source_len)
            staged_source = staged.device_flat[: staged.elements].view(
                packed_host.shape
            )
            destination.packed_cross_kv[
                :, slot : slot + 1, :, :source_len, :
            ].copy_(staged_source)
        enqueue_s = time.perf_counter() - started
        source.prefill_s += enqueue_s
        stages = dict(source.prefill_device_stage_s or {})
        stages["coordinator_staged_arena_admission_enqueue"] = enqueue_s
        source.prefill_device_stage_s = stages
        return enqueue_s, int(packed_host.nbytes), int(
            destination.cross_attention_mask[slot : slot + 1].numel()
            * destination.cross_attention_mask.element_size()
        )

    @staticmethod
    def _initial_token_ids(
        ready: ContinuousReadyItem,
        *,
        decoder_start_token_id: int,
    ) -> list[int]:
        if isinstance(ready.prefilled, UniRecPrefilledItem):
            return [
                int(token)
                for token in ready.prefilled.generated_ids[0].detach().cpu().tolist()
            ]
        return [int(decoder_start_token_id)]

    def _build_result(
        self,
        *,
        ready: ContinuousReadyItem,
        token_ids: list[int],
        decode_token_count: int,
        decode_active_s: float,
        compile_wrap_s: float | None,
        compile_meta: dict[str, Any] | None,
        cross_cache_len: int,
    ) -> dict[str, Any]:
        text = self.runner._decode_text_batch([token_ids])[0]
        prep = ready.prefilled.prep
        return {
            "image": str(prep["image"]),
            "text": text,
            "generated_ids": token_ids,
            "generated_token_count": max(0, len(token_ids) - 1),
            "prefill_generated_token_count": 0,
            "decode_generated_token_count": int(decode_token_count),
            "ttft_s": ready.prefilled.prefill_s,
            "prefill_device_stage_s": ready.prefilled.prefill_device_stage_s,
            "text_prefill_execution": ready.prefilled.text_prefill_execution,
            "text_prefill_real_source_tokens": (
                ready.prefilled.text_prefill_real_source_tokens
            ),
            "text_prefill_physical_source_tokens": (
                ready.prefilled.text_prefill_physical_source_tokens
            ),
            "decode_s": float(decode_active_s),
            "total_latency_s": (
                float(prep["prepare_total_s"])
                + ready.prefilled.prefill_s
                + float(decode_active_s)
            ),
            "decode_tokens_per_s": (
                float(decode_token_count) / decode_active_s
                if decode_active_s > 0 and decode_token_count > 0
                else None
            ),
            "compile_wrap_s": compile_wrap_s,
            "compile": compile_meta,
            "cross_cache_len": int(cross_cache_len),
            "static_self_kv_len": self.self_cache_length,
            "device": self.runner.device,
            "dtype": self.runner.dtype_name,
            "decode_mode": self.decode_mode,
            "compile_backend": (
                self.compile_backend
                if self.decode_mode.startswith("compiled")
                else None
            ),
            "prep": prep,
        }

    def run(
        self,
        source: Iterable[ContinuousReadyItem],
        *,
        on_complete: Callable[[ContinuousCompletedItem], None],
        graph_warmup_passes: int = 0,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if graph_warmup_passes < 0:
            raise ValueError("graph_warmup_passes must be non-negative")
        run_started = time.perf_counter()
        iterator = iter(source)
        source_exhausted = False
        submitted = 0
        completed = 0
        source_pull_s = 0.0
        initial_cache_stack_s = 0.0
        initial_arena_allocate_s = 0.0
        initial_arena_admission_enqueue_s = 0.0
        initial_state_build_s = 0.0
        cache_refill_copy_enqueue_s = 0.0
        cache_refill_direct_admission_enqueue_s = 0.0
        cache_refill_row_bytes = 0
        direct_admission_count = 0
        direct_cross_kv_bytes = 0
        direct_reset_bytes = 0
        result_build_s = 0.0
        completion_callback_s = 0.0
        decode_input_build_s = 0.0
        decode_input_buffer_setup_s = 0.0
        decode_input_host_pinned = False
        pre_decode_sync_s = 0.0
        production_graph_warmup_s = 0.0
        production_graph_warmup_pass_s: list[float] = []
        production_graph_warmup_warnings: list[str] = []
        diagnostic_step_callback_s = 0.0

        def next_ready() -> ContinuousReadyItem | None:
            nonlocal source_exhausted, source_pull_s, submitted
            if source_exhausted:
                return None
            pull_started = time.perf_counter()
            try:
                ready = next(iterator)
            except StopIteration:
                source_exhausted = True
                source_pull_s += time.perf_counter() - pull_started
                return None
            source_pull_s += time.perf_counter() - pull_started
            submitted += 1
            return ready

        first_ready = next_ready()
        if first_ready is None:
            return {
                "batch_size": self.batch_size,
                "submitted": 0,
                "completed": 0,
                "decode_iterations": 0,
                "raw_decode_token_slots": 0,
                "effective_decode_tokens": 0,
                "idle_decode_token_slots": 0,
                "decode_s": 0.0,
                "first_decode_step_s": None,
                "steady_decode_s": 0.0,
                "raw_decode_tokens_per_s": None,
                "effective_decode_tokens_per_s": None,
                "steady_raw_decode_tokens_per_s": None,
                "steady_effective_decode_tokens_per_s": None,
                "slot_refills": 0,
                "compile_wrap_s": None,
                "compile": None,
            }

        initial: list[ContinuousReadyItem] = [first_ready]
        direct_initial = isinstance(
            first_ready.prefilled,
            ContinuousWorkerPrefilledItem,
        )
        if direct_initial:
            arena_allocate_started = time.perf_counter()
            cache = self._allocate_empty_arena()
            initial_arena_allocate_s = time.perf_counter() - arena_allocate_started
            for slot in range(self.batch_size):
                if slot == 0:
                    ready = first_ready
                else:
                    ready = next_ready()
                    if ready is None:
                        break
                    initial.append(ready)
                if not isinstance(ready.prefilled, ContinuousWorkerPrefilledItem):
                    raise RuntimeError("Cannot mix worker and B1 initial prefill items")
                enqueue_s, transferred_bytes, reset_bytes = self._admit_worker_row(
                    cache,
                    slot,
                    ready.prefilled,
                    reset_reused_row=False,
                )
                initial_arena_admission_enqueue_s += enqueue_s
                direct_admission_count += 1
                direct_cross_kv_bytes += transferred_bytes
                direct_reset_bytes += reset_bytes
                ready.release_source_after_admission()
            if len(initial) < self.batch_size:
                # Keep inactive graph rows numerically valid, matching the old
                # behavior that padded the initial cohort with its last item.
                last_slot = len(initial) - 1
                with torch.inference_mode():
                    for slot in range(len(initial), self.batch_size):
                        for tensor_group in (
                            cache.key_cache,
                            cache.value_cache,
                            cache.cross_key_cache or (),
                            cache.cross_value_cache or (),
                        ):
                            for tensor in tensor_group:
                                tensor[slot : slot + 1].copy_(
                                    tensor[last_slot : last_slot + 1]
                                )
                        cache.cross_attention_mask[slot : slot + 1].copy_(
                            cache.cross_attention_mask[last_slot : last_slot + 1]
                        )
        else:
            for _ in range(1, self.batch_size):
                ready = next_ready()
                if ready is None:
                    break
                initial.append(ready)
            padded_initial = list(initial)
            while len(padded_initial) < self.batch_size:
                padded_initial.append(initial[-1])
            if any(
                not isinstance(item.prefilled, UniRecPrefilledItem)
                for item in padded_initial
            ):
                raise RuntimeError("Cannot mix worker and B1 initial prefill items")
            cache_stack_started = time.perf_counter()
            cache = self.runner._stack_prefilled_caches(
                [item.prefilled for item in padded_initial]
            )
            initial_cache_stack_s = time.perf_counter() - cache_stack_started
            for ready in initial:
                ready.release_source_after_admission()
        state_build_started = time.perf_counter()
        slots: list[ContinuousReadyItem | None] = [
            initial[index] if index < len(initial) else None
            for index in range(self.batch_size)
        ]
        token_ids: list[list[int]] = [
            (
                self._initial_token_ids(
                    slots[index],
                    decoder_start_token_id=int(
                        self.runner.config.decoder_start_token_id
                    ),
                )
                if slots[index] is not None
                else [
                    int(self.runner.config.decoder_start_token_id),
                    int(self.runner.config.eos_token_id),
                ]
            )
            for index in range(self.batch_size)
        ]
        last_tokens = [row[-1] for row in token_ids]
        cache_positions = [
            len(row) - 1 if slots[index] is not None else 1
            for index, row in enumerate(token_ids)
        ]
        slot_decode_counts = [0 for _ in range(self.batch_size)]
        slot_active_decode_s = [0.0 for _ in range(self.batch_size)]
        slot_admission_indices = [
            index if index < len(initial) else -1 for index in range(self.batch_size)
        ]
        next_admission_index = len(initial)
        slot_refills = 0
        eos_token_id = int(self.runner.config.eos_token_id)
        initial_state_build_s = time.perf_counter() - state_build_started

        self_attention_backend = (
            "increfa_all" if self.decode_mode == "compiled_ifa" else "eager"
        )
        decode_module = None
        compile_wrap_s = None
        compile_meta = None
        cross_cache_len = int(cache.cross_attention_mask.shape[-1])
        if self.decode_mode.startswith("compiled"):
            compile_started = time.perf_counter()
            decode_module, compile_meta = self.runner._compile_decode_module(
                backend=self.compile_backend,
                self_attention_backend=self_attention_backend,
                compile_dynamic=self.compile_dynamic,
                cross_cache_len=cross_cache_len,
                batch_size=self.batch_size,
                self_cache_len=self.self_cache_length,
            )
            compile_wrap_s = time.perf_counter() - compile_started

        admission_prefetcher: _WorkerAdmissionPrefetcher | None = None
        admission_prefetch_exhausted = False
        if direct_initial and self.admission_prefetch_depth:
            if cache.packed_cross_kv is None:
                raise RuntimeError("admission prefetch requires a packed arena")
            admission_prefetcher = _WorkerAdmissionPrefetcher(
                next_ready=next_ready,
                device=self.runner.device,
                dtype=self.runner.dtype,
                max_elements=int(
                    cache.packed_cross_kv[:, 0:1].numel()
                ),
                depth=self.admission_prefetch_depth,
            )

        def complete_slot(slot: int) -> None:
            nonlocal completed, completion_callback_s, result_build_s
            ready = slots[slot]
            if ready is None:
                return
            result_started = time.perf_counter()
            result = self._build_result(
                ready=ready,
                token_ids=token_ids[slot],
                decode_token_count=slot_decode_counts[slot],
                decode_active_s=slot_active_decode_s[slot],
                compile_wrap_s=compile_wrap_s,
                compile_meta=compile_meta,
                cross_cache_len=cross_cache_len,
            )
            result_build_s += time.perf_counter() - result_started
            callback_started = time.perf_counter()
            on_complete(
                ContinuousCompletedItem(
                    request_id=ready.request_id,
                    payload=ready.payload,
                    result=result,
                    slot=slot,
                    admission_index=slot_admission_indices[slot],
                    completion_index=completed,
                )
            )
            completion_callback_s += time.perf_counter() - callback_started
            completed += 1
            slots[slot] = None

        def refill_slot(slot: int) -> None:
            nonlocal cache_refill_copy_enqueue_s, cache_refill_row_bytes
            nonlocal cache_refill_direct_admission_enqueue_s
            nonlocal direct_admission_count, direct_cross_kv_bytes, direct_reset_bytes
            nonlocal next_admission_index, slot_refills
            nonlocal admission_prefetch_exhausted
            while slots[slot] is None:
                staged: _StagedWorkerAdmission | None = None
                if admission_prefetcher is None:
                    ready = next_ready()
                elif admission_prefetch_exhausted:
                    ready = None
                else:
                    staged = admission_prefetcher.take()
                    ready = staged.ready if staged is not None else None
                    if staged is None:
                        admission_prefetch_exhausted = True
                if ready is None:
                    token_ids[slot] = [
                        int(self.runner.config.decoder_start_token_id),
                        eos_token_id,
                    ]
                    last_tokens[slot] = eos_token_id
                    cache_positions[slot] = 1
                    return
                if isinstance(ready.prefilled, ContinuousWorkerPrefilledItem):
                    if staged is None:
                        enqueue_s, transferred_bytes, reset_bytes = (
                            self._admit_worker_row(
                                cache,
                                slot,
                                ready.prefilled,
                                reset_reused_row=True,
                            )
                        )
                    else:
                        enqueue_s, transferred_bytes, reset_bytes = (
                            self._admit_staged_worker_row(
                                cache,
                                slot,
                                staged,
                            )
                        )
                        admission_prefetcher.release(staged)
                    cache_refill_direct_admission_enqueue_s += enqueue_s
                    direct_admission_count += 1
                    direct_cross_kv_bytes += transferred_bytes
                    direct_reset_bytes += reset_bytes
                else:
                    if cache_refill_row_bytes == 0:
                        cache_refill_row_bytes = sum(
                            int(tensor[slot : slot + 1].numel() * tensor.element_size())
                            for tensor_group in (
                                cache.key_cache,
                                cache.value_cache,
                                cache.cross_key_cache or (),
                                cache.cross_value_cache or (),
                            )
                            for tensor in tensor_group
                        )
                        if cache.cross_attention_mask is not None:
                            mask_row = cache.cross_attention_mask[slot : slot + 1]
                            cache_refill_row_bytes += int(
                                mask_row.numel() * mask_row.element_size()
                            )
                    copy_started = time.perf_counter()
                    self._copy_cache_row(cache, slot, ready.prefilled.kv_cache)
                    cache_refill_copy_enqueue_s += time.perf_counter() - copy_started
                ready.release_source_after_admission()
                slots[slot] = ready
                token_ids[slot] = self._initial_token_ids(
                    ready,
                    decoder_start_token_id=int(
                        self.runner.config.decoder_start_token_id
                    ),
                )
                last_tokens[slot] = token_ids[slot][-1]
                cache_positions[slot] = len(token_ids[slot]) - 1
                slot_decode_counts[slot] = 0
                slot_active_decode_s[slot] = 0.0
                slot_admission_indices[slot] = next_admission_index
                next_admission_index += 1
                slot_refills += 1
                if last_tokens[slot] != eos_token_id:
                    return
                complete_slot(slot)

        # A request may produce EOS directly from B1 prefill. Complete and
        # refill it without launching a useless decode iteration.
        for slot in range(self.batch_size):
            if slots[slot] is not None and last_tokens[slot] == eos_token_id:
                complete_slot(slot)
                refill_slot(slot)

        decode_iterations = 0
        raw_decode_token_slots = 0
        effective_decode_tokens = 0
        idle_decode_token_slots = 0
        decode_s = 0.0
        first_decode_step_s: float | None = None

        input_buffer_setup_started = time.perf_counter()
        try:
            next_token_host = torch.empty(
                self.batch_size,
                dtype=torch.long,
                pin_memory=True,
            )
            cache_position_host = torch.empty(
                self.batch_size,
                dtype=torch.int64,
                pin_memory=True,
            )
            decode_input_host_pinned = True
        except RuntimeError:
            # Some non-accelerator test environments do not expose a pinned
            # allocator. Reuse pageable host storage there instead of falling
            # back to per-iteration device allocations.
            next_token_host = torch.empty(self.batch_size, dtype=torch.long)
            cache_position_host = torch.empty(
                self.batch_size,
                dtype=torch.int64,
            )
        next_token_host_array = next_token_host.numpy()
        cache_position_host_array = cache_position_host.numpy()
        next_token_tensor, cache_position_tensor = (
            self._allocate_decode_device_inputs(
                self.batch_size,
                self.runner.device,
            )
        )
        decode_input_buffer_setup_s = (
            time.perf_counter() - input_buffer_setup_started
        )

        # Warm the graph on the actual admitted production arena and the exact
        # long-lived device input tensors used by every measured decode step.
        # A synthetic arena can satisfy the isolated cache probe yet select a
        # second Dynamo contract when the real arena arrives on 310P. Repeated
        # warmup writes are safe: the first measured step writes the same
        # self-KV positions again before they are consumed by later steps.
        if decode_module is not None and graph_warmup_passes:
            from modeling_optimized_unirec import synchronize_device

            warmup_started = time.perf_counter()
            with torch.inference_mode(), warnings.catch_warnings(
                record=True
            ) as caught:
                warnings.simplefilter("always")
                next_token_host_array[:] = last_tokens
                cache_position_host_array[:] = cache_positions
                next_token_tensor.view(-1).copy_(
                    next_token_host,
                    non_blocking=decode_input_host_pinned,
                )
                cache_position_tensor.copy_(
                    cache_position_host,
                    non_blocking=decode_input_host_pinned,
                )
                for pass_index in range(graph_warmup_passes):
                    pass_started = time.perf_counter()
                    _ = decode_module(
                        next_token_tensor,
                        cache_position_tensor,
                        0,
                        cache.key_cache,
                        cache.value_cache,
                        cache.cross_key_cache,
                        cache.cross_value_cache,
                        cache.cross_attention_mask,
                    )
                    synchronize_device(self.runner.device)
                    pass_s = time.perf_counter() - pass_started
                    production_graph_warmup_pass_s.append(pass_s)
                    print(
                        "UNIREC_PRODUCTION_DECODE_WARMUP_PASS "
                        f"pass={pass_index + 1}/{graph_warmup_passes} "
                        f"wall_s={pass_s:.6f}",
                        flush=True,
                    )
                production_graph_warmup_warnings = [
                    str(warning.message) for warning in caught
                ]
            production_graph_warmup_s = time.perf_counter() - warmup_started
            recompile_warnings = [
                message
                for message in production_graph_warmup_warnings
                if (
                    "Skip cache as LocalUniRecCachedDecodeStepModule.forward"
                    in message
                    and "recompiled" in message
                )
            ]
            if recompile_warnings:
                raise RuntimeError(
                    "production decode arena invalidated the persisted graph "
                    "cache; refusing to enter the slow decode loop: "
                    + recompile_warnings[0]
                )

        try:
            with torch.inference_mode():
                while any(slot is not None for slot in slots):
                    iteration_started = time.perf_counter()
                    active_slots = [slot is not None for slot in slots]
                    active_positions = [
                        int(cache_positions[index])
                        for index, is_active in enumerate(active_slots)
                        if is_active
                    ]
                    active_source_lengths = [
                        int(slots[index].prefilled.actual_cross_attention_length)
                        for index, is_active in enumerate(active_slots)
                        if is_active and slots[index] is not None
                    ]
                    input_build_started = time.perf_counter()
                    next_token_host_array[:] = last_tokens
                    cache_position_host_array[:] = cache_positions
                    next_token_tensor.view(-1).copy_(
                        next_token_host,
                        non_blocking=decode_input_host_pinned,
                    )
                    cache_position_tensor.copy_(
                        cache_position_host,
                        non_blocking=decode_input_host_pinned,
                    )
                    input_build_s = time.perf_counter() - input_build_started
                    decode_input_build_s += input_build_s
                    # Default-stream ordering makes the input copies and any
                    # admitted arena rows visible to the graph. A device-wide
                    # synchronization here would also drain the independent
                    # admission transfer stream and defeat cross-K/V prefetch.
                    # The sampled-token CPU read below remains the required
                    # completion point for this iteration.
                    step_started = time.perf_counter()
                    graph_submit_started = step_started
                    if decode_module is None:
                        logits = self.runner.model.forward_cached_logits(
                            decoder_input_ids=next_token_tensor,
                            cache_position=cache_position_tensor,
                            active_length=0,
                            key_cache=cache.key_cache,
                            value_cache=cache.value_cache,
                            cross_key_cache=cache.cross_key_cache,
                            cross_value_cache=cache.cross_value_cache,
                            cross_attention_mask=cache.cross_attention_mask,
                            self_attention_backend="eager",
                        )
                    else:
                        logits = decode_module(
                            next_token_tensor,
                            cache_position_tensor,
                            0,
                            cache.key_cache,
                            cache.value_cache,
                            cache.cross_key_cache,
                            cache.cross_value_cache,
                            cache.cross_attention_mask,
                        )
                    graph_submit_s = time.perf_counter() - graph_submit_started
                    token_wait_started = time.perf_counter()
                    predicted = self.runner.model.select_next_token(logits)
                    predicted_ids = [
                        int(token)
                        for token in predicted.detach().cpu().view(-1).tolist()
                    ]
                    token_select_d2h_wait_s = (
                        time.perf_counter() - token_wait_started
                    )
                    step_s = time.perf_counter() - step_started
                    decode_s += step_s
                    if first_decode_step_s is None:
                        first_decode_step_s = step_s
                    decode_iterations += 1
                    raw_decode_token_slots += self.batch_size
                    active_count = sum(active_slots)
                    effective_decode_tokens += active_count
                    idle_decode_token_slots += self.batch_size - active_count

                    scheduler_started = time.perf_counter()
                    completed_slots = []
                    for slot, is_active in enumerate(active_slots):
                        if not is_active:
                            continue
                        token = predicted_ids[slot]
                        token_ids[slot].append(token)
                        last_tokens[slot] = token
                        cache_positions[slot] += 1
                        slot_decode_counts[slot] += 1
                        slot_active_decode_s[slot] += step_s
                        if (
                            token == eos_token_id
                            or len(token_ids[slot]) >= self.max_length
                        ):
                            completed_slots.append(slot)

                    refills_before = slot_refills
                    for slot in completed_slots:
                        complete_slot(slot)
                    for slot in completed_slots:
                        refill_slot(slot)
                    scheduler_s = time.perf_counter() - scheduler_started
                    if on_step is not None:
                        sorted_positions = sorted(active_positions)
                        sorted_source_lengths = sorted(active_source_lengths)
                        callback_started = time.perf_counter()
                        on_step(
                            {
                                "iteration": decode_iterations,
                                "active_count": active_count,
                                "cache_position_min": (
                                    sorted_positions[0]
                                    if sorted_positions
                                    else None
                                ),
                                "cache_position_p50": (
                                    sorted_positions[
                                        (len(sorted_positions) - 1) // 2
                                    ]
                                    if sorted_positions
                                    else None
                                ),
                                "cache_position_max": (
                                    sorted_positions[-1]
                                    if sorted_positions
                                    else None
                                ),
                                "cross_length_min": (
                                    sorted_source_lengths[0]
                                    if sorted_source_lengths
                                    else None
                                ),
                                "cross_length_p50": (
                                    sorted_source_lengths[
                                        (len(sorted_source_lengths) - 1) // 2
                                    ]
                                    if sorted_source_lengths
                                    else None
                                ),
                                "cross_length_max": (
                                    sorted_source_lengths[-1]
                                    if sorted_source_lengths
                                    else None
                                ),
                                "input_build_s": input_build_s,
                                "graph_submit_s": graph_submit_s,
                                "token_select_d2h_wait_s": (
                                    token_select_d2h_wait_s
                                ),
                                "decode_step_s": step_s,
                                "scheduler_s": scheduler_s,
                                "iteration_wall_s": (
                                    time.perf_counter() - iteration_started
                                ),
                                "completed_slots": len(completed_slots),
                                "refilled_slots": slot_refills - refills_before,
                                "active_after": sum(
                                    slot is not None for slot in slots
                                ),
                            }
                        )
                        diagnostic_step_callback_s += (
                            time.perf_counter() - callback_started
                        )
        finally:
            if admission_prefetcher is not None:
                admission_prefetcher.close()
        admission_prefetch_summary = (
            admission_prefetcher.summary()
            if admission_prefetcher is not None
            else None
        )

        steady_decode_s = decode_s - (first_decode_step_s or 0.0)
        steady_raw_slots = max(0, raw_decode_token_slots - self.batch_size)
        first_active = min(self.batch_size, len(initial))
        steady_effective_tokens = max(0, effective_decode_tokens - first_active)
        run_wall_s = time.perf_counter() - run_started
        directly_accounted_s = sum(
            (
                source_pull_s,
                initial_cache_stack_s,
                initial_arena_allocate_s,
                initial_arena_admission_enqueue_s,
                initial_state_build_s,
                compile_wrap_s or 0.0,
                cache_refill_copy_enqueue_s,
                cache_refill_direct_admission_enqueue_s,
                result_build_s,
                completion_callback_s,
                decode_input_buffer_setup_s,
                decode_input_build_s,
                pre_decode_sync_s,
                production_graph_warmup_s,
                diagnostic_step_callback_s,
                decode_s,
            )
        )
        return {
            "batch_size": self.batch_size,
            "submitted": submitted,
            "completed": completed,
            "decode_iterations": decode_iterations,
            "raw_decode_token_slots": raw_decode_token_slots,
            "effective_decode_tokens": effective_decode_tokens,
            "idle_decode_token_slots": idle_decode_token_slots,
            "decode_s": decode_s,
            "first_decode_step_s": first_decode_step_s,
            "steady_decode_s": steady_decode_s,
            "raw_decode_tokens_per_s": (
                raw_decode_token_slots / decode_s if decode_s > 0 else None
            ),
            "effective_decode_tokens_per_s": (
                effective_decode_tokens / decode_s if decode_s > 0 else None
            ),
            "steady_raw_decode_tokens_per_s": (
                steady_raw_slots / steady_decode_s if steady_decode_s > 0 else None
            ),
            "steady_effective_decode_tokens_per_s": (
                steady_effective_tokens / steady_decode_s
                if steady_decode_s > 0
                else None
            ),
            "slot_refills": slot_refills,
            "compile_wrap_s": compile_wrap_s,
            "compile": compile_meta,
            "production_graph_warmup": {
                "passes": int(graph_warmup_passes),
                "wall_s": production_graph_warmup_s,
                "pass_wall_s": production_graph_warmup_pass_s,
                "warnings": production_graph_warmup_warnings,
                "arena": "actual_admitted_decode_arena",
                "included_in_decode_s": False,
            },
            "timing_detail": {
                "run_wall_s": run_wall_s,
                "source_pull_s": source_pull_s,
                "initial_cache_stack_s": initial_cache_stack_s,
                "initial_arena_allocate_s": initial_arena_allocate_s,
                "initial_arena_admission_enqueue_s": (
                    initial_arena_admission_enqueue_s
                ),
                "initial_state_build_s": initial_state_build_s,
                "cache_refill_copy_enqueue_s": cache_refill_copy_enqueue_s,
                "cache_refill_direct_admission_enqueue_s": (
                    cache_refill_direct_admission_enqueue_s
                ),
                "cache_refill_row_bytes": cache_refill_row_bytes,
                "cache_refill_total_bytes": cache_refill_row_bytes * slot_refills,
                "direct_admission_count": direct_admission_count,
                "direct_cross_kv_bytes": direct_cross_kv_bytes,
                "direct_reset_bytes": direct_reset_bytes,
                "admission_prefetch": admission_prefetch_summary,
                "result_build_s": result_build_s,
                "completion_callback_s": completion_callback_s,
                "decode_input_buffer_setup_s": decode_input_buffer_setup_s,
                "decode_input_host_pinned": decode_input_host_pinned,
                "decode_input_device_tensors_inference": bool(
                    next_token_tensor.is_inference()
                    and cache_position_tensor.is_inference()
                ),
                "decode_input_build_s": decode_input_build_s,
                "explicit_pre_decode_sync": False,
                "pre_decode_sync_s": pre_decode_sync_s,
                "production_graph_warmup_s": production_graph_warmup_s,
                "diagnostic_step_callback_s": diagnostic_step_callback_s,
                "decode_s": decode_s,
                "scheduler_bookkeeping_residual_s": max(
                    0.0, run_wall_s - directly_accounted_s
                ),
            },
        }
