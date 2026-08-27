"""Multi-stream global K20 vision execution with bounded GE residency."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch_npu

from post_warmup_host_cleanup import purge_host_allocator_pages
from tbe_compiler_lifecycle import deinitialize_after_warmup
from torchair_ge_graph_compaction import release_loaded_ge_executors
from vision_bucket_presets import (
    assign_vision_bucket_cache_slots,
    plan_canvas_bucket_calls,
)
from vision_full_batch import (
    EXTENDED_FLAT_GLOBAL_CONTEXT_BUCKET_KEYS,
    FLAT_GLOBAL_CONTEXT_BUCKET_KEYS,
    BucketedFullVisionRuntime,
    EncodedVisionItem,
    PreprocessedVisionInput,
    VisionBucketSpec,
)


class BoundedVisionOwner:
    """Process all K20 calls while loading at most ``lanes`` graphs at once."""

    def __init__(
        self,
        runtime: BucketedFullVisionRuntime,
        *,
        lanes: int = 2,
        same_key_shards: int = 1,
        sharded_key_count: int = 4,
        fallback_runtime: BucketedFullVisionRuntime | None = None,
        deinitialize_tbe_after_first_group: bool = True,
    ) -> None:
        if not 1 <= lanes <= 4:
            raise ValueError("bounded vision lanes must be in [1, 4]")
        self.runtime = runtime
        self.lanes = lanes
        if not 1 <= same_key_shards <= lanes:
            raise ValueError("same-key vision shards must be in [1, lanes]")
        if sharded_key_count < 0:
            raise ValueError("sharded vision key count cannot be negative")
        self.same_key_shards = int(same_key_shards)
        self.sharded_key_count = int(sharded_key_count)
        self.fallback_runtime = fallback_runtime
        self.deinitialize_tbe_after_first_group = bool(
            deinitialize_tbe_after_first_group
        )
        self.streams = [
            torch.npu.Stream(device=torch.device(runtime.runner.device))
            for _ in range(lanes)
        ]
        self.pool = ThreadPoolExecutor(
            max_workers=lanes,
            thread_name_prefix="unirec-bounded-vision",
        )
        self.early_tbe_deinit: dict[str, Any] | None = None
        self._tbe_deinit_attempted = False
        self.host_purge_statuses: list[int | None] = []

    def _compile_method(self, key: str) -> Callable[..., torch.Tensor]:
        slots = assign_vision_bucket_cache_slots(
            self.runtime.specs,
            slot_count=max(10, len(self.runtime.specs)),
        )
        slot = dict(zip((spec.key for spec in self.runtime.specs), slots))[key]
        if key in (
            FLAT_GLOBAL_CONTEXT_BUCKET_KEYS
            | EXTENDED_FLAT_GLOBAL_CONTEXT_BUCKET_KEYS
        ):
            method_name = f"_forward_flat_bucket_slot_{slot}"
        else:
            method_name = f"_forward_bucket_slot_{slot}"
        return getattr(self.runtime.modules[key], method_name)

    def _clone_executor(
        self,
        key: str,
    ) -> tuple[Callable[..., torch.Tensor], dict[str, Any]]:
        cache_files = sorted(self.runtime.cache_dirs[key].rglob("compiled_module"))
        if len(cache_files) != 1:
            raise RuntimeError(
                f"expected one compiled_module for {key}, found {cache_files}"
            )
        try:
            from torch_npu.dynamo.torchair.inference._cache_compiler import (
                CompiledModel,
            )
        except ImportError:
            from torchair.inference._cache_compiler import CompiledModel
        namespace = {"__name__": f"unirec_shared_vision_{key}"}
        executor = CompiledModel.load(str(cache_files[0])).rebase(
            self.runtime.modules[key],
            global_vars=namespace,
            func=self._compile_method(key),
            cache_dir=str(cache_files[0].parent),
        )
        return executor, namespace

    @staticmethod
    def _mapping(descriptor: dict[str, Any]) -> np.memmap:
        return np.memmap(
            descriptor["path"],
            mode="r+",
            dtype=np.uint8,
            offset=int(descriptor["offset"]),
            shape=tuple(int(value) for value in descriptor["shape"]),
        )

    def _plan(
        self,
        records: Sequence[dict[str, Any]],
    ) -> tuple[
        dict[str, list[list[dict[str, Any]]]],
        dict[str, VisionBucketSpec],
        list[dict[str, Any]],
    ]:
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {
            canvas: [] for canvas in self.runtime.specs_by_canvas
        }
        fallbacks = []
        for record in records:
            width, height = (
                int(record["processed_image_size"][0]),
                int(record["processed_image_size"][1]),
            )
            canvas = self.runtime.select_canvas(width, height)
            if canvas is None:
                fallbacks.append(record)
            else:
                grouped[canvas].append(record)
        calls: dict[str, list[list[dict[str, Any]]]] = {}
        specs_by_key: dict[str, VisionBucketSpec] = {}
        for canvas in self.runtime.specs_by_canvas:
            pending = grouped[canvas]
            offset = 0
            for spec in plan_canvas_bucket_calls(
                self.runtime.specs_by_canvas[canvas],
                len(pending),
            ):
                real_rows = min(spec.batch_size, len(pending) - offset)
                calls.setdefault(spec.key, []).append(
                    pending[offset : offset + real_rows]
                )
                specs_by_key[spec.key] = spec
                offset += real_rows
            if offset != len(pending):
                raise RuntimeError(
                    f"K20 plan consumed {offset} of {len(pending)} rows for {canvas}"
                )
        return calls, specs_by_key, fallbacks

    def _run_key(
        self,
        lane: int,
        spec: VisionBucketSpec,
        calls: Sequence[Sequence[dict[str, Any]]],
        compiled_executor: Callable[..., torch.Tensor] | None = None,
    ) -> tuple[list[EncodedVisionItem], float]:
        torch_npu.npu.set_device(self.runtime.runner.device)
        outputs: list[EncodedVisionItem] = []
        started = time.perf_counter()
        with torch.npu.stream(self.streams[lane]):
            for call in calls:
                mappings = [
                    self._mapping(record["processed_pixel_values_descriptor"])
                    for record in call
                ]
                inputs = [
                    PreprocessedVisionInput(
                        source_index=int(record["source_index"]),
                        pixel_values=mapping,
                        original_image_size=tuple(
                            int(value) for value in record["source_image_size"]
                        ),
                        image_source=str(record["request_id"]),
                    )
                    for record, mapping in zip(call, mappings)
                ]
                outputs.extend(
                    self.runtime._run_bucket(
                        spec,
                        inputs,
                        compiled_executor=compiled_executor,
                    )
                )
                del inputs
                for mapping in mappings:
                    del mapping
        self.streams[lane].synchronize()
        return outputs, time.perf_counter() - started

    @staticmethod
    def _split_calls(
        calls: Sequence[Sequence[dict[str, Any]]],
        parts: int,
    ) -> list[list[Sequence[dict[str, Any]]]]:
        base, extra = divmod(len(calls), parts)
        output = []
        offset = 0
        for part in range(parts):
            count = base + int(part < extra)
            output.append(list(calls[offset : offset + count]))
            offset += count
        return [value for value in output if value]

    def _task_groups(
        self,
        calls: dict[str, list[list[dict[str, Any]]]],
        specs_by_key: dict[str, VisionBucketSpec],
    ) -> tuple[list[list[dict[str, Any]]], list[str]]:
        def estimated_ms(key: str) -> float:
            spec = specs_by_key[key]
            per_call = spec.planning_cost_ms
            if per_call is None:
                per_call = (
                    spec.batch_size * spec.width * spec.height / 1_000_000.0
                )
            return float(per_call) * len(calls[key])

        ranked = sorted(calls, key=lambda key: (-estimated_ms(key), key))
        shard_keys = (
            [key for key in ranked if len(calls[key]) >= self.same_key_shards][
                : self.sharded_key_count
            ]
            if self.same_key_shards > 1
            else []
        )
        groups: list[list[dict[str, Any]]] = []
        keys_per_sharded_group = max(1, self.lanes // self.same_key_shards)
        for start in range(0, len(shard_keys), keys_per_sharded_group):
            group = []
            for key in shard_keys[start : start + keys_per_sharded_group]:
                partitions = self._split_calls(
                    calls[key],
                    self.same_key_shards,
                )
                for shard_index, partition in enumerate(partitions):
                    group.append(
                        {
                            "key": key,
                            "shard_index": shard_index,
                            "shard_count": len(partitions),
                            "calls": partition,
                        }
                    )
            groups.append(group)
        remaining = [key for key in ranked if key not in set(shard_keys)]
        for start in range(0, len(remaining), self.lanes):
            groups.append(
                [
                    {
                        "key": key,
                        "shard_index": 0,
                        "shard_count": 1,
                        "calls": calls[key],
                    }
                    for key in remaining[start : start + self.lanes]
                ]
            )
        return groups, shard_keys

    def _release_loaded(self) -> dict[str, Any]:
        loaded = [
            value
            for value in self.runtime.compiled.values()
            if getattr(value, "_compiled_model", None) is not None
        ]
        if not loaded:
            return {"executor_count": 0, "executors": []}
        torch.npu.synchronize()
        report = release_loaded_ge_executors(loaded)
        gc.collect()
        torch.npu.empty_cache()
        self.host_purge_statuses.append(purge_host_allocator_pages())
        return report

    def _release_loaded_except(self, retained_keys: set[str]) -> dict[str, Any]:
        loaded_by_key = {
            key: value
            for key, value in self.runtime.compiled.items()
            if getattr(value, "_compiled_model", None) is not None
        }
        released_keys = sorted(set(loaded_by_key) - retained_keys)
        if not released_keys:
            return {
                "executor_count": 0,
                "executors": [],
                "released_keys": [],
                "retained_keys": sorted(set(loaded_by_key) & retained_keys),
            }
        torch.npu.synchronize()
        report = release_loaded_ge_executors(
            [loaded_by_key[key] for key in released_keys]
        )
        report["released_keys"] = released_keys
        report["retained_keys"] = sorted(set(loaded_by_key) & retained_keys)
        gc.collect()
        torch.npu.empty_cache()
        self.host_purge_statuses.append(purge_host_allocator_pages())
        return report

    def encode(
        self,
        records: Sequence[dict[str, Any]],
        *,
        on_encoded_batch: Callable[[list[EncodedVisionItem]], None] | None = None,
        retain_outputs: bool = True,
        retain_loaded_graphs: bool = False,
    ) -> tuple[list[EncodedVisionItem], dict[str, Any]]:
        if not retain_outputs and on_encoded_batch is None:
            raise ValueError("non-retained vision output requires a callback")
        calls, specs_by_key, fallbacks = self._plan(records)
        task_groups, shard_keys = self._task_groups(calls, specs_by_key)
        outputs: dict[int, EncodedVisionItem] = {}
        seen_outputs: set[int] = set()
        pair_reports = []
        residency_reports = []
        started = time.perf_counter()
        for group in task_groups:
            group_keys = {str(task["key"]) for task in group}
            residency_reports.append(
                self._release_loaded_except(group_keys)
                if retain_loaded_graphs
                else self._release_loaded()
            )
            primary_claimed: set[str] = set()
            executors: list[Callable[..., torch.Tensor]] = []
            clone_executors: list[Callable[..., torch.Tensor]] = []
            clone_namespaces: list[dict[str, Any]] = []
            for task in group:
                key = str(task["key"])
                if key not in primary_claimed:
                    executor = self.runtime.compiled[key]
                    primary_claimed.add(key)
                else:
                    executor, namespace = self._clone_executor(key)
                    clone_executors.append(executor)
                    clone_namespaces.append(namespace)
                executors.append(executor)
            futures = [
                self.pool.submit(
                    self._run_key,
                    lane,
                    specs_by_key[str(task["key"])],
                    task["calls"],
                    executor,
                )
                for lane, (task, executor) in enumerate(zip(group, executors))
            ]
            lane_reports = []
            for task, future in zip(group, futures):
                key = str(task["key"])
                encoded, lane_s = future.result()
                for item in encoded:
                    if item.source_index in seen_outputs:
                        raise RuntimeError(
                            f"duplicate encoded source {item.source_index}"
                        )
                    seen_outputs.add(item.source_index)
                    if retain_outputs:
                        outputs[item.source_index] = item
                if on_encoded_batch is not None:
                    on_encoded_batch(encoded)
                lane_reports.append(
                    {
                        "key": key,
                        "shard_index": int(task["shard_index"]),
                        "shard_count": int(task["shard_count"]),
                        "calls": len(task["calls"]),
                        "real_rows": sum(len(value) for value in task["calls"]),
                        "wall_s": lane_s,
                    }
                )
            clone_executors.clear()
            executors.clear()
            for namespace in clone_namespaces:
                namespace.clear()
            clone_namespaces.clear()
            gc.collect()
            torch.npu.empty_cache()
            self.host_purge_statuses.append(purge_host_allocator_pages())
            if (
                self.deinitialize_tbe_after_first_group
                and not self._tbe_deinit_attempted
            ):
                self._tbe_deinit_attempted = True
                self.early_tbe_deinit = deinitialize_after_warmup(
                    "bounded_vision_first_group_loaded"
                )
            pair_reports.append(lane_reports)
        fallback_started = time.perf_counter()
        fallback_release: dict[str, Any] | None = None
        compiled_fallbacks = 0
        eager_fallbacks = 0
        if fallbacks and self.fallback_runtime is not None:
            self._release_loaded()
        for record in fallbacks:
            mapping = self._mapping(record["processed_pixel_values_descriptor"])
            item = PreprocessedVisionInput(
                source_index=int(record["source_index"]),
                pixel_values=mapping,
                original_image_size=tuple(
                    int(value) for value in record["source_image_size"]
                ),
                image_source=str(record["request_id"]),
            )
            with torch.npu.stream(self.streams[0]):
                use_compiled_fallback = (
                    self.fallback_runtime is not None
                    and item.processed_width not in {320, 512, 576}
                )
                if not use_compiled_fallback:
                    output = self.runtime._run_fallback(item)
                    eager_fallbacks += 1
                else:
                    assert self.fallback_runtime is not None
                    spec = self.fallback_runtime.select_bucket(
                        item.processed_width,
                        item.processed_height,
                    )
                    if spec is None:
                        raise RuntimeError(
                            "compiled fallback graph cannot fit "
                            f"{item.processed_width}x{item.processed_height}"
                        )
                    output = self.fallback_runtime._run_bucket(spec, [item])[0]
                    compiled_fallbacks += 1
            self.streams[0].synchronize()
            if output.source_index in seen_outputs:
                raise RuntimeError(f"duplicate encoded source {output.source_index}")
            seen_outputs.add(output.source_index)
            if retain_outputs:
                outputs[output.source_index] = output
            if on_encoded_batch is not None:
                on_encoded_batch([output])
            del item, mapping
        fallback_s = time.perf_counter() - fallback_started
        if self.fallback_runtime is not None:
            loaded_fallbacks = [
                value
                for value in self.fallback_runtime.compiled.values()
                if getattr(value, "_compiled_model", None) is not None
            ]
            if loaded_fallbacks:
                torch.npu.synchronize()
                fallback_release = release_loaded_ge_executors(loaded_fallbacks)
                gc.collect()
                torch.npu.empty_cache()
                self.host_purge_statuses.append(purge_host_allocator_pages())
        release = (
            {
                "executor_count": 0,
                "executors": [],
                "retained_for_next_window": True,
            }
            if retain_loaded_graphs and not fallbacks
            else self._release_loaded()
        )
        resident_keys = sorted(
            key
            for key, value in self.runtime.compiled.items()
            if getattr(value, "_compiled_model", None) is not None
        )
        if len(resident_keys) > self.lanes:
            raise RuntimeError(
                "bounded vision retained more graphs than lanes: "
                f"{resident_keys}"
            )
        if len(seen_outputs) != len(records):
            raise RuntimeError(
                f"bounded vision produced {len(seen_outputs)} of {len(records)} crops"
            )
        ordered = (
            [outputs[index] for index in range(len(records))]
            if retain_outputs
            else []
        )
        report = {
            "wall_s": time.perf_counter() - started,
            "crop_count": len(records),
            "graph_count": len(calls),
            "task_count": sum(len(group) for group in task_groups),
            "pair_count": len(pair_reports),
            "pairs": pair_reports,
            "graph_residency": residency_reports,
            "retain_loaded_graphs": bool(retain_loaded_graphs),
            "resident_keys": resident_keys,
            "same_key_shards": self.same_key_shards,
            "sharded_key_count": self.sharded_key_count,
            "sharded_keys": shard_keys,
            "fallback_count": len(fallbacks),
            "fallback_wall_s": fallback_s,
            "fallback_execution": (
                "hybrid_compiled_960x1408_b1"
                if self.fallback_runtime is not None
                else "eager"
            ),
            "compiled_fallback_count": compiled_fallbacks,
            "eager_fallback_count": eager_fallbacks,
            "fallback_bucket_calls": (
                dict(self.fallback_runtime.stats["bucket_calls"])
                if self.fallback_runtime is not None
                else None
            ),
            "fallback_release": fallback_release,
            "final_release": release,
            "early_tbe_deinit": self.early_tbe_deinit,
            "host_purge_statuses": self.host_purge_statuses,
            "bucket_calls": dict(self.runtime.stats["bucket_calls"]),
            "bucket_real_rows": dict(self.runtime.stats["bucket_real_rows"]),
        }
        return ordered, report

    def close(self) -> None:
        self._release_loaded()
        self.pool.shutdown(wait=True)
