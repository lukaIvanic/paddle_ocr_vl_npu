"""Two-stream global K20 vision execution with bounded GE residency."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch_npu

from post_warmup_host_cleanup import cleanup_after_warmup
from torchair_ge_graph_compaction import release_loaded_ge_executors
from vision_bucket_presets import plan_canvas_bucket_calls
from vision_full_batch import (
    BucketedFullVisionRuntime,
    EncodedVisionItem,
    PreprocessedVisionInput,
    VisionBucketSpec,
)


class BoundedVisionOwner:
    """Process all K20 calls globally while loading only two graphs at once."""

    def __init__(
        self,
        runtime: BucketedFullVisionRuntime,
        *,
        lanes: int = 2,
    ) -> None:
        if lanes != 2:
            raise ValueError("bounded vision currently requires exactly two lanes")
        self.runtime = runtime
        self.lanes = lanes
        self.streams = [
            torch.npu.Stream(device=torch.device(runtime.runner.device))
            for _ in range(lanes)
        ]
        self.pool = ThreadPoolExecutor(
            max_workers=lanes,
            thread_name_prefix="unirec-bounded-vision",
        )

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
                outputs.extend(self.runtime._run_bucket(spec, inputs))
                del inputs
                for mapping in mappings:
                    del mapping
        self.streams[lane].synchronize()
        return outputs, time.perf_counter() - started

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
        cleanup_after_warmup("bounded_vision_pair_release")
        return report

    def encode(
        self,
        records: Sequence[dict[str, Any]],
    ) -> tuple[list[EncodedVisionItem], dict[str, Any]]:
        calls, specs_by_key, fallbacks = self._plan(records)
        weighted = sorted(
            calls,
            key=lambda key: (
                -len(calls[key])
                * specs_by_key[key].batch_size
                * specs_by_key[key].width
                * specs_by_key[key].height,
                key,
            ),
        )
        outputs: dict[int, EncodedVisionItem] = {}
        pair_reports = []
        started = time.perf_counter()
        for pair_start in range(0, len(weighted), self.lanes):
            self._release_loaded()
            pair = weighted[pair_start : pair_start + self.lanes]
            futures = [
                self.pool.submit(
                    self._run_key,
                    lane,
                    specs_by_key[key],
                    calls[key],
                )
                for lane, key in enumerate(pair)
            ]
            lane_reports = []
            for key, future in zip(pair, futures):
                encoded, lane_s = future.result()
                for item in encoded:
                    if item.source_index in outputs:
                        raise RuntimeError(
                            f"duplicate encoded source {item.source_index}"
                        )
                    outputs[item.source_index] = item
                lane_reports.append(
                    {
                        "key": key,
                        "calls": len(calls[key]),
                        "real_rows": sum(len(value) for value in calls[key]),
                        "wall_s": lane_s,
                    }
                )
            pair_reports.append(lane_reports)
        fallback_started = time.perf_counter()
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
                output = self.runtime._run_fallback(item)
            self.streams[0].synchronize()
            outputs[output.source_index] = output
            del item, mapping
        fallback_s = time.perf_counter() - fallback_started
        release = self._release_loaded()
        if len(outputs) != len(records):
            raise RuntimeError(
                f"bounded vision produced {len(outputs)} of {len(records)} crops"
            )
        ordered = [outputs[index] for index in range(len(records))]
        report = {
            "wall_s": time.perf_counter() - started,
            "crop_count": len(records),
            "graph_count": len(weighted),
            "pair_count": len(pair_reports),
            "pairs": pair_reports,
            "fallback_count": len(fallbacks),
            "fallback_wall_s": fallback_s,
            "final_release": release,
            "bucket_calls": dict(self.runtime.stats["bucket_calls"]),
            "bucket_real_rows": dict(self.runtime.stats["bucket_real_rows"]),
        }
        return ordered, report

    def close(self) -> None:
        self.pool.shutdown(wait=True)
