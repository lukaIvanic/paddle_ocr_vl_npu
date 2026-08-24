"""Concurrent cached layout executors sharing one model and one NPU owner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch_npu

from opendoc_layout_npu import (
    PPDocLayoutV2NpuAdapter,
    _layout_outputs_for_cpu_postprocess,
    post_process_layout_object_detection_exact,
    prepare_layout_resized_uint8_exact,
)
from post_warmup_host_cleanup import cleanup_after_warmup
from torchair_ge_graph_compaction import release_loaded_ge_executors


class SharedLayoutOwner:
    """Run four B2 layout graphs without four model/runtime processes."""

    def __init__(
        self,
        *,
        model_path: Path,
        cache_dir: Path,
        device: str,
        lanes: int = 4,
        batch_size: int = 2,
        threshold: float = 0.5,
    ) -> None:
        if lanes < 1 or batch_size < 1:
            raise ValueError("layout lane and batch counts must be positive")
        self.device = torch.device(device)
        self.lanes = int(lanes)
        self.batch_size = int(batch_size)
        self.threshold = float(threshold)
        self.adapter = PPDocLayoutV2NpuAdapter(
            model_path=model_path,
            device=device,
            dtype="float32",
            reading_order_dtype="float32",
            threshold=threshold,
            execution="torchair",
            compile_cache_dir=cache_dir,
            batch_size=batch_size,
            weight_format="native",
            depthwise_rewrite="native",
            input_color_order="rgb",
        )
        runtime = self.adapter.compiled_runtime
        if runtime is None:
            raise RuntimeError("compiled layout runtime was not created")
        cache_files = sorted(runtime.cache_dir.rglob("compiled_module"))
        if len(cache_files) != 1:
            raise RuntimeError(
                f"expected one layout compiled_module, found {cache_files}"
            )
        try:
            from torch_npu.dynamo.torchair.inference._cache_compiler import (
                CompiledModel,
            )
        except ImportError:
            from torchair.inference._cache_compiler import CompiledModel

        self.executors = [runtime.compiled]
        self.executor_namespaces: list[dict[str, Any]] = []
        for _ in range(1, lanes):
            compiled_model = CompiledModel.load(str(cache_files[0]))
            namespace: dict[str, Any] = {}
            self.executors.append(
                compiled_model.rebase(
                    runtime.stage,
                    global_vars=namespace,
                    func=runtime.stage.forward,
                    cache_dir=str(cache_files[0].parent),
                )
            )
            self.executor_namespaces.append(namespace)
        self.streams = [torch.npu.Stream(device=self.device) for _ in range(lanes)]
        self.pool = ThreadPoolExecutor(
            max_workers=lanes,
            thread_name_prefix="unirec-layout-owner",
        )
        self.calls = 0
        self.pages = 0
        self.wall_s = 0.0
        warm = np.zeros((800, 800, 3), dtype=np.uint8)
        futures = [
            self.pool.submit(self._predict_lane, lane, [warm] * batch_size)
            for lane in range(lanes)
        ]
        for future in futures:
            future.result()
        self.calls = 0
        self.pages = 0
        self.wall_s = 0.0

    def _predict_lane(
        self,
        lane: int,
        images: Sequence[np.ndarray],
    ) -> list[dict[str, Any]]:
        torch_npu.npu.set_device(self.device)
        if not images or len(images) > self.batch_size:
            raise ValueError(
                f"layout lane needs 1..{self.batch_size} pages, got {len(images)}"
            )
        real_count = len(images)
        padded = list(images) + [images[-1]] * (self.batch_size - real_count)
        target_sizes = [image.shape[:2] for image in padded]
        inputs = prepare_layout_resized_uint8_exact(padded)
        with torch.inference_mode(), torch.npu.stream(self.streams[lane]):
            device_pixels = inputs["pixel_values"].to(device=self.device)
            device_pixels = device_pixels.to(dtype=torch.float32).div_(255.0)
            logits, pred_boxes, order_logits = self.executors[lane](device_pixels)
        self.streams[lane].synchronize()
        cpu_outputs = _layout_outputs_for_cpu_postprocess(
            SimpleNamespace(
                logits=logits,
                pred_boxes=pred_boxes,
                order_logits=order_logits,
            )
        )
        prediction = post_process_layout_object_detection_exact(
            cpu_outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )
        results = []
        for image, item in zip(images, prediction[:real_count]):
            height, width = image.shape[:2]
            result_boxes = []
            for score, label_id, box, order in zip(
                item["scores"].tolist(),
                item["labels"].tolist(),
                item["boxes"].tolist(),
                item["order_seq"].tolist(),
            ):
                class_id = int(label_id)
                x1, y1, x2, y2 = box
                result_boxes.append(
                    {
                        "cls_id": class_id,
                        "label": self.adapter.LABEL_MAP.get(
                            class_id,
                            f"class_{class_id}",
                        ),
                        "score": float(score),
                        "coordinate": [
                            float(np.clip(x1, 0, width)),
                            float(np.clip(y1, 0, height)),
                            float(np.clip(x2, 0, width)),
                            float(np.clip(y2, 0, height)),
                        ],
                        "custom_value": float(order),
                    }
                )
            result = self.adapter._filter_overlap_boxes({"boxes": result_boxes})
            result["boxes"] = sorted(
                result["boxes"],
                key=lambda value: value["custom_value"],
            )
            for index, box in enumerate(result["boxes"], start=1):
                box["label"] = f"{box['label']}_{index:02d}"
            results.append(result)
        return results

    def predict(self, images: Sequence[np.ndarray]) -> list[dict[str, Any]]:
        if not images:
            return []
        started = time.perf_counter()
        batches = [
            list(images[start : start + self.batch_size])
            for start in range(0, len(images), self.batch_size)
        ]
        outputs: list[dict[str, Any]] = []
        for start in range(0, len(batches), self.lanes):
            group = batches[start : start + self.lanes]
            futures = [
                self.pool.submit(self._predict_lane, lane, batch)
                for lane, batch in enumerate(group)
            ]
            for future in futures:
                outputs.extend(future.result())
            self.calls += len(group)
        self.pages += len(images)
        self.wall_s += time.perf_counter() - started
        return outputs

    def release(self) -> dict[str, Any]:
        self.pool.shutdown(wait=True)
        lazy_executors = [
            executor
            for executor in self.executors
            if hasattr(executor, "_compiled_model")
        ]
        release = release_loaded_ge_executors(lazy_executors)
        release["rebased_function_count"] = len(self.executors) - len(
            lazy_executors
        )
        self.executors.clear()
        for namespace in self.executor_namespaces:
            namespace.clear()
        self.executor_namespaces.clear()
        self.streams.clear()
        self.adapter = None
        gc.collect()
        torch.npu.empty_cache()
        cleanup = cleanup_after_warmup("shared_layout_owner_release")
        return {
            "executors": release,
            "cleanup": cleanup,
            "pages": self.pages,
            "calls": self.calls,
            "wall_s": self.wall_s,
        }
