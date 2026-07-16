"""Real PP-DocLayoutV3 inference through Transformers/PyTorch."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from schema import Box, LayoutRegion
from timing import synchronize, timed_wall


class PPDocLayoutV3Runtime:
    def __init__(self, model_dir: Path, device: torch.device, threshold: float = 0.3):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.model_dir = model_dir.expanduser().resolve()
        self.device = device
        self.threshold = float(threshold)

        started = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_dir)
        self.model.eval().to(self.device)
        synchronize(self.device)
        self.setup_timing_s = {
            "layout_model_load": time.perf_counter() - started,
        }

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> tuple[list[LayoutRegion], dict[str, float]]:
        total_started = time.perf_counter()

        preprocess_started = time.perf_counter()
        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt")
        preprocess_s = time.perf_counter() - preprocess_started

        moved, transfer_s = timed_wall(
            self.device,
            lambda: {name: value.to(self.device) for name, value in inputs.items()},
        )
        outputs, inference_s = timed_wall(
            self.device,
            lambda: self.model(**moved),
        )

        post_started = time.perf_counter()
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        raw = self.processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )[0]
        regions = self._normalize(raw)
        postprocess_s = time.perf_counter() - post_started
        return regions, {
            "layout_preprocess": float(preprocess_s),
            "layout_h2d": float(transfer_s),
            "layout_inference": float(inference_s),
            "layout_postprocess": float(postprocess_s),
            "layout_total": float(time.perf_counter() - total_started),
        }

    def _normalize(self, result: dict[str, Any]) -> list[LayoutRegion]:
        polygons = result.get("polygon_points") or [None] * len(result["scores"])
        regions: list[LayoutRegion] = []
        for order, (score, label_id, box, polygon) in enumerate(
            zip(result["scores"], result["labels"], result["boxes"], polygons),
            start=1,
        ):
            values = [float(value) for value in box.detach().cpu().tolist()]
            normalized_polygon = None
            if polygon is not None:
                normalized_polygon = [
                    [float(point[0]), float(point[1])]
                    for point in polygon.tolist()
                ]
            label_index = int(label_id.item())
            id2label = self.model.config.id2label
            label = id2label.get(label_index, id2label.get(str(label_index), str(label_index)))
            region = LayoutRegion(
                order=order,
                label_id=label_index,
                label=str(label),
                score=float(score.item()),
                box=Box(*values),
                polygon_points=normalized_polygon,
            )
            if region.box.width > 0 and region.box.height > 0:
                regions.append(region)
        return regions
