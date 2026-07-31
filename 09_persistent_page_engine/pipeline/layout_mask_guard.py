"""Narrow compatibility guard for empty PP-DocLayoutV3 mask crops.

Transformers derives polygon points by cropping each 200x200 detection mask and
resizing that crop to the detected box.  A positive-size box can collapse to a
zero-width or zero-height mask slice after scaling and rounding.  OpenCV then
raises instead of using the bounding rectangle fallback already used for other
invalid polygon cases.

This guard preserves the installed implementation for every valid detection.
Only collapsed mask slices receive the exact integer bounding rectangle that
the upstream method constructs as its default polygon.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np


GUARD_VERSION = 2


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def find_empty_mask_crops(
    boxes: Any,
    masks: Any,
    scale_ratio: Any,
) -> list[dict[str, Any]]:
    """Return detections whose scaled mask crop has an empty dimension."""

    boxes_np = _as_numpy(boxes)
    masks_np = _as_numpy(masks)
    scale_width = _scalar(scale_ratio[0]) / 4
    scale_height = _scalar(scale_ratio[1]) / 4
    mask_height, mask_width = masks_np.shape[1:]
    records: list[dict[str, Any]] = []

    for region_index, box in enumerate(boxes_np):
        x_min, y_min, x_max, y_max = box.astype(np.int32)
        box_width = int(x_max - x_min)
        box_height = int(y_max - y_min)
        if box_width <= 0 or box_height <= 0:
            continue
        x_unclipped = [
            int(round(_scalar(x_min * scale_width))),
            int(round(_scalar(x_max * scale_width))),
        ]
        y_unclipped = [
            int(round(_scalar(y_min * scale_height))),
            int(round(_scalar(y_max * scale_height))),
        ]
        x_start, x_end = [
            int(value) for value in np.clip(x_unclipped, 0, mask_width)
        ]
        y_start, y_end = [
            int(value) for value in np.clip(y_unclipped, 0, mask_height)
        ]
        if x_start < x_end and y_start < y_end:
            continue
        records.append(
            {
                "region_index": int(region_index),
                "box_float": box.astype(np.float64).tolist(),
                "box_int": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "box_width": box_width,
                "box_height": box_height,
                "mask_shape": [int(mask_height), int(mask_width)],
                "scale_ratio": [
                    _scalar(scale_ratio[0]),
                    _scalar(scale_ratio[1]),
                ],
                "scaled_x_unclipped": x_unclipped,
                "scaled_y_unclipped": y_unclipped,
                "scaled_x_clipped": [x_start, x_end],
                "scaled_y_clipped": [y_start, y_end],
            }
        )
    return records


def _rectangle(box: Any) -> np.ndarray:
    x_min, y_min, x_max, y_max = _as_numpy(box).astype(np.int32)
    return np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )


class LayoutMaskGuardState:
    def __init__(self, original: Any) -> None:
        self.original = original
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []

    def record_fallback_regions(
        self,
        detections: int,
        fallback_regions: list[dict[str, Any]],
    ) -> None:
        """Record fallback telemetry from any installed extraction path."""

        if not fallback_regions:
            return
        with self._lock:
            self._events.append(
                {
                    "call_index": len(self._events),
                    "detections": int(detections),
                    "fallback_regions": [
                        dict(record) for record in fallback_regions
                    ],
                }
            )

    def guarded_extract(
        self,
        processor: Any,
        boxes: Any,
        masks: Any,
        scale_ratio: Any,
    ) -> list[Any]:
        try:
            return self.original(processor, boxes, masks, scale_ratio)
        except cv2.error as error:
            if "!ssize.empty()" not in str(error):
                raise

        boxes_np = _as_numpy(boxes)
        masks_np = _as_numpy(masks)
        polygons: list[Any] = []
        fallback_regions: list[dict[str, Any]] = []
        for index in range(len(boxes_np)):
            try:
                single = self.original(
                    processor,
                    boxes_np[index : index + 1],
                    masks_np[index : index + 1],
                    scale_ratio,
                )
            except cv2.error as error:
                if "!ssize.empty()" not in str(error):
                    raise
                diagnostics = find_empty_mask_crops(
                    boxes_np[index : index + 1],
                    masks_np[index : index + 1],
                    scale_ratio,
                )
                record = (
                    dict(diagnostics[0])
                    if diagnostics
                    else {
                        "box_float": boxes_np[index]
                        .astype(np.float64)
                        .tolist(),
                        "box_int": boxes_np[index]
                        .astype(np.int32)
                        .tolist(),
                        "mask_shape": list(masks_np[index].shape),
                        "scale_ratio": [
                            _scalar(scale_ratio[0]),
                            _scalar(scale_ratio[1]),
                        ],
                    }
                )
                record.update(
                    {
                        "region_index": int(index),
                        "precheck_detected": bool(diagnostics),
                        "upstream_error": str(error),
                    }
                )
                fallback_regions.append(record)
                polygons.append(_rectangle(boxes_np[index]))
            else:
                if len(single) != 1:
                    raise RuntimeError(
                        "PP-DocLayout mask extraction returned "
                        f"{len(single)} polygons for one detection"
                    )
                polygons.append(single[0])

        if not fallback_regions:
            raise RuntimeError(
                "PP-DocLayout batch mask extraction raised an empty-resize error, "
                "but every individual detection succeeded"
            )
        self.record_fallback_regions(len(boxes_np), fallback_regions)
        return polygons

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = [
                {
                    **event,
                    "fallback_regions": [
                        dict(record) for record in event["fallback_regions"]
                    ],
                }
                for event in self._events
            ]
        return {
            "guard_version": GUARD_VERSION,
            "behavior": "integer_bounding_rectangle_for_empty_scaled_mask_crop",
            "fallback_calls": len(events),
            "fallback_regions": sum(
                len(event["fallback_regions"]) for event in events
            ),
            "events": events,
        }

    def write_snapshot(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def install_layout_mask_guard(processor_cls: Any | None = None) -> LayoutMaskGuardState:
    """Install the process-wide guard and return its shared telemetry state."""

    if processor_cls is None:
        from transformers.models.pp_doclayout_v3.image_processing_pp_doclayout_v3 import (
            PPDocLayoutV3ImageProcessor,
        )

        processor_cls = PPDocLayoutV3ImageProcessor

    original = processor_cls._extract_polygon_points_by_masks
    existing = getattr(original, "_layout_mask_guard_state", None)
    if existing is not None:
        return existing

    state = LayoutMaskGuardState(original)

    def guarded(
        processor: Any,
        boxes: Any,
        masks: Any,
        scale_ratio: Any,
    ) -> list[Any]:
        return state.guarded_extract(processor, boxes, masks, scale_ratio)

    guarded._layout_mask_guard_state = state
    processor_cls._extract_polygon_points_by_masks = guarded
    return state
