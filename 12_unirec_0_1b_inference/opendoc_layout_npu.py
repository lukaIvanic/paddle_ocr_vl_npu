"""Eager NPU PP-DocLayoutV2 adapter for the official OpenDoc pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


class PPDocLayoutV2NpuAdapter:
    """Match ``LayoutDetectorONNX`` while running Transformers on NPU."""

    LABEL_MAP = {
        0: "abstract",
        1: "algorithm",
        2: "aside_text",
        3: "chart",
        4: "content",
        5: "display_formula",
        6: "doc_title",
        7: "figure_title",
        8: "footer",
        9: "footer_image",
        10: "footnote",
        11: "formula_number",
        12: "header",
        13: "header_image",
        14: "image",
        15: "inline_formula",
        16: "number",
        17: "paragraph_title",
        18: "reference",
        19: "reference_content",
        20: "seal",
        21: "table",
        22: "text",
        23: "vertical_text",
        24: "vision_footnote",
    }

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "npu:0",
        dtype: str = "float32",
        threshold: float = 0.5,
    ) -> None:
        if dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported layout dtype: {dtype}")
        if not str(device).startswith("npu"):
            raise ValueError("PPDocLayoutV2NpuAdapter requires an NPU device")

        import torch_npu  # noqa: F401
        from tools.infer_doc_onnx import filter_overlap_boxes
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.model_path = Path(model_path).expanduser().resolve()
        self.device = torch.device(device)
        self.dtype = DTYPE_MAP[dtype]
        self.threshold = float(threshold)
        self._filter_overlap_boxes = filter_overlap_boxes

        started = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_path)
        self.model.eval().to(device=self.device, dtype=self.dtype)
        torch.npu.synchronize()
        self.setup_s = time.perf_counter() - started
        self.page_count = 0
        self.forward_s = 0.0
        self.postprocess_s = 0.0

    @torch.inference_mode()
    def _predict_one(
        self,
        image: np.ndarray,
        threshold: float,
    ) -> dict[str, Any]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "OpenDoc layout input must be a BGR HxWx3 image, got "
                f"shape={image.shape}"
            )
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        moved = {
            name: tensor.to(
                device=self.device,
                dtype=self.dtype if tensor.is_floating_point() else tensor.dtype,
            )
            for name, tensor in inputs.items()
        }

        torch.npu.synchronize()
        started = time.perf_counter()
        outputs = self.model(**moved)
        torch.npu.synchronize()
        self.forward_s += time.perf_counter() - started

        started = time.perf_counter()
        prediction = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[(height, width)],
        )[0]
        scores = prediction["scores"].detach().cpu().tolist()
        labels = prediction["labels"].detach().cpu().tolist()
        boxes = prediction["boxes"].detach().cpu().tolist()
        order_sequence = prediction["order_seq"].detach().cpu().tolist()

        result_boxes: list[dict[str, Any]] = []
        for score, label_id, box, order in zip(
            scores,
            labels,
            boxes,
            order_sequence,
        ):
            class_id = int(label_id)
            x1, y1, x2, y2 = box
            result_boxes.append(
                {
                    "cls_id": class_id,
                    "label": self.LABEL_MAP.get(class_id, f"class_{class_id}"),
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

        result = self._filter_overlap_boxes({"boxes": result_boxes})
        result["boxes"] = sorted(
            result["boxes"],
            key=lambda box: box["custom_value"],
        )
        for index, box in enumerate(result["boxes"], start=1):
            box["label"] = f"{box['label']}_{index:02d}"
        self.postprocess_s += time.perf_counter() - started
        self.page_count += 1
        return result

    def __call__(
        self,
        images: np.ndarray | list[np.ndarray],
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(images, np.ndarray):
            images = [images]
        active_threshold = self.threshold if threshold is None else float(threshold)
        return [self._predict_one(image, active_threshold) for image in images]

    def timing_summary(self) -> dict[str, Any]:
        return {
            "setup_s": self.setup_s,
            "page_count": self.page_count,
            "forward_s": self.forward_s,
            "postprocess_s": self.postprocess_s,
        }
