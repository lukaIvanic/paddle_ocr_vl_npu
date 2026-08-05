"""Eager NPU PP-DocLayoutV2 adapter for the official OpenDoc pipeline."""

from __future__ import annotations

import time
from collections import defaultdict
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
        profile_stages: bool = False,
        execution: str = "eager",
        compile_cache_dir: str | Path | None = None,
        graph_warmup_passes: int = 2,
    ) -> None:
        if dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported layout dtype: {dtype}")
        if not str(device).startswith("npu"):
            raise ValueError("PPDocLayoutV2NpuAdapter requires an NPU device")
        if execution not in {"eager", "torchair"}:
            raise ValueError(f"Unsupported layout execution: {execution}")
        if execution == "torchair" and compile_cache_dir is None:
            raise ValueError("TorchAir layout execution requires compile_cache_dir")

        import torch_npu  # noqa: F401
        from tools.infer_doc_onnx import filter_overlap_boxes
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.model_path = Path(model_path).expanduser().resolve()
        self.device = torch.device(device)
        self.dtype = DTYPE_MAP[dtype]
        self.threshold = float(threshold)
        self.profile_stages = bool(profile_stages)
        self.execution = execution
        self._filter_overlap_boxes = filter_overlap_boxes

        started = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_path)
        self.model.eval().to(device=self.device, dtype=self.dtype)
        torch.npu.synchronize()
        self.compiled_runtime = None
        self.graph_warmup = None
        if execution == "torchair":
            from layout_torchair import LayoutFullGraphRuntime

            self.compiled_runtime = LayoutFullGraphRuntime(
                self.model,
                cache_root=Path(compile_cache_dir),
                dtype=self.dtype,
                device=self.device,
                warmup_passes=graph_warmup_passes,
            )
            self.graph_warmup = self.compiled_runtime.warmup
        self.setup_s = time.perf_counter() - started
        self.page_count = 0
        self.forward_s = 0.0
        self.postprocess_s = 0.0
        self.stage_s: dict[str, float] = defaultdict(float)

    def _record_stage(self, name: str, started: float) -> None:
        if self.profile_stages:
            self.stage_s[name] += time.perf_counter() - started

    def reset_timing(self) -> None:
        """Reset measured page work while retaining the loaded model."""
        self.page_count = 0
        self.forward_s = 0.0
        self.postprocess_s = 0.0
        self.stage_s.clear()

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

        total_started = time.perf_counter()
        started = time.perf_counter()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._record_stage("bgr_to_rgb_s", started)

        started = time.perf_counter()
        inputs = self.processor(images=rgb, return_tensors="pt")
        self._record_stage("processor_preprocess_s", started)

        started = time.perf_counter()
        moved = {
            name: tensor.to(
                device=self.device,
                dtype=self.dtype if tensor.is_floating_point() else tensor.dtype,
            )
            for name, tensor in inputs.items()
        }
        torch.npu.synchronize()
        self._record_stage("inputs_h2d_s", started)

        started = time.perf_counter()
        if self.compiled_runtime is None:
            outputs = self.model(**moved)
        else:
            from transformers.models.pp_doclayout_v2.modeling_pp_doclayout_v2 import (
                PPDocLayoutV2ForObjectDetectionOutput,
            )

            logits, pred_boxes, order_logits = self.compiled_runtime(
                moved["pixel_values"]
            )
            outputs = PPDocLayoutV2ForObjectDetectionOutput(
                logits=logits,
                pred_boxes=pred_boxes,
                order_logits=order_logits,
            )
        torch.npu.synchronize()
        forward_s = time.perf_counter() - started
        self.forward_s += forward_s
        if self.profile_stages:
            self.stage_s["model_forward_s"] += forward_s

        postprocess_started = time.perf_counter()
        started = postprocess_started
        prediction = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[(height, width)],
        )[0]
        if self.profile_stages:
            torch.npu.synchronize()
        self._record_stage("hf_box_decode_s", started)

        started = time.perf_counter()
        scores_cpu = prediction["scores"].detach().cpu()
        labels_cpu = prediction["labels"].detach().cpu()
        boxes_cpu = prediction["boxes"].detach().cpu()
        order_sequence_cpu = prediction["order_seq"].detach().cpu()
        scores = scores_cpu.tolist()
        labels = labels_cpu.tolist()
        boxes = boxes_cpu.tolist()
        order_sequence = order_sequence_cpu.tolist()
        self._record_stage("outputs_d2h_s", started)

        started = time.perf_counter()
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
        self._record_stage("result_box_build_s", started)

        started = time.perf_counter()
        result = self._filter_overlap_boxes({"boxes": result_boxes})
        self._record_stage("overlap_filter_s", started)

        started = time.perf_counter()
        result["boxes"] = sorted(
            result["boxes"],
            key=lambda box: box["custom_value"],
        )
        for index, box in enumerate(result["boxes"], start=1):
            box["label"] = f"{box['label']}_{index:02d}"
        self._record_stage("order_and_label_s", started)
        self.postprocess_s += time.perf_counter() - postprocess_started
        self._record_stage("detector_total_s", total_started)
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
        stage_s = dict(self.stage_s)
        return {
            "setup_s": self.setup_s,
            "page_count": self.page_count,
            "forward_s": self.forward_s,
            "postprocess_s": self.postprocess_s,
            "execution": self.execution,
            "graph_warmup": self.graph_warmup,
            "stage_s": stage_s,
            "stage_mean_ms": {
                name: seconds * 1000.0 / self.page_count
                for name, seconds in stage_s.items()
                if self.page_count
            },
        }
