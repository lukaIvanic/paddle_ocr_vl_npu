"""Self-contained PP-DocLayoutV3 to PaddleOCR-VL request frontend."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from paddleocr_vl.serving.types import RecognitionRequest
from utils.timeline import TimelineRecorder

from .layout_postprocess import (
    IMAGE_LABELS,
    LayoutPostprocessor,
    crop_formula_margin,
    crop_layout_regions,
    merge_blocks,
)
from .layout_output import (
    gather_document_images,
    tokenize_table_figures,
)
from .layout_model_runtime import (
    _MaskRectangleFastPath,
    _NpuGraphCoreForward,
    _decoder_forward_final_heads_only,
    _extract_custom_vertices_vectorized,
    _extract_polygon_points_by_masks_owned,
    _mask_to_box_capture_friendly,
    _post_process_selected_masks_only,
)


@dataclass(frozen=True)
class PreparedLayoutPage:
    ordinal: int
    image_path: Path
    image: np.ndarray
    layout_boxes: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    requests: list[RecognitionRequest]
    request_block_indices: list[int]
    figure_token_maps: dict[int, dict[str, str]]
    dropped_figure_paths: set[str]
    document_images: list[dict[str, Any]]
    timing_s: dict[str, float]
    statistics: dict[str, Any]


def _load_layout_labels(model_dir: Path) -> list[str]:
    metadata_path = model_dir / "inference.yml"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"PP-DocLayoutV3 metadata is missing: {metadata_path}"
        )
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    labels = metadata.get("label_list")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{metadata_path} has no label_list")
    return [str(label) for label in labels]


def _find_decoder(model: Any) -> Any:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        decoder = getattr(
            getattr(getattr(model, "model", None), "decoder", None),
            "decoder",
            None,
        )
    if decoder is None:
        decoder = getattr(getattr(model, "model", None), "decoder", None)
    if decoder is None or not hasattr(decoder, "layers"):
        raise RuntimeError("PP-DocLayoutV3 decoder was not found")
    return decoder


def _decode_bgr(path: Path) -> tuple[np.ndarray, dict[str, float | int]]:
    started = time.perf_counter()
    compressed = path.read_bytes()
    read_finished = time.perf_counter()
    encoded = np.frombuffer(compressed, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    decode_finished = time.perf_counter()
    if image is None:
        raise RuntimeError(f"OpenCV failed to decode {path}")
    return image, {
        "compressed_bytes": len(compressed),
        "file_read_s": read_finished - started,
        "image_decode_s": decode_finished - read_finished,
    }


def _prompt_for_label(label: str) -> str:
    if label == "table":
        return "Table Recognition:"
    if label == "chart":
        return "Chart Recognition:"
    if "formula" in label and label != "formula_number":
        return "Formula Recognition:"
    if label == "spotting":
        return "Spotting:"
    if label == "seal":
        return "Seal Recognition:"
    return "OCR:"


def _bgr_to_pil_rgb(image: np.ndarray) -> Image.Image:
    """Decode a BGR NumPy buffer directly into an independent RGB image."""

    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    height, width = image.shape[:2]
    return Image.frombuffer(
        "RGB",
        (width, height),
        image,
        "raw",
        "BGR",
        0,
        1,
    )


class OwnedLayoutFrontend:
    """Own the exact sequential page-to-recognition-request contract."""

    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
        *,
        threshold: float = 0.3,
        timeline: TimelineRecorder | None = None,
        graph_capture: bool = True,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.model_dir = model_dir.expanduser().resolve()
        self.device = device
        self.timeline = timeline
        self.labels = _load_layout_labels(self.model_dir)

        setup_started = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_dir
        )
        self.model.eval().to(self.device)
        self.model_dtype = next(self.model.parameters()).dtype

        decoder = _find_decoder(self.model)
        decoder.forward = MethodType(
            _decoder_forward_final_heads_only,
            decoder,
        )
        decoder._layout_emit_masks = True
        self.processor.extract_custom_vertices = MethodType(
            _extract_custom_vertices_vectorized,
            self.processor,
        )
        self.processor._extract_polygon_points_by_masks = MethodType(
            _extract_polygon_points_by_masks_owned,
            self.processor,
        )
        self.mask_fast_path = _MaskRectangleFastPath(
            self.processor._extract_polygon_points_by_masks
        )
        self.processor._extract_polygon_points_by_masks = self.mask_fast_path
        if graph_capture:
            from transformers.models.pp_doclayout_v3 import (
                modeling_pp_doclayout_v3,
            )

            modeling_pp_doclayout_v3.mask_to_box_coordinate = (
                _mask_to_box_capture_friendly
            )
            self.model.forward = _NpuGraphCoreForward(self.model.forward)
        self.postprocessor = LayoutPostprocessor(
            self.labels,
            threshold=threshold,
        )
        torch.npu.synchronize()
        self.setup_s = time.perf_counter() - setup_started
        self.graph_capture = bool(graph_capture)

    def _span(
        self,
        row: str,
        name: str,
        started_ns: int,
        *,
        flow_id: str,
        event_type: str = "work",
        args: dict[str, Any] | None = None,
    ) -> None:
        if self.timeline is not None:
            self.timeline.record_span(
                row,
                name,
                started_ns,
                time.perf_counter_ns(),
                flow_id=flow_id,
                event_type=event_type,
                args=args or {},
            )

    def _prepare_pixel_values(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Run the processor's exact singleton math without batch plumbing."""

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pixel_values = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        pixel_values = self.processor.resize(
            image=pixel_values,
            size=self.processor.size,
            resample=self.processor.resample,
            antialias=False,
        )
        pixel_values = self.processor.rescale_and_normalize(
            pixel_values,
            self.processor.do_rescale,
            self.processor.rescale_factor,
            self.processor.do_normalize,
            self.processor.image_mean,
            self.processor.image_std,
        )
        return pixel_values.to(
            device=self.device,
            dtype=self.model_dtype,
        )

    @torch.inference_mode()
    def _detect(
        self,
        image_bgr: np.ndarray,
        *,
        flow_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        timing: dict[str, float] = {}
        height, width = image_bgr.shape[:2]

        started = time.perf_counter()
        started_ns = time.perf_counter_ns()
        inputs = {"pixel_values": self._prepare_pixel_values(image_bgr)}
        timing["layout_preprocess_h2d_s"] = time.perf_counter() - started
        self._span(
            "Layout detection",
            "Owned layout preprocessing and H2D",
            started_ns,
            flow_id=flow_id,
        )

        started = time.perf_counter()
        started_ns = time.perf_counter_ns()
        outputs = self.model(**inputs)
        timing["layout_model_submit_s"] = time.perf_counter() - started
        self._span(
            "Layout detection",
            "Owned PP-DocLayoutV3 inference",
            started_ns,
            flow_id=flow_id,
        )

        started = time.perf_counter()
        started_ns = time.perf_counter_ns()
        predictions = _post_process_selected_masks_only(
            self.processor,
            outputs,
            threshold=self.postprocessor.threshold,
            target_sizes=[[height, width]],
        )
        boxes = self.postprocessor(predictions[0], (width, height))
        timing["layout_postprocess_s"] = time.perf_counter() - started
        self._span(
            "Layout postprocess",
            "Owned detector postprocessing",
            started_ns,
            flow_id=flow_id,
            args={"boxes": len(boxes)},
        )
        return boxes, timing

    def prepare_page(
        self,
        image_path: Path,
        ordinal: int,
        *,
        min_pixels: int | None = None,
        max_pixels: int = 1_003_520,
    ) -> PreparedLayoutPage:
        path = image_path.expanduser().resolve()
        flow_id = f"page:{ordinal}"
        page_started = time.perf_counter()
        page_started_ns = time.perf_counter_ns()

        decode_started_ns = time.perf_counter_ns()
        image, decode = _decode_bgr(path)
        self._span(
            "Page input",
            "Owned page read and decode",
            decode_started_ns,
            flow_id=flow_id,
            event_type="io",
            args={
                "bytes": decode["compressed_bytes"],
                "width": image.shape[1],
                "height": image.shape[0],
            },
        )

        boxes, detect_timing = self._detect(image, flow_id=flow_id)
        preparation_started = time.perf_counter()
        preparation_started_ns = time.perf_counter_ns()
        document_images = gather_document_images(image, boxes)
        blocks = crop_layout_regions(image, boxes)
        image_labels = IMAGE_LABELS + ["chart", "seal"]
        blocks = merge_blocks(
            blocks,
            non_merge_labels=image_labels + ["table"],
        )

        effective_min_pixels = int(min_pixels or 112_896)
        requests: list[RecognitionRequest] = []
        request_block_indices: list[int] = []
        figure_token_maps: dict[int, dict[str, str]] = {}
        dropped_figure_paths: set[str] = set()
        for block_index, block in enumerate(blocks):
            block_image = block["img"]
            label = block["label"]
            if label in image_labels or block_image is None:
                continue
            prompt = _prompt_for_label(label)
            if label == "table":
                (
                    block_image,
                    figure_token_map,
                    dropped,
                ) = tokenize_table_figures(
                    block_image,
                    block["box"],
                    document_images,
                )
                figure_token_maps[block_index] = figure_token_map
                dropped_figure_paths.update(dropped)
            elif "formula" in label and label != "formula_number":
                cropped = crop_formula_margin(block_image)
                crop_height, crop_width = cropped.shape[:2]
                if crop_height > 2 and crop_width > 2:
                    block_image = cropped
            requests.append(
                RecognitionRequest(
                    request_id=(
                        f"page_{ordinal:06d}_block_{block_index:06d}"
                    ),
                    crop=_bgr_to_pil_rgb(block_image),
                    prompt=prompt,
                    skip_special_tokens=True,
                    min_pixels=effective_min_pixels,
                    max_pixels=int(max_pixels),
                )
            )
            request_block_indices.append(block_index)
        preparation_s = time.perf_counter() - preparation_started
        self._span(
            "Crop / page preparation",
            "Owned filter, crop, merge, and request build",
            preparation_started_ns,
            flow_id=flow_id,
            args={"requests": len(requests)},
        )

        timing = {
            "file_read_s": float(decode["file_read_s"]),
            "image_decode_s": float(decode["image_decode_s"]),
            **detect_timing,
            "page_preparation_s": preparation_s,
            "page_total_s": time.perf_counter() - page_started,
        }
        self._span(
            "Page input",
            "Owned page frontend",
            page_started_ns,
            flow_id=flow_id,
            event_type="scope",
            args={"requests": len(requests)},
        )
        return PreparedLayoutPage(
            ordinal=ordinal,
            image_path=path,
            image=image,
            layout_boxes=boxes,
            blocks=blocks,
            requests=requests,
            request_block_indices=request_block_indices,
            figure_token_maps=figure_token_maps,
            dropped_figure_paths=dropped_figure_paths,
            document_images=document_images,
            timing_s=timing,
            statistics={
                "raw_layout_boxes": len(boxes),
                "filtered_layout_boxes": len(boxes),
                "merged_blocks": len(blocks),
                "requests": len(requests),
            },
        )
