#!/usr/bin/env python3
"""Localize a page-specific PP-DocLayoutV3 failure by synchronization stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from pipeline.layout_frontend import OwnedLayoutFrontend, _decode_rgb
from pipeline.omnidocbench_defaults import OMNIDOCBENCH_PAGE_COUNT


DEFAULT_DATASET_JSON = Path(
    "/workspace/datasets/OmniDocBench/OmniDocBench.json"
)
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")
DEFAULT_LAYOUT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_DATASET_JSON,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
    )
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=DEFAULT_LAYOUT_MODEL,
    )
    parser.add_argument("--page-index", type=int, required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "npu"),
        default="npu",
    )
    parser.add_argument(
        "--model-backend",
        choices=("transformers", "owned"),
        default="transformers",
    )
    parser.add_argument(
        "--layout-indexput-compat",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.page_index < 0:
        parser.error("--page-index must be non-negative")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_contract(tensor: torch.Tensor | None) -> Any:
    if tensor is None:
        return None
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": [int(value) for value in tensor.stride()],
        "contiguous": bool(tensor.is_contiguous()),
    }


class ProgressRecorder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.progress_path = output_dir / "progress.jsonl"
        self.summary_path = output_dir / "summary.json"
        self.started = time.perf_counter()
        self.records: list[dict[str, Any]] = []
        self.current_stage = "initialization"

    def record(
        self,
        stage: str,
        status: str,
        **details: Any,
    ) -> None:
        self.current_stage = stage
        record = {
            "elapsed_s": time.perf_counter() - self.started,
            "stage": stage,
            "status": status,
            **details,
        }
        self.records.append(record)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        print(
            f"[layout-page-probe] {status.upper()} {stage} "
            f"elapsed_s={record['elapsed_s']:.6f}",
            flush=True,
        )

    def write_summary(
        self,
        *,
        status: str,
        configuration: dict[str, Any],
        failure: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "status": status,
            "last_stage": self.current_stage,
            "configuration": configuration,
            "records": self.records,
            "failure": failure,
        }
        self.summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def _stage_start(
    recorder: ProgressRecorder,
    stage: str,
) -> None:
    recorder.record(stage, "start")


def _staged_postprocess(
    frontend: OwnedLayoutFrontend,
    outputs: Any,
    *,
    height: int,
    width: int,
    recorder: ProgressRecorder,
    device: torch.device,
) -> dict[str, Any]:
    """Run the production selection math with a sync after every op group."""

    processor = frontend.processor
    threshold = frontend.postprocessor.threshold
    boxes = outputs.pred_boxes
    logits = outputs.logits
    order_logits = outputs.order_logits
    masks = outputs.out_masks

    stage = "postprocess_order_sequence"
    _stage_start(recorder, stage)
    order_seqs = processor._get_order_seqs(order_logits)
    _synchronize(device)
    recorder.record(stage, "pass", output=_tensor_contract(order_seqs))

    stage = "postprocess_box_conversion_and_scale"
    _stage_start(recorder, stage)
    box_centers, box_dimensions = torch.split(boxes, 2, dim=-1)
    boxes = torch.cat(
        [
            box_centers - 0.5 * box_dimensions,
            box_centers + 0.5 * box_dimensions,
        ],
        dim=-1,
    )
    image_height, image_width = torch.as_tensor(
        [[height, width]]
    ).unbind(1)
    scale_factor = torch.stack(
        [image_width, image_height, image_width, image_height],
        dim=1,
    ).to(boxes.device)
    boxes = boxes * scale_factor[:, None, :]
    _synchronize(device)
    recorder.record(stage, "pass", output=_tensor_contract(boxes))

    stage = "postprocess_sigmoid_and_topk"
    _stage_start(recorder, stage)
    query_count = logits.shape[1]
    class_count = logits.shape[2]
    scores = torch.sigmoid(logits)
    scores, flattened_indices = torch.topk(
        scores.flatten(1),
        query_count,
        dim=-1,
    )
    labels = flattened_indices % class_count
    query_indices = flattened_indices // class_count
    _synchronize(device)
    recorder.record(
        stage,
        "pass",
        scores=_tensor_contract(scores),
        flattened_indices=_tensor_contract(flattened_indices),
    )

    stage = "postprocess_threshold_nonzero"
    _stage_start(recorder, stage)
    kept_positions = torch.nonzero(
        scores[0] >= threshold,
        as_tuple=False,
    ).squeeze(-1)
    _synchronize(device)
    recorder.record(
        stage,
        "pass",
        threshold=float(threshold),
        kept_positions=_tensor_contract(kept_positions),
    )

    stage = "postprocess_reading_order_gathers"
    _stage_start(recorder, stage)
    kept_order = order_seqs[0].gather(
        0,
        query_indices[0].gather(0, kept_positions),
    )
    kept_order, order_indices = torch.sort(kept_order)
    selected_positions = kept_positions.gather(0, order_indices)
    selected_queries = query_indices[0].gather(
        0,
        selected_positions,
    )
    _synchronize(device)
    recorder.record(
        stage,
        "pass",
        selected_queries=_tensor_contract(selected_queries),
    )

    stage = "postprocess_metadata_gathers"
    _stage_start(recorder, stage)
    selected_scores = scores[0].gather(0, selected_positions)
    selected_labels = labels[0].gather(0, selected_positions)
    selected_boxes = boxes[0].index_select(0, selected_queries)
    _synchronize(device)
    recorder.record(
        stage,
        "pass",
        scores=_tensor_contract(selected_scores),
        labels=_tensor_contract(selected_labels),
        boxes=_tensor_contract(selected_boxes),
    )

    stage = "postprocess_metadata_d2h"
    _stage_start(recorder, stage)
    cpu_boxes = selected_boxes.detach().cpu()
    cpu_scores = selected_scores.detach().cpu()
    cpu_labels = selected_labels.detach().cpu()
    cpu_order = kept_order.detach().cpu()
    _synchronize(device)
    recorder.record(
        stage,
        "pass",
        selected_count=int(cpu_scores.shape[0]),
    )

    stage = "postprocess_mask_index_select"
    _stage_start(recorder, stage)
    selected_masks = masks[0].index_select(0, selected_queries)
    _synchronize(device)
    recorder.record(
        stage,
        "pass",
        output=_tensor_contract(selected_masks),
    )

    stage = "postprocess_mask_sigmoid_and_threshold"
    _stage_start(recorder, stage)
    selected_masks = selected_masks.sigmoid() > threshold
    _synchronize(device)
    recorder.record(
        stage,
        "pass",
        output=_tensor_contract(selected_masks),
    )

    stage = "postprocess_mask_d2h"
    _stage_start(recorder, stage)
    cpu_masks = selected_masks.detach().cpu()
    _synchronize(device)
    recorder.record(stage, "pass", output=_tensor_contract(cpu_masks))

    stage = "postprocess_mask_polygon_cpu"
    _stage_start(recorder, stage)
    polygons = processor._extract_polygon_points_by_masks(
        cpu_boxes.numpy(),
        cpu_masks.numpy(),
        [
            processor.size["width"] / width,
            processor.size["height"] / height,
        ],
    )
    recorder.record(stage, "pass", polygon_count=len(polygons))
    return {
        "scores": cpu_scores,
        "labels": cpu_labels,
        "boxes": cpu_boxes,
        "order_seq": cpu_order,
        "polygon_points": polygons,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    dataset_json = args.dataset_json.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    model_dir = args.layout_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    recorder = ProgressRecorder(output_dir)

    configuration: dict[str, Any] = {
        "dataset_json": str(dataset_json),
        "images_dir": str(images_dir),
        "layout_model": str(model_dir),
        "page_index": int(args.page_index),
        "device": args.device,
        "model_backend": args.model_backend,
        "layout_indexput_compat": bool(args.layout_indexput_compat),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
        "ascend_process_log_path": os.environ.get(
            "ASCEND_PROCESS_LOG_PATH"
        ),
        "ascend_work_path": os.environ.get("ASCEND_WORK_PATH"),
    }

    try:
        stage = "dataset_and_image_resolution"
        _stage_start(recorder, stage)
        annotations = json.loads(dataset_json.read_text(encoding="utf-8"))
        if len(annotations) != OMNIDOCBENCH_PAGE_COUNT:
            raise ValueError(
                f"expected {OMNIDOCBENCH_PAGE_COUNT} pages, "
                f"got {len(annotations)}"
            )
        if args.page_index >= len(annotations):
            raise IndexError(
                f"page index {args.page_index} is outside "
                f"0..{len(annotations) - 1}"
            )
        relative = Path(
            annotations[args.page_index]["page_info"]["image_path"]
        )
        image_path = images_dir / relative.name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        configuration.update(
            {
                "image_path": str(image_path),
                "image_bytes": image_path.stat().st_size,
                "image_sha256": _sha256(image_path),
            }
        )
        recorder.record(
            stage,
            "pass",
            image_path=str(image_path),
            image_bytes=image_path.stat().st_size,
            image_sha256=configuration["image_sha256"],
        )

        stage = "page_read_and_decode"
        _stage_start(recorder, stage)
        image_rgb, decode_timing = _decode_rgb(image_path)
        recorder.record(
            stage,
            "pass",
            image_shape=[int(value) for value in image_rgb.shape],
            image_dtype=str(image_rgb.dtype),
            decode_timing=decode_timing,
        )

        stage = "device_initialization"
        _stage_start(recorder, stage)
        device = torch.device(
            "cpu" if args.device == "cpu" else "npu:0"
        )
        if device.type == "npu":
            import torch_npu

            if not torch.npu.is_available():
                raise RuntimeError("NPU is unavailable")
            torch.npu.set_compile_mode(jit_compile=False)
            configuration["torch_npu"] = torch_npu.__version__
            configuration["npu_name"] = torch.npu.get_device_name(0)
        recorder.record(
            stage,
            "pass",
            resolved_device=str(device),
            torch_npu=configuration.get("torch_npu"),
            npu_name=configuration.get("npu_name"),
        )

        stage = "frontend_and_model_setup"
        _stage_start(recorder, stage)
        frontend = OwnedLayoutFrontend(
            model_dir,
            device,
            graph_capture=False,
            device_stage_timing=False,
            npu_indexput_compat=args.layout_indexput_compat,
            model_backend=args.model_backend,
        )
        _synchronize(device)
        configuration["model_dtype"] = str(frontend.model_dtype)
        configuration["frontend_setup_s"] = float(frontend.setup_s)
        recorder.record(
            stage,
            "pass",
            setup_s=float(frontend.setup_s),
            model_dtype=str(frontend.model_dtype),
            graph_capture=bool(frontend.graph_capture),
            npu_indexput_compat=bool(frontend.npu_indexput_compat),
        )

        stage = "layout_preprocess_and_h2d"
        _stage_start(recorder, stage)
        pixel_values = frontend._prepare_pixel_values(image_rgb)
        _synchronize(device)
        recorder.record(
            stage,
            "pass",
            pixel_values=_tensor_contract(pixel_values),
        )

        stage = "fixed_shape_layout_model_forward"
        _stage_start(recorder, stage)
        outputs = frontend.model(pixel_values=pixel_values)
        _synchronize(device)
        output_contract = {
            name: _tensor_contract(getattr(outputs, name, None))
            for name in (
                "logits",
                "pred_boxes",
                "order_logits",
                "out_masks",
            )
        }
        recorder.record(
            stage,
            "pass",
            outputs=output_contract,
        )

        height, width = image_rgb.shape[:2]
        prediction = _staged_postprocess(
            frontend,
            outputs,
            height=height,
            width=width,
            recorder=recorder,
            device=device,
        )
        selected_count = int(prediction["scores"].shape[0])

        stage = "cpu_structural_layout_postprocess"
        _stage_start(recorder, stage)
        boxes = frontend.postprocessor(
            prediction,
            (width, height),
        )
        recorder.record(
            stage,
            "pass",
            selected_count=selected_count,
            filtered_box_count=len(boxes),
            label_counts={
                label: sum(box["label"] == label for box in boxes)
                for label in sorted({box["label"] for box in boxes})
            },
        )
    except BaseException as error:
        traceback_text = traceback.format_exc()
        (output_dir / "traceback.txt").write_text(
            traceback_text,
            encoding="utf-8",
        )
        failure = {
            "stage": recorder.current_stage,
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback_path": str(output_dir / "traceback.txt"),
        }
        recorder.record(
            recorder.current_stage,
            "fail",
            exception_type=type(error).__name__,
            exception=str(error),
        )
        recorder.write_summary(
            status="failed",
            configuration=configuration,
            failure=failure,
        )
        raise

    recorder.write_summary(
        status="completed",
        configuration=configuration,
    )
    print("LAYOUT_PAGE_FAILURE_PROBE: PASS", flush=True)


if __name__ == "__main__":
    main()
