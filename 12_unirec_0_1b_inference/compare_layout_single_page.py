#!/usr/bin/env python3
"""Compare official ONNX and local NPU PP-DocLayoutV2 on one page."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

from opendoc_layout_npu import PPDocLayoutV2NpuAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--transformers-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--execution", choices=("eager", "torchair"), default="torchair")
    parser.add_argument("--compile-cache-dir", type=Path)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.2, 0.4, 0.43, 0.5, 0.7),
    )
    return parser.parse_args()


def compact_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": box["label"],
            "score": float(box["score"]),
            "coordinate": [float(value) for value in box["coordinate"]],
            "order": float(box.get("custom_value", 0.0)),
        }
        for box in boxes
    ]


def run_onnx(
    *,
    image: Any,
    openocr_root: Path,
    model_path: Path,
    thresholds: list[float],
) -> dict[str, Any]:
    sys.path.insert(0, str(openocr_root))
    from tools.infer_doc_onnx import LayoutDetectorONNX

    detector = LayoutDetectorONNX(
        str(model_path),
        use_gpu=False,
        threshold=min(thresholds),
        auto_download=False,
    )
    inputs, scale, height, width = detector.preprocess(image)
    outputs = detector.session.run(detector.output_names, inputs)
    raw_rows = outputs[0]
    raw_candidates = []
    for row in raw_rows:
        class_id = int(row[0])
        raw_candidates.append(
            {
                "class_id": class_id,
                "label": detector.label_map.get(class_id, f"class_{class_id}"),
                "score": float(row[1]),
                "coordinate": [float(value) for value in row[2:6]],
                "order": float(row[6]),
            }
        )
    final = {}
    for threshold in thresholds:
        detector.threshold = threshold
        result = detector.postprocess(
            image,
            outputs,
            scale,
            height,
            width,
        )
        final[str(threshold)] = compact_boxes(result["boxes"])
    return {
        "raw_output_shape": list(raw_rows.shape),
        "raw_candidates": raw_candidates,
        "thresholds": final,
    }


def run_npu(
    *,
    image: Any,
    model_path: Path,
    device: str,
    dtype: str,
    execution: str,
    compile_cache_dir: Path | None,
    thresholds: list[float],
) -> dict[str, Any]:
    detector = PPDocLayoutV2NpuAdapter(
        model_path=model_path,
        device=device,
        dtype=dtype,
        threshold=min(thresholds),
        execution=execution,
        compile_cache_dir=compile_cache_dir,
        batch_size=1,
        weight_format="native",
        depthwise_rewrite="native",
        input_color_order="bgr",
    )
    original_filter = detector._filter_overlap_boxes
    detector._filter_overlap_boxes = lambda result: result
    pre_overlap = detector([image], threshold=min(thresholds))[0]
    detector._filter_overlap_boxes = original_filter

    final = {}
    for threshold in thresholds:
        result = detector([image], threshold=threshold)[0]
        final[str(threshold)] = compact_boxes(result["boxes"])
    return {
        "pre_overlap_at_min_threshold": compact_boxes(pre_overlap["boxes"]),
        "thresholds": final,
        "setup_s": detector.setup_s,
    }


def print_summary(name: str, result: dict[str, Any]) -> None:
    print(f"LAYOUT_COMPARE backend={name}", flush=True)
    for threshold, boxes in result["thresholds"].items():
        labels = ",".join(
            f"{box['label']}:{box['score']:.3f}" for box in boxes
        )
        print(
            f"LAYOUT_COMPARE threshold={threshold} boxes={len(boxes)} "
            f"labels={labels}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.execution == "torchair" and args.compile_cache_dir is None:
        raise ValueError("--compile-cache-dir is required for TorchAir execution")
    thresholds = sorted(set(args.thresholds))
    image = cv2.imread(str(args.input.expanduser().resolve()), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.input)

    onnx = run_onnx(
        image=image,
        openocr_root=args.openocr_root.expanduser().resolve(),
        model_path=args.onnx_model.expanduser().resolve(),
        thresholds=thresholds,
    )
    print_summary("onnx_cpu", onnx)

    npu = run_npu(
        image=image,
        model_path=args.transformers_model.expanduser().resolve(),
        device=args.device,
        dtype=args.dtype,
        execution=args.execution,
        compile_cache_dir=(
            args.compile_cache_dir.expanduser().resolve()
            if args.compile_cache_dir is not None
            else None
        ),
        thresholds=thresholds,
    )
    print_summary(f"npu_{args.execution}_{args.dtype}", npu)

    report = {
        "input": str(args.input.expanduser().resolve()),
        "thresholds": thresholds,
        "onnx_cpu": onnx,
        "npu": {
            "execution": args.execution,
            "dtype": args.dtype,
            **npu,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"LAYOUT_COMPARE output={output}", flush=True)


if __name__ == "__main__":
    main()
