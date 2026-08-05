#!/usr/bin/env python3
"""Run the official OpenDoc page pipeline with an explicit UniRec checkpoint.

OpenOCR's documented OmniDocBench command hard-codes a repository-relative
``unirec-0.1b/model.pth`` path and only selects CUDA or CPU.  This adapter keeps
the official OpenDoc pipeline intact while making the checkpoint and UniRec
device explicit.  Layout detection, crop construction, recognition routing,
page assembly, and result serialization still execute in OpenOCR.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official OpenDoc full-page inference adapter"
    )
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recognizer-device", default="cpu")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-visualization", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def patch_unirec_config(infer_doc: Any, model_path: Path) -> None:
    original_config = infer_doc.Config
    model_dir = model_path.parent

    class ExplicitCheckpointConfig:
        def __init__(self, config_path: str | Path) -> None:
            loaded = original_config(config_path)
            self.cfg = loaded.cfg
            global_cfg = self.cfg.get("Global", {})
            if global_cfg.get("use_transformers"):
                global_cfg["pretrained_model"] = str(model_path)
                global_cfg["vlm_ocr_config"] = str(model_dir)
                postprocess = self.cfg.get("PostProcess", {})
                if "tokenizer_path" in postprocess:
                    postprocess["tokenizer_path"] = str(model_dir)

    infer_doc.Config = ExplicitCheckpointConfig


def move_recognizer(pipeline: Any, device_name: str) -> None:
    if device_name == "cpu":
        return

    import torch

    if device_name.startswith("npu"):
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)

    device = torch.device(device_name)
    recognizer = pipeline.vl_rec_model
    recognizer.model.to(device)
    recognizer.device = device


def main() -> None:
    args = parse_args()
    openocr_root = args.openocr_root.resolve()
    model_path = args.model_path.resolve()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not (openocr_root / "tools" / "infer_doc.py").is_file():
        raise FileNotFoundError(f"OpenOCR source checkout not found: {openocr_root}")
    if not model_path.is_file():
        raise FileNotFoundError(f"UniRec checkpoint not found: {model_path}")

    sys.path.insert(0, str(openocr_root))
    from tools import infer_doc
    from tools.utils.utility import get_image_file_list

    patch_unirec_config(infer_doc, model_path)

    image_paths = sorted(get_image_file_list(str(input_path)))
    image_paths = image_paths[args.offset :]
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise ValueError(f"No input images found under {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("OPEN_DOC_SETUP_BEGIN", flush=True)
    pipeline = infer_doc.OpenDoc(gpuId=-1)
    move_recognizer(pipeline, args.recognizer_device)
    setup_s = time.perf_counter() - started
    print(f"OPEN_DOC_SETUP_END setup_s={setup_s:.3f}", flush=True)

    page_records: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        page_started = time.perf_counter()
        print(
            f"OPEN_DOC_PAGE_BEGIN index={index}/{len(image_paths)} "
            f"image={Path(image_path).name}",
            flush=True,
        )
        results = list(
            pipeline.predict(
                image_path,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        )
        if len(results) != 1:
            raise RuntimeError(
                f"Expected one page result for {image_path}, got {len(results)}"
            )
        result = results[0]
        result.save_to_json(save_path=str(output_dir))
        result.save_to_markdown(save_path=str(output_dir), pretty=args.pretty)
        if args.save_visualization:
            result.save_to_img(str(output_dir))
        page_s = time.perf_counter() - page_started
        page_records.append({"image": image_path, "wall_s": page_s})
        print(
            f"OPEN_DOC_PAGE_END index={index}/{len(image_paths)} "
            f"wall_s={page_s:.3f}",
            flush=True,
        )

    run = {
        "status": "ok",
        "openocr_root": str(openocr_root),
        "model_path": str(model_path),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "recognizer_device": args.recognizer_device,
        "offset": args.offset,
        "limit": args.limit,
        "setup_s": setup_s,
        "pipeline_wall_s": sum(record["wall_s"] for record in page_records),
        "pages": page_records,
    }
    summary_path = output_dir / "opendoc_run_summary.json"
    summary_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(f"OPEN_DOC_RUN_END summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
