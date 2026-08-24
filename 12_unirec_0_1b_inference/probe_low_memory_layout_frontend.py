#!/usr/bin/env python3
"""Gate shared layout and Torch-free W4/T8 crop preparation against production."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TE_PARALLEL_COMPILER", "1")
os.environ.setdefault("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "0")
os.environ.setdefault("UNIREC_DEINIT_TBE_AFTER_WARMUP", "1")

import numpy as np
import torch_npu

from host_memory_diagnostics import process_snapshot
from layout_page_input import decode_page_rgb, materialize_layout_rgb
from low_memory_frontend_pool import CpuCropPreparePool
from shared_layout_owner import SharedLayoutOwner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--layout-model", type=Path, required=True)
    parser.add_argument("--layout-cache", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> None:
    args = parse_args()
    visible = {
        int(value)
        for value in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    }
    if visible.intersection({5, 6}):
        raise RuntimeError("physical NPU 5 and NPU 6 are excluded")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    sys.path.insert(0, str(args.openocr_root.resolve()))
    from tools.utils.utility import get_image_file_list

    paths = [
        Path(value).resolve()
        for value in sorted(get_image_file_list(str(args.input.resolve())))
    ][: args.limit]
    if not paths:
        raise ValueError("no input pages")
    if args.spool_dir.exists() and any(args.spool_dir.iterdir()):
        raise RuntimeError(f"spool directory is not empty: {args.spool_dir}")
    args.spool_dir.mkdir(parents=True, exist_ok=True)

    owner = SharedLayoutOwner(
        model_path=args.layout_model,
        cache_dir=args.layout_cache,
        device=args.device,
        lanes=4,
        batch_size=2,
        threshold=0.5,
    )
    pages = []
    decode_started = time.perf_counter()
    for path in paths:
        rgb, _timing = decode_page_rgb(path)
        pages.append(materialize_layout_rgb(rgb))
    decode_s = time.perf_counter() - decode_started
    canonical = owner.adapter(pages, threshold=0.5)
    candidate = owner.predict(pages)
    layout_exact = canonical == candidate
    if not layout_exact:
        raise RuntimeError("shared layout output differs from canonical adapter")

    pool = CpuCropPreparePool(
        workers=args.workers,
        recognition_threads=args.threads,
        openocr_root=args.openocr_root,
        spool_root=args.spool_dir,
        cross_cache_length=1320,
    )
    for page_index, (path, rgb, layout_result) in enumerate(
        zip(paths, pages, candidate)
    ):
        pool.submit(
            page_index=page_index,
            path=path,
            rgb=rgb,
            layout_result=layout_result,
            started_at=time.perf_counter(),
        )
    payloads, frontend = pool.finish()
    pool.close()
    payload_digest = digest(
        [
            {
                "layout": payload["layout_results"],
                "blocks": payload["blocks"],
                "vlm_block_ids": payload["vlm_block_ids"],
                "crops": [
                    {
                        "crop_index": crop["crop_index"],
                        "label": crop["label"],
                        "figure_token_map": crop["figure_token_map"],
                        "source_image_size": crop["source_image_size"],
                        "processed_image_size": crop["processed_image_size"],
                    }
                    for crop in payload["crops"]
                ],
            }
            for payload in payloads
        ]
    )
    invalid_spools = []
    for payload in payloads:
        for crop in payload["crops"]:
            descriptor = crop["processed_pixel_values_descriptor"]
            array = np.memmap(
                descriptor["path"],
                mode="r+",
                dtype=np.uint8,
                offset=int(descriptor["offset"]),
                shape=tuple(descriptor["shape"]),
            )
            if not array.flags.writeable or not array.flags.c_contiguous:
                invalid_spools.append(descriptor)
            del array
    release = owner.release()
    report = {
        "status": "pass" if layout_exact and not invalid_spools else "fail",
        "chip": torch_npu.npu.get_device_name(0),
        "pages": len(paths),
        "layout_exact": layout_exact,
        "payload_digest": payload_digest,
        "decode_s": decode_s,
        "layout": release,
        "frontend": frontend.__dict__,
        "invalid_spools": invalid_spools,
        "final_memory": process_snapshot(),
    }
    print("UNIREC_LOW_MEMORY_FRONTEND " + json.dumps(report, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
