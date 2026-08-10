from __future__ import annotations

import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from pipeline.phone_text_frontend import PhoneText20Frontend


def test_phone_text_frontend_extracts_all_100_pages() -> None:
    bundle = (
        REPO_ROOT
        / "tmp/09_persistent_page_engine/phone_text_synthetic_100_20260810"
    )
    manifest = json.loads((bundle / "manifest.json").read_text())
    frontend = PhoneText20Frontend()

    for ordinal, sample in enumerate(manifest["samples"]):
        prepared = frontend.prepare_detected_page(
            frontend.detect_transferred_page(
                frontend.transfer_preprocessed_page(
                    frontend.preprocess_decoded_page(
                        frontend.decode_page(bundle / sample["file"], ordinal)
                    )
                )
            ),
            min_pixels=28_224,
            max_pixels=802_816,
            text_crop_scale=1.0,
        )

        assert len(prepared.blocks) == 20
        assert len(prepared.requests) == 20
        assert prepared.statistics["use_layout_detection"] is False
        assert all(block["label"] == "text" for block in prepared.blocks)
        assert min(block["box"][1] for block in prepared.blocks) >= 190

        for expected, block, request in zip(
            sample["lines"], prepared.blocks, prepared.requests
        ):
            expected_x0, expected_y0, expected_x1, expected_y1 = (
                expected["bbox_xyxy"]
            )
            actual_x0, actual_y0, actual_x1, actual_y1 = block["box"]
            assert actual_x0 <= expected_x0 <= expected_x1 <= actual_x1
            assert actual_y0 <= expected_y0 <= expected_y1 <= actual_y1
            assert request.request_id.endswith(
                f"_line_{expected['line_index']:02d}"
            )
