from pathlib import Path

from PIL import Image

from paddleocr_vl.serving.types import RecognitionRequest, RecognitionResult
from pipeline.layout_frontend import PreparedLayoutPage
from pipeline.persistent_page_service import (
    PageSubmission,
    _PageState,
    _recognition_metrics,
)


def test_page_recognition_metrics_preserve_crop_tokens_and_stage_totals() -> None:
    request = RecognitionRequest(
        request_id="crop-0",
        crop=Image.new("RGB", (64, 16)),
        prompt="OCR:",
        source_crop_size=(128, 32),
    )
    prepared = PreparedLayoutPage(
        ordinal=0,
        image_path=Path("page.png"),
        image_size=(1280, 1920),
        blocks=[{"label": "text"}],
        requests=[request],
        request_block_indices=[0],
        figure_token_maps={},
        dropped_figure_paths=set(),
        document_images=[],
        timing_s={"page_total_s": 0.2},
        statistics={},
    )
    result = RecognitionResult(
        request_id="crop-0",
        decode_schedule_id="schedule",
        decode_slot_index=0,
        decode_slot_epoch=1,
        prompt="OCR:",
        crop_size=(64, 16),
        text="hello",
        token_ids=[1, 2, 3],
        stop_reason="eos",
        input_tokens=40,
        projected_image_tokens=28,
        generated_tokens_including_eos=3,
        decode_tokens_after_prefill_including_eos=2,
        decode_calls_executed=2,
        timing_s={"request_total": 0.1},
        device_stage_s={"vision_prefill": 0.02, "text_prefill": 0.01},
        rates={},
        vision={
            "execution": "compiled",
            "bucket": 256,
            "real_vision_tokens": 112,
            "physical_vision_tokens": 256,
        },
        text_prefill={
            "execution": "compiled",
            "bucket": 128,
            "real_text_tokens": 40,
            "physical_text_tokens": 128,
        },
    )
    page = _PageState(
        submission=PageSubmission("page", Path("page.png"), 0.0),
        prepared=prepared,
        remaining=0,
        recognition={0: "hello"},
        recognition_results={0: result},
    )

    metrics = _recognition_metrics(page)

    assert metrics["crops"] == 1
    assert metrics["token_totals"]["real_vision"] == 112
    assert metrics["token_totals"]["generated_including_eos"] == 3
    assert metrics["device_stage_s"]["vision_prefill"] == 0.02
    assert metrics["crop_details"][0]["source_crop_size"] == [128, 32]
    assert metrics["crop_details"][0]["model_crop_size"] == [64, 16]
    assert metrics["rates"]["useful_vision_tok_per_device_s"] == 5600.0
