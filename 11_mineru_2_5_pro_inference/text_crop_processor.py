"""Immutable per-label processor selection for streaming recognition crops."""
import copy


def make_text_crop_processor(processor, max_pixels):
    if max_pixels is None:
        return None
    from run_official_transformers_omnidocbench import apply_processor_pixel_limits
    capped = copy.copy(processor)
    capped.image_processor = copy.deepcopy(processor.image_processor)
    apply_processor_pixel_limits(capped.image_processor, max_pixels=max_pixels)
    return capped


def select_crop_processor(processor, text_processor, block_type):
    # Layout labels are authoritative; prompts also cover titles and headers.
    return text_processor if block_type == "text" and text_processor is not None else processor
