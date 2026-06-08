#!/usr/bin/env python3
"""Run PaddleOCR-VL-1.6 recognition on one crop with Transformers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
DEFAULT_CROP = ROOT / "crops" / "crop_01_text_block_en.png"
MANIFEST = ROOT / "crops" / "manifest.json"


def default_prompt_for(crop: Path) -> str:
    if not MANIFEST.exists():
        return "OCR:"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in manifest:
        if item.get("file") == crop.name:
            return item.get("suggested_prompt") or "OCR:"
    return "OCR:"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--crop", type=Path, default=DEFAULT_CROP)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    crop = args.crop.expanduser().resolve()
    prompt = args.prompt or default_prompt_for(crop)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    image = Image.open(crop).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.model)
    model = (
        AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype)
        .to(device)
        .eval()
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    text = processor.decode(generated, skip_special_tokens=True)
    print(text.strip())


if __name__ == "__main__":
    main()
