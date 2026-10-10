"""Minimal text -> normalized dense BGE-M3 embeddings on Ascend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modeling_bge_m3 import Config, load_model


class Runner:
    def __init__(self, model_dir: str, *, device: str = "npu:0", max_length: int = 128,
                 compile_cache: str | None = None, weight_mode: str = "dense",
                 calibration_file: str | None = None):
        import torch_npu  # noqa: F401
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("This runner requires an Ascend NPU")
        torch.npu.set_device(self.device)
        if weight_mode not in {"dense", "ffn_w8a8", "full_w8a8"}:
            raise ValueError(f"Unknown weight mode: {weight_mode}")
        if weight_mode != "dense":
            torch.npu.config.allow_internal_format = True
        config = Config.from_directory(model_dir)
        if not 2 <= max_length <= config.max_position_embeddings - config.pad_token_id - 1:
            raise ValueError("max_length must be within the checkpoint position capacity")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="right")
        self.model = load_model(model_dir, self.device)
        self.quantization = None
        if weight_mode != "dense":
            from w8a8 import calibrate, convert, prepare_dense_weights
            path = Path(calibration_file) if calibration_file else Path(__file__).with_name("quantization_texts.json")
            texts = json.loads(path.read_text())["calibration"]
            self.max_length = min(256, config.max_position_embeddings - config.pad_token_id - 1)
            batches = [self.tokenize(texts[i:i+4]) for i in range(0, len(texts), 4)]
            scales = calibrate(self.model, batches)
            self.max_length = max_length
            self.quantization = convert(self.model, scales, weight_mode)
            self.quantization["dense_nz_linears"] = prepare_dense_weights(self.model)
            self.model.eval()
        if compile_cache:
            from torchair.inference import cache_compile
            from torchair.configs.compiler_config import CompilerConfig
            Path(compile_cache).mkdir(parents=True, exist_ok=True)
            self.model.forward = cache_compile(self.model.forward, config=CompilerConfig(),
                                               cache_dir=str(Path(compile_cache).resolve()),
                                               fullgraph=True, dynamic=False, ge_cache=True)

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        if not texts:
            raise ValueError("At least one text is required")
        tokens = self.tokenizer(texts, padding="max_length", truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        return {key: tokens[key].to(self.device) for key in ("input_ids", "attention_mask")}

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> torch.Tensor:
        return self.model(**self.tokenize(texts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--texts", nargs="+", default=["What is BGE M3?", "BGE M3 is a multilingual embedding model."])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--compile-cache")
    parser.add_argument("--weight-mode", choices=("dense", "ffn_w8a8", "full_w8a8"), default="dense")
    parser.add_argument("--calibration-file")
    args = parser.parse_args()
    runner = Runner(args.model_dir, device=args.device, max_length=args.max_length, compile_cache=args.compile_cache,
                    weight_mode=args.weight_mode, calibration_file=args.calibration_file)
    if runner.quantization:
        print({"quantization": runner.quantization})
    result = runner.encode(args.texts).float().cpu()
    print({"shape": list(result.shape), "norms": result.norm(dim=-1).tolist(),
           "similarities": (result @ result.T).tolist()})


if __name__ == "__main__":
    main()
