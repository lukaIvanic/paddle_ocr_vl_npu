"""Isolate eager/compiled quantization semantics on identical real-layer inputs."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from torchair.configs.compiler_config import CompilerConfig
from torchair.inference import cache_compile

from run_embedder import Runner
from w8a8 import W8A8Linear, calibrate


class Quantizer(torch.nn.Module):
    def __init__(self, scale, divide):
        super().__init__()
        self.register_buffer("scale", scale if divide else scale.reciprocal())
        self.divide = divide

    def forward(self, x):
        return torch_npu.npu_quantize(x, scales=self.scale, zero_points=None,
                                     dtype=torch.qint8, axis=0, div_mode=self.divide)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    runner = Runner("/workspace/model_downloads/bge-m3", max_length=256)
    texts = json.loads(Path(__file__).with_name("quantization_texts.json").read_text())
    scales = calibrate(runner.model, [runner.tokenize(texts["calibration"][i:i+4]) for i in range(0,12,4)])
    runner.max_length = 128
    tokens = runner.tokenize(["What is BGE M3?", "BGE M3 是一个支持多语言检索的嵌入模型。 Warszawa jest stolicą Polski."])
    captured = {}
    layer = runner.model.encoder.layer[0].intermediate.dense
    hook = layer.register_forward_pre_hook(lambda _m, inputs: captured.update(x=inputs[0].clone()))
    runner.model(**tokens)
    hook.remove()
    quant = W8A8Linear(layer, scales["encoder.layer.0.intermediate.dense"])
    x = captured["x"].reshape(-1, layer.in_features)
    report = {}
    for key, module in [("divide_quantizer", Quantizer(quant.input_scale, True)),
                        ("biased_linear", quant)]:
        torch._dynamo.reset()
        eager = module(x)
        compiled = cache_compile(module.forward, config=CompilerConfig(), dynamic=False, fullgraph=True,
                                 ge_cache=True, cache_dir=str(args.output.parent / key))
        actual = compiled(x)
        torch.npu.synchronize()
        a, b = eager.float().cpu(), actual.float().cpu()
        row = {"max_abs": float((a-b).abs().max()), "different_fraction": float((a!=b).float().mean()),
               "mean_abs": float((a-b).abs().mean()), "eager_range": [float(a.min()),float(a.max())]}
        report[key] = row
        args.output.write_text(json.dumps(report, indent=2))
        print(key, row, flush=True)


if __name__ == "__main__":
    main()
