"""No-checkpoint gate: direct V2 -> compiled V2 -> INT8 consumers, then profile."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from fused_norm_v2 import FusedBGEM3, initialize, norm_quant


class Probe(torch.nn.Module):
    def __init__(self, width, out_features, bias):
        super().__init__()
        from w8a8 import W8A8Linear
        self.norm = torch.nn.LayerNorm(width, device="npu", dtype=torch.float16)
        self.linear = W8A8Linear(torch.nn.Linear(width, out_features, device="npu", dtype=torch.float16), 1/16)
        self.register_buffer("scale", torch.tensor([16.], device="npu", dtype=torch.float16))
        self.register_buffer("bias", torch.randn(1, width, device="npu", dtype=torch.float16)*.1 if bias else None)

    def forward(self, x, residual):
        y, q = norm_quant(x, residual, self.norm.weight.reshape(1, -1),
                          self.norm.bias.reshape(1, -1), self.scale, self.bias, self.norm.eps)
        # Exercise both the already-INT8 consumer and FusedBGEM3's projection layout.
        projected = FusedBGEM3.project(self.linear, y)
        return y, q, self.linear.matmul(q), projected


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op-api", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-chip", choices=("910B", "310P"), required=True)
    p.add_argument("--rows", type=int, default=256)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    initialize(args.op_api)
    from benchmark_w8a8 import compiled_entrypoint
    from benchmark_norm_v2 import errors
    from profile_w8a8 import capture
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    chip = torch.npu.get_device_name(0)
    if args.expected_chip not in chip.upper():
        raise RuntimeError(f"Expected {args.expected_chip}, got {chip}")
    if args.expected_chip == "310P":
        torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(17)
    x, residual = [torch.randn(args.rows, 1024).half().npu() for _ in range(2)]
    result = {"device": chip, "physical_npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "passed": False, "cases": []}
    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    save()
    for out_features in (1024, 4096):
        for bias in (False, True):
            label = f"m{args.rows}_n{out_features}_bias{int(bias)}"
            block = Probe(1024, out_features, bias).eval()
            eager = block(x, residual)
            # Both paths must agree on weight orientation before compiling.
            torch.testing.assert_close(eager[3] + block.linear.bias, block.linear(eager[0]), atol=0, rtol=0)
            summed = x.float().cpu() + residual.float().cpu()
            if bias:
                summed += block.bias.float().cpu()
            reference = torch.nn.functional.layer_norm(summed, (1024,), eps=1e-5)
            torch.testing.assert_close(eager[0].float().cpu(), reference, atol=.004, rtol=.002)
            qref = (reference * 16).round().clamp(-128, 127)
            assert errors(eager[1], qref)["max_abs"] <= 1
            fn = compiled_entrypoint(block, label, args.output / "cache")
            compiled = fn(x, residual)
            comparison = [errors(a, b) for a, b in zip(compiled, eager)]
            for index in (0, 2, 3):
                torch.testing.assert_close(compiled[index], eager[index], atol=.004, rtol=.002)
            assert comparison[1]["max_abs"] <= 1
            row = {"name": label, "graph_transpose": block.linear.graph_transpose,
                   "compiled_vs_eager": comparison,
                   "profiles": {"compiled": capture(lambda: fn(x, residual), args.output / "profiles" / label, label, steps=10)}}
            result["cases"].append(row)
            save()
            print("PROBE " + json.dumps(row), flush=True)
    result["passed"] = True
    save()
    print("PASSED", flush=True)


if __name__ == "__main__":
    main()
