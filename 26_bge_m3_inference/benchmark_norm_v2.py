"""Compare upstream V2 with original full-W8A8 BGE on identical weights/scales."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from fused_norm_v2 import FusedBGEM3, initialize, norm_quant


def errors(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    delta = (a-b).abs()
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "exact_fraction": float((delta == 0).float().mean()),
            "min_cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=-1).min())}


class NormBlock(torch.nn.Module):
    def __init__(self, norm, scale, fused):
        super().__init__()
        self.norm, self.fused = norm, fused
        self.register_buffer("scale", scale)
        self.register_buffer("multiply", scale.reciprocal().half())

    def forward(self, projection, residual):
        if self.fused:
            return norm_quant(projection, residual, self.norm.weight.reshape(1, -1),
                              self.norm.bias.reshape(1, -1), self.multiply, None, self.norm.eps)
        import torch_npu
        y = self.norm(projection + residual)
        return y, torch_npu.npu_quantize(y, scales=self.scale, zero_points=None,
                                         dtype=torch.qint8, axis=0, div_mode=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op-api", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-dir", default="/workspace/model_downloads/bge-m3")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    initialize(args.op_api)
    import torch_npu
    from benchmark_w8a8 import compiled_entrypoint, metrics
    from download_model import verify
    from profile_w8a8 import capture
    from run_embedder import Runner
    from validate_910b import CASES
    from w8a8 import calibrate, convert
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    assert "910B" in torch.npu.get_device_name(0).upper()
    root = Path(__file__).parent
    release = json.loads((root / "release.json").read_text())
    assert all(verify(Path(args.model_dir) / n, v) for n, v in release["files"].items())
    texts = json.loads((root / "quantization_texts.json").read_text())
    runner = Runner(args.model_dir, max_length=256)
    scales = calibrate(runner.model, [runner.tokenize(texts["calibration"][i:i+4])
                                    for i in range(0, len(texts["calibration"]), 4)])
    runner.max_length = 128
    held_out = runner.tokenize(texts["held_out"])
    dense = runner.model(**held_out).cpu()
    convert(runner.model, scales, "full_w8a8")
    baseline = runner.model
    fused = FusedBGEM3(baseline).eval()  # Shares exact weights; baseline is unchanged.
    result = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "device": torch.npu.get_device_name(0), "physical_npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "library_sha256": hashlib.sha256(args.op_api.read_bytes()).hexdigest(),
              "checkpoint_revision": release["revision"], "checkpoint_hashes_verified": True,
              "scales": scales, "isolated": [], "cases": [], "passed": False}
    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    def log(label, row):
        print(label + " " + json.dumps(row), flush=True)
        save()

    # Capture real full-W8A8 activations at early, middle and final attention norms.
    captured, hooks = {}, []
    for i in (0, 11, 23):
        def hook(module, inputs, index=i):
            x, residual = inputs
            captured[index] = (module.dense(x).reshape(-1, 1024), residual.reshape(-1, 1024))
        hooks.append(baseline.encoder.layer[i].attention.output.register_forward_pre_hook(hook))
    sample = runner.tokenize(CASES[0]["texts"])
    baseline(**sample)
    for hook in hooks:
        hook.remove()
    for i, inputs in captured.items():
        layer = baseline.encoder.layer[i]
        blocks = {name: NormBlock(layer.attention.output.LayerNorm, layer.intermediate.dense.input_scale, name == "v2")
                  for name in ("original", "v2")}
        eager = {name: block(*inputs) for name, block in blocks.items()}
        y, q = eager["original"]
        reciprocal_q = torch_npu.npu_quantize(y, scales=blocks["v2"].multiply, zero_points=None,
                                             dtype=torch.qint8, axis=0, div_mode=False)
        row = {"layer": i, "shape": list(inputs[0].shape),
               "norm_v2_vs_original": errors(eager["v2"][0], y),
               "int8_v2_vs_original": errors(eager["v2"][1], q),
               "int8_scale_only": errors(reciprocal_q, q),
               "int8_v2_vs_reciprocal_control": errors(eager["v2"][1], reciprocal_q),
               "compiled_vs_eager": {}, "profiles": {}}
        for name, block in blocks.items():
            fn = compiled_entrypoint(block, f"block_{i}_{name}", args.output / "cache")
            cy, cq = fn(*inputs)
            # Isolated V2 GE must reproduce the direct ACLNN call on identical inputs.
            row["compiled_vs_eager"][name] = {"norm": errors(cy, eager[name][0]), "quant": errors(cq, eager[name][1])}
            if name == "v2":
                torch.testing.assert_close(cy, eager[name][0], atol=0.002, rtol=0.002)
                assert row["compiled_vs_eager"][name]["quant"]["max_abs"] <= 1
            if i == 0:
                row["profiles"][name] = capture(lambda: fn(*inputs), args.output / "profiles" / f"block_{name}", f"block_{name}", steps=10)
        result["isolated"].append(row)
        log("ISOLATED", row)

    eager_base, eager_fused = baseline(**held_out).cpu(), fused(**held_out).cpu()
    result["held_out"] = {"v2_vs_original": metrics(eager_fused, eager_base),
                           "original_vs_dense": metrics(eager_base, dense), "v2_vs_dense": metrics(eager_fused, dense)}
    assert all(r["finite"] for r in result["held_out"].values())
    log("HELD_OUT", result["held_out"])
    cases = CASES + [{"name": "batch4_full512", "length": 512, "texts": [t*100 for t in texts["held_out"][:4]]}]
    for case in cases:
        runner.max_length = case["length"]
        tokens = runner.tokenize(case["texts"])
        row = {"name": case["name"], "shape": list(tokens["input_ids"].shape), "profiles": {}, "compiled_vs_eager": {}}
        functions, outputs = {}, {}
        for name, model in (("original", baseline), ("v2", fused)):
            eager = model(**tokens).cpu()
            fn = compiled_entrypoint(model, case["name"] + "_" + name, args.output / "cache")
            functions[name] = fn
            outputs[name] = fn(**tokens).cpu()
            row["compiled_vs_eager"][name] = metrics(outputs[name], eager)
            assert row["compiled_vs_eager"][name]["finite"]
            assert float((outputs[name].float().norm(dim=-1)-1).abs().max()) < .005
            log("COMPILED", {"case": case["name"], "lane": name, **row["compiled_vs_eager"][name]})
        row["v2_vs_original"] = metrics(outputs["v2"], outputs["original"])
        if case["name"] == "mixed_multilingual":
            row["semantic"] = {}
            queries = runner.tokenize([texts["held_out"][0], texts["held_out"][2]])
            documents = runner.tokenize([texts["held_out"][1], texts["held_out"][3]])
            for name, fn in functions.items():
                scores = fn(**queries).float().cpu() @ fn(**documents).float().cpu().T
                row["semantic"][name] = {"scores": scores.tolist(), "top_document": scores.argmax(dim=1).tolist()}
        # Two blocks in opposite order: direct kernel evidence, not ctypes/host timing.
        for repeat, order in enumerate((("original", "v2"), ("v2", "original"))):
            for name in order:
                label = f"{case['name']}_{name}_{repeat}"
                row["profiles"][label] = capture(lambda: functions[name](**tokens), args.output / "profiles" / label, label, steps=10)
        result["cases"].append(row)
        log("CASE", row)
    result["passed"] = True
    log("DONE", {"output": str(args.output)})


if __name__ == "__main__":
    main()
