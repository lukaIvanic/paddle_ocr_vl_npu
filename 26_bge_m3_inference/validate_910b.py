"""Real-checkpoint eager and static TorchAir parity against Transformers on 910B."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoModel

from run_embedder import Runner
from download_model import verify

CASES = [
    {"name": "mixed_multilingual", "length": 128,
     "texts": ["What is BGE M3?", "BGE M3 是一个支持多语言检索的嵌入模型。 Warszawa jest stolicą Polski."]},
    {"name": "long_truncated", "length": 512,
     "texts": ["This document discusses multilingual information retrieval and neural embeddings. " * 100]},
]


def timed(fn):
    torch.npu.synchronize()
    start = time.perf_counter()
    output = fn()
    torch.npu.synchronize()
    return output, (time.perf_counter() - start) * 1000


def benchmark(fn):
    first, cold = timed(fn)
    for _ in range(2):
        fn()
    samples = [timed(fn)[1] for _ in range(5)]
    return first, {"first_call_ms": cold, "median_ms": statistics.median(samples), "samples_ms": samples}


def compare(actual, expected):
    a, b = actual.float(), expected.float()
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    maximum = float((a - b).abs().max())
    minimum = float(cosine.min())
    finite = bool(torch.isfinite(a).all())
    return {"max_abs": maximum, "min_cosine": minimum, "finite": finite,
            "passed": finite and maximum <= 0.002 and minimum >= 0.9999}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compile-cache", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    chip = torch.npu.get_device_name(0)
    if "910B" not in chip.upper():
        raise RuntimeError(f"This validation requires 910B; found {chip}")
    root = Path(__file__).resolve().parent
    result = {"device": chip, "host": socket.gethostname(),
              "physical_npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "model_dir": str(Path(args.model_dir).resolve()),
              "checkpoint_revision": "5617a9f61b028005a4858fdac845db406aefb181",
              "model_source_sha256": hashlib.sha256((root / "modeling_bge_m3.py").read_bytes()).hexdigest(),
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "transformers")},
              "dtype": "float16", "warmups": 2, "repeats": 5, "cases": [], "passed": False}
    release = json.loads((root / "release.json").read_text())
    for name, entry in release["files"].items():
        if not verify(Path(args.model_dir) / name, entry):
            raise RuntimeError(f"Checkpoint file does not match pinned release: {name}")
    result["checkpoint_hashes_verified"] = True
    print("ENVIRONMENT " + json.dumps(result), flush=True)
    reference = AutoModel.from_pretrained(args.model_dir, local_files_only=True,
                                         dtype=torch.float16, attn_implementation="eager",
                                         add_pooling_layer=False).to("npu:0").eval()
    runner = Runner(args.model_dir, max_length=128)
    with torch.inference_mode():
        for case in CASES:
            runner.max_length = case["length"]
            tokens = runner.tokenize(case["texts"])
            reference_hidden = reference(**tokens).last_hidden_state
            reference_embedding = torch.nn.functional.normalize(reference_hidden[:, 0], dim=-1).cpu()
            local_hidden = runner.model.forward_hidden_states(**tokens)
            mask = tokens["attention_mask"].bool()
            hidden_error = float((local_hidden[mask].float() - reference_hidden[mask].float()).abs().max().cpu())
            eager, eager_timing = benchmark(lambda: runner.model(**tokens))
            eager_parity = compare(eager.cpu(), reference_embedding)
            print("EAGER " + json.dumps({"case": case["name"], "parity": eager_parity,
                  "valid_hidden_max_abs": hidden_error, "timing": eager_timing}), flush=True)
            if not eager_parity["passed"] or hidden_error > 0.05:
                raise RuntimeError("Eager reference parity failed before compilation")
            from torchair.configs.compiler_config import CompilerConfig
            from torchair.inference import cache_compile
            compiled = cache_compile(runner.model.forward, config=CompilerConfig(), dynamic=False,
                                     fullgraph=True, ge_cache=True,
                                     cache_dir=str(Path(args.compile_cache).resolve() / case["name"]))
            compiled_output, compiled_timing = benchmark(lambda: compiled(**tokens))
            row = {"name": case["name"], "batch_size": len(case["texts"]), "sequence_length": case["length"],
                   "valid_tokens": tokens["attention_mask"].sum(dim=1).cpu().tolist(),
                   "eager_vs_reference": compare(eager.cpu(), reference_embedding),
                   "compiled_vs_reference": compare(compiled_output.cpu(), reference_embedding),
                   "compiled_vs_eager": compare(compiled_output.cpu(), eager.cpu()),
                   "valid_hidden_max_abs_eager_vs_reference": hidden_error,
                   "eager": eager_timing, "compiled": compiled_timing,
                   "embedding_shape": list(eager.shape),
                   "compiled_norm_max_error": float((compiled_output.float().norm(dim=-1) - 1).abs().max().cpu())}
            row["passed"] = (all(row[k]["passed"] for k in
                                 ("eager_vs_reference", "compiled_vs_reference", "compiled_vs_eager"))
                             and hidden_error <= 0.05 and row["compiled_norm_max_error"] <= 0.002)
            result["cases"].append(row)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print("CASE " + json.dumps(row), flush=True)
    result["passed"] = all(row["passed"] for row in result["cases"])
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps({"passed": result["passed"], "output": str(output)}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
