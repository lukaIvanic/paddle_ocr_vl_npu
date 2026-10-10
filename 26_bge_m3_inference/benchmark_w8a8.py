"""Paired compiled FP16 / static W8A8 smoke; quality metrics are not retrieval evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import types

import torch
import torch_npu
from torchair.configs.compiler_config import CompilerConfig
from torchair.inference import cache_compile

from download_model import verify
from modeling_bge_m3 import load_model
from run_embedder import Runner
from validate_910b import CASES, timed
from w8a8 import W8A8Linear, calibrate, convert, prepare_dense_weights

MODES = ("dense", "ffn_w8a8", "full_w8a8")


def metrics(actual, expected):
    a, b = actual.float().cpu(), expected.float().cpu()
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return {"finite": bool(torch.isfinite(a).all()), "max_abs": float((a-b).abs().max()),
            "mean_cosine": float(cosine.mean()), "min_cosine": float(cosine.min()),
            "similarity_matrix_max_abs": float((a @ a.T - b @ b.T).abs().max())}


def compiled_entrypoint(model, key, cache):
    # Separate code objects prevent Dynamo guards/cache watchers crossing lanes.
    original = model.forward.__func__
    function = types.FunctionType(original.__code__.replace(co_name=key), original.__globals__, key)
    return cache_compile(types.MethodType(function, model), config=CompilerConfig(), dynamic=False,
                         fullgraph=True, ge_cache=True, cache_dir=str(cache / key))


def biased_linear_probe():
    torch.manual_seed(31)
    linear = torch.nn.Linear(64, 32, device="npu:0", dtype=torch.float16).eval()
    linear.bias.copy_(torch.linspace(-0.75, 0.75, 32, device="npu:0", dtype=torch.float16))
    x = torch.linspace(-2, 2, 8*64, device="npu:0", dtype=torch.float16).view(8, 64)
    quantized = W8A8Linear(linear, 0.02)
    xq = quantized.quantize(x).cpu().int()
    weight = linear.weight.float().cpu()
    scales = weight.abs().amax(dim=1).clamp_min(1e-6) / 127
    wq = (weight / scales[:, None]).round().clamp(-127, 127).int()
    expected = ((xq @ wq.T).float() * (scales * 0.02)).half() + linear.bias.cpu()
    actual = quantized(x).cpu()
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.001)
    return {"max_abs_vs_integer_reference": float((actual.float()-expected.float()).abs().max()), "passed": True}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/workspace/model_downloads/bge-m3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, help="Capture CPU/NPU kernels after each unprofiled timing case")
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    chip = torch.npu.get_device_name(0)
    if "910B" not in chip.upper():
        raise RuntimeError(f"Expected 910B, got {chip}")
    root = Path(__file__).resolve().parent
    release = json.loads((root / "release.json").read_text())
    for name, entry in release["files"].items():
        if not verify(Path(args.model_dir) / name, entry):
            raise RuntimeError(f"Checkpoint hash mismatch: {name}")
    texts_file = root / "quantization_texts.json"
    texts = json.loads(texts_file.read_text())
    result = {"device": chip, "physical_npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "transformers")},
              "checkpoint_revision": release["revision"], "checkpoint_hashes_verified": True,
              "texts_sha256": hashlib.sha256(texts_file.read_bytes()).hexdigest(),
              "warmups": 3, "paired_repeats": 20, "passed": False, "cases": [], "modes": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    result["biased_linear_probe"] = biased_linear_probe()
    print("LINEAR_PROBE " + json.dumps(result["biased_linear_probe"]), flush=True)
    runner = Runner(args.model_dir, max_length=256)
    batches = [runner.tokenize(texts["calibration"][i:i+4]) for i in range(0, len(texts["calibration"]), 4)]
    start = time.perf_counter()
    scales = calibrate(runner.model, batches)
    result["calibration"] = {"texts": len(texts["calibration"]), "batch_size": 4, "padded_length": 256,
                             "seconds": time.perf_counter()-start, "scales": scales,
                             "method": "FP16 input absmax over three disjoint calibration batches"}
    models = {"dense": runner.model}
    for mode in MODES:
        started = time.perf_counter()
        if mode != "dense":
            models[mode] = load_model(args.model_dir, runner.device)
            result["modes"][mode] = convert(models[mode], scales, mode)
        else:
            result["modes"][mode] = {"mode": mode, "quantized_linears": 0}
        result["modes"][mode]["dense_nz_linears"] = prepare_dense_weights(models[mode])
        models[mode].eval()
        torch.npu.synchronize()
        result["modes"][mode]["setup_seconds"] = time.perf_counter()-started
        print("MODE " + json.dumps(result["modes"][mode]), flush=True)
    runner.max_length = 128
    held_out = runner.tokenize(texts["held_out"])
    expected = models["dense"](**held_out).cpu()
    result["held_out_quality"] = {mode: metrics(models[mode](**held_out), expected) for mode in MODES}
    print("HELD_OUT " + json.dumps(result["held_out_quality"]), flush=True)
    if not all(row["finite"] for row in result["held_out_quality"].values()):
        raise RuntimeError("Nonfinite held-out embeddings")
    save()
    cases = CASES + [{"name": "batch4_full512", "length": 512,
                      "texts": [text * 100 for text in texts["held_out"][:4]]}]
    for case in cases:
        runner.max_length = case["length"]
        tokens = runner.tokenize(case["texts"])
        dense_reference = models["dense"](**tokens).cpu()
        row = {"name": case["name"], "batch_size": len(case["texts"]), "sequence_length": case["length"],
               "valid_tokens": tokens["attention_mask"].sum(dim=1).cpu().tolist(), "modes": {}}
        compiled = {}
        for mode in MODES:
            eager = models[mode](**tokens).cpu()
            key = case["name"] + "_" + mode
            compiled[mode] = compiled_entrypoint(models[mode], key, args.cache)
            output, first = timed(lambda: compiled[mode](**tokens))
            parity = metrics(output, eager)
            norm_error = float((output.float().norm(dim=-1) - 1).abs().max().cpu())
            if not parity["finite"] or norm_error > 0.005:
                raise RuntimeError(f"Nonfinite or unnormalized embeddings for {key}: {parity}")
            if mode == "dense" and (parity["max_abs"] > 0.002 or parity["min_cosine"] < 0.9999):
                raise RuntimeError(f"Dense compiled/eager mismatch for {key}: {parity}")
            row["modes"][mode] = {"first_call_ms": first, "compiled_vs_eager": parity,
                                   "vs_dense": metrics(output, dense_reference), "samples_ms": []}
            print("COMPILED " + json.dumps({"case": case["name"], "mode": mode, **row["modes"][mode]}), flush=True)
        if case["name"] == "mixed_multilingual":
            queries = runner.tokenize([texts["held_out"][0], texts["held_out"][2]])
            documents = runner.tokenize([texts["held_out"][1], texts["held_out"][3]])
            result["semantic_sanity"] = {}
            for mode in MODES:
                query_vectors = compiled[mode](**queries).float().cpu()
                document_vectors = compiled[mode](**documents).float().cpu()
                scores = query_vectors @ document_vectors.T
                result["semantic_sanity"][mode] = {
                    "scores": scores.tolist(), "top_document": scores.argmax(dim=1).tolist(),
                    "expected_top_document": [0, 1],
                    "both_sensible": scores.argmax(dim=1).tolist() == [0, 1]}
            print("SEMANTIC_SANITY " + json.dumps(result["semantic_sanity"]), flush=True)
        for _ in range(3):
            for mode in MODES:
                compiled[mode](**tokens)
        # Rotate lane order to reduce clock/thermal/order bias; all use identical inputs.
        for repeat in range(20):
            order = MODES[repeat % 3:] + MODES[:repeat % 3]
            for mode in order:
                _, elapsed = timed(lambda: compiled[mode](**tokens))
                row["modes"][mode]["samples_ms"].append(elapsed)
        for mode, values in row["modes"].items():
            values["median_ms"] = statistics.median(values["samples_ms"])
            values["physical_tokens_per_second"] = len(case["texts"]) * case["length"] * 1000 / values["median_ms"]
        for mode in MODES:
            row["modes"][mode]["speedup_vs_dense"] = row["modes"]["dense"]["median_ms"] / row["modes"][mode]["median_ms"]
        result["cases"].append(row)
        save()
        print("CASE " + json.dumps(row), flush=True)
        if args.profile_dir:
            from profile_w8a8 import capture
            row["profiles"] = {}
            for mode in MODES:
                label = case["name"] + "_" + mode
                row["profiles"][mode] = capture(lambda: compiled[mode](**tokens), args.profile_dir / label, label)
                save()
                print("PROFILE " + json.dumps(row["profiles"][mode]), flush=True)
    result["passed"] = True  # Finite normalized execution and dense parity, NOT retrieval quality.
    save()
    print("RESULT " + json.dumps({"passed": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
