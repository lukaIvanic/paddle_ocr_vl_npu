"""Native Decision2 API with explicit Ascend BF16 autocast and FP32 head.

Not a stock vLLM endpoint. CUDA/ROCm graph/kernels and shared-context optimization
are disabled. There is no CPU inference fallback. Cases are functional checks,
not a benchmark of decision quality. Concurrent T2 timings are not clean speed.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback


def emit(event, **fields):
    print(json.dumps({"event": event, "unix_time": time.time(), **fields}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bf16"], required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("smoke_cases.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    phase = {"name": "imports"}
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(5):
            emit("heartbeat", phase=phase["name"])
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    result = {"status": "running", "chip": "910B2", "concurrent_workload": "T2 reranker",
              "command": sys.argv, "hostname": platform.node(),
              "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "rows": [], "activation_samples": {}}
    try:
        import torch
        import torch_npu  # noqa: F401 -- registers NPU backend
        if not torch.npu.is_available():
            raise RuntimeError("NPU unavailable; CPU fallback forbidden")
        torch.npu.set_device(0)
        free, total = torch.npu.mem_get_info()
        result["initial_memory"] = {"free_bytes": free, "total_bytes": total}
        if free < 8 * 1024**3:
            raise RuntimeError("Less than 8 GiB free; refusing to share this NPU")
        torch.set_num_threads(8)
        result["versions"] = {p: importlib.metadata.version(p) for p in ("torch", "torch-npu", "transformers")}
        sys.path.insert(0, str(args.model.resolve()))
        from decision2 import Decision2
        from decision2.qwen import keep_linear_bf16
        phase["name"] = "verified_model_load"
        started = time.perf_counter()
        runtime = Decision2.from_pretrained(str(args.model), device="npu:0", threads=8,
                                           graphs=False, kernels=False, share_context=False)
        result["bf16_residency"] = keep_linear_bf16(runtime.backend.model.backbone, torch)
        torch.npu.synchronize()
        result["load_s"] = time.perf_counter() - started
        model = runtime.backend.model
        if any(p.device.type != "npu" for p in model.parameters()):
            raise RuntimeError("Model contains non-NPU parameters")
        result["parameter_count"] = runtime.backend.parameter_count()
        result["parameter_dtypes"] = {}
        for name, param in model.named_parameters():
            group = "backbone" if name.startswith("backbone.") else "head"
            key = group + ":" + str(param.dtype)
            result["parameter_dtypes"][key] = result["parameter_dtypes"].get(key, 0) + param.numel()
            if group == "head" and param.dtype != torch.float32:
                raise RuntimeError("Decision head must remain FP32")
        def inspect_output(name):
            def hook(module, inputs, output):
                if name in result["activation_samples"]:
                    return
                if torch.is_tensor(output):
                    result["activation_samples"][name] = {"dtype": str(output.dtype),
                        "device": str(output.device), "shape": list(output.shape)}
            return hook
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and ("layers.0." in name or not name.startswith("backbone")):
                handles.append(module.register_forward_hook(inspect_output(name)))
        handles.append(model.register_forward_hook(inspect_output("decision_logits")))
        cases = json.loads(args.cases.read_text())
        emit("model_ready", **{k: result[k] for k in ("load_s", "versions", "parameter_dtypes", "bf16_residency")})
        for index, case in enumerate([cases[0], *cases]):
            warmup = index == 0
            phase["name"] = ("warmup:" if warmup else "request:") + case["id"]
            emit("request_start", id=case["id"], warmup=warmup)
            torch.npu.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("npu", dtype=torch.bfloat16):
                response = runtime.system_one(state=case["state"], questions=case["questions"])
            torch.npu.synchronize()
            elapsed = time.perf_counter() - started
            for answer in response["answers"].values():
                if "error" in answer:
                    raise RuntimeError(f"Invalid answer: {answer}")
                if answer["type"] == "noul":
                    assert math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1
                else:
                    probs = list(answer["probabilities"].values())
                    assert all(math.isfinite(v) and 0 <= v <= 1 for v in probs)
                    assert abs(sum(probs) - 1) < 1e-5
            row = {"id": case["id"], "warmup": warmup, "wall_s": elapsed, "response": response}
            result["rows"].append(row)
            emit("request_finished", **row)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        for handle in handles:
            handle.remove()
        assert result["activation_samples"]["decision_logits"]["dtype"] == "torch.float32"
        assert any(s["dtype"] == "torch.bfloat16" for n, s in result["activation_samples"].items() if n.startswith("backbone"))
        answers = {r["id"]: r["response"]["answers"] for r in result["rows"] if not r["warmup"]}
        result["sanity_observations"] = {
            "english_returns_selected": answers["english_returns"]["route"]["choice"] == "returns",
            "chinese_returns_selected": answers["chinese_returns"]["route"]["choice"] == "returns",
            "relevant_scores_above_irrelevant": answers["relevance_positive"]["relevance"]["score"] > answers["relevance_negative"]["relevance"]["score"],
        }
        result["status"] = "completed"
        result["peak_allocated_bytes"] = torch.npu.max_memory_allocated()
        emit("smoke_complete", observations=result["sanity_observations"], activations=result["activation_samples"])
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        stop.set()
        thread.join()
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
