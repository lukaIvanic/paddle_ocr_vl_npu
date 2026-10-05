"""Official Clef BF16 text-only NPU smoke; no local model replacement or optimization.

All timings come from complete real requests, not isolated stage replay. Event
spans include stream/host-enqueue gaps and are not summed kernel durations.
Semantic expectations are observations, not a formal accuracy benchmark.
"""
import argparse
from collections import Counter
import importlib.metadata
import importlib.util
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

from download_model import verify


def emit(event, **fields):
    print(json.dumps(dict(event=event, unix_time=time.time(), **fields), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bf16"], required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    release = json.loads(Path(__file__).with_name("release.json").read_text())
    phase, stop = {"name": "imports"}, threading.Event()
    result = dict(status="running", repository=release["repository"], revision=release["revision"],
                  hostname=platform.node(), command=sys.argv,
                  physical_device=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  timing_scope="synchronized B1 complete requests; cold/warm labeled; not a throughput benchmark",
                  event_scope="elapsed stream spans, including enqueue gaps; not isolated kernel execution",
                  rows=[], activation_samples={}, optimizations=[], cache_reuse=False, modality="text_only")
    handles = []

    def save():
        partial = args.output.with_suffix(".json.partial")
        partial.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        partial.replace(args.output)

    def heartbeat():
        while not stop.wait(5):
            emit("heartbeat", phase=phase["name"], completed_requests=len(result["rows"]))

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        import torch
        import torch_npu  # noqa: F401 -- register NPU backend, never CPU fallback
        if not torch.npu.is_available():
            raise RuntimeError("NPU unavailable; CPU inference fallback forbidden")
        torch.npu.set_device(0)
        torch.set_num_threads(8)
        torch.manual_seed(0)
        result["versions"] = {p: importlib.metadata.version(p)
                              for p in ["torch", "torch-npu", "transformers", "accelerate", "safetensors"]}
        result["device_name"] = torch.npu.get_device_name(0)
        if "910B" not in result["device_name"]:
            raise RuntimeError("This BF16 smoke is scoped to 910B only")
        free, total = torch.npu.mem_get_info()
        result["initial_memory"] = dict(free_bytes=free, total_bytes=total)
        if free < 26 * 1024**3:
            raise RuntimeError("Less than 26 GiB free; refusing Clef-flash model load")
        phase["name"] = "verify_release"
        # Check every file, including executable upstream code, before import.
        for name, entry in release["files"].items():
            if not verify(args.model / name, entry):
                raise ValueError(f"Pinned release verification failed: {name}")
            emit("file_verified", name=name)
        spec = importlib.util.spec_from_file_location("clef_official_joint_schema", args.model / "joint_schema_model.py")
        upstream = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = upstream  # dataclasses requires module registration
        spec.loader.exec_module(upstream)
        phase["name"] = "load_official_model"
        start = time.perf_counter()
        model, processor = upstream.load_release_model(
            args.model, device="npu:0", dtype=torch.bfloat16,
            local_files_only=True, attn_implementation="eager")
        torch.npu.synchronize()
        result["load_s"] = time.perf_counter() - start
        if any(p.device.type != "npu" or p.dtype != torch.bfloat16 for p in model.parameters()):
            raise RuntimeError("Expected all model parameters BF16 on NPU")
        counts = Counter()
        for name, parameter in model.named_parameters():
            group = "head" if name.startswith("head.") else "backbone"
            counts[f"{group}:{parameter.dtype}:{parameter.device.type}"] += parameter.numel()
        result["parameter_counts_by_dtype"] = dict(counts)
        result["buffer_counts_by_dtype"] = dict(Counter(str(b.dtype) for b in model.buffers()))
        config = model.language_model.config.text_config
        result["architecture"] = dict(model_type=config.model_type, hidden_size=config.hidden_size,
                                      layer_count=config.num_hidden_layers,
                                      layer_types=list(config.layer_types),
                                      attention_implementation=config._attn_implementation)
        text_model = model.language_model.model
        if hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        current_events = {}

        def stage_hooks(name, module):
            def before(_module, _inputs):
                event = torch.npu.Event(enable_timing=True)
                event.record()
                current_events[name + ":start"] = event

            def after(_module, _inputs, _output):
                event = torch.npu.Event(enable_timing=True)
                event.record()
                current_events[name + ":end"] = event

            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))

        stage_hooks("backbone", text_model)
        stage_hooks("decision_head", model.head)

        def sample(name):
            def hook(_module, _inputs, output):
                if name in result["activation_samples"]:
                    return
                if hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state
                if isinstance(output, tuple):
                    output = next((t for t in output if torch.is_tensor(t)), None)
                if torch.is_tensor(output):
                    result["activation_samples"][name] = dict(dtype=str(output.dtype),
                        device=str(output.device), shape=list(output.shape))
            return hook

        for name, module in model.named_modules():
            if name.endswith("layers.0") or name.endswith("layers.3") or name == "head.memory_projection":
                handles.append(module.register_forward_hook(sample(name)))
        handles.append(text_model.register_forward_hook(sample("backbone_last_hidden_state")))
        cases = json.loads(Path(__file__).with_name("smoke_cases.json").read_text())
        emit("model_ready", load_s=result["load_s"], versions=result["versions"],
             architecture=result["architecture"], parameter_counts=result["parameter_counts_by_dtype"])
        save()
        with torch.inference_mode():
            for index, case in enumerate([cases[0], *cases]):
                warmup = index == 0
                phase["name"] = ("warmup:" if warmup else "request:") + case["id"]
                emit("request_start", id=case["id"], warmup=warmup)
                torch.npu.synchronize()
                e2e_start = time.perf_counter()
                encoded = upstream.encode_record(processor.tokenizer, case, processor=processor)
                preprocessing_s = time.perf_counter() - e2e_start
                if encoded.media:
                    raise ValueError("Unexpected media in text-only smoke")
                # Detect any implicit truncation, rather than silently changing the input.
                full_state_ids = processor.tokenizer(upstream.render(case["state"]), add_special_tokens=False).input_ids
                unbounded = upstream.encode_record(processor.tokenizer, case, max_length=1_000_000, processor=processor)
                if encoded.input_ids != unbounded.input_ids:
                    raise ValueError("Smoke input was truncated")
                transfer_start = time.perf_counter()
                batch = upstream.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("npu:0"))
                torch.npu.synchronize()
                preparation_s = time.perf_counter() - transfer_start
                current_events.clear()
                start = time.perf_counter()
                logits = model(batch)[0]
                torch.npu.synchronize()
                model_wall_s = time.perf_counter() - start
                device_spans = {name: current_events[name+":start"].elapsed_time(current_events[name+":end"])
                                for name in ["backbone", "decision_head"]}
                output_start = time.perf_counter()
                answers, raw = {}, {}
                if len(logits) != len(encoded.questions):
                    raise ValueError("Question/logit count mismatch")
                for question, values in zip(encoded.questions, logits):
                    if values.numel() != len(question.option_ids) or values.device.type != "npu":
                        raise ValueError("Invalid decision output shape or device")
                    probabilities = values.float().softmax(-1).cpu().tolist()
                    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
                        raise ValueError("Nonfinite/invalid probabilities")
                    if not math.isclose(sum(probabilities), 1, abs_tol=1e-5):
                        raise ValueError("Probabilities do not sum to one")
                    raw[question.question_id] = dict(logits=values.float().cpu().tolist(),
                        logit_dtype=str(values.dtype), probabilities=dict(zip(question.option_ids, probabilities)))
                    answers[question.question_id] = upstream.systemone_answer(
                        case["questions"][question.question_id], raw[question.question_id]["probabilities"])
                e2e_s = time.perf_counter() - e2e_start
                row = dict(id=case["id"], warmup=warmup, input_tokens=len(encoded.input_ids),
                           state_tokens=len(full_state_ids), question_count=len(encoded.questions),
                           option_counts=[len(q.option_ids) for q in encoded.questions],
                           token_ids=list(encoded.input_ids), question_spans=[q.question_span for q in encoded.questions],
                           answers=answers, raw=raw, device_event_spans_ms=device_spans,
                           timing_s=dict(preprocessing=preprocessing_s, preparation=preparation_s,
                               synchronized_model=model_wall_s, output_postprocess=time.perf_counter()-output_start,
                               e2e=e2e_s),
                           model_input_tok_s=len(encoded.input_ids)/model_wall_s,
                           e2e_input_tok_s=len(encoded.input_ids)/e2e_s)
                result["rows"].append(row)
                emit("item_finished", **{k: v for k, v in row.items() if k not in ["token_ids", "question_spans"]})
                save()
        measured = {r["id"]: r for r in result["rows"] if not r["warmup"]}
        observations = {}
        for case in cases:
            for question, expected in case.get("expected_top_options", {}).items():
                probs = measured[case["id"]]["raw"][question]["probabilities"]
                observations[f"{case['id']}:{question}"] = max(probs, key=probs.get) == expected
        for language in ["english", "chinese"]:
            observations[language + "_relevance_ranking"] = (
                measured[language+"_relevant"]["answers"]["relevance"]["score"] >
                measured[language+"_irrelevant"]["answers"]["relevance"]["score"])
        result["semantic_observations"] = observations
        result["all_semantic_observations_passed"] = all(observations.values())
        if result["activation_samples"]["backbone_last_hidden_state"]["dtype"] != "torch.bfloat16":
            raise RuntimeError("Backbone output did not preserve the requested BF16 precision")
        if result["activation_samples"]["head.memory_projection"]["dtype"] != "torch.bfloat16":
            raise RuntimeError("Head activation did not preserve the requested BF16 precision")
        result["peak_allocated_bytes"] = torch.npu.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.npu.max_memory_reserved()
        result["status"] = "completed"
        emit("smoke_complete", semantic_observations=observations,
             peak_allocated_bytes=result["peak_allocated_bytes"], activations=result["activation_samples"])
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        for handle in handles:
            handle.remove()
        stop.set()
        thread.join()
        save()
        emit("complete", status=result["status"], output=str(args.output))


if __name__ == "__main__":
    main()
