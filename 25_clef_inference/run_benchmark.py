"""Score text choice benchmarks from Decision Index JSONL rows on one 910B.

Uses the local model by default; --backend transformers provides an independent
oracle. Results contain IDs, hashes and decisions, never benchmark text/tokens.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback

from run_local_smoke import NoTransformers, encode_record, systemone_answer


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("local", "transformers"), default="local")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N rows (a labeled subset)")
    parser.add_argument("--reference", type=Path, help="Require exact inputs/logits on every row in a prior result")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Expected nonempty rows with unique IDs")
    available = len(cases)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        cases = cases[:args.limit]
    for case in cases:
        if set(case["questions"]) != set(case["expected"]):
            raise ValueError("Every question must have a gold answer")
        for key, question in case["questions"].items():
            if question["type"] != "choice" or case["expected"][key] not in question["criteria"]:
                raise ValueError("This runner scores labeled choice questions only")
    references = {}
    if args.reference:
        reference = json.loads(args.reference.read_text())
        if reference["status"] != "completed":
            raise ValueError("Reference run is incomplete")
        references = {row["id"]: row for row in reference["rows"]}
        if not references or not references.keys() <= {case["id"] for case in cases}:
            raise ValueError("Every reference row must be included in this run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status="running", backend=args.backend, command=sys.argv,
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  hostname=platform.node(), physical_device=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  model_path=str(args.model.resolve()), rows_sha256=hashlib.sha256(args.rows.read_bytes()).hexdigest(),
                  available_requests=available, selected_requests=len(cases), subset=len(cases) < available,
                  dtype="bfloat16", modality="text_only", cache_reuse=False, rows=[])
    phase, stop = "imports", threading.Event()

    def emit(event, **fields):
        print(json.dumps(dict(event=event, **fields)), flush=True)

    def save():
        partial = args.output.with_suffix(".partial")
        partial.write_text(json.dumps(result, indent=2) + "\n")
        partial.replace(args.output)

    def heartbeat():
        while not stop.wait(10):
            emit("heartbeat", phase=phase, completed=len(result["rows"]), total=len(cases))

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        if args.backend == "local":
            sys.meta_path.insert(0, NoTransformers())
        import torch
        import torch_npu  # noqa: F401
        from tokenizers import Tokenizer

        if not torch.npu.is_available():
            raise RuntimeError("NPU required; no CPU fallback")
        torch.npu.set_device(0)
        torch.set_num_threads(8)
        torch.manual_seed(0)
        result["device_name"] = torch.npu.get_device_name(0)
        if "910B" not in result["device_name"]:
            raise RuntimeError("This BF16 experiment requires a 910B")
        free, total = torch.npu.mem_get_info()
        result["initial_memory"] = dict(free_bytes=free, total_bytes=total)
        if free < 26 * 1024**3:
            raise RuntimeError("Less than 26 GiB free; refusing model load")
        packages = ["torch", "torch-npu", "tokenizers", "safetensors"]
        if args.backend == "transformers":
            packages += ["transformers", "accelerate"]
        result["versions"] = {name: importlib.metadata.version(name) for name in packages}
        tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        # Validate complete inputs before loading the model. Never truncate.
        lengths = [len(encode_record(tokenizer, case).input_ids) for case in cases]
        result["input_tokens"] = dict(min=min(lengths), max=max(lengths), mean=statistics.mean(lengths))
        phase = "load_model"
        start = time.perf_counter()
        if args.backend == "local":
            from local_modeling_clef import load_model
            model = load_model(args.model, "npu:0", progress=lambda step: emit("load", step=step))
        else:
            spec = importlib.util.spec_from_file_location("clef_official_joint_schema", args.model / "joint_schema_model.py")
            upstream = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = upstream
            spec.loader.exec_module(upstream)
            model, processor = upstream.load_release_model(args.model, device="npu:0", dtype=torch.bfloat16,
                                                          local_files_only=True, attn_implementation="eager")
        model.eval()
        torch.npu.synchronize()
        result["load_s"] = time.perf_counter() - start
        if any(p.dtype != torch.bfloat16 or p.device.type != "npu" for p in model.parameters()):
            raise RuntimeError("Expected BF16 model parameters on NPU")
        emit("model_ready", load_s=result["load_s"], input_tokens=result["input_tokens"])
        save()
        with torch.inference_mode():
            for index, case in enumerate([cases[0], *cases]):
                warmup = index == 0
                phase = ("warmup:" if warmup else "request:") + case["id"]
                # Only state and schema enter the inference path; labels stay in the scorer.
                request = {"id": case["id"], "state": case["state"], "questions": case["questions"]}
                torch.npu.synchronize()
                start = time.perf_counter()
                encoded = encode_record(tokenizer, request)
                spans = [(q.question_id, q.question_span, q.option_spans, q.option_ids) for q in encoded.questions]
                input_hash = digest([encoded.input_ids, spans])
                if args.backend == "local":
                    batch = torch.tensor([encoded.input_ids], device="npu:0", dtype=torch.long)
                else:
                    official = upstream.encode_record(processor.tokenizer, request, max_length=16384, processor=processor)
                    official_spans = [(q.question_id, q.question_span, q.option_spans, q.option_ids) for q in official.questions]
                    if digest([official.input_ids, official_spans]) != input_hash:
                        raise AssertionError("Official and local input tokens/spans differ")
                    batch = upstream.collate_records([official], processor.tokenizer.pad_token_id, torch.device("npu:0"))
                torch.npu.synchronize()
                model_start = time.perf_counter()
                logits = model(batch, encoded) if args.backend == "local" else model(batch)[0]
                torch.npu.synchronize()
                model_s = time.perf_counter() - model_start
                if len(logits) != len(encoded.questions):
                    raise ValueError("Question/logit count mismatch")
                raw, answers = {}, {}
                for question, values in zip(encoded.questions, logits):
                    if (values.dtype != torch.bfloat16 or values.device.type != "npu"
                        or values.ndim != 1 or values.numel() != len(question.option_ids)
                        or not torch.isfinite(values).all().item()):
                        raise ValueError("Invalid BF16 decision logits")
                    probabilities = dict(zip(question.option_ids, values.float().softmax(-1).cpu().tolist()))
                    raw[question.question_id] = dict(logits=values.float().cpu().tolist(), probabilities=probabilities)
                    answers[question.question_id] = systemone_answer(request["questions"][question.question_id], probabilities)
                row = dict(id=case["id"], input_sha256=input_hash, input_tokens=len(encoded.input_ids), raw=raw,
                           predictions={key: answer["choice"] for key, answer in answers.items()},
                           expected=case["expected"], timing_s=dict(synchronized_model=model_s, e2e=time.perf_counter()-start))
                row["correct"] = sum(row["predictions"][key] == value for key, value in case["expected"].items())
                row["questions"] = len(case["expected"])
                if not warmup:
                    if case["id"] in references:
                        expected = references[case["id"]]
                        if any(row[key] != expected[key] for key in ("input_sha256", "raw", "predictions", "expected")):
                            raise AssertionError("Reference parity failed: " + case["id"])
                    result["rows"].append(row)
                    save()
                else:
                    result["warmup_s"] = row["timing_s"]
                emit("item_finished", id=case["id"], warmup=warmup, input_tokens=row["input_tokens"],
                     correct=row["correct"], questions=row["questions"], model_s=model_s)
        result["transformers_imported"] = any(n == "transformers" or n.startswith("transformers.") for n in sys.modules)
        if args.backend == "local" and result["transformers_imported"]:
            raise AssertionError("Transformers imported in local runtime")
        correct = sum(row["correct"] for row in result["rows"])
        questions = sum(row["questions"] for row in result["rows"])
        result["score"] = dict(correct=correct, questions=questions, accuracy_percent=100 * correct / questions)
        result["reference_rows_exact"] = len(references)
        result["timing_s"] = {}
        for key in ("synchronized_model", "e2e"):
            times = [row["timing_s"][key] for row in result["rows"]]
            result["timing_s"][key] = dict(total=sum(times), mean=statistics.mean(times), median=statistics.median(times),
                                          min=min(times), max=max(times))
        result["peak_allocated_bytes"] = torch.npu.max_memory_allocated()
        result["status"] = "completed"
        emit("score", **result["score"], timing_s=result["timing_s"], reference_rows_exact=len(references))
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        stop.set()
        thread.join()
        save()
        emit("complete", status=result["status"], output=str(args.output))


if __name__ == "__main__":
    main()
