"""Run the plain PyTorch Clef baseline against the saved Transformers smoke.

This process blocks Transformers imports, including transitive imports. Inputs,
schema spans and decision logits must match exactly; no tolerance is tuned to
the observed result. Timing is synchronized B1 smoke timing, not throughput.
"""
import argparse
from dataclasses import dataclass
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
from typing import Any


SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    record_id: str


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = 16384,
) -> EncodedRecord:
    if record.get("images") or record.get("videos"):
        raise ValueError("This baseline supports text only")
    if not isinstance(record.get("questions"), dict) or not record["questions"]:
        raise ValueError("At least one question is required")
    for question in record["questions"].values():
        if question.get("type") not in QUESTION_TYPES:
            raise ValueError("Question type must be noul, choice or score")
        if question["type"] == "choice" and not isinstance(question.get("criteria"), dict):
            raise ValueError("Choice criteria must be an object")
        if question["type"] == "score" and not isinstance(question.get("criteria"), list):
            raise ValueError("Score criteria must be a list")
        if question["type"] != "noul" and not question["criteria"]:
            raise ValueError("Criteria must not be empty")
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    for question_index, (question_id, question) in enumerate(record["questions"].items()):
        schema_ids.extend(
            _tokens(
                tokenizer,
                f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ",
            )
        )
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        question_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))

        option_spans: list[tuple[int, int]] = []
        option_ids: list[str] = []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(
            EncodedQuestion(
                question_id=str(question_id),
                question_type=QUESTION_TYPES[str(question["type"])],
                question_span=(question_start, question_end),
                option_spans=tuple(option_spans),
                option_ids=tuple(option_ids),
            )
        )

    prefix_ids = _tokens(
        tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
    )
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )
    state_ids = _tokens(tokenizer, render(record["state"]))
    length = len(prefix_ids) + len(state_ids) + len(schema_ids) + len(suffix_ids)
    if length > max_length:
        raise ValueError(f"Request has {length} tokens, maximum is {max_length}; truncation is disabled")
    schema_offset = len(prefix_ids) + len(state_ids)
    shifted_questions = tuple(
        EncodedQuestion(
            question_id=question.question_id,
            question_type=question.question_type,
            question_span=(
                question.question_span[0] + schema_offset,
                question.question_span[1] + schema_offset,
            ),
            option_spans=tuple(
                (start + schema_offset, end + schema_offset)
                for start, end in question.option_spans
            ),
            option_ids=question.option_ids,
        )
        for question in questions
    )
    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    if not input_ids or not shifted_questions:
        raise ValueError("record produced no model input or questions")
    return EncodedRecord(
        input_ids=input_ids,
        questions=shifted_questions,
        record_id=str(record.get("id", "unknown")),
    )


def systemone_answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    """Convert per-option probabilities for one question into a SystemOne answer."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {option: round(probabilities[option], 4) for option in options},
        }
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(sum(index * probabilities[level] for index, level in enumerate(levels)), 4),
        "confidence": round(max(probabilities[level] for level in levels), 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {level: round(probabilities[level], 4) for level in levels},
    }


class NoTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "transformers" or fullname.startswith("transformers."):
            raise ImportError("The local Clef runtime must not import Transformers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status="running", git_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), command=sys.argv,
        hostname=platform.node(), physical_device=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        dtype="bfloat16", modality="text_only", cache_reuse=False, rows=[])
    phase = "imports"
    stop = threading.Event()

    def emit(event, **fields):
        print(json.dumps(dict(event=event, **fields), ensure_ascii=False), flush=True)

    def save():
        partial = args.output.with_suffix(".partial")
        partial.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        partial.replace(args.output)

    def heartbeat():
        while not stop.wait(5):
            emit("heartbeat", phase=phase, completed=len(result["rows"]))

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        sys.meta_path.insert(0, NoTransformers())
        import torch
        import torch_npu  # noqa: F401
        from tokenizers import Tokenizer
        from local_modeling_clef import load_model

        if not torch.npu.is_available():
            raise RuntimeError("NPU unavailable; no CPU fallback")
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
        result["versions"] = {p: importlib.metadata.version(p)
                              for p in ("torch", "torch-npu", "tokenizers", "safetensors")}
        reference_path = root / "references/transformers_bf16_910b_f9bb6837/result.json"
        reference = json.loads(reference_path.read_text())
        result["reference_commit"] = reference["git_commit"]
        result["reference_revision"] = reference["revision"]
        result["model_path"] = str(args.model.resolve())
        references = {r["id"]: r for r in reference["rows"] if not r["warmup"]}
        phase = "load_model"
        start = time.perf_counter()
        model = load_model(args.model, "npu:0", progress=lambda name: emit("load", step=name))
        torch.npu.synchronize()
        result["load_s"] = time.perf_counter() - start
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        cases = json.loads((root / "smoke_cases.json").read_text())
        checks = []
        events = {}

        def before(name):
            def hook(module, inputs):
                events[name + ":start"] = torch.npu.Event(enable_timing=True)
                events[name + ":start"].record()
            return hook

        def after(name):
            def hook(module, inputs, output):
                events[name + ":end"] = torch.npu.Event(enable_timing=True)
                events[name + ":end"].record()
            return hook

        for name, module in (("backbone", model.backbone), ("head", model.head)):
            module.register_forward_pre_hook(before(name))
            module.register_forward_hook(after(name))
        with torch.inference_mode():
            for index, case in enumerate([cases[0], *cases]):
                warmup = index == 0
                phase = ("warmup:" if warmup else "request:") + case["id"]
                expected = references[case["id"]]
                # Fixture validation is outside measured request time.
                encoded = encode_record(tokenizer, case)
                if (list(encoded.input_ids) != expected["token_ids"]
                    or [list(q.question_span) for q in encoded.questions] != expected["question_spans"]
                    or [[list(s) for s in q.option_spans] for q in encoded.questions] != expected["option_spans"]):
                    raise AssertionError(f"Input token/span mismatch: {case['id']}")
                torch.npu.synchronize()
                start = time.perf_counter()
                encoded = encode_record(tokenizer, case)
                preprocessing_s = time.perf_counter() - start
                ids = torch.tensor([encoded.input_ids], device="npu:0", dtype=torch.long)
                torch.npu.synchronize()
                model_start = time.perf_counter()
                logits = model(ids, encoded)
                torch.npu.synchronize()
                model_s = time.perf_counter() - model_start
                raw, answers, parity = {}, {}, {}
                if len(logits) != len(encoded.questions):
                    raise AssertionError("Question/logit count mismatch")
                for question, values in zip(encoded.questions, logits):
                    if values.dtype != torch.bfloat16 or values.device.type != "npu":
                        raise AssertionError("Expected BF16 logits on NPU")
                    actual = values.float().cpu()
                    baseline = torch.tensor(expected["raw"][question.question_id]["logits"])
                    if actual.shape != baseline.shape or not torch.isfinite(actual).all():
                        raise AssertionError("Nonfinite or malformed logits")
                    exact = torch.equal(actual, baseline)
                    parity[question.question_id] = dict(exact=exact, max_abs=float((actual-baseline).abs().max()))
                    probabilities = dict(zip(question.option_ids, values.float().softmax(-1).cpu().tolist()))
                    raw[question.question_id] = dict(logits=actual.tolist(), probabilities=probabilities)
                    answers[question.question_id] = systemone_answer(case["questions"][question.question_id], probabilities)
                row = dict(id=case["id"], warmup=warmup, input_tokens=len(encoded.input_ids),
                           token_ids=list(encoded.input_ids), raw=raw, answers=answers, parity=parity,
                           answers_exact=answers == expected["answers"],
                           timing_s=dict(preprocessing=preprocessing_s, synchronized_model=model_s,
                                         e2e=time.perf_counter()-start),
                           device_event_spans_ms={n: events[n+":start"].elapsed_time(events[n+":end"])
                                                  for n in ("backbone", "head")})
                result["rows"].append(row)
                checks.append(all(p["exact"] for p in parity.values()) and row["answers_exact"])
                emit("item_finished", **{k: v for k, v in row.items() if k != "token_ids"})
                save()
        result["transformers_imported"] = any(n == "transformers" or n.startswith("transformers.") for n in sys.modules)
        result["peak_allocated_bytes"] = torch.npu.max_memory_allocated()
        result["all_exact"] = all(checks)
        if result["transformers_imported"] or not result["all_exact"]:
            raise AssertionError("Local runtime import/parity gate failed; inspect per-question differences")
        result["status"] = "completed"
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
