"""Official Clef BF16 text-only NPU smoke and local-backbone correctness checks.

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

REFERENCE_REPOSITORY = "Cloudflare/clef-flash"
REFERENCE_REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"


def emit(event, **fields):
    print(json.dumps(dict(event=event, unix_time=time.time(), **fields), ensure_ascii=False), flush=True)


def check_local_backbone():
    """Small NPU correctness checks; no checkpoint or full-model load required."""
    import unittest
    from types import SimpleNamespace

    import torch
    import torch_npu  # noqa: F401
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from local_modeling_clef import TextBackbone, chunk_gated_delta_rule

    class BackboneTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            if not torch.npu.is_available():
                raise RuntimeError("NPU required; do not report skipped checks as validation")
            torch.npu.set_device(0)
            torch.set_num_threads(8)

        @torch.inference_mode()
        def test_chunk_scan_against_token_recurrence(self):
            # Independent scalar-token recurrence tests causal masking and chunk carry.
            for length in (1, 63, 64, 65, 129):
                with self.subTest(length=length):
                    torch.manual_seed(length)
                    q, k, v = [torch.randn(1, length, 2, 8, device="npu:0") for _ in range(3)]
                    g = -torch.rand(1, length, 2, device="npu:0")
                    beta = torch.rand(1, length, 2, device="npu:0")
                    actual = chunk_gated_delta_rule(q, k, v, g, beta)
                    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * 8**-0.5
                    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
                    state = torch.zeros(1, 2, 8, 8, device="npu:0")
                    expected = []
                    for t in range(length):
                        state = state * g[:, t].exp()[..., None, None]
                        residual = v[:, t] - (k[:, t, :, :, None] * state).sum(-2)
                        state = state + k[:, t, :, :, None] * (beta[:, t, :, None] * residual)[..., None, :]
                        expected.append((q[:, t, :, :, None] * state).sum(-2))
                    torch.testing.assert_close(actual, torch.stack(expected, dim=1), atol=2e-6, rtol=2e-5)

        @torch.inference_mode()
        def test_hybrid_backbone_against_transformers(self):
            config = Qwen3_5TextConfig(
                vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                layer_types=["linear_attention", "full_attention"], num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
                linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
                linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
                rope_parameters=dict(rope_type="default", rope_theta=10000000,
                                     partial_rotary_factor=0.5, mrope_section=[1, 1, 2]))
            config._attn_implementation = "eager"
            torch.manual_seed(7)
            reference = Qwen3_5TextModel(config).to(device="npu:0", dtype=torch.bfloat16).eval()
            local = TextBackbone(SimpleNamespace(**config.to_dict())).to(device="npu:0", dtype=torch.bfloat16).eval()
            local.load_state_dict(reference.state_dict(), strict=True)
            # .to(bfloat16) casts buffers too; the release loader initializes RoPE in FP32.
            inv, _ = reference.rotary_emb.compute_default_rope_parameters(config)
            reference.rotary_emb.inv_freq = inv.to("npu:0")
            local.inv_freq = inv.to("npu:0")
            for length in (1, 65, 129):
                with self.subTest(length=length):
                    ids = torch.randint(0, config.vocab_size, (1, length), device="npu:0")
                    expected = reference(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
                    torch.testing.assert_close(local(ids), expected, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "one nonempty"):
                local(torch.zeros(2, 4, dtype=torch.long, device="npu:0"))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BackboneTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--check-backbone", action="store_true",
                        help="Run small local-backbone checks against recurrence and Transformers")
    args = parser.parse_args()
    if args.check_backbone:
        check_local_backbone()
        return
    if args.model is None or args.output is None:
        parser.error("--model and --output are required for the full Transformers smoke")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    phase, stop = {"name": "imports"}, threading.Event()
    result = dict(status="running", repository=REFERENCE_REPOSITORY, reference_revision=REFERENCE_REVISION,
                  model_path=str(args.model.resolve()),
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
                validation_start = time.perf_counter()
                full_state_ids = processor.tokenizer(upstream.render(case["state"]), add_special_tokens=False).input_ids
                unbounded = upstream.encode_record(processor.tokenizer, case, max_length=1_000_000, processor=processor)
                fixture_validation_s = time.perf_counter() - validation_start
                torch.npu.synchronize()
                e2e_start = time.perf_counter()
                encoded = upstream.encode_record(processor.tokenizer, case, processor=processor)
                preprocessing_s = time.perf_counter() - e2e_start
                if encoded.media:
                    raise ValueError("Unexpected media in text-only smoke")
                # Detect any implicit truncation, rather than silently changing the input.
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
                finished = time.perf_counter()
                e2e_s = finished - e2e_start
                row = dict(id=case["id"], warmup=warmup, input_tokens=len(encoded.input_ids),
                           state_tokens=len(full_state_ids), question_count=len(encoded.questions),
                           option_counts=[len(q.option_ids) for q in encoded.questions],
                           token_ids=list(encoded.input_ids), question_spans=[q.question_span for q in encoded.questions],
                           option_spans=[q.option_spans for q in encoded.questions],
                           answers=answers, raw=raw, device_event_spans_ms=device_spans,
                           timing_s=dict(preprocessing=preprocessing_s, preparation=preparation_s,
                               synchronized_model=model_wall_s, output_postprocess=finished-output_start,
                               e2e=e2e_s, fixture_validation_excluded=fixture_validation_s),
                           model_input_tok_s=len(encoded.input_ids)/model_wall_s,
                           e2e_input_tok_s=len(encoded.input_ids)/e2e_s)
                result["rows"].append(row)
                emit("item_finished", **{k: v for k, v in row.items() if k not in ["token_ids", "question_spans", "option_spans"]})
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
