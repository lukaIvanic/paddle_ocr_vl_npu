#!/usr/bin/env python3
"""PP-DocLayoutV3 with continuously scheduled UniRec/Paddle crop recognition."""
import argparse
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "09_persistent_page_engine"), str(ROOT / "12_unirec_0_1b_inference")]
from routing import Routing, add_arguments


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--input", type=Path, required=True, help="Directory of page images")
    parser.add_argument("--dataset-json", type=Path, help="Use annotation order instead of filename order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--detailed-timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layout-model", type=Path, default=Path("/workspace/models/PP-DocLayoutV3_safetensors"))
    parser.add_argument("--paddle-model-path", type=Path, default=Path("/workspace/models/PaddleOCR-VL-1.6"))
    parser.add_argument("--paddle-batch-size", type=int, default=64)
    parser.add_argument("--vision-promptfa-align-128", action="store_true")
    parser.add_argument("--unirec-model-path", type=Path)
    parser.add_argument("--openocr-root", type=Path)
    parser.add_argument("--unirec-vision-cache", type=Path)
    parser.add_argument("--unirec-decode-cache", type=Path)
    parser.add_argument("--unirec-batch-size", type=int, default=128)
    return parser.parse_args()


def engine_report(adapter):
    summary = adapter.summary
    if hasattr(summary, "__dataclass_fields__"):
        # Crop outputs already have their own trace; do not recursively copy
        # the scheduler's complete request history into the run summary.
        summary = {
            field.name: getattr(summary, field.name)
            for field in fields(summary)
            if field.name != "completions"
        }
    # These legacy timers span generator suspension, including other engines
    # and sometimes setup after iterator priming. Do not expose misleading
    # "exclusive"/"bookkeeping" names as normal stage measurements.
    summary = dict(summary)
    lifetime = {}
    for group, keys in {
        "timing_detail": ("run_wall_s", "scheduler_bookkeeping_residual_s"),
        "timing_s": ("continuous_decode_wall", "decode_host_exclusive_wall", "run_scoped_scheduler_wall"),
    }.items():
        if group in summary:
            summary[group] = dict(summary[group])
            for key in keys:
                if key in summary[group]:
                    lifetime[f"legacy_{key}"] = summary[group].pop(key)
    return {
        "graph_calls": adapter.graph_calls,
        "capacity": adapter.capacity,
        "ready_capacity": getattr(adapter, "ready_capacity", None),
        "prefill_request_counts": dict(getattr(adapter, "prefill_request_counts", {})),
        "summary": summary,
        "cooperative_pause_inclusive_legacy_timing": {
            "note": "Includes iterator suspension and possibly setup after priming. NOT exclusive work or scheduler overhead; do not use for throughput.",
            "values": lifetime,
        },
        "prefill_timing_basis": "Device-event elapsed envelopes; can include transfers and host submission gaps. NOT kernel-active time.",
        "cpu_preparation": adapter.cpu.summary() if getattr(adapter, "cpu", None) is not None else None,
        "prefill_tokens": dict(getattr(adapter, "prefill_tokens", {})),
        "prefill_device_s": dict(getattr(adapter, "prefill_device_s", {})),
        "vision_runtime": adapter.vision.summary() if hasattr(adapter, "vision") else None,
    }


def make_paddle(args, emit):
    from paddleocr_vl.serving.engine import ContinuousRecognizer
    from adapters.paddle import PaddleAdapter
    r = ContinuousRecognizer(
        model=str(args.paddle_model_path), dtype="fp16", decode_backend="torchair",
        decode_optimization="combined_apply_pse_sentinel", batch_size=args.paddle_batch_size,
        cache_length=4096, max_new_tokens=4096,
        torchair_cache_dir=ROOT / ".runtime_cache/09_persistent_page_engine_torchair",
        vision_backend="torchair", vision_attention="prompt_flash_attention",
        vision_promptfa_align_128=args.vision_promptfa_align_128,
        vision_mlp_intermediate_size=4352, vision_linear_weight_format="fractal_nz",
        vision_buckets=(256,384,512,640,768,1408,1920,2048,2944,4096),
        vision_torchair_cache_dir=ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair",
        vision_padding="bucket", vision_packing="greedy", vision_pack_target=768,
        text_backend="torchair", text_buckets=(1152,), text_padding="bucket",
        text_torchair_cache_dir=ROOT / ".runtime_cache/09_persistent_page_engine_text_torchair",
        text_packing="production_group", text_pack_buckets=(128,256,384,512,768,1024),
        text_pack_max_members=32,
        text_packed_cache_dir=ROOT / ".runtime_cache/09_persistent_page_engine_text_packed_torchair",
        preprocessor_min_pixels=28224, preprocessor_max_pixels=802816,
    )
    return PaddleAdapter(r, emit)


def make_unirec(args, emit):
    os.environ["UNIREC_STATIC_CACHE_LEN"] = "2048"
    os.environ["UNIREC_STATIC_CROSS_CACHE_LEN"] = "1320"
    os.environ["UNIREC_VISION_BUCKET_PRESET"] = "310p_k20_l4"
    os.environ["UNIREC_RECOGNITION_INPUT_CONTRACT"] = "compact_uint8_hwc"
    from modeling_optimized_unirec import OptimizedUniRecRunner
    from decode_model_optimizations import apply_decode_model_optimizations, decode_cache_variant_root
    from continuous_unirec import ContinuousUniRecDecoder, production_decode_cache_parent
    from vision_bucket_presets import resolve_vision_bucket_specs
    from vision_full_batch import BucketedFullVisionRuntime
    from adapters.unirec import UniRecAdapter
    sys.path.append(str(args.openocr_root.resolve()))
    from tools import infer_doc_onnx
    r = OptimizedUniRecRunner(model_path=args.unirec_model_path, device="npu:0", dtype="float16", compile_cache_dir=args.unirec_vision_cache)
    apply_decode_model_optimizations(r, weight_format="nz", lm_head_rows=57344)
    r._static_cross_cache_len_by_processor_max_side[tuple(r.processor.max_side)] = 1320
    vision = BucketedFullVisionRuntime(r, specs=resolve_vision_bucket_specs("310p_k20_l4"), focal_depthwise_rewrite="constant_grouped_all", weight_format="torchair_internal", preset_name="310p_k20_l4", synchronize_first_call=False)
    r._get_compiled_packed_text_prefill_runtime()
    r.compile_cache_dir = decode_cache_variant_root(production_decode_cache_parent(args.unirec_decode_cache), weight_format="nz", lm_head_rows=57344)
    decoder = ContinuousUniRecDecoder(runner=r, batch_size=args.unirec_batch_size, max_length=2048, decode_mode="compiled_ifa", compile_backend="torchair", admission_prefetch_depth=0, self_cache_length=2048, cross_cache_length=1320)
    return UniRecAdapter(r, vision, decoder, emit, infer_doc_onnx, collect_step_timing=args.detailed_timing)


def main():
    args = parse_args()
    routing = Routing(args.text_model, args.table_model, args.formula_model)
    os.environ.setdefault("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "0")
    os.environ.setdefault("TE_PARALLEL_COMPILER", "1")
    import torch
    import torch_npu
    from coordinator import Coordinator
    from page_source import PageInbox, PageSource
    from pipeline.layout_frontend import OwnedLayoutFrontend
    from pipeline.layout_mask_guard import install_layout_mask_guard
    torch.npu.set_device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    if args.dataset_json:
        annotations = json.loads(args.dataset_json.read_text())
        paths = [args.input / Path(row["page_info"]["image_path"]).name for row in annotations]
    else:
        paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    paths = paths[args.offset:][:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions = args.output_dir / "predictions"
    predictions.mkdir()
    inbox = PageInbox()
    for path in paths:
        inbox.submit(path)
    inbox.close()
    started = time.perf_counter()
    adapters = {}
    install_layout_mask_guard()
    with (args.output_dir / "recognition_trace.jsonl").open("w") as trace:
        def emit_page(result):
            with timing.scope("output.markdown_and_images_build_write"):
                result.save_to_markdown(str(predictions))
            with timing.scope("output.json_encode"):
                content = json.dumps(result.json, ensure_ascii=False, indent=2) + "\n"
            with timing.scope("output.json_write"):
                (predictions / (Path(result.data["input_path"]).stem + ".json")).write_text(content)
        def emit_trace(record):
            with timing.scope("output.crop_trace_encode_write"):
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
        # UniRec's loader changes the process-global format flag. Finish its
        # initialization first, then establish the shared production setting.
        holder = {}
        def emit(*values):
            holder["source"].complete(*values)
        if "unirec" in routing.models:
            adapters["unirec"] = make_unirec(args, emit)
        torch.npu.config.allow_internal_format = True
        if "paddle" in routing.models:
            adapters["paddle"] = make_paddle(args, emit)
        frontend = OwnedLayoutFrontend(args.layout_model, torch.device("npu:0"), graph_capture=False)
        source = PageSource(inbox, frontend, routing, emit_page, emit_trace)
        holder["source"] = source
        setup_s = time.perf_counter() - started
        print(f"HYBRID setup_finish setup_s={setup_s:.3f} routes={asdict(routing)}", flush=True)
        from hybrid_timing import PipelineTiming, install
        timing = PipelineTiming(enabled=args.detailed_timing)
        install(timing, adapters, source)
        try:
            with torch.inference_mode():
                result = Coordinator(adapters, source, decode_steps=args.decode_steps, timing=timing).run()
            if source.completed != len(paths) or source.pages or source.owners:
                raise RuntimeError(
                    f"Incomplete pipeline drain: completed={source.completed}/{len(paths)} "
                    f"pending_pages={len(source.pages)} pending_crops={len(source.owners)}"
                )
            result.update(
                pages=source.completed,
                pages_per_s=source.completed / result["wall_s"],
                setup_s=setup_s,
                routing=asdict(routing),
                engines={name: engine_report(a) for name, a in adapters.items()},
                page_preparation=source.summary(),
                detailed_timing=timing.summary(),
                peak_torch_allocated_bytes=torch.npu.max_memory_allocated(),
                peak_torch_reserved_bytes=torch.npu.max_memory_reserved(),
                arguments={
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            )
            diagnostics_started = time.perf_counter()
            if timing.enabled:
                timing.trace.write_json(args.output_dir / "timing_trace.json")
            result["diagnostic_trace_write_s_excluded_from_pipeline"] = time.perf_counter() - diagnostics_started
            (args.output_dir / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
        finally:
            source.close()
            for adapter in reversed(tuple(adapters.values())):
                adapter.close()
            frontend.close()


if __name__ == "__main__":
    main()
