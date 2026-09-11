#!/usr/bin/env python3
"""Continuous PPv3 + UniRec/MinerU service, using experiment 18's owner."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[1:1] = [str(ROOT / name) for name in (
    "18_unirec_paddle_hybrid_pipeline", "09_persistent_page_engine",
    "12_unirec_0_1b_inference", "11_mineru_2_5_pro_inference")]
spec = importlib.util.spec_from_file_location(
    "experiment18_runner", ROOT / "18_unirec_paddle_hybrid_pipeline/run_pipeline.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
from hybrid_routing import Routing, add_arguments


def build_parser():
    parser = shared.build_parser(add_routes=add_arguments, include_paddle=False)
    parser.description = __doc__
    parser.add_argument("--mineru-model-path", type=Path, default=Path("/workspace/models/MinerU2.5-Pro-2605-1.2B"))
    parser.add_argument("--mineru-batch-size", type=int, default=32)
    parser.add_argument("--mineru-ready-capacity", type=int, default=32)
    parser.add_argument("--mineru-cpu-capacity", type=int, default=64)
    parser.add_argument("--mineru-max-pixels", type=int, default=602112)
    cache = ROOT / ".runtime_cache/11_mineru_2_5_pro_inference"
    parser.add_argument("--mineru-vision-cache", type=Path, default=cache / "vision_prefill_b1_fp16_9511b2e")
    parser.add_argument("--mineru-text-cache", type=Path, default=cache / "text_prefill_packed_fp16")
    parser.add_argument("--mineru-decode-cache", type=Path, default=cache / "production_increfa_real_nz_compile")
    return parser


def make_mineru(args, emit):
    import torch
    from transformers import AutoProcessor
    from mineru_vl_utils import MinerUClient
    from local_modeling_mineru import (
        LocalMinerU2_5ForConditionalGeneration, configure_decode_packed_projections,
        configure_decode_weight_format, configure_decode_rotary_impl,
        configure_decode_attention_impl, configure_decode_increfa_length_mode)
    from native_custom_backend import LocalMinerUGenerateAdapter, make_local_fixed_batch_vlm_client
    from fixed_batch_engine import ContinuousBatchDecodeEngine
    from run_local_model_two_step_extract import CompiledSingleBatchRecognitionDecoder
    from run_official_transformers_omnidocbench import apply_processor_pixel_limits
    from vision_prefill_compile import MinerUVisionPrefillRuntime
    from text_prefill_compile import MinerUPackedTextPrefillRuntime
    from mineru_adapter import MinerUAdapter

    model_dir = args.mineru_model_path
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    apply_processor_pixel_limits(processor.image_processor, min_pixels=25088, max_pixels=args.mineru_max_pixels)
    model = LocalMinerU2_5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float16, device="npu:0")
    configure_decode_packed_projections(model)
    configure_decode_weight_format(model, "decode_nz")
    configure_decode_rotary_impl(model, "npu_apply")
    configure_decode_attention_impl(model, "increfa")
    configure_decode_increfa_length_mode(model, "pse_sentinel_310p")
    model.set_vision_attention_impl("prompt_flash_attention")
    vision = MinerUVisionPrefillRuntime(model.visual, buckets=(384,512,768,1024,1536,2048,3072,4224,5632),
        cache_root=args.mineru_vision_cache, model_dir=model_dir, device=model.device, dtype=model.dtype)
    model.set_vision_prefill_runtime(vision)
    text = MinerUPackedTextPrefillRuntime(model, buckets=(128,256,512,1024), max_members=32,
        cache_root=args.mineru_text_cache, model_dir=model_dir, device=model.device, dtype=model.dtype)
    decoder = CompiledSingleBatchRecognitionDecoder(model, cache_root=args.mineru_decode_cache,
        cache_length=4096, decode_weight_format="decode_nz", decode_rotary_impl="npu_apply",
        decode_attention_impl="increfa", decode_increfa_length_mode="pse_sentinel_310p")
    engine = ContinuousBatchDecodeEngine(model, decoder, batch_size=args.mineru_batch_size,
        cache_length=4096, eos_token_id=model.config.eos_token_id, pad_token_id=model.config.pad_token_id,
        collect_prefill_metrics=True, packed_text_prefill_runtime=text, vision_pack_target=768,
        vision_lookahead=32)
    owner = MinerUClient(backend="transformers", model=LocalMinerUGenerateAdapter(model), processor=processor,
                        image_analysis=False, batch_size=args.mineru_batch_size, use_tqdm=False)
    client = make_local_fixed_batch_vlm_client(model, processor, engine, batch_size=args.mineru_batch_size,
        continuous_refill=True, system_prompt=owner.client.system_prompt,
        allow_truncated_content=owner.client.allow_truncated_content)
    return MinerUAdapter(engine, client, owner.helper, emit,
                         ready_capacity=args.mineru_ready_capacity, cpu_capacity=args.mineru_cpu_capacity)


def main():
    args = build_parser().parse_args()
    shared.main(args, routing=Routing(args.text_model, args.table_model, args.formula_model),
                factories={"mineru": make_mineru})


if __name__ == "__main__":
    main()
