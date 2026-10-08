#!/usr/bin/env python3
"""Run the unchanged page pipeline with a vision-only PromptFA GE precision choice.

This experiment scopes the already-owned converter to vision graph lowering.
Text-prefill PromptFA and decode retain their stock converters and caches.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys


@contextmanager
def converter_scope(op, ge, install, record):
    stock_converter = op._ge_converter
    stock_ge = ge.PromptFlashAttention
    def audited_ge(*args, **kwargs):
        record(dict(event='vision_ge_promptfa', inner_precise=kwargs.get('inner_precise'),
                    input_layout=kwargs.get('input_layout'), sparse_mode=kwargs.get('sparse_mode')))
        if kwargs.get('inner_precise') != 4:
            raise RuntimeError('vision GE lowering did not request innerPrecise=4')
        return stock_ge(*args, **kwargs)
    try:
        install(4)
        ge.PromptFlashAttention = audited_ge
        yield
    finally:
        op._ge_converter = stock_converter
        ge.PromptFlashAttention = stock_ge
        if op._ge_converter is not stock_converter or ge.PromptFlashAttention is not stock_ge:
            raise RuntimeError('failed to restore stock attention converter')


def prewarm(selection_path, args, record):
    """Real crops only, same production runtime/cache; no benchmark result."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from local_modeling_mineru import LocalMinerU2_5ForConditionalGeneration
    from run_transformers_recognition_smoke import configure_npu, synchronize
    from vision_prefill_compile import MinerUVisionPrefillRuntime, parse_vision_buckets
    selection=json.loads(selection_path.read_text())
    assert selection['processor_fast'], 'production page pipeline uses the fast processor'
    cfg=selection['config']
    assert cfg['max_pixels'] == args.processor_max_pixels == 602112
    assert cfg['min_pixels'] == args.processor_min_pixels == 25088
    assert cfg['projection_impl'] == 'linear' and cfg['layer_norm_impl'] == 'manual_fp32'
    assert cfg['attention_impl'] == 'prompt_flash_attention' and cfg['allow_internal_format']
    assert cfg['promptfa_pad_head_dim_to'] == 0 and not cfg['approximate_precision']
    assert set(r['bucket'] for r in selection['selected']) == {384,512,768,1024,1536,2048,3072}
    torch.npu.config.allow_internal_format=True
    configure_npu()
    with torch.inference_mode():
        model=LocalMinerU2_5ForConditionalGeneration.from_pretrained(args.model,dtype=torch.float16,device='npu:0').eval()
        runtime=MinerUVisionPrefillRuntime(model.visual,buckets=parse_vision_buckets(args.local_vision_buckets),
            cache_root=args.local_vision_torchair_cache_dir,model_dir=args.model,device=torch.device('npu:0'),dtype=torch.float16)
        model.set_vision_attention_impl('prompt_flash_attention');model.set_vision_prefill_runtime(runtime)
        processor=AutoProcessor.from_pretrained(args.model,use_fast=True,local_files_only=True).image_processor
        processor.min_pixels,processor.max_pixels=25088,602112
        if getattr(processor,'size',None) is not None:
            processor.size['shortest_edge'],processor.size['longest_edge']=25088,602112
        for row in selection['selected']:
            path=Path(row['image'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['image_sha256']
            with Image.open(path) as image:
                inp=processor(images=[image.convert('RGB')],return_tensors='pt')
            assert int(inp.pixel_values.shape[0]) == row['real_tokens']
            assert inp.image_grid_thw.tolist() == row['grid']
            features=model.get_image_features(inp.pixel_values.to('npu:0',dtype=torch.float16),inp.image_grid_thw.to('npu:0'))
            synchronize()
            assert bool(torch.isfinite(features).all())
            record(dict(event='prewarm_crop_complete',image=row['id'],real_tokens=row['real_tokens'],bucket=row['bucket']))
        record(dict(event='prewarm_complete',runtime=runtime.metadata(),scope='cache warming only; no throughput result'))


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('--vision-inner-precise',type=int,choices=[1,4],required=True)
    p.add_argument('--precision-audit',type=Path,required=True)
    p.add_argument('--production-repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--allow-unsupported-mode4-probe',action='store_true')
    p.add_argument('--prewarm-only',type=Path,help='Real-crop selection JSON; warm production vision caches and exit without page inference')
    experiment,remaining=p.parse_known_args()
    production=experiment.production_repo.resolve()
    sys.path.insert(0,str(production/'11_mineru_2_5_pro_inference'))
    from run_page_pipeline import pipeline_args
    from run_official_transformers_omnidocbench import main as run_pipeline
    args=pipeline_args(remaining)
    experiment.precision_audit.parent.mkdir(parents=True,exist_ok=True)
    audit=experiment.precision_audit.open('x')
    def record(row):
        audit.write(json.dumps(row)+'\n');audit.flush()
        print('VISION_PRECISION '+json.dumps(row),flush=True)
    try:
        record(dict(event='requested', mode=experiment.vision_inner_precise,
            production_repo=str(production), scope='vision graphs only; stock text/decode'))
        if experiment.vision_inner_precise == 1:
            if experiment.prewarm_only:
                prewarm(experiment.prewarm_only,args,record)
            else:
                run_pipeline(args)
            record(dict(event='completed', mode=1, override_installed=False))
            return
        if args.local_vision_backend != 'torchair' or args.local_vision_attention != 'prompt_flash_attention':
            raise ValueError('mode4 requires compiled PromptFA vision')
        import torch
        import torch_npu
        # Match production before any device query can initialize the runtime.
        torch.npu.config.allow_internal_format=True
        soc=int(torch_npu.npu.get_soc_version())
        supported=200 <= soc <= 205
        if not supported and not experiment.allow_unsupported_mode4_probe:
            raise RuntimeError(f'mode4 is a 310P experiment; SOC {soc}; no fallback')
        if not supported and (args.limit > 2 or not args.input_images):
            raise ValueError('unsupported-device probe is limited to two explicit real pages')
        from vision_prefill_compile import MinerUVisionPrefillRuntime, _import_torchair
        _import_torchair()  # Establish the same torchair module aliases as production.
        stock=importlib.import_module('torchair._ge_concrete_graph.ge_converter.custom.flash_attention')
        helper_repo=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(helper_repo/'09_persistent_page_engine'))
        from scripts.vision_matmul_lab import _register_promptfa_inner_precise_converter
        helper=helper_repo/'09_persistent_page_engine/scripts/vision_matmul_lab.py'
        identity=hashlib.sha256(Path(__file__).read_bytes()+helper.read_bytes()).hexdigest()[:16]
        args.local_vision_torchair_cache_dir=Path(args.local_vision_torchair_cache_dir)/('vision_innerprecise4_'+identity)
        record(dict(event='configuration', soc=soc, device=torch_npu.npu.get_device_name(0),
            supported_310p=supported, unsupported_device_probe=not supported,allow_internal_format=True,
            helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
            vision_cache=str(args.local_vision_torchair_cache_dir),
            note='Separate vision cache prevents stock graphs from being reused for mode4. No text/decode converter override.'))
        original=MinerUVisionPrefillRuntime._compiled_for_bucket
        wrapped=set()
        def compile_selected(runtime,bucket):
            key=(id(runtime),bucket)
            if key in wrapped:
                return runtime.compiled[bucket]
            compiled=original(runtime,bucket)
            initialized=False
            def first_call_scoped(*inputs):
                nonlocal initialized
                if initialized:
                    return compiled(*inputs)
                record(dict(event='first_vision_call_start',bucket=bucket))
                op=torch.ops.npu.npu_prompt_flash_attention.default
                with converter_scope(op,stock.ge,_register_promptfa_inner_precise_converter,record):
                    output=compiled(*inputs)
                    torch_npu.npu.synchronize()
                initialized=True
                record(dict(event='first_vision_call_finish',bucket=bucket,stock_converter_restored=True))
                return output
            runtime.compiled[bucket]=first_call_scoped
            wrapped.add(key)
            return first_call_scoped
        MinerUVisionPrefillRuntime._compiled_for_bucket=compile_selected
        try:
            if experiment.prewarm_only:
                prewarm(experiment.prewarm_only,args,record)
            else:
                run_pipeline(args)
        finally:
            MinerUVisionPrefillRuntime._compiled_for_bucket=original
        record(dict(event='completed',mode=4,supported_310p=supported,
            interpretation='Successful execution on an unsupported device does not establish that its kernel used approximate arithmetic.'))
    except Exception as error:
        record(dict(event='failed',error=repr(error)))
        raise
    finally:
        audit.close()


if __name__=='__main__':main()
