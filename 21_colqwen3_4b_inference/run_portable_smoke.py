"""310P readiness and real-input eager/compiled B1 smoke; no model changes.

The default chip guard requires 310P. Explicit 910B mode validates this harness
only and must never be described as target-hardware validation.
"""
import argparse
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from run_hf_baseline import sha256
from download_hr_reference import FILES


def emit(phase, **values):
    print('PORTABLE_SMOKE '+json.dumps(dict(phase=phase, **values)), flush=True)


def verify_files(root, expected):
    actual = {}
    for name, digest in expected.items():
        path = root/name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = sha256(path)
        if actual[name] != digest:
            raise ValueError(f'Asset hash mismatch: {path}')
    return actual


def preflight(args, result):
    import torch
    import torch_npu
    if not torch.npu.is_available():
        raise RuntimeError('NPU required; no fallback')
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')
    if not visible or not visible.isdecimal() or torch.npu.device_count() != 1:
        raise ValueError('Select exactly one physical NPU with ASCEND_RT_VISIBLE_DEVICES')
    torch.npu.set_device(args.device)
    name = torch.npu.get_device_name()
    result.update(device=name, physical_npu=visible, python=sys.executable)
    if args.expected_chip.upper() not in name.upper():
        raise ValueError(f'Expected {args.expected_chip}, got {name}')
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(4)
    result['versions'] = {}
    for module in ('torch', 'torch_npu', 'transformers', 'tokenizers', 'huggingface_hub',
                   'pyarrow', 'numpy', 'PIL', 'safetensors', 'torchvision', 'pytrec_eval'):
        value = importlib.import_module(module)
        result['versions'][module] = getattr(value, '__version__', 'imported')
    if result['versions']['transformers'] != '4.57.1':
        raise RuntimeError('Use the validated Transformers 4.57.1 processor environment; do not patch checkpoint code')
    try:
        air = importlib.import_module('torchair')
    except ImportError:
        air = importlib.import_module('torch_npu.dynamo.torchair')
    inference = importlib.import_module(air.__name__+'.inference')
    if not callable(getattr(inference, 'cache_compile', None)):
        raise RuntimeError('Installed TorchAir lacks inference.cache_compile')
    if not hasattr(torch_npu, 'npu_prompt_flash_attention') or not hasattr(torch.ops.npu, 'npu_gelu'):
        raise RuntimeError('Required PromptFA/GELU APIs are missing')
    result['versions']['torchair'] = getattr(air, '__version__', 'imported')
    config = air.CompilerConfig()
    config.fusion_config.fusion_switch_file = str(Path(__file__).with_name('compile_fusion_switch.json'))
    expected = json.loads(Path(__file__).with_name('310p_assets.json').read_text())['files']
    emit('asset_verification_start')
    result['model_hashes'] = verify_files(args.model, expected)
    result['dataset_hashes'] = verify_files(args.dataset_root, FILES)
    from config import ColQwenConfig
    ColQwenConfig.from_model_dir(args.model)
    try:
        free, total = torch.npu.mem_get_info()
        result['device_memory_bytes'] = dict(free=free, total=total)
    except (AttributeError, RuntimeError) as error:
        result['device_memory_query_unavailable'] = str(error)
    health = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=False)
    (args.output_dir/'npu-smi.txt').write_text(health.stdout+health.stderr)
    if health.returncode:
        raise RuntimeError('npu-smi failed; verify the environment and device before inference')
    result['preflight_s'] = time.perf_counter()-args.started
    emit('preflight_passed', device=name, physical_npu=visible, versions=result['versions'],
         memory=result.get('device_memory_bytes'), target_310p=args.expected_chip=='310P')


def smoke(args, result):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from local_modeling_colqwen3 import LocalColQwen3
    from optimized_prefill import (Options, OptimizedVisionStage, OptimizedTextStage,
        configure_compiler, text_args_for_promptfa, prepare_310p_text_inputs)
    from prepared_prefill import StageCompiler, prepare_inputs, prepare_text, finish_embeddings
    from patch_embedding import LinearPatchEmbed, prepare_linear_patch_inputs
    from text_forward_variants import PreparedTextForward
    from profile_warm_forward import Forward, measure
    from bench_prepared_prefill import compare
    from bench_optimized_prefill import memory_stats
    from run_hr_evaluation import read_data

    class QueryForward:
        def __init__(self, model, batch, text, aligned):
            self.model, self.batch, self.text, self.aligned = model, batch, text, aligned

        def __call__(self):
            prepared = prepare_inputs(self.model, self.batch)
            tensors = text_args_for_promptfa(prepare_text(self.model, prepared, None))
            length = tensors[0].shape[1]
            if self.aligned:
                tensors = prepare_310p_text_inputs(*tensors)
            hidden = self.text(*tensors)[:, :length].contiguous()
            return finish_embeddings(self.model, prepared, hidden)

    with torch.inference_mode():
        corpus, queries, _ = read_data(args.dataset_root)
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
        emit('model_load_start')
        model = LocalColQwen3.from_pretrained(args.model, device=args.device)
        options = Options()
        patch = LinearPatchEmbed(model.visual.patch_embed).eval()
        vision, text = OptimizedVisionStage(model, options).eval(), OptimizedTextStage(model, options).eval()
        compiler = StageCompiler(args.model, args.cache_root, emit)
        configure_compiler(compiler, options)
        compiler.identity['internal_format'] = True
        result['memory_after_model'] = memory_stats()
        result['cases'] = []
        # A real small query first, then two distinct full-resolution HR pages.
        for kind, index in [('query', 0), ('page', 5), ('page', 0)]:
            item = queries[index] if kind == 'query' else corpus[index]
            emit('case_start', kind=kind, id=item['id'])
            if kind == 'query':
                inputs = processor.process_queries([item['text']])
            else:
                with Image.open(io.BytesIO(item['image']['bytes'])) as image:
                    inputs = processor.process_images([image.convert('RGB')])
            batch = {k:v.to(args.device) for k,v in inputs.items()}
            row = dict(kind=kind, id=item['id'], input_shapes={k:list(v.shape) for k,v in batch.items()})
            eager = QueryForward(model,batch,text,False) if kind=='query' else Forward(model,batch,patch,vision,text)
            expected = eager().cpu()
            started = time.perf_counter()
            if kind == 'query':
                prepared = prepare_inputs(model,batch)
                tensors = prepare_310p_text_inputs(*text_args_for_promptfa(prepare_text(model,prepared,None)))
                call = compiler.get('optimized_query_text_aligned',PreparedTextForward(text),tensors)
                compiled = QueryForward(model,batch,call,True)
            else:
                prepared = prepare_linear_patch_inputs(model,batch,patch)
                vc = compiler.get('optimized_vision',vision,prepared.vision_args[:3])
                outputs = vc(*prepared.vision_args[:3])
                tensors = text_args_for_promptfa(prepare_text(model,prepared,outputs))
                tc = compiler.get('optimized_text',text,tensors)
                compiled = Forward(model,batch,patch,vc,tc)
            actual = compiled()
            torch.npu.synchronize()
            row['compile_and_first_forward_s'] = time.perf_counter()-started
            row['compiled_vs_same_optimized_eager'] = compare(actual,expected)
            if not row['compiled_vs_same_optimized_eager']['passed']:
                result['cases'].append(row)
                raise RuntimeError('Same-implementation compiled/eager check failed; retain diagnostics and investigate')
            for name, fn in [('raw_eager',eager),('torchair',compiled)]:
                for _ in range(args.warmups):
                    fn()
                row[name], output = measure(fn,args.repeats)
                row[name]['forwards_per_second'] = 1000/row[name]['wall_ms']['mean']
                norms = output.float().norm(dim=-1)
                if not bool(torch.isfinite(output).all()) or float((norms-1).abs().max())>.002:
                    raise RuntimeError('Nonfinite or non-unit output')
                row[name]['replay_check'] = compare(output,expected)
                if not row[name]['replay_check']['passed']:
                    raise RuntimeError('Warm replay changed output')
            row['memory'] = memory_stats()
            torch.save(dict(inputs=inputs, eager=expected, compiled=output.cpu()),
                       args.output_dir/f'{kind}_{index}_embeddings.pt')
            result['cases'].append(row)
            emit('case_passed', **row)
        result['cache_records'] = compiler.records
        result['timing_scope'] = 'Warmed complete model forwards from processed NPU inputs to NPU embeddings; excludes preprocessing, external transfers, compile, validation and serialization.'
        result['quality_scope'] = 'Same-implementation eager/compiled integration checks, not HF-reference or retrieval-quality acceptance. Full HR scoring is the quality evaluation.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight','smoke'), default='smoke')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--device', choices=('npu:0',), default='npu:0')
    parser.add_argument('--expected-chip', choices=('310P','910B'), default='310P')
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.warmups<1 or args.repeats<1:
        parser.error('Warmups and repeats must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.started = time.perf_counter()
    result = dict(status='started', command=sys.argv, host=platform.node(),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  expected_chip=args.expected_chip, model=str(args.model.resolve()),
                  dataset_root=str(args.dataset_root.resolve()), cache_root=str(args.cache_root.resolve()))
    try:
        preflight(args,result)
        if args.phase=='smoke':
            smoke(args,result)
        result['status'] = 'passed'
    except Exception:
        result.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        result['total_s'] = time.perf_counter()-args.started
        (args.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        emit('finish',status=result['status'],output=str(args.output_dir/'result.json'))


if __name__=='__main__':
    main()
