"""Real-input optimization ladder; embedding/score drift is diagnostic, not accuracy.

This is not a ViDoRe accuracy evaluation or end-to-end throughput benchmark.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch

from local_modeling_colqwen3 import LocalColQwen3
from prepared_prefill import (PreparedVisionStage, PreparedTextStage, prepare_inputs,
    prepare_text, finish_embeddings, StageCompiler)
from optimized_prefill import (Options, OptimizedVisionStage, OptimizedTextStage,
    text_args_for_promptfa, configure_compiler, weight_formats)
from bench_prepared_prefill import emit, timed, measure, compare, load_case, cpu
from run_hf_baseline import sha256
from patch_embedding import LinearPatchEmbed,prepare_linear_patch_inputs


def patch_comparison(conv, linear, pixels, repeats):
    reference,conv_first=timed(conv,(pixels,))
    candidate,linear_first=timed(linear,(pixels,))
    for _ in range(3):
        timed(conv,(pixels,));timed(linear,(pixels,))
    samples={'conv3d':[],'linear':[]}
    calls={'conv3d':conv,'linear':linear}
    for index in range(repeats):
        for lane in (('conv3d','linear') if index%2==0 else ('linear','conv3d')):
            samples[lane].append(timed(calls[lane],(pixels,))[1])
    result={'input_shape':list(pixels.shape),'linear_weight_shape':list(linear.weight.shape),
        'comparison':compare(candidate,reference),'repeats':repeats,
        'timing_scope':'alternating synchronized warm wall time, patch projection only',
        'first_call_s':{'conv3d':conv_first,'linear':linear_first}}
    for lane,values in samples.items():
        ms=torch.tensor(values,dtype=torch.float64)*1000
        result[lane]={'samples_s':values,'mean_ms':float(ms.mean()),
                      'p50_ms':float(ms.quantile(.5)),'p90_ms':float(ms.quantile(.9)),
                      'max_ms':float(ms.max())}
    result['speedup']=result['conv3d']['mean_ms']/result['linear']['mean_ms']
    return result


def validity(value, reference):
    finite=bool(torch.isfinite(value).all())
    active=reference.float().norm(dim=-1)>0
    norms=value.float().norm(dim=-1)
    error=float(torch.where(active,(norms-1).abs(),torch.zeros_like(norms)).max())
    zeros=bool(((value==0) | active.unsqueeze(-1)).all())
    return {'finite':finite,'max_unit_norm_error':error,'zero_rows_preserved':zeros,
            'passed':finite and bool(active.any()) and error<.002 and zeros}


def memory_stats():
    return {'allocated_bytes':torch.npu.memory_allocated(),
            'reserved_bytes':torch.npu.memory_reserved(),
            'peak_allocated_bytes':torch.npu.max_memory_allocated(),
            'scope':'PyTorch allocator only; not all CANN/driver memory'}


def score_smoke(outputs):
    # Post-hoc FP32 CPU scoring of saved embeddings, not CPU model inference.
    queries=[r for r in outputs if not r['image']]
    documents=[r for r in outputs if r['image']]
    if not queries or not documents:
        raise ValueError('Score smoke requires query and image anchors')
    scores=[]
    rankings=[]
    for query in queries:
        ref_scores=[]
        candidate_scores=[]
        for document in documents:
            values=[]
            for lane in ('reference','candidate'):
                q=query[lane][0].float()
                d=document[lane][0].float()
                values.append(float((q@d.T).max(-1).values.sum()))
            delta=values[1]-values[0]
            scores.append({'query':query['name'],'document':document['name'],
                'reference':values[0],'candidate':values[1],'delta':delta,
                'relative_percent':100*delta/max(abs(values[0]),1e-12),
                'passed':abs(delta)<=.02+.001*abs(values[0])})
            ref_scores.append(values[0]);candidate_scores.append(values[1])
        ref_order=sorted(range(len(documents)),key=lambda i:ref_scores[i],reverse=True)
        new_order=sorted(range(len(documents)),key=lambda i:candidate_scores[i],reverse=True)
        rankings.append({'query':query['name'],'reference_order':ref_order,
                         'candidate_order':new_order,'same':ref_order==new_order})
    return {'scope':'small saved-anchor MaxSim comparison, not retrieval accuracy',
            'scoring':'CPU FP32','atol':.02,'rtol':.001,'scores':scores,'rankings':rankings,
            'passed':all(s['passed'] for s in scores)}


@torch.inference_mode()
def run(args,result):
    import torch_npu
    if not torch.npu.is_available():
        raise RuntimeError('NPU required')
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    # Set before model's first NPU allocation; explicit matching native control.
    internal_formats=args.enable_internal_format or args.weight_format=='fractal_nz'
    torch.npu.config.allow_internal_format=internal_formats
    options=Options(not args.unfused,args.weight_format,args.gqa,args.vision_norm)
    result.update(options=asdict(options),device_name=torch.npu.get_device_name(),
                  internal_format_requested=internal_formats,
                  versions={'torch':torch.__version__,'torch_npu':torch_npu.__version__})
    emit('model_load_start')
    model=LocalColQwen3.from_pretrained(args.model,device=args.device)
    patch_linear=LinearPatchEmbed(model.visual.patch_embed).eval()
    result['patch_embedding']=args.patch_embedding
    reference_vision,reference_text=PreparedVisionStage(model).eval(),PreparedTextStage(model).eval()
    torch.npu.synchronize()
    start=time.perf_counter()
    vision,text=OptimizedVisionStage(model,options).eval(),OptimizedTextStage(model,options).eval()
    torch.npu.synchronize()
    result['weight_setup_s']=time.perf_counter()-start
    result['setup_memory']=memory_stats()
    result['weight_formats']=weight_formats(vision,text)
    if args.weight_format=='fractal_nz' and set(result['weight_formats'])!={'29'}:
        raise RuntimeError('Not all target Linear weights are NZ')
    compiler=None if args.eager_only else StageCompiler(args.model,args.cache_root,emit)
    if compiler:
        configure_compiler(compiler,options)
        compiler.identity['internal_format']=internal_formats
    result['cache_records']=compiler.records if compiler else []
    emit('model_load_finish',weight_formats=result['weight_formats'])
    result['cases']=[]
    outputs=[]
    for index,path in enumerate(args.anchors):
        batch,anchor=load_case(path,args.row,args.device)
        name=f'{index}_{path.parent.parent.name}_{path.stem}'
        row={'name':name,'anchor':str(path),'anchor_sha256':sha256(path)}
        result['cases'].append(row)
        emit('case_start',name=name)
        expected,_=timed(lambda:model(**batch),())
        row['reference_vs_hf']=compare(expected,anchor)
        if not row['reference_vs_hf']['passed']:
            raise RuntimeError('Independent reference differs from HF anchor')
        reference_prepared=prepare_inputs(model,batch)
        prepared=(prepare_linear_patch_inputs(model,batch,patch_linear)
                  if args.patch_embedding=='linear' else reference_prepared)
        va=reference_prepared.vision_args
        if va is not None:
            length=va[0].shape[0]
            pixels=batch['pixel_values'][0,:length].contiguous()
            row['patch_projection']=patch_comparison(model.visual.patch_embed,patch_linear,pixels,args.patch_repeats)
            emit('patch_projection_measured',name=name,stats=row['patch_projection'])
        rv=reference_vision(*va) if va is not None else None
        ta=prepare_text(model,reference_prepared,rv)
        opt_va=prepared.vision_args[:3] if va is not None else None
        opt_ta=text_args_for_promptfa(ta)
        row['stages']={}
        calls={}
        for stage,module,tensors,control,control_args in (
            ('vision',vision,opt_va,reference_vision,va),
            ('text',text,opt_ta,reference_text,ta)):
            if tensors is None:
                continue
            eager,_=timed(module,tensors)
            stats={'input_shapes':[list(t.shape) for t in tensors],
                'optimized_eager_vs_manual':compare(eager,control(*control_args)),
                'manual_eager':measure(control,control_args,args.repeats),
                'optimized_eager':measure(module,tensors,args.repeats)}
            row['stages'][stage]=stats
            if compiler:
                call=compiler.get('optimized_'+stage,module,tensors)
                emit('graph_first_call_start',stage=stage,name=name)
                compiled,elapsed=timed(call,tensors)
                emit('graph_first_call_finish',stage=stage,name=name,seconds=elapsed)
                stats.update(first_call_s=elapsed,compiled_vs_own_eager=compare(compiled,eager),
                             compiled=measure(call,tensors,args.repeats))
                calls[stage]=call
            else:
                calls[stage]=module
            for lane in ('manual_eager','optimized_eager','compiled'):
                if lane in stats:
                    stats[lane]['real_tokens_per_s']=tensors[0].numel()/tensors[0].shape[-1]/(stats[lane]['mean_ms']/1000)
            emit('stage_measured',stage=stage,name=name,stats=stats)
        actual_v=calls['vision'](*opt_va) if va is not None else None
        actual_t=text_args_for_promptfa(prepare_text(model,prepared,actual_v))
        actual=finish_embeddings(model,prepared,calls['text'](*actual_t))
        if va is not None and args.patch_embedding=='linear':
            conv_v=calls['vision'](*va[:3])
            conv_t=text_args_for_promptfa(prepare_text(model,reference_prepared,conv_v))
            conv_embeddings=finish_embeddings(model,reference_prepared,calls['text'](*conv_t))
            row['linear_vs_conv3d_embeddings']=compare(actual,conv_embeddings)
        row['embeddings_vs_reference']=compare(actual,expected)
        row['validity']=validity(actual,expected)
        row['memory']=memory_stats()
        outputs.append({'name':name,'image':va is not None,'reference':cpu(expected),'candidate':cpu(actual)})
        torch.save({'inputs':{k:v.cpu() for k,v in batch.items()},**outputs[-1]},args.output_dir/f'{index}.pt')
        emit('case_finish',name=name,comparison=row['embeddings_vs_reference'],validity=row['validity'])
        (args.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if not row['validity']['passed']:
            raise RuntimeError('Nonfinite/non-normalized output')
    result['score_smoke']=score_smoke(outputs)
    emit('score_smoke',**result['score_smoke'])
    result['quality_assessment']='not_evaluated_on_vidore_v3'
    result['status']='completed_experimental'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--anchors',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,default=Path('.runtime_cache/21_colqwen3/prepared'))
    p.add_argument('--row',type=int,default=0)
    p.add_argument('--device',default='npu:0')
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--eager-only',action='store_true')
    p.add_argument('--unfused',action='store_true')
    p.add_argument('--weight-format',choices=('native','fractal_nz'),default='native')
    p.add_argument('--enable-internal-format',action='store_true')
    p.add_argument('--gqa',choices=('native','repeat'),default='repeat',
                   help='repeat follows the documented 310P contract; native is a target-specific experiment')
    p.add_argument('--vision-norm',choices=('module','manual_fp32'),default='manual_fp32')
    p.add_argument('--patch-embedding',choices=('linear','conv3d'),default='linear')
    p.add_argument('--patch-repeats',type=int,default=50)
    args=p.parse_args()
    if not args.device.startswith('npu:') or args.repeats<2 or args.patch_repeats<2 or args.row<0:
        p.error('NPU, >=2 repeats and nonnegative row required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    result={'status':'started','command':sys.argv,
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'host':platform.node(),'physical_npu':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'eager_only':args.eager_only,'scope':'exact B1 stages, not end-to-end throughput or retrieval accuracy'}
    try:
        run(args,result)
    except Exception:
        result.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        (args.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        emit('finished',status=result['status'])


if __name__=='__main__':
    main()
