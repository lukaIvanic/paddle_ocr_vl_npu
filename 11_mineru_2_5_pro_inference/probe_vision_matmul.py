#!/usr/bin/env python3
"""Synthetic FP16 projection CALIBRATION ONLY; never a model throughput result.

Exact vision shapes, F.linear including bias, ND/NZ weights, eager/compiled.
Each case is a bounded fresh process with its own telemetry and compile cache.
Only torch/torch-npu/TorchAir are imported by the worker; no model or vLLM load.
"""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
import warnings
from vision_diagnostic_runner import run_lane

SHAPES = {'qkv':(1280,3840),'proj':(1280,1280),'fc1':(1280,5120),'fc2':(5120,1280)}


def stats(values):
    ordered=sorted(values)
    def q(p):
        x=(len(ordered)-1)*p; i=int(x); j=min(i+1,len(ordered)-1)
        return ordered[i]+(ordered[j]-ordered[i])*(x-i)
    return dict(count=len(values),mean=statistics.mean(values),p50=q(.5),p90=q(.9),
                min=min(values),max=max(values))


def worker(a):
    import torch
    import torch.nn.functional as F
    import torch_npu
    root=a.output_dir;root.mkdir(parents=True,exist_ok=False)
    torch.npu.set_device('npu:0')
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = a.internal_format == 'on'
    m=a.m;k,n=SHAPES[a.projection]
    result=dict(kind='matmul_calibration',scope='synthetic calibration only; not model speed',
                shape=dict(M=m,K=k,N=n),projection=a.projection,weight_format_requested=a.weight_format,
                execution=a.execution,allow_internal_format=a.internal_format=='on',jit_compile=False,
                device=torch.npu.get_device_name(0),torch=torch.__version__,torch_npu=torch_npu.__version__,
                profile_forwards=3,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    def save(): (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    if a.weight_format=='nz' and a.internal_format=='off':
        result.update(status='unsupported_configuration',reason='NZ requested with internal formats off; no ND fallback')
        save();return
    generator=torch.Generator(device='cpu').manual_seed(310)
    # Identical bounded deterministic data in each format/execution pair.
    x=(torch.randn(m,k,generator=generator,dtype=torch.float32)*.02).half().to('npu:0')
    original=(torch.randn(n,k,generator=generator,dtype=torch.float32)*.02).half()
    weight=original.to('npu:0')
    bias=(torch.randn(n,generator=generator,dtype=torch.float32)*.02).half().to('npu:0')
    if a.weight_format=='nz':
        weight=torch_npu.npu_format_cast(weight,29)
        if int(torch_npu.get_npu_format(weight))!=29:
            raise RuntimeError('NZ descriptor not retained')
        if not torch.equal(torch_npu.npu_format_cast(weight,2).cpu(),original):
            raise RuntimeError('weight logical values changed on conversion')
    result['weight_format_actual']=int(torch_npu.get_npu_format(weight))
    def linear(x,w,b): return F.linear(x,w,b)
    # Unique code identity avoids sharing Dynamo guards across calibration cases.
    fn=types.FunctionType(linear.__code__.replace(co_name=f'mm_{m}_{k}_{n}_{a.weight_format}'),
                          linear.__globals__,argdefs=linear.__defaults__,closure=linear.__closure__)
    if a.execution=='compiled':
        try:
            import torchair
            CompilerConfig=torchair.CompilerConfig
        except ImportError:
            from torch_npu.dynamo import torchair
            from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig
        if not hasattr(torchair,'inference'):
            torchair.inference=importlib.import_module(f'{torchair.__name__}.inference')
        fn=torchair.inference.cache_compile(fn,config=CompilerConfig(),dynamic=False,
                    cache_dir=str(root/'cache'),ge_cache=True,fullgraph=True)
    with torch.inference_mode():
        start=time.perf_counter();first=fn(x,weight,bias);torch.npu.synchronize()
        result['first_call_s']=time.perf_counter()-start
        if not torch.isfinite(first).all().item():raise RuntimeError('nonfinite calibration output')
        # Small FP32 CPU slice checks calculation scale without timing CPU work.
        ref=x[:8].cpu().float() @ original[:32].float().T + bias[:32].cpu().float()
        observed=first[:8,:32].cpu().float()
        result['sample_fp32_reference_relative_l2']=float(torch.linalg.vector_norm(observed-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-12))
        for _ in range(3):fn(x,weight,bias)
        torch.npu.synchronize()
        before=int(torch._dynamo.utils.counters['stats']['unique_graphs'])
        wall=[];events=[]
        with warnings.catch_warnings(record=True) as caught:
            for _ in range(a.steps):
                torch.npu.synchronize()
                s=torch.npu.Event(enable_timing=True);e=torch.npu.Event(enable_timing=True)
                began=time.perf_counter();s.record();output=fn(x,weight,bias);e.record();e.synchronize()
                wall.append((time.perf_counter()-began)*1000);events.append(s.elapsed_time(e))
        result['timing_gate']=dict(new_graphs=int(torch._dynamo.utils.counters['stats']['unique_graphs'])-before,
                                  recompile_warnings=sum('recompiled' in str(w.message) for w in caught))
        result['timing']=dict(wall_ms=stats(wall),device_ms=stats(events),wall_samples_ms=wall,device_samples_ms=events)
        if any(result['timing_gate'].values()):
            result['status']='invalid_timing_compile_in_measurement';save();raise RuntimeError(result['status'])
        if not torch.equal(first,output):raise RuntimeError('calibration output not repeat-exact')
        import torch_npu.profiler as prof
        config=prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1,
                    aic_metrics=prof.AiCMetrics.PipeUtilization,export_type=prof.ExportType.Text)
        with prof.profile(activities=[prof.ProfilerActivity.CPU,prof.ProfilerActivity.NPU],
                schedule=prof.schedule(wait=0,warmup=0,active=3,repeat=1),record_shapes=True,
                experimental_config=config,
                on_trace_ready=prof.tensorboard_trace_handler(str(root/'profile'),analyse_flag=True)) as recording:
            for _ in range(3):
                fn(x,weight,bias);torch.npu.synchronize();recording.step()
    files=list((root/'profile').rglob('kernel_details.csv'))
    if len(files)!=1:raise RuntimeError(f'missing/ambiguous kernel CSV: {files}')
    from analyze_vision_diagnostics import bucket
    rows=list(csv.DictReader(files[0].open()))
    kernels=[r for r in rows if bucket(r)=='matmul']
    result['matmul_kernels']=[dict(type=r['Type'],block_num=r.get('Block Num'),mix_block_num=r.get('Mix Block Num'),
            accelerator_core=r.get('Accelerator Core'),duration_us=float(r['Duration(us)']),
            wait_us=r.get('Wait Time(us)'),input_shapes=r.get('Input Shapes'),input_formats=r.get('Input Formats'),
            achieved_tflops=2*m*k*n/(float(r['Duration(us)'])*1e6) if len(kernels)==3 else None) for r in kernels if float(r['Duration(us)'])>0]
    kernel_us_per_forward=sum(float(r['Duration(us)']) for r in kernels)/3
    result['matmul_kernel_us_per_forward']=kernel_us_per_forward
    result['achieved_tflops']=2*m*k*n/(kernel_us_per_forward*1e6) if kernel_us_per_forward>0 else None
    result['tflops_note']='2*M*K*N divided by summed matmul kernel duration per forward; excludes bias FLOPs. Per-call TFLOPS omitted for decomposed matmuls.'
    result['status']='completed' if kernels else 'missing_classified_matmul_kernel'
    save()
    if not kernels:raise RuntimeError(result['status'])
    print(json.dumps(result),flush=True)


def suite(a):
    root=a.output_dir;root.mkdir(parents=True,exist_ok=False)
    summary=dict(scope='CALIBRATION ONLY, never a model result',lanes=[])
    for m in map(int,a.ms.split(',')):
        for projection in a.projections.split(','):
            for execution in a.executions.split(','):
                for fmt in a.weight_formats.split(','):
                    name=f'M{m}_{projection}_{execution}_{fmt}_internal_{a.internal_format}'
                    command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--m',str(m),
                        '--projection',projection,'--execution',execution,'--weight-format',fmt,
                        '--internal-format',a.internal_format,'--steps',str(a.steps),'--output-dir',str(root/name)]
                    try:
                        receipt=run_lane(command,root/(name+'.receipt'),a.timeout_s,root/(name+'.log'))
                    except Exception:
                        summary['lanes'].append(dict(name=name,status='failed; inspect immutable receipt'))
                        (root/'summary.json').write_text(json.dumps(summary,indent=2));raise
                    summary['lanes'].append(dict(name=name,**receipt))
                    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print('MATMUL CALIBRATION complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('suite');w=sub.add_parser('worker')
    for cli in [s,w]:
        cli.add_argument('--output-dir',type=Path,required=True)
        cli.add_argument('--internal-format',choices=['on','off'],required=True)
        cli.add_argument('--steps',type=int,default=30)
    s.add_argument('--ms',default='768,3072,5632')
    s.add_argument('--projections',default='qkv,proj,fc1,fc2')
    s.add_argument('--executions',default='eager,compiled')
    s.add_argument('--weight-formats',default='nd,nz')
    s.add_argument('--timeout-s',type=int,default=1800)
    w.add_argument('--m',type=int,choices=[768,3072,5632],required=True)
    w.add_argument('--projection',choices=SHAPES,required=True)
    w.add_argument('--execution',choices=['eager','compiled'],required=True)
    w.add_argument('--weight-format',choices=['nd','nz'],required=True)
    a=p.parse_args()
    if a.steps<=0:p.error('steps must be positive')
    (suite if a.mode=='suite' else worker)(a)


if __name__=='__main__':main()
