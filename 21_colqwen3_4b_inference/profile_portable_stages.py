"""Profile the validated B1 HR page path, with no modeling or quality-gate changes.

Clean measurements are separate from profiler captures. Full means processed NPU
inputs through final NPU embeddings, not file-to-embedding throughput. Vision and
text use frozen prepared inputs; text retains production graph padding/trimming.
Both execution lanes prepare isolated text inputs from compiled vision, so that
eager/compiled text measurements use the same production preparation route.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
import traceback

import torch

from bench_prepared_prefill import compare, cpu
from forward_profile_analysis import distribution
from local_modeling_colqwen3 import LocalColQwen3
from optimized_prefill import Options, OptimizedVisionStage, OptimizedTextStage, configure_compiler, text_args_for_promptfa
from patch_embedding import LinearPatchEmbed, prepare_linear_patch_inputs
from prepared_prefill import StageCompiler, prepare_text
from profile_warm_forward import Forward, measure
from run_hf_baseline import sha256
from run_hr_evaluation import read_data
from run_portable_smoke import verify_files


def emit(phase, **data):
    print('PORTABLE_PROFILE '+json.dumps(dict(phase=phase,**data)),flush=True)


class Progress:
    def __init__(self,root):
        self.root=root
        self.phase='initializing'
        self.since=time.monotonic()
        self.done=threading.Event()
        self.thread=threading.Thread(target=self.loop,daemon=True)
        self.thread.start()

    def set(self,phase):
        self.phase,self.since=phase,time.monotonic()
        emit(phase)

    def loop(self):
        while not self.done.wait(10):
            state=dict(active=self.phase,active_seconds=time.monotonic()-self.since)
            emit('heartbeat',**state)
            path=self.root/'progress.json.tmp'
            path.write_text(json.dumps(state)+'\n')
            path.replace(self.root/'progress.json')

    def close(self):
        self.done.set()
        self.thread.join()


class Stage:
    def __init__(self,name,call,tensors):
        self.name,self.call,self.tensors=name,call,tensors
        self.annotate=False

    def __call__(self):
        context=torch.profiler.record_function('colqwen.'+self.name) if self.annotate else nullcontext()
        with context:
            return self.call(*self.tensors)


def exact_replay(actual,expected):
    a,b=(actual,expected) if isinstance(actual,tuple) else ((actual,),(expected,))
    return dict(exact=len(a)==len(b) and all(torch.equal(x,y) for x,y in zip(a,b)),
                finite=all(bool(torch.isfinite(x).all()) for x in a))


def tensor_hash(tensors):
    digest=hashlib.sha256()
    for t in tensors:
        a=t.detach().cpu().contiguous()
        digest.update(str((a.shape,a.dtype)).encode())
        digest.update(a.numpy().tobytes())
    return digest.hexdigest()


def projection_map(vision,text,model):
    rows=[]
    for prefix,root in [('vision',vision),('text',text),('mergers',model.visual.merger),
                        ('deep_mergers',model.visual.deepstack_merger_list),('retrieval',model.custom_text_proj)]:
        for name,module in root.named_modules():
            w=getattr(module,'weight',None)
            if w is None or w.ndim!=2:
                continue
            role=re.sub(r'\.\d+(?=\.|$)', '.*', prefix+'.'+name).rstrip('.')
            row=dict(role=role,in_features=w.shape[1],out_features=w.shape[0],
                     forward_source=inspect.getsourcefile(type(module).forward),
                     forward_line=inspect.getsourcelines(type(module).forward)[1])
            if row not in rows:
                rows.append(row)
    return rows


def capture(fn,args,scope,metric,expected):
    import torch_npu.profiler as p
    directory=Path('profiles')/scope/metric
    dest=args.output_dir/directory
    dest.mkdir(parents=True,exist_ok=False)
    counter={'pipe':p.AiCMetrics.PipeUtilization,'memory':p.AiCMetrics.Memory,
             'basic':p.AiCMetrics.AiCoreNone}[metric]
    config=p._ExperimentalConfig(profiler_level=p.ProfilerLevel.Level0 if metric=='basic' else p.ProfilerLevel.Level1,
                                  aic_metrics=counter,export_type=p.ExportType.Text)
    times=[]
    fn.annotate=True
    try:
        with p.profile(activities=[p.ProfilerActivity.CPU,p.ProfilerActivity.NPU],
                schedule=p.schedule(wait=0,warmup=1,active=args.profile_steps,repeat=1),
                record_shapes=True,with_stack=True,profile_memory=False,experimental_config=config,
                on_trace_ready=p.tensorboard_trace_handler(str(dest/'raw'),analyse_flag=True)) as prof:
            for step in range(args.profile_steps+1):
                torch.npu.synchronize()
                begin=time.perf_counter()
                with torch.profiler.record_function('colqwen.profile.'+scope):
                    output=fn()
                    torch.npu.synchronize()
                if step:
                    times.append((time.perf_counter()-begin)*1000)
                prof.step()
    finally:
        fn.annotate=False
    replay=exact_replay(cpu(output),expected)
    if not replay['exact'] or not replay['finite']:
        raise RuntimeError('Profiling changed same-lane replay; preserve captures and investigate')
    return dict(directory=str(directory),steps=args.profile_steps,metric=metric,
                profiled_wall_ms=distribution(times),replay=replay)


@torch.inference_mode()
def run(args,r,progress):
    import torch_npu
    import transformers
    from PIL import Image
    if not os.getenv('ASCEND_RT_VISIBLE_DEVICES','').isdecimal() or torch.npu.device_count()!=1:
        raise RuntimeError('Select exactly one physical NPU before running')
    torch.npu.set_device('npu:0')
    r.update(device=torch.npu.get_device_name(),torch=torch.__version__,torch_npu=torch_npu.__version__,
             transformers=transformers.__version__,python=sys.executable)
    if args.expected_chip not in r['device'].upper():
        raise RuntimeError(f'Expected {args.expected_chip}, got {r["device"]}')
    if transformers.__version__!='4.57.1':
        raise RuntimeError('Reuse the successful ColQwen Transformers 4.57.1 environment')
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format=True
    torch.npu.matmul.allow_hf32=False
    torch.set_num_threads(4)
    progress.set('verify_assets')
    r['model_hashes']=verify_files(args.model,json.loads(Path(__file__).with_name('310p_assets.json').read_text())['files'])
    corpus,_,_=read_data(args.dataset_root)
    item=corpus[args.page_index]
    r['page']=dict(index=args.page_index,id=item['id'],image_sha256=hashlib.sha256(item['image']['bytes']).hexdigest())
    progress.set('load_model_and_processor')
    model=LocalColQwen3.from_pretrained(args.model,device='npu:0')
    processor=transformers.AutoProcessor.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    with Image.open(io.BytesIO(item['image']['bytes'])) as image:
        inputs=processor.process_images([image.convert('RGB')])
    batch={k:v.to('npu:0') for k,v in inputs.items()}
    options=Options()
    r['options']=asdict(options)
    patch=LinearPatchEmbed(model.visual.patch_embed).eval()
    vision,text=OptimizedVisionStage(model,options).eval(),OptimizedTextStage(model,options).eval()
    r['model_map']=projection_map(vision,text,model)
    prepared=prepare_linear_patch_inputs(model,batch,patch)
    va=prepared.vision_args[:3]
    compiler=StageCompiler(args.model,args.cache_root,emit)
    configure_compiler(compiler,options)
    compiler.identity['internal_format']=True
    progress.set('load_or_compile_vision')
    vc=compiler.get('optimized_vision',vision,va)
    visual=vc(*va)
    ta=text_args_for_promptfa(prepare_text(model,prepared,visual))
    progress.set('load_or_compile_text')
    tc=compiler.get('optimized_text',text,ta)
    tc(*ta)
    torch.npu.synchronize()
    r['cache_records']=compiler.records
    r['input_contract']=dict(batch_shapes={k:list(v.shape) for k,v in batch.items()},
        vision_shapes=[list(t.shape) for t in va],text_shapes=[list(t.shape) for t in ta],
        vision_sha256=tensor_hash(va),text_sha256=tensor_hash(ta),
        frozen_text_source='compiled vision; production text preparation; no prealignment substitution')
    eager=Forward(model,batch,patch,vision,text)
    compiled=Forward(model,batch,patch,vc,tc)
    progress.set('same_implementation_diagnostic')
    r['compiled_vs_eager']=compare(compiled(),eager())
    # Accuracy has already been accepted through full HR. Record embedding
    # differences, but do not impose an uncalibrated cross-lane quality gate.
    calls=dict(full=compiled if args.execution=='torchair' else eager,
               vision=Stage('vision',vc if args.execution=='torchair' else vision,va),
               text=Stage('text',tc if args.execution=='torchair' else text,ta))
    r['scopes']={}
    for scope in args.scopes:
        fn=calls[scope]
        progress.set(scope+'_warmup_and_clean_before')
        for _ in range(args.warmups):
            fn()
        before,output=measure(fn,args.repeats)
        expected=cpu(output)
        if not exact_replay(expected,expected)['finite']:
            raise RuntimeError('Nonfinite clean output')
        row=dict(before=before,profiles={})
        r['scopes'][scope]=row
        emit('clean_before',scope=scope,wall_ms=before['wall_ms'])
        for metric in args.metrics:
            progress.set(scope+'_profile_'+metric+'_and_export')
            row['profiles'][metric]=capture(fn,args,scope,metric,expected)
            (args.output_dir/'result.json').write_text(json.dumps(r,indent=2)+'\n')
            emit('profile_exported',scope=scope,metric=metric)
        progress.set(scope+'_clean_after')
        row['after'],output=measure(fn,args.repeats)
        row['final_replay']=exact_replay(cpu(output),expected)
        if not row['final_replay']['exact'] or not row['final_replay']['finite']:
            raise RuntimeError('Same-lane output changed during replay')
        emit('clean_after',scope=scope,wall_ms=row['after']['wall_ms'])
    r['status']='completed'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--execution',choices=('torchair','raw_eager'),default='torchair')
    p.add_argument('--expected-chip',choices=('310P','910B'),default='310P')
    p.add_argument('--page-index',type=int,default=5)
    p.add_argument('--scopes',nargs='+',choices=('full','vision','text'),default=['full','vision','text'])
    p.add_argument('--metrics',nargs='+',choices=('pipe','memory','basic'),default=['pipe','memory'])
    p.add_argument('--warmups',type=int,default=5)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--profile-steps',type=int,default=3)
    args=p.parse_args()
    if not 0<=args.page_index<1110 or min(args.warmups,args.repeats,args.profile_steps)<1:
        p.error('Valid HR page index and positive measurement counts required')
    if len(set(args.scopes))!=len(args.scopes) or len(set(args.metrics))!=len(args.metrics):
        p.error('No duplicate scopes or metrics')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    r=dict(status='started',command=sys.argv,host=platform.node(),execution=args.execution,
        physical_npu=os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source={n:sha256(Path(__file__).with_name(n)) for n in
            ('profile_portable_stages.py','local_modeling_colqwen3.py','optimized_prefill.py','prepared_prefill.py','patch_embedding.py')},
        scope_definitions=dict(full='NPU inputs to NPU embeddings; preparation/mergers/projection included; file/preprocessing/transfers excluded',
            vision='Exact optimized_vision graph, frozen patch/position inputs; no patch embedding or mergers',
            text='Exact optimized_text page graph, frozen seven inputs; includes production 1274-to-1280 padding and output trim; no mergers or retrieval projection'))
    progress=Progress(args.output_dir)
    try:
        run(args,r,progress)
    except Exception:
        r.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        progress.close()
        (args.output_dir/'result.json').write_text(json.dumps(r,indent=2)+'\n')
        emit('finished',status=r['status'])


if __name__=='__main__':
    main()
