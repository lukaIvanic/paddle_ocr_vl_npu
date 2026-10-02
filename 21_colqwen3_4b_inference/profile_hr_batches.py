"""Bounded warmed NPU traces of existing full-resolution page encoding paths.

Two warmup and two captured batches at each size. Uses real fixed-dev pages,
never reports profiler-distorted times as benchmark throughput. No new graphs.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from types import SimpleNamespace

import torch
from hr_batch_encoding import encode_page_batches
from optimized_prefill import text_args_for_promptfa
from patch_embedding import prepare_linear_patch_inputs
from prepared_prefill import prepare_text, finish_embeddings
from run_hr_evaluation import Execution, read_data, select_workload
from local_modeling_colqwen3 import LocalColQwen3
from pipeline_timing import Journal


def encode_single(model, processor, execution, item, args, journal):
    from PIL import Image
    row=dict(kind='page',id=item['id'],sections={})
    start=time.perf_counter()
    with journal.section(row,'preprocess'):
        with Image.open(io.BytesIO(item['image']['bytes'])) as image:
            image=image.convert('RGB')
            batch=processor.process_images([image])
        row['image_sha256']=hashlib.sha256(item['image']['bytes']).hexdigest()
    vt=int(batch['image_grid_thw'].prod());tt=int(batch['attention_mask'].sum())
    with journal.section(row,'input_transfer',device=True):
        batch={k:v.to(args.device) for k,v in batch.items()}
    with journal.section(row,'vision_prepare',vt,device=True):
        prepared=prepare_linear_patch_inputs(model,batch,execution.patch)
    vision=execution.stage('vision',prepared.vision_args[:3],row)
    with journal.section(row,'text_prepare',tt,device=True):
        text_args=text_args_for_promptfa(prepare_text(model,prepared,vision))
    hidden=execution.stage('text',text_args,row)
    with journal.section(row,'retrieval_projection',tt,device=True):
        output=finish_embeddings(model,prepared,hidden)
    with journal.section(row,'output_materialize_wait',tt):
        output=output[0].cpu()
    with journal.section(row,'validate_retain'):
        if not bool(torch.isfinite(output).all()):raise ValueError('Nonfinite embeddings')
        norms=output.float().norm(dim=-1);active=norms>0
        if not bool(active.any()) or float((norms[active]-1).abs().max())>.002:
            raise ValueError('Invalid embedding norms')
    row['wall_s']=time.perf_counter()-start
    journal.complete(row,1,1,start)
    return output


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.device='npu:0';args.max_image_tokens=None;args.page_batch_size=32
    args.output_dir.mkdir(parents=True,exist_ok=False)
    def save(name,value):(args.output_dir/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
    def progress(phase,**kw):
        value=dict(phase=phase,pid=os.getpid(),**kw);save('state.json',value);print(json.dumps(value),flush=True)
    import torch_npu
    import torch_npu.profiler as profiler
    from transformers import AutoProcessor
    progress('setup')
    torch.npu.set_device(args.device);torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format=True;torch.npu.matmul.allow_hf32=False;torch.set_num_threads(4)
    corpus,queries,qrels=read_data(args.dataset_root);corpus,_,_=select_workload(corpus,queries,qrels,'dev')
    model=LocalColQwen3.from_pretrained(args.model,device=args.device)
    processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    observer_root=args.output_dir/'observer';observer_root.mkdir()
    journal=Journal(observer_root,profile=True)
    execution=Execution(model,args,journal)
    torch.npu.synchronize()
    save('metadata.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        host=platform.node(),physical_npu=os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
        command=subprocess.list2cmdline(__import__('sys').argv),processor=processor.image_processor.to_dict(),
        batches=[1,8,32],warmup_batches=2,captured_batches=2,mode='FP16 full resolution optimized eager pages',
        scope='Profile real dev prefixes; no final partial batch or padding, no queries/scoring. Not throughput measurements.',
        dev_ids=[x['id'] for x in corpus]))
    try:
        for size in [1,8,32]:
            args.page_batch_size=size
            items=corpus[:size]
            def encode():
                if size==1:return encode_single(model,processor,execution,items[0],args,journal)
                return encode_page_batches(model,processor,execution,items,args,journal,SimpleNamespace(step=lambda:None))
            progress('warmup',batch_size=size)
            for _ in range(2):encode()
            torch.npu.synchronize()
            trace=args.output_dir/f'batch_{size}'
            progress('capture',batch_size=size)
            with profiler.profile(activities=[profiler.ProfilerActivity.CPU,profiler.ProfilerActivity.NPU],
                    record_shapes=True,with_stack=False,profile_memory=False,
                    on_trace_ready=profiler.tensorboard_trace_handler(str(trace),analyse_flag=True)) as capture:
                for _ in range(2):encode();capture.step()
                torch.npu.synchronize()
            progress('exported',batch_size=size)
    finally:
        journal.close()
    progress('completed')

if __name__=='__main__':main()
