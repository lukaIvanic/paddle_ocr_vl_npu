"""Locate real-page batched/independent NPU differences; diagnostic only."""
import argparse, io, json
from pathlib import Path
import torch
from batched_prefill import BatchedVisionStage, prepare_batched_images
from local_modeling_colqwen3 import LocalColQwen3
from optimized_prefill import Options, OptimizedVisionStage, OptimizedTextStage, text_args_for_promptfa
from patch_embedding import LinearPatchEmbed, prepare_linear_patch_inputs
from prepared_prefill import prepare_text, finish_embeddings
from run_hr_evaluation import read_data, select_workload


def stats(a,b):
    a=a.float();b=b.float();d=(a-b).abs()
    result=dict(shape=list(a.shape),exact=bool(torch.equal(a,b)),max_abs=float(d.max()),mean_abs=float(d.mean()))
    if a.ndim>=2:
        av=a.reshape(-1,a.shape[-1]);bv=b.reshape_as(av)
        active=(av.norm(dim=-1)>0)&(bv.norm(dim=-1)>0)
        c=torch.nn.functional.cosine_similarity(av[active],bv[active],dim=-1)
        if len(c):result.update(cos_min=float(c.min()),cos_mean=float(c.mean()),below_999=int((c<.999).sum()),vectors=len(c))
    return result

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import torch_npu
    from transformers import AutoProcessor
    from PIL import Image
    torch.npu.set_device('npu:0');torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format=True;torch.npu.matmul.allow_hf32=False;torch.set_num_threads(4)
    corpus,queries,qrels=read_data(args.dataset_root);corpus,_,_=select_workload(corpus,queries,qrels,'dev')
    images=[];ids=[]
    for cid in ('corpus-test-5','corpus-test-375'):
        item=next(x for x in corpus if x['id']==cid)
        images.append(Image.open(io.BytesIO(item['image']['bytes'])).convert('RGB'));ids.append(cid)
    processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    model=LocalColQwen3.from_pretrained(args.model,device='npu:0')
    patch=LinearPatchEmbed(model.visual.patch_embed).eval();vs=OptimizedVisionStage(model,Options()).eval()
    vb=BatchedVisionStage(model,Options()).eval();text=OptimizedTextStage(model,Options()).eval()
    singles=[]
    def cpu(xs):return tuple(x.cpu() for x in xs)
    for image in images:
        batch={k:v.to('npu:0') for k,v in processor.process_images([image]).items()}
        prepared=prepare_linear_patch_inputs(model,batch,patch)
        vo=vs(*prepared.vision_args[:3]);ta=text_args_for_promptfa(prepare_text(model,prepared,vo));h=text(*ta)
        output=finish_embeddings(model,prepared,h)
        singles.append(dict(va=cpu(prepared.vision_args[:3]),vo=cpu(vo),ta=cpu(ta),hidden=h.cpu(),output=output.cpu()))
    batch={k:v.to('npu:0') for k,v in processor.process_images(images).items()}
    prepared=prepare_batched_images(model,batch,patch);vo=vb(*prepared.vision_args)
    ta=text_args_for_promptfa(prepare_text(model,prepared,vo));h=text(*ta);output=finish_embeddings(model,prepared,h).cpu()
    rows={}
    for category,current in [('va',cpu(prepared.vision_args)),('vo',cpu(vo)),('ta',cpu(ta))]:
        rows[category]=[]
        for i,value in enumerate(current):
            ref=torch.stack([s[category][i] for s in singles]) if category=='va' else torch.cat([s[category][i] for s in singles])
            rows[category].append(stats(value,ref))
    rows['hidden']=stats(h.cpu(),torch.cat([s['hidden'] for s in singles]))
    rows['output']=stats(output,torch.cat([s['output'] for s in singles]))
    # Feed exactly the independent text-stage inputs in a batch to isolate text
    # batching from any upstream batched vision rounding.
    exact_ta=tuple(torch.cat([s['ta'][i] for s in singles]).to('npu:0') for i in range(7))
    exact_h=text(*exact_ta)
    rows['text_with_independent_inputs']=stats(exact_h.cpu(),torch.cat([s['hidden'] for s in singles]))
    rows['output_with_independent_text_inputs']=stats(finish_embeddings(model,prepared,exact_h).cpu(),torch.cat([s['output'] for s in singles]))
    args.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
