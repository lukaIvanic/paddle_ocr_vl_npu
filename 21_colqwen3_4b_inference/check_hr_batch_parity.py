"""Real mixed-orientation development-page B1/B2/B3 NPU parity gate."""
import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from batched_prefill import BatchedVisionStage, prepare_batched_images
from local_modeling_colqwen3 import LocalColQwen3
from optimized_prefill import Options, OptimizedVisionStage, OptimizedTextStage, text_args_for_promptfa
from patch_embedding import LinearPatchEmbed, prepare_linear_patch_inputs
from prepared_prefill import prepare_text, finish_embeddings
from run_hr_evaluation import read_data, select_workload


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch_npu
    from transformers import AutoProcessor
    from PIL import Image
    torch.npu.set_device('npu:0')
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format=True
    torch.npu.matmul.allow_hf32=False
    torch.set_num_threads(4)
    corpus,queries,qrels=read_data(args.dataset_root)
    corpus,_,_=select_workload(corpus,queries,qrels,'dev')
    images=[]; ids=[]; landscape=None
    for item in corpus:
        image=Image.open(io.BytesIO(item['image']['bytes'])).convert('RGB')
        if image.width>image.height and landscape is None:
            landscape=(item['id'],image)
        elif image.width<=image.height and len(images)<2:
            images.append(image);ids.append(item['id'])
        else:
            image.close()
        if len(images)==2 and landscape is not None:break
    if landscape is None:raise ValueError('Expected a landscape dev page')
    ids.insert(1,landscape[0]);images.insert(1,landscape[1])
    processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    singles=[processor.process_images([image]) for image in images]
    model=LocalColQwen3.from_pretrained(args.model,device='npu:0')
    patch=LinearPatchEmbed(model.visual.patch_embed).eval()
    single_vision=OptimizedVisionStage(model,Options()).eval()
    vision=BatchedVisionStage(model,Options()).eval()
    text=OptimizedTextStage(model,Options()).eval()
    refs=[]
    for cpu in singles:
        inputs={k:v.to('npu:0') for k,v in cpu.items()}
        prepared=prepare_linear_patch_inputs(model,inputs,patch)
        vo=single_vision(*prepared.vision_args[:3])
        output=finish_embeddings(model,prepared,text(*text_args_for_promptfa(prepare_text(model,prepared,vo))))
        refs.append(output[0].cpu())
    results=[]
    for size in [1,2,3]:
        batch=processor.process_images(images[:size])
        input_exact=all(torch.equal(v[i:i+1],singles[i][k]) for k,v in batch.items() for i in range(size))
        if not input_exact:raise AssertionError('Batched processor inputs differ from independent pages')
        batch={k:v.to('npu:0') for k,v in batch.items()}
        prepared=prepare_batched_images(model,batch,patch)
        vo=vision(*prepared.vision_args)
        output=finish_embeddings(model,prepared,text(*text_args_for_promptfa(prepare_text(model,prepared,vo)))).cpu()
        expected=torch.stack(refs[:size])
        active=expected.float().norm(dim=-1)>0
        cos=torch.nn.functional.cosine_similarity(output.float()[active],expected.float()[active],dim=-1)
        delta=(output.float()-expected.float()).abs()
        row=dict(batch_size=size,ids=ids[:size],inputs_bit_exact=input_exact,
                 max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                 cosine_min=float(cos.min()),cosine_mean=float(cos.mean()),
                 shape=list(output.shape),finite=bool(torch.isfinite(output).all()))
        row['passed']=row['finite'] and row['max_abs']<=.003 and row['cosine_min']>=.999
        results.append(row)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if not row['passed']:raise AssertionError('Real-page batching parity failed')
    for image in images:image.close()

if __name__=='__main__':main()
