"""Exact real-crop checks through the production packed vision implementation.

Selection reuses frozen live layout geometry, verifies crop hashes and processor
prompt IDs against the full reference, and covers every reachable crop bucket.
No throughput claim: CPU snapshots deliberately synchronize the device.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-grid',choices=['cpu','npu'],default='cpu')
    p.add_argument('--candidate-transfer',default='blocking')
    p.add_argument('--count',type=int,default=200)
    a=p.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from run_transformers_recognition_smoke import configure_npu
    from local_modeling_mineru import LocalMinerU2_5ForConditionalGeneration
    from fixed_batch_engine import FixedBatchDecodeEngine
    from native_custom_backend import make_local_fixed_batch_vlm_client
    from vision_prefill_compile import MinerUVisionPrefillRuntime, parse_vision_buckets
    from run_official_transformers_omnidocbench import apply_processor_pixel_limits
    from prepare_crop_cap_replay import baseline_helper
    from generation_trace import image_fingerprint
    from run_page_pipeline import pipeline_args
    sys.path.append(str(Path(__file__).resolve().parents[1]/'09_persistent_page_engine'))
    from pipeline.layout_frontend import _decode_rgb
    from pipeline.layout_postprocess import crop_layout_regions
    configure_npu();torch.npu.config.allow_internal_format=True
    ref=a.reference_run/'output'
    summary=json.loads((ref/'run_summary_shard_00.json').read_text())
    model_path=Path(summary['model'])
    cfg=json.loads((model_path/'config.json').read_text())
    prod=pipeline_args(['--dataset-json',summary['dataset_json'],'--processor-max-pixels','602112','--output-dir',str(a.output)])
    prod.local_vision_buckets=parse_vision_buckets(prod.local_vision_buckets)
    buckets=[b for b in prod.local_vision_buckets if b<=3072]
    rows=[json.loads(line) for line in (ref/'generation_trace.jsonl').open()]
    counts=Counter();chosen=[];seen=set()
    def bucket(row):return next(b for b in buckets if row['prompt_token_ids'].count(cfg['image_token_id'])*4<=b)
    for row in rows:
        b=bucket(row)
        if counts[b]<20:
            chosen.append(row);seen.add(row['request_id']);counts[b]+=1
    for row in rows:
        if len(chosen)>=a.count:break
        if row['request_id'] not in seen:chosen.append(row);seen.add(row['request_id'])
    assert len(chosen)==a.count and set(counts)==set(buckets),(len(chosen),counts,buckets)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'selection.json').write_text(json.dumps(chosen,indent=2)+'\n')
    model=LocalMinerU2_5ForConditionalGeneration.from_pretrained(model_path,dtype=torch.float16,device='npu:0').eval()
    model.set_vision_attention_impl(prod.local_vision_attention)
    runtime=MinerUVisionPrefillRuntime(model.visual,buckets=prod.local_vision_buckets,
        cache_root=a.cache_root,model_dir=model_path,device=model.device,dtype=torch.float16)
    model.set_vision_prefill_runtime(runtime)
    engine=FixedBatchDecodeEngine(model,None,batch_size=32,cache_length=4096,
        eos_token_id=model.config.eos_token_id,pad_token_id=model.config.pad_token_id,
        vision_pack_target=768,vision_lookahead=32)
    processor=AutoProcessor.from_pretrained(model_path,use_fast=True,local_files_only=True)
    apply_processor_pixel_limits(processor.image_processor,min_pixels=25088,max_pixels=602112)
    kwargs=dict(batch_size=32,system_prompt='',allow_truncated_content=False)
    base=make_local_fixed_batch_vlm_client(model,processor,engine,vision_grid_device='npu',**kwargs)
    if a.candidate_transfer!='blocking':kwargs['input_transfer']=a.candidate_transfer
    candidate=make_local_fixed_batch_vlm_client(model,processor,engine,vision_grid_device=a.candidate_grid,**kwargs)
    helper=baseline_helper();routes=Counter();checked=0

    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,(tuple,list)):return tuple(cpu(v) for v in value)
        raise TypeError(type(value))
    def equal(x,y):
        if isinstance(x,torch.Tensor):return x.dtype==y.dtype and x.shape==y.shape and torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
        return len(x)==len(y) and all(equal(i,j) for i,j in zip(x,y))
    class Recorder:
        def __init__(self):self.values=[];self.routes=Counter()
        def measure(self,name,fn,*,tags=None):
            result=fn()
            if name in ['vision_patch_embed','vision_position_prepare','vision_transformer_blocks']:
                self.values.append((name,tags,cpu(result)))
            if tags:self.routes[tags['route']]+=1
            return result
    def image(row):
        geo=json.loads((ref/'layout_regions'/f"{Path(row['page']).stem}.json").read_text())
        record=geo['blocks'][row['block_index']]
        pixels,_=_decode_rgb(Path(summary['images_dir'])/row['page'])
        crop=Image.fromarray(crop_layout_regions(pixels,[dict(coordinate=record['bbox_pixels'],
            polygon_points=record['polygon_points'],label=record['source_label'])])[0]['img'])
        assert image_fingerprint(crop)==record['crop_pixel_sha256'],row['request_id']
        crop=helper.resize_by_need(crop)
        assert image_fingerprint(crop)==row['image_sha256'],row['request_id']
        return crop
    # Exercise each single-crop route explicitly, then the unchanged production
    # packer over 32-request cohorts; no invented padding-only crop.
    singles=[next(r for r in chosen if bucket(r)==b) for b in buckets]
    single_ids={r['request_id'] for r in singles}
    rest=[r for r in chosen if r['request_id'] not in single_ids]
    cohorts=[[r] for r in singles]+[rest[i:i+32] for i in range(0,len(rest),32)]
    with torch.inference_mode():
        for cohort in cohorts:
            start=checked;images=[image(r) for r in cohort];references=None
            for label,client in [('baseline',base),('candidate',candidate)]:
                requests=[]
                for offset,(row,img) in enumerate(zip(cohort,images)):
                    inputs,pos,delta,*_=client._prepare_cpu_inputs(img,row['chat_prompt'])
                    assert inputs.input_ids[0].tolist()==row['prompt_token_ids'],row['request_id']
                    request=client._finish_generation(inputs,None,pos,delta)
                    requests.append((0,start+offset,request))
                record=Recorder()
                outputs=engine._build_group_inputs_embeds(requests,record)
                record.values.append(('final_embeddings',None,cpu(outputs)))
                if references is None:references=record;routes.update(record.routes)
                else:
                    assert len(record.values)==len(references.values)
                    for index,(x,y) in enumerate(zip(references.values,record.values)):
                        if x[:2]!=y[:2] or not equal(x[2],y[2]):
                            torch.save(dict(baseline=x,candidate=y),a.output/'mismatch.pt')
                            raise AssertionError(f'bit mismatch cohort={start} entry={index} stage={x[0]}')
            checked+=len(cohort)
            print('EXACT_CROPS',checked,flush=True)
    expected={f'bucket_{b}' for b in buckets}|{'packed_768'}
    assert expected<=set(routes),(expected,routes)
    result=dict(chip=torch.npu.get_device_name(),crops=checked,exact=True,routes=dict(routes),
        checked_fields=['hidden_states','rope_cos','rope_sin','cu_seqlens','full_32_block_encoder_output','final_embeddings'],
        candidate_grid=a.candidate_grid,candidate_transfer=a.candidate_transfer,
        buckets_above_cap_not_applicable=[b for b in prod.local_vision_buckets if b>3072],
        scope='Level 2 correctness only; same production vision grouping and graphs, real hash-verified crops')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VISION_EXACT_PASS '+json.dumps(result),flush=True)


if __name__=='__main__':main()
