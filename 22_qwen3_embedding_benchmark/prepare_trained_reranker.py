"""Exact checkpoint export and document-first HF parity inputs; no training."""
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

from run_english_suite import tokenize_rerank, save, digest
from suite_protocol import ENGLISH
from reranker_protocol import PREFIX, SUFFIX

CHECKPOINT_SHA256 = '509278180d2c5bdac3a2c515027c2205443be4f073c60fc39c915affee08a839'


def export(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    assert digest(args.checkpoint) == CHECKPOINT_SHA256
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert state['scheduler']['completed_updates'] == 500
    assert state['config']['student_order'] == 'document_first'
    assert json.loads((args.original/'config.json').read_text())['tie_word_embeddings']
    raw = state['model']
    assert raw['lm_head.weight'].equal(raw['embed_tokens.weight'])
    model = {('lm_head.weight' if k=='lm_head.weight' else 'model.'+k):v.contiguous()
             for k,v in raw.items() if k!='lm_head.weight'}
    expected = {}
    for file in args.original.glob('*.safetensors'):
        with safe_open(file, framework='pt', device='cpu') as f:
            expected.update({k:tuple(f.get_slice(k).get_shape()) for k in f.keys()})
    expected.pop('lm_head.weight', None)
    assert {k:tuple(v.shape) for k,v in model.items()} == expected
    assert {str(v.dtype) for v in model.values()} == {'torch.float32'}
    args.output.mkdir(parents=True, exist_ok=False)
    for p in args.original.iterdir():
        if p.is_file() and p.suffix in ('.json','.txt','.jinja') and 'safetensors' not in p.name:
            shutil.copy2(p,args.output/p.name)
    path=args.output/'model.safetensors'
    save_file(model,str(path),metadata={'format':'pt'})
    with safe_open(path, framework='pt', device='cpu') as f:
        assert set(f.keys())==set(model)
        for k,v in model.items():
            assert f.get_tensor(k).equal(v),k
    training=json.loads(args.training_result.read_text())
    assert state['dataset_sha256']==training['dataset_sha256']==digest(args.dataset)
    data=json.loads(gzip.decompress(args.dataset.read_bytes()))
    # Fixed existing examples, independent of checkpoint outcomes or judgments.
    all_groups=data['benchmark']+data['reserved_benchmark']
    workloads=[]
    for task in ENGLISH:
        groups=[g for g in all_groups if g['task']==task]
        assert len(groups)==10 and all(g['instruction']==ENGLISH[task][2] for g in groups)
        chosen=[groups[0],groups[-1]]
        pairs=[{'qid':g['qid'],'did':did,'query':g['query'],'document':doc}
               for g in chosen for did,doc in zip(g['document_ids'],g['documents'])]
        assert len(pairs)==200
        workloads.append({'task':task,'pairs':pairs,'indices':[q*100+r for q in range(2) for r in (0,1,10,99)]})
    save(args.output/'parity_examples.json',workloads)
    save(args.output/'export_manifest.json',{'checkpoint':str(args.checkpoint),'checkpoint_sha256':CHECKPOINT_SHA256,
        'update':500,'input_order':'document_first','parameter_dtype':'float32','bitwise_export_verified':True,
        'model_files':{p.name:digest(p) for p in args.output.glob('*.safetensors')},
        'dataset_sha256':state['dataset_sha256'],'tied_embedding_head_verified':True,
        'key_mapping':'local key -> model.key; tied lm_head omitted as in released checkpoint'})
    print('EXPORT_VERIFIED',json.dumps({'output':str(args.output),'parameters':sum(v.numel() for v in model.values()),'bytes':path.stat().st_size}),flush=True)


def reference(args):
    import torch
    import torch_npu
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert torch.npu.is_available()
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    workloads=json.loads((args.model/'parity_examples.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    for w in workloads:
        ids,truncated=tokenize_rerank(tok,w['task'],w['pairs'],'document_first')
        # Verify exact training prompt constructor including prefix/suffix and cut.
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'13_qwen3_reranker'))
        from training_smoke_data import body
        from transformers_rerank import PREFIX as TRAIN_PREFIX, SUFFIX as TRAIN_SUFFIX
        assert PREFIX==TRAIN_PREFIX and SUFFIX==TRAIN_SUFFIX
        prefix=tok.encode(TRAIN_PREFIX,add_special_tokens=False);suffix=tok.encode(TRAIN_SUFFIX,add_special_tokens=False)
        for p,row in zip(w['pairs'],ids):
            expected=prefix+tok.encode(body(ENGLISH[w['task']][2],p['query'],p['document'],'document_first'),add_special_tokens=False)[:8192-len(prefix)-len(suffix)]+suffix
            assert row==expected
        w['input_ids_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        w['truncated']=truncated;w['lengths']=list(map(len,ids))
    save(args.output/'workloads.json',workloads)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.float16,attn_implementation='eager').eval().to('npu:0')
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert {str(p.dtype) for p in model.parameters()}=={'torch.float16'}
    refs=[]
    no,yes=tok.convert_tokens_to_ids('no'),tok.convert_tokens_to_ids('yes')
    with torch.inference_mode():
        for w in workloads:
            ids,_=tokenize_rerank(tok,w['task'],[w['pairs'][i] for i in w['indices']],'document_first')
            scores=[];margins=[]
            for row in ids:
                x=torch.tensor([row],device='npu:0')
                z=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits[:,-1,[no,yes]]
                scores.append(float(torch.nn.functional.log_softmax(z,dim=-1)[:,1].exp()[0].cpu()))
                margins.append(float((z.float()[:,1]-z.float()[:,0])[0].cpu()))
            refs.append({'task':w['task'],'indices':w['indices'],'scores':scores,'margins':margins})
            print('HF_REFERENCE',w['task'],len(scores),flush=True)
    save(args.output/'hf_reference.json',refs)
    save(args.output/'manifest.json',{'model':str(args.model),'input_order':'document_first','dtype':'float16',
        'prefix':PREFIX,'suffix':SUFFIX,'training_prompt_token_ids_verified':True,
        'workloads_sha256':digest(args.output/'workloads.json'),
        'model_files':{p.name:digest(p) for p in sorted(args.model.iterdir()) if p.name in ('config.json','tokenizer_config.json','tokenizer.json','model.safetensors')},
        'reference_pairs':sum(len(r['scores']) for r in refs)})
    print('REFERENCE_COMPLETE',80,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    e=sub.add_parser('export')
    for n in ('checkpoint','original','dataset','training-result','output'):e.add_argument('--'+n,type=Path,required=True)
    r=sub.add_parser('reference');r.add_argument('--model',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    args=p.parse_args();(export if args.phase=='export' else reference)(args)

if __name__=='__main__':main()
