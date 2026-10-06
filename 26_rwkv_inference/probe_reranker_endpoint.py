"""Validate right-padding endpoint states, then separate static backbone/head graphs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch
from local_modeling_rwkv_embedding import Embedding
from local_modeling_rwkv_reranker import Reranker
from probe_wkv7 import load_bridge, make_inputs, reference, compare
from run_cpu_reference import sha256
from run_nanoscidocs import save
from run_reranker_smoke import require, metrics
from wkv7_endpoint import load_endpoint, register_converter


class Backbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model=model
    def forward(self, ids, lengths):
        _,state,_=self.model.encode_states(ids,valid_lengths=lengths)
        return state


class Recurrence(torch.nn.Module):
    def forward(self,k,v,w,r,a,b,hi,lengths):
        return torch.ops.rwkv_endpoint.wkv7.default(k,v,w,r,a,b,hi,lengths)


def compiled(call, cache):
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    config=CompilerConfig()
    config.debug.graph_dump.type='pbtxt'
    config.debug.graph_dump.path=str(cache.parent/'graphs')
    return torchair.inference.cache_compile(call,config=config,dynamic=False,fullgraph=True,
        cache_dir=str(cache),ge_cache=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['build-root','reference-build','cases-root','checkpoint','reranker','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['recurrence','model'],required=True)
    p.add_argument('--backend',choices=['raw_eager','torchair'],default='raw_eager')
    p.add_argument('--bucket',type=int,choices=[256,512,1024,2048],default=512)
    p.add_argument('--dtype',choices=['fp16','fp32'],default='fp16')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();root=Path(__file__).parent
    report={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{n:sha256(root/n) for n in ['probe_reranker_endpoint.py','wkv7_endpoint.py',
            'local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py','data/wkv7_endpoint_sources.json']},
        'phase':args.phase,'backend':args.backend,'bucket':args.bucket,'dtype':args.dtype,
        'physical_npu':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),'shared_device':True,
        'contract':'Right-padded fixed shape; int32 device valid length includes EOS; extra right padding excluded from recurrent state',
        'recurrence_tolerance':{'atol':2e-5,'rtol':2e-4},
        'model_state_tolerance':{'atol':.02,'rtol':.005},
        'model_logit_tolerance':{'atol':.02,'rtol':.005},
        'cases':[],'all_checks_passed':False}
    try:
        import torch_npu
        torch.set_num_threads(4)
        status=subprocess.check_output(['/usr/local/bin/npu-status'],text=True)
        selected=next(x for x in status.splitlines() if x.startswith('NPU '+report['physical_npu']+': '))
        assert 'Health=OK' in selected
        torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False);torch.npu.config.allow_internal_format=False
        free,total=torch.npu.mem_get_info();assert free>2*1024**3
        report.update(device=torch.npu.get_device_name(0),torch=torch.__version__,torch_npu=torch_npu.__version__,hbm_before={'free_bytes':free,'total_bytes':total})
        assert torch.equal((torch.ones((64,64),device='npu')@torch.ones((64,64),device='npu')).cpu(),torch.full((64,64),64.))
        load_bridge(root/'wkv7_npu',args.reference_build);load_endpoint(args.build_root)
        report['package_manifest']=json.loads(Path(str(args.build_root)+'_source/manifest.json').read_text())
        api=args.build_root/'install/vendors/rwkv_endpoint/op_api/lib/libcust_opapi.so'
        report['op_api_sha256']=sha256(api)
        symbols=subprocess.check_output(['nm','-D',str(api)],text=True)
        assert 'aclnnRwkvEndpointWkv7' in symbols and 'aclnnRwkvReferenceWkv7' not in symbols
        report['independent_api_symbols_verified']=True
        register_converter()
        with torch.inference_mode():
            if args.phase=='recurrence':
                op=torch.ops.rwkv_endpoint.wkv7.default
                if args.backend=='torchair':op=compiled(Recurrence().forward,args.output/'recurrence_cache')
                inputs=[]
                shapes=[(1,1),(1,47),(1,48),(1,49),(2,128),(2,512),(1,2048)] if args.backend=='raw_eager' else [(2,128)]
                for batch,T in shapes:
                    for nonzero in [False,True]:
                        cpu=make_inputs(batch,T,nonzero,20261006+T)
                        dev=[x.to('npu') for x in cpu]
                        lengths=torch.empty((batch,),dtype=torch.int32,device='npu')
                        counts=sorted(set([0,1,min(47,T),min(48,T),min(49,T),T-1,T]))
                        if args.backend=='torchair':counts=counts+[1,T] # Same graph/buffers, changed endpoint then repeat.
                        for n in counts:
                            valid=[n] if batch==1 else [n,T-n]
                            lengths.copy_(torch.tensor(valid,dtype=torch.int32))
                            before=time.perf_counter();out,state=op(*dev,lengths);torch.npu.synchronize()
                            expected_out=torch.zeros_like(cpu[0]);expected_state=cpu[-1].clone()
                            for b,L in enumerate(valid):
                                if L:
                                    eo,es=reference([x[b:b+1,:,:L].contiguous() for x in cpu[:6]]+[cpu[-1][b:b+1]])
                                    expected_out[b:b+1,:,:L]=eo;expected_state[b:b+1]=es
                            parity=[compare(out.cpu(),expected_out),compare(state.cpu(),expected_state)]
                            assert all(x['allclose'] for x in parity),parity
                            direct=torch.ops.rwkv_endpoint.wkv7.default(*dev,lengths)
                            assert all(torch.equal(x,y) for x,y in zip((out,state),direct)), 'Compiled/eager or repeat mismatch'
                            assert all(torch.equal(x.cpu(),y) for x,y in zip(dev,cpu)), 'Input mutation'
                            row={'batch':batch,'T':T,'valid_lengths':valid,'nonzero_initial_state':nonzero,
                                 'comparisons':parity,'call_seconds':time.perf_counter()-before,'repeat_bitwise_equal':True}
                            report['cases'].append(row);inputs.append({'batch':batch,'T':T,'valid_lengths':valid,'seed':20261006+T,'nonzero_initial_state':nonzero})
                            print('RECURRENCE',json.dumps(row),flush=True)
                save(args.output/'inputs.json',inputs)
            else:
                folder=args.cases_root/f'reranker_b1_t{args.bucket}_cold_ea9aa394/probe'
                cases=json.loads((folder/'inputs.json').read_text())
                assert json.loads((folder/'result.json').read_text())['all_checks_passed']
                save(args.output/'inputs.json',cases)
                dtype={'fp16':torch.float16,'fp32':torch.float32}[args.dtype]
                model=Embedding(args.checkpoint,'npu:0',dtype);ranker=Reranker(args.reranker,'npu:0',dtype)
                report['checkpoint_sha256']={n:sha256(getattr(args,n)) for n in ['checkpoint','reranker']}
                backbone=Backbone(model).eval(); encode=backbone.forward; head=ranker.forward
                if args.backend=='torchair':
                    encode=compiled(encode,args.output/'backbone_cache')
                    head=compiled(head,args.output/'head_cache')
                ids=torch.empty((1,args.bucket),dtype=torch.long,device='npu')
                lengths=torch.empty((1,),dtype=torch.int32,device='npu')
                first_score=None
                for i,case in enumerate(cases+[cases[0]]):
                    raw=torch.tensor([case['input_ids']],dtype=torch.long,device='npu');L=raw.shape[1]
                    _,expected,_=model.encode_states(raw);expected_logit=ranker(expected[1])
                    ids.copy_(torch.tensor([case['input_ids']+[0]*(args.bucket-L)],dtype=torch.long))
                    lengths.fill_(L)
                    before=time.perf_counter();eager=backbone(ids,lengths);torch.npu.synchronize()
                    state_checks=[require(a,b,.02,.005) for a,b in zip(eager,expected)]
                    eager_logit=ranker(eager[1]);score_check=require(eager_logit,expected_logit,.02,.005)
                    print('MODEL_GRAPH_START',json.dumps({'T':args.bucket,'valid_length':L,'backend':args.backend}),flush=True)
                    actual=encode(ids,lengths);logit=head(actual[1]);torch.npu.synchronize()
                    assert all(torch.equal(a,b) for a,b in zip(actual,eager)), 'Compiled backbone differs from eager'
                    assert torch.equal(logit,eager_logit), 'Compiled head differs from eager'
                    assert torch.equal(head(encode(ids,lengths)[1]),logit), 'Repeat-call isolation'
                    if i==0:first_score=logit.clone()
                    if i==len(cases):assert torch.equal(logit,first_score), 'A/B/A device input replay'
                    row={'query_id':case['query_id'],'document_id':case['document_id'],'T':args.bucket,'valid_tokens':L,
                        'right_padding_tokens':args.bucket-L,'raw_logit':float(expected_logit.item()),'endpoint_logit':float(logit.item()),
                        'logit_delta':float((logit-expected_logit).item()),'state_parity':state_checks,'logit_parity':score_check,
                        'compiled_vs_eager_bitwise_equal':True,'call_seconds':time.perf_counter()-before,'repeat_first_case':i==len(cases)}
                    report['cases'].append(row);print('MODEL_CASE',json.dumps(row),flush=True)
                # Raw default path must still agree with the earlier saved unpadded scores.
                old=json.loads((folder/'result.json').read_text())
                for row in report['cases']:
                    prev=next(c for c in old['cases'] if c['query_id']==row['query_id'] and c['document_id']==row['document_id'])
                    require(torch.tensor(row['raw_logit']),torch.tensor(prev['raw_logit']),.02,.005)
                report['default_path_regression_passed']=True
            if args.backend=='torchair':
                graphs=list((args.output/'graphs').rglob('*.pbtxt'))
                assert graphs and any('RwkvEndpointWkv7' in f.read_text() for f in graphs), 'Missing distinct GE operator'
                report['distinct_GE_op_verified']=True
                report['graph_artifacts']=[{'path':str(f),'sha256':sha256(f)} for f in graphs]
                report['static_shapes_with_device_length_changes_passed']=True
            report['all_checks_passed']=True
    finally:
        report['total_seconds']=time.perf_counter()-start;save(args.output/'result.json',report)
        print('SUMMARY',json.dumps({'passed':report['all_checks_passed'],'cases':len(report['cases']),'seconds':report['total_seconds']}),flush=True)


if __name__=='__main__':main()
