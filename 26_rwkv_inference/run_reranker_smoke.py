"""Small independent CPU / eager NPU parity gate for the released 90M reranker."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import time

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from local_modeling_rwkv_embedding import Embedding
from local_modeling_rwkv_reranker import Reranker, CHECKPOINT_SHA256 as RERANKER_SHA256
from probe_wkv7 import load_bridge
from run_cpu_reference import CReference, CHECKPOINT_SHA256, sha256

UPSTREAM_HASHES = {'model.py': '1b062949606e7ab94aa0cb9ce473ee649750a6db9c6cc592e16d2ff295ac5696',
                  'rwkv7.py': 'e1e578e7f95bc811ac898008e617392fc8a001990b5c7bac178ca77f8d6800c0'}
PREFIX = 'Instruct: Given a query, retrieve documents that answer the query\nDocument: {document}\n'
SUFFIX = 'Query: {query}'


def cpu_recurrence(state, r, w, k, v, a, b):
    """Replace only upstream's CUDA batch call; inputs w are sigmoid-decay values."""
    B, T, C = r.shape
    H = C // 64
    r, w, k, v, a, b = [x.reshape(B, T, H, 64).float() for x in (r,w,k,v,a,b)]
    s, rows = state.float().clone(), []
    decay = (-.6065306597126334 * w).exp()
    for t in range(T):
        s = (s * decay[:, t, :, None, :] +
             (s @ a[:, t, :, :, None]) @ b[:, t, :, None, :] +
             v[:, t, :, :, None] @ k[:, t, :, None, :])
        rows.append((s @ r[:, t, :, :, None]).squeeze(-1))
    state.copy_(s)
    return torch.stack(rows, dim=1).reshape(B,T,C)


def reference_namespace(path, names, cpu_devices=False):
    """Execute only hash-pinned math definitions, excluding imports/CUDA builds/training."""
    if sha256(path) != UPSTREAM_HASHES[path.name]:
        raise ValueError('Upstream reference source hash mismatch')
    tree = ast.parse(path.read_text())
    nodes = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in names]
    if {x.name for x in nodes} != set(names):
        raise ValueError('Missing upstream math definition')
    class Adapt(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.decorator_list = []
            return self.generic_visit(node)
        def visit_Constant(self, node):
            if cpu_devices and node.value == 'cuda':
                return ast.copy_location(ast.Constant('cpu'), node)
            return node
    tree = ast.fix_missing_locations(Adapt().visit(ast.Module(body=nodes, type_ignores=[])))
    ns = dict(torch=torch, nn=nn, F=F, math=math, List=list, MyModule=nn.Module,
              DTYPE=torch.float32, HEAD_SIZE=64, RWKV7_BATCH_OP=cpu_recurrence)
    exec(compile(tree, str(path), 'exec'), ns)
    return ns


def cpu_models(args):
    ns = reference_namespace(args.upstream / 'rwkv7.py',
        ['RWKV_x070', 'RWKV_x070_TMix_seq_batch', 'RWKV_x070_CMix_seq_batch'], True)
    weights = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    backbone = ns['RWKV_x070'](SimpleNamespace(),
        {k.removeprefix('rwkv.'):v.clone() for k,v in weights.items() if k.startswith('rwkv.')})
    ns = reference_namespace(args.upstream / 'model.py',
        ['RWKV7_OP', 'RWKV_Tmix_x070', 'RWKV_CMix_x070', 'Block', 'ReRanker'])
    source = torch.load(args.reranker, map_location='cpu', mmap=True, weights_only=True)
    values = {k.removeprefix('reranker.'):v.float() for k,v in source.items() if k.startswith('reranker.')}
    if 'token.weight' in values:
        values['emb.weight'] = values.pop('token.weight')
    depth = max(int(k.split('.')[1]) for k in values if k.startswith('blocks.')) + 1
    H,S = values['blocks.0.att.r_k'].shape
    cfg = SimpleNamespace(n_layer=depth, n_embd=H*S, dim_att=H*S, head_size_a=S, head_size_divisor=8)
    with torch.device('meta'):
        ranker = ns['ReRanker'](cfg)
    ranker.load_state_dict(values, strict=True, assign=True)
    indices = tuple(int(i) for i in source.get('reranker_layer_idx', range(depth)))
    return backbone, ranker.eval(), indices


def metrics(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    delta = a - b
    return {'max_abs':float(delta.abs().max()),
            'normalized_rmse':float(delta.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-8)),
            'finite':bool(torch.isfinite(a).all())}


def require(a, b, atol, rtol):
    m = metrics(a,b)
    if not m['finite'] or not torch.allclose(a.cpu().float(), b.cpu().float(), atol=atol, rtol=rtol):
        raise AssertionError(m)
    return m


def timed(call, sync=lambda:None):
    start=time.perf_counter(); result=call(); sync()
    return result,time.perf_counter()-start


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','reranker','upstream','runtime','build-root','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allow-shared-device',action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    report={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_sha256':{x.name:sha256(x) for x in [Path(__file__),Path(__file__).with_name('local_modeling_rwkv_embedding.py'),Path(__file__).with_name('local_modeling_rwkv_reranker.py')]},
            'checkpoint_sha256':sha256(args.checkpoint),'reranker_sha256':sha256(args.reranker),
            'upstream_sha256':UPSTREAM_HASHES,'physical_npu':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
            'cpu_reference':'Pinned upstream PyTorch definitions; CPU FP32; CUDA batch recurrence replaced by explicit matrix math; jit decorators removed and cuda device literals mapped to cpu.',
            'npu_precision':'FP16 projections, FP32 state/pointwise','backend':'raw_eager',
            'thresholds':{'state_normalized_rmse':.02,'logit_atol':.02,'logit_rtol':.005,'continuation_atol':.002,'continuation_rtol':.001},
            'cases':[],'all_checks_passed':False}
    c=None;start=time.perf_counter()
    try:
        assert report['checkpoint_sha256']==CHECKPOINT_SHA256 and report['reranker_sha256']==RERANKER_SHA256
        import torch_npu
        torch.set_num_threads(4)
        status=subprocess.check_output(['/usr/local/bin/npu-status'],text=True)
        selected=next((x for x in status.splitlines() if x.startswith(f"NPU {report['physical_npu']}: ")), '')
        if 'Health=OK' not in selected or (not args.allow_shared_device and ': free ' not in selected):
            raise RuntimeError('Selected device unhealthy or shared without authorization')
        torch.npu.set_device(0); torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format=False
        free,total=torch.npu.mem_get_info()
        if free<2*1024**3: raise RuntimeError('Need 2 GiB free HBM')
        load_bridge(Path(__file__).parent/'wkv7_npu',args.build_root)
        report.update(hostname=os.uname().nodename,device=torch.npu.get_device_name(0),torch=torch.__version__,torch_npu=torch_npu.__version__,device_status=status,hbm_before={'free_bytes':free,'total_bytes':total})
        c=CReference(args.runtime,4)
        cpu,ranker,indices=cpu_models(args)
        embed=Embedding(args.checkpoint,'npu:0',torch.float16)
        rerank=Reranker(args.reranker,'npu:0',torch.float16)
        report['architecture']={'backbone_layers':cpu.n_layer,'width':cpu.n_embd,'reranker_layers':len(rerank.blocks),'selected_layers':list(indices)}
        cases=json.loads((Path(__file__).parent/'data/reranker_smoke_cases.json').read_text())['cases']
        report['cases_sha256']=sha256(Path(__file__).parent/'data/reranker_smoke_cases.json')
        # Regression gate for the embedding-only API after moving it into its own file.
        ids,mask=c.prepare_batch([cases[0]['document']])
        expected,_=c.encode(ids,mask,trace=False)
        with torch.inference_mode():
            actual=embed(torch.tensor(ids,device='npu'),torch.tensor(mask,dtype=torch.float32,device='npu'))
            report['embedding_regression']=require(actual,torch.from_numpy(expected),.002,.01)
            for group in [[x] for x in cases]+[cases[:2]]:
                tokens=[c.tokenize(PREFIX.format(**x)+SUFFIX.format(**x)).tolist()+[65535] for x in group]
                length=max(map(len,tokens))
                tokens=[[0]*(length-len(x))+x for x in tokens]
                # Online wrapper left-pads, then retains the last 2048 positions.
                tokens=[x[-2048:] for x in tokens]
                ids=torch.tensor(tokens,dtype=torch.long,device='npu')
                state=cpu.generate_zero_state(len(group))
                (cpu_hidden,cpu_time)=timed(lambda:cpu.forward_seq_batch(tokens,state,True))
                reference_layers=[]
                hooks=[b.register_forward_hook(lambda m,a,y:reference_layers.append(y[0].detach().clone())) for b in ranker.blocks]
                (expected,cpu_rank_time)=timed(lambda:ranker(torch.stack([state[1][i] for i in indices])).reshape(-1))
                for h in hooks:h.remove()
                ((hidden,nstate,layers),npu_time)=timed(lambda:embed.encode_states(ids,trace=True),torch.npu.synchronize)
                ((logits,rank_layers),npu_rank_time)=timed(lambda:rerank.run(nstate[1],True),torch.npu.synchronize)
                state_metrics=[metrics(a,b) for a,b in zip(nstate,state)]
                hidden_metric=metrics(hidden,cpu_hidden)
                rank_metrics=[metrics(a,b) for a,b in zip(rank_layers,reference_layers)]
                assert all(x['finite'] and x['normalized_rmse']<=.02 for x in state_metrics+[hidden_metric]+rank_metrics)
                score_metric=require(logits,expected,.02,.005)
                row={'case_ids':[x['id'] for x in group],'shape':list(ids.shape),'input_ids':tokens,'input_sha256':hashlib.sha256(np.asarray(tokens,dtype=np.int64).tobytes()).hexdigest(),
                     'cpu_logits':expected.tolist(),'npu_logits':logits.cpu().tolist(),'states':state_metrics,'hidden':hidden_metric,'reranker_layers':rank_metrics,'logits':score_metric,
                     'seconds':{'cpu_backbone':cpu_time,'cpu_reranker':cpu_rank_time,'npu_backbone':npu_time,'npu_reranker':npu_rank_time}}
                # Single rows exercise the real document/query boundary and an arbitrary split.
                if len(group)==1:
                    prefix=c.tokenize(PREFIX.format(**group[0]))
                    boundary=0
                    for a,b in zip(prefix,tokens[0]):
                        if a!=b:break
                        boundary+=1
                    splits=sorted({boundary,max(1,len(tokens[0])//2)})
                    row['continuation']=[]
                    for split in splits:
                        # CPU continuation independently exercises both token-shift and matrix state.
                        cpu_saved=cpu.generate_zero_state(1)
                        cpu.forward_seq_batch([tokens[0][:split]],cpu_saved,True)
                        cpu_continued=cpu.forward_seq_batch([tokens[0][split:]],cpu_saved,True)
                        cpu_continued_score=ranker(torch.stack([cpu_saved[1][i] for i in indices])).reshape(-1)
                        cpu_match=require(cpu_continued_score,expected,1e-4,1e-4)
                        _,saved,_=embed.encode_states(ids[:,:split])
                        before=[x.clone() for x in saved]
                        (continued,final,_),elapsed=timed(lambda:embed.encode_states(ids[:,split:],saved),torch.npu.synchronize)
                        continued_score=rerank(final[1])
                        score_match=require(continued_score,logits,.002,.001)
                        cm=[metrics(a,b) for a,b in zip(final,nstate)]
                        assert all(x['finite'] and x['normalized_rmse']<=.002 for x in cm)
                        assert all(torch.equal(a,b) for a,b in zip(saved,before))
                        assert torch.equal(rerank(final[1]),continued_score)
                        row['continuation'].append({'cpu_logits':cpu_match,'split':split,'suffix_tokens':ids.shape[1]-split,'states':cm,'logits':score_match,'query_forward_seconds':elapsed,'input_state_unchanged':True,'repeat_bitwise_equal':True})
                # Exact prepared rows must remain independent when batched.
                if len(group)>1:
                    individual=[]
                    for i in range(len(group)):
                        _,s,_=embed.encode_states(ids[i:i+1]);individual.append(rerank(s[1]))
                    row['batch_vs_single']=require(torch.cat(individual),logits,.02,.005)
                artifact=args.output/(group[0]['id']+('_b2' if len(group)>1 else '')+'.npz')
                np.savez_compressed(artifact,input_ids=np.asarray(tokens),cpu_hidden=cpu_hidden.numpy(),cpu_shift=state[0].numpy(),cpu_matrix=state[1].numpy(),cpu_logits=expected.numpy(),npu_logits=logits.cpu().numpy())
                row['artifact']={'path':str(artifact),'sha256':sha256(artifact),'bytes':artifact.stat().st_size}
                report['cases'].append(row);print(json.dumps(row),flush=True)
        report['all_checks_passed']=True
    except Exception as e:
        report['error']=f'{type(e).__name__}: {e}';raise
    finally:
        if c is not None:c.close()
        report['total_seconds']=time.perf_counter()-start
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print('RERANKER_SMOKE',json.dumps({k:v for k,v in report.items() if k not in ['cases','device_status']}),flush=True)

if __name__=='__main__':main()
