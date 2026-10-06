import ast,hashlib,json,math,os,sys,time
from pathlib import Path
from types import SimpleNamespace
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import numpy as np
import torch,torch_npu
from torch import nn
from torch.nn import functional as F
repo=Path('/workspace/repos/rwkv-cpu-reference');sys.path.insert(0,str(repo/'26_rwkv_inference'))
from probe_wkv7 import load_bridge
from run_reranker_smoke import cpu_models,cpu_recurrence,metrics
from local_modeling_rwkv_embedding import Embedding
from local_modeling_rwkv_reranker import Reranker
from run_nanoscidocs import ndcg_reference
src=Path(__file__).parent
out=src/'result.json'
torch.set_num_threads(4);torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format=False;torch.npu.matmul.allow_hf32=False
print('DEVICE',torch.npu.get_device_name(0),flush=True)
load_bridge(repo/'26_rwkv_inference/wkv7_npu',Path('/workspace/rwkv_reference/wkv7_build_178d16e6'))
args=SimpleNamespace(checkpoint=Path('/workspace/rwkv_reference/models/rwkv0b1-emb-curriculum.pth'),reranker=Path('/workspace/rwkv_reference/models/rwkv0b1-reranker.pth'),upstream=Path('/workspace/rwkv_reference/upstream/reranker'))
weights=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)
plain={k.removeprefix('rwkv.'):v.clone() for k,v in weights.items() if k.startswith('rwkv.')}
# Execute pinned upstream math only: exclude CUDA compilation, decorators and training imports.
# Substitute checkpoint loading with the same raw tensors and the CUDA recurrence with our validated NPU bridge.
class Adapt(ast.NodeTransformer):
 def visit_FunctionDef(self,n):n.decorator_list=[];return self.generic_visit(n)
 def visit_Constant(self,n):
  if n.value=='cuda':return ast.copy_location(ast.Constant('npu:0'),n)
  return n
 def visit_Assign(self,n):
  if len(n.targets)==1 and isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr=='z' and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='load':
   n.value=ast.Name(id='raw_weights',ctx=ast.Load())
  return self.generic_visit(n)
def definitions(p,names,ns):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.fix_missing_locations(Adapt().visit(ast.Module(body=nodes,type_ignores=[]))),str(p),'exec'),ns)
def bridge(state,r,w,k,v,a,b):
 B,T,C=r.shape;H=C//64
 def layout(x):return x.float().reshape(B,T,H,64).permute(0,2,1,3).contiguous()
 y,s=torch.ops.rwkv_reference.wkv7.default(layout(k),layout(v),layout(-.6065306597126334*w.float()),layout(r),layout(a),layout(b),state.contiguous())
 state.copy_(s)
 return y.permute(0,2,1,3).reshape(B,T,C).to(r.dtype)
ns=dict(torch=torch,nn=nn,F=F,math=math,List=list,MyModule=nn.Module,DTYPE=torch.float16,HEAD_SIZE=64,RWKV7_BATCH_OP=bridge,raw_weights=plain)
definitions(src/'backbone.py',['RWKV_x070','RWKV_x070_TMix_seq_batch','RWKV_x070_CMix_seq_batch'],ns)
cfg=SimpleNamespace(n_layer=12,n_embd=768,head_size_divisor=8,head_size_a=64,dim_att=768,reranker_layer_idx=list(range(12)),grad_cp=0,use_shared_state=False)
up=ns['RWKV_x070'](cfg);up.z={k:v.to('npu:0') for k,v in up.z.items()}
rns=dict(torch=torch,nn=nn,F=F,math=math,HEAD_SIZE=64)
definitions(src/'head.py',['RWKV7_OP','RWKV_Tmix_x070','RWKV_CMix_x070','Block','ReRanker'],rns)
rankweights=torch.load(args.reranker,map_location='cpu',mmap=True,weights_only=True)
values={k.removeprefix('reranker.'):v for k,v in rankweights.items() if k.startswith('reranker.')}
# Released checkpoint calls the learned token token.weight; packaged runtime aliases it to emb.weight.
values['emb.weight']=values.pop('token.weight')
with torch.device('meta'):head=rns['ReRanker'](cfg)
head.load_state_dict(values,strict=True,assign=True);head=head.bfloat16().to('npu:0').eval()
own=Embedding(args.checkpoint,'npu:0',torch.float32);ownhead=Reranker(args.reranker,'npu:0',torch.float32)
cpub,cpuh,indices=cpu_models(args)
run=repo/'tmp/26_rwkv_inference/nanobeir_reranker_dp2_b4_fp32_4fa423a8/probe'
# Locate prepared arrays and job lists from the completed run without changing them.
if not run.exists():run=run.parent
jobs=sum([json.loads(p.read_text()) for p in sorted(run.glob('jobs_*.json'))],[])
report={'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),src/'backbone.py',src/'head.py']},'chip':torch.npu.get_device_name(0),'physical_npu':'7','control':'Upstream FP16 backbone pointwise + BF16 head, identical prepared B32 rows split into B4; CUDA recurrence replaced by validated NPU bridge; no TorchAir','cases':[]}
for name in ['NanoSCIDOCSRetrieval','NanoArguAnaRetrieval','NanoClimateFeverRetrieval','NanoHotpotQARetrieval']:
 job=next(j for j in jobs if j['task']==name)
 p=run/'prepared'/name;flat=np.load(p/'tokens.npy',mmap_mode='r');off=np.load(p/'offsets.npy')
 scores=[];own_scores=[];headonly=[];cpu_checks=[];start=time.perf_counter()
 for pos in range(0,100,4):
  j=job['row_start']+pos;rows=np.stack([flat[off[k]:off[k+1]] for k in range(j,j+4)])
  ids=torch.from_numpy(rows.astype('int64')).to('npu:0')
  with torch.inference_mode():
   state=up.generate_zero_state(4,'npu:0');up.forward_seq_batch(rows.tolist(),state,False);score=head(state[1]).flatten().float().cpu();scores.extend(score.tolist())
   _,os_,_=own.encode_states(ids);old=ownhead(os_[1]).float().cpu();own_scores.extend(old.tolist());headonly.extend(head(os_[1]).flatten().float().cpu().tolist())
   if pos==0:
    cs=[torch.zeros((12,2,4,768)),torch.zeros((12,4,12,64,64))];cpub.forward_seq_batch(rows.tolist(),cs,False);cscore=cpuh(cs[1]).flatten()
    cpu_checks.append(dict(tokens=rows.shape[1],states=metrics(os_[1],cs[1]),logits=metrics(old,cscore),upstream_fp32_cpu_logits=cscore.tolist(),owned_fp32_npu_logits=old.tolist(),upstream_fp16_bf16_npu_logits=score.tolist()))
  del state,os_
 data=json.loads((Path('/workspace/rwkv_reference/nanobeir')/name/'data.json').read_text())
 qrels={job['query_id']:{r['corpus-id']:1 for r in data['qrels'] if r['query-id']==job['query_id']}}
 def ndcg(a):return ndcg_reference(qrels,{job['query_id']:dict(zip(job['document_ids'],a))})
 def order(a):return sorted(job['document_ids'],key=lambda d:(dict(zip(job['document_ids'],a))[d],d),reverse=True)
 row=dict(task=name,query_id=job['query_id'],seconds=time.perf_counter()-start,owned_fp32_ndcg=ndcg(own_scores),upstream_style_ndcg=ndcg(scores),bf16_head_only_ndcg=ndcg(headonly),max_logit_delta=max(abs(a-b) for a,b in zip(scores,own_scores)),top10_same=order(scores)[:10]==order(own_scores)[:10],cpu_checks=cpu_checks,document_ids=job['document_ids'],owned_logits=own_scores,upstream_style_logits=scores,bf16_head_only_logits=headonly)
 report['cases'].append(row);out.write_text(json.dumps(report,indent=2)+'\n')
 print('CASE',json.dumps({k:v for k,v in row.items() if 'logits' not in k and k!='document_ids'}),flush=True)
print('RESULT',out,flush=True)
