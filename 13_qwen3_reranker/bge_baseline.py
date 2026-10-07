"""Pinned FlagEmbedding semantics with an explicit Qwen scoring interface."""
import ast,hashlib,json,logging,math,os,pathlib,random,types
REF=pathlib.Path(__file__).parent/'bge_reference'
PIN=json.loads((REF/'PIN.json').read_text())
def reference_class(filename,name,namespace):
 raw=(REF/filename).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==PIN['code_sha256'][filename]
 tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
 # Execute the unmodified class from the pinned source; avoid importing unrelated
 # FlagEmbedding modules and their incompatible tokenizer/model dependencies.
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(REF/filename),'exec'),namespace)
 return namespace[name]
def reference_dataset_class():
 import datasets
 from torch.utils.data import Dataset
 import torch.distributed as dist
 return reference_class('AbsDataset.py','AbsRerankerTrainDataset',dict(
  Dataset=Dataset,AbsRerankerDataArguments=object,PreTrainedTokenizer=object,
  datasets=datasets,dist=dist,logger=logging.getLogger(__name__),os=os,random=random,math=math))
def candidate_indices(pos,neg,rng):
 if not pos or not neg:raise ValueError('Empty supplied pool')
 pi=rng.choice(list(range(len(pos))));all_idx=list(range(len(neg)))
 if len(neg)<7:
  num=math.ceil(7/len(neg));ni=rng.sample(all_idx*num,7)
 else:ni=rng.sample(all_idx,7)
 return pi,ni

def group_loss(student,teacher):
 """Reference CE(label=0) + CE(softmax(teacher),softmax(student)); mean groups."""
 import torch
 import torch.nn.functional as F
 if student.ndim!=2 or teacher.shape!=student.shape or student.shape[1]!=8:
  raise ValueError('Expected [query_groups,8] equally shaped scores')
 s=student.float();t=torch.softmax(teacher.detach().float(),dim=-1)
 return F.cross_entropy(s,torch.zeros(len(s),dtype=torch.long,device=s.device))-(t*F.log_softmax(s,dim=-1)).sum(-1).mean()
def loss_and_score_gradient(student,teacher):
 import torch
 with torch.enable_grad():
  scores=student.detach().float().reshape(1,8).requires_grad_(True)
  loss=group_loss(scores,teacher.reshape(1,8))
  grad,=torch.autograd.grad(loss,scores)
 return loss.detach(),grad.detach().reshape(8)

def reference_loss_class():
 import torch
 from abc import ABC,abstractmethod
 from typing import Union,List,Optional,Dict
 return reference_class('AbsModeling.py','AbsRerankerModel',dict(
  ABC=ABC,abstractmethod=abstractmethod,torch=torch,nn=torch.nn,Tensor=torch.Tensor,
  PreTrainedTokenizer=object,Union=Union,List=List,Optional=Optional,Dict=Dict,
  RerankerOutput=lambda **kw:types.SimpleNamespace(**kw)))
