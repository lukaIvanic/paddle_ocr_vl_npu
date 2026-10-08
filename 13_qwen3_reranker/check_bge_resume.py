"""Verify NpuFusedAdamW save/load continuation against uninterrupted updates."""
import argparse,io,json
from pathlib import Path
from distill_runtime import save

def cpu(x):
 import torch
 if isinstance(x,torch.Tensor):return x.detach().cpu()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 return x

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch,torch_npu
 torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False);torch.manual_seed(117)
 def new(v):
  ps=[torch.nn.Parameter(x.to('npu:0').clone()) for x in v]
  opt=torch_npu.optim.NpuFusedAdamW(ps,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)
  return ps,opt
 def step(ps,opt,n):
  opt.zero_grad(set_to_none=False)
  loss=sum(((p-(n*.05))**2).mean() for p in ps);loss.backward()
  torch.nn.utils.clip_grad_norm_(ps,1.0);opt.step();torch.npu.synchronize()
 ps,opt=new([torch.randn(7,11),torch.randn(13)])
 for n in range(1,4):step(ps,opt,n)
 buf=io.BytesIO();torch.save({'parameters':cpu(ps),'optimizer':cpu(opt.state_dict())},buf);buf.seek(0)
 state=torch.load(buf,map_location='cpu',weights_only=False)
 resumed,ropt=new(state['parameters']);ropt.load_state_dict(state['optimizer'])
 for n in range(4,7):step(ps,opt,n);step(resumed,ropt,n)
 errors=[float((p-q).abs().max()) for p,q in zip(ps,resumed)];assert max(errors)==0,errors
 orig=cpu(opt.state_dict());rest=cpu(ropt.state_dict())
 for k,v in orig['state'].items():
  for name,value in v.items():
   other=rest['state'][k][name]
   assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
 save(a.output,{'passed':True,'chip':'Ascend 910B2','optimizer':'NpuFusedAdamW','parameter_max_abs_errors':errors,'optimizer_state_exact':True,'steps_before_save':3,'steps_after_resume':3})
 print('RESUME_CONTROL_PASSED',errors,flush=True)
if __name__=='__main__':main()
