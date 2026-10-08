"""NPU mathematical control: paired replay vs independent pinned BGE losses."""
import argparse,collections,json,types
from pathlib import Path
from bge_baseline import reference_loss_class,PIN
from paired_bge_training import ORDERS,backward_window
from distill_runtime import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch,torch_npu
 torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False);torch.manual_seed(117)
 device=torch.device('npu:0');Ref=reference_loss_class()
 class Reference(Ref):
  def __init__(self,n):
   torch.nn.Module.__init__(self);self.train_batch_size=n;self.cross_entropy=torch.nn.CrossEntropyLoss(reduction='mean')
  def encode(self,features):return features.reshape(-1,1)
 class Runtime:
  def logits(self,model,rows):return (torch.stack([r['feature'] for r in rows])*model).sum(-1)
 runtime=Runtime();runtime.torch=torch;runtime.device=device
 results=[]
 for n in (1,3):
  for orders in [('document_first',),ORDERS]:
   window=[{'id':f'g{i}'} for i in range(n)];by={o:{} for o in orders};targets={g['id']:torch.randn(8).tolist() for g in window}
   for order in orders:
    for g in window:
     by[order][g['id']]=[{'group_id':g['id'],'candidate':j,'index':j,'ids':[1]*[17,129,255,513,2049,3073,4097,8192][j], 'feature':torch.randn(7,device=device)} for j in range(8)]
   start=torch.randn(7,device=device);replay=torch.nn.Parameter(start.clone());direct=torch.nn.Parameter(start.clone());calls=collections.Counter()
   def teacher(gid):calls[gid]+=1;return targets[gid]
   loss,per_order,micros=backward_window(runtime,replay,window,by,teacher)
   ref=Reference(n);teacher_flat=[v for g in window for v in targets[g['id']]]
   ref_losses=[ref(pair=torch.stack([runtime.logits(direct,by[o][g['id']]) for g in window]),teacher_scores=teacher_flat).loss for o in orders]
   expected=sum(ref_losses)/len(orders);expected.backward()
   loss_error=abs(loss.item()-expected.item());grad_error=(replay.grad-direct.grad).abs().max().item()
   assert calls==collections.Counter({g['id']:1 for g in window})
   assert torch.allclose(loss,expected,rtol=1e-5,atol=1e-6)
   assert torch.allclose(replay.grad,direct.grad,rtol=1e-5,atol=2e-6)
   # Verify the accumulated gradient produces the same single update.
   for model in [replay,direct]:
    torch.nn.utils.clip_grad_norm_([model],1.0)
    opt=torch.optim.AdamW([model],lr=1e-5,weight_decay=0,foreach=False);opt.step()
   update_error=(replay-direct).abs().max().item();assert update_error<2e-6
   results.append({'groups':n,'orders':list(orders),'loss_abs_error':loss_error,'gradient_max_abs_error':grad_error,'one_update_max_abs_error':update_error,'backward_microbatches':micros})
 out={'passed':True,'device':str(device),'chip':'Ascend 910B2','reference_commit':PIN['code_commit'],'cases':results}
 save(a.output,out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
