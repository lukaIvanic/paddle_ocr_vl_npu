"""Compare exact candidate draws, grouped loss and score gradients to pinned code."""
import argparse,json,pathlib,random,types,tempfile
from bge_baseline import reference_dataset_class,reference_loss_class,candidate_indices,group_loss,loss_and_score_gradient,PIN

def main():
 import torch
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 Dataset=reference_dataset_class();ref=object.__new__(Dataset)
 ref.args=types.SimpleNamespace(train_group_size=8,query_instruction_for_rerank=None,
  passage_instruction_for_rerank=None,shuffle_ratio=0.,knowledge_distillation=False)
 ref.create_one_example=lambda query,doc:doc
 tested=0;short_example=None
 for np_ in [1,3]:
  for nn in [1,2,3,6,7,8,100]:
   row={'query':'q','pos':[f'p{i}' for i in range(np_)],'neg':[f'n{i}' for i in range(nn)]};ref.dataset=[row]
   for seed in range(30):
    random.seed(seed);docs,_=ref[0];state=random.getstate()
    rng=random.Random(seed);pi,ni=candidate_indices(row['pos'],row['neg'],rng)
    assert docs==[row['pos'][pi]]+[row['neg'][i] for i in ni]
    assert state==rng.getstate();tested+=1
    if np_==1 and nn==2 and seed==0:short_example={'pool':row,'sampled':docs,'negative_indices':ni}
 # Exercise the pinned loader's per-file cap and explicit file concatenation.
 with tempfile.TemporaryDirectory() as tmp:
  root=pathlib.Path(tmp);files=[]
  for name,n in [('a',5),('b',2)]:
   f=root/(name+'.jsonl');f.write_text(''.join(json.dumps({'query':f'{name}{i}','pos':['p'],'neg':['n'],'pos_scores':[1.],'neg_scores':[0.]})+'\n' for i in range(n)));files.append(str(f))
  args=types.SimpleNamespace(train_data=files,max_example_num_per_dataset=3,knowledge_distillation=False,cache_path=str(root/'cache'),query_max_len=0,passage_max_len=8192)
  random.seed(19);loaded=Dataset(args,None).dataset
  expected=[f'a{i}' for i in random.Random(19).sample(list(range(5)),3)]+['b0','b1']
  assert loaded['query']==expected and 'pos_scores' not in loaded.column_names and 'neg_scores' not in loaded.column_names
  loader_check={'passed':True,'cap':3,'input_file_rows':[5,2],'result_order':loaded['query']}
 # Verify revisits consume a continuous RNG stream and resample, matching upstream.
 random.seed(1047);expected=[ref[0][0] for _ in range(5)];rng=random.Random(1047);actual=[]
 for _ in range(5):
  pi,ni=candidate_indices(row['pos'],row['neg'],rng);actual.append([row['pos'][pi]]+[row['neg'][i] for i in ni])
 assert actual==expected and len({tuple(x) for x in actual})>1
 Base=reference_loss_class()
 class Identity(Base):
  def encode(self,features):return features.reshape(-1,1)
 tok=lambda *args,**kw:{'input_ids':[1]}
 model=torch.nn.Identity();model.config=types.SimpleNamespace(pad_token_id=0)
 results=[]
 for groups in [1,3]:
  torch.manual_seed(1047);s=torch.randn(groups,8,requires_grad=True);t=torch.randn(groups,8)*4
  refmodel=Identity(model,tok,train_batch_size=groups).train()
  r=refmodel(pair=s,teacher_scores=t.tolist()).loss;g,=torch.autograd.grad(r,s)
  own=group_loss(s,t);og,=torch.autograd.grad(own,s)
  assert torch.equal(r,own) and torch.equal(g,og)
  accumulated=torch.stack([loss_and_score_gradient(s[i],t[i])[1]/groups for i in range(groups)])
  assert torch.allclose(accumulated,g,atol=1e-7,rtol=1e-6)
  results.append({'groups':groups,'reference_loss':r.item(),'loss_absolute_difference':(own-r).abs().item(),'gradient_max_absolute_difference':(g-og).abs().max().item(),'replay_gradient_max_difference':(accumulated-g).abs().max().item()})
 result={'passed':True,'reference_commit':PIN['code_commit'],'sampler_cases':tested,'loader_check':loader_check,'short_pool_example':short_example,'resampling_verified':True,'loss_checks':results,'scope':'CPU sampler and scalar-loss correctness; no model training or inference'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
