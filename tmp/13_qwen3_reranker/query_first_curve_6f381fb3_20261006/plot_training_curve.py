import argparse,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser()
p.add_argument('result',type=pathlib.Path)
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args()
x=json.loads(a.result.read_text())
steps=sorted(map(int,x['evaluations']))
ev=[x['evaluations'][str(s)] for s in steps]
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
color='#2563eb'
axes[0,0].plot(steps,[100*e['touche']['ndcg10'] for e in ev],'-o',color=color)
axes[0,0].axhline(100*x['frozen_qwen4b_reference']['ndcg10'],color='#64748b',linestyle='--',label='Frozen Qwen3-Reranker-4B')
axes[0,0].set(title=f"Touché NDCG@10 ({len(x['touche_query_ids'])} queries, 100 candidates each)",ylabel='NDCG@10 × 100')
axes[0,0].legend(fontsize=9)
axes[0,1].plot(steps,[100*e['validation']['overall']['ordering_accuracy'] for e in ev],'-o',color=color,label='Pair-weighted')
axes[0,1].plot(steps,[100*e['validation']['source_macro_ordering_accuracy'] for e in ev],'-o',color='#ea580c',label='Source macro')
axes[0,1].set(title=f"Held-out ordering ({len(x['validation_ids'])} queries)",ylabel='Positive above negative (%)')
axes[0,1].legend(fontsize=9)
axes[1,0].plot(steps,[e['validation']['pointwise_cross_entropy'] for e in ev],'-o',color=color)
axes[1,0].set(title='Held-out no/yes cross entropy',ylabel='Mean loss')
u=x['updates']
axes[1,1].plot([r['step'] for r in u],[r['loss'] for r in u],color=color,alpha=.45,linewidth=.8)
if len(u)>=10:
    axes[1,1].plot([r['step'] for r in u][9:],[sum(v['loss'] for v in u[i-9:i+1])/10 for i in range(9,len(u))],color=color,label='10-update mean')
    axes[1,1].legend(fontsize=9)
axes[1,1].set(title='Training loss',ylabel='No/yes cross entropy')
for ax in axes.flat:
    ax.set_xlabel('Gradient updates')
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Qwen3-Reranker-0.6B · query-first full fine-tuning · Ascend 910B2',fontsize=14)
a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output.with_suffix('.png'),dpi=180,bbox_inches='tight',pad_inches=.15)
fig.savefig(a.output.with_suffix('.svg'),bbox_inches='tight',pad_inches=.15)
print(a.output.with_suffix('.png'))
