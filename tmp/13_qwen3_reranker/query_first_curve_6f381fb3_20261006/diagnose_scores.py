import argparse,json,math,pathlib,statistics
p=argparse.ArgumentParser()
p.add_argument('result',type=pathlib.Path)
p.add_argument('fixture',type=pathlib.Path)
a=p.parse_args()
x=json.loads(a.result.read_text());fixture=json.loads(a.fixture.read_text())
def pearson(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    va=sum((v-ma)**2 for v in a);vb=sum((v-mb)**2 for v in b)
    return sum((v-ma)*(w-mb) for v,w in zip(a,b))/math.sqrt(va*vb) if va and vb else None
report={'status':x['status'],'updates':len(x['updates']),'evaluations':{}}
base=x['evaluations']['0']
for step,e in x['evaluations'].items():
    changes={q:e['touche']['per_query'][q]-base['touche']['per_query'][q] for q in e['touche']['per_query']}
    lengths,drifts=[],[]
    for q,scores in e['touche_margins'].items():
        dl=[len(fixture['documents'][d]['text']) for d in scores]
        delta=[v-base['touche_margins'][q][d] for d,v in scores.items()]
        lengths.extend(v-statistics.mean(dl) for v in dl)
        drifts.extend(v-statistics.mean(delta) for v in delta)
    vm=e['validation']
    report['evaluations'][step]={'touche_ndcg':100*e['touche']['ndcg10'],'delta_points':100*statistics.mean(changes.values()),'query_improved':sum(v>1e-9 for v in changes.values()),'query_worse':sum(v<-1e-9 for v in changes.values()),'query_equal':sum(abs(v)<=1e-9 for v in changes.values()),'validation_ordering':100*vm['overall']['ordering_accuracy'],'validation_source_macro_ordering':100*vm['source_macro_ordering_accuracy'],'validation_cross_entropy':vm['pointwise_cross_entropy'],'seconds':e['seconds'],'within_query_length_vs_margin_shift_pearson':pearson(lengths,drifts),'worst_queries':[{'qid':q,'text':fixture['queries'][q]['text'],'delta_points':100*changes[q]} for q in sorted(changes,key=changes.get)[:5]],'per_source_ordering_change_points':{s:100*(v['ordering_accuracy']-base['validation']['per_source'][s]['ordering_accuracy']) for s,v in vm['per_source'].items()}}
print(json.dumps(report,indent=2))
