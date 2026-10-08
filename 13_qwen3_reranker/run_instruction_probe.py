"""Paired instruction probe on the existing 180-query checkpoint panel.

Original released weights stay query-first; checkpoint 500 stays document-first.
Only the Instruct field varies within each model. No training or candidate mining.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from distill_runtime import Runtime, read, digest, model_manifest, save
from margin_distillation import benchmark_metrics
from training_smoke_data import body

DEFAULT_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'


def instruction_groups(groups, variant):
    if variant not in ('task_specific', 'default'):
        raise ValueError(variant)
    return [{**g, 'instruction': DEFAULT_INSTRUCTION if variant == 'default' else g['instruction']}
            for g in groups]


def ranking(g, scores):
    return sorted((i for i, d in enumerate(g['document_ids'])
                   if not (g['ignore_identical_ids'] and d == g['qid'])),
                  key=lambda i: (scores[i], g['document_ids'][i]), reverse=True)


def differences(groups, default, custom, model):
    dm = benchmark_metrics(groups, default)
    cm = benchmark_metrics(groups, custom)
    per_query = []
    tasks = {}
    for g in groups:
        a, b = default[g['id']], custom[g['id']]
        ar, br = ranking(g, a), ranking(g, b)
        aset, bset = set(ar[:10]), set(br[:10])
        nd = next(x['ndcg10'] for x in dm['per_task'][g['task']]['per_query'] if x['qid'] == g['qid'])
        nc = next(x['ndcg10'] for x in cm['per_task'][g['task']]['per_query'] if x['qid'] == g['qid'])
        ap, bp = {i: j+1 for j, i in enumerate(ar)}, {i: j+1 for j, i in enumerate(br)}
        relevant = sorted((i for i, d in enumerate(g['document_ids']) if g['qrels'].get(d, 0)>0),
                          key=lambda i: (-g['qrels'][g['document_ids'][i]], min(ap.get(i,101),bp.get(i,101))))
        selected = list(dict.fromkeys(ar[:3]+br[:3]+sorted(aset ^ bset)+relevant[:3]))
        row = {'model': model, 'group_id': g['id'], 'task': g['task'], 'language': g['language'],
               'qid': g['qid'], 'query': g['query'], 'default_instruction': DEFAULT_INSTRUCTION,
               'task_instruction': g['instruction'], 'ndcg_default': nd, 'ndcg_task_specific': nc,
               'delta_ndcg_points': 100*(nc-nd), 'top10_overlap': len(aset & bset),
               'top1_changed': ar[0] != br[0],
               'mean_abs_margin_change': sum(abs(x-y) for x,y in zip(a,b))/len(a),
               'top10_default': [g['document_ids'][i] for i in ar[:10]],
               'top10_task_specific': [g['document_ids'][i] for i in br[:10]],
               'documents': [{'document_id': g['document_ids'][i], 'text': g['documents'][i],
                              'judged': g['document_ids'][i] in g['qrels'],
                              'relevance': g['qrels'].get(g['document_ids'][i]),
                              'rank_default': ap.get(i), 'rank_task_specific': bp.get(i),
                              'margin_default': a[i], 'margin_task_specific': b[i]}
                             for i in selected]}
        per_query.append(row)
        tasks.setdefault(g['task'], []).append(row)
    summaries = {task: {'language': rows[0]['language'], 'queries': len(rows),
                        'default_ndcg': dm['per_task'][task]['ndcg10'],
                        'task_specific_ndcg': cm['per_task'][task]['ndcg10'],
                        'delta_ndcg_points': 100*(cm['per_task'][task]['ndcg10']-dm['per_task'][task]['ndcg10']),
                        'mean_top10_overlap': sum(r['top10_overlap'] for r in rows)/len(rows),
                        'top1_changes': sum(r['top1_changed'] for r in rows)}
                 for task, rows in tasks.items()}
    by_change = sorted(per_query, key=lambda r: (abs(r['delta_ndcg_points']), 10-r['top10_overlap']), reverse=True)
    by_ranking = sorted(per_query, key=lambda r: (10-r['top10_overlap'], abs(r['delta_ndcg_points'])), reverse=True)
    return {'per_task': summaries, 'suite_default': dm['suite_macro_ndcg10'],
            'suite_task_specific': cm['suite_macro_ndcg10'],
            'largest_ndcg_changes': by_change[:12], 'largest_top10_changes': by_ranking[:6]}, per_query


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--training-result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'preparing', 'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'configuration': {k:str(v) for k,v in vars(args).items()},
              'precision': 'FP32 parameters, BF16 autocast; existing training-evaluation runtime',
              'score': 'raw yes-minus-no logits', 'max_length':8192,
              'microbatch_max':16, 'token_budget':16384,
              'host':platform.node(), 'chip':'Ascend 910B2',
              'physical_npu':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), 'conditions':{}}
    save(args.output/'result.json', result)
    try:
        data = read(args.dataset)
        source = read(args.training_result)
        assert digest(args.dataset) == source['dataset_sha256']
        frequent, reserved = data['benchmark'], data['reserved_benchmark']
        groups = frequent + reserved
        assert len(frequent)==108 and len(reserved)==72 and len(groups)==180
        assert len({g['id'] for g in groups})==180
        assert len({(g['task'],g['qid']) for g in groups})==180
        counts=Counter(g['task'] for g in groups)
        assert len(counts)==18 and set(counts.values())=={10}
        assert all(len(g['documents'])==len(g['document_ids'])==100 and len(set(g['document_ids']))==100 for g in groups)
        result['dataset_sha256']=source['dataset_sha256']
        result['checkpoint_sha256']=digest(args.checkpoint)
        result['training_result_sha256']=digest(args.training_result)
        result['original_model_manifest']=model_manifest(args.model)
        assert result['original_model_manifest']==source['model_sha256']
        result['task_counts']=dict(counts)
        result['instructions']={g['task']:g['instruction'] for g in groups}
        result['generic_instruction']=DEFAULT_INSTRUCTION
        result['same_instruction_tasks']=sorted({g['task'] for g in groups if g['instruction']==DEFAULT_INSTRUCTION})
        result['panel_membership_sha256']=hashlib.sha256(json.dumps(sorted((g['task'],g['qid'],g['document_ids']) for g in groups)).encode()).hexdigest()
        runtime=Runtime(args.model)
        torch=runtime.torch
        import torch_npu
        result['environment']={'torch':torch.__version__,'torch_npu':torch_npu.__version__}
        records={}
        for name,order in [('original','query_first'),('checkpoint500','document_first')]:
            for variant in ('task_specific','default'):
                key=f'{name}_{variant}'
                records[key]={}
                for panel,gs in [('benchmark',frequent),('reserved_benchmark',reserved)]:
                    section=f'{key}_{panel}'
                    records[key][panel]=runtime.records(instruction_groups(gs,variant),section,order)
                    assert runtime.lengths[section]['truncated']==0, f'Unexpected truncation: {section}'
        result['lengths']=runtime.lengths
        result['status']='scoring'
        save(args.output/'result.json',result)
        print('PREPARED',json.dumps({k:result[k] for k in ('task_counts','same_instruction_tasks','physical_npu','checkpoint_sha256')}),flush=True)
        model=runtime.load(args.model)
        all_scores={}
        for name,order in [('original','query_first'),('checkpoint500','document_first')]:
            if name=='checkpoint500':
                state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
                assert state['scheduler']['completed_updates']==500
                assert state['dataset_sha256']==source['dataset_sha256']
                assert state['config']['student_order']=='document_first'
                assert state['model']['lm_head.weight'].equal(state['model']['embed_tokens.weight'])
                model.load_state_dict(state['model'],strict=True)
                assert model.lm_head.weight is model.embed_tokens.weight
                del state
                print('CHECKPOINT_LOADED',500,flush=True)
            for variant in ('task_specific','default'):
                key=f'{name}_{variant}'
                scores={}; seconds=0
                for panel,rs in records[key].items():
                    values,elapsed=runtime.score(model,rs,f'{key}_{panel}')
                    scores.update(values); seconds+=elapsed
                assert set(scores)=={g['id'] for g in groups}
                reference_check=None
                if variant=='task_specific':
                    expected=source['query_first_baseline']['benchmark_scores'] if name=='original' else source['evaluations']['500']['benchmark_scores']
                    diffs=[abs(a-b) for g in frequent for a,b in zip(scores[g['id']],expected[g['id']])]
                    reference_check={'same_runtime_saved_108_max_abs_margin_difference':max(diffs),'exact_scores':max(diffs)==0}
                    assert max(diffs)==0, 'Same-input checkpoint replay differs from saved scores; inspect before interpretation'
                metrics=benchmark_metrics(groups,scores)
                condition={'input_order':order,'instruction_variant':variant,'seconds':seconds,
                           'metrics':metrics,'saved_evaluation_replay':reference_check}
                save(args.output/f'{key}_scores.json',scores)
                result['conditions'][key]=condition
                all_scores[key]=scores
                save(args.output/'result.json',result)
                print('CONDITION_FINISHED',json.dumps({'condition':key,'seconds':seconds,'suite':metrics['suite_macro_ndcg10'],'replay':reference_check}),flush=True)
        result['comparisons']={}
        for name in ('original','checkpoint500'):
            summary,details=differences(groups,all_scores[f'{name}_default'],all_scores[f'{name}_task_specific'],name)
            result['comparisons'][name]=summary
            save(args.output/f'{name}_per_query_comparison.json',details)
        result['status']='completed'; result['total_seconds']=time.monotonic()-started
        result['pairs_scored']=sum(v['pairs'] for v in runtime.lengths.values())
        assert result['pairs_scored']==72000
        save(args.output/'result.json',result)
        print('COMPLETED',json.dumps({'seconds':result['total_seconds'],'pairs':result['pairs_scored']}),flush=True)
    except BaseException as e:
        result['status']='failed';result['error']=repr(e);result['total_seconds']=time.monotonic()-started
        save(args.output/'result.json',result)
        raise


if __name__=='__main__':
    main()
