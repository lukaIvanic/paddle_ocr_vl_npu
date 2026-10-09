"""Read-only comparison of complete MinerU page runs and frozen evaluator results."""
import argparse
from collections import Counter, defaultdict
import json
import re
from pathlib import Path
import statistics


def read(p):
    return json.loads(p.read_text())


def quality(root):
    r = root / 'evaluation/work/result'
    m = read(r / 'predictions_quick_match_metric_result.json')
    scores = {
        'text_accuracy': 100 * (1 - m['text_block']['page']['Edit_dist']['ALL']),
        'page_teds': 100 * m['table']['page']['TEDS']['ALL'],
        'page_cdm': 100 * m['display_formula']['page']['CDM']['ALL'],
    }
    scores['overall'] = statistics.mean(scores.values())
    per_page = {'text_accuracy': {k: 100 * (1-v) for k,v in read(r/'predictions_quick_match_text_block_per_page_edit.json').items()}}
    for kind,metric,label in [('table','TEDS','page_teds'),('display_formula','CDM','page_cdm')]:
        grouped = defaultdict(list)
        for row in read(r/f'predictions_quick_match_{kind}_result.json'):
            grouped[row['img_id']].append(100 * row['metric'][metric])
        per_page[label] = {k: statistics.mean(v) for k,v in grouped.items()}
    for label,pages in per_page.items():
        assert abs(statistics.mean(pages.values())-scores[label]) < 1e-9, label
    categories = {}
    for kind,metric,label in [('text_block','Edit_dist','text_accuracy'),('table','TEDS','page_teds'),('display_formula','CDM','page_cdm')]:
        categories[label] = {k: 100 * (1-v if metric == 'Edit_dist' else v) for k,v in m[kind]['page'][metric].items()}
    return scores,per_page,categories,read(r/'predictions_quick_match_stage_execution.json')


def load(root):
    s = read(root/'output/run_summary_shard_00.json')
    assert (s['completed'],s['failed'],s['skipped']) == (1651,0,0)
    q,p,c,stages = quality(root)
    gen = {}
    with (root/'output/generation_trace.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            assert row['request_id'] not in gen
            gen[row['request_id']]=row
    assert len(gen)==s['generation_trace']['requests']
    v=s['vision_timing']['all']
    g=s['local_compiled_generation']
    stream=s['streaming']
    log=(root/'evaluation/run.log').read_text().replace('\r','\n')
    final_cdm=[line for line in log.splitlines() if line.startswith('CDM: 100%')]
    cdm_rate=re.search(r',\s*([0-9.]+)it/s\]',final_cdm[-1]) if final_cdm else None
    observations=root.parent/'eval_worker_observations.jsonl'
    observed_pool=None
    if observations.exists():
        observed_pool=max((max(row['python_children_per_parent'].values(),default=0)
                           for row in map(json.loads,observations.open())),default=0)
    result={
        'chip':'910B2','pages':s['completed'],'requests':len(gen),
        'max_pixels':s['processor_max_pixels'],'setup_s':s['setup_s'],
        'pipeline_wall_s':s['pipeline_wall_s'],'pages_per_s':s['completed']/s['pipeline_wall_s'],
        'vision_s':v['device_s'],'vision_real_tokens':v['real_tokens'],
        'vision_physical_tokens':v['physical_tokens'],'vision_real_tok_s':v['real_tok_s'],
        'padding_fraction':1-v['real_tokens']/v['physical_tokens'],
        'generated_tokens':sum(len(r['generated_token_ids']) for r in gen.values()),
        'stop_reasons':dict(Counter(r['stop_reason'] for r in gen.values())),
        'non_eos_categories':dict(Counter(r['block_type'] for r in gen.values() if r['stop_reason']!='eos')),
        'decode_s':g['decode_s'],'prefill_metrics':g['prefill_metrics'],
        'cpu_prepare_wait_s':stream['cpu_prepare_wait_s'],'layout_host_wall_s':stream['layout_host_wall_s'],
        'decode_active_slot_fraction':stream['decode']['active_slot_fraction'],
        'accuracy':q,'evaluation_stages':stages,
        'evaluation_wall_s':float((root/'evaluation/wall_s.txt').read_text()),
        'cdm_final_progress_line':final_cdm[-1] if final_cdm else None,
        'cdm_aggregate_samples_per_s':float(cdm_rate.group(1)) if cdm_rate else None,
        'observed_max_python_children_per_parent':observed_pool,
    }
    return result,s,p,c,gen


def compare_pages(a,b):
    common=set(a)&set(b)
    rows=[{'image':k,'baseline':a[k],'comparison':b[k],'delta_pp':b[k]-a[k]} for k in common]
    rows.sort(key=lambda r:(r['delta_pp'],r['image']))
    worse=[r for r in rows if r['delta_pp'] < -1e-10]
    better=[r for r in rows if r['delta_pp'] > 1e-10]
    return {'common_pages':len(common),'baseline_only':sorted(set(a)-set(b)),
        'comparison_only':sorted(set(b)-set(a)),'worse':len(worse),'better':len(better),
        'equal':len(rows)-len(worse)-len(better),'worse_by_more_than_1pp':sum(r['delta_pp'] < -1 for r in rows),
        'worst_10':worse[:10],'best_10':list(reversed(better[-10:]))}


def compare_metric_inputs(ra, rb):
    result={}
    for kind,metric in [('display_formula','CDM'),('table','TEDS')]:
        maps=[]
        for root in [ra,rb]:
            rows=read(root/f'evaluation/work/result/predictions_quick_match_{kind}_result.json')
            values={(r['img_id'],json.dumps(r['gt_idx']),r['gt']):r for r in rows}
            assert len(values)==len(rows)
            maps.append(values)
        a,b=maps
        counts=Counter()
        for key in a.keys()&b.keys():
            x,y=a[key],b[key]
            same=x['pred']==y['pred']
            counts['same_prediction' if same else 'different_prediction']+=1
            if x['metric'][metric]!=y['metric'][metric]:
                counts['same_prediction_changed_score' if same else 'different_prediction_changed_score']+=1
        result[metric]={'baseline_samples':len(a),'comparison_samples':len(b),
                       'common_ground_truth_matches':len(a.keys()&b.keys()),**counts}
    return result


def compare(ra,rb):
    a,sa,pa,ca,ga=load(ra)
    b,sb,pb,cb,gb=load(rb)
    keys=['git_commit','model_hashes','layout_model_hashes','torch','torch_npu','transformers','mineru_vl_utils',
          'dtype','processor_fast','processor_min_pixels','attention','layout_backend','layout_graph_capture',
          'layout_image_size','image_analysis','batch_size','local_compiled_cache_length','local_vision_attention',
          'local_vision_backend','local_vision_buckets','local_vision_pack_target','local_vision_lookahead',
          'local_text_backend','local_text_buckets','local_text_max_members','local_decode_attention',
          'local_decode_weight_format','local_decode_rotary_impl','local_decode_increfa_length_mode',
          'streaming_page_window','local_prepare_prefetch_depth','offset','limit','warmup']
    cfg_diff={k:{'baseline':sa.get(k),'comparison':sb.get(k)} for k in keys if sa.get(k)!=sb.get(k)}
    common=set(ga)&set(gb)
    gen_counts=Counter()
    changed=[]
    for k in common:
        x,y=ga[k],gb[k]
        if x['generated_token_ids']!=y['generated_token_ids']:
            gen_counts['changed_output_requests']+=1
            changed.append(k)
        if x['prompt_token_ids']!=y['prompt_token_ids']: gen_counts['changed_prompt_requests']+=1
        if x['image_sha256']!=y['image_sha256']: gen_counts['changed_crop_image_hashes']+=1
        if x['chat_prompt']!=y['chat_prompt']: gen_counts['changed_prompt_templates']+=1
        if x['max_new_tokens']!=y['max_new_tokens']:
            gen_counts['changed_output_limits']+=1
            gen_counts['increased_output_limits' if y['max_new_tokens']>x['max_new_tokens'] else 'decreased_output_limits']+=1
    layout_changed=[]
    for x in sorted((ra/'output/layout_regions').glob('*.json')):
        y=rb/'output/layout_regions'/x.name
        if not y.exists(): layout_changed.append(x.name);continue
        xa,xb=read(x),read(y)
        xa.pop('layout_timing_s',None);xb.pop('layout_timing_s',None)
        if xa!=xb: layout_changed.append(x.name)
    cats={label:{k:{'baseline':ca[label][k],'comparison':cb[label][k],'delta_pp':cb[label][k]-ca[label][k]}
                 for k in sorted(ca[label].keys()&cb[label].keys())} for label in ca}
    return {'baseline':a,'comparison':b,'configuration_differences_except_max_pixels':cfg_diff,
        'throughput_gain_pct':100*(b['pages_per_s']/a['pages_per_s']-1),
        'real_vision_token_reduction_pct':100*(1-b['vision_real_tokens']/a['vision_real_tokens']),
        'accuracy_delta_pp':{k:b['accuracy'][k]-a['accuracy'][k] for k in a['accuracy']},
        'per_page':{k:compare_pages(pa[k],pb[k]) for k in pa},'categories':cats,
        'generation_comparison':{'common_requests':len(common),'baseline_only':sorted(set(ga)-set(gb)),
             'comparison_only':sorted(set(gb)-set(ga)),**gen_counts,'changed_request_ids':sorted(changed)},
        'layout_changed_pages':layout_changed,'metric_input_comparison':compare_metric_inputs(ra,rb)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=compare(a.baseline,a.comparison)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for name in ['baseline','comparison']:
        r=result[name]
        print(name,json.dumps({k:r[k] for k in ['chip','max_pixels','pipeline_wall_s','pages_per_s','vision_real_tokens','vision_real_tok_s','generated_tokens','stop_reasons','accuracy']}))
    print('accuracy_delta_pp',result['accuracy_delta_pp'])
    print('per_page',json.dumps(result['per_page']))
    print('configuration_differences_except_max_pixels',result['configuration_differences_except_max_pixels'])
