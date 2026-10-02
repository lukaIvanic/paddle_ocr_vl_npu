"""Instrumented, exact-corpus English HR retrieval; NPU encoding and FP32 MaxSim.

Only existing compatible transformer caches are used. Uncached shapes explicitly
use optimized raw eager; no per-shape compilation or silent failure fallback.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import faulthandler
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
SCRIPT_STARTED=time.monotonic()
import traceback

import numpy as np
import torch

from download_hr_reference import FILES, REPO, REVISION
from local_modeling_colqwen3 import LocalColQwen3
from optimized_prefill import (Options, OptimizedVisionStage, OptimizedTextStage,
                               configure_compiler, text_args_for_promptfa)
from patch_embedding import LinearPatchEmbed, prepare_linear_patch_inputs
from prepared_prefill import StageCompiler, prepare_text, finish_embeddings
from run_hf_baseline import sha256
from pipeline_timing import Journal, PipelineProfiler


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {'count': 0}
    return dict(count=len(a), sum=float(a.sum()), mean=float(a.mean()),
                p50=float(np.quantile(a,.5)), p90=float(np.quantile(a,.9)),
                p99=float(np.quantile(a,.99)), max=float(a.max()))


def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        for name, s in r['sections'].items():
            groups[(r['kind'],name,s.get('route','none'),s.get('tokens',0))].append(s)
    def summarize(items):
        wall = distribution([s['host_s'] for s in items])
        device = distribution([s['device_interval_ms'] for s in items if 'device_interval_ms' in s])
        tokens = sum(s.get('tokens',0) for s in items)
        measured_tokens=sum(s.get('tokens',0) for s in items if 'device_interval_ms' in s)
        return dict(host_s=wall, device_interval_ms=device, tokens=tokens,
                    device_tok_s=measured_tokens/(device['sum']/1000) if device.get('sum',0) else None)
    by_shape = [dict(kind=k[0],section=k[1],route=k[2],length=k[3],**summarize(v))
                for k,v in sorted(groups.items())]
    totals = defaultdict(list)
    for k,v in groups.items():
        totals[(k[0],k[1])].extend(v)
    return {'by_shape':by_shape, 'by_section':[
        dict(kind=k[0],section=k[1],**summarize(v)) for k,v in sorted(totals.items())],
        'item_wall_s':{kind:distribution([r['wall_s'] for r in records if r['kind']==kind])
                       for kind in sorted({r['kind'] for r in records})},
        'unattributed_host_s':distribution([r.get('unattributed_host_s',0) for r in records])}


def cached_path(compiler, stage, tensors):
    signature={'stage':stage,'inputs':[{'shape':list(a.shape),'stride':list(a.stride()),
                'dtype':str(a.dtype)} for a in tensors], **compiler.identity}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:24]
    return compiler.cache_root/f'{stage}_{digest}'


class Execution:
    def __init__(self, model, args, journal):
        options=Options()
        self.vision=OptimizedVisionStage(model,options).eval()
        self.text=OptimizedTextStage(model,options).eval()
        self.patch=LinearPatchEmbed(model.visual.patch_embed).eval()
        self.compiler=StageCompiler(args.model,args.cache_root,journal.emit)
        configure_compiler(self.compiler,options)
        self.compiler.identity['internal_format']=True
        self.ready=set()
        self.journal=journal

    def stage(self, name, tensors, record):
        module=getattr(self,name)
        tokens=tensors[0].numel()//tensors[0].shape[-1]
        with self.journal.section(record,name+'_dispatch'):
            path=cached_path(self.compiler,'optimized_'+name,tensors)
            warm=path.exists() and any(path.rglob('*.om'))
            call=self.compiler.get('optimized_'+name,module,tensors) if warm else module
            route='compiled_cache' if warm else 'optimized_eager_uncached'
            first=str(path) not in self.ready
            self.ready.add(str(path))
            record.setdefault('stage_first_use',{})[name]=first
        # First real invocation includes cache loading, if any; never replay.
        with self.journal.section(record,name+'_transformer',tokens,device=True,route=route):
            result=call(*tensors)
        return result


def field(row, *names):
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f'Missing {names}: {list(row)}')


def read_data(root):
    import pyarrow.parquet as pq
    for name,digest in FILES.items():
        if sha256(root/name)!=digest:
            raise ValueError(f'Dataset hash mismatch: {name}')
    def read(component):
        return pq.read_table(root/f'english-{component}/test-00000-of-00001.parquet').to_pylist()
    corpus=[dict(id=str(field(r,'_id','corpus_id','id')),image=r['image']) for r in read('corpus')]
    queries=[dict(id=str(field(r,'_id','query_id','id')),text=field(r,'text','query')) for r in read('queries')]
    qrels=defaultdict(dict)
    for r in read('qrels'):
        qid=str(field(r,'query-id','query_id')); cid=str(field(r,'corpus-id','corpus_id'))
        if cid in qrels[qid]:
            raise ValueError('Duplicate qrel')
        qrels[qid][cid]=int(r['score'])
    if len(corpus)!=1110 or len(queries)!=318:
        raise ValueError('Unexpected HR English counts')
    cids={r['id'] for r in corpus}; qids={r['id'] for r in queries}
    if len(cids)!=len(corpus) or len(qids)!=len(queries):
        raise ValueError('Duplicate IDs')
    # The English MTEB qrels file deliberately contains all 1,908 translated
    # query IDs; only the 318 IDs in English queries are evaluated.
    if not qids<=set(qrels) or any(not set(v)<=cids for v in qrels.values()):
        raise ValueError('Invalid qrel references')
    return corpus,queries,dict(qrels)


def maxsim_column(query_flat, document, offsets):
    # FP32 dot products on NPU; small query-token maxima reduced on CPU. Keep
    # zero document rows: the official padded scorer includes their zero floor.
    values=(query_flat @ document.float().T).amax(-1).cpu().numpy()
    return np.add.reduceat(values,np.asarray(offsets,dtype=np.int64))


def select_workload(corpus, queries, qrels, workload):
    if workload=='full':
        return corpus,queries,qrels
    # Spread pages through the stable corpus order, rather than its first 10%.
    corpus=corpus[5::10]
    cids={r['id'] for r in corpus}
    eligible=[q for q in queries if any(cid in cids and s>0 for cid,s in qrels[q['id']].items())]
    ordered=sorted(eligible,key=lambda q:hashlib.sha256(('hr-dev-v1:'+q['id']).encode()).hexdigest())
    selected={q['id'] for q in ordered[:32]}
    queries=[q for q in queries if q['id'] in selected]
    if len(corpus)!=111 or len(queries)!=32:
        raise ValueError('Incomplete fixed HR development selection')
    return corpus,queries,{q['id']:{c:s for c,s in qrels[q['id']].items() if c in cids} for q in queries}


@torch.inference_mode()
def run(args, result, journal):
    records=[]
    setup=dict(kind='setup',id='setup',sections={})
    setup_start=time.perf_counter()
    with journal.section(setup,'runtime_initialize'):
        import torch_npu
        import pytrec_eval
        from transformers import AutoProcessor
        from PIL import Image
        if not torch.npu.is_available():
            raise RuntimeError('NPU required')
        torch.npu.set_device(args.device)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format=True
        torch.npu.matmul.allow_hf32=False
        torch.set_num_threads(4)
    result.update(device=torch.npu.get_device_name(),physical_npu=os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
                  torch=torch.__version__,torch_npu=torch_npu.__version__,options=asdict(Options()))
    with journal.section(setup,'dataset_read_verify'):
        corpus,queries,qrels=read_data(args.dataset_root)
    result['qrels_inventory']={'query_ids_in_file':len(qrels),'english_query_ids':len(queries),
                              'policy':'evaluate only IDs in the English query component'}
    with journal.section(setup,'workload_select'):
        corpus,queries,qrels=select_workload(corpus,queries,qrels,args.workload)
        manifest=dict(selection='hr-dev-v1' if args.workload=='dev' else 'full',
                      corpus=[r['id'] for r in corpus],queries=[r['id'] for r in queries])
        manifest['sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        (args.output_dir/'workload.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result['workload_manifest']=manifest
    result.update(pages=len(corpus),queries=len(queries),full_domain=args.workload=='full')
    journal.emit('dataset_verify_finish',pages=len(corpus),queries=len(queries))
    journal.emit('model_load_start')
    with journal.section(setup,'model_load'):
        model=LocalColQwen3.from_pretrained(args.model,device=args.device)
    with journal.section(setup,'processor_and_execution_setup'):
        processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
        execution=Execution(model,args,journal)
        torch.npu.synchronize()  # Existing setup-completion boundary, not subsection timing.
    result['processor_image_config']=processor.image_processor.to_dict()
    journal.emit('model_load_finish')
    setup['wall_s']=time.perf_counter()-setup_start
    journal.complete(setup,1,1,setup_start); records.append(setup)
    result['setup_s']=time.monotonic()-journal.start
    query_embeddings=[]
    page_embeddings=[]
    profiler=PipelineProfiler(args.output_dir,len(corpus),len(queries),args.profile)
    journal.profiler=profiler
    # Index the corpus first; query-to-ranked batch timing then uses existing embeddings.
    query_workflow_start=None
    encoding_start=time.perf_counter()
    for kind,items in [('page',corpus),('query',queries)]:
        window=time.perf_counter()
        if kind=='query':
            query_workflow_start=window
        for index,item in enumerate(items):
            begin=time.perf_counter()
            row=dict(kind=kind,id=item['id'],index=index,sections={})
            journal.emit('item_start',kind=kind,id=item['id'],index=index,total=len(items))
            with journal.section(row,'preprocess'):
                if kind=='page':
                    payload=item['image']['bytes']
                    with Image.open(io.BytesIO(payload)) as image:
                        image=image.convert('RGB')
                        row['image_size']=list(image.size)
                        batch=processor.process_images([image])
                    row['image_sha256']=hashlib.sha256(payload).hexdigest()
                else:
                    batch=processor.process_queries([item['text']])
            vt=int(batch['image_grid_thw'].prod()) if 'image_grid_thw' in batch else 0
            tt=int(batch['attention_mask'].sum())
            row.update(vision_tokens=vt,text_tokens=tt,image_grid_thw=batch.get('image_grid_thw',torch.empty(0)).tolist())
            row['merged_image_tokens']=int((batch['input_ids']==model.config.image_token_id).sum())
            row['prompt_or_query_tokens']=tt-row['merged_image_tokens']
            with journal.section(row,'input_transfer',device=True):
                batch={k:v.to(args.device) for k,v in batch.items()}
            with journal.section(row,'vision_prepare',vt,device=True):
                prepared=prepare_linear_patch_inputs(model,batch,execution.patch)
            vision=execution.stage('vision',prepared.vision_args[:3],row) if vt else None
            with journal.section(row,'text_prepare',tt,device=True):
                text_args=text_args_for_promptfa(prepare_text(model,prepared,vision))
            hidden=execution.stage('text',text_args,row)
            with journal.section(row,'retrieval_projection',tt,device=True):
                output=finish_embeddings(model,prepared,hidden)
            # Blocking materialization is required by the CPU embedding consumer.
            # It can wait on earlier kernels; do not label it pure copy execution.
            with journal.section(row,'output_materialize_wait',tt):
                output=output[0].cpu()
            with journal.section(row,'validate_retain'):
                if not bool(torch.isfinite(output).all()):
                    raise ValueError('Nonfinite embeddings')
                norms=output.float().norm(dim=-1)
                active=norms>0
                if not bool(active.any()) or float((norms[active]-1).abs().max())>.002:
                    raise ValueError('Invalid embedding norms')
                row.update(embedding_rows=len(output),active_embedding_rows=int(active.sum()))
                if kind=='query':
                    query_embeddings.append(output.float())
                else:
                    page_embeddings.append(output)
            row['usable_embedding_s']=time.perf_counter()-begin
            row['wall_s']=time.perf_counter()-begin
            records.append(row)
            journal.complete(row,index+1,len(items),window)
            profiler.step()
        result[kind+'_encoding_s']=time.perf_counter()-window
    result['encoding_s']=time.perf_counter()-encoding_start
    result['encoding_outside_item_spans_s']=result['encoding_s']-sum(
        r['wall_s'] for r in records if r['kind'] in ('page','query'))
    result['page_per_s']=len(corpus)/result['page_encoding_s']
    result['cache_records']=execution.compiler.records
    score_setup=dict(kind='setup',id='scoring_setup',sections={})
    score_setup_start=time.perf_counter()
    with journal.section(score_setup,'scoring_prepare'):
        offsets=np.cumsum([0]+[len(q) for q in query_embeddings[:-1]]).tolist()
        query_flat=torch.cat(query_embeddings).to(args.device)
    score_setup['wall_s']=time.perf_counter()-score_setup_start
    journal.complete(score_setup,1,1,score_setup_start); records.append(score_setup)
    max_document_rows=max(r['embedding_rows'] for r in records if r['kind']=='page')
    result['scoring_document_padding']='zero floor for documents shorter than corpus maximum, matching MTEB global padding'
    scores=np.empty((len(queries),len(corpus)),dtype=np.float32)
    scoring_start=time.perf_counter()
    score_records=[]
    for index,item in enumerate(corpus):
        row=dict(kind='score',id=item['id'],index=index,sections={})
        start=time.perf_counter()
        row.update(query_tokens=len(query_flat),document_tokens=len(page_embeddings[index]),queries=len(queries))
        with journal.section(row,'embedding_transfer',device=True):
            document=page_embeddings[index].to(args.device)
            if len(document)<max_document_rows:
                # One zero row has the same MaxSim effect as all MTEB padding
                # rows, without performing a larger all-zero matmul.
                document=torch.cat((document,document.new_zeros(1,document.shape[-1])))
        with journal.section(row,'maxsim',device=True):
            maxima=(query_flat @ document.float().T).amax(-1)
        with journal.section(row,'score_materialize_wait'):
            values=maxima.cpu().numpy()
        with journal.section(row,'query_score_reduction'):
            scores[:,index]=np.add.reduceat(values,np.asarray(offsets,dtype=np.int64))
        row['wall_s']=time.perf_counter()-start
        score_records.append(row)
        journal.complete(row,index+1,len(corpus),scoring_start)
        profiler.step()
    result['scoring_s']=time.perf_counter()-scoring_start
    profiler.close()
    journal.resolve()
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite scores')
    cids=[x['id'] for x in corpus]; qids=[x['id'] for x in queries]
    final=dict(kind='finalize',id='metrics_outputs',sections={})
    final_start=time.perf_counter()
    with journal.section(final,'ranking'):
        rankings=[dict(query_id=qid,ranked_corpus_ids=[cids[j] for j in
                    sorted(range(len(cids)),key=lambda j:(float(scores[i,j]),cids[j]),reverse=True)])
                    for i,qid in enumerate(qids)]
    result['query_to_ranked_batch_s']=time.perf_counter()-query_workflow_start
    with journal.section(final,'quality_metrics'):
        run_scores={qid:{cid:float(scores[i,j]) for j,cid in enumerate(cids)} for i,qid in enumerate(qids)}
        evaluator=pytrec_eval.RelevanceEvaluator({q:qrels[q] for q in qids}, {'ndcg_cut.10','recall.10','map_cut.10'})
        per_query=evaluator.evaluate(run_scores)
    if len(per_query)!=len(qids):
        raise ValueError('Metric query count mismatch')
    metrics={k:float(np.mean([v[k] for v in per_query.values()])) for k in next(iter(per_query.values()))}
    with journal.section(final,'artifact_write'):
        np.save(args.output_dir/'scores.npy',scores)
        (args.output_dir/'ids.json').write_text(json.dumps(dict(corpus=cids,queries=qids),indent=2)+'\n')
        (args.output_dir/'per_query_metrics.json').write_text(json.dumps(per_query,indent=2)+'\n')
        with (args.output_dir/'rankings.jsonl').open('w') as f:
            for ranking in rankings:
                f.write(json.dumps(ranking)+'\n')
    final['wall_s']=time.perf_counter()-final_start
    journal.complete(final,1,1,final_start); records.append(final)
    result['timings']=aggregate(records)
    result['scoring_timings']=aggregate(score_records)
    result.update(metrics=metrics,status='completed' if result['full_domain'] else 'completed_development',
                  reference_ndcg_at_10=.66088,
                  reference_delta=metrics['ndcg_cut_10']-.66088 if result['full_domain'] else None)
    journal.emit('evaluation_finish',metrics=metrics,full_domain=result['full_domain'],
                 reference_delta=result['reference_delta'],page_per_s=result['page_per_s'])


def main(observer_factory=Journal):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='npu:0')
    p.add_argument('--cache-root',type=Path,default=Path('.runtime_cache/21_colqwen3/prepared'))
    p.add_argument('--workload',choices=('dev','full'),default='dev',help='Full HR only by explicit request')
    p.add_argument('--profile',action='store_true',help='Profile selected real items in the complete pipeline')
    args=p.parse_args()
    if not args.device.startswith('npu:'):
        p.error('NPU required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    journal=observer_factory(args.output_dir,profile=args.profile)
    faulthandler.enable()
    result=dict(status='started',command=sys.argv,host=platform.node(),dataset=REPO,revision=REVISION,
                python_import_and_cli_s=time.monotonic()-SCRIPT_STARTED,
                model_config_sha256=sha256(Path(args.model)/'config.json'),
                processor_sha256=sha256(Path(args.model)/'processing_ops_colqwen3.py'),
                commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                workload=args.workload,profiler=args.profile,
                scope='B1 sequential real pipeline; deferred event intervals, host spans, no embedding serialization',
                scoring='FP32 NPU dot/max, CPU FP32 per-query sum; full 2560 dimensions; pytrec_eval metrics')
    try:
        run(args,result,journal)
    except Exception:
        result.update(status='failed',error=traceback.format_exc())
        journal.emit('failed',error=result['error'])
        raise
    finally:
        if hasattr(journal,'profiler'):
            journal.profiler.close()
        journal.close()
        result['total_s']=time.monotonic()-journal.start
        result['script_wall_s']=time.monotonic()-SCRIPT_STARTED
        result['timing_semantics']='host_s is submission/blocking wall time, not device execution; device_interval_ms can include launch gaps and waits; no inter-section timing barriers'
        result['embedding_storage']='CPU memory only; released at process exit'
        (args.output_dir/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        faulthandler.cancel_dump_traceback_later()

if __name__=='__main__':
    main()
