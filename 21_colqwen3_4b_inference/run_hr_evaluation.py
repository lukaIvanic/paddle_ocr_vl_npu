"""Instrumented, exact-corpus English HR retrieval; NPU encoding and FP32 MaxSim.

Only existing compatible transformer caches are used. Uncached shapes explicitly
use optimized raw eager; no per-shape compilation or silent failure fallback.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import faulthandler
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
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
        wall = distribution([s['wall_s'] for s in items])
        device = distribution([s['device_ms'] for s in items if 'device_ms' in s])
        tokens = sum(s.get('tokens',0) for s in items)
        return dict(wall_s=wall, device_ms=device, tokens=tokens,
                    wall_tok_s=tokens/wall['sum'] if wall['sum'] else None,
                    device_tok_s=tokens/(device['sum']/1000) if device.get('sum',0) else None)
    by_shape = [dict(kind=k[0],section=k[1],route=k[2],length=k[3],**summarize(v))
                for k,v in sorted(groups.items())]
    totals = defaultdict(list)
    for k,v in groups.items():
        totals[(k[0],k[1])].extend(v)
    return {'by_shape':by_shape, 'by_section':[
        dict(kind=k[0],section=k[1],**summarize(v)) for k,v in sorted(totals.items())],
        'item_wall_s':{kind:distribution([r['wall_s'] for r in records if r['kind']==kind])
                       for kind in sorted({r['kind'] for r in records})}}


class Journal:
    def __init__(self, root):
        self.start = time.monotonic()
        self.events = (root/'events.jsonl').open('w', buffering=1)
        self.items = (root/'items.jsonl').open('w', buffering=1)
        self.lock = threading.Lock()
        self.active = None
        self.done = threading.Event()
        self.worker = threading.Thread(target=self.heartbeat, daemon=True)
        self.worker.start()

    def emit(self, phase, **data):
        row = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_s=time.monotonic()-self.start, **data)
        with self.lock:
            line=json.dumps(row)
            self.events.write(line+'\n')
            print('HR_EVAL '+line, flush=True)

    def heartbeat(self):
        while not self.done.wait(10):
            active=self.active
            self.emit('heartbeat', active=active,
                      active_elapsed_s=time.monotonic()-active['start'] if active else None)

    @contextmanager
    def section(self, record, name, tokens=0, device=False, route=None):
        self.active=dict(kind=record['kind'],id=record['id'],section=name,start=time.monotonic())
        self.emit('section_start',kind=record['kind'],id=record['id'],section=name,
                  tokens=tokens,route=route)
        begin=time.perf_counter()
        faulthandler.dump_traceback_later(120,repeat=True)
        if device:
            torch.npu.synchronize()
            a,b=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            a.record()
        yield
        if device:
            b.record(); b.synchronize()
        stats=dict(wall_s=time.perf_counter()-begin,tokens=tokens)
        if device:
            stats['device_ms']=a.elapsed_time(b)
        if route:
            stats['route']=route
        record['sections'][name]=stats
        self.emit('section_finish',kind=record['kind'],id=record['id'],section=name,**stats)
        faulthandler.cancel_dump_traceback_later()
        self.active=None

    def close(self):
        self.done.set(); self.worker.join(timeout=2)
        self.events.close(); self.items.close()


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
        path=cached_path(self.compiler,'optimized_'+name,tensors)
        tokens=tensors[0].numel()//tensors[0].shape[-1]
        warm=path.exists() and any(path.rglob('*.om'))
        if warm:
            call=self.compiler.get('optimized_'+name,module,tensors)
            route='compiled_cache'
            if str(path) not in self.ready:
                with self.journal.section(record,name+'_cache_load',tokens,device=True,route=route):
                    call(*tensors)
                self.ready.add(str(path))
        else:
            call,route=module,'optimized_eager_uncached'
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


@torch.inference_mode()
def run(args, result, journal):
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
    journal.emit('dataset_verify_start')
    corpus,queries,qrels=read_data(args.dataset_root)
    result['qrels_inventory']={'query_ids_in_file':len(qrels),'english_query_ids':len(queries),
                              'policy':'evaluate only IDs in the English query component'}
    corpus=corpus[:args.limit_pages] if args.limit_pages else corpus
    queries=queries[:args.limit_queries] if args.limit_queries else queries
    result.update(pages=len(corpus),queries=len(queries),full_domain=not(args.limit_pages or args.limit_queries))
    journal.emit('dataset_verify_finish',pages=len(corpus),queries=len(queries))
    journal.emit('model_load_start')
    model=LocalColQwen3.from_pretrained(args.model,device=args.device)
    processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    execution=Execution(model,args,journal)
    torch.npu.synchronize()
    result['processor_image_config']=processor.image_processor.to_dict()
    journal.emit('model_load_finish')
    records=[]
    result['setup_s']=time.monotonic()-journal.start
    query_embeddings=[]
    encoding_start=time.perf_counter()
    for kind,items in [('query',queries),('page',corpus)]:
        window=time.perf_counter()
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
            with journal.section(row,'output_transfer',tt,device=True):
                output=output[0].cpu()
            with journal.section(row,'validate_save'):
                if not bool(torch.isfinite(output).all()):
                    raise ValueError('Nonfinite embeddings')
                norms=output.float().norm(dim=-1)
                active=norms>0
                if not bool(active.any()) or float((norms[active]-1).abs().max())>.002:
                    raise ValueError('Invalid embedding norms')
                row.update(embedding_rows=len(output),active_embedding_rows=int(active.sum()))
                torch.save(output,args.output_dir/'embeddings'/f'{kind}_{index:04d}.pt')
                if kind=='query':
                    query_embeddings.append(output.float())
            row['wall_s']=time.perf_counter()-begin
            records.append(row)
            journal.items.write(json.dumps(row)+'\n')
            elapsed=time.perf_counter()-window
            journal.emit('item_finish',kind=kind,id=item['id'],completed=index+1,total=len(items),
                         wall_s=row['wall_s'],vision_tokens=vt,text_tokens=tt,
                         items_per_s=(index+1)/elapsed,eta_s=(len(items)-index-1)*elapsed/(index+1))
            if (index+1)%10==0:
                (args.output_dir/'progress.json').write_text(json.dumps(dict(kind=kind,completed=index+1,
                    total=len(items),elapsed_s=elapsed,items_per_s=(index+1)/elapsed),indent=2)+'\n')
        result[kind+'_encoding_s']=time.perf_counter()-window
    result['encoding_s']=time.perf_counter()-encoding_start
    result['page_per_s']=len(corpus)/result['page_encoding_s']
    result['timings']=aggregate(records)
    result['cache_records']=execution.compiler.records
    (args.output_dir/'encoding_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    offsets=np.cumsum([0]+[len(q) for q in query_embeddings[:-1]]).tolist()
    query_flat=torch.cat(query_embeddings).to(args.device)
    scores=np.empty((len(queries),len(corpus)),dtype=np.float32)
    scoring_start=time.perf_counter()
    score_records=[]
    for index,item in enumerate(corpus):
        row=dict(kind='score',id=item['id'],index=index,sections={})
        start=time.perf_counter()
        with journal.section(row,'embedding_load_transfer',device=True):
            document=torch.load(args.output_dir/'embeddings'/f'page_{index:04d}.pt',weights_only=True).to(args.device)
        with journal.section(row,'maxsim',device=True):
            scores[:,index]=maxsim_column(query_flat,document,offsets)
        row['wall_s']=time.perf_counter()-start
        score_records.append(row); journal.items.write(json.dumps(row)+'\n')
        if index==0:
            # Independent official processor formula, using the same FP32
            # embeddings. CPU reduction is scoring, not model inference.
            expected=processor.score_multi_vector(query_embeddings[:2],[document.cpu().float()],device=args.device)
            error=float(np.abs(expected[:,0].numpy()-scores[:2,0]).max())
            result['score_contract_max_abs']=error
            if error>.0005:
                raise ValueError(f'MaxSim contract mismatch: {error}')
        if (index+1)%10==0 or index+1==len(corpus):
            elapsed=time.perf_counter()-scoring_start
            journal.emit('scoring_progress',completed=index+1,total=len(corpus),elapsed_s=elapsed,
                         eta_s=(len(corpus)-index-1)*elapsed/(index+1))
            (args.output_dir/'progress.json').write_text(json.dumps(dict(kind='scoring',completed=index+1,
                total=len(corpus),elapsed_s=elapsed),indent=2)+'\n')
    result['scoring_s']=time.perf_counter()-scoring_start
    result['scoring_timings']=aggregate(score_records)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite scores')
    np.save(args.output_dir/'scores.npy',scores)
    cids=[x['id'] for x in corpus]; qids=[x['id'] for x in queries]
    (args.output_dir/'ids.json').write_text(json.dumps(dict(corpus=cids,queries=qids),indent=2)+'\n')
    run_scores={qid:{cid:float(scores[i,j]) for j,cid in enumerate(cids)} for i,qid in enumerate(qids)}
    # pytrec_eval uses graded linear-gain nDCG, matching MTEB's TREC evaluator.
    evaluator=pytrec_eval.RelevanceEvaluator({q:qrels[q] for q in qids}, {'ndcg_cut.10','recall.10','map_cut.10'})
    per_query=evaluator.evaluate(run_scores)
    if len(per_query)!=len(qids):
        raise ValueError('Metric query count mismatch')
    metrics={k:float(np.mean([v[k] for v in per_query.values()])) for k in next(iter(per_query.values()))}
    (args.output_dir/'per_query_metrics.json').write_text(json.dumps(per_query,indent=2)+'\n')
    with (args.output_dir/'rankings.jsonl').open('w') as f:
        for i,qid in enumerate(qids):
            order=sorted(range(len(cids)),key=lambda j:(float(scores[i,j]),cids[j]),reverse=True)
            f.write(json.dumps(dict(query_id=qid,ranked_corpus_ids=[cids[j] for j in order]))+'\n')
    result.update(metrics=metrics,status='completed' if result['full_domain'] else 'completed_partial_smoke',
                  reference_ndcg_at_10=.66088,
                  reference_delta=metrics['ndcg_cut_10']-.66088 if result['full_domain'] else None)
    journal.emit('evaluation_finish',metrics=metrics,full_domain=result['full_domain'],
                 reference_delta=result['reference_delta'],page_per_s=result['page_per_s'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='npu:0')
    p.add_argument('--cache-root',type=Path,default=Path('.runtime_cache/21_colqwen3/prepared'))
    p.add_argument('--limit-pages',type=int,default=0)
    p.add_argument('--limit-queries',type=int,default=0)
    args=p.parse_args()
    if not args.device.startswith('npu:') or min(args.limit_pages,args.limit_queries)<0:
        p.error('NPU and nonnegative smoke limits required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'embeddings').mkdir()
    journal=Journal(args.output_dir)
    faulthandler.enable()
    faulthandler.dump_traceback_later(120,repeat=True)
    result=dict(status='started',command=sys.argv,host=platform.node(),dataset=REPO,revision=REVISION,
                model_config_sha256=sha256(Path(args.model)/'config.json'),
                processor_sha256=sha256(Path(args.model)/'processing_ops_colqwen3.py'),
                commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                scope='English HR full corpus unless explicit smoke limits; synchronized diagnostic timings',
                scoring='FP32 NPU dot/max, CPU FP32 per-query sum; full 2560 dimensions; pytrec_eval metrics')
    try:
        run(args,result,journal)
    except Exception:
        result.update(status='failed',error=traceback.format_exc())
        journal.emit('failed',error=result['error'])
        raise
    finally:
        result['total_s']=time.monotonic()-journal.start
        (args.output_dir/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        faulthandler.cancel_dump_traceback_later()
        journal.close()

if __name__=='__main__':
    main()
