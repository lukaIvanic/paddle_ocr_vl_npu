"""Prepare fixed suite panels and a bounded English/Chinese retrieval pilot.

Read existing top100 retrieval results; never rerun retrieval or inject golds.
Reuse cached BGE rows and acquire short Chinese MIRACL shards from the mirror.
"""
import argparse,collections,gc,gzip,hashlib,json,os,pathlib,random,sys,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from mixture_data import MIRROR,REVISION,EXCLUDED,text_hash

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'22_qwen3_embedding_benchmark'))
from protocol import TASKS,validate_task
from suite_protocol import ENGLISH


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    raw=(json.dumps(x,ensure_ascii=False)+'\n').encode()
    p.write_bytes(gzip.compress(raw,mtime=0) if p.suffix=='.gz' else raw)
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())


def raw_pool(args):
    cached=args.output/'raw_pool.json.gz'
    if cached.exists():
        value=read(cached)
        return value['rows'],value['provenance']
    old=read(args.previous_mixture)
    rows=[];seen=set()
    allowed={'nq':'en','squad':'en','miracl_en':'en','miracl_zh':'zh'}
    for r in old['provenance']['responses']:
        config=r['config'];source=config.split('_len-')[0]
        if source not in allowed:continue
        key=hashlib.sha256(r['url'].encode()).hexdigest()+'.json.gz'
        p=args.row_cache/key
        if not p.exists():continue
        for z in read(p)['rows']:
            identity=f'{config}/{z["row_idx"]}'
            if identity in seen:continue
            seen.add(identity)
            rows.append(z['row']|{'id':identity,'source':source,'language':allowed[source]})
    # MIRACL's short retrieval rows have few supplied negatives; later lexical
    # mining supplements them from a disjoint source-specific document bank.
    config_names=['miracl_zh_len-0-500','miracl_zh_len-500-1000',
                  'miracl_en_len-0-500']
    endpoint='https://hf-mirror.com'
    metadata=json.load(urllib.request.urlopen(f'{endpoint}/api/datasets/{MIRROR}/revision/{REVISION}',timeout=30))
    assert metadata['sha']==REVISION
    args.downloads.mkdir(exist_ok=True,parents=True)
    import pyarrow.parquet as pq
    all_files={config:sorted(s['rfilename'] for s in metadata['siblings']
               if s['rfilename'].startswith(config+'/') and s['rfilename'].endswith('.parquet'))
               for config in config_names}
    assert all(all_files.values())
    def download(item):
            config,name=item
            p=args.downloads/pathlib.Path(name).name.replace('train-',config+'-')
            if not p.exists():
                started=time.monotonic()
                print('DOWNLOAD',json.dumps({'file':name}),flush=True)
                url=f'{endpoint}/datasets/{MIRROR}/resolve/{REVISION}/{name}?download=true'
                with urllib.request.urlopen(url,timeout=90) as stream,p.with_suffix('.partial').open('wb') as out:
                    while chunk:=stream.read(1024**2):out.write(chunk)
                p.with_suffix('.partial').replace(p)
                print('DOWNLOADED',json.dumps({'file':name,'bytes':p.stat().st_size,
                      'seconds':time.monotonic()-started}),flush=True)
            return p
    jobs=[(c,n) for c,files in all_files.items() for n in files]
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded=dict(zip(jobs,pool.map(download,jobs)))
    for config,files in all_files.items():
        offset=0
        for name in files:
            p=downloaded[(config,name)]
            data=pq.read_table(p).to_pylist()
            for i,row in enumerate(data):
                identity=f'{config}/{offset+i}'
                if identity not in seen:
                    source=config.split('_len-')[0]
                    rows.append(row|{'id':identity,'source':source,'language':allowed[source]})
                    seen.add(identity)
            offset+=len(data)
    origin={'original':'Shitao/bge-m3-data','mirror':MIRROR,'revision':REVISION,
                 'previous_sample_sha256':sha(args.previous_mixture),
                 'download_sha256':{p.name:sha(p) for p in args.downloads.glob('*.parquet')}}
    save(cached,{'rows':rows,'provenance':origin})
    return rows,origin


def build_groups(rows,blocked_queries,blocked_docs,seed,train_queries,val_queries):
    rng=random.Random(seed);rows=list(rows);rng.shuffle(rows)
    clean=[];seen=set()
    for row in rows:
        qh=text_hash(row['query'])
        if qh in seen or qh in blocked_queries:continue
        docs=[];dh=set()
        for d in row['pos']+row['neg']:
            h=text_hash(d)
            if d.strip() and h not in blocked_docs and h not in dh:
                docs.append(d);dh.add(h)
        pos=[d for d in row['pos'] if d in docs]
        neg=[d for d in row['neg'] if d in docs and d not in pos]
        if not pos or not neg:continue
        seen.add(qh);clean.append(row|{'pos':pos,'neg':neg})
    validation=[];train=[];vd=set()
    for lang in ['en','zh']:
        selected=[r for r in clean if r['language']==lang]
        # English source cycling prevents NQ's larger cached sample dominating.
        banks={s:collections.deque(r for r in selected if r['source']==s)
               for s in sorted({r['source'] for r in selected})}
        ordered=[]
        while any(banks.values()):
            for bank in banks.values():
                if bank:ordered.append(bank.popleft())
        v=ordered[:val_queries//2];validation.extend(v)
        vd.update(text_hash(d) for r in v for d in r['pos']+r['neg'])
    vq={text_hash(r['query']) for r in validation}
    for lang in ['en','zh']:
        selected=[]
        for r in clean:
            if r['language']!=lang or text_hash(r['query']) in vq:continue
            pos=[d for d in r['pos'] if text_hash(d) not in vd]
            neg=[d for d in r['neg'] if text_hash(d) not in vd]
            if pos and neg:selected.append(r|{'pos':pos,'neg':neg})
        banks={s:collections.deque(r for r in selected if r['source']==s)
               for s in sorted({r['source'] for r in selected})}
        while len([r for r in train if r['language']==lang])<train_queries//2 and any(banks.values()):
            for bank in banks.values():
                if bank and len([r for r in train if r['language']==lang])<train_queries//2:
                    train.append(bank.popleft())
    assert len(train)==train_queries and len(validation)==val_queries, (len(train),len(validation))

    def expand(selected):
        from sklearn.feature_extraction.text import TfidfVectorizer
        result=[]
        for source in sorted({r['source'] for r in selected}):
            rs=[r for r in selected if r['source']==source]
            # Candidate bank stays inside this split and source.
            bank=list(dict.fromkeys(d for r in rs for d in r['pos']+r['neg']))
            vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,3),max_features=100000)
            matrix=vectorizer.fit_transform(bank)
            for r in rs:
                documents=[r['pos'][0]]+r['neg'][:7]
                sims=(matrix@vectorizer.transform([r['query']]).T).toarray().ravel()
                for index in sorted(range(len(bank)),key=lambda i:(-sims[i],i)):
                    if len(documents)>=8:break
                    if bank[index] not in documents:documents.append(bank[index])
                assert len(documents)==8
                result.append({k:r[k] for k in ['id','source','language','query']}|
                              {'documents':documents,'instruction':
                               'Given a web search query, retrieve relevant passages that answer the query'})
        rng.shuffle(result)
        return result
    train,validation=expand(train),expand(validation)
    hashes=lambda gs:{text_hash(d) for g in gs for d in g['documents']}
    assert not hashes(train)&hashes(validation)
    assert not {text_hash(g['query']) for g in train}&vq
    assert not (hashes(train)|hashes(validation))&blocked_docs
    return train,validation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--english',type=pathlib.Path,required=True)
    p.add_argument('--chinese',type=pathlib.Path,required=True)
    p.add_argument('--previous-mixture',type=pathlib.Path,required=True)
    p.add_argument('--row-cache',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--downloads',type=pathlib.Path,required=True)
    p.add_argument('--panel-queries',type=int,default=6)
    p.add_argument('--reserved-queries',type=int,default=4)
    p.add_argument('--train-queries',type=int,default=1600)
    p.add_argument('--validation-queries',type=int,default=64)
    p.add_argument('--seed',type=int,default=1047)
    p.add_argument('--acquire-only',action='store_true')
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    if args.acquire_only:
        rows,origin=raw_pool(args)
        print('ACQUIRED',len(rows),json.dumps(collections.Counter(r['source'] for r in rows)),flush=True)
        return
    import mteb,importlib.metadata
    assert importlib.metadata.version('mteb')=='1.38.9'
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    from run_english_suite import load_task
    # MTEB's generic qrels read omits the default config name. Specify it for
    # offline cache resolution while preserving the suite's pinned revisions.
    import importlib
    module=importlib.import_module('mteb.abstasks.AbsTaskRetrieval')
    original_loader=module.load_dataset
    def explicit_default(repo,*a,**kw):
        if not a and 'name' not in kw:kw['name']='default'
        return original_loader(repo,*a,**kw)
    module.load_dataset=explicit_default
    class Observer:state={}
    panel=[];reserved=[];bq=set();bd=set();provenance={}
    for name in list(ENGLISH)+list(TASKS):
        print('TASK_LOADING',name,flush=True)
        if name in ENGLISH:
            task,meta=load_task(name,Observer());split='test';lang='en';instruction=ENGLISH[name][2]
            cp=args.english/name/'mteb'/f'{name}_default_predictions.json'
        else:
            task=mteb.get_tasks(tasks=[name])[0];validate_task(task);task.load_data()
            split='dev';lang='zh';instruction=TASKS[name][2];meta=dict(task.metadata.dataset)
            cp=args.chinese/f'{name}_default_predictions.json'
        candidates=json.loads(cp.read_text());queries=task.queries[split];qrels=task.relevant_docs[split]
        assert set(candidates)==set(qrels)
        bq.update(text_hash(q) for q in queries.values())
        order=sorted(qrels,key=lambda q:hashlib.sha256(f'{args.seed}/{name}/{q}'.encode()).hexdigest())
        for index,qid in enumerate(order[:args.panel_queries+args.reserved_queries]):
            ids=sorted(candidates[qid],key=lambda d:(-candidates[qid][d],d))
            assert len(ids)==100
            docs=corpus_to_str([task.corpus[split][d] for d in ids])
            bd.update(text_hash(d) for d in docs)
            g={'id':f'{name}/{qid}','task':name,'qid':qid,'language':lang,
               'query':queries[qid],'instruction':instruction,'documents':docs,'document_ids':ids,
               'qrels':qrels[qid],'ignore_identical_ids':task.ignore_identical_ids}
            (panel if index<args.panel_queries else reserved).append(g)
        provenance[name]={'candidate_sha256':sha(cp),'dataset':meta,'split':split,
                          'panel_qids':[q for q in order[:args.panel_queries]],
                          'reserved_qids':order[args.panel_queries:args.panel_queries+args.reserved_queries]}
        save(args.output/'benchmark_panels.json.gz',{'panel':panel,'reserved':reserved,'provenance':provenance})
        print('TASK_READY',name,'panel',args.panel_queries,'reserved',args.reserved_queries,flush=True)
        del task,candidates;gc.collect()
    rows,origin=raw_pool(args)
    train,val=build_groups(rows,bq,bd,args.seed,args.train_queries,args.validation_queries)
    result={'train':train,'validation':val,'benchmark':panel,'reserved_benchmark':reserved,
            'provenance':{'bge':origin,'benchmarks':provenance,'seed':args.seed,
             'excluded_sources':EXCLUDED,'language_balance':'equal English and Chinese queries',
             'candidate_selection':'first positive and up to seven supplied negatives; supplement from split/source-disjoint lexical bank',
             'filter_scope':'excluded source families, all suite query texts, selected panel candidate texts; not exhaustive corpus decontamination'},
            'distribution':dict(collections.Counter(g['source'] for g in train)),
            'checks':{'train_validation_queries_disjoint':True,'train_validation_documents_disjoint':True,
                      'benchmark_query_texts_excluded':True,'panel_candidate_texts_excluded':True}}
    save(args.output/'dataset.json.gz',result)
    print('PREPARED',json.dumps({'train':len(train),'validation':len(val),'benchmark':len(panel),
          'reserved':len(reserved),'distribution':result['distribution'],'sha256':sha(args.output/'dataset.json.gz')}),flush=True)

if __name__=='__main__':main()
