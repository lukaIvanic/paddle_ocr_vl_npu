"""Profile real HTTP requests, preserving the original Eos head and prompt."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:18423')
    p.add_argument('--concurrency', type=int, default=32)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sys.path.insert(0, '/workspace/models/Decision-2.0-Eos-0.8B')
    from transformers import AutoTokenizer
    from decision2._vendor.dev2model.decision_model import encode
    from decision2._vendor.dev2model.infer import question_to_row, product_answer
    root = Path('tmp/22_qwen3_embedding_benchmark/reranker_serving_cc6c6841/prepared/workloads.json')
    w = next(x for x in json.loads(root.read_text()) if x['task']=='EcomRetrieval')
    tok = AutoTokenizer.from_pretrained('/workspace/models/Decision-2.0-Eos-0.8B', local_files_only=True)
    def call(path, payload=None):
        req = urllib.request.Request(a.url+path, data=json.dumps(payload or {}).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            content = response.read()
            return json.loads(content) if content else None
    def one(pair):
        start=time.perf_counter()
        row=question_to_row({'id':pair['qid']+'/'+pair['did'], 'state':{'query':pair['query'],'document':pair['document']}}, 'relevance',
                            {'type':'noul','instructions':'Does the document satisfy this retrieval instruction for the query? Given a user query from an e-commerce website, retrieve description sentences of relevant products'})
        encoded=encode(row,tok,2048)
        payload={'model':'eos-0.8b','input':encoded['ids'],'task':'classify','use_activation':False,'encoding_format':'float','add_special_tokens':False,
                 'decision2':{'candidate_positions':encoded['candidate_positions'],'query_position':encoded['query_position'],'token_count':len(encoded['ids'])}}
        response=call('/pooling',payload)
        logits=response['data'][0]['data']
        score=product_answer('noul',encoded['keys'],logits,1.,[o['description'] for o in row['options']])['noul']
        assert response['usage']['prompt_tokens']==len(encoded['ids'])
        return {'qid':pair['qid'],'did':pair['did'],'tokens':len(encoded['ids']),'logits':logits,'score':score,'wall_s':time.perf_counter()-start}
    groups=[]
    def run(label, concurrency, pairs):
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results=list(pool.map(one,pairs))
        group={'label':label,'concurrency':concurrency,'wall_s':time.perf_counter()-start,'started_at':time.time()-(time.perf_counter()-start),'results':results}
        groups.append(group)
        print(json.dumps({k:v for k,v in group.items() if k!='results'}),flush=True)
        a.output.write_text(json.dumps({'profile':a.profile,'groups':groups},indent=2)+'\n')
    run('warmup_b1',1,w['pairs'][:8])
    for i in range(3):
        run('warmup_c'+str(a.concurrency)+'_'+str(i),a.concurrency,w['pairs'][:a.concurrency])
    if a.profile:
        call('/start_profile')
    try:
        run('measured_c1',1,w['pairs'][:8])
        for i in range(2):
            run('measured_c'+str(a.concurrency)+'_'+str(i),a.concurrency,w['pairs'][:128])
    finally:
        if a.profile:
            call('/stop_profile')


if __name__ == '__main__':
    main()
