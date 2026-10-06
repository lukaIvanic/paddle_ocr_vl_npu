"""Eager NanoSCIDOCS with owned NPU forward and the pinned MTEB retrieval contract.

MTEB 1.38.60: revision 484eb905..., train split, descending corpus IDs,
unit relevance, cosine scoring, identical IDs retained, pytrec_eval NDCG@10.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import numpy as np
import torch
from probe_wkv7 import load_bridge
from run_cpu_reference import CReference, INSTRUCTION, CHECKPOINT_SHA256, sha256
from run_npu_embedding import Embedding, comparison

REVISION = '484eb90549fc3f0b9c42b3551e80ceb999515537'
DATA_SHA = 'c1be79f83593929a7dd1c9d900ec91534de0f532e19a38293c1352c27112f557'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def ndcg_reference(qrels, results):
    values = {}
    discounts = 1 / np.log2(np.arange(2, 12))
    for qid, relevant in qrels.items():
        # TREC breaks equal-score ties by descending document ID.
        ranked = sorted(results[qid], key=lambda did: (results[qid][did], did), reverse=True)[:10]
        gains = np.array([relevant.get(did, 0) for did in ranked])
        ideal = discounts[:min(10, len(relevant))].sum()
        values[qid] = float((gains * discounts[:len(gains)]).sum() / ideal)
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--build-root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, choices=(1, 4, 8, 16), default=1)
    p.add_argument('--dtype', choices=('fp16', 'fp32'), default='fp16')
    p.add_argument('--allow-shared-device', action='store_true')
    p.add_argument('--max-preflight-from', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    start = time.perf_counter()
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'script_sha256': sha256(__file__), 'model_script_sha256': sha256(root/'run_npu_embedding.py'),
              'checkpoint_sha256': CHECKPOINT_SHA256, 'data_sha256': sha256(args.data),
              'dataset': 'zeta-alpha-ai/NanoSCIDOCS', 'revision': REVISION, 'split': 'train',
              'hostname': platform.node(), 'physical_npu': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
              'backend': 'raw_eager', 'batch_size': args.batch_size, 'dense_dtype': args.dtype,
              'state_pointwise_dtype': 'float32', 'shared_device': args.allow_shared_device,
              'context': 2048, 'eos_chunk': 512, 'instruction': INSTRUCTION,
              'mteb_contract_version': '1.38.60', 'mteb_framework_used': False,
              'corpus_order': 'descending document ID', 'ignore_identical_ids': False,
              'stages': {}, 'all_checks_passed': False}
    cpu = None
    try:
        import torch_npu
        import pytrec_eval
        report['pytrec_eval'] = importlib.metadata.version('pytrec-eval-terrier')
        torch.set_num_threads(4)
        assert report['data_sha256'] == DATA_SHA, 'Dataset snapshot SHA mismatch'
        data = json.loads(args.data.read_text())
        assert data['revision'] == REVISION
        corpus = {row['_id']: ((row.get('title') or '') + ' ' + row['text']).strip() for row in data['corpus']}
        queries = {row['_id']: INSTRUCTION.format(query=row['text']) for row in data['queries']}
        assert len(corpus) == len(data['corpus']) == 2210 and len(queries) == len(data['queries']) == 50
        qrels = {qid: {} for qid in queries}
        for row in data['qrels']:
            assert row['query-id'] in queries and row['corpus-id'] in corpus
            qrels[row['query-id']][row['corpus-id']] = 1
        assert sum(map(len, qrels.values())) == 244 and all(qrels.values())
        load_bridge(root/'wkv7_npu', args.build_root)
        status = subprocess.check_output(['/usr/local/bin/npu-status'], text=True)
        report['device_status'] = status
        selected = next((x for x in status.splitlines() if x.startswith(f"NPU {report['physical_npu']}: ")), '')
        if 'Health=OK' not in selected or (not args.allow_shared_device and ': free ' not in selected):
            raise RuntimeError('Selected device unhealthy or occupied without shared authorization')
        torch.npu.set_device(0)
        free, total = torch.npu.mem_get_info()
        report['hbm_before_model'] = dict(free_bytes=free, total_bytes=total)
        if free < 2*1024**3:
            raise RuntimeError('Need 2 GiB free HBM')
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        report.update(device=torch.npu.get_device_name(0), torch=torch.__version__, torch_npu=torch_npu.__version__)
        cpu = CReference(args.runtime, 4)
        model = Embedding(args.checkpoint, 'npu:0', {'fp16':torch.float16, 'fp32':torch.float32}[args.dtype])
        torch.npu.reset_peak_memory_stats()
        report['setup_seconds'] = time.perf_counter()-start
        layer_limit, emb_limit = ((1e-4,1e-5) if args.dtype=='fp32' else (.02,.002))
        report['thresholds'] = dict(layer_normalized_rmse=layer_limit, embedding_max_abs=emb_limit, min_cosine=.9995)
        with torch.inference_mode():
            if args.max_preflight_from:
                previous = json.loads(args.max_preflight_from.read_text())
                for key in ['model_script_sha256', 'checkpoint_sha256', 'data_sha256', 'dense_dtype']:
                    assert previous[key] == report[key], f'Preflight identity mismatch: {key}'
                assert previous['maximum_preflight']['passed']
                report['maximum_preflight'] = dict(previous['maximum_preflight'], reused_from=str(args.max_preflight_from))
            else:
                print('PREFLIGHT: identifying largest prepared document', flush=True)
                maximum = max(corpus, key=lambda did: cpu.prepare_batch([corpus[did]])[0].shape[1])
                ids, mask = cpu.prepare_batch([corpus[maximum]])
                before = time.perf_counter()
                expected, layers = cpu.encode(ids, mask, True)
                ni = torch.from_numpy(ids.astype(np.int64)).to('npu')
                nm = torch.from_numpy(mask.astype(np.float32)).to('npu')
                got, traced = model.run(ni,nm,True)
                layer_metrics = [comparison(x.cpu().numpy(),y) for x,y in zip(traced,layers)]
                embedding = comparison(got.cpu().numpy(),expected)
                cosine = float((got.cpu().numpy()*expected).sum())
                assert all(x['normalized_rmse'] <= layer_limit for x in layer_metrics)
                assert embedding['max_abs'] <= emb_limit and cosine >= .9995
                report['maximum_preflight'] = dict(passed=True,document_id=maximum,shape=list(ids.shape),
                    layers=layer_metrics,embedding=embedding,cosine=cosine,seconds=time.perf_counter()-before)
                print('PREFLIGHT',json.dumps(report['maximum_preflight']),flush=True)
            if args.batch_size > 1:
                ids, mask = cpu.prepare_batch(list(queries.values())[:args.batch_size])
                expected,_ = cpu.encode(ids,mask,False)
                ni = torch.from_numpy(ids.astype(np.int64)).to('npu')
                nm = torch.from_numpy(mask.astype(np.float32)).to('npu')
                got = model(ni,nm).cpu().numpy()
                metrics = comparison(got,expected)
                assert metrics['max_abs'] <= emb_limit and np.min((got*expected).sum(-1)) >= .9995
                report['batch_preflight'] = dict(passed=True,shape=list(ids.shape),embedding=metrics)
                print('BATCH_PREFLIGHT',json.dumps(report['batch_preflight']),flush=True)
            ids, mask = cpu.prepare_batch(list(queries.values())[:args.batch_size])
            ni = torch.from_numpy(ids.astype(np.int64)).to('npu')
            nm = torch.from_numpy(mask.astype(np.float32)).to('npu')
            for _ in range(2): model(ni,nm)
            torch.npu.synchronize()
            report['preflight_and_warmup_seconds'] = time.perf_counter()-start-report['setup_seconds']
            evaluation_start = time.perf_counter()
            embeddings, id_sets = {}, {}
            all_batches = []
            for group, texts in [('queries',queries),('corpus',corpus)]:
                ids_order = list(texts) if group=='queries' else sorted(texts,reverse=True)
                id_sets[group] = ids_order
                stage_start = time.perf_counter()
                outputs = []
                forward_seconds = 0
                prepared_slots = 0
                for offset in range(0,len(ids_order),args.batch_size):
                    keys = ids_order[offset:offset+args.batch_size]
                    ids,mask = cpu.prepare_batch([texts[key] for key in keys])
                    if ids.shape[1] > 2048: raise RuntimeError('Prepared input exceeds validated contract')
                    ni = torch.from_numpy(ids.astype(np.int64)).to('npu')
                    nm = torch.from_numpy(mask.astype(np.float32)).to('npu')
                    torch.npu.synchronize()
                    before = time.perf_counter()
                    output = model(ni,nm)
                    torch.npu.synchronize()
                    seconds = time.perf_counter()-before
                    forward_seconds += seconds
                    vectors = output.cpu().numpy()
                    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors,axis=-1),1,atol=1e-4)
                    outputs.append(vectors)
                    prepared_slots += ids.size
                    all_batches.append(dict(group=group,offset=offset,ids=keys,shape=list(ids.shape),forward_seconds=seconds,
                        input_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
                    done = offset+len(keys)
                    if offset==0 or done//50 != offset//50 or done==len(ids_order):
                        elapsed=time.perf_counter()-stage_start
                        progress=dict(group=group,done=done,total=len(ids_order),elapsed_seconds=elapsed,
                            forward_seconds=forward_seconds,texts_per_second=done/elapsed,
                            estimated_remaining_seconds=elapsed*(len(ids_order)-done)/done)
                        save(args.output/'progress.json',progress)
                        print('PROGRESS',json.dumps(progress),flush=True)
                embeddings[group] = np.concatenate(outputs)
                report['stages'][group] = dict(texts=len(ids_order),batches=len(outputs),forward_seconds=forward_seconds,
                    wall_seconds=time.perf_counter()-stage_start,prepared_token_slots=prepared_slots)
            np.savez_compressed(args.output/'embeddings.npz',corpus=embeddings['corpus'],queries=embeddings['queries'])
            save(args.output/'ids.json',id_sets)
            save(args.output/'batch_timings.json',all_batches)
            similarities = torch.from_numpy(embeddings['queries']) @ torch.from_numpy(embeddings['corpus']).T
            results, top100 = {}, {}
            for i,qid in enumerate(id_sets['queries']):
                values,positions = torch.topk(similarities[i],1000,sorted=True)
                results[qid] = {id_sets['corpus'][j]:float(score) for j,score in zip(positions.tolist(),values.tolist())}
                top100[qid] = [dict(document_id=id_sets['corpus'][j],score=float(score))
                              for j,score in zip(positions[:100].tolist(),values[:100].tolist())]
            scores = pytrec_eval.RelevanceEvaluator(qrels,{'ndcg_cut.10'}).evaluate(results)
            manual = ndcg_reference(qrels,results)
            assert set(scores)==set(queries) and all(abs(scores[q]['ndcg_cut_10']-manual[q])<1e-12 for q in queries)
            mean = sum(scores[q]['ndcg_cut_10'] for q in queries)/len(queries)
            report.update(ndcg_at_10_unrounded=mean, ndcg_at_10=round(mean,5)*100,
                upstream_gpu_bf16_ndcg_at_10=40.988,upstream_cpu_fp32_ndcg_at_10=41.058,
                delta_gpu_percentage_points=round(mean,5)*100-40.988,
                delta_cpu_percentage_points=round(mean,5)*100-41.058,
                metric_crosscheck_passed=True,eval_wall_seconds=time.perf_counter()-evaluation_start)
            save(args.output/'per_query_scores.json',scores)
            save(args.output/'top100.json',top100)
            report['artifacts'] = {name:dict(sha256=sha256(args.output/name),bytes=(args.output/name).stat().st_size)
                for name in ['embeddings.npz','ids.json','batch_timings.json','per_query_scores.json','top100.json']}
            report['peak_hbm_allocated_bytes'] = torch.npu.max_memory_allocated()
            report['peak_hbm_reserved_bytes'] = torch.npu.max_memory_reserved()
            report['all_checks_passed'] = True
            print('SUMMARY',json.dumps(report),flush=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if cpu is not None: cpu.close()
        report['total_seconds'] = time.perf_counter()-start
        save(args.output/'result.json',report)


if __name__=='__main__': main()
