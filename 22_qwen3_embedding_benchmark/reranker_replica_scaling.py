"""Fixed-token, independent-client 1/2/4 replica throughput check.

Each replica processes the identical full workload, not a shard. Tokenization
is outside timing; HTTP, score parsing and finite/coverage checks are timed.
Only servers created by the reused server context are stopped on exit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from reranker_protocol import PREFIX, SUFFIX, MAX_LENGTH
from protocol import TASKS
from reranker_serving_sweep import digest
from run_evaluation import emit


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def worker(args):
    data = json.loads(args.input.read_text())
    ids = data['input_ids']
    batches = [ids[i:i+128] for i in range(0, len(ids), 128)]
    payloads = [json.dumps(dict(model='reranker-diagnostic', input=b, task='classify',
        use_activation=True, encoding_format='float', add_special_tokens=False)).encode()
        for b in batches]
    batch_tokens = [sum(map(len, b)) for b in batches]
    root = args.output
    root.mkdir()

    def one(index):
        req = urllib.request.Request(args.endpoint+'/pooling', data=payloads[index],
            headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=180) as response:
            response = json.load(response)
        rows = sorted(response['data'], key=lambda r: r['index'])
        if [r['index'] for r in rows] != list(range(len(batches[index]))):
            raise ValueError('Missing or duplicate response indices')
        if any(len(r['data']) != 1 for r in rows):
            raise ValueError('Expected one activated relevance score per input')
        values = [float(r['data'][0]) for r in rows]
        if not all(math.isfinite(float(x)) and 0 <= x <= 1 for x in values):
            raise ValueError('Invalid score')
        if response['usage']['prompt_tokens'] != batch_tokens[index]:
            raise ValueError('Server token accounting differs')
        return index, values, time.monotonic()-start

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Warm the same full workload, including every sequence length.
        for _ in range(2):
            for offset in range(0, len(batches), 2):
                for future in [pool.submit(one, i) for i in range(offset, min(offset+2, len(batches)))]:
                    future.result()
        save(root/'ready.json', dict(pid=os.getpid(), input_sha256=digest(args.input)))
        while not args.start_file.exists():
            time.sleep(.02)
        start_at = json.loads(args.start_file.read_text())['start_at']
        while time.time() < start_at:
            time.sleep(.005)
        started_wall = time.time()
        started = time.monotonic()
        sequence = iter((cycle, i) for cycle in range(args.cycles) for i in range(len(batches)))
        pending = {}
        first_scores = [None]*len(ids)
        pairs = tokens = requests = 0
        cycle_counts = [0]*args.cycles
        http_total = 0.
        def submit():
            item = next(sequence, None)
            if item is not None:
                cycle, index = item
                pending[pool.submit(one, index)] = (cycle, index)
        submit(); submit()
        with (root/'progress.jsonl').open('w') as log:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    cycle, index = pending.pop(future)
                    _, values, latency = future.result()
                    if cycle == 0:
                        first_scores[index*128:index*128+len(values)] = [float(x) for x in values]
                    pairs += len(values)
                    tokens += batch_tokens[index]
                    requests += 1
                    http_total += latency
                    cycle_counts[cycle] += 1
                    if cycle_counts[cycle] == len(batches):
                        log.write(json.dumps(dict(cycle=cycle+1, pairs=pairs, tokens=tokens,
                            elapsed_s=time.monotonic()-started))+'\n')
                        log.flush()
                    submit()
        elapsed = time.monotonic()-started
    assert pairs == len(ids)*args.cycles
    assert tokens == sum(map(len, ids))*args.cycles
    assert all(x is not None for x in first_scores)
    save(root/'first_scores.json', first_scores)
    save(root/'result.json', dict(device=args.device, pairs=pairs, tokens=tokens,
        elapsed_s=elapsed, started_wall=started_wall, ended_wall=time.time(),
        input_tok_s=tokens/elapsed, requests=requests,
        mean_request_s=http_total/requests, cycles=args.cycles,
        input_sha256=digest(args.input)))


def phase(args, endpoints, count, reference):
    root = args.output/f'replicas_{count}'
    root.mkdir()
    processes, logs = [], []
    start_file = root/'start.json'
    try:
        for i in range(count):
            log = (root/f'worker_{i}.log').open('w')
            logs.append(log)
            cmd = [sys.executable, __file__, '--worker', '--input', str(args.output/'inputs.json'),
                '--output', str(root/f'npu_{args.devices[i]}'), '--endpoint', endpoints[i],
                '--device', str(args.devices[i]), '--cycles', str(args.cycles),
                '--start-file', str(start_file)]
            processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'},
                start_new_session=True))
        deadline = time.monotonic()+300
        while not all((root/f'npu_{d}'/'ready.json').exists() for d in args.devices[:count]):
            if any(p.poll() is not None for p in processes):
                raise RuntimeError(f'Worker failed before warmup: {root}')
            if time.monotonic() > deadline:
                raise TimeoutError('Worker warmup timeout')
            time.sleep(.1)
        save(start_file, dict(start_at=time.time()+1))
        emit('phase_started', replicas=count, cycles=args.cycles)
        deadline = time.monotonic()+1200
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f'Worker failed: {root}')
            if time.monotonic() > deadline:
                raise TimeoutError('Measurement timeout')
            time.sleep(.2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError(f'Worker failed: {root}')
        rows = [json.loads((root/f'npu_{d}'/'result.json').read_text()) for d in args.devices[:count]]
        delta = 0.
        for d in args.devices[:count]:
            scores = json.loads((root/f'npu_{d}'/'first_scores.json').read_text())
            if reference is None:
                reference = scores
            if len(scores) != len(reference):
                raise ValueError('Replica score coverage differs')
            delta = max(delta, max(abs(a-b) for a,b in zip(reference, scores)))
        if delta > .03:
            raise ValueError(f'Replica score mismatch: {delta}')
        wall = max(r['ended_wall'] for r in rows)-min(r['started_wall'] for r in rows)
        result = dict(replicas=count, workers=rows, tokens=sum(r['tokens'] for r in rows),
            pairs=sum(r['pairs'] for r in rows), elapsed_s=wall,
            input_tok_s=sum(r['tokens'] for r in rows)/wall,
            max_score_difference_from_single=delta,
            start_spread_s=max(r['started_wall'] for r in rows)-min(r['started_wall'] for r in rows))
        save(root/'result.json', result)
        emit('phase_finished', **result)
        return result, reference
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model')
    p.add_argument('--prepared', type=Path)
    p.add_argument('--devices', type=int, nargs='+', default=[0,1,2,3])
    p.add_argument('--port', type=int, default=18710)
    p.add_argument('--cycles', type=int, default=24)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--input', type=Path)
    p.add_argument('--endpoint')
    p.add_argument('--device', type=int)
    p.add_argument('--start-file', type=Path)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if args.devices != [0,1,2,3] or args.output.exists():
        p.error('Requires devices 0 1 2 3 and fresh output')
    args.output.mkdir(parents=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    import run_reranker_evaluation as evaluator
    from run_english_suite import servers
    from transformers import AutoTokenizer
    evaluator.MODEL = args.model
    args.allow_occupied_devices = [0]
    snapshot = subprocess.check_output(['npu-smi','info'], text=True, timeout=120)
    import re
    busy = [(int(a), int(b)) for a,b in re.findall(
        r'^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|', snapshot, re.M)]
    if any(d in args.devices and (d != 0 or pid != 635074) for d,pid in busy):
        raise RuntimeError(f'Unexpected process on selected device: {busy}')
    (args.output/'initial_npu_snapshot.txt').write_text(snapshot)
    source = args.prepared/'workloads.json'
    workloads = json.loads(source.read_text())
    workload = next(w for w in workloads if w['task']=='T2Retrieval')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prefix, suffix = [tok.encode(x, add_special_tokens=False) for x in (PREFIX, SUFFIX)]
    texts = [f'<Instruct>: {TASKS["T2Retrieval"][2]}\n<Document>: {r["document"]}\n<Query>: {r["query"]}'
        for r in workload['pairs']]
    bodies = tok(texts, add_special_tokens=False, padding=False, truncation=False)['input_ids']
    limit = MAX_LENGTH-len(prefix)-len(suffix)
    ids = [prefix+b[:limit]+suffix for b in bodies]
    save(args.output/'inputs.json', dict(input_ids=ids))
    save(args.output/'manifest.json', dict(model=args.model, input_order='document_first',
        task='T2Retrieval', source=str(source), source_sha256=digest(source),
        inputs_sha256=digest(args.output/'inputs.json'), pairs_per_cycle=len(ids),
        tokens_per_cycle=sum(map(len, ids)), cycles=args.cycles,
        truncated_pairs_per_cycle=sum(len(b)>limit for b in bodies),
        devices=args.devices, dtype='float16', max_length=MAX_LENGTH,
        max_num_seqs=32, max_num_batched_tokens=16384, client_batch=128,
        concurrency_per_replica=2, warmup_cycles=2, tokenization_timed=False,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model_sha256={f.name:digest(f) for f in Path(args.model).glob('*.safetensors')}))
    class Observer:
        state = {}
    results, reference = [], None
    try:
        with servers(args, 'reranker', Observer()) as endpoints:
            for count in [1,2,4]:
                row, reference = phase(args, endpoints, count, reference)
                results.append(row)
                for r in results:
                    r['speedup'] = r['input_tok_s']/results[0]['input_tok_s']
                    r['scaling_efficiency'] = r['speedup']/r['replicas']
                save(args.output/'results.json', results)
        save(args.output/'completion.json', dict(status='complete', results=results))
    except BaseException as exc:
        save(args.output/'failure.json', dict(error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
