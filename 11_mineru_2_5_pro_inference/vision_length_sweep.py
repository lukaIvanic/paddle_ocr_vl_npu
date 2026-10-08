#!/usr/bin/env python3
"""Real-crop selection and paired full-encoder replay; never synthetic tokens."""
import argparse
from contextlib import ExitStack, contextmanager
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from vision_diagnostic_config import load_config
from vision_diagnostic_runner import run_lane

HERE = Path(__file__).resolve().parent


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def ownership(path):
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'another sweep owns {path}; no concurrent cache writers') from None
        yield


def sources():
    # The coordinator may change to add recovery; model/timing code may not.
    names = ['bench_production_vision_attention.py', 'vision_prefill_compile.py',
             'local_modeling_mineru.py', 'mineru_prefill_timing.py',
             'profile_production_vision_routes.py', 'run_transformers_recognition_smoke.py',
             'vision_diagnostic_config.py']
    paths = [HERE/name for name in names]
    paths.append(HERE.parent/'09_persistent_page_engine/scripts/vision_matmul_lab.py')
    return {str(p.relative_to(HERE.parent)): sha(p) for p in paths}


def verify_source_commit(commit, expected):
    for name, digest in expected.items():
        source = subprocess.check_output(['git','-C',str(HERE.parent),'show',f'{commit}:{name}'])
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError(f'benchmark source changed since {commit}: {name}; cannot resume this sweep')


def environment():
    packages = {}
    for name in ['torch','torch-npu','transformers','torchair']:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    cann = os.environ.get('ASCEND_HOME_PATH')
    version = Path(cann)/'version.info' if cann else None
    return dict(python=sys.version, hostname=platform.node(), packages=packages,
        physical_device=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), cann_path=cann,
        cann_version_sha256=sha(version) if version and version.is_file() else None)


def command_value(receipt, flag):
    argv = json.loads((receipt/'command.json').read_text())['argv']
    return argv[argv.index(flag)+1]


def complete(receipt):
    path = receipt/'exit.json'
    return path.is_file() and json.loads(path.read_text()).get('status') == 'completed'


def preserve_attempt(root, names, reason):
    existing = [root/name for name in names if (root/name).exists()]
    if not existing: return
    # A dead launcher can leave compiler children alive. Never race that group.
    for path in existing:
        process = path/'process.json'
        if process.is_file():
            pg = json.loads(process.read_text())['process_group']
            try: os.killpg(pg, 0)
            except ProcessLookupError: pass
            else: raise RuntimeError(f'previous process group {pg} is still alive; stop its owner before resuming')
    archive = root/'attempts'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    archive.mkdir(parents=True)
    for path in existing: path.rename(archive/path.name)
    save(archive/'reason.json',dict(reason=reason, preserved=[p.name for p in existing]))
    print(f'SWEEP preserved previous attempt: {archive}',flush=True)


def check_capture(root, selection, cfg, code):
    manifest = json.loads((root/'capture/manifest.json').read_text())
    if load_config(inherited=manifest['vision_config']) != cfg:
        raise ValueError('saved capture configuration changed')
    verify_source_commit(manifest['commit'],code)
    for name, expected in manifest['model_hashes'].items():
        if sha(Path(selection['model'])/name) != expected:
            raise ValueError(f'model changed: {name}')
    if len(manifest['routes']) != len(selection['selected']):
        raise ValueError('capture did not preserve selected crop count')
    for entry, chosen in zip(manifest['routes'].values(),selection['selected']):
        if (entry['tags']['real_tokens'],entry['bucket'],entry['image_sha256']) != (
                chosen['real_tokens'],chosen['bucket'],chosen['image_sha256']):
            raise ValueError('saved capture token grid/bucket/image differs from selection')
        if sha(root/'capture'/entry['file']) != entry['sha256']:
            raise ValueError('saved capture tensor hash mismatch')
    return manifest


def reusable_lane(root, lane, manifest, cfg, steps, code):
    receipt = root/(lane['name']+'.receipt')
    result_path = root/lane['name']/'result.json'
    if not complete(receipt) or not result_path.is_file(): return False
    result = json.loads(result_path.read_text())
    if result.get('status') not in ['completed','unsupported_on_this_device']: return False
    entry = manifest['routes'][lane['route']]
    expected = dict(cfg,approximate_precision=lane['variant']=='pfa_approx')
    if (result['route'] != lane['route'] or result['variant'] != lane['variant']
            or result['source_capture_sha256'] != entry['sha256']
            or result['model_hashes'] != manifest['model_hashes']
            or result['capture_config'] != cfg or result['vision_config'] != expected
            or result['tags'] != entry['tags']
            or int(command_value(receipt,'--steps')) != steps):
        raise ValueError(f'saved successful lane metadata differs: {lane["name"]}')
    verify_source_commit(result['commit'],code)
    if result['status'] == 'completed':
        if any(result['timing_gate'].values()) or not result['repeat_parity']['exact']:
            raise ValueError(f'invalid saved timing/repeat gate: {lane["name"]}')
        if lane['variant']=='baseline' and not result['full_encoder_parity']['exact']:
            raise ValueError('saved baseline was not bit-exact with capture')
    return True


def choose(inventory, buckets, count):
    """Pick distinct real crops nearest evenly spaced positions in each bucket."""
    selected = []
    lower = 0
    for upper in sorted(buckets):
        available = [r for r in inventory if lower < r['real_tokens'] <= upper]
        if len(available) < count:
            raise ValueError(f'bucket {upper}: only {len(available)} real crops, need {count}; no fabricated fallback')
        for i in range(count):
            target = lower + (upper-lower) * (i+1)/(count+1)
            row = min(available, key=lambda r: (abs(r['real_tokens']-target), r['id']))
            available.remove(row)
            selected.append(dict(row, bucket=upper, selection_target=target))
        lower = upper
    return selected


def select(args):
    # Capture uses slow; the production cache prewarm explicitly selects fast.
    from PIL import Image
    from transformers import AutoProcessor
    cfg = load_config(args.config_json)
    processor = AutoProcessor.from_pretrained(args.model, use_fast=args.processor == 'fast', local_files_only=True).image_processor
    processor.min_pixels, processor.max_pixels = cfg['min_pixels'], cfg['max_pixels']
    if getattr(processor, 'size', None) is not None:
        processor.size['shortest_edge'], processor.size['longest_edge'] = cfg['min_pixels'], cfg['max_pixels']
    inventory = []
    for row in json.loads(args.manifest.read_text()):
        path = args.manifest.parent / row['file']
        with Image.open(path) as source:
            size = list(source.size)
            inp = processor(images=[source.convert('RGB')], return_tensors='pt')
        real = int(inp.pixel_values.shape[0])
        if real * 196 > cfg['max_pixels']:
            raise ValueError('processor exceeded cap')
        inventory.append(dict(id=row['id'], image=str(path.resolve()), original_size=size,
            image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            real_tokens=real, grid=inp.image_grid_thw.tolist(), category=row['category_type']))
    picked = choose(inventory, cfg['buckets'], args.per_bucket)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    save(args.output, dict(config=cfg, model=str(args.model.resolve()), per_bucket=args.per_bucket,processor_fast=args.processor == 'fast',
        inventory=inventory, selected=picked,
        scope='real single crops at normal processor sizes; no small-crop packing or production frequency weighting'))
    for row in picked:
        print(f"SELECT {row['id']} useful_tokens={row['real_tokens']} bucket={row['bucket']}", flush=True)


def report(root):
    metadata = json.loads((root/'sweep.json').read_text())
    rows = []
    for lane in metadata['lanes']:
        result_path = root/lane['name']/'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        receipt_path = root/(lane['name']+'.receipt')/'exit.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        status = result.get('status', 'missing_result')
        if receipt.get('status') != 'completed':
            status = receipt.get('status', 'missing_receipt')
        timing = result.get('timing', {})
        rows.append(dict(lane, status=status, chip=result.get('device', 'UNKNOWN'),
            real_tokens=result.get('tags', {}).get('real_tokens'),
            bucket=result.get('tags', {}).get('physical_tokens'), timing=timing,
            wall_tok_s=result.get('wall_real_tok_s'), event_tok_s=result.get('real_tok_s'),
            drift=result.get('full_encoder_parity'), timing_gate=result.get('timing_gate'),
            source_commit=result.get('commit'),
            receipt_dir=str(receipt_path.parent)))
    pairs = []
    for route in metadata['routes']:
        for repeat in range(metadata['repeats']):
            pair = [r for r in rows if r['route'] == route and r['repeat'] == repeat]
            by_variant = {r['variant']: r for r in pair}
            if all(by_variant.get(v, {}).get('status') == 'completed' for v in ['baseline','pfa_approx']):
                base, approx = by_variant['baseline'], by_variant['pfa_approx']
                pairs.append(dict(route=route, repeat=repeat,
                    wall_tok_s_gain_pct=100*(base['timing']['wall_ms']['mean']/approx['timing']['wall_ms']['mean']-1),
                    event_tok_s_gain_pct=100*(base['timing']['device_ms']['mean']/approx['timing']['device_ms']['mean']-1)))
    history_path = root/'resume_history.jsonl'
    history = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
    summary = dict(scope=metadata['scope'], rows=rows, pairs=pairs, resume_history=history,
        production_weighted_gain=None,
        note='No improvement is computed for failed, unsupported or missing pairs. Feature drift does not suppress valid throughput.')
    save(root/'length_results.json', summary)
    lines = ['Full 32-block MinerU vision stack; warm compiled FP16 D80, ND weights.',
        metadata['scope'], '',
        '| Chip | Lane | Useful / bucket tokens | Wall ms | Event ms | Wall tok/s | Event tok/s | Status |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    def number(v):
        return '—' if v is None else f'{v:.3f}'
    for row in rows:
        t = row['timing']
        lines.append(f"| {row['chip']} | {row['name']} | {row['real_tokens']} / {row['bucket']} | "
            f"{number(t.get('wall_ms', {}).get('mean'))} | {number(t.get('device_ms', {}).get('mean'))} | "
            f"{number(row['wall_tok_s'])} | {number(row['event_tok_s'])} | {row['status']} |")
    lines += ['', '| Route / repeat | Wall tok/s gain | Event tok/s gain |', '|---|---:|---:|']
    for pair in pairs:
        lines.append(f"| {pair['route']} / {pair['repeat']+1} | {pair['wall_tok_s_gain_pct']:+.2f}% | {pair['event_tok_s_gain_pct']:+.2f}% |")
    lines += ['', '| Lane | Feature relative L2 | Max absolute difference | Bit-exact |', '|---|---:|---:|---|']
    for row in rows:
        d = row.get('drift') or {}
        lines.append(f"| {row['name']} | {number(d.get('relative_l2'))} | {number(d.get('max_abs'))} | {d.get('exact', '—')} |")
    lines += ['', 'Failed / unsupported / missing lanes:']
    lines += [f"- {r['name']}: {r['status']}" for r in rows if r['status'] != 'completed'] or ['- None']
    lines += ['', 'No production-weighted improvement is claimed. JSON retains mean/p50/p99 timings and timing gates.']
    if any(row.get('resume') for row in history):
        lines += ['This sweep was resumed. Completed pairs retain their original timings and receipts; incomplete pairs were rerun together. See resume_history.jsonl and attempts/.',
                  'Pairs can come from different execution times; inspect their saved host/device snapshots before comparing repeats.']
    if any(row['event']=='legacy_environment_limit' for row in history):
        lines += ['Legacy resume: the old run did not record a complete environment fingerprint. Full old CANN/processor environment equality cannot be established; original lane runtime metadata remains available.']
    content = '\n'.join(lines)+'\n'
    (root/'length_results.md').write_text(content)
    print(content, flush=True)
    return summary


def run(args):
    selection = json.loads(args.selection.read_text())
    if selection.get('processor_fast',False):
        raise ValueError('capture-crops uses slow processor; fast selection is for the production prewarm, not this controlled replay')
    cfg = load_config(inherited=selection['config'])
    if cfg['approximate_precision'] or cfg['attention_impl'] != 'prompt_flash_attention':
        raise ValueError('capture must use non-approximate PromptFA baseline')
    for row in selection['selected']:
        if sha(row['image']) != row['image_sha256']:
            raise ValueError(f'image changed: {row["image"]}')
    root = args.output_dir.resolve()
    if args.resume:
        if not root.is_dir(): raise ValueError('--resume requires the existing sweep output directory')
    else:
        root.mkdir(parents=True,exist_ok=False)
    with ExitStack() as locks:
        locks.enter_context(ownership(root/'.sweep.lock'))
        state_path = root/'run_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        cache = args.cache_root.resolve() if args.cache_root else (
            Path(state['cache_root']) if state else root/'cache')
        cache.mkdir(parents=True,exist_ok=True)
        locks.enter_context(ownership(cache/'.vision_length_writer.lock'))
        run_owned(args,root,cache,selection,cfg,state)


def run_owned(args,root,cache,selection,cfg,state):
    code = sources()
    env = environment()
    def record(event,**values):
        row = dict(event=event,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
        with (root/'resume_history.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n');handle.flush()
        print('SWEEP '+json.dumps(row),flush=True)
    for name,value in [('selection.json',selection),('config.json',cfg)]:
        path = root/name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'{name} differs from existing run; use a new output directory')
        if not path.exists(): save(path,value)
    wanted = dict(cache_root=str(cache),steps=args.steps,repeats=args.repeats,
        source_hashes=code,environment=env)
    if state is not None and state != wanted:
        raise ValueError('cache, steps, repeats, model/timing source or environment changed; use a new output directory')
    old = json.loads((root/'sweep.json').read_text()) if (root/'sweep.json').exists() else None
    if old and (old['steps'],old['repeats']) != (args.steps,args.repeats):
        raise ValueError('steps/repeats differ from saved sweep')
    receipt = root/'capture.receipt'
    if (receipt/'command.json').exists() and Path(command_value(receipt,'--cache-root')).resolve() != cache:
        raise ValueError('cache root differs from saved capture; resume with the original cache root')
    legacy = args.resume and state is None
    record('start',resume=args.resume,cache_root=str(cache),legacy_resume=legacy,
        environment=env,source_hashes=code)
    if legacy:
        record('legacy_environment_limit',note='Old sweeps lack a complete environment fingerprint. Saved source/model/tensor/config checks apply; retained pairs keep their original runtime metadata and receipts. Full old CANN/processor environment equality cannot be established.')
    if state is None: save(root/'run_state.json',wanted)
    bench = HERE/'bench_production_vision_attention.py'
    command = [sys.executable,'-u',str(bench),'capture-crops','--model',selection['model'],
        '--config-json',str(root/'config.json'),'--cache-root',str(cache),'--output-dir',str(root/'capture')]
    for row in selection['selected']: command.extend(['--image',row['image']])
    if args.resume and complete(receipt):
        record('reuse_capture')
    else:
        if args.resume:
            preserve_attempt(root,['capture','capture.receipt'],'capture incomplete; recapture using existing graph cache')
        run_lane(command,receipt,args.capture_timeout_s)
    manifest = check_capture(root,selection,cfg,code)
    routes = list(manifest['routes'])
    lanes = [dict(name=f'{route}_r{repeat+1}_{variant}',route=route,repeat=repeat,variant=variant)
        for route in routes for repeat in range(args.repeats) for variant in ['baseline','pfa_approx']]
    metadata = dict(routes=routes,repeats=args.repeats,lanes=lanes,
        scope=selection['scope'],steps=args.steps,order='baseline, approximate, baseline, approximate per crop')
    if old and old != metadata: raise ValueError('saved sweep schedule differs; refusing to mix experiments')
    if not old: save(root/'sweep.json',metadata)
    try:
        for index in range(0,len(lanes),2):
            pair = lanes[index:index+2]
            if args.resume and all(reusable_lane(root,lane,manifest,cfg,args.steps,code) for lane in pair):
                results = [json.loads((root/lane['name']/'result.json').read_text()) for lane in pair]
                for field in ['device','soc_version','torch','torch_npu']:
                    if results[0][field] != results[1][field]:
                        raise ValueError(f'saved pair runtime differs: {field}')
                record('retain_pair',lanes=[lane['name'] for lane in pair])
                continue
            if args.resume:
                names = [name for lane in pair for name in [lane['name'],lane['name']+'.receipt']]
                preserve_attempt(root,names,'pair incomplete; rerun both adjacent members without changing measurement method')
                record('run_pair',lanes=[lane['name'] for lane in pair])
            for lane in pair:
                command = [sys.executable,'-u',str(bench),'replay','--capture-dir',str(root/'capture'),
                    '--route',lane['route'],'--variant',lane['variant'],'--cache-root',str(cache),
                    '--output-dir',str(root/lane['name']),'--steps',str(args.steps)]
                print('SWEEP start '+lane['name'],flush=True)
                run_lane(command,root/(lane['name']+'.receipt'),args.lane_timeout_s)
                result = json.loads((root/lane['name']/'result.json').read_text())
                if result['status'] not in ['completed','unsupported_on_this_device']:
                    raise RuntimeError(f'invalid lane {lane["name"]}: {result["status"]}')
    finally:
        report(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('select')
    s.add_argument('--model', type=Path, required=True)
    s.add_argument('--manifest', type=Path, default=HERE.parent/'crops'/'hotswap_100_manifest.json')
    s.add_argument('--config-json', type=Path, default=HERE/'vision_length_config.json')
    s.add_argument('--per-bucket', type=int, default=2)
    s.add_argument('--processor',choices=['slow','fast'],default='slow')
    s.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--selection', type=Path, required=True)
    r.add_argument('--output-dir', type=Path, required=True)
    r.add_argument('--cache-root', type=Path, help='Independent reusable graph-cache directory; default OUTPUT/cache')
    r.add_argument('--resume', action='store_true', help='Reuse verified capture and complete pairs in an existing output directory')
    r.add_argument('--steps', type=int, default=30)
    r.add_argument('--repeats', type=int, default=2)
    r.add_argument('--capture-timeout-s', type=int, default=3600)
    r.add_argument('--lane-timeout-s', type=int, default=1800)
    a = sub.add_parser('report')
    a.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    for field in ['per_bucket','steps','repeats','capture_timeout_s','lane_timeout_s']:
        if hasattr(args, field) and getattr(args, field) <= 0:
            p.error(f'{field} must be positive')
    if args.mode == 'select': select(args)
    elif args.mode == 'run': run(args)
    else: report(args.output_dir)


if __name__ == '__main__':
    main()
