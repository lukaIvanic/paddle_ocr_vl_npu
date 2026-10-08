#!/usr/bin/env python3
"""Real-crop selection and paired full-encoder replay; never synthetic tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from vision_diagnostic_config import load_config
from vision_diagnostic_runner import run_lane

HERE = Path(__file__).resolve().parent


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


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
            drift=result.get('full_encoder_parity'), timing_gate=result.get('timing_gate')))
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
    summary = dict(scope=metadata['scope'], rows=rows, pairs=pairs,
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
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    save(root/'selection.json', selection)
    save(root/'config.json', cfg)
    cache = root/'cache'
    bench = HERE/'bench_production_vision_attention.py'
    command = [sys.executable, '-u', str(bench), 'capture-crops', '--model', selection['model'],
        '--config-json', str(root/'config.json'), '--cache-root', str(cache), '--output-dir', str(root/'capture')]
    for row in selection['selected']:
        path = Path(row['image'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError(f'image changed: {path}')
        command.extend(['--image', str(path)])
    run_lane(command, root/'capture.receipt', args.capture_timeout_s)
    manifest = json.loads((root/'capture'/'manifest.json').read_text())
    routes = list(manifest['routes'])
    if len(routes) != len(selection['selected']):
        raise ValueError('capture did not preserve selected crop count')
    for entry, chosen in zip(manifest['routes'].values(), selection['selected']):
        if (entry['tags']['real_tokens'], entry['bucket'], entry['image_sha256']) != (
                chosen['real_tokens'], chosen['bucket'], chosen['image_sha256']):
            raise ValueError('capture token grid/bucket/image differs from selection')
    lanes = [dict(name=f'{route}_r{repeat+1}_{variant}', route=route, repeat=repeat, variant=variant)
        for route in routes for repeat in range(args.repeats) for variant in ['baseline','pfa_approx']]
    save(root/'sweep.json', dict(routes=routes, repeats=args.repeats, lanes=lanes,
        scope=selection['scope'], steps=args.steps, order='baseline, approximate, baseline, approximate per crop'))
    try:
        for lane in lanes:
            command = [sys.executable, '-u', str(bench), 'replay', '--capture-dir', str(root/'capture'),
                '--route', lane['route'], '--variant', lane['variant'], '--cache-root', str(cache),
                '--output-dir', str(root/lane['name']), '--steps', str(args.steps)]
            print('SWEEP start '+lane['name'], flush=True)
            run_lane(command, root/(lane['name']+'.receipt'), args.lane_timeout_s)
            result = json.loads((root/lane['name']/'result.json').read_text())
            if result['status'] not in ['completed','unsupported_on_this_device']:
                raise RuntimeError(f"invalid lane {lane['name']}: {result['status']}")
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
