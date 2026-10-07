"""Pinned BGE grouped supervised/distillation loss with the existing Qwen NPU interface."""
import argparse
import collections
import os
from pathlib import Path
import platform
import subprocess
import time
import json

from distill_runtime import Runtime, read, digest, model_manifest, save, plans
from margin_distillation import (agreement, benchmark_metrics, lr_at,
                                validate_teacher_inputs)

from bge_baseline import loss_and_score_gradient, PIN
from bge_filtered_runtime import update_windows, filtered_reference_baseline, validation_objective

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['teacher', 'profile', 'train'], required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--teacher', type=Path)
    p.add_argument('--control', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--schedule', choices=['constant', 'warmup_linear'], default='constant')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--profile-updates', type=int, default=3)
    p.add_argument('--profile-eval', action='store_true', help='Time unchanged-weight evaluation during optimizer-free profiling')
    p.add_argument('--queries-per-update', type=int, default=32)
    p.add_argument('--batch-schedule', choices=['contiguous', 'contiguous_partial', 'retained_original_slots'], default='contiguous')
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--wall-time-limit', type=float, default=2400)
    p.add_argument('--student-order', choices=['query_first', 'contents_swapped', 'document_first'], default='query_first')
    p.add_argument('--query-first-reference', type=Path,
                   help='Completed query-first student result.json supplying the original baseline')
    p.add_argument('--reserved-eval-steps', type=int, nargs='*', default=[],
                   help='Additional evaluated updates receiving the reserved panel')
    args = p.parse_args()
    if args.mode == 'teacher':
        assert args.student_order == 'query_first', 'Teacher targets remain query first'
    if args.student_order != 'query_first':
        assert args.query_first_reference, 'Reordered input requires the original query-first reference'
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    data = read(args.dataset)
    assert data['reference']['code_commit'] == PIN['code_commit']
    runtime = Runtime(args.model)
    torch = runtime.torch
    import torch_npu
    import transformers
    sections = ['train', 'validation', 'benchmark', 'reserved_benchmark']
    if args.student_order != 'query_first':
        for s in sections:
            runtime.records(data[s], s)
    canonical_lengths = dict(runtime.lengths)
    records = {s: runtime.records(data[s], s, args.student_order) for s in sections}
    if args.student_order == 'query_first':
        canonical_lengths = dict(runtime.lengths)
    model = runtime.load(args.model)
    result = {'status': 'running', 'dataset_sha256': digest(args.dataset),
              'model_sha256': model_manifest(args.model), 'lengths': runtime.lengths,
              'environment': {'host': platform.node(), 'chip': 'Ascend 910B2',
                              'physical_npu': os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
                              'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                              'transformers': transformers.__version__},
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'distribution': data['distribution'], 'split_checks': data['checks'],
              'evaluations': {}, 'updates': [], 'checkpoints': []}
    save(args.output / 'result.json', result)
    scoring_config = {'order':'query_first','margin':'yes-minus-no raw logits','max_length':8192,'body_truncation':'right before fixed suffix','weights':'FP32','autocast':'BF16','attention':'fusion_attention','tokenizer_files':{f.name:digest(f) for f in args.model.glob('*token*') if f.is_file()},'prefix_suffix_source_sha256':digest(Path(__file__).parent/'transformers_rerank.py')}
    if args.mode == 'teacher':
        target_path = args.output / 'teacher.json'
        target = read(target_path) if target_path.exists() else {
            'dataset_sha256': result['dataset_sha256'], 'model_sha256': result['model_sha256'],
            'lengths': runtime.lengths,
            'scoring': 'FP32 weights, BF16 autocast, yes-minus-no logits, query first, total max8192',
            'scoring_config':scoring_config, 'scores': {}, 'seconds': {}}
        assert target['dataset_sha256'] == result['dataset_sha256']
        assert target['model_sha256'] == result['model_sha256']
        assert target['scoring_config'] == scoring_config
        assert target['lengths'] == runtime.lengths
        for section, rows in records.items():
            if section in target['scores']:
                continue
            scores, seconds = runtime.score(model, rows, section)
            target['scores'][section], target['seconds'][section] = scores, seconds
            if 'benchmark' in section:
                target.setdefault('metrics', {})[section] = benchmark_metrics(data[section], scores)
            save(target_path, target)
            print('TEACHER_SECTION', json.dumps({'section': section, 'seconds': seconds}), flush=True)
        result['status'] = 'completed'
        result['total_seconds'] = time.monotonic() - started
        save(args.output / 'result.json', result)
        return
    assert args.teacher and args.control
    control = read(args.control)
    assert control['passed'], 'Implementation control must pass before training'
    assert control['reference_commit'] == PIN['code_commit'], 'Pinned reference differs'
    teacher = read(args.teacher)
    assert teacher['dataset_sha256'] == result['dataset_sha256']
    validate_teacher_inputs(teacher['lengths'], canonical_lengths, runtime.lengths, args.student_order)
    result['teacher_query_first_lengths'] = canonical_lengths
    if args.query_first_reference:
        reference = read(args.query_first_reference)
        assert reference['status'] == 'completed'
        assert reference['model_sha256'] == result['model_sha256']
        assert reference['recipe']['prompt_order'] == 'query_first'
        result['query_first_reference_sha256'] = digest(args.query_first_reference)
        if reference['dataset_sha256'] == result['dataset_sha256']:
            assert reference['lengths'] == canonical_lengths
            assert reference['teacher_sha256'] == digest(args.teacher)
            result['query_first_baseline'] = reference['evaluations']['0']
        else:
            result['query_first_baseline'] = filtered_reference_baseline(
                data, reference, digest(args.query_first_reference), teacher, canonical_lengths)
            result['reference_comparison_scope'] = 'Same benchmark queries and released initialization; prior query-first training included groups removed by the disclosed length filter.'
    result['teacher_sha256'] = digest(args.teacher)
    result['control_sha256'] = digest(args.control)
    result['recipe'] = {'loss': 'BGE supervised group CE + teacher-distribution CE, weights 1:1, mean query groups',
        'optimizer': 'fresh NpuFusedAdamW, betas0.9/0.999, eps1e-8, weight_decay0',
        'gradient_clip': 1.0, 'microbatch_max': 4, 'train_token_budget': 8192,
        'padding': 'left, microbatch max rounded up to128', 'prompt_order': args.student_order,
        'score_gradient_replay': 'deterministic no-grad pass followed by microbatch backward',
        'weights': 'FP32', 'autocast': 'BF16', 'warmup_updates': 5, 'reference_commit': PIN['code_commit'], 'optimizer_updates_allowed': args.mode == 'train'}
    optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=args.learning_rate,
                        betas=(.9, .999), eps=1e-8, weight_decay=0)
    assert not optimizer.state
    by_group = collections.defaultdict(list)
    for r in records['train']:
        by_group[r['group_id']].append(r)
    windows = update_windows(data['train'], args.steps, args.queries_per_update, args.batch_schedule)
    result['training_group_counts_per_update'] = [len(w) for w in windows]
    if args.batch_schedule == 'retained_original_slots':
        assert data['derivation']['kind'] == 'whole_group_length_filter'
    assert all(len(g['documents']) == 8 for g in data['train'])

    def checkpoint(step):
        t = time.monotonic()
        def cpu(x):
            if isinstance(x, torch.Tensor): return x.detach().cpu()
            if isinstance(x, dict): return {k: cpu(v) for k, v in x.items()}
            if isinstance(x, list): return [cpu(v) for v in x]
            if isinstance(x, tuple): return tuple(cpu(v) for v in x)
            return x
        state = {'model': cpu(model.state_dict()), 'optimizer': cpu(optimizer.state_dict()),
                 'scheduler': {'schedule': args.schedule, 'completed_updates': step,
                               'steps': args.steps, 'peak_lr': args.learning_rate, 'warmup': 5},
                 'rng': torch.get_rng_state(), 'npu_rng': torch.npu.get_rng_state(),
                 'dataset_sha256': result['dataset_sha256'], 'teacher_sha256': result['teacher_sha256'],
                 'training_order_ids': [g['id'] for g in data['train']], 'config': result['config']}
        path = args.output / f'checkpoint_{step:03d}.pt'
        torch.save(state, path.with_suffix('.partial'))
        path.with_suffix('.partial').replace(path)
        result['checkpoints'].append({'step': step, 'path': str(path), 'bytes': path.stat().st_size,
                                      'seconds': time.monotonic() - t})
        print('CHECKPOINT', json.dumps(result['checkpoints'][-1]), flush=True)

    # Both curves have fixed membership; the 180-query view never replaces 108.
    import hashlib
    panel_key = lambda g: (g['task'], str(g['qid']))
    frequent_keys = {panel_key(g) for g in data['benchmark']}
    reserved_keys = {panel_key(g) for g in data['reserved_benchmark']}
    assert len(data['benchmark']) == len(frequent_keys) == 108
    assert len(data['reserved_benchmark']) == len(reserved_keys) == 72
    assert not frequent_keys & reserved_keys
    result['evaluation_panels'] = {
        'trajectory_108': {'queries': 108, 'steps': [0, 1, 3, 10, 25, 50]},
        'baseline_endpoint_180': {'queries': 180, 'steps': [0, args.steps]},
        'membership_sha256': hashlib.sha256(json.dumps(
            {'frequent': sorted(frequent_keys), 'reserved': sorted(reserved_keys)}
        ).encode()).hexdigest()}

    def evaluate(step, endpoint=False):
        t = time.monotonic()
        scores, seconds = runtime.score(model, records['benchmark'], 'benchmark')
        val, val_s = runtime.score(model, records['validation'], 'validation')
        item = {'benchmark': benchmark_metrics(data['benchmark'], scores),
                'agreement': agreement(data['validation'], val, teacher['scores']['validation']),
                'seconds': {'benchmark': seconds, 'validation': val_s},
                'benchmark_scores': scores, 'validation_scores': val}
        item['validation_objective'] = validation_objective(val, teacher['scores']['validation'])
        item['trajectory_108'] = item['benchmark']
        if endpoint or step == 0:
            rs, rt = runtime.score(model, records['reserved_benchmark'], 'reserved_benchmark')
            item['reserved_benchmark'] = benchmark_metrics(data['reserved_benchmark'], rs)
            item['reserved_scores'] = rs
            item['baseline_endpoint_180'] = benchmark_metrics(
                data['benchmark'] + data['reserved_benchmark'], scores | rs)
            item['seconds']['reserved_benchmark'] = rt
        item['seconds']['total'] = time.monotonic() - t
        result['evaluations'][str(step)] = item
        save(args.output / 'result.json', result)
        print('EVALUATION', json.dumps({'step': step,
              'trajectory_108': item['trajectory_108']['suite_macro_ndcg10'],
              'baseline_endpoint_180': item.get('baseline_endpoint_180', {}).get('suite_macro_ndcg10'),
              'agreement': item['agreement'], 'validation_objective': item['validation_objective'], 'seconds': item['seconds']}), flush=True)
        if 'query_first_baseline' in result:
            original = result['query_first_baseline']['benchmark']['suite_macro_ndcg10']
            print('QUERY_FIRST_COMPARISON', json.dumps({'step': step,
                  'original': original,
                  'delta_ndcg_points': {lang: 100 * (value-original[lang]) for lang, value in
                                       item['benchmark']['suite_macro_ndcg10'].items()}}), flush=True)
        assert seconds + val_s < 180, 'Frequent evaluation exceeds three-minute budget; resize fixture before training'

    try:
        if args.mode != 'profile' or args.profile_eval: evaluate(0)
        torch.npu.reset_peak_memory_stats()
        for step in range(1, args.steps + 1):
            if time.monotonic() - started > args.wall_time_limit:
                result['status'] = 'time_limit'
                if args.mode != 'profile': checkpoint(step - 1)
                break
            begin = time.monotonic()
            # This checkpoint has no dropout: eval mode retains full autograd.
            model.eval()
            optimizer.zero_grad(set_to_none=False)
            learning_rate = lr_at(step, args.steps, args.learning_rate, args.schedule)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            window = windows[step - 1]
            total_loss = torch.zeros((), device=runtime.device)
            micros = 0
            for group in window:
                rows = by_group[group['id']]
                microplan = list(plans(rows, 4, 8192))
                values = torch.empty(8, device=runtime.device)
                with torch.no_grad():
                    for micro in microplan:
                        z = runtime.logits(model, micro)
                        values[[r['candidate'] for r in micro]] = z
                target = torch.tensor(teacher['scores']['train'][group['id']], device=runtime.device)
                loss, gradient = loss_and_score_gradient(values, target)
                total_loss += loss.detach() / len(window)
                for micro in microplan:
                    z = runtime.logits(model, micro)
                    indices = [r['candidate'] for r in micro]
                    assert torch.equal(z.detach(), values[indices]), 'Replay must be deterministic'
                    (z * (gradient[indices] / len(window))).sum().backward()
                    micros += 1
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if args.mode != 'profile': optimizer.step()
            torch.npu.synchronize()
            row = {'step': step, 'optimizer_step_applied':args.mode == 'train', 'lr': learning_rate, 'loss': total_loss.item(), 'gradient_norm': norm.item(),
                   'seconds': time.monotonic() - begin, 'queries': len(window), 'pairs': len(window)*8,
                   'backward_microbatches': micros, 'peak_allocated_gib': torch.npu.max_memory_allocated()/1024**3}
            result['updates'].append(row)
            print('UPDATE', json.dumps(row), flush=True)
            if args.mode == 'profile':
                if step >= args.profile_updates:
                    result['status'] = 'profile_completed_no_optimizer_updates'
                    break
                continue
            if step in {1, 3, 10, 25, 50} or step == args.steps:
                checkpoint(step)
                evaluate(step, endpoint=step == args.steps or step in args.reserved_eval_steps)
            save(args.output / 'result.json', result)
        else:
            result['status'] = 'completed'
    except Exception as e:
        result.update(status='failed', error=repr(e))
        raise
    finally:
        result['total_seconds'] = time.monotonic() - started
        save(args.output / 'result.json', result)
        print('FINISHED', json.dumps({'status': result['status'], 'updates': len(result['updates']),
                                    'seconds': result['total_seconds']}), flush=True)


if __name__ == '__main__':
    main()
