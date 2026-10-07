"""Explicit scheduling and reference checks for the authorized length-filtered run."""
import copy
import math
from margin_distillation import agreement

def update_windows(groups, steps, batch_size, mode):
    if mode == 'retained_original_slots':
        windows = [[] for _ in range(steps)]
        positions = []
        for group in groups:
            prefix, index = group['id'].split('/')
            assert prefix == 'train'
            index = int(index)
            assert 0 <= index < steps * batch_size
            positions.append(index)
            windows[index // batch_size].append(group)
        assert positions == sorted(set(positions)), 'Retained order or uniqueness changed'
    else:
        if mode == 'contiguous':
            assert len(groups) >= steps * batch_size
        elif mode == 'contiguous_partial':
            assert (steps - 1) * batch_size < len(groups) <= steps * batch_size
        else:
            raise ValueError(mode)
        windows = [groups[i * batch_size:(i + 1) * batch_size] for i in range(steps)]
    assert len(windows) == steps and all(0 < len(w) <= batch_size for w in windows)
    return windows

def filtered_reference_baseline(data, reference, reference_sha, teacher, canonical_lengths):
    derivation = data['derivation']
    assert derivation['kind'] == 'whole_group_length_filter'
    assert derivation['parent_dataset_sha256'] == reference['dataset_sha256']
    assert derivation['parent_teacher_sha256'] == reference['teacher_sha256']
    assert derivation['parent_reference_sha256'] == reference_sha
    assert teacher['cache_derivation'] == derivation
    assert derivation['benchmark_unchanged']
    for section in ['benchmark', 'reserved_benchmark']:
        assert not derivation['removed_group_ids'][section]
        assert canonical_lengths[section] == reference['lengths'][section]
    baseline = copy.deepcopy(reference['evaluations']['0'])
    baseline['validation_scores'] = {g['id']:baseline['validation_scores'][g['id']] for g in data['validation']}
    baseline['agreement'] = agreement(data['validation'], baseline['validation_scores'], teacher['scores']['validation'])
    baseline['validation_objective'] = validation_objective(baseline['validation_scores'], teacher['scores']['validation'])
    baseline['reference_projection'] = {'benchmark_scores_unchanged':True,
        'validation_filtered_to_groups':len(data['validation']),
        'original_query_first_training_included_extra_groups':len(derivation['removed_group_ids']['train'])}
    return baseline

def validation_objective(scores, teacher):
    def lse(values):
        maximum = max(values)
        return maximum + math.log(sum(math.exp(v - maximum) for v in values))
    supervised = []
    distillation = []
    for key, values in scores.items():
        target = teacher[key]
        assert len(values) == len(target) == 8
        student_normalizer, teacher_normalizer = lse(values), lse(target)
        supervised.append(student_normalizer - values[0])
        distillation.append(sum(math.exp(t - teacher_normalizer) * (student_normalizer - s)
                                for s, t in zip(values, target)))
    sup, kd = sum(supervised)/len(supervised), sum(distillation)/len(distillation)
    return {'supervised_ce':sup, 'teacher_ce':kd, 'total':sup+kd, 'query_groups':len(scores)}


def expanded_reference_baseline(data, reference_data, reference, canonical_lengths, teacher):
    """Reuse released-weight scores only for provably identical evaluation inputs."""
    assert data['derivation']['kind'] == 'expanded_bge_length_filter'
    for section in ('benchmark', 'reserved_benchmark'):
        assert data[section] == reference_data[section]
        assert canonical_lengths[section] == reference['lengths'][section]
    old_validation = {g['id']:g for g in reference_data['validation']}
    assert all(g == old_validation[g['id']] for g in data['validation'])
    baseline = copy.deepcopy(reference['evaluations']['0'])
    baseline['validation_scores'] = {g['id']:baseline['validation_scores'][g['id']] for g in data['validation']}
    baseline['agreement'] = agreement(data['validation'], baseline['validation_scores'], teacher['scores']['validation'])
    baseline['validation_objective'] = validation_objective(baseline['validation_scores'], teacher['scores']['validation'])
    baseline['reference_projection'] = {'benchmark_scores_unchanged':True, 'validation_filtered_to_groups':len(data['validation']), 'comparison':'Released-weight baseline only; expanded training exposure differs from earlier runs'}
    return baseline
