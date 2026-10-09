"""Strict instruction-only derivation checks; published text is supplied by a manifest."""
import copy

SECTIONS = ('train', 'validation', 'benchmark', 'reserved_benchmark')


def validate_derivation(data, control, manifest):
    assert manifest['status'] == 'APPROVED'
    mapping = {row['source']: row['instruction'] for row in manifest['rows']}
    assert all(isinstance(v, str) and v for v in mapping.values())
    changes = {}
    for section in SECTIONS:
        assert len(data[section]) == len(control[section])
        changes[section] = []
        for new, old in zip(data[section], control[section]):
            expected = dict(old)
            if section in ('train', 'validation'):
                expected['instruction'] = mapping[old['source']]
            assert new == expected, (section, old['id'], 'Change beyond approved instruction')
            if new['instruction'] != old['instruction']:
                changes[section].append(new['id'])
    for key in control:
        if key not in (*SECTIONS, 'derivation'):
            assert data[key] == control[key], ('metadata changed', key)
    assert data['derivation']['control_derivation'] == control['derivation']
    return changes


def benchmark_reference(data, reference_data, reference, lengths):
    for section in ('benchmark', 'reserved_benchmark'):
        assert data[section] == reference_data[section]
        assert lengths[section] == reference['lengths'][section]
    baseline = copy.deepcopy(reference['evaluations']['0'])
    # Changed validation instructions invalidate old validation scores.
    for key in ('validation_scores', 'agreement', 'validation_objective', 'seconds'):
        baseline.pop(key, None)
    baseline['reference_projection'] = {'benchmark_scores_unchanged': True,
        'validation_scores_reused': False, 'comparison': 'Identical benchmark panels; instruction ablation'}
    return baseline
