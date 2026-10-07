"""Small, model-independent accounting helpers for warm forward profiles."""
import csv
import math
from collections import defaultdict
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Expected nonempty finite, nonnegative timing samples')
    def quantile(p):
        index = (len(values) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return dict(count=len(values), mean=sum(values)/len(values), p50=quantile(.5),
                p90=quantile(.9), p99=quantile(.99), min=values[0], max=values[-1])


def interval_union_us(intervals):
    """Count overlapping device tasks once; kernel sums can exceed elapsed time."""
    end = None
    covered = 0.0
    for start, stop in sorted(intervals):
        if not all(math.isfinite(x) for x in (start, stop)) or stop < start:
            raise ValueError('Invalid device task interval')
        covered += max(0.0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return covered


def summarize_kernel_csv(path, steps):
    if steps <= 0:
        raise ValueError('Profile step count must be positive')
    groups = defaultdict(lambda: dict(count=0, duration_us=0.0))
    intervals = []
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        duration = float(row.get('Duration(us)') or row.get('Task Duration(us)') or 0)
        kind = row.get('Type') or row.get('Op Type') or row.get('Task Type') or 'unknown'
        group = groups[kind]
        group['count'] += 1
        group['duration_us'] += duration
        start = row.get('Start Time(us)')
        if start and duration > 0:
            start = float(start)
            intervals.append((start, start + duration))
    total = sum(g['duration_us'] for g in groups.values())
    union = interval_union_us(intervals)
    envelope = max(b for _, b in intervals)-min(a for a, _ in intervals) if intervals else None
    return dict(kernel_rows=len(rows), profile_steps=steps,
                kernel_duration_sum_ms_per_forward=total/steps/1000,
                interval_union_ms=union/1000 if intervals else None,
                interval_envelope_ms=envelope/1000 if envelope is not None else None,
                uncovered_envelope_ms=(envelope-union)/1000 if envelope is not None else None,
                timeline_scope='All captured task intervals; gaps include boundaries between forwards. '
                               'No claim of pure compute utilization or CPU causality.',
                top_types=[dict(type=k, **v, duration_ms_per_forward=v['duration_us']/steps/1000,
                                count_per_forward=v['count']/steps,
                                kernel_sum_percent=100*v['duration_us']/total if total else 0)
                           for k, v in sorted(groups.items(), key=lambda pair: pair[1]['duration_us'], reverse=True)])
