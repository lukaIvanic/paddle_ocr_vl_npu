"""Production vision event statistics; also explains existing run summaries."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def group_stats(rows):
    times = [r["device_s"] for r in rows]
    if any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError("vision samples must have finite positive device duration")
    total = sum(times)
    real = sum(r["real_tokens"] for r in rows)
    physical = sum(r["physical_tokens"] for r in rows)
    return {
        "calls": len(rows), "members": sum(r["members"] for r in rows),
        "device_s": total, "real_tokens": real, "physical_tokens": physical,
        "real_tok_s": real / total, "physical_tok_s": physical / total,
        "useful_token_fraction": real / physical,
        "latency_ms": {
            "mean": statistics.mean(times) * 1000,
            "std": statistics.pstdev(times) * 1000,
            "min": min(times) * 1000, "max": max(times) * 1000,
            **{f"p{p}": percentile(times, p / 100) * 1000 for p in (50, 90, 95, 99)},
        },
    }


def summarize_vision_samples(rows):
    routes = defaultdict(list)
    shapes = defaultdict(list)
    for row in rows:
        routes[row["route"]].append(row)
        shapes[f'{row["route"]}:S{row["physical_tokens"]}'].append(row)
    return {
        "schema_version": 1,
        "scope": "vision_transformer_blocks_device_event_region",
        "percentile_method": "linear interpolation at (n-1)*q; per-call unweighted",
        "rate_method": "sum(tokens)/sum(device_s); never mean of per-call rates",
        "notes": "Production event regions include launch gaps; direct routes include padding preparation. Packed mask construction is outside the region. Not pure kernel time.",
        "all": group_stats(rows),
        "by_route": {k: group_stats(v) for k, v in sorted(routes.items())},
        "by_exact_shape": {k: group_stats(v) for k, v in sorted(shapes.items())},
        "slowest_calls": sorted(rows, key=lambda r: r["device_s"], reverse=True)[:20],
    }


def explain(summary):
    wall = summary["pipeline_wall_s"]
    decode = summary["streaming"]["decode"]
    prefill = decode["prefill_metrics"]
    print(f'pages={summary["completed"]} failed={summary["failed"]} hot_wall_s={wall:.3f} pg/s={summary["completed"]/wall:.6f}')
    print(f'commit={summary["git_commit"]} PSE={summary["local_decode_increfa_length_mode"]}')
    print('Rates below use device-event stage time, except production output tok/s uses full hot wall time.')
    for label, tokens, seconds in (
        ("vision real", prefill["raw_vision_tokens"], prefill["vision_transformer_blocks"]),
        ("text prefill real", prefill["text_prefill_tokens"], prefill["text_transformer_prefill"]),
        ("decode effective", decode["decode_calls"], decode["decode_s"]),
        ("decode raw B32 slots", decode["raw_decode_token_slots"], decode["decode_s"]),
    ):
        print(f'{label}: tokens={tokens} device_s={seconds:.3f} tok/s={tokens/seconds:.3f}')
    print(f'decode mean graph ms={1000*decode["decode_s"]/decode["graph_calls"]:.3f}; active-slot occupancy={100*decode["active_slot_fraction"]:.3f}% (not NPU utilization)')
    print(f'production effective decode tokens/hot wall s={decode["decode_calls"]/wall:.3f} (prefill-sampled first tokens excluded)')
    timing = summary.get("vision_timing")
    if not timing:
        print('Per-route latency samples were not collected in this run; route counts and aggregate times remain available.')
        print(json.dumps(summary["local_compiled_vision"]["route_counts"], sort_keys=True))
        return
    print('route | calls | device s | real tok/s | physical tok/s | mean/p50/p90/p95/p99/max ms')
    for route, stats in timing["by_route"].items():
        latency = stats["latency_ms"]
        values = '/'.join(f'{latency[k]:.3f}' for k in ('mean','p50','p90','p95','p99','max'))
        print(f'{route} | {stats["calls"]} | {stats["device_s"]:.3f} | {stats["real_tok_s"]:.1f} | {stats["physical_tok_s"]:.1f} | {values}')
    print('Exact shapes, including eager overflow:')
    print(json.dumps(timing["by_exact_shape"], indent=2))
    print('Slowest calls:')
    print(json.dumps(timing["slowest_calls"][:5], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    explain(json.loads(args.summary.read_text()))
