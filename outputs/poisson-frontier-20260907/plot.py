import json
import argparse
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, MaxNLocator

root = Path(__file__).resolve().parent
rows = json.loads((root / 'results.json').read_text())
assert len(rows) == 30 and all(r['valid'] for r in rows)
parser = argparse.ArgumentParser()
parser.add_argument('--comparison', action='store_true')
parser.add_argument('--vllm-results', type=Path)
parser.add_argument('--output-dir', type=Path, default=root)
args = parser.parse_args()


def frontier(points, metric):
    return sorted([r for r in points if not any(
        s['target_qps'] >= r['target_qps'] and s[metric] <= r[metric]
        and (s['target_qps'] > r['target_qps'] or s[metric] < r[metric])
        for s in points)], key=lambda r: r['target_qps'])


def comparison_charts():
    """Only observed 1k results; never substitute 100-request measurements."""
    vllm = json.loads(args.vllm_results.read_text()) if args.vllm_results else []
    assert all(r['count'] == 1000 and r['valid'] for r in rows + vllm)
    assert all(r['batch'] == 64 and r['token_budget'] == 16384 for r in vllm)
    assert len({r['sequence_sha256'] for r in rows + vllm}) == 1
    assert len({r['target_qps'] for r in vllm}) == len(vllm)
    expected_rates = {r['target_qps'] for r in rows}
    complete = {r['target_qps'] for r in vllm} == expected_rates
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False})
    manifest = dict(custom_source=str(root/'results.json'),
                    vllm_source=str(args.vllm_results) if args.vllm_results else None,
                    status='complete' if complete else 'draft',
                    x_axis='request latency, seconds, logarithmic',
                    y_axis='offered Poisson QPS', curves={})
    for metric, title, name in [('p95_s', 'P95 latency', 'pareto-p95'),
                               ('mean_s', 'Mean latency', 'pareto-mean')]:
        candidates = [r for r in rows if metric != 'p95_s' or r['target_qps'] not in (7, 8)]
        custom = frontier(candidates, metric)
        measured = sorted(vllm, key=lambda r: r['target_qps'])
        fig, ax = plt.subplots(figsize=(11.2, 6.8), dpi=180)
        fig.subplots_adjust(left=.105, right=.965, bottom=.14, top=.78)
        fig.text(.105, .94, f'Table OCR: {title}', fontsize=24, weight='bold', color='#172D3A')
        fig.text(.105, .884, 'PaddleOCR-VL-1.6 · OmniDocBench v1.6 · One 910B2 · 1,000 requests per point',
                 fontsize=11.5, color='#546878')
        ax.set_xscale('log')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E4E9ED', linewidth=.7)
        ax.grid(axis='x', which='major', color='#E4E9ED', linewidth=.7)
        series = [('Custom (Pareto frontier)', custom, '#087F8C', 'o'),
                  ('vLLM-Ascend', measured, '#DB7837', 's')]
        for label, points, color, marker in series:
            if points:
                ax.plot([r[metric] for r in points], [r['target_qps'] for r in points],
                        color=color, marker=marker, linewidth=2.5, markersize=6,
                        markeredgecolor='white', markeredgewidth=.9, label=label)
                for point in points:
                    # Label the custom curve into the open space between curves;
                    # this also keeps low-latency labels off the y axis.
                    is_custom = label.startswith('Custom')
                    ax.annotate(f"{point['target_qps']:g} QPS · {point[metric]:.2f} s",
                                (point[metric], point['target_qps']),
                                xytext=(10, 6) if is_custom else (-10, 12),
                                textcoords='offset points', ha='left' if is_custom else 'right',
                                va='bottom', fontsize=10.5, color='#172D3A',
                                bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1.5))
            else:
                ax.plot([], [], color=color, marker=marker, linewidth=2.5,
                        label='vLLM-Ascend (results pending)')
        xs = [r[metric] for r in custom + measured]
        lo, hi = min(xs), max(xs)
        margin = max(math.log(hi/lo)*.09, .2)
        ax.set_xlim(lo/math.exp(margin), hi*math.exp(margin))
        xmin, xmax = ax.get_xlim()
        # Equal log-axis spacing: 1, 1.5, 2.25, 3.375, ... (and below 1).
        first_power = math.ceil(math.log(xmin, 1.5))
        last_power = math.floor(math.log(xmax, 1.5))
        ticks = [1.5**power for power in range(first_power, last_power + 1)]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2f}'.rstrip('0').rstrip('.')))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_ylim(.6, 8.45)
        ax.set_yticks(range(1,9))
        ax.set_xlabel(f'{title} (seconds, log scale)', labelpad=13)
        ax.set_ylabel('Offered Poisson QPS', labelpad=12)
        ax.legend(loc='upper left', ncol=1,
                  frameon=False, borderaxespad=1, fontsize=11)
        for spine in ax.spines.values():
            spine.set_color('#B6C1C8')
        if not complete:
            fig.text(.965,.94,'DRAFT',ha='right',fontsize=10,color='#7A8790')
        for extension in ('png','svg'):
            fig.savefig(args.output_dir/f'{name}.{extension}', facecolor='white')
        plt.close(fig)
        manifest['curves'][metric] = dict(custom=custom, vllm=measured,
            omitted_custom_qps=[7,8] if metric == 'p95_s' else [])
        print(args.output_dir/f'{name}.png')
    (args.output_dir/'pareto-comparison-data.json').write_text(json.dumps(manifest,indent=2)+'\n')
    combined_chart(manifest)


def combined_chart(manifest):
    """Overlay the two already-selected metric frontiers without changing data."""
    fig, ax = plt.subplots(figsize=(13.6, 8.2), dpi=180)
    fig.subplots_adjust(left=.085, right=.965, bottom=.12, top=.83)
    fig.text(.085, .95, 'Table OCR: Mean & P95 latency', fontsize=24,
             weight='bold', color='#172D3A')
    fig.text(.085, .905,
             'PaddleOCR-VL-1.6 · OmniDocBench v1.6 · One 910B2 · 1,000 requests per point',
             fontsize=11.5, color='#546878')
    all_x = []
    for system, color in [('custom', '#087F8C'), ('vllm', '#DB7837')]:
        for metric, metric_label, style, marker in [
                ('mean_s', 'Mean', '-', 'o'), ('p95_s', 'P95', '--', 's')]:
            points = manifest['curves'][metric][system]
            all_x.extend(r[metric] for r in points)
            label = f"{'Custom' if system == 'custom' else 'vLLM-Ascend'} · {metric_label}"
            ax.plot([r[metric] for r in points], [r['target_qps'] for r in points],
                    color=color, linestyle=style, marker=marker, linewidth=2.3,
                    markersize=5.5, markeredgecolor='white', markeredgewidth=.8,
                    label=label)
            for r in points:
                right_label = system == 'vllm' and metric == 'p95_s'
                ax.annotate(f"{r['target_qps']:g} QPS · {r[metric]:.2f} s",
                            (r[metric], r['target_qps']),
                            xytext=(8 if right_label else -8, -9 if metric == 'mean_s' else 9),
                            textcoords='offset points', ha='left' if right_label else 'right',
                            va='top' if metric == 'mean_s' else 'bottom',
                            fontsize=9.5, color='#172D3A',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1))
    ax.set_xscale('log')
    # Leave room for the left-facing point labels inside the plotting area.
    ax.set_xlim(min(all_x)/2.25, max(all_x)*2.25)
    xmin, xmax = ax.get_xlim()
    ticks = [1.5**p for p in range(math.ceil(math.log(xmin,1.5)),
                                  math.floor(math.log(xmax,1.5))+1)]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2f}'.rstrip('0').rstrip('.')))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_ylim(.4, 9.25)
    ax.set_yticks(range(1,9))
    ax.set_axisbelow(True)
    ax.grid(which='major', color='#E4E9ED', linewidth=.7)
    ax.set_xlabel('Request latency (seconds, log scale)', labelpad=12)
    ax.set_ylabel('Offered Poisson QPS', labelpad=12)
    ax.legend(loc='upper left', ncol=2, frameon=False,
              borderaxespad=1, fontsize=11, columnspacing=2.5, handlelength=3)
    for spine in ax.spines.values():
        spine.set_color('#B6C1C8')
    for extension in ('png', 'svg'):
        fig.savefig(args.output_dir/f'pareto-mean-p95.{extension}', facecolor='white')
    plt.close(fig)
    print(args.output_dir/'pareto-mean-p95.png')


if args.comparison:
    comparison_charts()
    raise SystemExit(0)

front = sorted([r for r in rows if not any(
    s['target_qps'] >= r['target_qps'] and s['p95_s'] <= r['p95_s']
    and (s['target_qps'] > r['target_qps'] or s['p95_s'] < r['p95_s'])
    for s in rows)], key=lambda r: r['target_qps'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(12, 7.4), dpi=180)
fig.subplots_adjust(left=.095, right=.96, bottom=.12, top=.81)
fig.text(.095,.94,'Table OCR: QPS–latency Pareto frontier',fontsize=23,weight='bold',color='#172D3A')
fig.text(.095,.89,'One 910B2 · PaddleOCR-VL-1.6 · OmniDocBench v1.6 tables · 1,000 requests per point',
         fontsize=12,color='#546878')
ax.set_axisbelow(True)
ax.grid(color='#DFE6EC', linewidth=.7)
ax.set_xlim(1.5,6.65)
ax.set_ylim(.5,8.6)
ax.set_xticks([1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5])
ax.set_yticks(range(1,9))
ax.set_xlabel('P95 request latency (seconds)  →  lower is better',labelpad=13)
ax.set_ylabel('Offered Poisson arrival rate (QPS)  →  higher is better',labelpad=12)
ax.scatter([r['p95_s'] for r in rows],[r['target_qps'] for r in rows],
           s=59,color='#A5B3BF',edgecolor='white',linewidth=.9,zorder=3)
ax.plot([r['p95_s'] for r in front],[r['target_qps'] for r in front],
        color='#087F8C',linewidth=2.4,zorder=4)
ax.scatter([r['p95_s'] for r in front],[r['target_qps'] for r in front],
           s=90,color='#087F8C',edgecolor='white',linewidth=1.4,zorder=5)
offsets={1:(36,-14),2:(-37,12),3:(-37,12),4:(-37,10),5:(-38,10),
         5.5:(-39,13),6:(28,-20),6.5:(28,12),7:(-28,18),8:(-30,18)}
for r in front:
    dx,dy=offsets[r['target_qps']]
    ax.annotate(f"B{r['batch']} · {r['target_qps']:g} QPS",(r['p95_s'],r['target_qps']),
                xytext=(dx,dy),textcoords='offset points',fontsize=10.5,
                color='#172D3A',ha='center',va='center',
                bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=1.4))
for spine in ax.spines.values():
    spine.set_color('#AAB7C0')
ax.legend(handles=[
    Line2D([0],[0],color='#087F8C',marker='o',linewidth=2,label='Measured Pareto frontier'),
    Line2D([0],[0],color='#A5B3BF',marker='o',linestyle='',label='Other tested configurations')],
    loc='upper left',frameon=False,fontsize=11)
fig.savefig(root / 'pareto-frontier.png', facecolor='white')
fig.savefig(root / 'pareto-frontier.svg', facecolor='white')
print(root / 'pareto-frontier.png')
