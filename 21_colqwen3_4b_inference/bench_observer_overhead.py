"""Development-only ABBA check of observer overhead on complete HR-dev runs.

No production observation-level toggle. Both controls execute the SAME pipeline,
including scoring, validation and outputs. Never replays an isolated section.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


class UnobservedControl:
    """Private test fixture; deliberately not exposed by the pipeline CLI."""
    def __init__(self, root, profile=False):
        assert not profile
        self.start=time.monotonic()
    def emit(self, *args, **kwargs):
        pass
    @contextmanager
    def section(self, *args, **kwargs):
        yield
    def resolve(self):
        pass
    def complete(self, *args, **kwargs):
        pass
    def close(self):
        pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--dataset-root',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    rows=[]
    baseline=None
    for index,observed in enumerate((False,True,True,False)):
        root=a.output_dir/f'{index}_{"observed" if observed else "control"}'
        if observed:
            command=[sys.executable,'-u',str(Path(__file__).with_name('run_hr_evaluation.py'))]
        else:
            code=(f'import sys; sys.path.insert(0,{str(Path(__file__).resolve().parent)!r}); '
                  'from bench_observer_overhead import UnobservedControl; '
                  'from run_hr_evaluation import main; main(observer_factory=UnobservedControl)')
            command=[sys.executable,'-u','-c',code]
        command+=['--model',a.model,'--dataset-root',a.dataset_root,'--output-dir',str(root),'--workload','dev']
        (a.output_dir/f'{index}.command.json').write_text(json.dumps(command,indent=2)+'\n')
        print(json.dumps(dict(phase='comparison_run_start',index=index,observed=observed)),flush=True)
        start=time.perf_counter()
        with (a.output_dir/f'{index}.log').open('w') as out:
            subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=300)
        r=json.loads((root/'result.json').read_text())
        score=np.load(root/'scores.npy')
        ids=json.loads((root/'ids.json').read_text())
        if baseline is None:
            baseline=(score,ids)
        if ids!=baseline[1] or not np.array_equal(score,baseline[0]):
            raise RuntimeError('Observer changed scores or workload')
        rows.append(dict(index=index,observed=observed,subprocess_s=time.perf_counter()-start,
                         page_s=r['page_encoding_s'],query_s=r['query_encoding_s'],
                         scoring_s=r['scoring_s'],encoding_s=r['encoding_s'],total_s=r['total_s'],
                         page_per_s=r['page_per_s'],scores_exact=True))
        summary=dict(runs=rows,status='running')
        if len(rows)==4:
            summary['status']='completed'
            for key in ('page_s','query_s','encoding_s','scoring_s','total_s'):
                on=np.mean([x[key] for x in rows if x['observed']])
                off=np.mean([x[key] for x in rows if not x['observed']])
                summary[key]=dict(observed_mean_s=on,control_mean_s=off,overhead_percent=100*(on/off-1))
        (a.output_dir/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(phase='comparison_run_finish',**rows[-1])),flush=True)

if __name__=='__main__':
    main()
