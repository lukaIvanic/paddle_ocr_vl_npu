"""Small real-workload ABBA validation; no isolated model-stage replay."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time

repo=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(repo/'25_clef_inference'))


class Control:
    """Development-only control: same requests and validation, no observation."""
    instrument_model=False
    def __init__(self, root, profile=False):
        assert not profile
        self.start=time.perf_counter(); self.rows=[]; self.pending=[]
    def emit(self,*args,**kwargs): pass
    def close(self): pass
    @contextmanager
    def section(self,*args,**kwargs): yield
    @contextmanager
    def item(self,row):
        row['sections']=[]
        start=time.perf_counter()
        try: yield row
        finally:
            row['wall_s']=time.perf_counter()-start
            row['unattributed_host_s']=row['wall_s']
            self.rows.append(row)


def main():
    from run_reranking_smoke import digest, save
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    fixture_path=repo/'tmp/25_clef_inference/reranking_lengths_44ca1ad4/fixture.json'
    source=json.loads(fixture_path.read_text())
    selection=[0,25,39]
    fixture=dict(source,pairs=[source['pairs'][i] for i in selection],expected_pairs=len(selection),
        quality_scope='Three-case plumbing validation: first pair, prior Medical boundary case, longest pair; not benchmark accuracy or aggregate latency')
    path=args.output_dir/'fixture.json'; save(path,fixture)
    output={'status':'running','original_fixture_sha256':digest(fixture_path),'selected_indices':selection,'runs':[]}
    baseline=None
    try:
        for index,observed in enumerate((False,True,True,False,True)):
            profile=index==4
            folder=args.output_dir/f'{index}_{"profile" if profile else "observed" if observed else "control"}'
            if observed:
                command=[sys.executable,'-u',str(repo/'25_clef_inference/benchmark_local.py')]
            else:
                code=(f'import sys; sys.path.insert(0,{str(Path(__file__).resolve().parent)!r}); '
                      'from validate import Control; from benchmark_local import main; main(observer_factory=Control)')
                command=[sys.executable,'-u','-c',code]
            command+=['--model',str(args.model),'--fixture',str(path),'--output-dir',str(folder),'--mode','all']
            if profile: command+=['--profile']
            save(args.output_dir/f'command-{index}.json',command)
            with (args.output_dir/f'run-{index}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((folder/'result.json').read_text())
            assert result['status']=='completed' and result['pending_device_events']==0
            assert not result['transformers_imported']
            if baseline is None: baseline=result['scores']
            assert baseline==result['scores'],'Instrumentation changed scores'
            assert result['model_source_sha256']==digest(repo/'25_clef_inference/local_modeling_clef.py')
            entry={'observed':observed,'profile':profile,'scores_exact':True,
                   'total_s':result['total_s'],'phase_s':{k:v['phase_wall_s_including_observation'] for k,v in result['phase_memory'].items()}}
            if profile:
                traces=list((folder/'profiler').rglob('trace_view.json'))
                assert len(traces)==3, f'Expected 3 real-item traces, got {len(traces)}'
                ranges=set()
                for trace in traces:
                    content=json.loads(trace.read_text())
                    events=content.get('traceEvents',[]) if isinstance(content,dict) else content
                    ranges.update(e.get('name','') for e in events if e.get('name','').startswith('clef/'))
                assert 'clef/gdn_scan' in ranges and 'clef/decision_head' in ranges
                entry.update(traces=len(traces),clef_ranges=sorted(ranges))
            output['runs'].append(entry)
            save(args.output_dir/'validation.json',output)
            print(json.dumps({'run':index,**entry}),flush=True)
        output['overhead_percent']={}
        for phase in ('prepare','uncached','cached'):
            on=sum(r['phase_s'][phase] for r in output['runs'][:4] if r['observed'])/2
            off=sum(r['phase_s'][phase] for r in output['runs'][:4] if not r['observed'])/2
            output['overhead_percent'][phase]=100*(on/off-1)
        output['status']='completed'
    except BaseException as exc:
        output.update(status='failed',error=repr(exc))
        raise
    finally: save(args.output_dir/'validation.json',output)


if __name__=='__main__': main()
