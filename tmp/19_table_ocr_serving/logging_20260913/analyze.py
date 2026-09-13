"""Check the saved logging A/B runs, original native outputs and emitted events."""
import argparse
from collections import Counter
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    reference={r['sequence']:r for r in rows(args.reference/'results.jsonl')}
    schedule=rows(args.reference/'schedule.jsonl')
    report={}
    for name in ('control_cached','basic','detailed'):
        folder=args.root/name
        measured=folder/'b8/measured/results'
        results=rows(measured/'results.jsonl')
        assert len(results)==100
        assert rows(measured/'schedule.jsonl')==schedule
        mismatches={field:[] for field in ('token_ids','raw_text','text','stop_reason','crop_size','input_tokens','projected_image_tokens')}
        for row in results:
            assert row['status']=='ok'
            old=reference[row['sequence']]
            assert old['request_id']==row['request_id']
            for field in mismatches:
                if old['service_result']['response'][field]!=row['service_result']['response'][field]:
                    mismatches[field].append(row['sequence'])
        summary=json.loads((measured/'summary.json').read_text())
        ownership=rows(folder/'ownership.jsonl')
        assert not ownership[-1]['pids']
        assert all(set(r['pids'])<=set(r['owned']) for r in ownership)
        item=dict(latency_s=summary['request_latency_s'],completed_qps=100/summary['run_wall_s'],
                  errors=summary['failed_request_count'],mismatches=mismatches,
                  ownership_checks=len(ownership),npu_released=True)
        assert not any(mismatches.values()),(name,mismatches)
        event_file=folder/'b8/service_logs/events.jsonl'
        if name.startswith('control'):
            assert not event_file.exists(),'Control unexpectedly logged events'
        else:
            events=rows(event_file)
            counts=Counter(event['event'] for event in events)
            for event in ('request_accepted','request_finished','response_sent'):
                assert counts[event]==101,(name,event,counts[event])
            assert counts['logging_warning']==0
            assert counts['request_failed']==counts['request_timeout']==counts['request_rejected']==counts['inference_failed']==0
            assert counts['shutdown_finished']==1
            finished=[e for e in events if e['event']=='request_finished']
            total_tokens=sum(e['generated_tokens_including_eos'] for e in finished)
            heartbeats=[e for e in events if e['event']=='heartbeat']
            assert len(heartbeats)>=2
            assert heartbeats[-1]['output_tokens_including_eos']==total_tokens
            for event in finished:
                assert ('timing_s' in event)==(name=='detailed')
                assert not {'text','raw_text','token_ids','image_bytes'} & event.keys()
            # Same operational event is printed and persisted, without reformatting.
            console=[]
            for line in (folder/'b8/server.log').read_text().splitlines():
                try: value=json.loads(line)
                except json.JSONDecodeError: continue
                if isinstance(value,dict) and 'event' in value: console.append(value)
            assert console==events,'File/console event streams differ'
            for kind in ('request_accepted','request_finished','response_sent'):
                identifiers=[e['request_id'] for e in events if e['event']==kind]
                assert len(set(identifiers))==101
            assert {e['request_id'] for e in finished}=={e['request_id'] for e in events if e['event']=='request_accepted'}
            item.update(event_counts=dict(counts),retained_tokens_including_eos=total_tokens,
                        file_console_identical=True,heartbeats=heartbeats)
        report[name]=item
    baseline=report['control_cached']
    for name in ('basic','detailed'):
        report[name]['difference_vs_cached_control_percent']={
            'mean_latency':100*(report[name]['latency_s']['mean']/baseline['latency_s']['mean']-1),
            'p95_latency':100*(report[name]['latency_s']['p95']/baseline['latency_s']['p95']-1),
            'completed_qps':100*(report[name]['completed_qps']/baseline['completed_qps']-1)}
    (args.root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,item in report.items():
        print(name,json.dumps({key:value for key,value in item.items() if key!='heartbeats'}))


if __name__=='__main__':
    main()
