"""Read-only remote native-token aggregation; no tokenizer or inference involved."""
import collections
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REMOTE=r'''
import collections, hashlib, json
from pathlib import Path
root=Path('/data1/lukaiv/workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine')
names=['910b_full_e2e_eval_8634d3a_r1/output/recognition_trace.jsonl',
       '910b_full_sacrificial_drain_d9197ea/output/recognition_trace.jsonl',
       'table_spec_full_d1e6d00/whole/row_ocr_records.jsonl']
results=[]
for name in names:
 p=root/name
 counts=collections.defaultdict(collections.Counter)
 records=collections.Counter(); stops=collections.Counter(); pages=set(); requests=set()
 h=hashlib.sha256(); missing=0
 with p.open('rb') as f:
  for line in f:
   h.update(line)
   if not line.strip(): continue
   r=json.loads(line)
   if 'page_input_index' in r: pages.add(r['page_input_index'])
   requests.add(r.get('request_id',len(requests)))
   for row in (r.get('rows') if 'rows' in r else [r]):
    ids=row.get('token_ids')
    if ids is None: missing+=1; continue
    label=row.get('label',r.get('label','table'))
    counts[label].update(ids); records[label]+=1
    stops[row.get('stop_reason','unspecified')]+=1
 summary=p.parent/'run_summary.json'
 summary_data=json.loads(summary.read_text()) if summary.exists() else {}
 results.append(dict(source=str(p),source_sha256=h.hexdigest(),page_count=len(pages),
    page_min=min(pages) if pages else None,page_max=max(pages) if pages else None,
    request_count=len(requests),records=dict(records),missing_ids=missing,stops=dict(stops),
    counts={k:dict(v) for k,v in counts.items()},
    summary_top_level_keys=list(summary_data),
    summary_scalar_fields={k:v for k,v in summary_data.items() if isinstance(v,(str,int,float,bool))}))
print(json.dumps(results,separators=(',',':')))
'''

cmd=['ssh','-S','/tmp/paddle-blue-zone-master.sock','-o','IdentitiesOnly=yes',
     '-i','/Users/lukaivanic/Downloads/key.pem','-o','ConnectTimeout=8',
     'root@116.204.40.238','python3 -']
result=subprocess.run(cmd,input=REMOTE,text=True,capture_output=True,timeout=180,check=True)
raw=json.loads(result.stdout)
selection=json.loads((ROOT/'09_persistent_page_engine/presets/table_compact_vocab/b1_verifier_topfreq_16384.json').read_text())
kept=set(selection['token_ids'])
for source in raw:
 total=collections.Counter()
 source['by_label']={}
 for label,freq in source['counts'].items():
  c=collections.Counter({int(k):v for k,v in freq.items()}); total.update(c)
  source['by_label'][label]=dict(records=source['records'][label],unique_ids=len(c),
     token_occurrences=sum(c.values()),outside_existing_16k=len(set(c)-kept))
 source['unique_ids']=len(total)
 source['token_occurrences']=sum(total.values())
 source['outside_existing_16k']=len(set(total)-kept)
 source['all_token_ids']=sorted(total)
(HERE/'native_generation_counts.json').write_text(json.dumps(raw,indent=2)+'\n')
print(json.dumps([{k:v for k,v in s.items() if k not in ('counts','all_token_ids')} for s in raw],indent=2))
