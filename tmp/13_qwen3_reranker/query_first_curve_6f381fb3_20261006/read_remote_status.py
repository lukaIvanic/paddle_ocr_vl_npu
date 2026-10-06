import subprocess,json
code='''import json,pathlib,statistics
p=pathlib.Path('/tmp/qwen-document-first-training-smoke-small/tmp/13_qwen3_reranker/query_first_curve_6f381fb3_20261006')
x=json.loads((p/'result.json').read_text());u=x['updates']
print(json.dumps({'status':x['status'],'updates':len(u),'mean_update_s':statistics.mean(r['seconds'] for r in u),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in u),'total_seconds':x.get('total_seconds'),'exit_code':(p/'exit_code.txt').read_text().strip() if (p/'exit_code.txt').exists() else None,'evaluations':{s:{'touche':100*e['touche']['ndcg10'],'ordering':100*e['validation']['overall']['ordering_accuracy'],'seconds':e['seconds']['round']} for s,e in x['evaluations'].items()},'tail':(p/'run.log').read_text().splitlines()[-1]}))
'''
a=['ssh','-F','/home/luka/Documents/Codex/2026-10-01/can-you-connect-to-my-mac/work/ssh-blue-zone/config','-o','ControlPath=none','-o','ControlMaster=no','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','blue_zone_npu_server','docker exec -i research_vllm_ascend_023_external_workspace /usr/local/python3.12.13/bin/python3 -']
r=subprocess.run(a,input=code,text=True,capture_output=True,check=True)
x=json.loads(r.stdout)
open('/tmp/qwen-curve-latest-status.json','w').write(json.dumps(x,indent=2))
print(json.dumps(x,indent=2))
