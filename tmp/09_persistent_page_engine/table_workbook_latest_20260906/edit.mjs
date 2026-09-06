import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';
const here=path.dirname(new URL(import.meta.url).pathname);
const repo=path.resolve(here,'../../..');
const file=path.join(repo,'outputs/01a0735d-f277-7262-b1d0-b87d6db95456/Table OCR latency comparison.xlsx');
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(file));
const s=wb.worksheets.getItem('Latency comparison');
const scope=[['Latency comparison','A1:Z27'],['vLLM data','A1:K15'],['Sources and notes','A1:B8'],['Sources and notes','A10:B12']];
const snapshot=w=>scope.map(([name,range])=>{const r=w.worksheets.getItem(name).getRange(range);return JSON.stringify({values:r.values,formulas:r.formulas});});
const original=snapshot(wb);
const root=path.join(repo,'tmp/09_persistent_page_engine');
const sweep='table_optimized_c1_to_c7_9f6e486d_20260906';
const audit=JSON.parse(await fs.readFile(path.join(root,sweep,'analysis.json'),'utf8'));
if(Object.keys(audit).length!==8 || Object.values(audit).some(x=>!x.clean_timing))throw Error('Validation missing or contaminated');
const groups=[
 ['B1 / C1',`${sweep}/b1c1/measured`],
 ['B2 / C2',`${sweep}/b2c2/measured`,'table_packed_noevents_23d5518c_20260906/validation1000_a','table_packed_noevents_23d5518c_20260906/validation1000_b'],
 ['B3 / C3',`${sweep}/b3c3/measured`],
 ['B4 / C4',`${sweep}/b4c4/measured`],
 ['B5 / C5','table_packed_noevents_b5c5_b0f1cac7_20260906/validation1000_a','table_b5_clean_repeat_d958f186_20260906/validation1000_b'],
 ['B6 / C6',`${sweep}/b6c6/measured`],
 ['B7 / C7',`${sweep}/b7c7/measured`],
 ['B8 / C7',`${sweep}/b8c7/measured`],
 ['B8 / C8',`${sweep}/b8c8/measured`],
];
const selected=[];
for(const [label,...paths] of groups){
 const runs=await Promise.all(paths.map(async id=>({id,data:JSON.parse(await fs.readFile(path.join(root,id,'summary.json'),'utf8'))})));
 for(const {data} of runs)if(data.request_count!==1000 || data.failed_request_count!==0)throw Error('Incomplete run');
 runs.sort((a,b)=>b.data.latency_s.p95-a.data.latency_s.p95 || a.data.completion_qps-b.data.completion_qps);
 selected.push({label,...runs[0]});
}
const rows=selected.map(({label,data:d})=>[label,...['mean','p50','p90','p95','p99','max'].map(k=>d.latency_s[k]),d.completion_qps]);
s.getRange('C28').values=[['Latest optimizations · 1,000 requests · 2026-09-06']];
s.getRange('C34:J36').unmerge();
s.getRange('C38:J39').unmerge();
s.getRange('C31:J39').clear({applyTo:'all'});
s.getRange('C31:J39').format={font:{name:'Arial',size:11,color:'#20231F'},fill:'#E2EFDA',borders:{preset:'all',style:'thin',color:'#61615B'},rowHeightPx:33,verticalAlignment:'center'};
s.getRange('C31:J39').values=rows;
s.getRange('D31:J39').setNumberFormat('0.###');
s.getRange('D31:J39').format.horizontalAlignment='right';
s.getRange('G31:G39').format.font.bold=true;
s.getRange('J31:J39').format.font.bold=true;
for(const [range,text] of [
 ['C41:J42','Seed 3: all 665 tables once, then 335 distinct tables from the same corpus. Identical order in every lane. B = static decode batch; C = maximum requests in flight.'],
 ['C44:J45','One new 1,000-request run per tested configuration. B2 shows the higher-P95 run of all three validations (the new reconfirmation). B5 retains the higher-P95 run of its two earlier validations. All metrics in each row come from one run.']]){
 const r=s.getRange(range);r.merge();r.values=[[text]];r.format={font:{name:'Arial',size:10,color:'#64705F'},wrapText:true,verticalAlignment:'center',rowHeightPx:31};
}
s.getRange('L34').values=[['Ordinary decoding. Packed MLP, complete-layer prefetch, RoPE lookup, linear vision patch projection, vision padding, setup GC freeze and no optional decode timing events. Unchanged input policy, native vocabulary map and KV4096.']];
s.getRange('L38').values=[['Each selected run: 1,000 responses, 988 EOS completions, 12 unchanged KV4096-cap outputs, no request errors. All 1,000 are included in latency and QPS. One physical 910B2 (NPU6), with exclusive ownership verified during measurement.']];
const notes=wb.worksheets.getItem('Sources and notes');
notes.getRange('B9').values=[[`Current sweep: ${sweep}, each lane's measured summary. B2 uses the new reconfirmation (highest P95 of three). B5: table_packed_noevents_b5c5_b0f1cac7_20260906/validation1000_a (higher P95 of two). Same seed-3 sequence. B8/C7 and B8/C8 share a warmed server, with a separate complete-request warmup before each run.`]];
notes.getRange('A9:B9').format.rowHeightPx=95;
if(JSON.stringify(original)!==JSON.stringify(snapshot(wb)))throw Error('Unrelated content changed');
console.log(JSON.stringify({selected:selected.map(x=>({label:x.label,source:x.id})),rows}));
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#SPILL!',options:{useRegex:true,maxResults:30},maxChars:1000})).ndjson);
for(const [sheetName,range,name] of [['Latency comparison','B28:Z45','after'],['Sources and notes','A9:B10','sources']]){
 const png=await wb.render({sheetName,range,scale:1.2,format:'png'});
 await fs.writeFile(path.join(here,name+'.png'),new Uint8Array(await png.arrayBuffer()));
}
await fs.copyFile(file,path.join(here,'before-sweep.xlsx'));
await(await SpreadsheetFile.exportXlsx(wb)).save(file);
const saved=await SpreadsheetFile.importXlsx(await FileBlob.load(file));
if(JSON.stringify(original)!==JSON.stringify(snapshot(saved)))throw Error('Export altered unrelated values or formulas');
if(JSON.stringify(saved.worksheets.getItem('Latency comparison').getRange('C31:J39').values)!==JSON.stringify(rows))throw Error('Export data mismatch');
console.log('Saved; nine rows verified; historical values and formulas unchanged.');
