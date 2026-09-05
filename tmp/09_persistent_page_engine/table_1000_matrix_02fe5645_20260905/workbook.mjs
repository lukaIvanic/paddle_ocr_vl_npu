import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const here=path.dirname(new URL(import.meta.url).pathname);
const repo=path.resolve(here,'../../..');
const out=path.join(repo,'outputs/01a0735d-f277-7262-b1d0-b87d6db95456');
const read=async p=>JSON.parse(await fs.readFile(path.join(repo,'tmp/09_persistent_page_engine',p),'utf8'));
const hard=await read('table_vllm_closed50_shuffle1_c124_95fb6d8c_20260903/analysis.json');
const all=await read('table_vllm_random100_seed1_c124_95fb6d8c_20260903/analysis.json');
const high=await read('table_vllm_c8_c16_both_9ed4c364_20260903/analysis.json');
const measured=await read('table_1000_matrix_02fe5645_20260905/analysis.json');
const wb=Workbook.create();
const sheet=wb.worksheets.add('Latency comparison');
const data=wb.worksheets.add('vLLM data');
const notes=wb.worksheets.add('Sources and notes');
const border={preset:'all',style:'thin',color:'#61615B'};
const customHard=[
 ['B=1',2.456,2.201,3.641,3.65,3.693,3.695,.407],
 ['B=2',2.882,2.491,4.368,4.426,4.862,5.101,.685],
 ['B=4',3.467,3.023,5.102,5.353,6.145,6.242,1.085],
 ['B=8',5.184,4.763,8.162,8.345,8.79,8.922,1.433],
 ['B=16',7.511,6.945,11.157,12.811,14.011,14.14,1.873],
];
const customAll=[
 ['B=1',.635,.446,1.214,1.722,3.45,3.681,1.574],
 ['B=2',.793,.527,1.492,2.151,4.915,5.081,2.483],
 ['B=4',1.184,.73,2.553,3.35,7.289,7.785,3.241],
 ['B=8',1.857,1.181,3.815,5.181,12.185,14.403,4.003],
 ['B=16',3.168,2.161,6.132,8.284,15.868,20.789,4.513],
];
const headers=['Concurrency/Latency','Mean(s)','P50(s)','P90(s)','P95(s)','P99(s)','Max(s)','QPS'];
for(const s of [sheet,data,notes])s.showGridLines=false;
sheet.getRange('A1:Z41').format={font:{name:'Arial',size:11,color:'#20231F'},rowHeightPx:31,verticalAlignment:'center'};
sheet.getRange('A1:B41').format.columnWidthPx=22;
sheet.getRange('C1:C41').format.columnWidthPx=205;
sheet.getRange('D1:J41').format.columnWidthPx=105;
sheet.getRange('K1:K41').format.columnWidthPx=40;
sheet.getRange('L1:L41').format.columnWidthPx=85;
sheet.getRange('M1:O41').format.columnWidthPx=72;
sheet.getRange('P1:Q41').format.columnWidthPx=40;
sheet.getRange('R1:R41').format.columnWidthPx=205;
sheet.getRange('S1:Y41').format.columnWidthPx=105;
sheet.getRange('Z1:Z41').format.columnWidthPx=22;
sheet.getRange('A1:Z2').format.rowHeightPx=15;
function merged(s,range,text,format={}){const r=s.getRange(range);r.merge();r.values=[[text]];r.format=format;}
function frame(row,col,title,fill,band,count=5){
 const r=sheet.getRangeByIndexes(row-1,col,1,3);r.merge();r.values=[[title]];
 r.format={fill:band,font:{bold:true,size:18},rowHeightPx:35,borders:{preset:'outside',style:'thin',color:'#61615B'}};
 const table=sheet.getRangeByIndexes(row,col,count+1,8);
 table.format={fill,borders:border,rowHeightPx:33,verticalAlignment:'center'};
 sheet.getRangeByIndexes(row,col,1,8).values=[headers];
 sheet.getRangeByIndexes(row,col,1,8).format.font.bold=true;
 sheet.getRangeByIndexes(row,col+1,count+1,7).format.horizontalAlignment='right';
 sheet.getRangeByIndexes(row+1,col+1,count,7).setNumberFormat('0.###');
 sheet.getRangeByIndexes(row+1,col+4,count,1).format.font.bold=true;
 sheet.getRangeByIndexes(row+1,col+7,count,1).format.font.bold=true;
}
merged(sheet,'C3:J3','Custom PaddleOCR-VL',{font:{bold:true,color:'#697265'}});
merged(sheet,'L3:O3','Ratios above 1× favor custom',{font:{size:10,color:'#697265'}});
merged(sheet,'R3:Y3','vLLM-Ascend 0.23.0rc1 · FULL_AND_PIECEWISE',{font:{bold:true,color:'#697265'}});
data.getRange('A1:K15').format={font:{name:'Arial',size:11},rowHeightPx:30,verticalAlignment:'center'};
data.getRange('A1:A15').format.columnWidthPx=150;
data.getRange('B1:K15').format.columnWidthPx=110;
data.getRange('A1:K1').values=[['Cohort','Concurrency','Requests','Elapsed(s)','Mean(s)','P50(s)','P90(s)','P95(s)','P99(s)','Max(s)','QPS']];
data.getRange('A1:K1').format={fill:'#D9E2D2',font:{bold:true},borders:border};
let raw=2;
for(const [group,row,custom,fill,band] of [
 ['p90',4,customHard,'#FFF2CC','#E8B13E'],['random',15,customAll,'#E2EFDA','#92C96F']]){
 const name=group==='p90'?'>P90 tables':'All tables';
 frame(row,2,'Custom: '+name,fill,band);
 frame(row,17,'vLLM: '+name,fill,band);
 sheet.getRangeByIndexes(row+1,2,5,8).values=custom;
 const start=row+2;
 merged(sheet,`L${row}:O${row}`,'Speedup',{fill:band,font:{bold:true,size:18},borders:{preset:'outside',style:'thin',color:'#61615B'}});
 sheet.getRange(`L${row+1}:O${row+6}`).format={fill,borders:border};
 sheet.getRange(`L${row+1}:O${row+1}`).values=[['B / C','Mean','P95','QPS']];
 sheet.getRange(`L${row+1}:O${row+1}`).format.font.bold=true;
 for(const [i,c] of [1,2,4,8,16].entries()){
  const v=c<=4?(group==='p90'?hard:all)[String(c)].vllm:high[group][String(c)].vllm;
  const d=v.latency_s, r=start+i;
  data.getRange(`A${raw}:K${raw}`).values=[[group==='p90'?'Hardest 50':'Random 100',c,v.requests,v.wall_s,d.mean,d.p50,d.p90,d.p95,d.p99,d.max,null]];
  data.getRange(`A${raw}:K${raw}`).format={fill,borders:border};
  data.getRange(`D${raw}:K${raw}`).setNumberFormat('0.000000');
  data.getRange(`K${raw}`).formulas=[[`=C${raw}/D${raw}`]];
  sheet.getRange(`R${r}`).values=[[`B=${c}`]];
  sheet.getRange(`S${r}:Y${r}`).formulas=[['E','F','G','H','I','J','K'].map(k=>`=ROUND('vLLM data'!${k}${raw},3)` )];
  sheet.getRange(`L${r}`).values=[[c]];
  sheet.getRange(`M${r}:O${r}`).formulas=[[`=S${r}/D${r}`,`=V${r}/G${r}`,`=J${r}/Y${r}`]];
  const speed=sheet.getRange(`M${r}:O${r}`);
  speed.setNumberFormat('0.00"×"');speed.format.font.bold=true;
  speed.format.horizontalAlignment='right';
  speed.conditionalFormats.add('cellIs',{operator:'greaterThan',formula:1,format:{font:{color:'#27643B'}}});
  speed.conditionalFormats.add('cellIs',{operator:'lessThan',formula:1,format:{font:{color:'#A33B30'}}});
  raw++;
 }
 merged(sheet,`L${row+8}:O${row+9}`,'Latency = vLLM ÷ custom\nQPS = custom ÷ vLLM',{font:{size:10,color:'#64705F'},wrapText:true});
}
merged(sheet,'C12:J12','50 hardest tables, shuffled in the same order.',{font:{size:10,color:'#64705F'}});
merged(sheet,'C23:J23','100 tables sampled from the full 665-table corpus.',{font:{size:10,color:'#64705F'}});
merged(sheet,'R12:Y12','C = maximum requests in flight; vLLM batches dynamically.',{font:{size:10,color:'#64705F'}});
merged(sheet,'R23:Y23','Same 100 tables and dispatch order as custom.',{font:{size:10,color:'#64705F'}});
merged(sheet,'C25:Y25','Latency is in seconds. QPS is achieved closed-loop throughput. Ascend 910B2; startup and warmup excluded.',{font:{size:10,color:'#64705F'}});

merged(sheet,'C28:J28','New optimizations · 1,000 requests · 2026-09-05',{font:{bold:true,color:'#697265'}});
frame(29,2,'Custom: all 665 + 335', '#E2EFDA','#92C96F',6);
const lanes=['b1','b2','b4c3','b4c4','b8','b16'];
const labels=['B1 / C1','B2 / C2','B4 / C3','B4 / C4','B8 / C8','B16 / C16'];
for(const [i,name] of lanes.entries()){
 const r=31+i, a=measured[name];
 if(a){const d=a.latency_s;sheet.getRange(`C${r}:J${r}`).values=[[labels[i],d.mean,d.p50,d.p90,d.p95,d.p99,d.max,a.completed_tables_per_s]];}
 else{sheet.getRange(`C${r}`).values=[[labels[i]]];merged(sheet,`D${r}:J${r}`,'Pending',{font:{italic:true,color:'#737970'},horizontalAlignment:'center'});}
}
merged(sheet,'C38:J39','Seed 3: all 665 tables once, then 335 distinct tables from the same corpus. Identical order in every lane. B = static decode batch; C = maximum requests in flight.',{font:{size:10,color:'#64705F'},wrapText:true});
merged(sheet,'L30:Y32','The new block uses a larger, different workload than the historical random-100 block. No matched new vLLM run was requested; cross-block changes are not controlled speedup measurements.',{font:{size:11,color:'#64705F'},wrapText:true});
merged(sheet,'L34:Y36','Ordinary decoding only. Latest CPU preparation and vision padding optimizations; unchanged model, input pixel policy, greedy selection, native vocabulary map, KV4096 and stopping rules.',{font:{size:11,color:'#64705F'},wrapText:true});
const limits=lanes.filter(name=>measured[name]).map(name=>`${name.toUpperCase()}: ${measured[name].stop_reasons.kv_cache_full||0}`).join('; ');
merged(sheet,'L38:Y40',`KV4096 limit stops, retained in every latency denominator: ${limits}. QPS counts returned requests, including these capped outputs; not proof of complete-output goal validation.`,{font:{size:11,color:'#64705F'},wrapText:true});

notes.getRange('A1:B12').format={font:{name:'Arial',size:11},rowHeightPx:60,verticalAlignment:'center',wrapText:true};
notes.getRange('A1:A12').format.columnWidthPx=190;
notes.getRange('B1:B12').format.columnWidthPx=940;
notes.getRange('A1:B1').values=[['Item','Source or explanation']];
notes.getRange('A1:B1').format={fill:'#92C96F',font:{bold:true},rowHeightPx:32};
notes.getRange('A2:B10').values=[
 ['Reconstruction','The earlier temporary workbook is missing. Historical tables were reconstructed from the three supplied screenshots and saved full-precision vLLM records.'],
 ['Historical custom data','Values and displayed rounding reproduce the screenshots. The >P90 B8 P95 is retained as 8.345 s; the saved benchmark value was 8.353625551 s.'],
 ['Historical cohorts','Hardest 50: original-B1 tail ranks, shuffled with seed 1. All tables: a 100-table sample from all 665, seed 1, not all 665 measured.'],
 ['Historical vLLM sources','table_vllm_closed50_shuffle1_c124_95fb6d8c_20260903; table_vllm_random100_seed1_c124_95fb6d8c_20260903; table_vllm_c8_c16_both_9ed4c364_20260903.'],
 ['Historical configuration','vLLM-Ascend 0.23.0rc1, FULL_AND_PIECEWISE. Earlier C1/C2/C4 used max-num-seqs 4; C8/C16 used 16.'],
 ['Historical output limits','Hardest 50 had eight context-limit stops per lane; random 100 completed normally. Those historical measurements are preserved, not presented as a new quality validation.'],
 ['Speedup formulas','Ratios use the photo-displayed, rounded values for both engines. Mean and P95 = vLLM / custom; QPS = custom / vLLM. Values above 1× favor custom. Full-precision vLLM measurements remain in the data tab.'],
 ['New comparison','table_1000_matrix_02fe5645_20260905. Fixed seed 3; 1,000 submissions per lane, 665 unique crops. Full HTTP submission-to-response timing; payload preparation, model startup and full-request warmup outside timing.'],
 ['Timing scope','Queueing, image decoding, CPU preparation, vision/text prefill, decode, control and postprocessing remain in latency. Overlapping work is not subtracted. QPS includes initial filling and final draining.'],
];
notes.getRange('A2:A10').format.font.bold=true;
notes.getRange('A2:B10').format.borders={preset:'inside',style:'thin',color:'#D9DFD4'};
if(JSON.stringify(sheet.getRange('C6:J10').values)!==JSON.stringify(customHard))throw Error('Historical hard table changed');
if(JSON.stringify(sheet.getRange('C17:J21').values)!==JSON.stringify(customAll))throw Error('Historical all-table values changed');
console.log((await wb.inspect({kind:'table',range:"'Latency comparison'!C29:J36",include:'values,formulas',tableMaxRows:8,tableMaxCols:8,maxChars:3500})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#SPILL!',options:{useRegex:true,maxResults:30},maxChars:1500})).ndjson);
await fs.mkdir(out,{recursive:true});
for(const [sheetName,range,file] of [['Latency comparison','B3:Z25','historical'],['Latency comparison','B28:Z40','new'],['vLLM data','A1:K11','data'],['Sources and notes','A1:B10','notes']]){
 if(['historical','data','notes'].includes(file)){
  try{await fs.access(path.join(here,file+'.png'));continue;}catch{}
 }
 const png=await wb.render({sheetName,range,scale:1.2,format:'png'});
 await fs.writeFile(path.join(here,file+'.png'),new Uint8Array(await png.arrayBuffer()));
}
await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'Table OCR latency comparison.xlsx'));
