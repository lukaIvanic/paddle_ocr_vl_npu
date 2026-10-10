"""Small unprofiled call matrix; enable ASCEND_GLOBAL_LOG_LEVEL=0 for tiler logs."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fused_norm_v2 import initialize
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--op-api',type=Path,required=True)
a=p.parse_args()
initialize(a.op_api)
from test_add_layer_norm_quant_v2 import V2
op=V2(a.op_api)
torch.npu.set_device(0)
with torch.inference_mode():
    for rows in (256,512,2048):
        x=torch.zeros(rows,1024,dtype=torch.float16,device='npu')
        g=torch.ones(1,1024,dtype=torch.float16,device='npu')
        b=torch.zeros_like(g);s=torch.ones(1,dtype=torch.float16,device='npu')
        for kind,bias in [('none',None),('broadcast',b),('elementwise',x)]:
            y,q=op(x,x,g,b,s,bias)
            assert not torch.count_nonzero(y).item() and not torch.count_nonzero(q).item()
            print(json.dumps({'pid':os.getpid(),'rows':rows,'bias':kind,'op_api':str(a.op_api)}),flush=True)
print(Path('/proc/self/maps').read_text())
