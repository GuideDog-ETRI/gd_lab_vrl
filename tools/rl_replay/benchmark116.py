"""CPU-only 4-camera fixture benchmark; not an Isaac throughput measurement."""
import json,time,platform
from pathlib import Path
import numpy as np
from test_heightmap116 import scene,P,Q,K,BODY
from heightmap116 import ElevationMap
out={}
for mode in ('h0','h1'):
    d,v,_=scene([.35,.65],[0,-1.2,0]);d=np.repeat(d,4,0);v=np.repeat(v,4,0);ir=np.full_like(d,.7);ir[:,:,30:55]=.03
    m=ElevationMap(mode);costs=[];samples=[]
    for j in range(40):
        tick=time.perf_counter();m.integrate(d,v,ir,np.repeat(P[None],4,0),np.repeat(Q[None],4,0),np.repeat(K[None],4,0),BODY,Q,j*.1);costs.append((time.perf_counter()-tick)*1000)
        tick=time.perf_counter();m.sample((0,0,0),.8,j*.1);samples.append((time.perf_counter()-tick)*1000)
    out[mode]={label:dict(n=len(vals),mean_ms=float(np.mean(vals)),p95_ms=float(np.quantile(vals,.95))) for label,vals in [('integrate_4view_one_env',costs[5:]),('sample_187_one_env',samples[5:])]}
out['platform']=platform.platform();out['scope']='single process CPU, analytic fixture, 4 repeated camera views, excludes Isaac/transport/Torch encoder'
p=Path(__file__).resolve().parents[2]/'outputs/cpu_benchmark.json';p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
