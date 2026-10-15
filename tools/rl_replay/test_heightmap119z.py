"""Pre-rollout analytic checks against measured floor heights and original h0."""
import json
from pathlib import Path
from test_heightmap116 import scene,K,Q,P,BODY
from heightmap116 import ElevationMap,rays
from heightmap119z import BatchMap
import numpy as np
results={}
def run(name,edges,levels,missing=False,dark=False,mode='h2',times=(0.,.1,.2,.3)):
 d,v,ray=scene(edges,levels);ir=np.full_like(d,.7)
 if dark:ir[:,:,25:55]=.02
 if missing:d[:]=5;v[:]=False
 b=BatchMap(1,mode);old=ElevationMap('h0')
 for t in times:
  b.integrate([0],np.tile(d,(1,4,1,1)),np.tile(v,(1,4,1,1)),np.tile(ir,(1,4,1,1)),np.tile(P,(1,4,1)),np.tile(Q,(1,4,1)),np.tile(K,(1,4,1,1)),BODY[None],Q[None],t)
  old.integrate(d,v,ir,P[None],Q[None],K[None],BODY,Q,t)
 packed=b.sample(np.zeros((1,3)),np.array([.8]),times[-1])[0];valid=packed[187:]>0
 if mode=='h0':np.testing.assert_allclose(packed,old.sample([0,0,0],.8,times[-1])[0],atol=1e-6)
 results[name]={'valid_cells':int(valid.sum()),'saturated_cells':int(((packed[:187]==5)&valid).sum())}
 return b,packed,valid
b,h,v=run('deep_floor_bright_ir',[.25,.75],[0,-1.2,0]);assert ((h[:187]==5)&v).sum()>0;assert not b.maps[0].gap.any()
b,h,v=run('lower_stairs_dark_ir',[.2,.5,.75],[0,-.15,-.30,-.45],dark=True)
x=np.tile(np.linspace(-.8,.8,17),11);expected=5*np.clip(.3-np.array([0,-.15,-.30,-.45])[np.searchsorted([.2,.5,.75],x,side='right')],-1,1)
assert v.sum()>15;assert not b.maps[0].gap.any();assert np.mean(np.abs(h[:187][v]-expected[v]))/5<.055
assert ((h[:187]>1.5)&(h[:187]<5)&v).any()
results['lower_stairs_dark_ir']['height_mae_m']=float(np.mean(np.abs(h[:187][v]-expected[v]))/5)
b,h,v=run('h2_first_return_unconfirmed',[],[-1.2],times=(0.,));assert not v.any()
b,h,v=run('h2_missing_bright',[],[0],missing=True);assert not v.any()
b,h,v=run('h2_missing_dark',[],[0],missing=True,dark=True);assert ((h[:187]==5)&v).any()
for name,edges,levels,missing in [('h0_deep',[.25,.75],[0,-1.2,0],False),('h0_stairs',[.2,.5,.75],[0,-.15,-.30,-.45],False),('h0_missing',[],[0],True)]:
 b,h,v=run(name,edges,levels,missing=missing,dark=True,mode='h0',times=(0.,));assert not b.maps[0].gap.any()
 if missing:assert not v.any()
 else:assert v.any()
result={'passed':True,'fixtures':results}
path=Path(__file__).resolve().parents[2]/'outputs/update119z/cpu_fixtures.json';path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
