"""CPU synthetic tile preview of actor augmentation, plus isolated CPU timing."""
import sys,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'tests'))
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_mapper_aug149 import fixture
from gd_lab.gast.mapper_aug import MapperAugCfg,MapperAugmentation
torch.set_num_threads(1);out=root/'archive/update149_validation';out.mkdir(exist_ok=True)
for kind in ('gap','up','down'):
 h,v,ir,args,cast=fixture(4,kind);a=MapperAugmentation(4,'cpu',MapperAugCfg(enabled=True));a.params[:,0]=.9;a.params[:,1]=0
 a.params[:,6:8]=.5;a.params[:,12]=0
 q=a(h,v,ir,*args,50000,cast)
 fig,ax=plt.subplots(4,4,figsize=(12,9))
 for i in range(4):
  for j,(name,value) in enumerate([('off height',h),('on height',q[0]),('on valid',q[1]),('on IR',q[2])]):
   im=ax[i,j].imshow(value[i].reshape(11,17).float(),origin='lower',extent=[-.8,.8,-.5,.5],vmin=0 if j>1 else -1,vmax=1 if j>1 else 4)
   ax[i,j].set_title(f'env{i}: {name}');fig.colorbar(im,ax=ax[i,j],shrink=.65)
 fig.suptitle(f'UPDATE149 synthetic {kind}: same input, augmentation off/on; critic source unchanged')
 fig.tight_layout();fig.savefig(out/f'{kind}.png',dpi=120);plt.close(fig)
 torch.save(dict(clean_height=h,aug_height=q[0],valid=q[1],ir=q[2],points=args[0]),out/f'{kind}_actor_inputs.pt')
h,v,ir,args,cast=fixture(1024);a=MapperAugmentation(1024,'cpu',MapperAugCfg(enabled=True))
for _ in range(3):a(h,v,ir,*args,50000,cast)
t=time.perf_counter()
for _ in range(10):a(h,v,ir,*args,50000,cast)
ms=1000*(time.perf_counter()-t)/10
(out/'cpu_timing.json').write_text(json.dumps(dict(envs=1024,threads=1,capture_ms=ms,estimated_25_captures_ms=ms*25,cast='analytic CPU fixture, NOT Warp/GPU or end-to-end iteration'),indent=2));print('CPU analytic capture ms',ms)
