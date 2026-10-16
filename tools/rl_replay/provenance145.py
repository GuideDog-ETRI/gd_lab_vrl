"""Opt-in, sensor-only observation hooks. Never changes mapper decisions.
JSONL capped incl. a terminal truncation marker; raw stream is lossless NPZ frames.
No ground-truth selection, no course names/coordinates. Readers reject capped/incomplete raw.
"""
import io,json,struct,threading,hashlib,warnings
from pathlib import Path
import numpy as np

def plain(x):
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v) for v in x]
 return x
class JsonSink:
 def __init__(self,path,max_bytes=128*1024**2):
  if max_bytes<1024:raise ValueError('max_bytes >=1024 required')
  self.f=open(path,'wb',buffering=256*1024);self.limit=max_bytes;self.n=0;self.count=0;self.capped=False;self.lock=threading.Lock()
 def emit(self,event,**data):
  with self.lock:
   if self.capped:return
   b=(json.dumps(plain(dict(event=event,**data)),separators=(',',':'),allow_nan=False)+'\n').encode()
   if self.n+len(b)>self.limit-256:
    b=b'{"event":"truncated","reason":"byte_limit"}\n';self.capped=True
   self.f.write(b);self.n+=len(b);self.count+=1
   if self.count%64==0 or self.capped:self.f.flush()
 def close(self):
  with self.lock:self.f.flush();self.f.close()
class RawWriter:
 def __init__(self,path,metadata,max_bytes=1024**3):
  if max_bytes<4096:raise ValueError('raw limit >=4096 required')
  self.f=open(path,'wb',buffering=256*1024);self.f.write(b'R145NPZ1');self.n=8;self.limit=max_bytes;self.capped=False;self.closed=False;self.lock=threading.Lock();self.count=0
  self.write('header',metadata)
 def _bytes(self,event,meta,arrays):
  buf=io.BytesIO();np.savez_compressed(buf,__meta__=json.dumps(dict(event=event,**meta),allow_nan=False),**arrays);b=buf.getvalue();return struct.pack('<Q',len(b))+b
 def write(self,event,meta=None,arrays=None):
  with self.lock:
   if self.capped or self.closed:return
   b=self._bytes(event,meta or {},arrays or {})
   if self.n+len(b)>self.limit-1024:
    b=self._bytes('truncated',{'reason':'byte_limit'},{});self.capped=True
    warnings.warn('raw recorder byte limit reached; exact replay will reject this incomplete stream')
   self.f.write(b);self.n+=len(b);self.count+=1
   # Flush each operation: a crash loses at most an incomplete final frame, detected by reader.
   self.f.flush()
 def close(self):
  if self.closed:return
  self.write('end')
  with self.lock:self.f.flush();self.f.close();self.closed=True

def read_raw(path):
 with open(path,'rb') as f:
  if f.read(8)!=b'R145NPZ1':raise ValueError('not an UPDATE145 raw stream')
  ended=False
  while True:
   b=f.read(8)
   if not b:break
   if len(b)!=8:raise ValueError('truncated frame length')
   n=struct.unpack('<Q',b)[0]
   if n>256*1024**2:raise ValueError('oversized raw record')
   b=f.read(n)
   if len(b)!=n:raise ValueError('truncated frame')
   with np.load(io.BytesIO(b),allow_pickle=False) as z:
    meta=json.loads(str(z['__meta__']));arr={k:z[k].copy() for k in z.files if k!='__meta__'}
   if meta['event']=='truncated':raise ValueError('raw byte cap: no complete replay claim')
   if ended:raise ValueError('record after end')
   if meta['event']=='end':ended=True
   yield meta,arr
  if not ended:raise ValueError('missing clean end; partial capture cannot verify complete payload')

def emit(obj,event,**kw):
 sink=getattr(obj,'provenance',None)
 if sink is not None:sink.emit(event,**getattr(obj,'provenance_context',{}),**kw)

def install_structure(cls,S):
 update,ret,expire,overlay,reset=cls.update,cls._retract,cls.expire,cls.overlay,cls.reset
 def wrap_update(self,res,stamp):
  if getattr(self,'provenance',None) is None:return update(self,res,stamp)
  for grid,(G,off) in enumerate(self.grids):
   keys=set(map(tuple,self.cells(res['gap_xy'],off).tolist()))
   for k in sorted(keys):emit(self,'gap_mark_attempt',grid=grid,key=k,offset=off,previous=G.get(k),stamp=stamp)
  emit(self,'structure_evidence',stamp=stamp,gap_xy=res['gap_xy'],gap_z=res['gap_z'],gap_depth=res['gap_depth'],support_xyz=res['support_xyz'],pending_xy=res['pending_xy'],evid_dbg=res.get('evid_dbg'),n_points=res.get('n_points'),n_evidence=res.get('n_evidence'))
  old=[dict(G) for G,_ in self.grids];oldpending=dict(self.pending)
  result=update(self,res,stamp)
  if not len(res['support_xyz']):
   for grid,(G,off) in enumerate(self.grids):
    for k,g in G.items():emit(self,'retraction_decision',grid=grid,key=k,offset=off,stored=g,removed=False,reasons=['no_support_capture_no_retraction_call'])
  for grid,(G,off) in enumerate(self.grids):
   for k,v in G.items():
    if k not in old[grid] or v!=old[grid][k]:emit(self,'mark' if k not in old[grid] else 'refresh',grid=grid,key=k,offset=off,state=v,previous=old[grid].get(k),stamp=stamp)
  for k,t in self.pending.items():
   if oldpending.get(k)!=t:emit(self,'pending_mark',key=k,stamp=t)
  return result
 def wrap_retract(self,G,off,now,sx,pend):
  if getattr(self,'provenance',None) is None:return ret(self,G,off,now,sx,pend)
  # Observational copy of the exact predicates; original method still decides and mutates.
  before=dict(G);pending=dict(self.pending);f=(sx[:,:2]-off)/self.res;cc=np.floor(f+1e-7).astype(np.int64);fr=f-cc;m=S['retract_margin']/self.res
  inner=((fr>m)&(fr<1-m)).all(1);quad=(fr[:,0]>=.5).astype(int)*2+(fr[:,1]>=.5).astype(int)
  ret(self,G,off,now,sx,pend)
  for k in sorted(set(before)|(set(pending) if pend else set())):
   take=(cc==np.array(k)).all(1);ins=take&inner;g=before.get(k);band=np.zeros(len(sx),bool);low=band.copy()
   if g is not None:
    band=ins&(abs(sx[:,2]-g[0])<=S['retract_band']);low=ins&(sx[:,2]>=g[0]-S['step_max'])&(sx[:,2]<g[0]-S['retract_band'])
   elif pend and k in pending:band=ins
   nb=int(band.sum());nl=int(low.sum());qb=len(set(quad[band].tolist()));ql=len(set(quad[low].tolist()))
   sufficient=(nb>=S['retract_points'] and qb>=2) or (nl>=2*S['retract_points'] and ql>=3)
   reasons=[]
   if not take.any():reasons.append('no_support_in_cell')
   elif not ins.any():reasons.append('interior_margin')
   elif not (band|low).any():reasons.append('height_band')
   if not sufficient:reasons.append('support_count_or_quadrants')
   if k in now:reasons.append('same_capture_remark_veto')
   if g is not None and k not in G:reasons.append('retracted');emit(self,'retract',key=k,offset=off,previous=g)
   emit(self,'retraction_decision',key=k,offset=off,stored=g,marked_now=k in now,points_in_cell=int(take.sum()),interior_points=int(ins.sum()),band_points=nb,band_quadrants=qb,low_points=nl,low_quadrants=ql,sufficient=sufficient,removed=g is not None and k not in G,reasons=reasons,support_points=sx[take],fractions=fr[take],interior=inner[take])
 def wrap_expire(self,pose,stamp):
  if getattr(self,'provenance',None) is None:return expire(self,pose,stamp)
  old=[dict(G) for G,_ in self.grids];p=dict(self.pending);sup=dict(self.sup);expire(self,pose,stamp)
  c,s=np.cos(pose[2]),np.sin(pose[2])
  for j,(G,off) in enumerate(self.grids):
   for k,g in old[j].items():
    if k not in G:
     behind=c*((k[0]+.5)*self.res+off-pose[0])+s*((k[1]+.5)*self.res+off-pose[1])<S['hind_pass_x']
     emit(self,'expire',grid=j,key=k,offset=off,previous=g,stamp=stamp,reason='hind_pass' if behind else 'age')
  for k in p.keys()-self.pending.keys():emit(self,'pending_expire',key=k,stamp=stamp)
  for k in sup.keys()-self.sup.keys():emit(self,'support_expire',key=k,stamp=stamp)
 def wrap_overlay(self,packed,pose,z,stamp,source):
  if getattr(self,'provenance',None) is None:return overlay(self,packed,pose,z,stamp,source)
  if stamp-getattr(self,'_prov_last_sample',-float('inf')) < .1:
   return overlay(self,packed,pose,z,stamp,source)
  self._prov_last_sample=stamp
  before=packed.copy();overlay(self,packed,pose,z,stamp,source)
  y,x=np.meshgrid(np.linspace(-.5,.5,11),np.linspace(-.8,.8,17),indexing='ij');c,s=np.cos(pose[2]),np.sin(pose[2]);xy=np.c_[pose[0]+c*x.ravel()-s*y.ravel(),pose[1]+s*x.ravel()+c*y.ravel()]
  k0=self.cells(xy);k1=self.cells(xy,S['dual_offset'])
  for i,p in enumerate(xy):
   g0=self.gap.get(tuple(k0[i]));g1=self.gap2.get(tuple(k1[i]));winner=0 if g0 else (1 if g1 else None)
   supkey=tuple(np.floor(p/S['sup_res']+1e-7).astype(int));support=self.sup.get(supkey)
   direct=getattr(self,'provenance_direct',None);direct_age=None
   if direct is not None and direct.origin is not None:
    ij,inside=direct.indices(p[None]);ix,iy=ij[0]
    if inside[0]:direct_age=stamp-direct.last[iy,ix]
   emit(self,'policy_cell',index=i,world_xy=p,pose=pose,scanner_z=z,stamp=stamp,pre_h=float(before[i]),pre_valid=float(before[187+i]),out_h=float(packed[i]),out_valid=float(packed[187+i]),source=int(source[i]),direct_cell_age=direct_age,support_cell_key=supkey,support_state=support,support_age=stamp-support[1] if support else None,source_kind='inferred_structural_gap' if source[i]==4 else ('pending' if source[i]==5 else 'direct_or_unknown'),grid0_key=k0[i],grid1_key=k1[i],grid0=g0,grid1=g1,grid0_age=stamp-g0[3] if g0 else None,grid1_age=stamp-g1[3] if g1 else None,winning_grid=winner if source[i]==4 and S['dual_grid'] else None,pending=tuple(k0[i]) in self.pending,dual_or=S.get('dual_or',False))
 def wrap_reset(self):
  emit(self,'structure_reset');self._prov_last_sample=-float('inf');return reset(self)
 cls.update=wrap_update;cls._retract=wrap_retract;cls.expire=wrap_expire;cls.overlay=wrap_overlay;cls.reset=wrap_reset

def install_maps(cls):
 init,integ,sample,reset=cls.__init__,cls.integrate,cls.sample,cls.reset
 def new_init(self,n,arm,provenance=None):
  init(self,n,arm);self.provenance=provenance;self.provenance_context={}
  for e,m in enumerate(self.struct):m.provenance=provenance;m.provenance_context={'env':e};m.provenance_direct=self.maps[e]
 def new_integrate(self,ids,packet,stamp,now):
  if self.provenance is not None:
   for e in ids:
    self.struct[e].provenance_context=dict(self.provenance_context,env=int(e),capture_stamp=stamp,integration_time=now)
   emit(self,'integration',ids=list(ids),capture_stamp=stamp,integration_time=now)
  return integ(self,ids,packet,stamp,now)
 def new_sample(self,poses,scanner,stamp):
  if self.provenance is not None:
   for e,m in enumerate(self.struct):m.provenance_context=dict(self.provenance_context,env=e)
  return sample(self,poses,scanner,stamp)
 def new_reset(self,ids):
  emit(self,'maps_reset',ids=list(ids));return reset(self,ids)
 cls.__init__=new_init;cls.integrate=new_integrate;cls.sample=new_sample;cls.reset=new_reset
