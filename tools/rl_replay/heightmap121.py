"""Sensor-only h2m/h3/h3u. No geometry, reflectance or injection-label inputs."""
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage
from heightmap116 import R, rays, rotation, support_normals, vertical_returns
from heightmap119z import BatchMap as LegacyBatchMap

UNKNOWN, OBSERVED_HEIGHT, INFERRED_GAP = 0, 1, 2

def candidates(ir):
    good=np.isfinite(ir)&(ir>=0)&(ir<.995)
    if not good.any():return np.zeros_like(good)
    dark=good&(ir<=np.quantile(ir[good],.2))&(ir<np.max(ir[good]));paired=np.zeros_like(good)
    for transpose in (False,True):
        a=dark.T if transpose else dark;v=ir.T if transpose else ir;g=good.T if transpose else good;p=paired.T if transpose else paired
        for row in range(a.shape[0]):
            changes=np.diff(np.r_[False,a[row],False].astype(int))
            for start,end in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
                if start<3 or end>a.shape[1]-3 or end-start<2:continue
                left=v[row,start-3:start];right=v[row,end:end+3]
                if not g[row,start-3:end+3].all():continue
                surround=np.r_[left,right];mad=max(.015,float(np.median(np.abs(surround-np.median(surround)))))
                if min(np.median(left),np.median(right))-np.median(v[row,start:end])>=2.5*mad:p[row,start:end]=True
    labels,_=ndimage.label(dark);out=np.zeros_like(good)
    for label,sl in enumerate(ndimage.find_objects(labels),1):
        patch=labels[sl]==label;votes=paired[sl]&patch
        if sl[0].start==0 or sl[1].start==0 or sl[0].stop==ir.shape[0] or sl[1].stop==ir.shape[1]:continue
        # Require transverse interior; length alone must never qualify a line.
        thick=ndimage.binary_opening(patch,structure=np.ones((2,2),bool))
        if not thick.any():continue
        coords=np.argwhere(votes&thick)
        if len(coords)<3:continue
        centered=coords-coords.mean(axis=0)
        if np.linalg.matrix_rank(centered.T@centered/(len(coords)-1))<2:continue
        _,_,axes=np.linalg.svd(centered,full_matrices=False)
        length=np.ptp(centered@axes[0])+1
        if length>=6 and votes.sum()>=.5*patch.sum():out[sl]|=thick
    return out

def fit_plane(points):
    if len(points)<6:return None
    a=np.c_[points[:,:2],np.ones(len(points))]
    if np.linalg.matrix_rank(a)<3:return None
    coef=np.linalg.lstsq(a,points[:,2],rcond=None)[0];error=points[:,2]-a@coef
    rms=float(np.sqrt(np.mean(error**2)))
    if rms>.02 or np.linalg.norm(coef[:2])>.75:return None
    return coef,rms,np.linalg.pinv(a.T@a)*max(rms**2,1e-6)

def plane_intersection(position,ray,coef):
    denom=ray[...,2]-ray[...,:2]@coef[:2]
    return np.divide(coef[2]+position[:2]@coef[:2]-position[2],denom,out=np.full(denom.shape,np.nan),where=np.abs(denom)>.1)

def tread_certificate(points,valid,support):
    """Contiguous finite planar tread plus observed adjacent vertical riser.
    No dilation across missing pixels and no 25cm height connectivity rule.
    """
    riser=vertical_returns(points,valid);out=np.zeros_like(valid)
    labels,_=ndimage.label(support)
    for lab,sl in enumerate(ndimage.find_objects(labels),1):
        mask=labels==lab;p=points[mask]
        if len(p)<12 or fit_plane(p) is None:continue
        if np.linalg.norm(np.ptp(p[:,:2],axis=0))<.10:continue
        adjacent=mask.copy()
        for _ in range(5):adjacent=ndimage.binary_dilation(adjacent)&valid
        contact=adjacent&riser
        if contact.sum()<3:continue
        coef=fit_plane(p)[0];rp=points[contact]
        residual=rp[:,2]-(rp[:,:2]@coef[:2]+coef[2])
        edge=points[adjacent&~mask]
        edge_residual=edge[:,2]-(edge[:,:2]@coef[:2]+coef[2])
        if np.ptp(residual)>=.03 and np.min(np.abs(edge_residual))<=.05:out|=mask
    return out

@dataclass
class Cell:
    z:float=np.nan
    variance:float=np.inf
    observed:float=-np.inf
    score:float=0.
    score_stamp:float=0.
    votes:list=field(default_factory=list)
    latch:bool=False
    positive:float=-np.inf
    blocked:bool=False
    conflict:bool=False
    release:list=field(default_factory=list)

class H3Map:
    def __init__(self,mode):self.mode=mode;self.reset()
    def reset(self):self.cells={};self.last_capture=-np.inf;self.last_id=-1;self.pose_sigma=.01
    def key(self,xy):return tuple(np.floor(np.asarray(xy)/R.resolution+1e-7).astype(int))
    def evidence(self,key,stamp,capture_id,position,yaw,conditions,positive=False,z=None,conflict=False,normal_ir=False):
        cell=self.cells.setdefault(key,Cell());cell.score*=2**(-max(0,stamp-cell.score_stamp)/.5);cell.score_stamp=stamp
        if conflict:
            cell.conflict=True;cell.blocked=True;cell.release=[];return
        if positive:
            cell.blocked=True;cell.release=[];cell.score+=1
            cell.votes=[v for v in cell.votes if stamp-v[0]<=.8]
            if all(v[1]!=capture_id for v in cell.votes):cell.votes.append((stamp,capture_id,np.array(position),yaw,set(conditions)))
            votes=cell.votes
            span=stamp-votes[0][0]
            movement=max(np.linalg.norm(v[2]-votes[0][2]) for v in votes)
            angle=max(abs(np.arctan2(np.sin(v[3]-votes[0][3]),np.cos(v[3]-votes[0][3]))) for v in votes)
            diversity=set().union(*(v[4] for v in votes))
            qualified=len(votes)>=3 and span>=.2-1e-9 and (movement>=.1-1e-9 or angle>=np.deg2rad(10)) and len(diversity)>=2 and cell.score>=2
            if qualified and not cell.conflict:cell.latch=True
            if cell.latch and not cell.conflict:cell.positive=stamp
        elif z is not None:
            if cell.blocked:
                cell.conflict=True
                if normal_ir:
                    if cell.release and abs(z-cell.release[-1][2])>.05:cell.release=[]
                    if not cell.release or cell.release[-1][1]!=capture_id:cell.release.append((stamp,capture_id,z))
                    cell.release=[v for v in cell.release if stamp-v[0]<=.8]
                    if len(cell.release)>=3:
                        cell.blocked=False;cell.conflict=False;cell.latch=False;cell.votes=[];cell.score=0
                else:cell.release=[]
            if not cell.blocked:cell.z=float(z);cell.variance=0.;cell.observed=stamp
    def integrate(self,depth,valid,ir,positions,quats,ks,body_pos,body_quat,stamp,capture_id=None,now=None,pose_sigma=.01):
        capture_id=stamp if capture_id is None else capture_id
        if stamp<=self.last_capture or capture_id<=self.last_id:return
        self.last_capture=stamp;self.last_id=capture_id
        if now is not None and now-stamp>.3:return
        yaw=np.arctan2(rotation(body_quat)[1,0],rotation(body_quat)[0,0]);events={}
        def add(xy,kind,z,camera,distance,normal=False):
            key=self.key(xy);ev=events.setdefault(key,{'positive':False,'z':[],'normal':False,'conditions':set(),'unknown':False})
            ev[kind]=True if kind!='z' else ev[kind]
            if kind=='z':ev['z'].append(float(z));ev['normal']|=normal
            ev['conditions'].add((camera,int(distance/.15)))
        for c in range(len(depth)):
            ray=rays(ks[c],quats[c],*depth[c].shape);points=positions[c]+ray*depth[c][...,None]
            finite=valid[c]&np.isfinite(depth[c])
            local=(points-body_pos)@rotation(body_quat)
            selfmask=(abs(local[...,0])<.43)&(abs(local[...,1])<.27)&(local[...,2]>-.5)&(local[...,2]<.2)
            support=support_normals(points,finite)&~selfmask
            dark=candidates(ir[c]) if self.mode!='h2m' else np.zeros_like(finite)
            suspect=np.zeros_like(finite)
            certified=tread_certificate(points,finite,support)
            regions=[]
            # Missing is shared by h3 and h2m and never gated by IR.
            for missing,source in ((True,~finite),(False,dark&finite)):
                labels,_=ndimage.label(source)
                for lab,sl in enumerate(ndimage.find_objects(labels),1):
                    mask=labels==lab
                    if missing and not ndimage.binary_opening(mask,structure=np.ones((2,2),bool)).any():continue
                    regions.append((missing,mask))
            for missing,mask in regions:
                ring=ndimage.binary_dilation(mask,iterations=5)&support
                if not missing:ring&=~dark
                # Require two actual support sides, not a reference from old fill.
                coords=np.argwhere(mask)
                # Finite-depth intersection can fragment a thick IR band.
                # Reject degenerate regions before requesting two PCA axes.
                if len(coords)<3:continue
                center_xy=coords.mean(0);centered=coords-center_xy
                if np.linalg.matrix_rank(centered.T@centered/(len(coords)-1))<2:continue
                _,_,axes=np.linalg.svd(centered,full_matrices=False)
                grid=np.einsum('ihw,i->hw',np.indices(mask.shape)-center_xy[:,None,None],axes[1]);center=0.
                a=ring&(grid<center);b=ring&(grid>=center)
                plane=fit_plane(points[ring]) if a.sum()>=3 and b.sum()>=3 else None
                if plane is None or pose_sigma>.05:
                    suspect|=mask&~certified
                    for p in points[mask&finite&~certified]:add(p[:2],'unknown',None,c,np.linalg.norm(p-positions[c]))
                    continue
                coef,rms,cov=plane;t=plane_intersection(positions[c],ray,coef);projected=positions[c]+ray*np.nan_to_num(t)[...,None]
                if missing:
                    # Each projected missing cell must be bracketed by measured
                    # ground along body forward, in the same lateral strip.
                    forward=np.array([np.cos(yaw),np.sin(yaw)])
                    lateral=np.array([-np.sin(yaw),np.cos(yaw)])
                    measured=points[ring,:2];footprint=projected[mask,:2];bracketed=np.zeros_like(mask)
                    for yy,xx in np.argwhere(mask):
                        delta=measured-projected[yy,xx,:2]
                        along=delta@forward;across=np.abs(delta@lateral)
                        strip=across<=R.resolution
                        region_delta=footprint-projected[yy,xx,:2]
                        region_along=(region_delta@forward)[np.abs(region_delta@lateral)<=R.resolution]
                        bracketed[yy,xx]=(np.count_nonzero(strip&(along<region_along.min()-.5*R.resolution))>=3 and np.count_nonzero(strip&(along>region_along.max()+.5*R.resolution))>=3)
                    mask=mask&bracketed
                predicted=points[...,:2]@coef[:2]+coef[2];near=np.abs(points[...,2]-predicted)<=.05
                chosen=mask&(~finite if missing else finite&near)&~certified&~selfmask
                suspect|=chosen
                for y,x in np.argwhere(chosen):
                    p=points[y,x] if finite[y,x] else projected[y,x]
                    design=np.r_[p[:2],1.];sigma=np.sqrt(design@cov@design+pose_sigma**2)
                    if not np.isfinite(t[y,x]) or not .15<t[y,x]<5 or sigma>.05:continue
                    add(p[:2],'positive',None,c,np.linalg.norm(p-positions[c]))
            for y,x in np.argwhere(support&~suspect):
                p=points[y,x];normal=self.mode=='h2m' or (np.isfinite(ir[c,y,x]) and 0<=ir[c,y,x]<.995 and not dark[y,x])
                add(p[:2],'z',p[2],c,np.linalg.norm(p-positions[c]),normal)
        for key,ev in events.items():
            z=ev['z'];mixed=bool(z) and (np.ptp(z)>.05 or ev['positive'])
            self.evidence(key,stamp,capture_id,body_pos[:2],yaw,ev['conditions'],positive=ev['positive'],z=np.median(z) if z else None,conflict=mixed or ev['unknown'],normal_ir=ev['normal'])
        # Spatially bounded memory; blocked/expired cells cannot resurrect old ground.
        self.cells={k:v for k,v in self.cells.items() if np.linalg.norm(np.array(k)*R.resolution-body_pos[:2])<3}
    def sample_points(self,xy,scanner,stamp):
        h=np.zeros(len(xy));valid=np.zeros(len(xy));provenance=np.zeros(len(xy),np.uint8);confirmed=np.zeros(len(xy),bool)
        for n,p in enumerate(xy):
            cell=self.cells.get(self.key(p))
            if cell is None:continue
            if cell.latch and not cell.conflict and stamp-cell.positive<=1.0:
                confirmed[n]=True;provenance[n]=INFERRED_GAP
                if self.mode in ('h3','h2m'):h[n]=5;valid[n]=1
            elif not cell.blocked and stamp-cell.observed<=R.ttl and np.isfinite(cell.z):
                h[n]=5*np.clip(scanner-cell.z-.5,-1,1);valid[n]=1;provenance[n]=OBSERVED_HEIGHT
        return h,valid,provenance,confirmed

class BatchMap:
    def __init__(self,n,mode='h3'):
        self.mode=mode;self.legacy=LegacyBatchMap(n,mode) if mode in ('h0','h0d','h0d2','h0p','h0c','h2') else None
        self.ground_legacy=LegacyBatchMap(n,'h2') if mode=='h2m' else None
        self.maps=self.legacy.maps if self.legacy else [H3Map(mode) for _ in range(n)]
        y,x=np.meshgrid(np.linspace(-.5,.5,11),np.linspace(-.8,.8,17),indexing='ij');self.x=x.ravel();self.y=y.ravel()
    def reset(self,ids):
        if self.ground_legacy:self.ground_legacy.reset(ids)
        if self.legacy:self.legacy.reset(ids)
        else:
            for e in ids:self.maps[e].reset()
    def integrate(self,ids,depth,valid,ir,positions,quats,ks,body_pos,body_quat,stamp,now=None,pose_sigma=.01):
        if self.legacy:return self.legacy.integrate(ids,depth,valid,ir,positions,quats,ks,body_pos,body_quat,stamp)
        if self.ground_legacy:
            self.ground_legacy.integrate(ids,depth,valid,np.full_like(depth,-1.),positions,quats,ks,body_pos,body_quat,stamp)
        for e in ids:self.maps[e].integrate(depth[e],valid[e],ir[e],positions[e],quats[e],ks[e],body_pos[e],body_quat[e],stamp,now=now,pose_sigma=pose_sigma)
    def sample(self,poses,scanner,stamp):
        if self.legacy:
            out=self.legacy.sample(poses,scanner,stamp);self.provenance=np.where(out[:,187:]>0,OBSERVED_HEIGHT,UNKNOWN).astype(np.uint8);self.confirmed=np.zeros((len(poses),187),bool);return out
        baseline=self.ground_legacy.sample(poses,scanner,stamp) if self.ground_legacy else None
        output=[];provenance=[];confirmed=[]
        for e,p in enumerate(poses):
            c,s=np.cos(p[2]),np.sin(p[2]);xy=np.c_[p[0]+c*self.x-s*self.y,p[1]+s*self.x+c*self.y]
            h,v,src,gap=self.maps[e].sample_points(xy,scanner[e],stamp)
            if baseline is not None:
                blocked=np.array([self.maps[e].cells.get(self.maps[e].key(p),Cell()).blocked for p in xy])
                h=np.where(blocked,h,baseline[e,:187]);v=np.where(blocked,v,baseline[e,187:])
                src=np.where(blocked,src,np.where(v>0,OBSERVED_HEIGHT,UNKNOWN)).astype(np.uint8)
            output.append(np.r_[h,v]);provenance.append(src);confirmed.append(gap)
        self.provenance=np.array(provenance);self.confirmed=np.array(confirmed);return np.array(output,dtype=np.float32)
