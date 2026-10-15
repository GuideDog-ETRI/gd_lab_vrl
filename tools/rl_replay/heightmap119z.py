"""UPDATE117 batch geometry, grouped reductions and 187-cell sampling.
Frozen rules imported verbatim; image components and sparse noise fill retain original logic.
"""
from heightmap116 import *

# UPDATE129: fixed a priori, not tuned on results.
DEEP_M=.10
CONFIRM_DEEP=1.5  # >=2 captures within ~0.5 s (decay tau 0.8 s); fixed a priori

PATCH=dict(half_cells=5,min_cells=6,max_rms=.02,max_slope=.75)  # +-0.25 m window, fixed a priori

def patch_planes(m,stamp):
    """Local plane under every map cell, fitted to fresh certified ground in a +-0.25 m window.
    Cells flagged deep (m.gap>0.01 in h0p) are excluded. Returns (plane_z, ok)."""
    r=m.r;S=r.size;k=2*PATCH['half_cells']+1
    ok=np.isfinite(m.z)&(stamp-m.finite<r.ttl)&(m.var<.0025)&~(m.gap>.01)
    w=ok.astype(float);z=np.where(ok,m.z,0.)
    yy,xx=np.mgrid[:S,:S].astype(float)*r.resolution
    box=lambda a:ndimage.uniform_filter(a,size=k,mode='constant')*k*k
    Sw=box(w);Sx=box(w*xx);Sy=box(w*yy);Sz=box(w*z);Sxx=box(w*xx*xx);Sxy=box(w*xx*yy);Syy=box(w*yy*yy);Sxz=box(w*xx*z);Syz=box(w*yy*z);Szz=box(w*z*z)
    A=np.stack([np.stack([Sxx,Sxy,Sx],-1),np.stack([Sxy,Syy,Sy],-1),np.stack([Sx,Sy,Sw],-1)],-2);B=np.stack([Sxz,Syz,Sz],-1)
    det=np.linalg.det(A);good=(Sw>=PATCH['min_cells'])&(np.abs(det)>1e-12)
    coef=np.zeros(B.shape);coef[good]=np.linalg.solve(A[good],B[good][...,None])[...,0]
    a,b,c=coef[...,0],coef[...,1],coef[...,2]
    sse=Szz-(a*Sxz+b*Syz+c*Sz);rms=np.sqrt(np.maximum(sse,0)/np.maximum(Sw,1))
    good&=(rms<=PATCH['max_rms'])&(np.hypot(a,b)<=PATCH['max_slope'])
    return a*xx+b*yy+c,good

class FastMap(ElevationMap):
    def integrate(self,depth,valid,ir,positions,quats,ks,body_pos,body_quat,stamp):
        if stamp<=self.last_capture:return # duplicate/reordered captures cannot add evidence
        r=self.r;self.recenter(body_pos[:2]);reference=self.reference(body_pos[:2],body_pos[2])
        decay=np.exp(-(stamp-self.clock)/r.evidence_tau) if self.clock is not None else 1.
        self.ground*=decay;self.gap*=decay;self.noise*=decay;self.clock=stamp;self.last_capture=stamp
        size=r.size*r.size;zs=[]; zindices=[];gap_cells=set();noise_cells=set()
        if self.mode in ('h0p','h0c'):plane,plane_ok=patch_planes(self,stamp)
        for c in range(len(depth)):
            ray=self.precomputed_rays[c]
            points=positions[c]+ray*depth[c][...,None]
            support=support_normals(points,valid[c],r)
            local=(points-body_pos)@rotation(body_quat)
            # Conservative torso/leg envelope; hidden ground must come from memory.
            self_mask=(np.abs(local[...,0])<.43)&(np.abs(local[...,1])<.27)&(local[...,2]>-.50)&(local[...,2]<.20)
            support&=~self_mask
            dark=dark_regions(ir[c],r) if self.mode in ('h1','h2') else np.zeros_like(valid[c])
            ij,inside=self.indices(points[...,:2])
            # Reference nearby already observed support, keeping descending stairs
            # local rather than comparing every lower tread to the robot's deck.
            local_ref=np.full_like(depth[c],reference)
            ii=ij[...,0].clip(0,r.size-1);jj=ij[...,1].clip(0,r.size-1)
            known=inside&np.isfinite(self.z[jj,ii])&(stamp-self.finite[jj,ii]<r.ttl)
            local_ref[known]=self.z[jj[known],ii[known]]
            deep=points[...,2]<local_ref-r.gap_depth-1e-7
            # Treads connected by <=25cm finite steps are support even when
            # their total descent below the initial deck exceeds 30cm.
            reachable=support&~deep
            for _ in range(12):
                hi=ndimage.maximum_filter(np.where(reachable,points[...,2],-np.inf),size=7)
                lo=ndimage.minimum_filter(np.where(reachable,points[...,2],np.inf),size=7)
                linked=support&((np.abs(points[...,2]-hi)<=r.finite_step)|(np.abs(points[...,2]-lo)<=r.finite_step))
                expanded=reachable|linked
                if np.array_equal(expanded,reachable):break
                reachable=expanded
            deep&=~reachable
            ground=support&inside&~deep
            if self.mode in ('h0','h2'):ground=support&inside
            if self.mode in ('h0p','h0c'):
                # UPDATE129 plane-patch deep test: below the local ground plane of that cell by >=DEEP_M.
                pz=plane[jj,ii];pok=inside&plane_ok[jj,ii]
                ground=(support|(valid[c]&~self_mask&pok&(points[...,2]<pz-DEEP_M)))&inside
            if self.mode in ('h0d','h0d2'):
                # UPDATE129 deep-first: returns >=10 cm below local support are kept even when
                # their beam neighbours jump (narrow pit bottoms), never discarded as non-support.
                ground=(support|(valid[c]&~self_mask&(points[...,2]<local_ref-DEEP_M)))&inside
            zindices.append((ij[...,1]*r.size+ij[...,0])[ground]);zs.append(points[...,2][ground])
            if self.mode not in ('h1','h2'):continue
            ambiguous=((~valid[c])|(deep&~self_mask))&~vertical_returns(points,valid[c])
            if self.mode=='h2':ambiguous=(~valid[c])&~vertical_returns(points,valid[c])
            t=np.divide(reference-positions[c,2],ray[...,2],out=np.full_like(depth[c],np.nan),where=ray[...,2]<-.1)
            projected=positions[c]+ray*np.nan_to_num(t,nan=0)[...,None]
            footprint,ok=self.indices(projected[...,:2]);ok&=np.isfinite(t)&(t>.15)&(t<5)
            footprint_local=(projected-body_pos)@rotation(body_quat)
            ok&=~((np.abs(footprint_local[...,0])<.43)&(np.abs(footprint_local[...,1])<.27))
            cells=footprint[...,1]*r.size+footprint[...,0]
            gap_cells.update(cells[ok&ambiguous&dark].tolist())
            noise_cells.update(cells[ok&ambiguous&~dark&np.isfinite(ir[c])&(ir[c]>=0)].tolist())
        ground_cells=set()
        indices=np.concatenate(zindices);values=np.concatenate(zs).astype(float)
        if len(indices):
            order=np.lexsort((values,indices));indices=indices[order];values=values[order]
            ids,starts,counts=np.unique(indices,return_index=True,return_counts=True)
            ends=starts+counts-1
            zs=(values[starts+(counts-1)//2]+values[starts+counts//2])*.5
            means=np.add.reduceat(values,starts)/counts
            variances=np.add.reduceat((values-np.repeat(means,counts))**2,starts)/counts
            keep=values[ends]-values[starts]<=r.finite_step
            if self.mode=='h0p':
                cj,ci=np.divmod(ids,r.size)
                deepcell=plane_ok[cj,ci]&(values[starts]<plane[cj,ci]-DEEP_M)
                zs=np.where(deepcell,values[starts],zs);variances=np.where(deepcell,0.,variances);keep|=deepcell
                deep_ids=set(ids[deepcell].tolist())
            if self.mode=='h0c':
                # Camera: a plane-deep cell is accepted only after deep evidence in >=2 captures
                # (decayed count in self.noise, tau 0.8 s); unconfirmed deep cells are left as in h0.
                cj,ci=np.divmod(ids,r.size)
                cand=plane_ok[cj,ci]&(values[starts]<plane[cj,ci]-DEEP_M)
                self.noise[cj[cand],ci[cand]]+=1.
                conf=cand&(self.noise[cj,ci]>=CONFIRM_DEEP)
                zs=np.where(conf,values[starts],zs);variances=np.where(conf,0.,variances)
                keep=(keep&~cand)|conf
                deep_ids=set(ids[conf].tolist())
            if self.mode in ('h0d','h0d2'):
                # Mixed cells holding a deep return take the deepest value instead of being dropped.
                deepcell=values[starts]<reference-DEEP_M
                if self.mode=='h0d2':
                    # Confirmation: >=2 deep returns in this capture, or the cell was already deep and fresh.
                    deepcount=np.add.reduceat((values<reference-DEEP_M).astype(int),starts)
                    pj,pi=np.divmod(ids,r.size);prev=self.z[pj,pi]
                    was=np.isfinite(prev)&(prev<reference-DEEP_M)&(stamp-self.finite[pj,pi]<r.ttl)
                    deepcell&=(deepcount>=2)|was
                zs=np.where(deepcell,values[starts],zs);variances=np.where(deepcell,0.,variances);keep|=deepcell
            ids=ids[keep];zs=zs[keep];variances=variances[keep];j,i=np.divmod(ids,r.size)
            ground_cells=set(ids.tolist());old=self.z[j,i]
            consistent=np.isfinite(old)&(np.abs(old-zs)<=r.height_consistency)
            self.ground[j,i]=np.where(consistent,self.ground[j,i]+1,1)
            if self.mode in ('h0p','h0c'):
                # A fresh deep-flagged cell is not overwritten by a later shallower reading.
                hold=np.isfinite(old)&(self.gap[j,i]>.01)&(stamp-self.finite[j,i]<r.ttl)&(zs>old+r.height_consistency)
                zs=np.where(hold,old,zs);variances=np.where(hold,0.,variances);consistent&=~hold
                newflag=np.array([x in deep_ids for x in ids.tolist()],bool)|hold
            if self.mode in ('h0d','h0d2'):
                # A fresh deep cell is not overwritten by a later shallower reading.
                hold=np.isfinite(old)&(old<reference-DEEP_M)&(stamp-self.finite[j,i]<r.ttl)&(zs>old+r.height_consistency)
                zs=np.where(hold,old,zs);variances=np.where(hold,0.,variances);consistent&=~hold
            self.z[j,i]=np.where(consistent,.5*(old+zs),zs)
            self.var[j,i]=variances;self.last[j,i]=stamp;self.finite[j,i]=stamp
            self.gap[j,i]=0
            if self.mode!='h0c':self.noise[j,i]=0
            if self.mode in ('h0p','h0c'):self.gap[j,i]=np.where(newflag,1.,0.)  # deep flag (<confirm, never emitted as +5)
        for cells,array in [(gap_cells,self.gap),(noise_cells,self.noise)]:
            for idx in cells-ground_cells:
                j,i=divmod(idx,r.size)
                if stamp-self.finite[j,i]<=r.veto_seconds:continue
                array[j,i]+=1;self.last[j,i]=stamp

class BatchMap:
    def __init__(self,n,mode='h2'):
        assert mode in ('h2','h0','h0d','h0d2','h0p','h0c');self.mode=mode
        self.maps=[FastMap(mode) for _ in range(n)]
        self.arrays={name:np.stack([getattr(m,name) for m in self.maps]) for name in ('z','var','last','finite','ground','gap','noise')}
        for name,a in self.arrays.items():
            for e,m in enumerate(self.maps):setattr(m,name,a[e])
        y,x=np.meshgrid(np.linspace(-.5,.5,11),np.linspace(-.8,.8,17),indexing='ij')
        self.x=x.ravel();self.y=y.ravel()
    def reset(self,ids):
        for e in ids:
            m=self.maps[e];m.reset()
            for name,a in self.arrays.items():a[e]=getattr(m,name);setattr(m,name,a[e])
    def integrate(self,ids,depth,valid,ir,positions,quats,ks,body_pos,body_quat,stamp):
        # Bound temporary memory while vectorizing camera geometry across env/view/pixels.
        for offset in range(0,len(ids),16):
            rows=np.asarray(ids[offset:offset+16]);k=ks[rows];q=quats[rows]
            v,u=np.mgrid[:depth.shape[-2],:depth.shape[-1]]
            local=np.stack((np.broadcast_to((u-k[...,0,2,None,None])/k[...,0,0,None,None],(len(rows),4,*u.shape)),
                            np.broadcast_to(-(v-k[...,1,2,None,None])/k[...,1,1,None,None],(len(rows),4,*u.shape)),
                            -np.ones((len(rows),4,*u.shape))),-1)
            rays_batch=np.einsum('nvhwj,nvij->nvhwi',local,rotation(q))
            for j,e in enumerate(rows):
                m=self.maps[e];m.precomputed_rays=rays_batch[j]
                m.integrate(depth[e],valid[e],ir[e],positions[e],quats[e],ks[e],body_pos[e],body_quat[e],stamp)
                m.precomputed_rays=None
    def sample(self,poses,scanner,stamp):
        origins=np.array([m.origin if m.origin is not None else (0,0) for m in self.maps]);active=np.array([m.origin is not None for m in self.maps])
        c=np.cos(poses[:,2,None]);s=np.sin(poses[:,2,None])
        xy=np.stack((poses[:,0,None]+c*self.x-s*self.y,poses[:,1,None]+s*self.x+c*self.y),-1)
        ij=np.floor(xy/R.resolution+1e-7).astype(int)-origins[:,None]
        inside=(ij>=0).all(-1)&(ij<R.size).all(-1)&active[:,None]
        i=ij[...,0].clip(0,R.size-1);j=ij[...,1].clip(0,R.size-1);e=np.arange(len(poses))[:,None]
        a={name:value[e,j,i] for name,value in self.arrays.items()}
        fresh=inside&(stamp-a['last']<=R.ttl);z=a['z'].copy()
        valid=fresh&np.isfinite(z)&(a['var']<.0025)
        if self.mode in ('h0d','h0d2','h0p','h0c'):
            # Policy cell (10 cm) reads the deepest valid 5 cm map cell under its footprint.
            for dx,dy in ((-.025,-.025),(-.025,.025),(.025,-.025),(.025,.025)):
                ox=self.x+dx;oy=self.y+dy
                q=np.stack((poses[:,0,None]+c*ox-s*oy,poses[:,1,None]+s*ox+c*oy),-1)
                qij=np.floor(q/R.resolution+1e-7).astype(int)-origins[:,None]
                qin=(qij>=0).all(-1)&(qij<R.size).all(-1)&active[:,None]
                qi=qij[...,0].clip(0,R.size-1);qj=qij[...,1].clip(0,R.size-1)
                qz=self.arrays['z'][e,qj,qi];qok=qin&(stamp-self.arrays['last'][e,qj,qi]<=R.ttl)&np.isfinite(qz)&(self.arrays['var'][e,qj,qi]<.0025)
                better=qok&(~valid|(qz<z))
                z=np.where(better,qz,z);valid|=qok
        if self.mode not in ('h0','h0d','h0d2','h0p','h0c'):valid&=a['ground']>=R.ground_confirm
        gap=fresh&(a['gap']>=R.confirm)&(a['gap']>a['noise'])&(stamp-a['finite']>R.veto_seconds)
        for row,col in np.argwhere(fresh&~valid&(a['noise']>=R.confirm)&(self.mode!='h0c')):
            m=self.maps[row];jj,ii=j[row,col],i[row,col];sl=(slice(max(0,jj-2),min(R.size,jj+3)),slice(max(0,ii-2),min(R.size,ii+3)))
            known=(m.ground[sl]>=R.ground_confirm)&(stamp-m.finite[sl]<R.ttl)&np.isfinite(m.z[sl]);vals=m.z[sl][known]
            if vals.size>=3 and np.ptp(vals)<=R.height_consistency and not (m.gap[sl]>=R.confirm).any():z[row,col]=np.median(vals);valid[row,col]=True
        h=5*np.clip(scanner[:,None]-z-.5,-1,1);h=np.where(gap,5,np.where(valid,h,0));valid|=gap
        return np.concatenate((h,valid),-1).astype(np.float32)
