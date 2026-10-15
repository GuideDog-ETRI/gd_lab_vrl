"""CPU-only causal elevation mapping. No terrain/scanner/GT imports.

All distances metres, optical depth, camera-to-world OpenGL wxyz. A missing
ray is localized on the estimated support plane, NOT at the far sentinel.
This localization is a hypothesis, and cannot resolve arbitrary occlusion.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage

@dataclass(frozen=True)
class Rules:
    resolution: float = .05
    size: int = 80
    ttl: float = 2.
    evidence_tau: float = .8
    gap_depth: float = .30
    finite_step: float = .25
    height_consistency: float = .05
    normal_z: float = .80
    confirm: float = 2.4
    ground_confirm: float = 1.7
    veto_seconds: float = .4
    ir_contrast: float = .15
    band_min_pixels: int = 6
    band_min_width: int = 2
    band_min_length: int = 3
    band_paired_fraction: float = .25
    noise_sigma: float = .01
    dropout: float = .03
    deep_hole: float = .60
    false_fill: float = .25
    flying: float = .15
    spike: float = .006
    ir_sigma: float = .015
R=Rules()

def rotation(q):
    q=np.asarray(q,dtype=float);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(q.shape[:-1]+(3,3))

def rays(k,q,h,w):
    v,u=np.mgrid[:h,:w]
    local=np.stack(((u-k[0,2])/k[0,0],-(v-k[1,2])/k[1,1],-np.ones((h,w))),-1)
    return local@rotation(q).T

def dark_regions(ir,r=R):
    # Per-image affine-invariant normalization. No absolute IR threshold and
    # no rectangle/box occupancy assumption. Curved/perspective bands can vary
    # in width and direction. Require bright-dark-bright cross-sections on
    # either image axis; a one-sided silhouette alone is not a gap band.
    ir=np.asarray(ir,dtype=float)
    finite=np.isfinite(ir)&(ir>=0)
    if not finite.any():return np.zeros(ir.shape,dtype=bool)
    lo,hi=np.quantile(ir[finite],[.01,.99])
    if hi-lo<=np.finfo(float).eps*max(1.,abs(hi)):return np.zeros(ir.shape,dtype=bool)
    norm=np.clip((np.where(finite,ir,hi)-lo)/(hi-lo),0,1)
    dark=finite&(norm<=np.quantile(norm[finite],.75)-r.ir_contrast)
    # Preserve the band interior even when wider than the local surround kernel;
    # the two endpoint comparisons below supply the local boundary evidence.
    paired=np.zeros_like(dark)
    for transpose in (False,True):
        mask=dark.T if transpose else dark
        values=norm.T if transpose else norm
        votes=paired.T if transpose else paired
        for row in range(mask.shape[0]):
            changes=np.diff(np.r_[False,mask[row],False].astype(int))
            for start,end in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
                if start==0 or end==mask.shape[1] or end-start<r.band_min_width:continue
                interior=np.median(values[row,start:end])
                left=np.median(values[row,max(0,start-3):start])
                right=np.median(values[row,end:min(end+3,mask.shape[1])])
                if min(left,right)-interior>=r.ir_contrast:votes[row,start:end]=True
    labels,n=ndimage.label(dark);accepted=np.zeros_like(dark)
    for i,sl in enumerate(ndimage.find_objects(labels),1):
        if sl is None:continue
        patch=labels[sl]==i;count=int(patch.sum())
        paired_patch=paired[sl]&patch
        length=max(np.count_nonzero(paired_patch.any(0)),np.count_nonzero(paired_patch.any(1)))
        if count>=r.band_min_pixels and length>=r.band_min_length and paired_patch.sum()/count>=r.band_paired_fraction:
            accepted[sl]|=patch
    return accepted

def support_normals(points,valid,r=R):
    # Central differences reject mixed-depth edges and vertical risers. Missing
    # neighbors cannot certify a horizontal return.
    dx=np.zeros_like(points);dy=np.zeros_like(points)
    dx[:,1:-1]=points[:,2:]-points[:,:-2];dy[1:-1]=points[2:]-points[:-2]
    normal=np.cross(dx,dy);norm=np.linalg.norm(normal,axis=-1)
    good=valid.copy();good[[0,-1]]=False;good[:,[0,-1]]=False
    good[1:-1,1:-1]&=valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]
    # Permit sloped terrain up to 37 degrees; discontinuities >25cm are not treads.
    return good&(np.abs(normal[...,2])>=r.normal_z*np.maximum(norm,1e-12))&(norm>1e-10)&(np.abs(dx[...,2])<=r.finite_step)&(np.abs(dy[...,2])<=r.finite_step)

def vertical_returns(points,valid):
    """Certify finite vertical surfaces; never use a riser as missing floor."""
    dx=np.zeros_like(points);dy=np.zeros_like(points)
    dx[:,1:-1]=points[:,2:]-points[:,:-2];dy[1:-1]=points[2:]-points[:-2]
    normal=np.cross(dx,dy);norm=np.linalg.norm(normal,axis=-1)
    neighbors=ndimage.minimum_filter(valid.astype(np.uint8),size=3,mode='constant',cval=0)>0
    return neighbors&(norm>1e-10)&(np.abs(normal[...,2])<.3*norm)

def corrupt(depth,valid,proxy,positions,quats,ks,reference_z,rng,noisy,ir_mode,r=R):
    """Independent sensor RNG. Synthetic IR uses PRE-corruption optical range.
    Return-loss holes are coupled to dark IR; false-fill leaves IR tied to the
    underlying range. This favourable simulated cue is explicitly not evidence
    of real active-IR performance. No terrain-family or GT labels are used.
    """
    ir_rng=np.random.default_rng(rng.integers(0,2**63)) # identical depth RNG across IR modes
    d=depth.copy();v=valid.copy();irs=proxy.copy();counts={k:0 for k in ['hole','false_fill','flying','spike','dropout']}
    for c in range(len(d)):
        ray=rays(ks[c],quats[c],*d[c].shape)
        plane=np.divide(reference_z-positions[c,2],ray[...,2],out=np.full_like(d[c],np.nan),where=ray[...,2]<-.1)
        z=positions[c,2]+ray[...,2]*depth[c]
        deep=valid[c]&(z<reference_z-r.gap_depth)&(plane>.15)&(plane<5)
        holes=np.zeros_like(deep)
        if noisy:
            holes=deep&(rng.random(deep.shape)<r.deep_hole)
            fill=deep&~holes&(rng.random(deep.shape)<r.false_fill)
            edge=(ndimage.maximum_filter(depth[c],3)-ndimage.minimum_filter(depth[c],3)>.15)&valid[c]
            fly=edge&(rng.random(deep.shape)<r.flying)
            spike=valid[c]&(rng.random(deep.shape)<r.spike)
            drop=rng.random(deep.shape)<r.dropout
            d[c][fill]=plane[fill]
            d[c][fly]=.5*(ndimage.maximum_filter(depth[c],3)[fly]+ndimage.minimum_filter(depth[c],3)[fly])
            d[c][spike]=rng.uniform(.15,5,size=spike.sum())
            d[c]+=rng.normal(0,r.noise_sigma,d[c].shape)
            v[c]&=~holes&~drop&(d[c]>.15)&(d[c]<5)
            for key,mask in [('hole',holes),('false_fill',fill),('flying',fly),('spike',spike),('dropout',drop)]:counts[key]+=int(mask.sum())
        if ir_mode=='synthetic':
            # Reflectance is spatially smooth, independent of terrain identity.
            reflectance=.55+.1*ndimage.gaussian_filter(ir_rng.normal(size=d[c].shape),3)
            irs[c]=np.clip(.04+reflectance/((np.maximum(depth[c],.15)*np.linalg.norm(ray,axis=-1))**2+.2),0,1)
            irs[c][~valid[c]|holes]=.025
            irs[c]=np.clip(irs[c]+ir_rng.normal(0,r.ir_sigma,d[c].shape),0,1)
        elif ir_mode!='proxy':raise ValueError(ir_mode)
    d[~v]=5.
    return d,v,irs,counts

class ElevationMap:
    def __init__(self,mode='h1',rules=R):
        self.r=rules;self.mode=mode;self.reset()
    def reset(self):
        s=(self.r.size,self.r.size)
        self.origin=None;self.z=np.full(s,np.nan);self.var=np.full(s,np.inf)
        self.last=np.full(s,-np.inf);self.finite=np.full(s,-np.inf)
        self.ground=np.zeros(s);self.gap=np.zeros(s);self.noise=np.zeros(s)
        self.clock=None;self.last_capture=-np.inf
    def recenter(self,xy):
        new=np.floor(np.asarray(xy)/self.r.resolution).astype(int)-self.r.size//2
        if self.origin is not None:
            shift=new-self.origin
            for a in [self.z,self.var,self.last,self.finite,self.ground,self.gap,self.noise]:
                fill=np.nan if a is self.z else np.inf if a is self.var else -np.inf if a is self.last or a is self.finite else 0.
                if np.max(np.abs(shift))>=self.r.size:a[:]=fill;continue
                a[:]=np.roll(a,(-shift[1],-shift[0]),axis=(0,1))
                if shift[0]>0:a[:,-shift[0]:]=fill
                elif shift[0]<0:a[:,:-shift[0]]=fill
                if shift[1]>0:a[-shift[1]:]=fill
                elif shift[1]<0:a[:-shift[1]]=fill
        self.origin=new
    def indices(self,xy):
        ij=np.floor(np.asarray(xy)/self.r.resolution+1e-7).astype(int)-self.origin
        return ij,(ij[...,0]>=0)&(ij[...,0]<self.r.size)&(ij[...,1]>=0)&(ij[...,1]<self.r.size)
    def reference(self,xy,base_z):
        if self.origin is None:return base_z-.5
        fresh=(self.last>self.last_capture-self.r.ttl)&np.isfinite(self.z)&(self.ground>=self.r.ground_confirm)
        yy,xx=np.mgrid[:self.r.size,:self.r.size];wx=(xx+self.origin[0]+.5)*self.r.resolution;wy=(yy+self.origin[1]+.5)*self.r.resolution
        near=fresh&(np.hypot(wx-xy[0],wy-xy[1])<.45)
        return float(np.median(self.z[near])) if near.any() else base_z-.5
    def integrate(self,depth,valid,ir,positions,quats,ks,body_pos,body_quat,stamp):
        if stamp<=self.last_capture:return # duplicate/reordered captures cannot add evidence
        r=self.r;self.recenter(body_pos[:2]);reference=self.reference(body_pos[:2],body_pos[2])
        decay=np.exp(-(stamp-self.clock)/r.evidence_tau) if self.clock is not None else 1.
        self.ground*=decay;self.gap*=decay;self.noise*=decay;self.clock=stamp;self.last_capture=stamp
        size=r.size*r.size;zs=[[] for _ in range(size)];gap_cells=set();noise_cells=set()
        for c in range(len(depth)):
            ray=rays(ks[c],quats[c],*depth[c].shape)
            points=positions[c]+ray*depth[c][...,None]
            support=support_normals(points,valid[c],r)
            local=(points-body_pos)@rotation(body_quat)
            # Conservative torso/leg envelope; hidden ground must come from memory.
            self_mask=(np.abs(local[...,0])<.43)&(np.abs(local[...,1])<.27)&(local[...,2]>-.50)&(local[...,2]<.20)
            support&=~self_mask
            dark=dark_regions(ir[c],r) if self.mode=='h1' else np.zeros_like(valid[c])
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
            if self.mode=='h0':ground=support&inside
            for idx,z in zip((ij[...,1]*r.size+ij[...,0])[ground],points[...,2][ground]):zs[int(idx)].append(float(z))
            if self.mode!='h1':continue
            ambiguous=((~valid[c])|(deep&~self_mask))&~vertical_returns(points,valid[c])
            t=np.divide(reference-positions[c,2],ray[...,2],out=np.full_like(depth[c],np.nan),where=ray[...,2]<-.1)
            projected=positions[c]+ray*np.nan_to_num(t,nan=0)[...,None]
            footprint,ok=self.indices(projected[...,:2]);ok&=np.isfinite(t)&(t>.15)&(t<5)
            footprint_local=(projected-body_pos)@rotation(body_quat)
            ok&=~((np.abs(footprint_local[...,0])<.43)&(np.abs(footprint_local[...,1])<.27))
            cells=footprint[...,1]*r.size+footprint[...,0]
            gap_cells.update(cells[ok&ambiguous&dark].tolist())
            noise_cells.update(cells[ok&ambiguous&~dark&np.isfinite(ir[c])&(ir[c]>=0)].tolist())
        ground_cells=set()
        for idx,values in enumerate(zs):
            if not values:continue
            j,i=divmod(idx,r.size);v=np.asarray(values);z=float(np.median(v));var=float(np.var(v))
            # Multi-height/riser mixtures are never averaged into a ramp.
            if np.ptp(v)>r.finite_step:continue
            ground_cells.add(idx)
            old=self.z[j,i];consistent=np.isfinite(old) and abs(old-z)<=r.height_consistency
            self.ground[j,i]=self.ground[j,i]+1 if consistent else 1
            self.z[j,i]=.5*(old+z) if consistent else z
            self.var[j,i]=var;self.last[j,i]=stamp;self.finite[j,i]=stamp
            # Finite treads decisively veto dark-IR gap evidence, incl false fill.
            self.gap[j,i]=0;self.noise[j,i]=0
        for cells,array in [(gap_cells,self.gap),(noise_cells,self.noise)]:
            for idx in cells-ground_cells:
                j,i=divmod(idx,r.size)
                if stamp-self.finite[j,i]<=r.veto_seconds:continue
                array[j,i]+=1;self.last[j,i]=stamp
    def sample(self,xyyaw,scanner_z,stamp):
        y,x=np.meshgrid(np.linspace(-.5,.5,11),np.linspace(-.8,.8,17),indexing='ij')
        c,s=np.cos(xyyaw[2]),np.sin(xyyaw[2]);xy=np.stack((xyyaw[0]+c*x-s*y,xyyaw[1]+s*x+c*y),-1)
        ij,inside=self.indices(xy);i=ij[...,0].clip(0,self.r.size-1);j=ij[...,1].clip(0,self.r.size-1)
        fresh=inside&(stamp-self.last[j,i]<=self.r.ttl)
        z=self.z[j,i].copy();g=self.ground[j,i]
        valid=fresh&np.isfinite(z)&(self.var[j,i]<.0025)
        if self.mode=='h1':valid&=g>=self.r.ground_confirm
        gap=fresh&(self.gap[j,i]>=self.r.confirm)&(self.gap[j,i]>self.noise[j,i])&(stamp-self.finite[j,i]>self.r.veto_seconds)
        if self.mode=='h1':
            # Noise fill requires >=3 nearby mutually consistent certified ground
            # cells, no certified gap neighbor, and remains invalid otherwise.
            for a,b in np.argwhere(fresh&~valid&(self.noise[j,i]>=self.r.confirm)):
                jj,ii=j[a,b],i[a,b];sl=(slice(max(0,jj-2),min(self.r.size,jj+3)),slice(max(0,ii-2),min(self.r.size,ii+3)))
                known=(self.ground[sl]>=self.r.ground_confirm)&(stamp-self.finite[sl]<self.r.ttl)&np.isfinite(self.z[sl])
                vals=self.z[sl][known]
                if vals.size>=3 and np.ptp(vals)<=self.r.height_consistency and not (self.gap[sl]>=self.r.confirm).any():z[a,b]=np.median(vals);valid[a,b]=True
        else:gap[:]=False
        h=5*np.clip(scanner_z-z-.5,-1,1);h=np.where(gap,5,np.where(valid,h,0));valid|=gap
        return np.concatenate((h.ravel(),valid.ravel())).astype(np.float32),gap.ravel()
