"""Privileged generator only. Geometry and audit masks never enter mapper API."""
import numpy as np
from scipy import ndimage
from heightmap116 import R, rays, rotation

def rng(seed,env,episode,capture,camera,channel):
    return np.random.default_rng(np.random.SeedSequence([seed,env,episode,capture,camera,channel]))

def inside_polygon(xy,polygon):
    inside=np.zeros(xy.shape[:-1],bool);x,y=xy[...,0],xy[...,1]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        inside^=((a[1]>y)!=(b[1]>y))&(x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1]+1e-30)+a[0])
    return inside

def clip_polygon(poly,bounds):
    for axis,limit,sign in [(0,bounds[0],1),(0,bounds[1],-1),(1,bounds[2],1),(1,bounds[3],-1)]:
        out=[]
        for a,b in zip(poly,np.roll(poly,-1,axis=0)):
            ain=sign*(a[axis]-limit)>=0;bin=sign*(b[axis]-limit)>=0
            if ain:out.append(a)
            if ain!=bin:out.append(a+(b-a)*(limit-a[axis])/(b[axis]-a[axis]))
        poly=np.asarray(out)
        if not len(poly):break
    return poly

class PatchField:
    """Static union area measured on 2mm world quadrature, not image pixels."""
    def __init__(self,bounds,seed):
        self.bounds=bounds;self.polygons=[];self.offsets=[]
        x0,x1,y0,y1=bounds
        nx=max(1,int(np.ceil((x1-x0)/.002)));ny=max(1,int(np.ceil((y1-y0)/.002)))
        y,x=np.meshgrid(y0+(np.arange(ny)+.5)*(y1-y0)/ny,x0+(np.arange(nx)+.5)*(x1-x0)/nx,indexing='ij')
        xy=np.stack((x,y),-1);covered=np.zeros(x.shape,bool);target=int(round(.25*covered.size));random=np.random.default_rng(seed)
        for attempt in range(10000):
            if covered.sum()>=target:break
            center=np.array([random.uniform(x0,x1),random.uniform(y0,y1)]);n=int(random.integers(4,9))
            angles=np.arange(n)*2*np.pi/n+random.uniform(0,2*np.pi)
            length=random.uniform(.2,.6);width=random.uniform(.05,.20)
            # Long direction follows the slot; original polygon has 4--8 vertices.
            poly=center+np.c_[np.cos(angles)*width/2,np.sin(angles)*length/2]
            poly=clip_polygon(poly,bounds)
            if len(poly)<3:continue
            select=inside_polygon(xy,poly)&~covered
            if covered.sum()+select.sum()>target:
                low,high=0.,1.
                for _ in range(22):
                    mid=(low+high)/2;candidate=center+(poly-center)*mid
                    count=(inside_polygon(xy,candidate)&~covered).sum()
                    if covered.sum()+count<=target:low=mid
                    else:high=mid
                poly=center+(poly-center)*low;select=inside_polygon(xy,poly)&~covered
            if not select.any():continue
            covered|=select;self.polygons.append(poly);self.offsets.append(random.uniform(-.02,.02))
            if target-covered.sum()<=2:break
        self.area_fraction=float(covered.mean());self.quadrature_m=.002
    def sample(self,xy):
        selected=np.zeros(xy.shape[:-1],bool);offset=np.zeros(xy.shape[:-1])
        for poly,z in zip(self.polygons,self.offsets):
            mask=inside_polygon(xy,poly);selected|=mask;offset[mask]=z
        return selected,offset

def materials(xyz):
    """World-coordinate material field independent of terrain labels.
    10% periodic dark paint; 20% bright tiles, including gap floors.
    """
    x,y,z=np.moveaxis(np.nan_to_num(xyz),-1,0)
    dark=np.mod(x+.37*y,2.)<.2
    bright=np.mod(y+.11*x,1.)<.2
    shadow=np.mod(y-.23*x,3.)<.10
    reflectance=np.where(bright,.95,.50);reflectance=np.where(dark|shadow,.035,reflectance)
    return reflectance,dark,bright&~dark&~shadow,shadow

def physical_ir(points,normals,hit,projector,visible,random):
    rho,dark,bright,shadow=materials(points)
    delta=projector-points;distance=np.linalg.norm(delta,axis=-1);direction=delta/np.maximum(distance[...,None],1e-9)
    cosine=np.maximum(0,np.sum(normals*direction,-1))
    clean=.04+1.3*rho*cosine*visible/(distance**2+.05)
    exposure=random.uniform(.75,1.25);image=np.clip(exposure*clean+random.normal(0,.015,clean.shape),0,1)
    missing=(random.random(clean.shape)<.03)|~hit;image[missing]=-1
    return image,dict(dark=dark,bright=bright,shadow=shadow,reflectance=rho,ir_clean=clean,ir_observable=hit&~missing&(image<.995))

def generate(depth,hit,points,normals,projectors,projector_visible,positions,quats,ks,origin,slots,decks,patches,seed,env,episode,capture):
    d=depth.copy();valid=hit&(d>=.15)&(d<5);ir=np.empty_like(d);audit=[];counts={}
    def count(name,mask):counts[name]=counts.get(name,0)+int(np.sum(mask))
    for c in range(len(d)):
        ray=rays(ks[c],quats[c],*d[c].shape)
        image,labels=physical_ir(points[c],normals[c],hit[c],projectors[c],projector_visible[c],rng(seed,env,episode,capture,c,1));ir[c]=image
        gap=np.zeros(d[c].shape,bool);fill=gap.copy();plane=np.full(d[c].shape,np.nan);offset=np.zeros_like(plane)
        for slot,deck,field in zip(slots,decks,patches):
            t=np.divide(origin[2]+deck-positions[c,2],ray[...,2],out=np.full_like(plane,np.nan),where=ray[...,2]<-.1)
            surface=positions[c]+ray*np.nan_to_num(t)[...,None];local=surface-origin
            mask=(local[...,0]>slot[0])&(local[...,0]<slot[1])&(abs(local[...,1])<field.bounds[3])&hit[c]&(points[c,...,2]<origin[2]+deck-.05)&(t>.15)&(t<5)
            selected,dz=field.sample(local[...,:2]);gap|=mask;use=mask&selected
            fill|=use;plane[use]=t[use];offset[use]=dz[use]
        # Optical depth = ray / plane intersection; world z is never used as depth.
        displaced=plane+np.divide(offset,ray[...,2],out=np.zeros_like(offset),where=abs(ray[...,2])>.1)
        d[c][fill]=displaced[fill];valid[c][fill]=True
        hole=gap&(rng(seed,env,episode,capture,c,2).random(plane.shape)<R.deep_hole)
        safe=np.where(hit[c],depth[c],5.)
        edge=(ndimage.maximum_filter(safe,3)-ndimage.minimum_filter(safe,3)>.15)&valid[c]
        fly=edge&(rng(seed,env,episode,capture,c,3).random(plane.shape)<R.flying)
        spike=valid[c]&(rng(seed,env,episode,capture,c,4).random(plane.shape)<R.spike)
        drop=rng(seed,env,episode,capture,c,5).random(plane.shape)<R.dropout
        d[c][fly]=.5*(ndimage.maximum_filter(safe,3)[fly]+ndimage.minimum_filter(safe,3)[fly])
        spike_values=rng(seed,env,episode,capture,c,6).uniform(.15,5,plane.shape);d[c][spike]=spike_values[spike]
        d[c]+=rng(seed,env,episode,capture,c,7).normal(0,R.noise_sigma,plane.shape)
        valid[c]&=~hole&~drop&np.isfinite(d[c])&(d[c]>=.15)&(d[c]<5)
        surviving=fill&valid[c]&(abs((d[c]-plane)*ray[...,2])<=.05)
        labels.update(points=points[c],hit=hit[c],horizontal=np.abs(normals[c,...,2])>.8,gap=gap,injected=fill,surviving=surviving,projected=positions[c]+ray*np.nan_to_num(plane)[...,None])
        audit.append(labels)
        for name,mask in [('gap_pixels',gap),('bright_gap_pixels',gap&labels['bright']),('dark_nongap_pixels',~gap&labels['dark']&hit[c]),('hit_pixels',hit[c]),('observable_gap_pixels',gap&labels['ir_observable']),('injected',fill),('surviving',surviving),('hole',hole),('flying',fly),('spike',spike),('dropout',drop),('ir_missing',image<0),('pixels',np.ones_like(gap))]:count(name,mask)
    d[~valid]=5.
    return d,valid,ir,counts,audit

class IsaacSensor:
    def __init__(self,base,geometry,seed=42):
        self.base=base;self.geometry=geometry;self.seed=seed;self.fields={}
        scanner=base.scene['height_scanner'];self.mesh=scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
    def capture(self,positions,quats,ks,episodes,capture,shape):
        import torch
        from isaaclab.utils.warp import raycast_mesh
        n,c=positions.shape[:2];h,w=shape
        directions=np.array([[rays(ks[e,k],quats[e,k],h,w) for k in range(c)] for e in range(n)])
        lengths=np.linalg.norm(directions,axis=-1);unit=directions/lengths[...,None]
        starts=np.broadcast_to(positions[:,:,None,None,:],unit.shape).copy()
        cast=lambda p,r,normal=False:raycast_mesh(torch.tensor(p.reshape(1,-1,3),device=self.base.device,dtype=torch.float32),torch.tensor(r.reshape(1,-1,3),device=self.base.device,dtype=torch.float32),self.mesh,max_dist=100.,return_distance=True,return_normal=normal)
        pts,dist,normals,_=cast(starts,unit,True)
        points=pts.cpu().numpy().reshape(unit.shape);distance=dist.cpu().numpy().reshape(lengths.shape);normals=normals.cpu().numpy().reshape(unit.shape)
        hit=np.isfinite(distance)&(distance<100)&np.isfinite(points).all(-1)
        safe=np.where(hit[...,None],points,starts+unit*100.)
        projectors=positions+np.einsum('...ij,j->...i',rotation(quats),np.array([.05,0,0]))
        lightstart=np.broadcast_to(projectors[:,:,None,None,:],unit.shape);delta=safe-lightstart;lightdistance=np.linalg.norm(delta,axis=-1)
        _,lightdist,_,_=cast(lightstart,delta/np.maximum(lightdistance[...,None],1e-9));lightdist=lightdist.cpu().numpy().reshape(lengths.shape)
        visible=hit&(lightdist>=lightdistance-.015)
        slots,decks=self.geometry.for_envs(self.base);slots=slots.cpu().numpy();decks=decks.cpu().numpy();origins=self.base.scene.env_origins.cpu().numpy()
        from gd_lab.mdp.terrain_families import terrain_family_gate
        isgap=(terrain_family_gate(self.base,('platform_gap',))==0).cpu().numpy()
        results=[]
        for e in range(n):
            key=tuple(origins[e])+tuple(slots[e].ravel())
            if key not in self.fields:
                width=float(self.base.cfg.scene.terrain.terrain_generator.size[1])/2
                self.fields[key]=[PatchField((a,b,-width,width),np.random.SeedSequence([self.seed,e,j])) for j,(a,b) in enumerate(slots[e])] if isgap[e] else []
            fields=self.fields[key];s=slots[e] if isgap[e] else [];z=decks[e] if isgap[e] else []
            results.append(generate(distance[e]/lengths[e],hit[e],safe[e],normals[e],projectors[e],visible[e],positions[e],quats[e],ks[e],origins[e],s,z,fields,self.seed,e,int(episodes[e]),capture))
        return results
