"""Read-only v25or post-sample tap. Never calls sample/overlay/expire or changes actor data."""
from dataclasses import dataclass
import numpy as np
import torch

@dataclass(frozen=True)
class TerrainFrame177:
    surface_z: torch.Tensor   # [N,11,17], metres relative to sample base z; NOT normalized h
    valid: torch.Tensor       # native output validity/provenance, not a new free certificate
    gap: torch.Tensor         # explicit structure130 source4, synthetic depth is NOT support
    pending: torch.Tensor     # source5 / unresolved native output
    seen_ms: torch.Tensor     # original per-cell observation time; -1 when unavailable
    source: torch.Tensor      # native codes0 unknown,1 direct,4 GAP,5 pending
    pose: torch.Tensor        # [N,4] world x,y,z,yaw of THIS sampled grid, FP64
    epoch: torch.Tensor       # [N] actual mapper epoch supplied by caller
    sample_ms: torch.Tensor   # read/sample chart time, NEVER observation time
    last_capture_ms: torch.Tensor # actual latest mapper input, NEVER sample_ms


def _ms(v):
    return int(np.floor(v*1000+1e-6)) if np.isfinite(v) and v>=0 else -1


def tap_v25or177(maps,packed,pose_xyzyaw,scanner_z,*,sample_ms,epoch,structure_rules,device='cpu'):
    """Call IMMEDIATELY after actor's existing maps.sample(poses,scanner,stamp).

    Reads the exact already-produced packed array + last_sources. Reconstructs
    ONLY selected donor/provenance from h0d's centre + four footprint samples.
    Does not integrate, resample the mapper, reset, expire or mutate any arrays.
    Scanner z is the SAME value used by maps.sample, NOT a GT floor query.
    Strict L0S/h0d/dual-OR/DD=.7 contract; any numeric mismatch fails closed.
    """
    if maps.arm!='L0S' or maps.direct.mode!='h0d' or not structure_rules.get('dual_grid') or not structure_rules.get('dual_or') or abs(structure_rules['d_deep']-.7)>1e-8:raise ValueError('expected existing v25or L0S/h0d dual-OR DD=.7')
    packed=np.asarray(packed);pose=np.asarray(pose_xyzyaw,dtype=np.float64);scanner=np.asarray(scanner_z,dtype=np.float64)
    n=pose.shape[0];times=np.asarray(sample_ms);epochs=np.asarray(epoch)
    if pose.shape!=(n,4) or packed.shape!=(n,374) or scanner.shape!=(n,) or times.shape!=(n,) or epochs.shape!=(n,) or not np.isfinite(pose).all() or not np.isfinite(scanner).all():raise ValueError('tap shapes/finite pose')
    if times.dtype!=np.int64 or epochs.dtype!=np.int64 or (times<0).any() or (epochs<0).any():raise ValueError('time/epoch int64')
    source=np.asarray(maps.last_sources).copy()
    if source.shape!=(n,187) or not np.isin(source,[0,1,4,5]).all():raise ValueError('not a camera v25or source map')
    z=np.full((n,187),np.nan);seen=np.full((n,187),-1,np.int64);valid=np.zeros((n,187),bool)
    gaps=(source==4)&(packed[:,187:]>0);pending=source==5
    captures=np.full(n,-1,np.int64)
    for e,m in enumerate(maps.direct.maps):
        captures[e]=_ms(m.last_capture)
        if m.origin is None:continue
        r=m.r;c,s=np.cos(pose[e,3]),np.sin(pose[e,3]);x=maps.direct.x;y=maps.direct.y
        donor_z=np.full(187,np.nan);donor_t=np.full(187,-1,np.int64);selected=np.zeros(187,bool)
        for dx,dy in ((0,0),(-.025,-.025),(-.025,.025),(.025,-.025),(.025,.025)):
            ox=x+dx;oy=y+dy
            xy=np.c_[pose[e,0]+c*ox-s*oy,pose[e,1]+s*ox+c*oy]
            ij=np.floor(xy/r.resolution+1e-7).astype(np.int64)-m.origin
            inside=(ij>=0).all(1)&(ij<r.size).all(1)
            i=ij[:,0].clip(0,r.size-1);j=ij[:,1].clip(0,r.size-1)
            zz=m.z[j,i];last=m.last[j,i];finite=m.finite[j,i]
            now=times[e]/1000
            ok=inside&np.isfinite(zz)&(m.var[j,i]<.0025)&(now-last>=-1e-9)&(now-last<=r.ttl)
            better=ok&(~selected|(zz<donor_z))
            # Older of height-update and finite-hit provenance is conservative.
            tt=np.minimum(last,finite)
            stamp=np.array([_ms(t) for t in tt],dtype=np.int64)
            donor_z=np.where(better,zz,donor_z);donor_t=np.where(better,stamp,donor_t);selected|=ok
        encoded=5*np.clip(scanner[e]-donor_z-.5,-1,1)
        direct=(source[e]==1)&(packed[e,187:]>0)&selected&np.isfinite(encoded)&(np.abs(encoded-packed[e,:187])<2e-5)&(donor_t>=0)
        z[e,direct]=donor_z[direct];seen[e,direct]=donor_t[direct];valid[e,direct]=True
        mem=maps.struct[e]
        xy=np.c_[pose[e,0]+c*x-s*y,pose[e,1]+s*x+c*y]
        k0=np.floor(xy/mem.res+1e-7).astype(np.int64)
        k1=np.floor((xy-structure_rules['dual_offset'])/mem.res+1e-7).astype(np.int64)
        for q in np.flatnonzero(gaps[e]):
            # Match overlay's exact winning entry (grid0 if present, otherwise grid1).
            g=mem.gap.get(tuple(k0[q])) or mem.gap2.get(tuple(k1[q]))
            if g is None:continue
            synthetic=g[0]-structure_rules['d_deep'];h=5*np.clip(scanner[e]-synthetic-.5,-1,1)
            t=_ms(g[3])
            if abs(h-packed[e,q])<2e-5 and 0<=t<=times[e]:
                z[e,q]=synthetic;seen[e,q]=t;valid[e,q]=True
        for q in np.flatnonzero(pending[e]):
            seen[e,q]=_ms(mem.pending.get(tuple(k0[q]),-np.inf))
    # Invalid final heights, even if finite in packed, are never permission.
    valid &=np.isfinite(z)&(seen>=0)&(seen<=times[:,None])
    shape=(n,11,17)
    def tensor(a,dtype=None):return torch.as_tensor(np.array(a,copy=True),device=device,dtype=dtype).clone()
    return TerrainFrame177(tensor(z-pose[:,2,None],torch.float32).reshape(shape),tensor(valid).reshape(shape),tensor(gaps).reshape(shape),tensor(pending).reshape(shape),tensor(seen).reshape(shape),tensor(source,torch.uint8).reshape(shape),tensor(pose),tensor(epochs),tensor(times),tensor(captures))
