"""Area-overlap categorical yaw transform, adapted from existing warp.py.

Source 10cm, destination 5/10cm; possibly different grid extents. Integer
monotonic timestamps are gathered, never converted to absolute float32 time.
Free needs all positive-area overlapping source cells free; occupied wins any.
"""
import torch
from .oracle_occgrid176 import UNKNOWN,FREE,OCCUPIED


def resample(state,seen,source,height,capture,profile,old_pose,new_pose,old_grid,new_grid):
    n=state.shape[0];dev=state.device;dtype=height.dtype
    if old_grid.cell_m!=.1 or new_grid.cell_m not in (.05,.1):raise ValueError('terrain 10cm to supervisor 5/10cm')
    r=old_grid.cell_m;half=r/2;newhalf=new_grid.cell_m/2;eps=1e-6
    yy,xx=torch.meshgrid(torch.arange(new_grid.ny,device=dev,dtype=dtype),torch.arange(new_grid.nx,device=dev,dtype=dtype),indexing='ij')
    xyx=new_grid.x_min_m+(xx+.5)*new_grid.cell_m;xyy=new_grid.y_min_m+(yy+.5)*new_grid.cell_m
    # Small FP64 pose ledger; per-cell arithmetic remains FP32.
    angle=(new_pose[:,3]-old_pose[:,3]);ca=angle.cos().to(dtype)[:,None,None];sa=angle.sin().to(dtype)[:,None,None]
    delta=new_pose[:,:2]-old_pose[:,:2];co=old_pose[:,3].cos();si=old_pose[:,3].sin()
    tx=(co*delta[:,0]+si*delta[:,1]).to(dtype)[:,None,None];ty=(-si*delta[:,0]+co*delta[:,1]).to(dtype)[:,None,None]
    x=ca*xyx-sa*xyy+tx;y=sa*xyx+ca*xyy+ty
    ix=torch.floor((x-old_grid.x_min_m)/r).long();iy=torch.floor((y-old_grid.y_min_m)/r).long()
    extent=newhalf*(ca.abs()+sa.abs());bound=half+extent-eps
    bound_new=newhalf+half*(ca.abs()+sa.abs())-eps
    inside=(x-extent>=old_grid.x_min_m-eps)&(x+extent<=old_grid.x_max_m+eps)&(y-extent>=old_grid.y_min_m-eps)&(y+extent<=old_grid.y_max_m+eps)
    free=inside.clone();occupied=torch.zeros_like(free);count=torch.zeros_like(ix)
    big=torch.iinfo(torch.int64).max
    free_seen=torch.full_like(ix,big);occ_seen=torch.full_like(ix,big)
    free_cap=torch.full_like(ix,big);occ_cap=torch.full_like(ix,-1)
    bits=torch.zeros_like(ix,dtype=torch.uint8);occ_bits=torch.zeros_like(bits)
    h=torch.full_like(x,torch.nan);occ_h=h.clone();prof=torch.full_like(ix,-1,dtype=torch.int32);occ_prof=prof.clone()
    h_agrees=inside.clone()
    fields=[v.flatten(1) for v in (state,seen,source,height,capture,profile)]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            cx=ix+dx;cy=iy+dy
            vx=old_grid.x_min_m+(cx+.5)*r-x;vy=old_grid.y_min_m+(cy+.5)*r-y
            overlap=(vx.abs()<bound)&(vy.abs()<bound)&((ca*vx+sa*vy).abs()<bound_new)&((-sa*vx+ca*vy).abs()<bound_new)
            valid=(cx>=0)&(cx<old_grid.nx)&(cy>=0)&(cy<old_grid.ny);real=overlap&valid
            index=(cy.clamp(0,old_grid.ny-1)*old_grid.nx+cx.clamp(0,old_grid.nx-1)).flatten(1)
            s,t,b,z,c,p=[v.gather(1,index).reshape_as(ix) for v in fields]
            occ=real&(s==OCCUPIED);occupied |=occ
            free &= ~(overlap&(~valid|(s!=FREE)))
            free_seen=torch.minimum(free_seen,torch.where(real,t,big));free_cap=torch.minimum(free_cap,torch.where(real,c,big))
            first=real&(count==0);h=torch.where(first,z,h);prof=torch.where(first,p,prof)
            h_agrees &= ~(real&(~torch.isfinite(z)|((z-h).abs()>.020001)))
            prof=torch.where(real&(p!=prof),-2,prof)
            bits |=torch.where(real,b,0)
            choose=occ&(t<occ_seen)
            occ_seen=torch.where(choose,t,occ_seen);occ_h=torch.where(choose,z,occ_h)
            occ_cap=torch.where(choose,c,occ_cap);occ_prof=torch.where(choose,p,occ_prof)
            occ_bits |=torch.where(occ,b,0);count+=real.long()
    free &=count>0
    out=torch.where(occupied,OCCUPIED,torch.where(free,FREE,UNKNOWN)).to(torch.uint8)
    stamp=torch.where(occupied,occ_seen,torch.where(free,free_seen,-1))
    out_h=torch.where(occupied,occ_h,torch.where(free&h_agrees,h,torch.nan))
    cap=torch.where(occupied,occ_cap,torch.where(free,free_cap,-1))
    src=torch.where(occupied,occ_bits,torch.where(free,bits,0))
    tag=torch.where(occupied,occ_prof,torch.where(free,prof,-1))
    return out,stamp,src,out_h,cap,tag
