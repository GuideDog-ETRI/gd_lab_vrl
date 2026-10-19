"""Observed gap runs along current translational stop axis; no invented support."""
import torch

def walkable_gap177b(gap,support,seen,*,cell_m,max_width_m=.26,enabled=None):
    """Current-heading grid: NAV stop axis +x. Every run needs BOTH observed banks.

    Occupied/unknown cells are not gap observations. Full grid-cell extents give
    a conservative width (2x10cm passes; 3x10cm fails). Return original oldest
    run/bank time; this is a permission sidecar, never a support height.
    """
    n,ny,nx=gap.shape
    if support.shape!=gap.shape or seen.shape!=gap.shape:raise ValueError('gap grid')
    x=torch.arange(nx,device=gap.device).expand(n,ny,-1)
    left=torch.where(~gap,x,-1).cummax(-1).values
    right=torch.where(~gap,x,nx).flip(-1).cummin(-1).values.flip(-1)
    li=left.clamp(0,nx-1);ri=right.clamp(0,nx-1)
    ok=gap&(left>=0)&(right<nx)&support.gather(-1,li)&support.gather(-1,ri)
    ok &= (right-left-1)*cell_m<=max_width_m+1e-7
    oldest=torch.minimum(seen.gather(-1,li),seen.gather(-1,ri))
    for offset in range(1,int(max_width_m/cell_m)+2):
        idx=(left+offset).clamp(0,nx-1)
        oldest=torch.minimum(oldest,torch.where(left+offset<right,seen.gather(-1,idx),oldest))
    ok &= oldest>=0
    if enabled is not None:ok &= enabled[:,None,None]
    return ok,torch.where(ok,oldest,-1)
