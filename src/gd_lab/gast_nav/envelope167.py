"""Measured command-specific raster envelope; unsupported combinations unknown."""
import torch


def select_envelope(artifacts, mode, value):
    if mode=='NAV' and value<0:return None
    if mode=='TURN' and value>=0:return None
    key=f'{mode}_{abs(value):.1f}'
    if abs(abs(value)-round(abs(value),1))>1e-7:return None
    return artifacts.get(key)


def resample_required(entry, grid, *, device):
    """Conservative cell-overlap resampling without a whole-robot bounding hull.

    Return (mask, contained): clipping any required source cell fails closed.
    """
    source=entry['mask'].to(device);g=entry['grid'];y,x=source.nonzero(as_tuple=True)
    centres=torch.stack((g['x0']+x*g['res'],g['y0']+y*g['res']),-1)
    target=grid.xy(device=device,dtype=torch.float32).flatten(0,1)
    required=torch.zeros(len(target),device=device,dtype=torch.bool)
    reach=(g['res']+grid.res)/2
    for first in range(0,len(centres),128):
        required |= ((target[:,None]-centres[None,first:first+128]).abs()<=reach+1e-7).all(-1).any(-1)
    lower=centres-g['res']/2;upper=centres+g['res']/2
    contained=((lower[:,0]>=grid.x0-grid.res/2-1e-7)&(upper[:,0]<=grid.x0+(grid.nx-.5)*grid.res+1e-7)&
               (lower[:,1]>=grid.y0-grid.res/2-1e-7)&(upper[:,1]<=grid.y0+(grid.ny-.5)*grid.res+1e-7)).all()
    return required.reshape(grid.ny,grid.nx),contained
