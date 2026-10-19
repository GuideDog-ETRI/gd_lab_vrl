"""Replace only reset-episode rows after simulated entry captures."""
import torch


def copy_seeded_rows183(destination,seeded,rows):
 if rows.dtype!=torch.bool or rows.shape!=(destination.n,) or destination.n!=seeded.n:raise ValueError('entry row mask')
 if destination.cfg!=seeded.cfg or destination.world_cfg!=seeded.world_cfg:raise ValueError('same memory grids required')
 profile=seeded.profile.clone()
 for i,value in enumerate(seeded.profiles):
  if value not in destination.profiles:
   if len(destination.profiles)>=64:raise ValueError('profile registry full')
   destination.profiles.append(value)
  profile[seeded.profile==i]=destination.profiles.index(value)
 for name in ('state','seen','source','height','cap','foot_z','foot_seen','anchor','epoch','last_ms','sequence'):
  getattr(destination,name)[rows]=getattr(seeded,name)[rows]
 destination.profile[rows]=profile[rows]
