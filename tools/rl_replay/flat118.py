import math
class FlatPitchGeometry118:
    def __init__(self,base,gap):
        from gd_lab.mdp.gap_stair_v2 import _StairGeometry
        from gd_lab.mdp.platform_gap_terms import boarding_family_mask
        self.base=base;self.gap=gap;self.stair=_StairGeometry(base);self.family=boarding_family_mask
        self.total=0.;self.count=0;self.block_total=[0.]*4;self.block_count=[0]*4
    def sample(self,ids,pitch):
        import torch
        b=self.base;p=b.scene['robot'].data.root_pos_w-b.scene.env_origins
        slots,_=self.gap.for_envs(b)
        distance=(slots.flatten(1)-p[:,0,None]).abs().min(-1).values
        gap_ok=~self.family(b) | (distance>=1.3)
        cols=b.scene.terrain.terrain_types.long();edges=self.stair.edges[cols]
        # Euclidean distance to square pyramid perimeters, including corners.
        xy=p[:,:2].abs();outside=(xy[:,None,:]-edges[:,:,None]).clamp_min(0).norm(dim=-1)
        inside=edges-xy.max(-1).values[:,None]
        sd=torch.where(inside>=0,inside,outside)
        sd=torch.where(torch.isfinite(edges),sd,float('inf')).min(-1).values
        safe=gap_ok & (~self.stair.is_stair[cols] | (sd>=1.3))
        # A scan confirms the edge-distant surface is level, excluding ramps.
        clean=b.observation_manager.compute_group('gast_clean')
        h=clean[:,:187]/5;valid=(clean[:,187:374]>.5)&torch.isfinite(h)
        span=torch.where(valid,h,-float('inf')).max(-1).values-torch.where(valid,h,float('inf')).min(-1).values
        safe &= (valid.float().mean(-1)>.9)&(span<=.02)&torch.isfinite(pitch)
        values=torch.rad2deg(pitch[ids][safe[ids]])
        self.total+=float(values.double().sum());self.count+=len(values)
        for bi in range(4):
            selected=ids[safe[ids] & (ids>=bi*128) & (ids<(bi+1)*128)]
            self.block_total[bi]+=float(torch.rad2deg(pitch[selected]).double().sum());self.block_count[bi]+=len(selected)
    def result(self):
        return {'mean_deg':self.total/self.count if self.count else None,'samples':self.count,
          'definition':'same run all env policy samples; all local gap/stair edges >=1.3m; clean scan >90% valid and span <=.02m',
          'geometry_verified':True,'blocks':[{'mean_deg':v/n if n else None,'samples':n} for v,n in zip(self.block_total,self.block_count)]}
