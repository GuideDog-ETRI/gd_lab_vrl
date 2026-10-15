"""B3 adds an independent non-preparation baseline action preservation mean."""
import torch
from gd_lab.gast.b2_objective import WindowLoss as B2Window

class WindowLoss(B2Window):
    def __init__(self,mode='prep4'):
        super().__init__(mode);self.preserve_sum=None;self.preserve_n=0
    def preserve(self,values,labels,valid):
        mask=(labels!=1)&valid;n=int(mask.sum())
        if n:
            value=values[mask].sum()
            self.preserve_sum=value if self.preserve_sum is None else self.preserve_sum+value
            self.preserve_n+=n
    def value(self,zero):
        loss,stats=super().value(zero)
        preserve=self.preserve_sum/self.preserve_n if self.preserve_n else zero
        stats.update(preserve=float(preserve.detach()),preserve_n=self.preserve_n)
        for key,group in [('action',self.action),('raw',self.raw)]:
            stats[key+'_group_mean']={k:float(group.sums[k].detach()/n) if n else None for k,n in group.counts.items()}
        return loss+preserve,stats
