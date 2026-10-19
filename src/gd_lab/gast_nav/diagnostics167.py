"""Required-cell attribution without inferring provenance from UNKNOWN alone.

Masks overlap. Missing optional instrumentation is reported unavailable, never
as zero. A self-return flag by itself is insufficient: depth clipping shares
that flag, and true blockage requires a cell-local ray/collision attribution.
"""
import torch


def required_causes(required, state, support_known, *, self_blocked=None,
                    warp_hole=None, ttl_expired=None, reference_conflict=None):
    observed=support_known & (state[:,:2]!=0).all(1)
    missing=required & ~observed
    masks={'support_missing':missing & ~support_known,
           'B1_subbin_incomplete':missing & support_known & (state[:,0]==0),
           'B2_subbin_incomplete':missing & support_known & (state[:,1]==0)}
    for name,value in dict(self_blocked=self_blocked,warp_hole=warp_hole,
                           ttl_expired=ttl_expired,reference_conflict=reference_conflict).items():
        masks[name]=None if value is None else missing & value
    return missing,masks
