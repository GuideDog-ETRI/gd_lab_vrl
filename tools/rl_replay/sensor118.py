"""Realistic depth artifacts, independently corrupted aligned luminance IR.
Depth artifact probabilities/geometry retain UPDATE116 Rules. No depth mask
or depth return is used to construct, darken or drop the IR channel.
"""
import numpy as np
from heightmap116 import corrupt as depth_artifacts
from sensor_independent116 import corrupt as independent

def corrupt(depth,valid,proxy,positions,quats,ks,reference_z,rng,noisy,ir_mode):
    assert noisy=='realistic' and ir_mode=='proxy'
    ds,irs=rng.integers(0,2**63,size=2)
    d,v,_,counts=depth_artifacts(depth,valid,proxy,positions,quats,ks,reference_z,np.random.default_rng(ds),True,'proxy')
    _,_,ir,ic=independent(depth,valid,proxy,positions,quats,ks,reference_z,np.random.default_rng(irs),'ir_only','proxy')
    counts.update(pixels=d.size,depth_dropout=counts.pop('dropout'),ir_dropout=ic['ir_dropout'],depth_invalid=int((~v).sum()),ir_invalid=ic['ir_invalid'])
    return d,v,ir,counts
