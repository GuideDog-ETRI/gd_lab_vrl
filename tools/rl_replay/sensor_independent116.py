"""Aligned clean images FIRST, independent additive noise/dropout SECOND.

IR is the same-camera RGB luminance proxy, not a physical active-IR sensor.
No terrain/depth/missing-return labels participate in clean IR construction.
IR invalid pixels use -1 internally; the map ignores them and the student
receives zero there, without changing the checkpoint's input dimensions.
"""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class SensorRules:
    depth_sigma_m: float = .01
    depth_dropout: float = .03
    ir_sigma: float = .015
    ir_dropout: float = .03

RULES = SensorRules()

def corrupt(depth, valid, proxy, positions, quats, ks, reference_z, rng,
            noisy, ir_mode, r=RULES):
    # Signature retains the capture geometry contract; corruption never reads
    # the geometry or reference height. No depth-conditioned dropout probability.
    mode = ('realistic' if noisy else 'clean') if isinstance(noisy, (bool, np.bool_)) else noisy
    if mode not in ('clean', 'depth_only', 'ir_only', 'realistic'):
        raise ValueError(mode)
    if ir_mode not in ('proxy', 'synthetic'):
        raise ValueError(ir_mode)
    depth = np.asarray(depth)
    if depth.shape != valid.shape or depth.shape != proxy.shape:
        raise ValueError('depth/valid/IR must share the canonical pixel grid')
    # Draw all four substream seeds unconditionally, even in clean mode.
    seeds = rng.integers(0, 2**63, size=4)
    dn, dd, inn, ind = [np.random.default_rng(int(s)) for s in seeds]
    d = depth.copy()
    v = np.asarray(valid, dtype=bool).copy() & np.isfinite(d) & (d >= .15) & (d < 5)
    ir_valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 1)
    ir = np.clip(np.nan_to_num(proxy, nan=.5), 0, 1).copy()
    if ir_mode == 'synthetic':
        # Optional fixed exposure transform, NOT inverse-depth-generated IR.
        ir = .1 + .8 * ir
    depth_mask = np.zeros(d.shape, bool)
    ir_mask = np.zeros(d.shape, bool)
    if mode in ('depth_only', 'realistic'):
        d += dn.normal(0, r.depth_sigma_m, d.shape)
        depth_mask = dd.random(d.shape) < r.depth_dropout
        v &= ~depth_mask & np.isfinite(d) & (d >= .15) & (d < 5)
    if mode in ('ir_only', 'realistic'):
        ir = np.clip(ir + inn.normal(0, r.ir_sigma, ir.shape), 0, 1)
        ir_mask = ind.random(ir.shape) < r.ir_dropout
        ir_valid &= ~ir_mask
    d[~v] = 5.
    ir[~ir_valid] = -1.
    counts = dict(pixels=d.size, depth_dropout=int(depth_mask.sum()),
                  ir_dropout=int(ir_mask.sum()),
                  both_dropout=int((depth_mask & ir_mask).sum()),
                  depth_only_dropout=int((depth_mask & ~ir_mask).sum()),
                  ir_only_dropout=int((ir_mask & ~depth_mask).sum()),
                  depth_invalid=int((~v).sum()), ir_invalid=int((~ir_valid).sum()))
    return d, v, ir, counts
