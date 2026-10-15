"""UPDATE136 B: camera-side depth outlier rejection before the shared mapper (deployable on the real D4xx stream).

Per camera image, single pass on the unfiltered frame, 3x3 neighbourhood excluding the centre, valid pixels only:
  m = median of valid neighbour optical depths, tau = max(0.10 m, 0.05 m),
  c = number of valid neighbours within tau of the centre depth.
  keep if n >= 3 and (|d - m| <= tau or c >= 2)      (c >= 2 keeps 1-px true edges / thin bands)
  keep if n == 2 and c == 2;  reject if n < 2.
Rejected pixels become invalid (unknown); nothing is filled. No labels, no RNG, no future frames.
Thresholds fixed a priori (Codex90 UPDATE136 review), not tuned on rollout results.
"""
import numpy as np

TAU_ABS, TAU_REL = .10, .05


def reject_outliers(depth, valid):
    """depth, valid: (..., H, W). Returns the kept mask (subset of valid)."""
    d = np.where(valid, depth, np.nan)
    pad = np.pad(d, [(0, 0)] * (d.ndim - 2) + [(1, 1), (1, 1)], constant_values=np.nan)
    h, w = d.shape[-2:]
    nb = np.stack([pad[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    ok = np.isfinite(nb)
    n = ok.sum(0)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            m = np.nanmedian(nb, axis=0)
        tau = np.maximum(TAU_ABS, TAU_REL * np.nan_to_num(m, nan=0.))
        c = (ok & (np.abs(nb - d) <= tau)).sum(0)
        near = np.abs(d - m) <= tau
    keep = ((n >= 3) & (near | (c >= 2))) | ((n == 2) & (c == 2))
    return valid & keep


if __name__ == '__main__':
    # Fixtures: flat floor rows at 1.0-2.0 m, deep bands 1/2/3 px wide (+0.6 m), a slanted edge, isolated spikes.
    rng = np.random.default_rng(0)
    H, W = 90, 160
    base = np.linspace(1., 2., H)[:, None].repeat(W, 1) + rng.normal(0, .01, (H, W))
    report = {}
    for width in (1, 2, 3):
        d = base.copy(); band = np.zeros((H, W), bool); band[:, 80:80 + width] = True; d[band] += .6
        k = reject_outliers(d, np.ones((H, W), bool))
        report[f'band{width}px_kept'] = k[band].mean(); report[f'band{width}px_floor_kept'] = k[~band].mean()
    d = base.copy(); yy, xx = np.mgrid[:H, :W]; edge = xx > 40 + yy; d[edge] += .6
    k = reject_outliers(d, np.ones((H, W), bool)); report['slant_kept'] = k.mean()
    d = base.copy(); spike = rng.random((H, W)) < .006; d[spike] = rng.uniform(.15, 5, spike.sum())
    far = spike & (np.abs(d - base) > .1)
    k = reject_outliers(d, np.ones((H, W), bool))
    report['spike_far_rejected'] = 1 - k[far].mean(); report['floor_kept'] = k[~spike].mean()
    v = rng.random((H, W)) > .3; k = reject_outliers(base, v); report['sparse70_floor_kept'] = k[v].mean()
    for key, val in report.items(): print(f'{key:22s} {val:.4f}')
