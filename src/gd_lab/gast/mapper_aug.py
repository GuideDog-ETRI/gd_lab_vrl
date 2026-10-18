"""UPDATE149 actor-only mapper-error augmentation; no simulator imports.

Randomness has a private generator so enabling this term never consumes the
teacher's normal noise/dropout RNG. Parameters persist per episode. Extra GT
queries are augmentation inputs only; no labels/coordinates are returned.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MapperAugCfg:
    enabled: bool = False
    ramp_steps: int = 50000  # 500 updates at the approved 100-step horizon
    clean_fraction: float = .25
    seed: int = 14942


class MapperAugmentation:
    def __init__(self, n, device, cfg=MapperAugCfg()):
        if cfg.ramp_steps <= 0 or not 0 <= cfg.clean_fraction <= 1:
            raise ValueError('invalid mapper augmentation ramp/clean fraction')
        self.cfg = cfg
        self.rng = torch.Generator(device=device).manual_seed(cfg.seed)
        self.params = torch.zeros(n, 14, device=device)
        self.reset()
        self.last_diagnostics = {}
        self._active = torch.zeros(n, dtype=torch.bool, device=device)
        self.removed = torch.zeros(n, 187, dtype=torch.bool, device=device)
        self.widened = self.removed.clone()

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.params[ids] = torch.rand(self.params[ids].shape, device=self.params.device, generator=self.rng)

    def geometry(self, points, scanner_z, support_z, yaw, gap_family, step, cast):
        """Return changed metric z BEFORE the existing noise/clip/dropout chain."""
        z = points[..., 2]
        strength = min(1., max(0., float(step) / self.cfg.ramp_steps))
        self.removed.zero_()
        self.widened = torch.zeros_like(z, dtype=torch.bool)
        self.last_diagnostics = {"strength": strength, "active_episodes": 0, "new_unknown": 0, "height_changes": 0}
        if not self.cfg.enabled or strength == 0:
            self._active.zero_()
            return z
        p = self.params
        active = (p[:, 0] >= self.cfg.clean_fraction) & (p[:, 1] < strength)
        self._active = active
        # One synchronization at the existing camera refresh, not every policy step.
        if not active.any():
            self._active.zero_()
            return z
        z = points[..., 2]
        finite = torch.isfinite(points).all(-1)
        # S0 reads actual GT at the four diagonal offsets, not a max-pool of a
        # 10 cm grid (which cannot represent a 2.5 cm footprint).
        offsets = torch.tensor([[-.025, -.025], [-.025, .025], [.025, -.025], [.025, .025]], device=z.device)
        base = offsets[None].expand(len(z), -1, -1)
        extra = base.clone()
        extra[:, :2, 0] -= .075 * strength * p[:, 2:3]
        extra[:, 2:, 0] += .075 * strength * p[:, 3:4]
        extra[:, :, 0] += (.025 * (p[:, 4:5] - .5)) * strength
        extra[..., 0].clamp_(-.10, .10)  # total S0+extra+phase <=10 cm, no stress tail
        extra *= (.10 / extra.norm(dim=-1, keepdim=True).clamp_min(.10))
        offsets = torch.cat((base, extra), 1)
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        world = torch.stack((c * offsets[..., 0] - s * offsets[..., 1],
                             s * offsets[..., 0] + c * offsets[..., 1]), -1)
        starts = points[:, :, None, :].nan_to_num(0.).expand(-1, -1, 8, -1).clone()
        starts[..., :2] += world[:, None]
        starts[..., 2] = scanner_z[:, None, None] + 2.
        directions = torch.zeros_like(starts); directions[..., 2] = -1
        distance = cast(starts.reshape(-1, 3), directions.reshape(-1, 3), 10.).reshape(len(z), -1, 8)
        hits = starts[..., 2] - distance
        hits = torch.where(torch.isfinite(distance) & (distance < 10.), hits, torch.inf)
        footprint_z = torch.minimum(z, hits[..., :4].amin(-1))
        # Only the extra dilation/DDEEP are pit-specific. Ordinary stairs still
        # receive S0 footprint and unknown corruption, but no fabricated pit floor.
        extra_pit = gap_family[:, None, None] & (hits[..., 4:] < support_z[:, None, None] - .3)
        dilated = torch.where(extra_pit, hits[..., 4:], torch.inf).amin(-1)
        aug_z = torch.minimum(footprint_z, dilated)
        pit = gap_family[:, None] & (aug_z < support_z[:, None] - .3)
        aug_z = torch.where(pit & (p[:, 5:6] < .5), support_z[:, None] - .7, aug_z)
        apply = active[:, None] & finite
        changed = apply & ((aug_z - z).abs() > 1e-6)
        self.widened = apply & (z >= support_z[:, None]-.3) & pit
        self.last_diagnostics.update(active_episodes=active.sum().detach(),
            height_changes=changed.sum().detach(),
            widened_cells=self.widened.sum().detach(),
            query_radius_max_m=offsets.norm(dim=-1).amax().detach(),
            outward_query_max_m=offsets[..., 0].abs().amax().detach(),
            near_query_mean_m=(-extra[:, :2, 0]).mean().detach(),
            far_query_mean_m=extra[:, 2:, 0].mean().detach())
        return torch.where(apply, aug_z, z)

    def unknown(self, height, valid, ir, points, yaw):
        p = self.params
        strength = self.last_diagnostics.get("strength", 0.)
        if not self.cfg.enabled or strength == 0:
            return height, valid, ir
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        finite = torch.isfinite(points).all(-1)
        # Episode-fixed egocentric ellipse and one map-border band; do not invent
        # visibility outside the existing four-camera mask.
        delta = points[..., :2] - points[:, 93:94, :2]  # grid center
        x = delta[..., 0] * c + delta[..., 1] * s
        y = -delta[..., 0] * s + delta[..., 1] * c
        cx = (p[:, 6:7] * 1.6 - .8); cy = (p[:, 7:8] - .5)
        rx = (.05 + .15 * p[:, 8:9]) * strength
        ry = (.04 + .10 * p[:, 9:10]) * strength
        blob = ((x-cx)/rx.clamp_min(1e-6)).square() + ((y-cy)/ry.clamp_min(1e-6)).square() < 1
        side = (p[:, 10:11] * 4).long(); width = .15 * p[:, 11:12] * strength
        border = ((side == 0) & (x < -.8+width)) | ((side == 1) & (x > .8-width)) | ((side == 2) & (y < -.5+width)) | ((side == 3) & (y > .5-width))
        holes = (blob & (p[:, 12:13] < .6)) | (border & (p[:, 13:14] < .5))
        candidates = holes & self._active[:, None] & finite & valid
        # Deterministic per-frame truncation in grid order; never remove >15%
        # of the CURRENTLY valid cells. A small footprint may remove fewer than5%.
        cap = torch.floor(valid.sum(-1) * (.05 + .10*p[:, 4]) * strength).long()
        removed = candidates & (candidates.long().cumsum(-1) <= cap[:, None])
        self.removed = removed
        v = valid & ~removed
        fraction = removed.sum(-1) / valid.sum(-1).clamp_min(1)
        self.last_diagnostics.update(new_unknown=removed.sum().detach(),
            widened_visible_cells=(self.widened & valid).sum().detach(),
            widened_output_cells=(self.widened & v).sum().detach(),
            extra_unknown_fraction_mean=fraction.mean().detach(),
            extra_unknown_fraction_max=fraction.max().detach())
        return torch.where(v, height, 0.), v, torch.where(v, ir, 0.)

    def __call__(self, height, valid, ir, points, scanner_z, support_z, yaw, gap_family, step, cast):
        # Synthetic fixtures only. Live integration uses geometry before noise.
        z = self.geometry(points, scanner_z, support_z, yaw, gap_family, step, cast)
        if not self.cfg.enabled or self.last_diagnostics['strength'] == 0:
            return height, valid, ir
        h = ((scanner_z[:, None] - z - .5).clamp(-1, 1)*5)
        h = torch.where(self._active[:, None], h, height)
        return self.unknown(h, valid, ir, points, yaw)
