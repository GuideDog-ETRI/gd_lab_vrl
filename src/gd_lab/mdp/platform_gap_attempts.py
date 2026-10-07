"""Gap-attempt state machine for the gap fine-tuning arms (pure torch).

No simulator imports: every input is an explicit tensor, so the logic is
CPU-testable. Frames: ``x`` is relative to the tile centre (== env origin),
``z`` is world height. A foot position is the collision-sphere centre.

An *attempt* starts when the leading foot is within ``APPROACH_DISTANCE`` of the
near edge of the slot in front of the robot. The slot and direction are fixed at
the start. It ends, in priority order, with (1) the original crossing event
(``clean`` if no foot of that slot ever reached ``CLEAN_DEPTH`` below the lower
deck, else ``nonclean``), (2) retreat beyond ``RETREAT_DISTANCE``, (3) heading
reversal, or (4) the end of the episode (``close_episode``). A closed attempt is
never restarted in the same step. The clean bonus is paid at most once per
direction and episode, but every attempt is counted for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gd_lab.mdp.platform_gap_math import gap_intrusion_cost

FOOT_RADIUS = 0.03
CLEAN_DEPTH = 0.03
APPROACH_DISTANCE = 0.30
START_TOLERANCE = 0.05
RETREAT_DISTANCE = 0.40
CONTACT_FORCE = 10.0
MAX_RECORDS = 64

CLEAN, NONCLEAN, RETREAT, REVERSE, EPISODE_END = 1, 2, 3, 4, 5
OUTCOME_NAMES = {CLEAN: "clean", NONCLEAN: "nonclean", RETREAT: "retreat", REVERSE: "reverse", EPISODE_END: "episode_end"}


@dataclass
class StepResult:
    cost: torch.Tensor  # [N] intrusion cost in [0, 1], zero outside gap tiles
    clean_event: torch.Tensor  # [N] bool, one-shot clean-crossing bonus
    started: torch.Tensor  # [N] bool
    closed: torch.Tensor  # [N] bool


class GapAttemptTracker:
    def __init__(self, num_envs: int, device, strict_contact: bool = False):
        # strict_contact (v2): a foot touching anything over the held slot (+- radius) makes the attempt non-clean.
        self.strict_contact = bool(strict_contact)
        self.n, self.device = num_envs, torch.device(device)
        zeros = lambda *shape, dtype=torch.long: torch.zeros(*shape, dtype=dtype, device=self.device)  # noqa: E731
        self.active = zeros(num_envs, dtype=torch.bool)
        self.slot, self.direction = zeros(num_envs), zeros(num_envs)
        self.clean = zeros(num_envs, dtype=torch.bool)
        self.paid = zeros(num_envs, 2, dtype=torch.bool)
        self.att_depth, self.att_width, self.att_deck = (zeros(num_envs, dtype=torch.float32) for _ in range(3))
        self.att_dwell, self.att_contact, self.att_steps = zeros(num_envs), zeros(num_envs), zeros(num_envs)
        # episode counters: outcomes[:, 0] = started, [:, CLEAN..EPISODE_END] = closed by outcome
        self.outcomes = zeros(num_envs, 6)
        self.family_steps, self.dwell_steps, self.contact_steps, self.deep_steps = (zeros(num_envs) for _ in range(4))
        self.cost_sum = zeros(num_envs, dtype=torch.float32)
        # raw per-attempt records (fixed capacity, overflow counted in rec_count)
        self.rec_count = zeros(num_envs)
        self.rec_outcome, self.rec_dwell, self.rec_contact, self.rec_steps, self.rec_slot, self.rec_dir = (
            zeros(num_envs, MAX_RECORDS) for _ in range(6)
        )
        self.rec_depth, self.rec_width, self.rec_deck = (zeros(num_envs, MAX_RECORDS, dtype=torch.float32) for _ in range(3))

    # -- lifecycle -----------------------------------------------------------------
    def _ids(self, env_ids):
        everyone = torch.arange(self.n, device=self.device)
        if env_ids is None:
            return everyone
        if isinstance(env_ids, slice):
            return everyone[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        for tensor in (
            self.active, self.slot, self.direction, self.clean, self.paid, self.att_depth, self.att_width,
            self.att_deck, self.att_dwell, self.att_contact, self.att_steps, self.outcomes, self.family_steps,
            self.dwell_steps, self.contact_steps, self.deep_steps, self.cost_sum, self.rec_count, self.rec_outcome,
            self.rec_dwell, self.rec_contact, self.rec_steps, self.rec_slot, self.rec_dir, self.rec_depth,
            self.rec_width, self.rec_deck,
        ):
            tensor[ids] = 0

    def _close(self, closing: torch.Tensor, outcome: torch.Tensor):
        rows = torch.arange(self.n, device=self.device)
        position = self.rec_count.clamp(max=MAX_RECORDS - 1)
        write = closing & (self.rec_count < MAX_RECORDS)
        for buffer, value in (
            (self.rec_outcome, outcome), (self.rec_depth, self.att_depth), (self.rec_dwell, self.att_dwell),
            (self.rec_contact, self.att_contact), (self.rec_steps, self.att_steps), (self.rec_slot, self.slot),
            (self.rec_dir, self.direction), (self.rec_width, self.att_width), (self.rec_deck, self.att_deck),
        ):
            buffer[rows, position] = torch.where(write, value.to(buffer.dtype), buffer[rows, position])
        self.rec_count += closing.long()
        self.outcomes.scatter_add_(1, outcome[:, None], closing[:, None].long())
        self.active = self.active & ~closing
        for counter in (self.att_depth, self.att_dwell, self.att_contact, self.att_steps):
            counter[closing] = 0

    def close_episode(self, env_ids):
        """Fail every attempt still open in ``env_ids`` as ``episode_end`` (call before reset)."""
        mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        mask[self._ids(env_ids)] = True
        closing = self.active & mask
        self._close(closing, torch.full((self.n,), EPISODE_END, dtype=torch.long, device=self.device))

    # -- per-step update -----------------------------------------------------------
    def update(self, foot_x, foot_z, slots, lower_deck_z, forward, family, crossing_event, contact) -> StepResult:
        """foot_x/foot_z/contact [N,4]; slots [N,2,2] (left,right x-range); lower_deck_z [N,2] world z;
        forward/family [N] bool (heading +x, gap tile); crossing_event [N,2] bool (left,right)."""
        rows = torch.arange(self.n, device=self.device)
        in_slot = (foot_x[..., None] > slots[:, None, :, 0]) & (foot_x[..., None] < slots[:, None, :, 1])
        in_slot = in_slot & family[:, None, None]
        foot_z3 = foot_z[..., None].expand_as(in_slot)
        deck3 = lower_deck_z[:, None, :].expand_as(in_slot)
        cost = gap_intrusion_cost(foot_z3, in_slot, deck3)
        depth = (deck3 - (foot_z3 - FOOT_RADIUS)).clamp_min(0.0)
        deep = in_slot & (depth >= CLEAN_DEPTH)

        # episode-level diagnostics (independent of attempts)
        self.family_steps += family.long()
        self.dwell_steps += in_slot.any(dim=(1, 2)).long()
        self.contact_steps += (in_slot.any(dim=2) & contact).any(dim=1).long()
        self.deep_steps += deep.any(dim=(1, 2)).long()
        self.cost_sum += cost

        # 1) start attempts on envs that were idle at the beginning of the step
        near = torch.where(forward, slots[:, 1, 0], slots[:, 0, 1])
        far = torch.where(forward, slots[:, 1, 1], slots[:, 0, 0])
        lead = torch.where(forward, foot_x.max(dim=1).values, foot_x.min(dim=1).values)
        distance = torch.where(forward, near - lead, lead - near)
        before_far = torch.where(forward, lead < far, lead > far)
        start = family & ~self.active & (distance <= APPROACH_DISTANCE) & (distance >= -START_TOLERANCE) & before_far
        wanted = forward.long()
        self.slot = torch.where(start, wanted, self.slot)
        self.direction = torch.where(start, torch.where(forward, 1, -1), self.direction)
        self.clean = torch.where(start, torch.ones_like(self.clean), self.clean)
        self.att_width = torch.where(start, slots[rows, wanted, 1] - slots[rows, wanted, 0], self.att_width)
        self.att_deck = torch.where(start, lower_deck_z[rows, wanted], self.att_deck)
        for counter in (self.att_depth, self.att_dwell, self.att_contact, self.att_steps):
            counter[start] = 0
        self.outcomes[:, 0] += start.long()
        self.active = self.active | start

        # 2) update the held attempts with this step's geometry
        held_index = self.slot[:, None, None].expand(-1, foot_x.shape[1], 1)
        held_in = in_slot.gather(2, held_index).squeeze(2)
        held_depth = (depth.gather(2, held_index).squeeze(2) * held_in).amax(dim=1)
        violation = deep.gather(2, held_index).squeeze(2).any(dim=1)
        self.att_steps += self.active.long()
        self.att_dwell += (self.active & held_in.any(dim=1)).long()
        self.att_contact += (self.active & (held_in & contact).any(dim=1)).long()
        self.att_depth = torch.where(self.active, torch.maximum(self.att_depth, held_depth), self.att_depth)
        if self.strict_contact:
            # v2.1: same boundary as the per-foot clean / slot terms: slot +- (foot radius - 5 mm)
            edge = FOOT_RADIUS - 0.005
            held = slots[torch.arange(self.n, device=self.device), self.slot]  # [N, 2]
            near_slot = (foot_x > held[:, None, 0] - edge) & (foot_x < held[:, None, 1] + edge)
            violation = violation | (near_slot & contact & family[:, None]).any(dim=1)
        self.clean = self.clean & ~(self.active & violation)

        # 3) close attempts: success > retreat > reversal
        held = slots[rows, self.slot]  # [N, 2] (lo, hi) of the held slot
        forward_held = self.direction > 0
        near_held = torch.where(forward_held, held[:, 0], held[:, 1])
        lead_held = torch.where(forward_held, foot_x.max(dim=1).values, foot_x.min(dim=1).values)
        retreated = torch.where(
            forward_held, lead_held < near_held - RETREAT_DISTANCE, lead_held > near_held + RETREAT_DISTANCE
        )
        success = self.active & crossing_event.gather(1, self.slot[:, None]).squeeze(1)
        fail_retreat = self.active & ~success & retreated
        fail_reverse = self.active & ~success & ~retreated & (forward != forward_held)
        closing = success | fail_retreat | fail_reverse
        outcome = torch.zeros(self.n, dtype=torch.long, device=self.device)
        outcome = torch.where(success, torch.where(self.clean, CLEAN, NONCLEAN), outcome)
        outcome = torch.where(fail_retreat, RETREAT, outcome)
        outcome = torch.where(fail_reverse, REVERSE, outcome)

        already_paid = self.paid.gather(1, self.slot[:, None]).squeeze(1)
        bonus = success & self.clean & ~already_paid
        self.paid = self.paid | (bonus[:, None] & (torch.arange(2, device=self.device)[None, :] == self.slot[:, None]))
        self._close(closing, outcome)
        return StepResult(cost=cost, clean_event=bonus, started=start, closed=closing)

    # -- reporting -----------------------------------------------------------------
    def snapshot(self, env_ids):
        """Additive sums and raw records for the gap envs among ``env_ids`` (call after close_episode).

        Sums can be added across resets and envs and turned into rates with ``summarize``.
        Returns None when none of the envs spent a step on a gap tile.
        """
        ids = self._ids(env_ids)
        ids = ids[self.family_steps[ids] > 0]
        if ids.numel() == 0:
            return None
        outcomes = self.outcomes[ids]
        sums = {
            "episodes": int(ids.numel()),
            "started": int(outcomes[:, 0].sum()),
            "closed": [int(v) for v in outcomes[:, 1:].sum(dim=0)],  # clean, nonclean, retreat, reverse, episode_end
            "family_steps": int(self.family_steps[ids].sum()),
            "dwell_steps": int(self.dwell_steps[ids].sum()),
            "contact_steps": int(self.contact_steps[ids].sum()),
            "deep_steps": int(self.deep_steps[ids].sum()),
            "cost_sum": float(self.cost_sum[ids].sum()),
            "overflow": int((self.rec_count[ids] - MAX_RECORDS).clamp_min(0).sum()),
        }
        count = self.rec_count[ids].clamp(max=MAX_RECORDS).tolist()
        fields = {
            name: getattr(self, "rec_" + name)[ids].cpu()
            for name in ("outcome", "depth", "dwell", "contact", "steps", "slot", "dir", "width", "deck")
        }
        records = []
        for local, env_id in enumerate(ids.tolist()):
            for k in range(count[local]):
                records.append({
                    "env": env_id,
                    "outcome": OUTCOME_NAMES[int(fields["outcome"][local, k])],
                    "max_depth": float(fields["depth"][local, k]),
                    "slot_steps": int(fields["dwell"][local, k]),
                    "contact_steps": int(fields["contact"][local, k]),
                    "attempt_steps": int(fields["steps"][local, k]),
                    "slot": int(fields["slot"][local, k]),
                    "direction": int(fields["dir"][local, k]),
                    "width": float(fields["width"][local, k]),
                    "lower_deck": float(fields["deck"][local, k]),
                })
        return sums, records

    @staticmethod
    def merge(total, sums):
        if total is None:
            return {**sums, "closed": list(sums["closed"])}
        merged = {key: total[key] + sums[key] for key in total if key != "closed"}
        merged["closed"] = [a + b for a, b in zip(total["closed"], sums["closed"], strict=True)]
        return merged

    @staticmethod
    def summarize(sums, records):
        """Scalar stats (python floats) from additive sums; depth percentiles come from the raw records."""
        if sums is None:
            return {}
        started, steps, closed = float(sums["started"]), float(sums["family_steps"]), sums["closed"]
        rate = lambda value: float(value) / max(started, 1.0)  # noqa: E731
        stats = {
            "gap_episodes": float(sums["episodes"]),
            "attempts": started,
            "attempts_per_episode": started / sums["episodes"],
            "clean_rate": rate(closed[CLEAN - 1]),
            "nonclean_rate": rate(closed[NONCLEAN - 1]),
            "fail_retreat_rate": rate(closed[RETREAT - 1]),
            "fail_reverse_rate": rate(closed[REVERSE - 1]),
            "fail_episode_end_rate": rate(closed[EPISODE_END - 1]),
            "slot_dwell_frac": sums["dwell_steps"] / steps,
            "slot_contact_frac": sums["contact_steps"] / steps,
            "deep_intrusion_frac": sums["deep_steps"] / steps,
            "intrusion_cost_mean": sums["cost_sum"] / steps,
            "record_overflow": float(sums["overflow"]),
        }
        depths = torch.tensor([record["max_depth"] for record in records])
        if depths.numel():
            stats["attempt_depth_p50"] = float(depths.quantile(0.5))
            stats["attempt_depth_p95"] = float(depths.quantile(0.95))
            stats["attempt_depth_max"] = float(depths.max())
        return stats

    def collect(self, env_ids):
        """Scalar stats and raw records for the gap envs among ``env_ids`` (call after close_episode)."""
        snap = self.snapshot(env_ids)
        if snap is None:
            return {}, []
        return self.summarize(*snap), snap[1]


class EpisodeArchive:
    """Keeps the FIRST episode of every env across resets (evaluation and smoke runs).

    ``GapMonitor.reset`` wipes the tracker whenever an env resets, so anything read afterwards
    would miss every finished episode. The archive hooks ``monitor.reset``: it fails the open
    attempts as ``episode_end``, snapshots the env's counters and records, and only then lets
    the reset happen. Later episodes of the same env are ignored, so the result is exactly one
    episode per env regardless of early terminations.
    """

    def __init__(self, tracker: GapAttemptTracker):
        self.tracker = tracker
        self.finished = torch.zeros(tracker.n, dtype=torch.bool, device=tracker.device)
        self.sums, self.records, self.truncated = None, [], 0
        self._monitor, self._original, self._had_own_reset = None, None, False

    def archive(self, env_ids, *, truncated: bool = False):
        ids = self.tracker._ids(env_ids)
        fresh = ids[~self.finished[ids]]
        if fresh.numel() == 0:
            return
        self.tracker.close_episode(fresh)
        snap = self.tracker.snapshot(fresh)
        self.finished[fresh] = True
        self.truncated += int(fresh.numel()) if truncated else 0
        if snap is not None:
            self.sums = self.tracker.merge(self.sums, snap[0])
            self.records.extend(snap[1])

    def install(self, monitor):
        self._monitor, self._original = monitor, monitor.reset
        self._had_own_reset = "reset" in vars(monitor)

        def reset(env_ids=None):
            self.archive(env_ids)
            self._original(env_ids)

        monitor.reset = reset

    def finish(self):
        """Archive envs whose first episode never ended, restore ``monitor.reset``, return (stats, records)."""
        self.archive(None, truncated=True)
        if self._monitor is not None:
            if self._had_own_reset:
                self._monitor.reset = self._original
            else:
                vars(self._monitor).pop("reset", None)  # fall back to the class method
            self._monitor = None
        return self.tracker.summarize(self.sums, self.records), self.records
