"""Per-step rollout log shared by record_rollout.py (teacher drives) and train_student_live.py --replay_out
(student drives). Call ``before_step`` with what the policy saw and chose, ``after_step`` with what the
environment returned, and ``finish`` to add returns/GAE and write the JSON that viewer.html plays.

Optionally streams every step as one NDJSON line (``live``) so the viewer can follow a running simulator.
"""
import hashlib
import json
import os
from pathlib import Path

import torch


def _round(t, digits=4):
    return torch.round(t.float() * 10**digits).div(10**digits).cpu().tolist()


def policy_terrain_input(obs, n, device):
    """The terrain group the teacher's encoder reads: 'terrain', else the newest gast_history frame."""
    keys = obs.keys()
    if "terrain" in keys:
        return obs["terrain"]
    if "gast_history" in keys:
        return obs["gast_history"].reshape(n, 8, -1)[:, -1, :374]
    return torch.zeros(n, 0, device=device)


class RolloutLog:
    def __init__(self, base, *, task, checkpoint, gamma, lam, steps, stochastic=False, vx=None,
                 live=None, student_view=False, camera_profile=None, cloud_envs=2, cloud_stride=4, extra_meta=None):
        self.base, self.steps = base, steps
        self.robot = base.scene["robot"]
        self.rewards = base.reward_manager
        sensors = base.scene.sensors if hasattr(base.scene, "sensors") else {}
        self.scanner = sensors.get("height_scanner")
        self.contacts = sensors.get("contact_forces")
        self.foot_ids = [self.robot.body_names.index(n) for n in self.robot.body_names if n.endswith("_foot")]
        self.contact_foot_ids = ([self.contacts.body_names.index(self.robot.body_names[i]) for i in self.foot_ids]
                                 if self.contacts is not None else [])
        self.dt, self.n = base.step_dt, base.num_envs
        self.rec = {}
        self.clouds = []
        self.student_view = None
        if student_view:
            self._init_student_view(camera_profile, cloud_envs, cloud_stride)
        self.meta = {"task": task, "checkpoint": os.path.abspath(checkpoint),
                     "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                     "dt": self.dt, "gamma": gamma, "lam": lam, "steps": steps, "num_envs": self.n,
                     "stochastic": stochastic, "vx": vx, "term_names": list(self.rewards.active_terms),
                     "body_names": list(self.robot.body_names), "joint_names": list(self.robot.joint_names),
                     "foot_names": [self.robot.body_names[i] for i in self.foot_ids], **(extra_meta or {})}
        self.live = None
        if live:
            Path(live).parent.mkdir(parents=True, exist_ok=True)
            self.live = open(live, "w")
            head = {**self.meta, "steps_planned": steps, "live": True}
            head.pop("steps")
            self.live.write(json.dumps({"meta": head}, separators=(",", ":")) + "\n")
            self.live.flush()

    def _init_student_view(self, profile, k, stride):
        from gd_lab.core.camera_contract import DEFAULT_CAMERA_PROFILE, load_camera_contract
        from gd_lab.core.camera_geometry import depth_pixels_world
        from gd_lab.mdp.camera_observations import canonical_camera_snapshot
        contract = load_camera_contract(profile or DEFAULT_CAMERA_PROFILE)
        k = min(k, self.n)

        def view():
            """World points of the four belly depth images (what the student sees), flat [x,y,z,...] per env."""
            _, depths, positions, rotations, intrinsics = canonical_camera_snapshot(self.base.scene, contract)
            pts = depth_pixels_world(depths[:k], positions[:k], rotations[:k], intrinsics[:k])
            pts, dep = pts[:, :, ::stride, ::stride].reshape(k, -1, 3), depths[:k, :, ::stride, ::stride].reshape(k, -1)
            lo, hi = contract.depth_clip
            ok = torch.isfinite(dep) & (dep > lo) & (dep < hi * 0.999) & torch.isfinite(pts).all(-1)
            return [torch.round(pts[e][ok[e]] * 1000).div(1000).flatten().cpu().tolist() for e in range(k)]
        self.student_view = view

    def _add(self, key, value):
        self.rec.setdefault(key, []).append(value.detach().clone())

    def before_step(self, obs, *, mean, std, action, value, latent, command, extra=None):
        """State and decision at step t. ``extra``: more per-env tensors to keep (e.g. student latent error)."""
        d, n = self.robot.data, self.n
        dev = mean.device
        for key, val in (("base_pos", d.root_pos_w), ("base_quat", d.root_quat_w), ("base_lin_vel_b", d.root_lin_vel_b),
                         ("command", command), ("bodies", d.body_pos_w), ("joint_pos", d.joint_pos),
                         ("joint_vel", d.joint_vel), ("action_mean", mean), ("action_std", std.expand_as(mean)),
                         ("action", action), ("value", value), ("latent", latent),
                         ("terrain_obs", policy_terrain_input(obs, n, dev))):
            self._add(key, val)
        self._add("scan", self.scanner.data.ray_hits_w if self.scanner is not None else torch.zeros(n, 0, 3, device=dev))
        self._add("scan_z", self.scanner.data.pos_w[:, 2] if self.scanner is not None else torch.zeros(n, device=dev))
        for key, val in (extra or {}).items():
            self._add(key, val)
        if self.student_view is not None:
            self.clouds.append(self.student_view())

    def after_step(self, reward, done):
        """What the environment returned for step t; streams the step when live."""
        dev = reward.device
        # RewardManager keeps each weighted term as a rate; the per-step contribution is rate * dt.
        self._add("terms", self.rewards._step_reward * self.dt)
        self._add("reward", reward.reshape(-1))
        self._add("done", done.reshape(-1).bool())
        if self.contacts is not None:
            self._add("foot_contact", self.contacts.data.net_forces_w[:, self.contact_foot_ids].norm(dim=-1) > 1.0)
        else:
            self._add("foot_contact", torch.zeros(self.n, len(self.foot_ids), dtype=torch.bool, device=dev))
        if self.live is not None:
            i = len(self.rec["reward"]) - 1
            self.live.write(json.dumps({"t": i, "envs": [self._row(e, i) for e in range(self.n)]},
                                       separators=(",", ":")) + "\n")
            self.live.flush()

    def _row(self, e, i):
        row = {k: _round(v[i][e]) for k, v in self.rec.items() if k not in ("done", "foot_contact")}
        row["done"] = int(self.rec["done"][i][e])
        row["foot_contact"] = self.rec["foot_contact"][i][e].int().tolist()
        if self.clouds:
            row["cloud"] = self.clouds[i][e] if e < len(self.clouds[i]) else []
        return row

    def full(self):
        return len(self.rec.get("reward", [])) >= self.steps

    def finish(self, bootstrap, out):
        """Discounted return per term, TD error and GAE with the critic, then write ``out``."""
        gamma, lam, steps = self.meta["gamma"], self.meta["lam"], len(self.rec["reward"])
        stacked = {k: torch.stack(v, 1) for k, v in self.rec.items()}  # [n, T, ...]
        r, v, d, terms = stacked["reward"], stacked["value"], stacked["done"].float(), stacked["terms"]
        returns, adv, delta = torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r)
        term_returns = torch.zeros_like(terms)
        bootstrap = bootstrap.reshape(-1)
        next_value, next_return, next_term, next_adv = bootstrap, bootstrap, torch.zeros_like(terms[:, 0]), torch.zeros_like(bootstrap)
        for t in reversed(range(steps)):
            alive = 1.0 - d[:, t]
            delta[:, t] = r[:, t] + gamma * next_value * alive - v[:, t]
            adv[:, t] = delta[:, t] + gamma * lam * alive * next_adv
            returns[:, t] = r[:, t] + gamma * alive * next_return
            term_returns[:, t] = terms[:, t] + gamma * alive[:, None] * next_term
            next_value, next_return, next_term, next_adv = v[:, t], returns[:, t], term_returns[:, t], adv[:, t]
        meta = {**self.meta, "steps": steps, "notes": {
            "terms": "weighted reward per step (rate*dt), summing to 'reward'",
            "scan": "height-scanner hits (privileged ground truth, 11x17)",
            "terrain_obs": "the teacher encoder's terrain input: 187 heights (scan_z - z - 0.5, clipped +-1, x5) + 187 validity",
            "cloud": "student view: belly-camera depth pixels as world points, flat [x,y,z,...]",
            "term_returns": "discounted return-to-go of each term; their sum is 'return' (bootstrap V(s_T) in 'return' only)",
            "advantage": "GAE(gamma, lam) with the critic value", "delta": "TD error"}}
        result = {"meta": meta, "envs": []}
        for e in range(self.n):
            env_out = {k: _round(stacked[k][e]) for k in stacked if k not in ("done", "foot_contact")}
            env_out["done"] = stacked["done"][e].cpu().int().tolist()
            env_out["foot_contact"] = stacked["foot_contact"][e].cpu().int().tolist()
            env_out.update({"return": _round(returns[e]), "term_returns": _round(term_returns[e]),
                            "advantage": _round(adv[e]), "delta": _round(delta[e])})
            if self.clouds and e < len(self.clouds[0]):
                env_out["cloud"] = [c[e] for c in self.clouds]
            result["envs"].append(env_out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, separators=(",", ":")))
        if self.live is not None:
            self.live.write(json.dumps({"end": True, "out": str(out)}) + "\n")
            self.live.close()
        print(f"[INFO] wrote {out}: {self.n} envs x {steps} steps, {len(meta['term_names'])} reward terms", flush=True)
