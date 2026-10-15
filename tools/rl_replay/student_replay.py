"""Inference-only student replay. Never creates optimizers, writers or training artifacts."""
import hashlib
from pathlib import Path
import torch
from rollout_log import RolloutLog
from runtime import fix_velocity, environment_metadata


class SupervisionCoverage:
    """P2: of the real gap cells (semantic gap & known), how many does the student's spatial loss supervise
    (gap & known & teacher-visible & valid, geometry.targets mask), counted at every capture like training."""

    def __init__(self):
        self.c = {k: 0 for k in ("captures", "rows", "near_gap_rows", "gap_cells", "gap_supervised",
                                 "gap_cells_ahead", "gap_supervised_ahead", "gap_height_supervised")}

    def add(self, clean, teacher_terrain):
        from gd_lab.gast.geometry import near_gap_rows, targets
        _, mask = targets(clean, teacher_terrain)                     # [B, 187, 6]
        _, _, gap, known = clean.reshape(-1, 4, 11, 17).unbind(1)
        real = ((gap > .5) & known.bool()).flatten(1)                # [B, 187]
        sup = mask[..., 2].bool() & real
        ahead = torch.zeros(11, 17, dtype=torch.bool, device=clean.device); ahead[:, 9:] = True
        ahead = ahead.flatten()[None]
        c = self.c
        c["captures"] += 1; c["rows"] += clean.shape[0]; c["near_gap_rows"] += int(near_gap_rows(clean).sum())
        c["gap_cells"] += int(real.sum()); c["gap_supervised"] += int(sup.sum())
        c["gap_cells_ahead"] += int((real & ahead).sum()); c["gap_supervised_ahead"] += int((sup & ahead).sum())
        c["gap_height_supervised"] += int((mask[..., 0].bool() & real).sum())

    def report(self):
        c = self.c
        ratio = lambda a, b: (a / b) if b else None
        return {**c, "gap_coverage": ratio(c["gap_supervised"], c["gap_cells"]),
                "gap_coverage_ahead": ratio(c["gap_supervised_ahead"], c["gap_cells_ahead"]),
                "gap_height_coverage": ratio(c["gap_height_supervised"], c["gap_cells"]),
                "near_gap_row_fraction": ratio(c["near_gap_rows"], c["rows"]),
                "definition": "coverage = sum(gap & known & supervised) / sum(gap & known); "
                              "supervised = geometry.targets mask (teacher-visible & scan-valid & finite)"}


class DecoderScore:
    """Isaac counterpart of the deploy decoder_metrics: the student's decoded gap channel (spatial head, same
    tensor the ONNX 'decoded' output exports) at each delivered frame, against the clean gap label of that
    frame's capture (same pose), corridor |y| <= 0.3 m, rows that have a known gap cell in the grid."""

    def __init__(self, num_envs=None, device=None, period_s=None):
        self.s = {}
        # Hidden cells split into "seen in the last 2 s then hidden" (memory) and "not seen" (e.g. pit interior).
        from gd_lab.gast.geometry import SeenCells
        self.seen = SeenCells(num_envs, device, 2.0, period_s) if num_envs else None

    def reset(self, done):
        if self.seen is not None:
            self.seen.reset(done)

    def add(self, spatial, clean, teacher_terrain, rows=None, pose=None, stamp=None):
        h, _, gap, known = clean.reshape(-1, 4, 11, 17).unbind(1)
        visible = teacher_terrain.reshape(-1, 2, 11, 17)[:, 1] > .5
        memory = None
        if self.seen is not None and rows is not None:
            memory = self.seen.update(rows, visible.flatten(1), pose[:, :3], stamp).reshape(-1, 11, 17) & ~visible
        near = ((gap > .5) & known.bool()).flatten(1).any(1)
        if not near.any():
            return
        pred = spatial.reshape(-1, 11, 17, 6)[near]
        gap, known, visible, h = gap[near] > .5, known[near].bool(), visible[near], h[near] / 5
        memory = memory[near] if memory is not None else None
        corridor = torch.zeros(11, 17, dtype=torch.bool, device=gap.device); corridor[2:9] = True
        # Offset profile: decoded gap prob vs column offset (cells, + = ahead) to the nearest real gap cell in the
        # same row. A memory that is in the right place peaks at offset 0.
        if memory is not None:
            col = torch.arange(17, device=gap.device)
            gcols = torch.where(gap & known, col.expand_as(gap), 1000)            # [B,11,17]
            off = col[None, None, :, None] - gcols[:, :, None, :]              # [B,11,17(cell),17(gapcol)]
            nearest = off.gather(-1, off.abs().argmin(-1, keepdim=True))[..., 0]  # [B,11,17]
            prob = torch.sigmoid(pred[..., 2])
            for name, m in (("visible", visible & known & corridor), ("hidden_seen_2s", memory & known & corridor)):
                ok = m & (nearest.abs() <= 6)
                b = self.s.setdefault("offset/" + name, {"sum": torch.zeros(13), "n": torch.zeros(13)})
                idx = (nearest[ok] + 6).cpu()
                b["sum"].index_add_(0, idx, prob[ok].float().cpu()); b["n"].index_add_(0, idx, torch.ones(len(idx)))
        ahead = torch.zeros_like(corridor); ahead[:, 9:] = True
        for zone, sel in (("ahead", corridor & ahead), ("under_behind", corridor & ~ahead)):
            kinds = ("all", "teacher_visible", "teacher_hidden") + (("hidden_seen_2s", "hidden_never_seen") if memory is not None else ())
            for vis in kinds:
                m = known & sel
                if vis == "teacher_visible": m = m & visible
                if vis == "teacher_hidden": m = m & ~visible
                if vis == "hidden_seen_2s": m = m & memory
                if vis == "hidden_never_seen": m = m & ~visible & ~memory
                b = self.s.setdefault(f"{zone}/{vis}", {"gp": [], "dp": [], "gh": [], "dh": [], "th": [], "td": []})
                b["gp"].append(pred[..., 2][m & gap].cpu()); b["dp"].append(pred[..., 2][m & ~gap].cpu())
                b["gh"].append(pred[..., 0][m & gap].cpu()); b["dh"].append(pred[..., 0][m & ~gap].cpu())
                b["th"].append(h[m & gap].cpu()); b["td"].append(h[m & ~gap].cpu())

    def report(self):
        def auc(pos, neg):
            """P(pos > neg) + 0.5 P(pos == neg): ties count half (all-equal -> 0.5)."""
            if len(pos) == 0 or len(neg) == 0:
                return None
            neg = neg.double().sort().values
            pos = pos.double()
            below = torch.searchsorted(neg, pos, right=False).double()
            not_above = torch.searchsorted(neg, pos, right=True).double()
            return float((below + .5 * (not_above - below)).sum() / (len(pos) * len(neg)))
        out = {}
        for k, b in self.s.items():
            if k.startswith("offset/"):
                out[k] = {"offset_cells": list(range(-6, 7)), "mean_gap_prob": (b["sum"] / b["n"].clamp_min(1)).tolist(),
                          "cells": b["n"].tolist()}
                continue
            gp, dp, gh, dh, th, td = (torch.cat(b[x]) if b[x] else torch.zeros(0) for x in ("gp", "dp", "gh", "dh", "th", "td"))
            mean = lambda t: float(t.mean()) if len(t) else None
            out[k] = {"gap_cells": len(gp), "deck_cells": len(dp), "gap_auc": auc(gp, dp),
                      "height_gap_minus_deck_m": (mean(gh) - mean(dh)) if len(gh) and len(dh) else None,
                      "label_height_gap_minus_deck_m": (mean(th) - mean(td)) if len(th) and len(td) else None}
        return out


@torch.no_grad()
def run(env, teacher, student, attention, transport, args, agent_cfg, env_cfg, teacher_path, iteration):
    base = env.unwrapped
    student.eval()
    fix_velocity(base.command_manager.get_term("base_velocity"), args.replay_vx)
    obs, _ = env.reset()
    dt, device, n = base.step_dt, base.device, base.num_envs
    hidden = student.init_hidden(n, device)
    episodes = torch.zeros(n, dtype=torch.long, device=device)
    last = torch.full_like(episodes, -1)
    skip = round(args.replay_warmup_seconds / dt)
    log = None
    coverage = SupervisionCoverage() if getattr(args, "replay_supervision_out", None) else None
    decoder = (DecoderScore(n, device, transport.interval[0] * dt) if getattr(args, "replay_decoder_out", None)
               else None)
    from gd_lab.core.camera_transport import delivery_mask
    while True:
        capture_step = transport.next_capture(base.common_step_counter)
        base._vrl_camera_capture_step = capture_step
        while base.common_step_counter < capture_step:
            age = base.common_step_counter * dt - attention.stamp
            fresh = (age < .3) & attention.ready
            latent = attention.latent * fresh[:, None]
            actions = teacher.act_with_terrain_latent(obs, latent)
            if skip <= 0:
                if log is None:
                    log = RolloutLog(base, task=args.task, checkpoint=args.student_resume,
                        gamma=float(agent_cfg.algorithm.gamma), lam=float(agent_cfg.algorithm.lam),
                        steps=round(args.replay_seconds / dt), vx=args.replay_vx, live=args.replay_live,
                        student_view=True, camera_profile=env_cfg.camera_profile, cloud_envs=args.replay_cloud_envs,
                        terrain_history=getattr(args, "replay_terrain_history", False),
                        terrain_source="gast_history" if type(teacher).__name__ == "GastTeacherView" else "terrain",
                        extra_meta={"driver": "student", "teacher_checkpoint": str(Path(teacher_path).resolve()),
                            "teacher_sha256": hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest(),
                            "student_iteration": iteration, "training_augmentation": False,
                            "camera_noise": False, "gap_ghost": False,
                            "camera_transport": transport.config.manifest(dt),
                            "camera_interval_ms": args.camera_interval_ms, "camera_delay_ms": args.camera_delay_ms,
                            "camera_drop_prob": args.camera_drop_prob, **environment_metadata(base)})
                log.before_step(obs, mean=actions, std=torch.zeros_like(actions), action=actions,
                    value=teacher.evaluate(obs).reshape(-1), latent=latent,
                    command=base.command_manager.get_command("base_velocity"),
                    extra={"latent_err": (latent-teacher.terrain_latent(obs)).pow(2).mean(-1),
                           "teacher_action": teacher.act_inference(obs), "student_fresh": fresh.float()})
            obs, reward, done, _ = env.step(actions)
            if skip > 0:
                skip -= 1
            else:
                log.after_step(reward, done)
                if log.full():
                    log.finish(teacher.evaluate(obs).reshape(-1), args.replay_out)
                    if coverage is not None:
                        import json
                        Path(args.replay_supervision_out).write_text(json.dumps(coverage.report(), indent=1))
                        print(f"[INFO] supervision coverage: {coverage.report()}", flush=True)
                    if decoder is not None:
                        import json
                        Path(args.replay_decoder_out).write_text(json.dumps(decoder.report(), indent=1))
                        print(f"[INFO] decoder score: {decoder.report()}", flush=True)
                    return
            reset = done.reshape(-1).bool()
            hidden[reset] = 0
            episodes += reset.long()
            attention.reset(reset)
            if decoder is not None:
                decoder.reset(reset)
            step = base.common_step_counter
            if step == capture_step:
                snapshot = base._vrl_camera_snapshot
                extra = attention.capture(obs, teacher.act_inference(obs))
                transport.capture(step, (snapshot[0].clone(), episodes.clone(), extra))
                if coverage is not None:
                    coverage.add(extra[1], extra[2])
            for packet in transport.receive(step):
                frames, captured_episodes, extra = packet.payload
                valid = delivery_mask(packet.capture_step, captured_episodes, episodes, last)
                valid &= torch.isfinite(frames).flatten(1).all(-1)
                valid &= torch.isfinite(extra[3]).all(-1)
                if not valid.any():
                    continue
                rows = valid.nonzero(as_tuple=False).flatten()
                output, memory, spatial, _, _ = student.encode(frames[rows], hidden[rows], extra[3][rows],
                    (step-packet.capture_step)*dt, torch.ones(len(rows), device=device))
                if not torch.isfinite(output).all() or not torch.isfinite(memory).all():
                    raise RuntimeError("non-finite student inference")
                hidden[rows] = memory
                if decoder is not None:
                    decoder.add(spatial, extra[1][rows], extra[2][rows], rows, extra[3][rows], extra[4][rows])
                attention.latent[rows] = output
                attention.ready[rows] = True
                attention.stamp[rows] = extra[4][rows]
                last[rows] = packet.capture_step
