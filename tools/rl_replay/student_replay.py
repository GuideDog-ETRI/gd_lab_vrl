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
                    return
            reset = done.reshape(-1).bool()
            hidden[reset] = 0
            episodes += reset.long()
            attention.reset(reset)
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
                output, memory, _, _, _ = student.encode(frames[rows], hidden[rows], extra[3][rows],
                    (step-packet.capture_step)*dt, torch.ones(len(rows), device=device))
                if not torch.isfinite(output).all() or not torch.isfinite(memory).all():
                    raise RuntimeError("non-finite student inference")
                hidden[rows] = memory
                attention.latent[rows] = output
                attention.ready[rows] = True
                attention.stamp[rows] = extra[4][rows]
                last[rows] = packet.capture_step
