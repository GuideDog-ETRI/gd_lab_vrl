"""MuJoCo check of a handstand-skill checkpoint (Python, no Pilot): scripted mode command, metrics, optional mp4.

Same observation contract as training (blind DreamWaQ 46 + skill_command 3 per frame, history 5 term-major,
"repeat_first" history init, actor 4 newest frames), actions -> soft-limited joint targets, PD with the DC-motor
torque-speed limits of the training actuators. Robot = the deploy MuJoCo RBQ10 model with the 6 kg payload removed.

  python mujoco_check.py --checkpoint model_499.pt --mjcf <deploy>/simulation/terrains/gap_pit65/rbq_payload.xml \
      --mesh_root <RBQ-nightly>/resources --scenario hold|abort|idle [--video out.mp4]
Needs mujoco (+ PyOpenGL, imageio-ffmpeg for video) on PYTHONPATH; MUJOCO_GL=glfw with a display for video.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

import os  # noqa: E402
REPO = Path(os.environ.get("GD_LAB_VRL_ROOT", Path(__file__).resolve().parents[3])).resolve()
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from gd_lab.rl.actor_critic import DreamwaqActorCritic  # noqa: E402

POLICY_JOINTS = ["FL_HIP", "FR_HIP", "HL_HIP", "HR_HIP", "FL_THIGH", "FR_THIGH", "HL_THIGH", "HR_THIGH",
                 "FL_KNEE", "FR_KNEE", "HL_KNEE", "HR_KNEE"]
MJ_JOINT = {"FL_HIP": "joint9_FLR", "FL_THIGH": "joint10_FLP", "FL_KNEE": "joint11_FLK",
            "FR_HIP": "joint6_FRR", "FR_THIGH": "joint7_FRP", "FR_KNEE": "joint8_FRK",
            "HL_HIP": "joint3_HLR", "HL_THIGH": "joint4_HLP", "HL_KNEE": "joint5_HLK",
            "HR_HIP": "joint0_HRR", "HR_THIGH": "joint1_HRP", "HR_KNEE": "joint2_HRK"}
Q0 = np.array([0.0] * 4 + [0.76] * 4 + [-1.45] * 4)
KNEE = np.array(["KNEE" in j for j in POLICY_JOINTS])
KP = np.where(KNEE, 127.77, 123.39)
KD = np.full(12, 2.4)
EFFORT = np.where(KNEE, 126.0, 93.6)
SAT = np.where(KNEE, 140.0, 104.0)
VMAX = np.where(KNEE, 11.15, 14.4)
TERM_DIMS = [3, 3, 3, 12, 12, 12, 1, 3]
HISTORY = 5
TRUNK_MASS, TRUNK_COM = 16.758, "-0.013265 0 0.011765"
FEET = ["FL", "FR", "RL", "RR"]


def walk_flag(name: str, t: float) -> float:
    """S2 'step' scenario: stepping requested 4.5-10.5 s (inside the 2-13 s request)."""
    return 1.0 if name == "step" and 4.5 <= t < 10.5 else 0.0


def scenario(name: str, t: float) -> tuple[float, float, float]:
    """(request, abort, time since change / 3 s)."""
    if name == "idle":
        return 0.0, 0.0, min(t / 3.0, 1.0)
    t_on = 2.0
    t_off = {"hold": 12.0, "step": 13.0}.get(name, 3.0)
    abort = 1.0 if name == "abort" else 0.0
    if t < t_on:
        return 0.0, 0.0, min(t / 3.0, 1.0)
    if t < t_off:
        return 1.0, 0.0, min((t - t_on) / 3.0, 1.0)
    return 0.0, abort, min((t - t_off) / 3.0, 1.0)


def build_model(mjcf: str, mesh_root: str, frictionloss: float | None = None, foot_condim: int | None = None,
                armature_scale: float | None = None) -> mujoco.MjModel:
    xml = Path(mjcf).read_text().replace("/workspace/RBQ/resources", mesh_root)
    # sim-to-sim probes: joint Coulomb friction, foot contact dimensionality, rotor armature
    if frictionloss is not None:
        xml = re.sub(r'frictionloss="[^"]*"', f'frictionloss="{frictionloss}"', xml)
    if foot_condim is not None:
        xml = xml.replace('condim="6" friction="0.8 0.8 0.8"', f'condim="{foot_condim}" friction="0.8 0.005 0.0001"')
    if armature_scale is not None:
        xml = re.sub(r'armature="([^"]*)"', lambda m: f'armature="{float(m.group(1)) * armature_scale:.6g}"', xml)
    # payload 0: trunk back to the URDF mass / COM, inertia scaled by the mass ratio
    m = re.search(r'(<body name="base_link"[^>]*>\s*<inertial pos=")([^"]*)(" mass=")([^"]*)(" fullinertia=")([^"]*)(")', xml)
    ratio = TRUNK_MASS / float(m.group(4))
    inertia = " ".join(f"{float(v) * ratio:.9g}" for v in m.group(6).split())
    xml = xml[:m.start()] + m.group(1) + TRUNK_COM + m.group(3) + f"{TRUNK_MASS}" + m.group(5) + inertia + m.group(7) + xml[m.end():]
    floor = ('<geom name="floor" type="plane" size="0 0 0.05" friction="0.8 0.8 0.8" rgba="0.75 0.78 0.8 1" '
             'contype="1" conaffinity="1"/><light pos="0 0 4" dir="0 0 -1" directional="true"/>')
    xml = xml.replace("<worldbody>", "<worldbody>" + floor, 1)
    return mujoco.MjModel.from_xml_string(xml)


def load_policy(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    dims = TERM_DIMS + ([1] if sd["actor_obs_normalizer._mean"].shape[1] // HISTORY == sum(TERM_DIMS) + 1 else [])
    n_critic = sd["critic_obs_normalizer._mean"].shape[1]
    obs = TensorDict({"policy": torch.zeros(1, sum(dims) * HISTORY), "critic": torch.zeros(1, n_critic)}, batch_size=[1])
    pol = DreamwaqActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 12, history_length=HISTORY,
                              policy_term_dims=dims, velocity_target_slice=(45, 48), actor_history_steps=4,
                              actor_obs_normalization=True, critic_obs_normalization=True,
                              actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu",
                              init_noise_std=1.0, noise_std_type="scalar", inference_only=True)
    pol.load_state_dict(sd, strict=True)
    return pol.eval(), len(dims) == len(TERM_DIMS) + 1, n_critic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--mesh_root", required=True)
    ap.add_argument("--scenario", default="hold", choices=["hold", "abort", "idle", "step"])
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="repeat trials: training-like observation noise + small initial joint offsets")
    ap.add_argument("--frictionloss", type=float, default=None)
    ap.add_argument("--foot_condim", type=int, default=None)
    ap.add_argument("--armature_scale", type=float, default=None)
    ap.add_argument("--soft_margin_deg", type=float, default=None,
                    help="joint-target soft margin of the training action term (S1 5, S2 8); default by layout")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    duration = a.duration or {"hold": 17.0, "abort": 9.0, "idle": 6.0, "step": 16.0}[a.scenario]

    pol, has_walk, n_critic = load_policy(a.checkpoint)
    model = build_model(a.mjcf, a.mesh_root, a.frictionloss, a.foot_condim, a.armature_scale)
    data = mujoco.MjData(model)
    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MJ_JOINT[j])] for j in POLICY_JOINTS])
    vadr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MJ_JOINT[j])] for j in POLICY_JOINTS])
    act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, MJ_JOINT[j].replace("joint", "motor")) for j in POLICY_JOINTS])
    lo = model.jnt_range[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MJ_JOINT[j]) for j in POLICY_JOINTS], 0]
    hi = model.jnt_range[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MJ_JOINT[j]) for j in POLICY_JOINTS], 1]
    margin = math.radians(a.soft_margin_deg if a.soft_margin_deg is not None else (8.0 if has_walk else 5.0))
    margin_vendor = math.radians(3.0)
    limited = hi - lo < 2 * math.pi + 1   # thigh range is +-2pi (effectively unlimited)
    lo_s, hi_s = np.where(limited, lo + margin, -1e9), np.where(limited, hi - margin, 1e9)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FEET]
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    rng = np.random.default_rng(a.seed) if a.seed is not None else None
    # stand pose on the floor
    data.qpos[qadr] = Q0 + (rng.uniform(-0.05, 0.05, 12) if rng is not None else 0.0)
    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    lowest = min(data.geom_xpos[g][2] for g in foot_geoms) - 0.03
    data.qpos[2] -= lowest - 0.002
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    decim = int(round(0.01 / dt))
    last_action = np.zeros(12)
    target = Q0.copy()
    hist = None
    renderer = None
    frames = []
    if a.video:
        renderer = mujoco.Renderer(model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 2.6, 90.0, -12.0
    stats = dict(max_pitch_deg=0.0, handstand_s=0.0, trunk_floor_contact=False, fell_at=None)
    # joint-level stats against the vendor Motion protections (Motion.ini: POSITION_LIMIT_deg with J_LIM_MARGIN 3 deg,
    # TORQUE_LIMIT_Nm 100/100/150, CURRENT_LIMIT 50 A) and the size of the target steps the drives must follow
    js = dict(q_min=np.full(12, 1e9), q_max=np.full(12, -1e9), tau_max=np.zeros(12), tau_raw_max=np.zeros(12),
              dq_max=np.zeros(12), err_max=np.zeros(12), dtarget_max=np.zeros(12), near_limit_s=np.zeros(12))
    prev_target = None
    walk_steps, prev_front, walk_xy0, walk_drift = 0, None, None, 0.0
    log = []
    steps = int(duration / dt)
    for k in range(steps):
        t = k * dt
        if k % decim == 0:
            quat = data.qpos[3:7]
            R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
            pg = R.T @ np.array([0.0, 0.0, -1.0])
            w_b = data.qvel[3:6].copy()
            cmd = scenario(a.scenario, t)
            noise = (lambda lo, n: rng.uniform(-lo, lo, n)) if rng is not None else (lambda lo, n: 0.0)
            terms = [(w_b + noise(0.2, 3)) * 0.25, pg + noise(0.05, 3), np.zeros(3), data.qpos[qadr] - Q0 + noise(0.01, 12),
                     (data.qvel[vadr] + noise(1.5, 12)) * 0.05, last_action,
                     np.zeros(1), np.array(cmd)] + ([np.array([walk_flag(a.scenario, t)])] if has_walk else [])
            if hist is None:
                hist = [np.repeat(x[None], HISTORY, 0) for x in terms]
            else:
                hist = [np.concatenate([h[1:], x[None]], 0) for h, x in zip(hist, terms)]
            flat = np.concatenate([h.reshape(-1) for h in hist])
            with torch.no_grad():
                action = pol.act_inference(TensorDict({"policy": torch.tensor(flat[None], dtype=torch.float32),
                                                       "critic": torch.zeros(1, n_critic)}, batch_size=[1]))[0].numpy()
            target = np.clip(Q0 + 0.25 * action, lo_s, hi_s)
            last_action = (target - Q0) / 0.25
            if prev_target is not None:
                js["dtarget_max"] = np.maximum(js["dtarget_max"], np.abs(target - prev_target))
            prev_target = target.copy()
            pitch = math.degrees(math.atan2(pg[0], -pg[2]))
            contacts = set()
            trunk_hit = False
            for c in data.contact[: data.ncon]:
                pair = {c.geom1, c.geom2}
                if floor in pair:
                    other = (pair - {floor}).pop() if len(pair) > 1 else floor
                    if other in foot_geoms:
                        contacts.add(FEET[foot_geoms.index(other)])
                    elif model.geom_bodyid[other] == trunk:
                        trunk_hit = True
            wf = walk_flag(a.scenario, t)
            front_now = {f for f in ("FL", "FR") if f in contacts}
            if wf > 0.5:
                if walk_xy0 is None:
                    walk_xy0 = data.qpos[:2].copy()
                walk_drift = max(walk_drift, float(np.linalg.norm(data.qpos[:2] - walk_xy0)))
                if prev_front is not None:
                    walk_steps += len(front_now - prev_front)
            prev_front = front_now
            if cmd[0] > 0.5:
                stats["max_pitch_deg"] = max(stats["max_pitch_deg"], pitch)
                if pitch > 75 and not ({"RL", "RR"} & contacts) and {"FL", "FR"} <= contacts:
                    stats["handstand_s"] += 0.01
            if trunk_hit and stats["fell_at"] is None:
                stats["trunk_floor_contact"], stats["fell_at"] = True, round(t, 2)
            if k % (decim * 10) == 0:
                log.append(dict(t=round(t, 2), cmd=cmd[:2], pitch=round(pitch, 1), feet="".join(sorted(contacts)),
                                z=round(float(data.qpos[2]), 3)))
        q, v = data.qpos[qadr], data.qvel[vadr]
        tau = KP * (target - q) - KD * v
        tmax = np.clip(SAT * (1.0 - v / VMAX), 0.0, EFFORT)
        tmin = np.clip(SAT * (-1.0 - v / VMAX), -EFFORT, 0.0)
        data.ctrl[act] = np.clip(tau, tmin, tmax)
        js["q_min"] = np.minimum(js["q_min"], q); js["q_max"] = np.maximum(js["q_max"], q)
        js["tau_max"] = np.maximum(js["tau_max"], np.abs(data.ctrl[act])); js["tau_raw_max"] = np.maximum(js["tau_raw_max"], np.abs(tau))
        js["dq_max"] = np.maximum(js["dq_max"], np.abs(v)); js["err_max"] = np.maximum(js["err_max"], np.abs(target - q))
        js["near_limit_s"] += (limited & ((q < lo + margin_vendor) | (q > hi - margin_vendor))) * dt
        mujoco.mj_step(model, data)
        if renderer is not None and k % int(round(1 / 30 / dt)) == 0:
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, cam)
            frames.append(renderer.render())
    R = np.zeros(9); mujoco.mju_quat2Mat(R, data.qpos[3:7]); pg = R.reshape(3, 3).T @ np.array([0, 0, -1.0])
    stats["final_upright"] = bool(pg[2] < -0.95)
    if a.scenario == "step":
        stats["walk_front_touchdowns"] = walk_steps
        stats["walk_drift_m"] = round(walk_drift, 3)
    stats["final_base_z"] = round(float(data.qpos[2]), 3)
    stats["max_pitch_deg"] = round(stats["max_pitch_deg"], 1)
    stats["handstand_s"] = round(stats["handstand_s"], 2)
    deg = lambda x: [round(float(math.degrees(v)), 1) for v in x]  # noqa: E731
    stats["joints"] = dict(names=POLICY_JOINTS, range_deg=[[round(math.degrees(l), 1), round(math.degrees(h), 1)] for l, h in zip(lo, hi)],
                           q_min_deg=deg(js["q_min"]), q_max_deg=deg(js["q_max"]),
                           tau_max_Nm=[round(float(x), 1) for x in js["tau_max"]], tau_unclipped_max_Nm=[round(float(x), 1) for x in js["tau_raw_max"]],
                           dq_max_rad_s=[round(float(x), 2) for x in js["dq_max"]], track_err_max_deg=deg(js["err_max"]),
                           target_step_max_deg=deg(js["dtarget_max"]), within_3deg_of_limit_s=[round(float(x), 2) for x in js["near_limit_s"]])
    print(json.dumps(dict(scenario=a.scenario, checkpoint=a.checkpoint, **stats)))
    for row in log:
        print(row)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(stats=stats, log=log), indent=1))
    if a.video and frames:
        import imageio
        imageio.mimsave(a.video, frames, fps=30, macro_block_size=1)
        print("video", a.video, len(frames), "frames")


if __name__ == "__main__":
    main()
