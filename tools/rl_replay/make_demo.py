"""Synthetic rollout for trying tools/rl_replay/viewer.html without a simulator: python3 make_demo.py -> demo.json."""
import json, math, random
random.seed(0)
T, dt, g, lam = 400, 0.01, 0.99, 0.95
legs = {"FR": (0.33, -0.12), "FL": (0.33, 0.12), "RR": (-0.33, -0.12), "RL": (-0.33, 0.12)}
body_names = ["base"] + [f"{l}_{p}" for l in legs for p in ("hip", "thigh", "calf", "foot")]
terms = ["track_lin_vel_xy", "track_ang_vel_z", "lin_vel_z", "ang_vel_xy", "joint_torques", "action_rate",
         "feet_air_time", "base_contact", "gap_foothold_margin", "gap_hind_hop", "stair_handle_disturbance", "orientation"]
def gap_floor(x): return -0.65 if (1.4 < x < 1.6) or (2.6 < x < 2.85) else 0.0
env = {k: [] for k in ["base_pos","base_quat","base_lin_vel_b","command","bodies","joint_pos","joint_vel","action_mean","action_std","action","value","terms","reward","done","foot_contact","scan","latent"]}
for t in range(T):
    x = 0.6 * t * dt; z = 0.5 + 0.01 * math.sin(t * 0.3)
    env["base_pos"].append([x, 0, z]); env["base_quat"].append([1, 0, 0, 0]); env["base_lin_vel_b"].append([0.6, 0, 0]); env["command"].append([0.6, 0, 0])
    bodies = [[x, 0, z]]; contact = []
    for k, (lx, ly) in enumerate(legs.values()):
        ph = t * 0.25 + (0 if k in (0, 3) else math.pi)
        lift = max(0, math.sin(ph)) * 0.12; fx = x + lx + 0.08 * math.cos(ph)
        hip = [x + lx, ly, z]; thigh = [x + lx, ly * 1.6, z - 0.02]; calf = [fx - 0.05, ly * 1.6, z - 0.25 + lift / 2]; foot = [fx, ly * 1.6, gap_floor(fx) * 0 + lift + 0.02]
        bodies += [hip, thigh, calf, foot]; contact.append(int(lift < 0.01))
    env["bodies"].append(bodies); env["foot_contact"].append(contact)
    a = [0.3 * math.sin(t * 0.25 + j) for j in range(12)]
    env["joint_pos"].append(a); env["joint_vel"].append([0.0] * 12); env["action_mean"].append(a); env["action_std"].append([0.15] * 12); env["action"].append(a)
    near_gap = any(abs(x - c) < 0.3 for c in (1.5, 2.72))
    r = [0.02 * (1 - 0.1 * random.random()), 0.005, -0.001, -0.0005, -0.002, -0.001 * (2 if near_gap else 1), 0.003 if contact[0] else 0,
         0, 0.004 if near_gap else 0, -0.006 if near_gap and t % 30 < 5 else 0, 0, -0.0008]
    env["terms"].append(r); env["reward"].append(sum(r)); env["done"].append(0)
    env["value"].append(1.8 + 0.2 * math.sin(t * 0.02) - (0.3 if near_gap else 0))
    env["scan"].append([[x + (i - 8) * 0.1, (j - 5) * 0.1, gap_floor(x + (i - 8) * 0.1)] for j in range(11) for i in range(17)])
    env["latent"].append([0.0] * 32)
ret, tret, adv, delta = [0] * T, [[0] * len(terms) for _ in range(T)], [0] * T, [0] * T
nv, nr, ntr, na = env["value"][-1], env["value"][-1], [0] * len(terms), 0
for t in reversed(range(T)):
    delta[t] = env["reward"][t] + g * nv - env["value"][t]; adv[t] = delta[t] + g * lam * na
    ret[t] = env["reward"][t] + g * nr; tret[t] = [env["terms"][t][k] + g * ntr[k] for k in range(len(terms))]
    nv, nr, ntr, na = env["value"][t], ret[t], tret[t], adv[t]
env.update({"return": ret, "term_returns": tret, "advantage": adv, "delta": delta})
meta = {"task": "demo (synthetic)", "checkpoint": "-", "checkpoint_sha256": "-", "dt": dt, "gamma": g, "lam": lam, "steps": T, "num_envs": 1,
        "stochastic": False, "vx": 0.6, "term_names": terms, "body_names": body_names,
        "joint_names": [f"{l}_{j}" for l in ("FR", "FL", "HR", "HL") for j in ("HIP", "THIGH", "KNEE")], "foot_names": [f"{l}_foot" for l in legs]}
json.dump({"meta": meta, "envs": [env]}, open("demo.json", "w"))
