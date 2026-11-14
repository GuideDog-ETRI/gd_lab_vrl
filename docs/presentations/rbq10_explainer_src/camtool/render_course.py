"""Re-render a recorded MuJoCo course run (sim-state-probe JSONL) with a scripted camera.

Side follow view on the gaps / stairs, smoothly switching to a near top-down view from ``top_from_x``
(the walls hide the robot in the side view). Robot = deploy rbq_payload.xml, course = the same terrain folder.
  python render_course.py <state.jsonl> <terrain_dir> <mesh_root> <out.mp4> [top_from_x]
"""
import json
import sys
import tempfile
from pathlib import Path

import imageio
import mujoco
import numpy as np

state_path, terrain, mesh_root, out = sys.argv[1:5]
top_from = float(sys.argv[5]) if len(sys.argv) > 5 else None
terrain = Path(terrain)
if top_from is None:
    top_from = json.loads((terrain / "sections.json").read_text())["door1"] - 2.0

tmp = Path(tempfile.mkdtemp())
(tmp / "robot.xml").write_text((terrain / "rbq_payload.xml").read_text().replace("/workspace/RBQ/resources", mesh_root))
(tmp / "course.xml").write_text((terrain / "environment.xml").read_text())
(tmp / "scene.xml").write_text('<mujoco model="replay"><include file="robot.xml"/><include file="course.xml"/></mujoco>')
model = mujoco.MjModel.from_xml_path(str(tmp / "scene.xml"))
model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
data = mujoco.MjData(model)
joints = [f"joint{i}_{n}" for i, n in enumerate(["HRR", "HRP", "HRK", "HLR", "HLP", "HLK", "FRR", "FRP", "FRK",
                                                  "FLR", "FLP", "FLK"])]
qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints])

rows = [json.loads(l) for l in open(state_path) if l.startswith("{")]
t = np.array([r["wall"] for r in rows]); t -= t[0]
pos = np.array([r["pos"] for r in rows])
quat = np.array([[r["imu_xyzw"][3], *r["imu_xyzw"][:3]] for r in rows])   # -> w, x, y, z
q = np.array([r["q"] for r in rows])
for i in range(1, len(quat)):                                              # keep the hemisphere for interpolation
    if np.dot(quat[i], quat[i - 1]) < 0:
        quat[i] = -quat[i]

W, H, fps = 960, 540, 30
renderer = mujoco.Renderer(model, H, W)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
look = pos[0].copy()
frames = []
for ft in np.arange(0.0, t[-1], 1.0 / fps):
    k = min(np.searchsorted(t, ft), len(t) - 1)
    k0 = max(k - 1, 0)
    a = 0.0 if t[k] == t[k0] else (ft - t[k0]) / (t[k] - t[k0])
    p = (1 - a) * pos[k0] + a * pos[k]
    qq = (1 - a) * quat[k0] + a * quat[k]
    qq /= np.linalg.norm(qq)
    data.qpos[:3], data.qpos[3:7] = p, qq
    data.qpos[qadr] = (1 - a) * q[k0] + a * q[k]
    mujoco.mj_forward(model, data)
    look = 0.85 * look + 0.15 * p                                          # damped follow
    w = float(np.clip((p[0] - top_from) / 1.5, 0.0, 1.0))                   # 0 side view -> 1 top view
    cam.lookat[:] = look
    cam.azimuth = 90.0
    cam.elevation = (1 - w) * -12.0 + w * -78.0
    cam.distance = (1 - w) * 2.8 + w * 3.6
    renderer.update_scene(data, cam)
    frames.append(renderer.render())
imageio.mimsave(out, frames, fps=fps, macro_block_size=1, quality=7)
print("rendered", out, len(frames), "frames", f"{t[-1]:.1f} s", "top view from x =", top_from)
