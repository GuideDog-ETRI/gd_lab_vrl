#!/usr/bin/env python3
"""E2 orchestrator (host python, needs PyYAML): convert models, run the selected E2 trials (and E1 jobs) on the
configured GPUs, then summarise.

  python3 scripts/gast_narrow/e2/run_e2.py CONFIG.yaml [--only gap,narrow,e1] [--models a,b] [--conditions c,d]
                                            [--out DIR] [--dry_run] [--summary_only]

- Output: <out_root>/<name>/ (or --out): models/, plans/, trials/<model>__<condition>/result.json, e1/, summary.md/json.
- Finished trials (result.json present) are skipped, so a rerun resumes.
- E1 jobs reuse the E1 harness as UPDATE136/153 did (run_trial128.sh, 32 envs x 30 s, vx 1.0); E1 sees the
  narrow models with a zero LiDAR map and the margin folded in (convert_models.py --to e1).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ROBOT_WIDTH = 0.48  # mdp_v2.ROBOT_WIDTH (host side: no torch / gd_lab import); passable when width >= 0.48 + 2 m


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build_cells(cfg: dict, suites: list[str]) -> list[dict]:
    s = cfg["suites"]
    cells = []
    if "gap" in suites:
        g = s["gap"]
        for d, lv, v in itertools.product(g["directions"], g["levels"], g["speeds"]):
            cells.append(dict(suite="gap", family="platform_gap", direction=d, level=lv, speed=v, envs=g["envs"],
                              key=f"gap/{d}/L{lv}/v{v}"))
    if "stairs" in suites:
        g = s["stairs"]
        for f, d, lv in itertools.product(g["families"], g["directions"], g["levels"]):
            cells.append(dict(suite="stairs", family=f, direction=d, level=lv, speed=g["speed"], envs=g["envs"],
                              key=f"stairs/{f}/{d}/L{lv}"))
    if "terrain" in suites:
        g = s["terrain"]
        for f, lv in itertools.product(g["families"], g["levels"]):
            cells.append(dict(suite="terrain", family=f, direction=g.get("direction", "outward"), level=lv,
                              speed=g["speed"], envs=g["envs"], key=f"terrain/{f}/L{lv}"))
    if "narrow" in suites:
        g = s["narrow"]
        for f, case, d in itertools.product(g["families"], g["cases"], g.get("directions", ["forward"])):
            if d == "backward" and case["width"] < ROBOT_WIDTH + 2 * case["margin"]:
                continue  # blocked passages are always driven forward (the v2 zone watches the robot front)
            tag = "/back" if d == "backward" else ""
            cells.append(dict(suite="narrow", family=f, case=case["name"], width=case["width"], margin=case["margin"],
                              direction=d, envs=g["envs"], key=f"narrow/{f}/{case['name']}{tag}"))
    if "turn" in suites:  # v24 only: in-place turn pockets (docs/design/GAST_NARROW_V24_TURN_DESIGN.md section 8.1)
        g = s["turn"]
        for group in g["groups"]:
            for f, case, d, rate in itertools.product(group["families"], group["cases"], g["directions"],
                                                      g.get("rates", [0.5])):
                rtag = f"/w{rate}" if len(g.get("rates", [0.5])) > 1 else ""
                cells.append(dict(suite="turn", family=f, case=case["name"], radius=case["radius"],
                                  margin=case["margin"], direction=d, angle_deg=g.get("angle_deg", 180.0), rate=rate,
                                  envs=g["envs"], key=f"turn/{f}/{case['name']}/{d}{rtag}"))
    if "dead_end" in suites:
        g = s["dead_end"]
        for f, lv in itertools.product(g["families"], g["levels"]):
            cells.append(dict(suite="dead_end", family=f, case=f"L{lv}", level=lv, margin=g["margin"], envs=g["envs"],
                              key=f"dead_end/{f}/L{lv}"))
    return cells


def container(cfg, gpu: int | None, extra_env: dict | None = None, binds: list[str] | None = None) -> tuple[list, dict]:
    hw = cfg["hardware"]
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / "src"), TRAIN_ARM="4", PYTHONUNBUFFERED="1", OMNI_KIT_ACCEPT_EULA="YES",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", GAST_MATH_SDPA="1", GAST_NAN_DIAG="0",
               GD_LAB_BACKGAP_RAMP_STEPS="0", GD_LAB_V21_SPEED_RAMP_STEPS="0", GD_LAB_V2_FORCE_RAMP_STEPS="0")
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.update(extra_env or {})
    cmd = ["apptainer", "exec"] + (["--nv"] if gpu is not None else []) + ["--writable-tmpfs"]
    for b in binds or []:
        cmd += ["--bind", b]
    return cmd + [hw["sif"], hw["python"]], env


def run_pool(jobs: list[dict], gpus: list[int], per_gpu: int, label: str) -> list[dict]:
    """jobs: dict(name, out, make(gpu, slot) -> (cmd, env, cwd)). Runs each job once, result.json marks done."""
    todo = queue.Queue()
    for job in jobs:
        todo.put(job)
    done, lock = [], threading.Lock()

    def worker(gpu, slot):
        while True:
            try:
                job = todo.get_nowait()
            except queue.Empty:
                return
            out = Path(job["out"])
            if (out / "result.json").is_file():
                continue
            out.mkdir(parents=True, exist_ok=True)
            cmd, env, cwd = job["make"](gpu, slot)
            t = time.time()
            with open(out / "run.log", "w") as log:
                proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                code = proc.wait()
            ok = (out / "result.json").is_file()
            line = f"{time.strftime('%H:%M:%S')} [{label}] gpu{gpu}.{slot} {job['name']} exit={code} " \
                   f"result={int(ok)} wall={time.time() - t:.0f}s"
            print(line, flush=True)
            with lock:
                done.append(dict(name=job["name"], exit=code, ok=ok))

    threads = []
    for gpu in gpus:
        for slot in range(per_gpu):
            th = threading.Thread(target=worker, args=(gpu, slot), daemon=True)
            th.start()
            threads.append(th)
            time.sleep(15)  # staggered Kit start-up
    for th in threads:
        th.join()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", default="", help="comma list of suites (gap,stairs,terrain,narrow,dead_end,turn,e1)")
    ap.add_argument("--models", default="")
    ap.add_argument("--conditions", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--summary_only", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    only = set(filter(None, a.only.split(",")))
    pick = lambda name, chosen: not chosen or name in chosen  # noqa: E731
    models = {k: resolve(v) for k, v in cfg["models"].items() if pick(k, set(filter(None, a.models.split(","))))}
    gpus = [int(g) for g in cfg["hardware"]["gpus"]]
    if socket.gethostname().lower().startswith("onvlm2") and 3 in gpus:
        sys.exit("GPU 3 is not ours on onvlm2")
    out = resolve(a.out) if a.out else resolve(cfg["out_root"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, out / "config.yaml")
    suites_on = [k for k, v in cfg["suites"].items() if v.get("enabled", True) and pick(k, only)]
    e1 = cfg.get("e1", {})
    e1_on = bool(e1.get("enabled", False)) and pick("e1", only) or ("e1" in only)
    conds = {k: v for k, v in cfg["conditions"].items() if v.get("enabled", True)
             and pick(k, set(filter(None, a.conditions.split(","))))}

    # plans
    trials = []
    for cname, cond in conds.items():
        suites = [s for s in cond["suites"] if s in suites_on]
        if not suites:
            continue
        cells = build_cells(cfg, suites)
        plan = dict(task=cfg.get("task", "v2"), condition=dict(cond, name=cname), seed=cfg["seed"],
                    seconds=cfg["seconds"], cells=cells)
        (out / "plans").mkdir(exist_ok=True)
        (out / "plans" / f"{cname}.json").write_text(json.dumps(plan, indent=1))
        for m in models:
            trials.append((m, cname, sum(c["envs"] for c in cells)))
    print(f"[E2] out={out} models={list(models)} conditions={list(conds)} suites={suites_on} e1={e1_on}")
    for m, c, n in trials:
        print(f"  trial {m} x {c}: {n} envs")
    if a.dry_run:
        return
    if not a.summary_only:
        # models
        mdir = out / "models"
        mdir.mkdir(exist_ok=True)
        for name, src in models.items():
            dst = mdir / f"{name}_v2.pt"
            if not dst.is_file():
                cmd, env = container(cfg, None, binds=[f"{ROOT}:{ROOT}"])
                subprocess.run(cmd + [str(HERE / "convert_models.py"), "--to", "v2", str(src), str(dst)], env=env,
                               check=True)
        # E2 trials
        def make_trial(model, cname):
            def make(gpu, slot):
                tmp = out / "usd_tmp" / f"g{gpu}s{slot}"
                tmp.mkdir(parents=True, exist_ok=True)
                cmd, env = container(cfg, gpu, binds=[f"{tmp}:/tmp/IsaacLab"])
                return (cmd + [str(HERE / "e2_trial.py"), "--headless", "--plan", str(out / "plans" / f"{cname}.json"),
                               "--model", str(mdir / f"{model}_v2.pt"), "--out", str(out / "trials" / f"{model}__{cname}")],
                        env, str(ROOT))
            return dict(name=f"{model}__{cname}", out=out / "trials" / f"{model}__{cname}", make=make)
        jobs = [make_trial(m, c) for m, c, _ in trials]
        if jobs:
            run_pool(jobs, gpus, int(cfg["hardware"]["e2_slots_per_gpu"]), "E2")
        if e1_on:
            run_e1(cfg, e1, models, out, gpus)
    subprocess.run([sys.executable, str(HERE / "summarize_e2.py"), str(out)], check=True)


def run_e1(cfg, e1, models, out, gpus):
    harness = Path(e1["harness"])
    stage = harness / "models_e2" / out.name
    jobs = []
    for name, src in models.items():
        d = stage / name / "best_top5"
        d.mkdir(parents=True, exist_ok=True)
        if not (stage / name / "params").exists():
            shutil.copytree(harness / e1["params_from"], stage / name / "params")
        pt = d / f"{name}.pt"
        if not pt.is_file():
            cmd, env = container(cfg, None, binds=[f"{ROOT}:{ROOT}", f"{harness}:{harness}"])
            subprocess.run(cmd + [str(HERE / "convert_models.py"), "--to", "e1", "--margin_obs",
                                  str(e1.get("margin_obs", 0.0)), str(src), str(pt)], env=env, check=True)
        for (label, fam), arm, seed, ir in itertools.product(e1["families"].items(), e1["arms"], e1["seeds"], e1["ir"]):
            jname = f"{name}{'ir0' if ir == 'zero' else ''}_s{seed}_{arm}_{label}"
            jobs.append(dict(name=jname, out=out / "e1" / jname, make=e1_make(cfg, harness, fam, arm, seed, ir, pt,
                                                                              out / "e1" / jname)))
    run_pool(jobs, gpus, int(cfg["hardware"]["e1_slots_per_gpu"]), "E1")


def e1_make(cfg, harness, fam, arm, seed, ir, pt, odir):
    def make(gpu, slot):
        env = dict(os.environ)
        a, extra = arm, {}
        if arm == "v25or":
            a, extra = "L0S", {"U130_DDEEP": "0.7", "U130_DUAL": "2"}
        env.update(GAST_IMAGE=cfg["hardware"]["sif"], GAST_CONTAINER_PYTHON=cfg["hardware"]["python"], GAST_GPU=str(gpu),
                   GD_LAB_REPLAY_LOCK=str(harness / "logs" / f"replay_e2_g{gpu}s{slot}.lock"),
                   GAST_USD_TMP=str(harness / "logs" / f"usd_tmp_e2_g{gpu}s{slot}"), U129_FAMILY=fam,
                   UPDATE116_GPU_LOCK_HELD="1", U127_SENSOR_COUNT="1", OMP_NUM_THREADS="2", U136_IR=ir,
                   U130_SENSOR="lidar", U136_CAM_OFF="", U130_TEACHER=str(pt), **extra)
        cmd = ["timeout", "-k", "60", "2400", "bash", "run_trial128.sh", a, str(seed), "32", str(odir / "result.json"), "30"]
        return cmd, env, str(harness)
    return make


if __name__ == "__main__":
    main()
