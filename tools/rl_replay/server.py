"""Local server for tools/rl_replay/viewer.html: serves the repository (URDF, meshes, recordings) and runs
record_rollout.py on request, so recording, opening and playing all happen in the viewer.

  python3 tools/rl_replay/server.py [--port 8765]      then open http://127.0.0.1:8765/tools/rl_replay/viewer.html

API (JSON, 127.0.0.1 only):
  GET  /api/recordings              recordings in logs/rl_replay/ (newest first, with their meta)
  GET  /api/sources                 tasks and teacher checkpoints that can be recorded
  POST /api/record                  {task, checkpoint, num_envs, seconds, vx, stochastic, name, force, live, student_view}
  GET  /api/record                  state of the current/last recording job (log tail, output, exit code)
  GET  /api/live?path=..&offset=N   complete lines from byte offset N of a live stream
One recording at a time. Compute GPU processes block recording; no force bypass.
"""
import argparse
import urllib.parse
import json
import os
import re
import subprocess
import threading
import time
import math
from runtime import validate_window, task_source
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORDINGS = REPO / "logs" / "rl_replay"
SIF = os.environ.get("GD_LAB_SIF", str(Path.home() / "workspace/gd_lab_isaaclab.sif"))
PYTHON = os.environ.get("GD_LAB_PYTHON", str(Path.home() / "workspace/venv_apptainer/bin/python"))
TASKS = [  # policy family -> the task it was trained on
    ("GAST teacher v2.1", "Gd-GastGapCleanV21-Rbq10-Dreamwaq-v0"),
    ("GAST teacher v2", "Gd-GastGapCleanV2-Rbq10-Dreamwaq-v0"),
    ("BIVT-Ray Clean v2.1", "Gd-VrlGapFinetuneCleanV21Raycast-Rbq10-Dreamwaq-v0"),
    ("BIVT-Ray Clean", "Gd-VrlGapFinetuneCleanRaycast-Rbq10-Dreamwaq-v0"),
]
job = {"state": "idle"}
lock = threading.Lock()


def recordings():
    out = []
    for p in sorted(RECORDINGS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with p.open() as f:
                head = f.read(1 << 16)
            meta, _ = json.JSONDecoder().raw_decode(head, head.index('"meta"') + head[head.index('"meta"'):].index("{"))
            meta = {k: meta.get(k) for k in ("task", "num_envs", "steps", "vx", "stochastic", "checkpoint")}
        except Exception:
            meta = {}
        out.append({"path": str(p.relative_to(REPO)), "name": p.name, "bytes": p.stat().st_size,
                    "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)), "meta": meta})
    for p in sorted(RECORDINGS.glob("*.live.ndjson"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with p.open() as f:
                meta = json.loads(f.readline())["meta"]
            meta = {k: meta.get(k) for k in ("task", "num_envs", "vx", "stochastic", "checkpoint")}
        except Exception:
            meta = {}
        finished = (RECORDINGS / p.name.replace(".live.ndjson", ".json")).is_file()
        out.insert(0, {"path": str(p.relative_to(REPO)), "name": p.name, "bytes": p.stat().st_size, "live": True,
                       "finished": finished, "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                       "meta": meta})
    return out


def live_lines(query):
    """Complete lines from `from` on of a live stream under logs/rl_replay/ (a partial last line waits)."""
    q = urllib.parse.parse_qs(query)
    path = (REPO / q.get("path", [""])[0]).resolve()
    if path.parent != RECORDINGS.resolve() or not path.name.endswith(".live.ndjson") or not path.is_file():
        return 404, {"error": "no such live stream"}
    try:
        start = int(q.get("offset", ["0"])[0])
        if start < 0 or start > path.stat().st_size:
            raise ValueError("invalid offset")
        rows = []
        with path.open("rb") as f:
            f.seek(start)
            if start:
                f.seek(start-1)
                if f.read(1) != b"\n":
                    raise ValueError("offset must be a line boundary")
            next_offset = start
            for _ in range(32):
                line = f.readline()
                if not line.endswith(b"\n"):
                    break
                rows.append(json.loads(line))
                next_offset = f.tell()
        return 200, {"lines": rows, "next": next_offset}
    except (ValueError, UnicodeError):
        return 400, {"error": "invalid offset or stream"}


def sources():
    ckpts = sorted(str(p.relative_to(REPO)) for p in (REPO / "checkpoints/teachers").rglob("*.pt"))
    return {"tasks": [{"label": a, "task": b} for a, b in TASKS], "checkpoints": ckpts}


def gpu_busy():
    try:
        gpu = os.environ.get("RL_REPLAY_GPU", "0")
        processes = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
                                    "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if processes.returncode:
            return ["GPU status unavailable (fail closed)"]
        pids = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
        if any(not pid.isdecimal() for pid in pids):
            return ["GPU compute-process status unavailable (fail closed)"]
        return [f"GPU {gpu}: compute PID {pid}" for pid in pids]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return ["GPU status unavailable (fail closed)"]


def validate_request(req):
    if not isinstance(req, dict):
        raise ValueError("JSON object required")
    for key in ("force", "live", "student_view", "stochastic", "terrain_history"):
        if key in req and type(req[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    vx = req.get("vx")
    if vx == "":
        vx = None
    # UI sends numeric input as string; accept only finite numeric strings.
    if isinstance(vx, str):
        vx = float(vx)
    validate_window(req.get("num_envs", 8), req.get("seconds", 4), 1, vx)
    task, ckpt = req.get("task"), req.get("checkpoint")
    if task not in [t for _, t in TASKS] or not isinstance(ckpt, str):
        raise ValueError("unknown task/checkpoint")
    path = (REPO / ckpt).resolve()
    if not path.is_relative_to(REPO.resolve()) or not path.is_file() or path.suffix != ".pt":
        raise ValueError("checkpoint must be a .pt under this repository")
    name = req.get("name") or f"{path.stem}_{time.time_ns()}"
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
        raise ValueError("invalid recording name")
    return {**req, "vx": vx, "name": name}


def start(req):
    try:
        req = validate_request(req)
    except (ValueError, TypeError) as exc:
        return 400, {"error": str(exc)}
    if not os.environ.get("TRAIN_ARM"):
        return 400, {"error": "server requires explicit TRAIN_ARM"}
    with lock:
        if job.get("state") == "running":
            return 409, {"error": "a recording is already running"}
        busy = gpu_busy()
        if busy:
            return 409, {"error": "GPU busy or unavailable; force override disabled", "busy": busy}
        task, ckpt = req.get("task", ""), req.get("checkpoint", "")
        if task not in [t for _, t in TASKS] or not (REPO / ckpt).is_file():
            return 400, {"error": "unknown task or checkpoint"}
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", req.get("name") or f"{Path(ckpt).stem}_{time.strftime('%Y%m%d_%H%M%S')}")
        out = RECORDINGS / (name if name.endswith(".json") else name + ".json")
        if any(p.exists() for p in (out, out.with_suffix(".log"), out.with_suffix(".live.ndjson"))):
            return 409, {"error": "recording name already exists"}
        RECORDINGS.mkdir(parents=True, exist_ok=True)
        cmd = ["apptainer", "exec", "--nv", "--writable-tmpfs", SIF, PYTHON, "tools/rl_replay/record_rollout.py",
               "--headless", "--task", task, "--checkpoint", ckpt, "--num_envs", str(int(req.get("num_envs", 8))),
               "--seconds", str(float(req.get("seconds", 4))), "--out", str(out)]
        if req.get("vx") not in (None, ""):
            cmd += ["--vx", str(float(req["vx"]))]
        if req.get("stochastic"):
            cmd.append("--stochastic")
        if req.get("student_view"):
            cmd.append("--student_view")
        if req.get("terrain_history"):
            cmd.append("--terrain_history")
        live = None
        if req.get("live", True):
            live = out.with_name(out.stem + ".live.ndjson")
            cmd += ["--live", str(live)]
        log = RECORDINGS / (out.stem + ".log")
        source = task_source(REPO, task)
        env = dict(os.environ, PYTHONPATH=str(source), OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1",
                   CUDA_VISIBLE_DEVICES=os.environ.get("RL_REPLAY_GPU", "0"))
        try:
            with log.open("x") as output:
                proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT)
        except OSError as exc:
            return 409, {"error": str(exc)}
        job.clear()
        job.update(state="running", pid=proc.pid, out=str(out.relative_to(REPO)), log=str(log), started=time.time(), cmd=" ".join(cmd),
                   live=str(live.relative_to(REPO)) if live else None)

        def wait():
            code = proc.wait()
            with lock:
                job.update(state="done" if code == 0 and out.is_file() else "failed", exit=code, ended=time.time())
        threading.Thread(target=wait, daemon=True).start()
        return 200, {"ok": True, "out": job["out"], "live": job["live"]}


def status():
    with lock:
        s = dict(job)
    if "log" in s and Path(s["log"]).is_file():
        with Path(s["log"]).open("rb") as f:
            f.seek(max(0, Path(s["log"]).stat().st_size - 65536))
            tail = f.read().decode(errors="replace")
        lines = [line for line in tail.splitlines()
                 if "[Warning]" not in line and "[omni" not in line]
        s["tail"] = lines[-12:]
    if "started" in s:
        s["elapsed"] = round((s.get("ended") or time.time()) - s["started"])
    return s


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(REPO), **k)

    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/recordings"):
            return self._json(200, recordings())
        if self.path.startswith("/api/sources"):
            return self._json(200, sources())
        if self.path.startswith("/api/record"):
            return self._json(200, status())
        if self.path.startswith("/api/live"):
            return self._json(*live_lines(urllib.parse.urlsplit(self.path).query))
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/record":
            return self._json(404, {"error": "not found"})
        host = self.headers.get("Host")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed or self.headers.get("Origin") not in (None, f"http://{host}"):
            return self._json(403, {"error": "same-origin localhost required"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 16384:
                raise ValueError("invalid length")
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeError):
            return self._json(400, {"error": "bad json"})
        code, body = start(req)
        return self._json(code, body)

    def log_message(self, fmt, *args):
        if "/api/" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    port = ap.parse_args().port
    print(f"viewer: http://127.0.0.1:{port}/tools/rl_replay/viewer.html   (repo {REPO})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
