"""Local server for tools/rl_replay/viewer.html: serves the repository (URDF, meshes, recordings) and runs
record_rollout.py on request, so recording, opening and playing all happen in the viewer.

  python3 tools/rl_replay/server.py [--port 8765]      then open http://127.0.0.1:8765/tools/rl_replay/viewer.html

API (JSON, 127.0.0.1 only):
  GET  /api/recordings              recordings in logs/rl_replay/ (newest first, with their meta)
  GET  /api/sources                 tasks and teacher checkpoints that can be recorded
  POST /api/record                  {task, checkpoint, num_envs, seconds, vx, stochastic, name, force, live, student_view}
  GET  /api/record                  state of the current/last recording job (log tail, output, exit code)
  GET  /api/live?path=..&from=N     lines N.. of a live stream (*.live.ndjson) that a recorder is writing
One recording at a time. It refuses while a training/distillation process holds the GPU unless force=true.
"""
import argparse
import urllib.parse
import json
import os
import re
import subprocess
import threading
import time
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
BUSY = re.compile(r"train_student_live\.py|train_student\.py|distill_student\.py|train_teacher\.py|train_bivt\.py")
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
    start = int(q.get("from", ["0"])[0])
    with path.open() as f:
        text = f.read()
    lines = text.split("\n")[:-1]  # the last element is "" or an unfinished line
    return 200, {"lines": [json.loads(x) for x in lines[start:]], "next": len(lines)}


def sources():
    ckpts = sorted(str(p.relative_to(REPO)) for p in (REPO / "checkpoints/teachers").rglob("*.pt"))
    return {"tasks": [{"label": a, "task": b} for a, b in TASKS], "checkpoints": ckpts}


def gpu_busy():
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return [line.strip()[:120] for line in ps.splitlines() if BUSY.search(line) and "server.py" not in line]


def start(req):
    with lock:
        if job.get("state") == "running":
            return 409, {"error": "a recording is already running"}
        busy = gpu_busy()
        if busy and not req.get("force"):
            return 409, {"error": "GPU is busy with training; tick 'force' to record anyway", "busy": busy}
        task, ckpt = req.get("task", ""), req.get("checkpoint", "")
        if task not in [t for _, t in TASKS] or not (REPO / ckpt).is_file():
            return 400, {"error": "unknown task or checkpoint"}
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", req.get("name") or f"{Path(ckpt).stem}_{time.strftime('%Y%m%d_%H%M%S')}")
        out = RECORDINGS / (name if name.endswith(".json") else name + ".json")
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
        live = None
        if req.get("live", True):
            live = out.with_name(out.stem + ".live.ndjson")
            live.unlink(missing_ok=True)
            cmd += ["--live", str(live)]
        log = RECORDINGS / (out.stem + ".log")
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"), TRAIN_ARM="4", OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)
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
        lines = [line for line in Path(s["log"]).read_text(errors="replace").splitlines()
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
        if not self.path.startswith("/api/record"):
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
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
