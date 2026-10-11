#!/usr/bin/env python3
"""Compare GAST Top-5 and recent checkpoints using the same online records.

The script is read-only. It scans the large decision log on the training host,
so only a small JSON summary travels over SSH.
"""

import argparse
import json
import pathlib
import subprocess
import sys


DEFAULT_RUN = (
    "/data/users/bsseo/gd_lab_vrl_worktrees/gast-v21-20261007/gast/logs/"
    "gast/arm4/2026-10-09_08-40-05_gast_gapclean_from_bivt_train_resume8400mathsdpa"
)

REMOTE_PROGRAM = r'''
import datetime
import json
import pathlib
import pathlib
import re

request = json.loads(REQUEST)
run = pathlib.Path(request["run"])
if not run.is_absolute() or not run.is_dir():
    raise SystemExit(f"Run directory is missing or not absolute: {run}")

leaderboard_path = run / "best_top5" / "leaderboard.json"
decisions_path = run / "top5_decisions.jsonl"
if not leaderboard_path.is_file() or not decisions_path.is_file():
    raise SystemExit("Top-5 leaderboard or decision log is missing")

leaderboard = json.loads(leaderboard_path.read_text())
entries = leaderboard.get("entries", [])
if not isinstance(entries, list):
    raise SystemExit("Unexpected leaderboard format: entries is not a list")

regular = []
for path in run.glob("model_*.pt"):
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match:
        regular.append((int(match.group(1)), str(path)))
regular.sort(reverse=True)
regular = regular[:request["recent"]]

top_iterations = {int(entry["iteration"]) for entry in entries}
targets = top_iterations | {iteration for iteration, _ in regular}
decisions = {}
with decisions_path.open(encoding="utf-8") as stream:
    for line in stream:
        try:
            decision = json.loads(line)
            iteration = int(decision.get("iteration", -1))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue  # A live writer may leave a partial final line.
        if iteration in targets and isinstance(decision.get("candidate"), dict):
            decisions[iteration] = decision

print(json.dumps({
    "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run": str(run),
    "top5": entries,
    "regular": [{"iteration": i, "path": p} for i, p in regular],
    "decisions": decisions,
}, ensure_ascii=False))
'''


def collect(host: str, run: str, recent: int) -> dict:
    request = json.dumps({"run": run, "recent": recent})
    program = "REQUEST = " + repr(request) + "\n" + REMOTE_PROGRAM
    command = (["python3", "-"] if pathlib.Path(run).is_dir() else
               ["ssh", "-o", "ConnectTimeout=10", host, "python3", "-"])
    process = subprocess.run(
        command,
        input=program,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "SSH failed")
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid response from {host}: {error}") from error


def number(value, decimals=3, percent=False):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if percent:
        value *= 100
    return f"{value:.{decimals}f}"


def metric_row(label, iteration, candidate):
    candidate = candidate or {}
    metrics = candidate.get("criteria_metrics") or {}
    counts = candidate.get("sample_counts") or {}
    gap_samples = sum(
        value for key, value in counts.items()
        if key.startswith("platform_gap:") and isinstance(value, int)
    ) if isinstance(counts, dict) else 0
    return [
        label, str(iteration), number(candidate.get("score"), 5),
        number(metrics.get("gap_clean_crossing_rate"), 1, True),
        number(metrics.get("gap_crossing_rate"), 1, True),
        number(metrics.get("base_contact_rate"), 1, True),
        number(metrics.get("platform_gap_termination_rate"), 1, True),
        number(metrics.get("stairs_termination_rate"), 1, True),
        number(metrics.get("mean_terrain_level"), 2),
        str(gap_samples) if gap_samples else "—",
    ]


def render(data):
    decisions = data["decisions"]
    lines = [
        f"확인 시각(UTC): {data['checked_at']}",
        f"Run: {data['run']}",
        "",
        "| 구분 | Iter | 점수 | 클린 갭 % | 전체 갭 % | 베이스 접촉 % | 갭 종료 % | 계단 종료 % | 평균 지형 레벨 | 갭 표본 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, entry in enumerate(data["top5"], 1):
        iteration = int(entry["iteration"])
        decision = decisions.get(str(iteration), {})
        candidate = decision.get("candidate") or entry
        lines.append("| " + " | ".join(metric_row(f"Top-{rank}", iteration, candidate)) + " |")
    for item in data["regular"]:
        iteration = item["iteration"]
        decision = decisions.get(str(iteration), {})
        lines.append("| " + " | ".join(metric_row("정기", iteration, decision.get("candidate"))) + " |")
    lines += [
        "",
        "점수와 비율은 저장 시점의 온라인 후보 평가값이며, 별도 고정 시드 보행 평가가 아닙니다.",
        "갭 표본은 platform_gap 지형의 레벨별 sample_counts 합계입니다. —는 기록 부재입니다.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="onvlm2")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--recent", type=int, choices=range(5, 11), default=5,
                        metavar="5-10", help="Number of recent regular checkpoints")
    parser.add_argument("--json", action="store_true", help="Print machine-readable data")
    args = parser.parse_args()
    try:
        data = collect(args.host, args.run, args.recent)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else render(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
