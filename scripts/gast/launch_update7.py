"""Launch one UPDATE7 prescription; forward existing trainer arguments.

Examples:
  python scripts/gast/launch_update7.py a --dry-run -- \
    --teacher_checkpoint TEACHER --student_resume STUDENT \
    --iterations 30000 --perception_run_name update7_a

  python scripts/gast/launch_update7.py b --weight 4 -- \
    --teacher_checkpoint TEACHER --student_resume STUDENT \
    --iterations 30000 --gap_loss_weight 2 \
    --perception_run_name update7_b

  python scripts/gast/launch_update7.py c --weight 4 -- \
    --teacher_checkpoint TEACHER --student_resume STUDENT \
    --iterations 30000 --gap_loss_weight 2 \
    --perception_run_name update7_c

TRAIN_ARM and other existing environment variables are inherited.
iterations remains a total target, as in train_student_live.py.
No prescription is selected automatically from a diagnostic.
"""
import argparse
import math
import shlex
import subprocess
import sys
from pathlib import Path


CONTROLLED = {
    "--gap_ghost_probability",
    "--hidden_gap_latent_weight",
    "--ahead_gap_latent_weight",
    "--gap_action_weight",
    "--gap_observed_supervision",
}


def numeric_option(arguments, name, default):
    value = default
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item == name:
            if index + 1 >= len(arguments):
                raise ValueError(f"missing value for {name}")
            value = float(arguments[index + 1])
            index += 1
        elif item.startswith(name + "="):
            value = float(item.split("=", 1)[1])
        index += 1
    return value


def prescription_flags(kind, weight, base_gap_weight):
    if kind == "a":
        return ["--gap_ghost_probability", "0", "--gap_observed_supervision"]
    if kind not in ("b", "c"):
        raise ValueError("prescription must be a, b or c")
    if not math.isfinite(weight) or weight <= base_gap_weight:
        raise ValueError(
            "--weight must exceed existing --gap_loss_weight; "
            "new weights are absolute floors"
        )
    option = (
        "--hidden_gap_latent_weight"
        if kind == "b" else "--gap_action_weight"
    )
    return [option, str(weight)]


def build_command(kind, weight, executable, forwarded):
    forwarded = list(forwarded)
    for item in forwarded:
        option = item.split("=", 1)[0]
        if option in CONTROLLED:
            raise ValueError(
                f"{option} is controlled by this launcher; "
                "use the trainer directly to combine prescriptions"
            )
        if option == "--replay_out":
            raise ValueError("prescription launcher requires training mode")
    if kind == "a" and "--no_camera_noise" in forwarded:
        raise ValueError(
            "prescription a keeps generic sensor noise enabled; "
            "remove --no_camera_noise"
        )
    base_gap_weight = numeric_option(
        forwarded, "--gap_loss_weight", 1.0
    )
    if not math.isfinite(base_gap_weight) or base_gap_weight < 1:
        raise ValueError("--gap_loss_weight must be finite and >= 1")
    trainer = Path(__file__).resolve().with_name("train_student_live.py")
    return [
        executable,
        str(trainer),
        *forwarded,
        *prescription_flags(kind, weight, base_gap_weight),
    ]


def main():
    raw = sys.argv[1:]
    if "--" in raw:
        boundary = raw.index("--")
        launcher_arguments = raw[:boundary]
        forwarded = raw[boundary + 1:]
    else:
        launcher_arguments, forwarded = raw, []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prescription", choices=("a", "b", "c"))
    parser.add_argument("--weight", type=float, default=3.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(launcher_arguments)
    try:
        command = build_command(
            args.prescription, args.weight, args.python, forwarded
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
