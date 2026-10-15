# UPDATE54 — optional belief terrain supervision

UPDATE51 files and artifacts are unchanged. This directory records their hashes
and the small deltas in the copied UPDATE54 runner/entry points. No git metadata
exists in this copied worktree.

New training entry: `scripts/gast/train_student_live_update54.py`.
New prepared remote launcher: `run_v3_update54_90.sh`.
`--v3_aux_lambda` defaults to **0**; launcher uses `${V3_AUX_LAMBDA:-0}`.
The original `run_v3_90.sh` continues to run UPDATE51, without UPDATE54 contracts.
No remote launcher was executed here.

At zero the auxiliary object has no head/parameters, consumes no RNG, and returns
the exact original loss tensor. At positive lambda a separate 128-wide MLP reads
post-tick belief and predicts 187 x 5 terrain channels. The model state dict and
inference graphs do not acquire auxiliary keys. Checkpoints store the head in
`belief_aux`, separately from `model`; the UPDATE51 exporter remains usable.

Only the latest accepted capture per row in this tick is eligible; old episode,
reordered, duplicate, future, stale and absent packets provide no auxiliary
supervision. The current execution group determines auxiliary prep4 weighting.
Targets use inverse yaw-grid bilinear sampling and scanner-height datum shift.
Every height interpolation contributor must be visible, finite and unsaturated;
out-of-grid and invalid rows are masked. Edges and local level support are
recomputed in the current grid using visible neighboring cells. Visibility
means availability from the delivered capture, not a new current camera view.
Support does not mean safe foothold. No privileged target enters inference.

Per-row loss is the mean of five channel losses, each normalized by its masked
weighted cell sum (clamped to 1). Positive up/down cells weigh 5. Row losses use
the existing prep4 GroupMean. Logs include nominal 20 versus 5 multipliers and
actual positive-edge coefficient min/max after lambda, group counts, five
channels and per-row edge denominators. Raw numerator weights alone are not
normalized gradient coefficients.

CPU reproduction (no GPU, simulator or optimizer steps):

```bash
bash tools/update51/cpu.sh tools/update54/verify_cpu.py
bash tools/update51/cpu.sh tools/update54/verify_onnx.py
bash tools/update51/cpu.sh tools/update54/verify_runner.py
```

Outputs: `outputs/update54_cpu/`. `verify_runner.py` uses actual runners and the
GAST2000 actor with a deterministic fake host; it replaces optimizer.step with a
no-op. This is not a simulator smoke test or a training performance result.
