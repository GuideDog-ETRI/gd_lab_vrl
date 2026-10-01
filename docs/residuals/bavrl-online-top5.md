# BAVRL online Top-5 — 2026-09-30

Upstream `49c4db5` was fast-forwarded without undoing the local layout migration.
The upstream change raises the three termination-rate gates from 6% to 9%;
the per-iteration ranking implementation was already present in the common RL layer.

## New layout and integration

- Shared scoring, selection and atomic publication: `src/gd_lab/rl/online_top5.py`.
- Pre-reset completed-episode capture: `src/gd_lab/rl/online_rollout.py`.
- Reloadable criteria: `configs/online_top5.json` (or `--top5-criteria`).
- BAVRL integration/status logging: `src/gd_lab/residuals/bavrl/online_quality.py`.
- Trainer: `scripts/train_bavrl.py`; tests: `tests/test_bavrl_online_quality.py`.

Each PPO iteration drains completed rollout episodes and invokes the shared scorer.
`quality_samples.jsonl` records every iteration, including missing gap samples,
failed gates, score, criteria and checkpoint selection status. TensorBoard records
`Quality/S_online` when available. This is an online training-rollout proxy, NOT
held-out evaluation. As in the upstream runner, rollout statistics select the
post-update checkpoint; they are not a separate evaluation of its updated weights.

`best_top5/leaderboard.json` and `<iteration>_top<rank>.pt` are atomically published.
Files contain the BAVRL model, optimizer, teacher hash, iteration and deployment
context, and can be used with `scripts/export_bavrl.py`. The original teacher is
still required for export. Never substitute another method's checkpoint.

Current gates: mean terrain level >=9, mean gap level >=8, overall/gap/stairs
base-contact rates <=9%, gap success regression <=2 percentage points and
base-contact worsening <=1 percentage point. Ranking uses family/level macro means,
not a raw global episode fraction. Selected checkpoints have minimum spacing100.
Missing gap/stairs episodes or low curriculum levels may leave Top-5 empty,
especially with32env/horizon16 after a curriculum reset. No thresholds were relaxed
locally. Periodic50-iteration checkpoints remain available but are not Top-5.

SIGINT/SIGTERM now finish the current iteration, save a checkpoint and exit.

## Resumed run

Source: `logs/bavrl/dwb38000_bavrl_20000_resume1000_20260930/bavrl_4400.pt`.
New run: `logs/bavrl/dwb38000_bavrl_20000_top5_resume4400_20260930/`.
Target: total20000, additional15600;32env,horizon16,epochs4,batch32.
Model/optimizer resume; physics, curriculum, sensor queues and episode statistics
start fresh. Old run and its checkpoints are preserved. No Git push.

```bash
bash scripts/train_bavrl_5090.sh dwb38000_bavrl_20000_top5_resume4400_20260930 \
  --num_envs 32 --iterations 15600 --horizon 16 --epochs 4 --batch-size 32 \
  --resume logs/bavrl/dwb38000_bavrl_20000_resume1000_20260930/bavrl_4400.pt
```
