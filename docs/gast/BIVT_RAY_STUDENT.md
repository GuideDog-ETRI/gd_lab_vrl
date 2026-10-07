# DWB-38000 → BIVT-Ray-4500 → GAST student

Requested 2026-10-01: 516 environments (not 512), BPTT 16, 20,000 camera capture
attempts. Fresh student; original actor/CENet and raycast-visible terrain teacher
are frozen. GAST clean geometry is auxiliary supervision only, not actor input.
The teacher checkpoint is verified by SHA256 in the launcher.

```bash
./gast/scripts/train_bivt4500_env516_bptt16.sh
```

Other settings: Arm4, seed 42, Adam LR .0003, checkpoint every approximately 200
captures, camera interval 70–100 ms, delay 0–150 ms, 5% packet drop. Initial
teacher rollout transitions to student rollout after 1,000 captures over 4,000
captures. Simulator root pose is used for grid-memory alignment; real odometry
and the new deployment contract require separate validation.

Output: `gast/logs/gast/arm4/bivt4500_gast_env516_bptt16_20000_20261001/`.
Preflight: `gast/logs/bivt4500_env516_preflight.console.log`.
Short capacity/compatibility checks do not establish locomotion performance or
memory headroom throughout a full run. No existing model is overwritten.

## Verified launch

2026-10-01: 516-env/BPTT16 preflight completed 32 captures in 37.96 learning
seconds and saved a checkpoint. The fresh 20,000-capture run was launched in
tmux `bivt4500-gast-516-20k`. At 18:05 KST it reached 1,536 captures; latest
saved checkpoint was `perception_1408.pt`. Measured mean was 1.06 seconds per
capture, provisional ETA 23:30 KST. This is a timestamped observation, not a
completion claim. Console: `gast/logs/bivt4500_gast_env516_20000.console.log`.
GPU memory was 29,174 / 32,607 MiB; avoid concurrent GPU workloads.
