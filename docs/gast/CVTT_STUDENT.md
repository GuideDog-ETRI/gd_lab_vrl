# CVTT-7761 → GAST student, 20,000 captures

This hybrid experiment does not train a GAST teacher. The fixed CVTT-7761
teacher retains its original camera-visible terrain encoder, actor and CENet.
Only the GAST student receives optimizer updates. The separate task adds clean
geometry targets for auxiliary supervision without replacing teacher inputs.

Student: camera CNN + grid cross-attention, pose-aligned cell GRU, temporal
attention and 32-D latent. Pose is simulator root pose; real odometry drift is
not covered. Missing/stale/blur supervision uses the teacher's zero-terrain
response, not an assurance of equivalence to DWB-38000.

Settings: Arm4, seed 42, 16 envs, learning rate 0.0003, BPTT 64, capture interval
70–100 ms, delivery delay 0–150 ms, packet drop 5%, checkpoint interval 200
(rounded to the BPTT window boundary). Goal: 20,000 capture attempts, not PPO
updates or optimizer steps. Student rollout starts after 1,000 captures and
ramps over 4,000. Extra corruption includes blur and camera dropout.

```bash
./gast/scripts/train_cvtt7761_student_20k.sh
```

This launcher verifies the fixed teacher SHA256 before starting. Existing GAVD
weights are not reused: this student is trained from scratch. The new hidden
state/pose contract requires a separate deployment exporter before MuJoCo use.
Do not use the legacy GAVD exporter. No Git push is performed by this launcher.

## Requested restart: 1024 env / BPTT 16

The original 16-env/BPTT64 run was stopped at the user's request; its saved
`perception_3200.pt` is preserved (unsaved progress was discarded). The new run
starts from scratch, target 20,000 captures, via:

```bash
./gast/scripts/train_cvtt7761_env1024_bptt16.sh
```

The corresponding BIVT-Ray-4500 → GAVD launcher is
`scripts/train_bivt_ray4500_gavd_env1024_bptt16.sh`. Both use seed 42, LR .0003,
70–100 ms capture interval, 0–150 ms delay and 5% packet loss. Their teachers and
student objectives differ, so this is not an isolated architecture ablation.
1024 env capacity has NOT yet been validated. Do not run both concurrently.
Increasing from 64 to 1024 env also increases samples per capture attempt 16x;
equal iteration counts do not imply equal data or computation budgets.
