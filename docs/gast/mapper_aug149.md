# UPDATE149B protocol (supersedes UPDATE149)

No GPU training/smoke has been run by Codex. Claude launches after review, from the prepared aug149 worktree with server-specific `GD_LAB_SIF` and `GD_LAB_PYTHON` configured:

```bash
# onvlm2, GPU0 and GPU1 respectively
bash scripts/gast/train_aug149_A2769.sh smoke
bash scripts/gast/train_aug149_C2769.sh smoke
# 90.111, GPU0
bash scripts/gast/train_aug149_A3906.sh smoke
```

Smoke:64 env ×16 steps ×5 updates, local mapper ramp32 steps. Then replace `smoke` with `train` for1024 env ×100 steps ×2000 ADDITIONAL updates, mapper ramp50,000 steps. GPU3 is never exposed. `AUG149_DRY_RUN=1` validates the source SHA and prints the command without launching. Host placement is the caller's responsibility.

All arms now use `--resume_rayir149`: exact source SHA/task, full native model, PPO and CENet optimizer states/LR/groups, source iteration+1. `resume149.py` validates source metadata; after runner.load it compares every model/optimizer tensor and group value with the source and refuses any mismatch. No warm-start fallback. Source2769 starts2770 (last trained index4769 after2000 updates); source3906 starts3907 (last5906). Source horizon is independently verified as100, including the16-step smoke. Absolute noise offsets are277000/390700. Force/speed use the explicitly requested mature-source full-strength protocol (ramp settings0: force scale1, speed ceiling1.2); noise is already1. Global camera/history/reset counters remain local. Only the NEW mapper augmentation ramps0→500 local updates.

Every family must have saved level metadata. Floor(saved mean) and matching terrain origins are checked; artificial startup resets temporarily suppress curriculum computation, then restore the manager method and verify levels/origins stayed identical. Ordinary later episode resets resume normal curriculum. This restores family-level means, not exact per-env physical state.

25% clean episodes means unchanged Ray+IR baseline with noise/dropout/visibility/blackout, not GT. Parameters use a private RNG and persist per episode. Default switch OFF preserves baseline path/RNG. Source tasks/rewards/terrain proportions are unchanged.

Geometry: original center plus four±2.5cm S0 GT rays; four extra queries have independently sampled0–7.5cm near/far residual plus±1.25cm phase. Clamp the combined offset (S0 INCLUDED) to10cm Euclidean radius. No added stress tail. Sampled dilation can still miss narrow pits; this is not a fitted mapper-error distribution or full continuum morphological dilation. Gap-family/support tests are internal only. DDEEP replaces qualifying pit z with support−.7m at episode probability.5. Stair down steps do not get DDEEP.

Live order: geometry metric z → existing clip/noise/bias/dropout ONCE → extra unknown → held capture → blackout/history. Invalid height/IR are zero. Critic/clean reconstruction targets remain independent. IR uses the original GT hit/material/visibility cue: no false-gap darkening or visibility expansion.

Unknown: episode-fixed blob and **map-border band** (not camera-image border). Per-env removal cap=floor(currently_valid × uniform[.05,.15] × ramp). Deterministic grid-order truncation keeps the cap; sparse candidate masks may remove less than5%, including0. A cap is not a forced minimum. New diagnostics record mean/max realized removal and actual newly lowered ground cells before/after validity. Query radius is a geometric bound, not a measured edge-shift statistic. Temporal counters track longest held-frame extra-unknown run and total unknown run including baseline visibility/dropout/blackout, reset per env and retain the run maximum.

Top-5 start uses local+500; episode collector excludes episodes whose start was before full mapper strength, including the control to keep comparison consistent. Rank gates and spacing500 remain unchanged. Every local+250 checkpoint is explicitly retained (`local_0250.pt` through `local_2000.pt`); original periodic/final saves also remain. Top5 is capacity, not a guarantee of five qualifying candidates. E1 final selection remains Claude's task; no automatic gate relaxation.

Smoke stdout/completion and `aug149_updates.jsonl` provide source SHA and optimizer step/LR summaries, exact loaded-state checks, restored family levels, actual force/speed/noise values, local mapper strength, widened visible/output cell counts, unknown fractions/run durations, finite frame/loss/parameters, per-update wall seconds, peak allocated GPU bytes. A2769 versus C2769 provides ON/OFF measurements. A3906 has no matched control arm. GPU values are unmeasured until Claude runs smoke. Full1024-env throughput cannot be inferred reliably from64-env smoke.

Additional ray budget remains8×187×1024=1,531,904 rays/capture, nominal25 and worst staggered100 captures per100-step update (38.3M–153.2M extra rays). Diagnostics add synchronization/reductions; measure actual overhead. CPU synthetic preview timing is not a GPU estimate. Fixtures and tests are under `archive/update149_validation`, with UPDATE149B logs distinguished by149b suffix. Optional accumulated-mask mixing/stale GAP memory remain deferred.
