# UPDATE162 pipeline optimization

The training authorization is unchanged: remote onvlm2 GPUs 0/1/2 only, A0 pilot
200 iterations, after Claude commits and syncs. Local measurements are fixed-3906
inference with no optimizer steps. Do not treat local rollout time as PPO iteration time.

## Changes

- Capture-aligned actor maps are held between 10Hz captures. Health age still
  advances each control tick. A reset invalidates only its held rows (UNKNOWN),
  preserving earlier rollout tensor references, until the next scheduled capture.
  It no longer forces a whole-batch capture at every individual reset.
- `map_cuda.py` enumerates dominant-axis ray slabs plus three minor-axis candidates,
  then applies the original exact slab/strict-volume/sub-bin checks. No sensor rays,
  band boundaries or sub-bins are dropped. CPU `ray_evidence_reference` remains.
- CUDA clearance reduces exact distances to occupied cell boundaries within the
  clipped radius. CPU `clearance_reference` remains. All tested outputs are bit equal.
- Support fitting fuses per-cell neighbourhood moments. Heights and observed masks
  are exact; fitted normals allow absolute 1e-4 reduction-order tolerance. The same
  torch eigensolver runs on observed candidates, in chunks <=32768 to avoid a
  cuSOLVER invalid-batch error seen when all 1366-env cells were solved at once.
- `BatchedOracle` preloads immutable scene geometry once and computes critic/reward
  route projection, width/rotation feasibility and clearance on the GPU. Comparison
  tolerance vs the float64 scalar geometry is 1e-5m; tested feasibility bits match.
  Oracle data is not fed into actor maps.
- `NavState.prepare` batches sensor evidence and outputs. Stateful command/FSM/
  supervisor logic remains unchanged on CPU, receiving one batched feature transfer
  and one epoch transfer; per-env CUDA scalar reads/writes were removed.
- Reset spawn/axis metadata is cached by tile. Scene YAML is loaded once, rather
  than once for each oracle call. Remaining Python protocol loops are intentional.
- Launcher places Triton/CUDA caches inside the launch record. A late SIGUSR1 during
  teardown no longer restores its default terminating handler. Runner errors write
  partial timing/optimizer diagnostics before teardown where possible.

## Verification

- CPU: `bash tests/gast_nav/run_cpu.sh` (238 pass, 7 GPU-only tests skip).
- GPU: pytest `test_maps.py test_cuda_random162.py -p cuda_map_plugin` (98 pass).
  The plugin moves the unchanged golden/counterexample ray/clearance calls to CUDA
  and compares states, sub-bin diagnostics and clearance bit-for-bit to CPU.
- Randomized CUDA rays include parallel axes/self hits/unknown support; plane
  moments and final support code/connectivity are also compared.
- Full CPU suite must run separately from CUDA plugin. `test_env160.py` explicitly
  rejects an initialized CUDA context; combining both in one pytest process is an
  invalid test invocation (15 setup errors), not a reason to weaken that guard.
- `test_batched_oracle.py` checks scalar geometry equivalence and partial-reset
  UNKNOWN invalidation without modifying stored observations.
- Performance and tolerance contract: `configs/gast_nav/performance162.yaml`.

## Profiling

`GAST_RUN_ID=<unique> bash scripts/gast_nav/profile_pipeline.sh --envs 1366 --steps 100`

Add `--profile` for inclusive function wall/CUDA-event timing. These CUDA event
intervals include host dispatch gaps; they are not isolated GPU kernel durations.
`--snapshot` saves the initial actual sensor capture for offline equivalence work.
No original smoke160B marker is reused. Local profile output is under
`logs/gast/nav_v1/<run_id>/`. GPU3 is never part of the remote device set.

Baseline 16 env/20 steps: 6.276s; ray evidence ~1.816s, scalar oracle ~2.259s.
Optimized 1366 env/100 steps: 27.994s with inclusive profiling, 23.567s without.
The 1366 profiled run had exactly 10 captures, pipeline capture 0.866s total,
physics 1.155s total, PyTorch peak reserved 3.590GiB. PPO/DDP/rollout storage are
excluded from these inference measurements and require the remote pilot.
