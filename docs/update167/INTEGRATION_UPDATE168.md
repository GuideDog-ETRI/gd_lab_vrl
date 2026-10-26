# UPDATE168 producer integration readiness

The five new files from `update168_from90/INTEGRATE_ONLY168.txt` are imported.
Their manifest and camera/robot/URDF dependency hashes were verified. The live
adapter is `isaac_producer168.py`; the original reference consumer remains intact.

## Prepared diagnostic entry point

```
GAST_RUN_ID=update168_easy_c_<unique> bash scripts/gast_nav/diagnose_easy_c168.sh
```

Defaults are fixed at 64 environments, 600 policy ticks, seed 168, easy C geometry,
correctly scaled frozen 3906, measured envelope lookup, no optimizer updates.
`resolved_config.json` records the runtime configuration. `result.json` includes
an explicit t>=3 s gate over windows 3/4/5. This entry point uses `producer168`, preserving current-hit occupied veto and
requiring the final CPU JUnit counterexamples for a positive gate.
This is an integration baseline, not a command that secretly enables synthetic
free volumes. Previous raw diagnostic flags are retained for comparison only;
legacy R1/R2 relaxations cannot bypass the measured certificate.

## Adapter contract and conservative first integration

* The producer owns simulation sensor queries for supported scene primitives and
  actual robot collision geometry. Unsupported or missing geometry invalidates
  the guarantee. Full-block conservative optical-z first-hit lower bounds and
  explicit volume_valid must come from this producer; point depths are insufficient.
* Camera K, camera-to-current-yaw transforms, block rectangles, depth lower bounds,
  capture timestamp, and producer qualification/version must be recorded together.
* Invoke frusta_from_d4xx and retain bounded wide/fine FreeEvidenceHistory stores
  (support grids 43x49 and extended 45x57 respectively). Copy measured support;
  never substitute scene floor height. Use mapper epochs and clear/reset both stores.
* Supply cumulative travelled-distance bounds, not net displacement, and explicit
  delay linear/yaw bounds. Dynamic history defaults null. Only closed, static C
  fixtures may set historical_obstacle_speed_bound_mps=0, with a run-config entry.
* Merge through ray_evidence/build_near with current ray first-hit clipping and
  occupied veto. Preserve source age, and expire B1/B2 proofs between captures too;
  simply holding a previously valid TTL-edge certificate is not sufficient.
* Keep actor/critic input layout unchanged. Never use geometry oracle occupancy
  directly for supervisor certification, nor infer valid volume from observed ratio.

## Acceptance before C READY

Verify supplied manifest and diffs; report source collisions rather than overwrite.
Run imported counterexamples 1-6 and the complete CPU suite. Verify independent
producer fixtures (subpixel/edge/near/far/thin pillar/self/wall+person) and supported
shape coverage. Benchmark producer and complete proof separately; producer-only
speed does not establish total 4096-env capture cost.

Run the prepared 64-env diagnostic with the real adapter. Require completed 600
ticks, t>=3s HOLD <=30%, progress>0, current producer counterexamples passed, finite
observations, and no unsupported envelope falsely certified. Record NAV-only and
all-env denominators. Preserve empirical envelope limitations: flat stopped subset,
per-cell occupancy percentile rather than a pathwise coverage guarantee. Three
fixed degrees of TURN overshoot are provisional and are not a validated bound.

## Implemented limitations

The initial live adapter uses current-capture evidence only (no history reuse),
and invalidates certification between captures. It records all generated NAV
mesh boxes including floors through a sensor-only inventory. Unsupported legacy
tiles and unrepresented outer border within a conservative camera reach sphere
invalidate the entire environment; these are not declared empty. All supported
world NAV primitives and live robot link collisions are queried, plus the own-env
live cylinder. The inventory is never a policy/support map input.

P2 delay linear/yaw working bounds are 5 m/s and 10 rad/s in this diagnostic,
with current observed violations closing the capture. These deliberately large
diagnostic assumptions are not a measured continuous-time deployment guarantee.
No reduction of the original P2/support/range uncertainty is applied.

C-only self_trunk_aperture is approved in review, with user confirmation pending;
it is not described as user approval. Default producer168.yaml remains strict.
The current-only adapter and high delay bounds can reject otherwise potentially
usable historical evidence; negative HOLD results do not prove the producer
cannot improve observation under a separately validated temporal adapter.
