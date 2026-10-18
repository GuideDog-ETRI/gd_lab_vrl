# UPDATE161 A/B/C integration

Shared `SceneTileMeta`, `StaticBox`, `DynamicCapsules` and scene generator callback
signatures are unchanged. Copy the changed terrains.py / moving_cylinder.py,
new turn_clutter.py, and updated configs/gast_nav/scenes.yaml together.

## Turn geometry

`build_scene('nav_turn_room', difficulty, cfg, seed=...)` now constructs a 3--5m
clutter workspace, with 4--12 chair boxes, square pillars and blocks. Optional
perimeter walls are separate from the clutter count. A chosen spot has exact
minimum obstacle-surface distance `turn_radius + reference_margin + delta(level)`.
The default reference margin is 0.10m; runtime condition margin can differ and
must still be passed separately to the oracle. Difficulty controls delta rather
than room dimensions. Level is the existing clamped floor(difficulty*levels).

Route is entry -> spot. Entry also identifies the reserved person corridor;
`cylinder_slot(meta, config)` recovers its perpendicular nominal path without
adding a shared metadata field. The path, maximum cylinder radius, safety
clearance and max sway offset are all protected during clutter placement.

`turn_free_polygon(meta, config=..., samples=None)` is a visible-free debug
polygon. Rays end at actual OBBs or the declared visualization radius. It is not
the complete connected free component or a substitute safety oracle/map input.
The GT polygon and obstacle list must not bypass sensor first-hit processing.

## Sway API (environment task E)

```python
# Once per physics tick, after computing the train profile in vehicle axes:
cylinder.update(dt, a_train_xy)  # a_train_xy: torch.Tensor[N,2], m/s^2
cylinder.step(dt, robot_positions_world, robot_velocities_world)
cylinder.write_colliders(env.scene['nav_cylinder'])
# Build optical capsule only from cylinder.capsules() AFTER accepted motion.
```

`update()` is filter-only. It returns desired offset[N,2] and never moves collider
or advances the nominal path. LPF is exact first-order discrete integration:
`alpha=-expm1(-2*pi*cutoff_hz*dt)`. Offset `-gain_s2*a_lp` has a Euclidean norm cap.
Default gain=0.08s², cutoff=0.8Hz, max offset=0.12m; all are YAML settings.
Fields `a_train_lp`, `sway_offset`, `applied_sway_offset` are available for logs.
Axes are vehicle-floor xy. Current env frames share world axes, with translation
only through env_origins. A rotated vehicle needs an explicit vector rotation.

`step()` progresses the NOMINAL path, adds the desired absolute offset, checks
the entire relative swept segment against robot motion and static OBBs, and only
then commits position/velocity. A blocked tick freezes both progress and applied
offset. LPF can continue evolving. Recovery is checked as another full segment;
no hidden nominal progress or teleport. Endpoint reversal uses nominal coords.
Per-tick static checks are batched; geometry cache refreshes on SceneIndex binding
changes. Reset clears LPF and both offsets only in reset env rows.

No `update()` call -> zero-offset backward-compatible behavior. `speed` remains
nominal route speed; `velocity` is ACTUAL accepted motion including sway. Any
safety-envelope speed bound must include sway, not just nominal 0--0.8m/s. With
|train jerk|<=10 and gain0.08, a conservative additional speed bound is 0.8m/s
(for zero-initialized LPF and bounded profile), before checking runtime limits.

`EpisodeCylinder` already owns approved skip/parking/counters. It is unchanged.
Do not classify its blocked/initial_overlap proxy as a PhysX person-contact
termination: task D uses actual link/collider contact forces and its own threshold.
Tasks D/E, contact termination and env.py wiring were not modified by Codex90.

## CPU evidence

Run `tests/gast_nav/run_scenes161_cpu.sh`. It executes scene/map/environment and
wiring CPU regressions, excludes policy fixtures that perform optimizer steps,
and uses no simulator/GPU. Staging overlay is test-only, not copied into gd_lab
root or installed into the shared venv. GPU/Isaac throughput is not verified.
See .runtime/turn161_stats.json for all level denominators and rejection counters.
