# UPDATE160 scene API / integration

Owned files: scene_types.py, scene_config.py, scene_geometry.py, terrains.py,
feasibility.py, moving_cylinder.py; settings: configs/gast_nav/scenes.yaml.

## Coordinates and Isaac integration

- `build_scene(kind,difficulty,cfg=None,seed=...)` returns `(meshes,origin,meta)`.
  Mesh coordinates are `[0,size.x] × [0,size.y]`, origin is `(size.x/2,size.y/2,0)`.
  ALL metadata coordinates are relative to that origin (not lower-left corner).
- Four `nav_*_terrain(difficulty,cfg)` functions return `(meshes,origin)` exactly
  like existing `platform_gap_terrain`. Metadata is attached to the first mesh,
  but plain Isaac concatenation/OBJ caching must not be assumed to preserve it.
- After starting the Isaac app, call `make_isaac_cfg(kind,...)` for real
  SubTerrainBaseCfg instances, and set generator `class_type` to the class
  returned by `make_isaac_generator_class()`. Use `use_cache=False`. The subclass
  retains `scene_grid[(row,col)]` with the correct terrain family and geometry.
  Existing legacy terrain callbacks are delegated to the base unchanged.
- The supplied size is 16×12m. A smaller size may not fit the full 6m corridor,
  two approaches, or stair flights. Builders reject cropping rather than silently
  overlap neighboring tiles. Choose one common size for legacy+Nav in the owner env.
- Bind after assignment with `SceneIndex.from_grid(generator.scene_grid,
  terrain_levels,terrain_types)`; call `.bind` when curriculum changes env tiles.
- Spawn at `.tensors()['spawn_xy']`: route start, or turn_spot centre. Add actual
  `env.scene.env_origins`; do not change env origin to the route start.
- Metadata is for command emulation, critic/reward and oracle. Actor sensor maps
  must still be built from ray hits using the map pipeline.

## Tensors

`SceneIndex.lookup(env_ids) -> list[SceneTileMeta]`, `.tensors(device)` returns
kind/level, padded route_points/route_mask, width_profile/width_mask,
segment_type/segment_bounds/segment_mask, landing/landing_mask,
turn_center/turn_radius/turn_mask, route_axis_yaw, available_width,
route_length, train_inertia, spawn_xy. Geometry stays env-local.

## Oracle

`passage_feasible(meta,position_xy=None,swept_width_m=None,margin_m=None,config=None)`
uses minimum remaining obstacle-reduced width. `turn_feasible(...,
turn_swept_radius_m=None,...)` uses room clearance at the actual position.
Both return FeasibilityResult(feasible,available,required,remaining_clearance,reason).
Thresholds: temporary W=0.50m, turn radius=0.65m PLUS margin. No tolerance is added
to the feasibility inequality. Dynamic cylinder contacts and sensor unknown are
separate environment inputs; these functions are the STATIC scene width oracle,
not proof of full body dynamics, braking or gap/stair traversability.
`batch_feasibility(index,positions_xy,turn_mask,config=None)` is a CPU-first
reference adapter; no 4096-env GPU throughput claim is made.

## Cylinder

`MovingCylinder(index,stage='C',seed=0,device='cpu',env_origins=None,config=None)`
maintains one cylinder per enabled non-legacy C env, no cylinder in A0/A/B.
`reset(env_ids=None,stage=None,robot_positions=None)` samples dimensions and a safe
static path. Without robot_positions it excludes the metadata spawn location.
`step(dt,robot_positions,robot_velocities=None)` tests continuous relative motion
and stops for the whole tick before collision; blocked/initial_overlap distinguish
prediction from preexisting overlap. Straight endpoints hold; pingpong reverses.

`capsules()` returns DynamicCapsules in WORLD coordinates. Transform both segment
endpoints to the sample body-yaw frame before passing them to map ray intersections.
The capsule conservatively encloses the flat-ended physical cylinder and extends
one radius above/below it; its physical cylinder height remains 1.7m.
`collider_specs()` supplies radius/height/kinematic flags. On reset the env must
create or scale cylinder collider GEOMETRY to these values. `write_colliders(obj)`
then writes root pose WXYZ and actual velocity through Isaac RigidObject methods.
Do not merely change the observation radius while leaving collider size unchanged.

`NoCylinderPathError` is an explicit initialization rejection, with env_id/kind/
level/radius_m/attempts diagnostics. A cylinder is never silently disabled to make
a C scene pass. The owning env must record rejected initial geometry separately
from policy contact failures; resampling policy requires review confirmation.

## Validation

`tests/gast_nav/run_scenes160_cpu.sh` runs only new scene fixtures and existing map
regressions. Existing policy tests contain optimizer steps and are intentionally
not run by UPDATE160. Optional GPU/Isaac creation smoke was not run.
