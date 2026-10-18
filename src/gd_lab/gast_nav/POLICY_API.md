# UPDATE159 policy-side API (CPU implementation; no training launcher)

Source authority: `docs/design/GAST_NAV_V1_DESIGN.md` appendix C and the UPDATE159 review reply.
`schema.py` is unchanged. All runtime thresholds are in `configs/gast_nav/runtime.yaml`.

## Policy and migration

`GastNavActorCritic(*legacy_args, nav_privileged_dim=<explicit>, **legacy_kwargs)` inherits the real
`GastRayIrActorCritic`. Additional TensorDict keys:

- `nav_wide`: `[N,27,43,49]` (map-side schema encoding).
- `nav_near`: `[N,3,25,37]`.
- `nav_features`: `[N,16]`: mode onehot3, effective_go, v_cap, effective_turn_rate,
  phase onehot6, dwell progress, applied y_off/margin/turn_cap.
- `nav_privileged`: `[N,P]`: caller-defined explicit critic layout, width P passed to constructor.
  Actor never consumes this key. Critic has its own CNNs.

`migrate_checkpoint(policy, path, expected_sha256=...)` verifies SHA, source iteration3906,
allowlisted state expansion and strict load. Use `source_3906.yaml`; no source optimizer state is loaded.
`apply_manifest(policy, yaml_path, learning_rate=...)` returns PPO parameter groups and frozen checksums;
`assert_frozen(policy, manifest['frozen'])` detects parameter/buffer changes. CENet remains a separate
optimizer. A0 disables CENet and terrain auxiliary steps. A enables old parameters with lr×0.1.

## PPO

`GastNavPPO(policy, nav_config=<runtime['ppo']>, trainable_manifest=..., schedule='fixed', device='cpu', ...)`
constructs a manifest-filtered optimizer (A0 by default). Do not restore a legacy optimizer.
`configure_trainable(path)` changes stage and starts a fresh optimizer; checkpoint/resume orchestration
is a future runner responsibility, not strict resume from3906.

Per tick: supervisor selects controller **before** `act_selected(obs, controller_ids)`.
The returned proposal is executed only on NAV_POLICY rows. `process_env_step` requires extras:
`nav_controller`, `nav_executed_action`, `nav_override_reason` (integer reason code supplied by runner).
It rejects controller changes after proposal and modified NAV_POLICY actions. Storage retains proposals,
logprob/value, mask, executed action/controller/reason. HOLD first-tick reward is adjusted there.
Existing timeout bootstrap is inherited. GAE includes all ticks; actor advantages normalize valid rows
only. `compute_returns` then `update` implement masked actor/entropy/KL, all-row value, and global
valid-count reductions. Entirely invalid actor minibatches do not touch actor optimizer moments.
Normalization moments remain fixed through rollout/replay and update afterwards. Frozen checksums are
checked before and after each update. Nonfinite gradients abort all ranks before optimizer step.

`mirror_legacy(obs, actions, mode, env=..., privileged_mirror=...)` returns reflected LEGACY rows only
plus original indices; caller keeps original rows. It uses the native proprio/terrain mirror and reflects
new map y axes and command/y_off signs. Phase/mode are invariant. A semantic reflection callback is
mandatory for the caller-defined privileged layout. NAV/TURN are excluded. Upstream unconditional
symmetry_cfg is rejected; environment/runner augmentation must explicitly use this helper. Automatic
mirror augmentation is not silently enabled before that environment-specific contract is supplied.

## Commands and FSM

`encode(NavCommand|NavAck)` / `decode(bytes,cfg=...,now_ms=...)` use header `<4sHII>`:
NAV1/NAK1 magic, version1, payload byte length, CRC32 of canonical UTF-8 JSON. All clocks monotonic ms.
Typed dataclasses include route hint, release timestamp and request correlation. This is local IPC;
external gdm transport attachment is not part of this CPU task.

`CommandAdapter.accept(...,stationary=...)`, `.tick(...stable=...,hold=...,fsm_go=...,pre_brake_v=...)`
implement invalid/go/hold precedence and STOPPING→STABLE→ACTIVE. Duplicate requests cannot restart
mode transitions. Missing hint uses the initial frozen yaw. `omega_max_verified` is intentionally null
until measured; TURN fails closed with TURN_UNVERIFIED unless explicitly configured.

`LandingFSM.tick(now_ms,dt,command,observation,...)` consumes `LandingObservation` from the common
sensor detector interface (no GT phase inference). AUTO enters the same release queue/validator.
Release requires announce≥announce_s and continuous dwell≥announce_s+stable_after_announce_s,
plus command/hint/view/contact/go/hold checks. Defaults2.0+0.5 s are config values. go0 pauses both;
slip resets dwell and preserves announcement; hold/risk resets both. NO_DESCENT_VIEW blocks release.
ACK entries identify accepted/rejected seq/source. Reset context includes route/mode/landing/odom epoch.

## Supervisor

`Supervisor.select(HealthInput,simulation=...)` returns Decision(health,controller,fallback_action,decelerate).
INFER_FAIL always controller=DAMP; sim fallback_action=VISION_HOLD, hardware=DAMP, per the review reply.
No actuator writes occur here; RlWalker integration must execute the action and remains outside this worktree.
`observe_turn(...)` keeps positive-yaw accumulation by route epoch/manoeuvre ID across mode switches.
`predict_horizon(...)` returns future time and person inflation/residual margins, with envelope violation
flag. Caller reconstructs sample→now from odom and supplies swept geometry; this skeleton does not claim
collision detection, stopping guarantees, or a verified hold controller.

## CPU tests

Run `bash scripts/gast_nav/run_cpu_tests.sh`. No --nv, CUDA_VISIBLE_DEVICES='', tensors/device=CPU.
Only the simulator runner config import is stubbed; actual3906, terrain/CENet, rsl_rl and mirror code run.
Synthetic optimizer updates are unit fixtures, not environment training. `.runtime` holds allowed temp
files; Python bytecode writes are disabled. CPU tests do not establish GPU/DDP throughput, ONNX parity,
physical swept width, map-side integration, V0/V1 physical safety or deployment performance.
