# UPDATE161 environment contract

`person_contact.py` is simulator independent. `PersonContactTracker.observe`
consumes per-body, NavCylinder-filtered world contact force histories [N,...,3].
Each vector is thresholded separately; neither opposing links nor physics steps
are summed. `classify` is read-only; `commit` counts once until selected-row reset.

Isaac configuration adds one ContactSensor per robot rigid link (including feet),
filtered to that environment's NavCylinder, at physics_dt with decimation history.
`NavState` fails explicitly if actual robot body names differ from the configured
coverage list. Termination is evaluated in the same policy step after physics
substeps, without waiting for another policy action. Contact and invalid masks
are disjoint. A transient contact in an earlier substep survives in sensor history.

Explicit invalid reasons: INITIAL_OVERLAP, NONFINITE_STATE, ENGINE_FAILURE,
PERSON_MODEL_INTRUSION, NONFINITE_CONTACT. A later `MovingCylinder.initial_overlap`
flag or proximity/blocked status is NOT accepted as proof of engine invalidity.
Reset performs an actual PhysX overlap_shape query against robot collider paths,
using a separate non-colliding USD cylinder with the sampled physical dimensions.
Articulated kinematics are refreshed after reset poses and before the query.
This avoids both inflated proximity labels and stale USD poses for tensor-written
kinematic cylinders. Query evaluation is only at reset, before the first action.
The tracker rejects INITIAL_OVERLAP labels after begin_action().

Additional reset diagnostics can use `tracker.reset(ids, initial_overlap=...)`, or
`tracker.mark_invalid(ids, reason)` before termination evaluation. Already-counted
policy failures cannot be relabelled retrospectively. Env extras preserve the
pre-reset person_contact/invalid_code plus cumulative counts and reason totals.

Scene owner API for E:

```python
MovingCylinder.update(dt, a_train_xy)  # [N,2], fixed vehicle-floor coordinates
MovingCylinder.step(dt, robot_positions, robot_velocities)
MovingCylinder.write_colliders(...)
```

Environment calls update once per physics step, after TrainInertia.step, before
collision-aware step and collider write. No robot-yaw rotation is applied: the
train inertia output is already in the fixed floor/world axes, with randomized
vehicle acceleration direction, independently of robot yaw. `EpisodeCylinder`
forwards update to the scene-owner implementation when present. Until B arrives,
it records last_a_train_xy and exposes sway_api_stub=True; it does NOT fabricate
sway. The owner must reset LP/offset state in its reset and perform collision
checks on the offset-adjusted candidate before changing physical position.

No GPU/Isaac runtime validation was authorized for UPDATE161. CPU fixtures and
source/API checks do not validate PhysX filter installation in a running scene.
