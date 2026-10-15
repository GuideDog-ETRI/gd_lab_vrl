# UPDATE19 export / runtime contract

Use `export_bundle.py`, not deploy's unmodified `export/gast_checkpoint.py`: the latter does not know `student_policy_state` and would silently export the teacher actor.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src nice -n 19 apptainer exec /home/user/workspace/gd_lab_isaaclab.sif \
/home/user/workspace/venv_apptainer/bin/python -B tools/update19_export/export_bundle.py \
  --checkpoint /absolute/path/to/NEW_STUDENT.pt \
  --output-dir /home/user/gd_project/archive/aux_worktrees/update19_b/exported/UNIQUE_RUN \
  --expected-student-iteration 42624 \
  --expected-teacher-sha256 db500e96ddf2325d415ac960417d2e34417ded87d804f630ae165675eaba24b4
```

The output directory must not exist. Exports are restricted to update19_b or notes. The teacher path recorded in the checkpoint must remain readable with its `params/agent.yaml`. Exports load policy state strictly, check teacher SHA, compare source backbone tensors, test 21 recurrent student cases and 21 actor cases on CPU ONNX Runtime, then write a manifest. A missing owned policy is an error, never a frozen fallback.

## Copy / frozen (including A*)

Actor: `direct_obs [B,46], cenet_obs [B,230], terrain_latent [B,32] -> actions [B,12], z_t [B,19]`.
Student: existing five inputs, two outputs. `copy` exports the student's learned actor and teacher CENet/normalizer; frozen exports the teacher actor. `A*` changes training targets only: execution/export still applies the normal gate. Existing Pilot graph arities work for these bundles; no rollout success is claimed from offline parity.

## Residual: explicit version-2 interface, NOT directly runnable in current Pilot

Actor: the same three inputs plus `age_seconds [B]` and `quality_gate [B]`. Outputs remain `actions,z_t`. Student: the same five inputs, but three outputs `terrain_latent [1,32], hidden_out [1,6116], quality_gate [1,1]`. The returned gate is the **effective delivery gate** after sigmoid, availability and delivery-age attenuation. Never apply this gate to the latent a second time.

The residual receives `[base_actor(203), executed_latent(32), age(1), gate(1)]`. Age is execution-clock minus the same packet's capture timestamp, clamped to [0,1] in the graph. Gate is held from the delivered packet; the training freshness rule multiplies **both** latent and gate by `ready && age<0.3`. It does not reapply continuous age attenuation on every actor tick. Carry latent, gate, timestamp, and reset/sequence identity atomically.

Exact deploy integration required (read-only reviewed; NOT applied):

1. `perception/common/VisionStudentThread.cpp`: initialization near lines104–200 currently permits GAST5-input/2-output. Add an explicitly versioned 5/3 variant. At `studentLoop` near445–446 request the third output, validate finite gate in[0,1], and publish it under the same mutex as latent and `m_latentStampMs` near704–710. Extend `VisionStudentThread.hpp` and `latestLatent` API to return the gate and packet identity with the existing capture timestamp. Reset all metadata together. Do not retrieve gate and latent in separate calls.
2. `pilot/src/PolicyBackend.cpp` near448–458 dispatches only a3-input vision graph. Accept the explicit residual5-input contract after checking metadata/names/shapes, otherwise reject.
3. `perception/common/PolicyBackendVrl.cpp` near125–157 validates3 inputs. Add5-input validation and the age/gate tensors to inference near262–295; preserve the existing normalizer/CENet packing and action output scaling. Actor timestamp must use the same monotonic clock domain as capture/`latestLatent`. Pass the held effective gate, not sigmoid/raw quality.
4. **Freshness mismatch requiring review:** existing `VisionLatentGate.hpp` uses Fresh<250ms, holds until1000ms, then faults. It explicitly avoids synthesizing a zero latent; training inherited the old300ms zero-latent rule. This mismatch predates UPDATE19. Do not silently change the deployed fault path. For exact residual feature parity in delayed cases, review and version a policy-input adapter that zeroes both latent+gate at300ms while preserving the outer1000ms fault/Waiting path, OR align the new training mode to the reviewed runtime hold rule before collecting B2. The latter changes the experiment and needs a new contract/checkpoint. Current code implements the specified/default300ms training behavior. Normal fresh-case ONNX parity alone does not resolve this.
5. Add runtime tests for ordinary delivery, hold, out-of-order packets, reset with in-flight packets, missing input,250/300/1000ms boundaries, and simultaneous latent/gate updates. Confirm same frame/current observation against Python and CPU ONNX before a new MuJoCo evaluation. Do not install B2 in the active deployment checkout meanwhile.

The manifest marks residual bundles `runtime_v2_REQUIRED` and `deploy_ready_without_runtime_changes=false`. Existing runtime arity checks reject them. No GPU/Isaac/MuJoCo/Pilot process was launched for this implementation.

## Synthetic smoke artifacts

`smoke_bundle.py` uses the real seen0 checkpoint and teacher backbone, but adds a small **synthetic** actor delta to ensure export does not accidentally substitute the teacher. These test bundles/checkpoints under `.tmp/bundle_smoke_*` are not learned candidates. The manifest marks `synthetic_smoke_only_NOT_TRAINED=true`. The test is reproducible and uses a new temporary directory for each invocation.
