# UPDATE55: optional extensions on UPDATE51/54

All paths below are relative to this copied worktree. No GPU job was launched.
The prior UPDATE51/54 entrypoints remain usable. UPDATE55 uses
`scripts/gast/train_student_live_update55.py` and `v3_runner_update55.py`.

## Options and units

| Flag | Default | Meaning |
| --- | --- | --- |
| `--v3_async` | false | Single GRU, event-gated visual projection, separate validity/event and full capture-to-current transform |
| `--v3_predict_correct` | false | Separate experiment; requires `--v3_async`; body predictor GRU plus event-only corrector GRU. **Not enabled by +1** |
| `--v3_aux_lambda` | 0 | Existing UPDATE54 five-channel terrain objective |
| `--v3_front_lambda` | 0 | Same map head, x>0 height and local-level support only |
| `--v3_future_lambda` | 0 | Training-only transition to teacher raw latent at +10/+20/+40 control ticks |
| `--v3_replay_lambda` | 0 | Mean of category-balanced replay B3 losses, including 34624 preservation |
| `--v3_bptt_steps` | 16 | **Scheduled captures**, with all intervening ticks; choices 16/32/64, preserving UPDATE51 units |
| `--v3_sequence_length` | 16 | Loaded/replayed training origins in **control ticks**; choices 16/32/64 |
| `--v3_burn_in` | 16 | Observation-prefix control ticks, current v3 weights, no gradient |
| `--v3_replay_capacity` | 8 | Protected sequence capacity per category; optional offline auxiliary cache bounded to 5x this |
| `--v3_sequence_files` | empty | Explicit training UPDATE55 dagger files; needs replay or auxiliary coefficient >0 |

A future sequence has burn-in + train-length + 40 target-suffix ticks. The
suffix is evaluated without gradients, contributes no B3/current-action loss,
and supplies detached executed actions/teacher targets for training origins.
Only origins with exact 100 Hz timestamps, unchanged epoch, no intervening done,
and finite action/target are supervised. Online windows mask unavailable tails
instead of crossing an optimizer/window boundary. The online future objective
uses all eligible origins within its capture-count BPTT window.

Sequence recording (actor base, root z, delivered packet payloads) is enabled
when replay or any auxiliary is enabled. Older UPDATE51/54 files lack the new
replay schema and are rejected by the loader rather than copying their hidden.
Files must belong to the authorized training split; no test dataset is selected
by these launchers. The live sampler remains the original 24/4/4 sampler.

Replay keeps flat/stairs/force/landing/recovery in separate bounded queues,
including terminal landing/recovery labels. Each update samples one sequence
from **each populated category**. Unobserved categories are logged in
`replay_missing`; coverage is not fabricated. Burn-in starts from zero and
recomputes visual and belief state from recorded observations using current v3.
This is truncated burn-in, not exact reconstruction of infinite episode history.
The prefix length is tunable; no stored hidden (including old model hidden) is
read. Episode/reset consistency is checked. Recorded 34624 action targets stay
detached; B3 non-preparation preservation is reused unchanged.

## Async tick ABI (two graphs total)

Encode graph is unchanged. Tick retains the prior ordered inputs, renames
`camera_fresh` to `visual_valid`, then adds:

- `new_packet`: bool `[B]`, host accepted a new synchronized four-view packet
  before **this** tick. Rejected duplicates, out-of-order, future, old-epoch
  packets do not set it. Consume once after the tick; reset clears pending bit.
- `capture_to_now`: float32 `[B,3,4]`, upper block of `T_now^-1 T_capture`.
  Full body SE(3), row-major rotation/translation, implicit homogeneous last row
  `[0,0,0,1]`. Translation is in metres; capture/current pose use wxyz quaternion
  and root z. Host computes this from sim-root odometry proxy; no GT distance.

`visual_valid` includes ready and capture-age <300 ms. The visual projection is
injected only when valid AND new_packet; held images are not re-injected.
Relative tick motion and previous executed action retain UPDATE51 semantics.
Reset hidden before a new-epoch packet; host already does this in the bank.
Graph callers must also clear visual state through encode/reset as documented
in UPDATE51. Clock duplicate/reversal halts rather than silently replaying a tick.

A legacy checkpoint can initialize the asynchronous single GRU by copying old
input weights and zero-initializing added columns. The optional predictor is a
new initialized module. Resume does not invent optimizer state restoration.
Future/map modules live outside the student and never enter either ONNX graph.
Use `tools/update55/export_v3.py` for asynchronous checkpoints; the old exporter
cannot load a wider belief input. Default checkpoint exports remain byte-equal.

## Prepared 90.111 ablations — not run

From a separate `/home/onvlm/.../update51_student_v3` checkout:

```bash
bash tools/update55/ablation_baseline_90.sh small 42
bash tools/update55/ablation_plus1_90.sh small 42
bash tools/update55/ablation_plus2a_90.sh small 42
bash tools/update55/ablation_plus2b_90.sh small 42
bash tools/update55/ablation_plus3_90.sh small 42
```

Run separately after existing GPU work is finished. These are not an automatic
sweep. The launcher has the inherited `/home/onvlm` location guard and flock.
Baseline and +1 have no auxiliary coefficient; +2a/+2b/+3 use their own `.1`
example coefficient, not a validated recommendation. `V3_FRONT_LAMBDA`,
`V3_FUTURE_LAMBDA`, `V3_REPLAY_LAMBDA` override only the matching variant.
All five force UPDATE54 lambda=0 to isolate these comparisons. The Python
entrypoint still supports combinations explicitly. `V3_BPTT_STEPS=32|64`,
`V3_SEQUENCE_LENGTH=32|64`, `V3_BURN_IN`, `V3_RESUME`, `V3_ROUND` are supported.
No launcher enables predict/correct automatically.

## CPU verification

```bash
bash tools/update51/cpu.sh tools/update55/verify_cpu.py
bash tools/update51/cpu.sh tools/update55/verify_runner.py
bash tools/update51/cpu.sh tools/update55/verify_onnx.py
bash tools/update51/cpu.sh tools/update55/verify_regression54.py
```

The wrapper hides CUDA and does not use Apptainer GPU passthrough. Tests run
forward/backward and ORT CPU; runner optimizer.step is replaced by a no-op.
Results are under `outputs/update55_cpu/`. No training-performance or real
simulator/robot/odometry claim is made by these checks.
