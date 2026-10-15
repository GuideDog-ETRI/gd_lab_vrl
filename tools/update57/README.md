# UPDATE57: optional realtime training and production export

Small (1,686,071 training parameters) remains the default; large (7,996,983)
is explicit. Old 51/54/55 sources and artifacts are untouched. No Pilot code.

## Production export

```bash
bash tools/update51/cpu.sh tools/update57/export_v3.py checkpoint.pt new_output
bash tools/update51/cpu.sh tools/update57/export_v3.py checkpoint.pt new_cached_output --projection-cache
```

Default encode outputs **h_visual_out, quality**. Cached encode additionally outputs
**visual_projection** (small128 / large512). Cached tick replaces input h_visual with
visual_projection, all other inputs and output meanings unchanged. Async55 and
predict/correct55 supported. Default tick is byte-identical to the old exporter
for the same checkpoint. Encode intentionally changes: capture/spatial heads are
pruned. Aux/lookahead remain checkpoint-only training objects.

Host must invalidate cache/ready on reset, episode/model epoch change; a newly
completed same-epoch packet can restore it. Invalidity goes through valid/ready;
setting the projection's input to zero is incorrect because of projection bias.
The encode worker writes visual state and its unmasked projection; tick owns
belief and applies the freshness/event mask AFTER projection. Reset is a serialized
barrier for both owners. External multithreaded Pilot integration is not provided.

## Existing 10 ms online training host

New entrypoint: `scripts/gast/train_student_live_update57.py`. Same CLI and small
default as UPDATE55, with all new flags off:

- `--v3_projection_cache`
- `--v3_latest_encode` (in-flight1, newest complete waiting1)
- `--v3_encode_delay_ticks N` (0..100, requires latest encode)
- `--v3_action_delay_ticks 0|1|2`

Use the existing training environment/setup to launch this Python entrypoint;
no remote/GPU job was run as part of UPDATE57. The test uses its actual runner on
a CPU fake host with real actor weights. Epoch checks reject queued actions and
packets across resets. Drop counts and peak queue occupancy enter window logs.
Capture-to-capture motion uses the last committed packet, not the last arrival.
The old transport's physical arrival is retained as arrival_step; completion is
used as feature availability time, capture_step is never overwritten.

`previous_action` is the clipped action actually passed to env.step, after delay.
Decision-time teacher/action/preservation losses keep the existing definition;
future auxiliary transitions use recorded executed_action. Delayed commands are
not supervised using future state. No stale teacher inference fallback.

## Millisecond jitter / separated sensor timing

`gd_lab.gast.v3_realtime.TrainingHost` is an additional, executable **simulation
host interface**, not an IsaacLab physics patch. It requires `advance(.001,
applied_action)`, `observe()` and `capture()` callbacks. `run_until(end_ms, ...)`
advances fixed 1ms physics substeps and independently releases actual controller
events; skipped time is never filled with fake GRU ticks. The 10ms Isaac runner
above does NOT pretend to support ±1ms by rewriting timestamp metadata.

```python
host = TrainingHost(student, n, Timing(
    jitter_ms=2, long_interval_ms=30, long_probability=.01,
    action_delay_ticks=2, encode_ms=120,
    burst_probability=.02, burst_packets=3,
    long_vision_probability=.01, long_vision_ms=1200,
    proprio_delay_ms=5, odometry_std_m=.01,
    timestamp_offset_ms=0, timestamp_drift_ppm=0,
), projection_cache=True)
host.run_until(2000, advance=physics_step, observe=available_observation,
               capture=synchronized_capture, actor=available_input_actor,
               on_tick=execution_loss, on_encode=capture_loss)
```

All Timing perturbations default to zero/off. Sweep jitter_ms 1/2;
long_interval_ms 20/30; action_delay_ticks 0/1/2. Probabilities, delay1200 and sensor
error magnitudes are experimental starting values, not measured distributions.
Physics callback may return a done mask: reset clears state, action and sensor
history at the next boundary. One environment reset conservatively clears shared
proprio history; no old-episode observation is supplied after reset.

`observe()` gives proprio[B,230], observed pose[B,7] (xy,yaw,wxyz), z[B]. Pose noise
is independently seeded and affects observation only. Capture original time,
arrival time and observed timestamp are separate fields. Offset/drift do not
replace original capture metadata; future or old-epoch observations are rejected.
Network delay storage is separate from the bounded encoder queue and bounded by
configured maximum delay / capture period. Capture and encode callbacks get
labels only for loss. Actor never receives packet labels or future observations.

Training callbacks can accumulate existing raw/spatial/action/aux losses from
returned tensors. After each TBPTT optimizer update call
`host.state.detach(weights_changed=True)` to reproject held features with updated
weights. Cache accumulation gradients are tolerance-equal, not guaranteed bit-equal
because sum order changes. The no-option legacy runner remains bit-identical.

This host is CPU tested including backward. Actual Isaac/MuJoCo callback integration,
full jitter training and task performance are not claimed; a simulator exposing
1ms substeps must supply these callbacks before using jitter in a real training run.

## Verification and Z890 benchmark

```bash
bash tools/update51/cpu.sh tools/update57/verify_cpu.py
bash tools/update51/cpu.sh tools/update57/verify_onnx.py
bash tools/update51/cpu.sh tools/update57/verify_runner.py
bash tools/update51/cpu.sh tools/update57/benchmark.py --size all --samples 10000
python3 tools/update57/verify_static.py
```

Benchmark requires verify_onnx artifacts first. It records lscpu, board, kernel,
ORT/PyTorch versions, affinity, load, env, hashes and raw timing samples. CPU EP,
FP32, batch1, intra/inter-op1, EXTENDED. Measures encode/tick/real GAST2000 actor
MLP and actual serial tick+actor / encode+tick+actor p50/p95/p99/p99.9/max.
It excludes normalizer/CENet/camera/transport/DDS and concurrent CPU contention.
Untrained student fixtures are reference graphs. No deployment deadline guarantee.
No Pilot PC measurement is authorized or performed here.
