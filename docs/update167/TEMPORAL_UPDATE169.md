# UPDATE169 temporal C integration

Opt in with `nav_settings.temporal169_config = configs/gast_nav/temporal169_training_c.yaml`.
Deployment/default producer remains current-only, with default historical speed
bound null. The static-C setting is review-approved, not described as user approval.
Enabled cylinders cause an error; there is no dynamic-scene free reuse.

`TemporalEvidence169` retains at most six immutable sensor captures per support
grid. At physics sample boundaries it integrates the configured speed upper bound
and checks actual displacement/speed/yaw. This bounds the travelled path under
that operating assumption, including a return to the same position. A sampled
violation invalidates the old chain; reset rows invalidate old epochs. This is
not a proof of a continuous-time physical speed bound between samples.

Each policy tick evaluates frusta in the current pose with the accumulated path
and existing P2 budget. Support is conservatively warped from observed support,
never supplied from terrain metadata. Evidence age is source age, not query time;
TTL remains 0.5 s. The latest observed occupied/self endpoints are transformed
as additional blockers, never used to manufacture free space. Actor maps and
raw clearance veto remain on their existing path.

The C-only working delay bounds are 1.0 m/s (0.6 cap +0.4 allowance) and 1.5 rad/s
(1.0 command cap +0.5 allowance). These allowances are assumptions, not measured
worst-case guarantees. Violations are counted per physics sample and fail closed.

Run `GAST_RUN_ID=<unique> bash scripts/gast_nav/diagnose_easy_c169.sh` for the
64-env/600-tick gate; `--steps 2` is a plumbing smoke that cannot pass C READY.
The UPDATE169 90.111 inventory and optimized proof are now integrated. The
primitive-only runner disables legacy gap/stair reward terms and curriculum,
keeps nav_total/C rewards, disables cylinder collisions, and audits the live USD
terrain triangles against the primitive source before asserting a closed scene.
This is a new all-corridor fixture, not the previous mixed-terrain comparison.

Probe snapshots at ticks 0/299/599 contain actual certificates and required masks.
The offline analyzer asserts live/replay equality and computes diagnostic-only
counterfactuals (latest-only, no temporal reuse, no P2 erosion, no erosion). These
counterfactuals never change the live controller. Progress for the final gate is
averaged over t>=3 s, like HOLD; whole-run reward means remain separately logged.
