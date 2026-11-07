# GAST-Narrow v2.3 (Hangang 128-node run) - 3 checkpoints for MuJoCo tests

Same network layout as narrowv22_371.pt (drop-in replacement). Test order: 602 -> 557 -> 604.

Final E2 (Isaac, 512 envs per cell, nominal condition, %):

| item | 6566 | 371 | it602 | it557 | it604 |
|---|---|---|---|---|---|
| gap forward cross | 93.6 | 94.5 | 95.2 | 93.7 | 94.4 |
| gap forward fall | 6.4 | 5.5 | 4.8 | 6.3 | 5.6 |
| gap backward cross | 99.4 | 99.6 | 99.6 | 99.4 | 99.7 |
| gap forward nose-up > 20 deg | 15.3 | 16.0 | 2.7 | 3.3 | 2.9 |
| stairs pass | 97.1 | 97.3 | 97.1 | 96.9 | 97.2 |
| dead-end stand | 2.7 | 79.6 | 98.1 | 96.0 | 95.8 |
| blocked passage hold | 0 | 95.9 | 99.4 | 99.6 | 99.5 |
| open passage pass | 78.4 | 90.6 | 98.7 | 98.4 | 96.9 |

MuJoCo checks: gap forward/backward nose pitch, stairs, dead-end stop, L-corner leg snag, high-speed joystick.
params/: agent.yaml, env.yaml, gast_warm_start.yaml of the run. Verify: sha256sum -c SHA256SUMS
