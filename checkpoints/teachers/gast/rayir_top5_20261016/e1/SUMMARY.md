# E1 results: Ray+IR teachers 2769 / 2836 / 3906

Isaac Lab E1, 32 env × 30 s, vx 1.0, camera transport 100 ms / 0–150 ms / 5% drop, ARM4 gains, seeds 42–45.
Cells: clean/reached (gap) or passed/reached (stairs), summed over seeds; fd = approaches with a foot drop, ct = body contact.
Same seed → same initial_state_sha256 across teachers and paths (paired episodes). `stdown` = pyramid_stairs_inv (pit climb-out), `stup` = pyramid_stairs.
Camera without filter was only run for 2769. Only the Top-1 of each run was evaluated; Top-2..5 have online scores only.

## IR native

| path | terrain | 2769 | 2836 | 3906 |
|---|---|---|---|---|
| G | gap | 76/76 = 100.0% (fd 0) | 79/92 = 85.9% (fd 12) | 91/92 = 98.9% (fd 1) |
| G | stdown | 165/169 = 97.6% (ct 2) | 166/170 = 97.6% (ct 6) | 164/165 = 99.4% (ct 2) |
| G | stup | 176/176 = 100.0% (ct 31) | 176/177 = 99.4% (ct 33) | 176/176 = 100.0% (ct 31) |
| GV | gap | 91/91 = 100.0% (fd 0) | 85/85 = 100.0% (fd 0) | 82/82 = 100.0% (fd 0) |
| GV | stdown | 165/171 = 96.5% (ct 4) | 164/169 = 97.0% (ct 0) | 165/166 = 99.4% (ct 2) |
| GV | stup | 176/176 = 100.0% (ct 31) | 176/176 = 100.0% (ct 33) | 176/176 = 100.0% (ct 31) |
| LiDAR v25or | gap | 96/97 = 99.0% (fd 1) | 75/86 = 87.2% (fd 11) | 90/93 = 96.8% (fd 3) |
| LiDAR v25or | stdown | 166/168 = 98.8% (ct 2) | 166/169 = 98.2% (ct 4) | 163/163 = 100.0% (ct 2) |
| LiDAR v25or | stup | 176/176 = 100.0% (ct 31) | 176/177 = 99.4% (ct 31) | 176/176 = 100.0% (ct 31) |
| camera v25or (no filter) | gap | 41/161 = 25.5% (fd 119) | — | — |
| camera v25or (no filter) | stdown | 162/170 = 95.3% (ct 24) | — | — |
| camera v25or (no filter) | stup | 176/182 = 96.7% (ct 41) | — | — |
| camera+filter v25or | gap | 81/86 = 94.2% (fd 5) | 80/92 = 87.0% (fd 11) | 76/79 = 96.2% (fd 2) |
| camera+filter v25or | stdown | 165/170 = 97.1% (ct 2) | 166/167 = 99.4% (ct 2) | 165/165 = 100.0% (ct 2) |
| camera+filter v25or | stup | 176/176 = 100.0% (ct 31) | 176/176 = 100.0% (ct 31) | 176/176 = 100.0% (ct 31) |

## IR = 0 at inference

| path | terrain | 2769 | 2836 | 3906 |
|---|---|---|---|---|
| G | gap | 75/77 = 97.4% (fd 2) | 82/94 = 87.2% (fd 11) | 89/90 = 98.9% (fd 1) |
| G | stdown | 166/168 = 98.8% (ct 0) | 166/169 = 98.2% (ct 2) | 164/166 = 98.8% (ct 6) |
| G | stup | 176/176 = 100.0% (ct 31) | 176/176 = 100.0% (ct 31) | 176/176 = 100.0% (ct 31) |
| GV | gap | 85/85 = 100.0% (fd 0) | 91/91 = 100.0% (fd 0) | 89/92 = 96.7% (fd 3) |
| GV | stdown | 166/169 = 98.2% (ct 8) | 162/167 = 97.0% (ct 0) | 166/166 = 100.0% (ct 0) |
| GV | stup | 176/176 = 100.0% (ct 31) | 176/176 = 100.0% (ct 31) | 176/177 = 99.4% (ct 33) |
| LiDAR v25or | gap | 93/95 = 97.9% (fd 1) | 90/98 = 91.8% (fd 8) | 85/91 = 93.4% (fd 6) |
| LiDAR v25or | stdown | 167/168 = 99.4% (ct 4) | 167/170 = 98.2% (ct 4) | 162/167 = 97.0% (ct 4) |
| LiDAR v25or | stup | 176/176 = 100.0% (ct 31) | 176/176 = 100.0% (ct 31) | 176/176 = 100.0% (ct 31) |
| camera v25or (no filter) | gap | 34/165 = 20.6% (fd 131) | — | — |
| camera v25or (no filter) | stdown | 162/169 = 95.9% (ct 16) | — | — |
| camera v25or (no filter) | stup | 181/184 = 98.4% (ct 49) | — | — |
| camera+filter v25or | gap | 78/85 = 91.8% (fd 7) | 85/92 = 92.4% (fd 7) | 82/87 = 94.3% (fd 5) |
| camera+filter v25or | stdown | 166/170 = 97.6% (ct 6) | 166/170 = 97.6% (ct 6) | 163/166 = 98.2% (ct 4) |
| camera+filter v25or | stup | 177/177 = 100.0% (ct 33) | 176/176 = 100.0% (ct 31) | 176/177 = 99.4% (ct 33) |

## Camera noise diagnosis (2769, gap, seed 42; `dcam_*`)

U136_CAM_OFF zeroes sensor121 rates. Spikes (0.6% of pixels at U(0.15, 5 m)) + the min-z cell reducer turned flat cells into "deep";
camfilter136 (3×3 neighbour-median outlier rejection) fixes it without changing the noise model.

- fly off: 10/47, foot drops 37
- spike off: 22/23, foot drops 1
- spikefly off: 20/21, foot drops 1
- allnoise off: 17/19, foot drops 2
