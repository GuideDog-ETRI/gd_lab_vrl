# E2 결과: v21_top5

- 기준 모델: b6566, 기존 능력 지표가 기준보다 2%p 넘게 낮으면 ⚠
- seed 42, 에피소드 20 s, env마다 첫 에피소드 1회

## 조건 nominal (라이다 both, 탑재물 5.0 kg, 밀기 False, 계단 외란 False)

### 갭

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| gap/forward/L7/v0.5 | 건넘 | 97.9 | 95.8 | 97.9 | 97.9 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.5 | clean/시도 | 76.6 | 61.7 | 80.9 | 72.3 | 75.0 | 68.8 | 75.0 |
| gap/forward/L7/v0.5 | 넘어짐 | 2.1 | 4.2 | 2.1 | 2.1 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.9 | clean/시도 | 64.6 | 75.0 | 83.3 | 70.8 | 75.0 | 68.8 | 75.0 |
| gap/forward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 건넘 | 91.7 | 95.8 | 91.7 | 91.7 | 91.7 | 89.6 | 89.6 |
| gap/forward/L11/v0.5 | clean/시도 | 79.5 | 69.6 | 77.3 | 75.0 | 88.6 | 83.7 | 74.4 |
| gap/forward/L11/v0.5 | 넘어짐 | 8.3 | 4.2 | 8.3 | 8.3 | 8.3 | 10.4 | 10.4 |
| gap/forward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 97.9 | 100.0 | 100.0 |
| gap/forward/L11/v0.9 | clean/시도 | 85.4 | 87.5 | 83.3 | 70.8 | 80.9 | 85.4 | 72.9 |
| gap/forward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 2.1 | 0.0 | 0.0 |
| gap/forward/L14/v0.5 | 건넘 | 85.4 | 81.2 | 83.3 | 85.4 | 77.1 | 87.5 | 85.4 |
| gap/forward/L14/v0.5 | clean/시도 | 90.2 | 79.5 | 80.0 | 75.6 | 83.8 | 83.3 | 92.7 |
| gap/forward/L14/v0.5 | 넘어짐 | 14.6 | 16.7 | 16.7 | 14.6 | 22.9 | 12.5 | 14.6 |
| gap/forward/L14/v0.9 | 건넘 | 93.8 | 91.7 | 87.5 | 85.4 | 79.2 | 85.4 | 91.7 |
| gap/forward/L14/v0.9 | clean/시도 | 53.3 | 72.7 | 57.1 | 51.2 | 57.9 | 51.2 | 36.4 |
| gap/forward/L14/v0.9 | 넘어짐 | 6.2 | 8.3 | 12.5 | 14.6 | 20.8 | 14.6 | 8.3 |
| gap/backward/L7/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 97.9 | 100.0 | 100.0 |
| gap/backward/L7/v0.5 | clean/시도 | 81.2 | 66.7 | 75.0 | 72.9 | 76.6 | 75.0 | 79.2 |
| gap/backward/L7/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 2.1 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.9 | clean/시도 | 79.2 | 81.2 | 79.2 | 83.3 | 81.2 | 72.9 | 75.0 |
| gap/backward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 건넘 | 97.9 | 100.0 | 100.0 | 97.9 | 97.9 | 100.0 | 97.9 |
| gap/backward/L11/v0.5 | clean/시도 | 68.1 | 62.5 | 58.3 | 66.0 | 68.1 | 66.7 | 63.8 |
| gap/backward/L11/v0.5 | 넘어짐 | 2.1 | 0.0 | 0.0 | 2.1 | 2.1 | 0.0 | 2.1 |
| gap/backward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.9 | clean/시도 | 89.6 | 87.5 | 75.0 | 85.4 | 87.5 | 75.0 | 91.7 |
| gap/backward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 건넘 | 97.9 | 97.9 | 100.0 | 97.9 | 97.9 | 97.9 | 93.8 |
| gap/backward/L14/v0.5 | clean/시도 | 61.7 | 68.1 | 54.2 | 66.0 | 61.7 | 68.1 | 57.8 |
| gap/backward/L14/v0.5 | 넘어짐 | 2.1 | 2.1 | 0.0 | 2.1 | 2.1 | 2.1 | 6.2 |
| gap/backward/L14/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L14/v0.9 | clean/시도 | 70.8 | 56.2 | 68.8 | 64.6 | 72.9 | 66.7 | 68.8 |
| gap/backward/L14/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 건넘 | 97.0 | 96.9 | 96.7 | 96.4 | 95.0 ⚠ | 96.7 | 96.5 |
| 평균 | clean/시도 | 75.0 | 72.4 ⚠ | 72.7 ⚠ | 71.2 ⚠ | 75.8 | 72.1 ⚠ | 71.9 ⚠ |
| 평균 | 넘어짐 | 3.0 | 3.0 | 3.3 | 3.6 | 5.0 ⚠ | 3.3 | 3.5 |

### 계단

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| stairs/pyramid_stairs/up/L9 | 통과 | 96.9 | 96.9 | 96.9 | 96.9 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L9 | 몸통 접촉 | 34.4 | 12.5 | 15.6 | 46.9 | 37.5 | 43.8 | 40.6 |
| stairs/pyramid_stairs/up/L14 | 통과 | 96.9 | 93.8 | 93.8 | 96.9 | 100.0 | 93.8 | 96.9 |
| stairs/pyramid_stairs/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L14 | 몸통 접촉 | 43.8 | 12.5 | 62.5 | 46.9 | 46.9 | 56.2 | 53.1 |
| stairs/pyramid_stairs/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L9 | 몸통 접촉 | 53.1 | 84.4 | 50.0 | 50.0 | 50.0 | 46.9 | 53.1 |
| stairs/pyramid_stairs/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L14 | 몸통 접촉 | 87.5 | 93.8 | 84.4 | 90.6 | 90.6 | 81.2 | 96.9 |
| stairs/pyramid_stairs_inv/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L9 | 몸통 접촉 | 31.2 | 18.8 | 25.0 | 37.5 | 21.9 | 25.0 | 31.2 |
| stairs/pyramid_stairs_inv/up/L14 | 통과 | 96.9 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L14 | 넘어짐 | 3.1 | 0.0 | 0.0 | 0.0 | 3.1 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L14 | 몸통 접촉 | 59.4 | 25.0 | 68.8 | 62.5 | 56.2 | 65.6 | 46.9 |
| stairs/pyramid_stairs_inv/down/L9 | 통과 | 96.9 | 93.8 | 96.9 | 96.9 | 100.0 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L9 | 몸통 접촉 | 40.6 | 50.0 | 53.1 | 40.6 | 53.1 | 28.1 | 34.4 |
| stairs/pyramid_stairs_inv/down/L14 | 통과 | 93.8 | 96.9 | 100.0 | 96.9 | 100.0 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L14 | 몸통 접촉 | 90.6 | 71.9 | 87.5 | 78.1 | 90.6 | 68.8 | 78.1 |
| stairs/pyramid_stairs_wide/up/L9 | 통과 | 96.9 | 100.0 | 96.9 | 96.9 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L9 | 몸통 접촉 | 21.9 | 6.2 | 15.6 | 21.9 | 9.4 | 25.0 | 28.1 |
| stairs/pyramid_stairs_wide/up/L14 | 통과 | 78.1 | 75.0 | 75.0 | 75.0 | 87.5 | 78.1 | 84.4 |
| stairs/pyramid_stairs_wide/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.1 | 0.0 |
| stairs/pyramid_stairs_wide/up/L14 | 몸통 접촉 | 46.9 | 62.5 | 50.0 | 37.5 | 46.9 | 50.0 | 50.0 |
| stairs/pyramid_stairs_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L9 | 몸통 접촉 | 59.4 | 93.8 | 68.8 | 65.6 | 71.9 | 65.6 | 62.5 |
| stairs/pyramid_stairs_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L14 | 몸통 접촉 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 몸통 접촉 | 46.9 | 25.0 | 37.5 | 50.0 | 40.6 | 50.0 | 50.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 몸통 접촉 | 53.1 | 53.1 | 59.4 | 40.6 | 59.4 | 37.5 | 53.1 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 통과 | 96.9 | 100.0 | 96.9 | 100.0 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 몸통 접촉 | 78.1 | 84.4 | 62.5 | 65.6 | 75.0 | 50.0 | 53.1 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 몸통 접촉 | 100.0 | 96.9 | 96.9 | 96.9 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_nose/up/L9 | 통과 | 96.9 | 93.8 | 96.9 | 100.0 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/up/L9 | 몸통 접촉 | 31.2 | 25.0 | 15.6 | 21.9 | 34.4 | 28.1 | 31.2 |
| stairs/pyramid_stairs_nose/up/L14 | 통과 | 65.6 | 93.8 | 81.2 | 78.1 | 78.1 | 87.5 | 75.0 |
| stairs/pyramid_stairs_nose/up/L14 | 넘어짐 | 34.4 | 3.1 | 12.5 | 18.8 | 18.8 | 6.2 | 21.9 |
| stairs/pyramid_stairs_nose/up/L14 | 몸통 접촉 | 75.0 | 71.9 | 71.9 | 81.2 | 75.0 | 71.9 | 71.9 |
| stairs/pyramid_stairs_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L9 | 몸통 접촉 | 78.1 | 87.5 | 56.2 | 65.6 | 71.9 | 62.5 | 59.4 |
| stairs/pyramid_stairs_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 통과 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 몸통 접촉 | 71.9 | 65.6 | 68.8 | 71.9 | 71.9 | 68.8 | 84.4 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 통과 | 100.0 | 100.0 | 93.8 | 96.9 | 100.0 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 넘어짐 | 0.0 | 0.0 | 6.2 | 3.1 | 0.0 | 3.1 | 3.1 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 몸통 접촉 | 81.2 | 62.5 | 84.4 | 84.4 | 90.6 | 81.2 | 81.2 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 몸통 접촉 | 43.8 | 68.8 | 53.1 | 37.5 | 53.1 | 37.5 | 40.6 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 통과 | 96.9 | 96.9 | 100.0 | 96.9 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 몸통 접촉 | 84.4 | 96.9 | 81.2 | 78.1 | 78.1 | 75.0 | 81.2 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 통과 | 96.4 | 97.4 | 97.0 | 97.1 | 97.8 | 97.3 | 97.4 |
| 평균 | 넘어짐 | 1.6 | 0.1 | 0.8 | 0.9 | 0.9 | 0.5 | 1.0 |
| 평균 | 몸통 접촉 | 62.9 | 61.2 | 61.2 | 61.3 | 63.4 | 59.0 | 61.5 |

### 경사·거친 지면

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| terrain/hf_pyramid_slope/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/hf_pyramid_slope/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/hf_pyramid_slope/L14 | 속도 오차 m/s | 0.161 | 0.171 | 0.156 | 0.161 | 0.158 | 0.155 | 0.159 |
| terrain/hf_pyramid_slope_inv/L14 | 통과 | 81.2 | 81.2 | 84.4 | 81.2 | 81.2 | 84.4 | 81.2 |
| terrain/hf_pyramid_slope_inv/L14 | 넘어짐 | 0.0 | 9.4 | 0.0 | 0.0 | 0.0 | 3.1 | 3.1 |
| terrain/hf_pyramid_slope_inv/L14 | 속도 오차 m/s | 0.300 | 0.344 | 0.283 | 0.296 | 0.289 | 0.289 | 0.303 |
| terrain/random_rough/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/random_rough/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/random_rough/L14 | 속도 오차 m/s | 0.161 | 0.174 | 0.160 | 0.164 | 0.163 | 0.169 | 0.163 |
| terrain/ring_fence/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/ring_fence/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/ring_fence/L14 | 속도 오차 m/s | 0.162 | 0.177 | 0.160 | 0.161 | 0.157 | 0.168 | 0.167 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 통과 | 95.3 | 95.3 | 96.1 | 95.3 | 95.3 | 96.1 | 95.3 |
| 평균 | 넘어짐 | 0.0 | 2.3 ⚠ | 0.0 | 0.0 | 0.0 | 0.8 | 0.8 |
| 평균 | 속도 오차 m/s | 0.196 | 0.217 | 0.190 | 0.195 | 0.192 | 0.195 | 0.198 |

### 좁은 통로

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| narrow/narrow_corridor/pass_wide (폭 0.80) | 통과 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 무접촉 | 81.2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.065 | 0.099 | 0.096 | 0.104 | 0.099 | 0.101 | 0.110 |
| narrow/narrow_corridor/pass_tight (폭 0.66) | 통과 | 78.1 | 90.6 | 75.0 | 40.6 | 46.9 | 71.9 | 59.4 |
| narrow/narrow_corridor/pass_tight (폭 0.66) | 무접촉 | 28.1 | 93.8 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 |
| narrow/narrow_corridor/pass_tight (폭 0.66) | 벽 거리 중앙값 m | -0.001 | 0.051 | 0.045 | 0.244 | 0.061 | 0.058 | 0.071 |
| narrow/narrow_corridor/block_margin (폭 0.66) | 서기 | 0.0 | 90.6 | 96.9 | 96.9 | 93.8 | 96.9 | 93.8 |
| narrow/narrow_corridor/block_margin (폭 0.66) | 침범 | 100.0 | 6.2 | 3.1 | 3.1 | 6.2 | 3.1 | 6.2 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 서기 | 0.0 | 90.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 침범 | 100.0 | 9.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.79) | 통과 | 84.4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.79) | 무접촉 | 71.9 | 81.2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.79) | 벽 거리 중앙값 m | 0.043 | 0.027 | 0.099 | 0.104 | 0.103 | 0.108 | 0.111 |
| narrow/narrow_door_pos/pass_tight (폭 0.67) | 통과 | 71.9 | 96.9 | 93.8 | 21.9 | 93.8 | 90.6 | 71.9 |
| narrow/narrow_door_pos/pass_tight (폭 0.67) | 무접촉 | 43.8 | 65.6 | 93.8 | 96.9 | 90.6 | 96.9 | 96.9 |
| narrow/narrow_door_pos/pass_tight (폭 0.67) | 벽 거리 중앙값 m | 0.000 | 0.010 | 0.055 | 0.371 | 0.049 | 0.052 | 0.054 |
| narrow/narrow_door_pos/block_margin (폭 0.67) | 서기 | 0.0 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/block_margin (폭 0.67) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_pos/block_narrow (폭 0.52) | 서기 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/block_narrow (폭 0.52) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.80) | 통과 | 93.8 | 93.8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.80) | 무접촉 | 84.4 | 87.5 | 100.0 | 96.9 | 100.0 | 96.9 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.044 | 0.035 | 0.104 | 0.105 | 0.100 | 0.104 | 0.109 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 통과 | 84.4 | 96.9 | 75.0 | 18.8 | 65.6 | 62.5 | 56.2 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 무접촉 | 62.5 | 71.9 | 96.9 | 96.9 | 96.9 | 93.8 | 93.8 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 벽 거리 중앙값 m | 0.002 | -0.004 | 0.068 | 0.444 | 0.057 | 0.055 | 0.076 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 서기 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.53) | 서기 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.53) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 통과 | 87.5 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 무접촉 | 43.8 | 96.9 | 100.0 | 100.0 | 100.0 | 96.9 | 96.9 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 벽 거리 중앙값 m | -0.000 | 0.028 | 0.059 | 0.060 | 0.063 | 0.049 | 0.054 |
| narrow/narrow_bend_left/pass_tight (폭 0.65) | 통과 | 46.9 | 81.2 | 53.1 | 0.0 | 15.6 | 28.1 | 3.1 |
| narrow/narrow_bend_left/pass_tight (폭 0.65) | 무접촉 | 12.5 | 65.6 | 71.9 | 100.0 | 96.9 | 87.5 | 100.0 |
| narrow/narrow_bend_left/pass_tight (폭 0.65) | 벽 거리 중앙값 m | -0.019 | 0.002 | 0.002 | 0.431 | 0.255 | 0.260 | 0.312 |
| narrow/narrow_bend_left/block_margin (폭 0.65) | 서기 | 0.0 | 81.2 | 96.9 | 96.9 | 93.8 | 93.8 | 93.8 |
| narrow/narrow_bend_left/block_margin (폭 0.65) | 침범 | 100.0 | 15.6 | 3.1 | 3.1 | 3.1 | 6.2 | 6.2 |
| narrow/narrow_bend_left/block_narrow (폭 0.52) | 서기 | 0.0 | 90.6 | 100.0 | 96.9 | 100.0 | 96.9 | 100.0 |
| narrow/narrow_bend_left/block_narrow (폭 0.52) | 침범 | 100.0 | 9.4 | 0.0 | 3.1 | 0.0 | 3.1 | 0.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.81) | 통과 | 84.4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.81) | 무접촉 | 34.4 | 100.0 | 96.9 | 100.0 | 100.0 | 96.9 | 96.9 |
| narrow/narrow_bend_right/pass_wide (폭 0.81) | 벽 거리 중앙값 m | -0.008 | 0.047 | 0.057 | 0.068 | 0.066 | 0.059 | 0.063 |
| narrow/narrow_bend_right/pass_tight (폭 0.65) | 통과 | 50.0 | 65.6 | 56.2 | 6.2 | 25.0 | 40.6 | 21.9 |
| narrow/narrow_bend_right/pass_tight (폭 0.65) | 무접촉 | 12.5 | 59.4 | 75.0 | 100.0 | 84.4 | 65.6 | 87.5 |
| narrow/narrow_bend_right/pass_tight (폭 0.65) | 벽 거리 중앙값 m | -0.017 | -0.003 | 0.012 | 0.409 | 0.322 | 0.272 | 0.393 |
| narrow/narrow_bend_right/block_margin (폭 0.65) | 서기 | 0.0 | 81.2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/block_margin (폭 0.65) | 침범 | 100.0 | 18.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_right/block_narrow (폭 0.50) | 서기 | 0.0 | 90.6 | 100.0 | 96.9 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_bend_right/block_narrow (폭 0.50) | 침범 | 100.0 | 3.1 | 0.0 | 3.1 | 0.0 | 0.0 | 3.1 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 통과 | 77.8 | 92.5 | 85.3 | 58.8 | 74.7 | 79.4 | 71.2 |
| 평균 | 무접촉 | 47.5 | 82.2 | 93.4 | 99.1 | 96.9 | 93.1 | 97.2 |
| 평균 | 벽 거리 중앙값 m | 0.011 | 0.029 | 0.060 | 0.234 | 0.117 | 0.112 | 0.135 |
| 평균 | 서기 | 0.0 | 92.2 | 99.4 | 98.8 | 98.8 | 98.8 | 98.4 |
| 평균 | 침범 | 100.0 | 6.2 | 0.6 | 1.2 | 0.9 | 1.2 | 1.6 |

### 막힌 통로

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| dead_end/narrow_dead_end/L7 | 서기 | 0.0 | 93.8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| dead_end/narrow_dead_end/L7 | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dead_end/narrow_dead_end/L14 | 서기 | 6.2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| dead_end/narrow_dead_end/L14 | 침범 | 90.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 서기 | 3.1 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 평균 | 침범 | 95.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## 조건 lidar_off (라이다 off, 탑재물 5.0 kg, 밀기 False, 계단 외란 False)

### 갭

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| gap/forward/L7/v0.5 | 건넘 | 100.0 | 95.8 | 100.0 | 100.0 | 100.0 | 100.0 | 97.9 |
| gap/forward/L7/v0.5 | clean/시도 | 70.8 | 78.3 | 72.9 | 72.9 | 75.0 | 72.9 | 78.7 |
| gap/forward/L7/v0.5 | 넘어짐 | 0.0 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 | 2.1 |
| gap/forward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.9 | clean/시도 | 87.5 | 87.5 | 79.2 | 81.2 | 85.4 | 85.4 | 81.2 |
| gap/forward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 건넘 | 91.7 | 95.8 | 95.8 | 97.9 | 95.8 | 97.9 | 100.0 |
| gap/forward/L11/v0.5 | clean/시도 | 86.4 | 56.5 | 71.7 | 89.4 | 82.6 | 87.2 | 83.3 |
| gap/forward/L11/v0.5 | 넘어짐 | 8.3 | 4.2 | 4.2 | 2.1 | 4.2 | 2.1 | 0.0 |
| gap/forward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 97.9 | 97.9 | 100.0 | 97.9 | 100.0 |
| gap/forward/L11/v0.9 | clean/시도 | 77.1 | 81.2 | 78.7 | 78.7 | 79.2 | 76.6 | 79.2 |
| gap/forward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 2.1 | 2.1 | 0.0 | 2.1 | 0.0 |
| gap/forward/L14/v0.5 | 건넘 | 87.5 | 77.1 | 89.6 | 87.5 | 77.1 | 77.1 | 85.4 |
| gap/forward/L14/v0.5 | clean/시도 | 78.6 | 73.0 | 79.1 | 83.3 | 86.5 | 75.7 | 78.0 |
| gap/forward/L14/v0.5 | 넘어짐 | 12.5 | 22.9 | 10.4 | 12.5 | 22.9 | 22.9 | 14.6 |
| gap/forward/L14/v0.9 | 건넘 | 93.8 | 91.7 | 89.6 | 91.7 | 93.8 | 93.8 | 91.7 |
| gap/forward/L14/v0.9 | clean/시도 | 64.4 | 84.1 | 67.4 | 54.5 | 62.2 | 64.4 | 61.4 |
| gap/forward/L14/v0.9 | 넘어짐 | 6.2 | 8.3 | 10.4 | 8.3 | 6.2 | 6.2 | 8.3 |
| gap/backward/L7/v0.5 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 | 97.9 | 100.0 |
| gap/backward/L7/v0.5 | clean/시도 | 74.5 | 70.8 | 75.0 | 77.1 | 72.9 | 85.1 | 64.6 |
| gap/backward/L7/v0.5 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 | 2.1 | 0.0 |
| gap/backward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.9 | clean/시도 | 89.6 | 85.4 | 85.4 | 87.5 | 85.4 | 87.5 | 85.4 |
| gap/backward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 건넘 | 100.0 | 100.0 | 97.9 | 95.8 | 100.0 | 100.0 | 97.9 |
| gap/backward/L11/v0.5 | clean/시도 | 70.8 | 77.1 | 66.0 | 67.4 | 77.1 | 77.1 | 72.3 |
| gap/backward/L11/v0.5 | 넘어짐 | 0.0 | 0.0 | 2.1 | 4.2 | 0.0 | 0.0 | 2.1 |
| gap/backward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.9 | clean/시도 | 83.3 | 83.3 | 87.5 | 87.5 | 81.2 | 89.6 | 83.3 |
| gap/backward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 건넘 | 100.0 | 95.8 | 100.0 | 95.8 | 100.0 | 100.0 | 95.8 |
| gap/backward/L14/v0.5 | clean/시도 | 47.9 | 60.9 | 62.5 | 47.8 | 56.2 | 62.5 | 58.7 |
| gap/backward/L14/v0.5 | 넘어짐 | 0.0 | 4.2 | 0.0 | 4.2 | 0.0 | 0.0 | 4.2 |
| gap/backward/L14/v0.9 | 건넘 | 100.0 | 95.8 | 100.0 | 100.0 | 100.0 | 97.9 | 100.0 |
| gap/backward/L14/v0.9 | clean/시도 | 58.3 | 43.5 | 72.9 | 62.5 | 64.6 | 57.4 | 66.7 |
| gap/backward/L14/v0.9 | 넘어짐 | 0.0 | 4.2 | 0.0 | 0.0 | 0.0 | 2.1 | 0.0 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 건넘 | 97.6 | 96.0 | 97.6 | 97.2 | 97.2 | 96.9 | 97.4 |
| 평균 | clean/시도 | 74.1 | 73.5 | 74.9 | 74.2 | 75.7 | 76.8 | 74.4 |
| 평균 | 넘어짐 | 2.4 | 3.8 | 2.4 | 2.8 | 2.8 | 3.1 | 2.6 |

### 계단

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| stairs/pyramid_stairs/up/L9 | 통과 | 87.5 | 84.4 | 90.6 | 93.8 | 84.4 | 93.8 | 87.5 |
| stairs/pyramid_stairs/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L9 | 몸통 접촉 | 34.4 | 25.0 | 34.4 | 37.5 | 18.8 | 31.2 | 31.2 |
| stairs/pyramid_stairs/up/L14 | 통과 | 90.6 | 87.5 | 93.8 | 87.5 | 90.6 | 87.5 | 90.6 |
| stairs/pyramid_stairs/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L14 | 몸통 접촉 | 56.2 | 28.1 | 46.9 | 65.6 | 50.0 | 53.1 | 59.4 |
| stairs/pyramid_stairs/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L9 | 몸통 접촉 | 50.0 | 75.0 | 59.4 | 53.1 | 53.1 | 62.5 | 56.2 |
| stairs/pyramid_stairs/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L14 | 몸통 접촉 | 93.8 | 93.8 | 93.8 | 87.5 | 90.6 | 90.6 | 93.8 |
| stairs/pyramid_stairs_inv/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L9 | 몸통 접촉 | 28.1 | 28.1 | 46.9 | 53.1 | 50.0 | 56.2 | 59.4 |
| stairs/pyramid_stairs_inv/up/L14 | 통과 | 100.0 | 90.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L14 | 넘어짐 | 0.0 | 9.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L14 | 몸통 접촉 | 50.0 | 37.5 | 50.0 | 62.5 | 46.9 | 56.2 | 53.1 |
| stairs/pyramid_stairs_inv/down/L9 | 통과 | 100.0 | 93.8 | 100.0 | 96.9 | 100.0 | 93.8 | 100.0 |
| stairs/pyramid_stairs_inv/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L9 | 몸통 접촉 | 34.4 | 75.0 | 37.5 | 34.4 | 37.5 | 31.2 | 40.6 |
| stairs/pyramid_stairs_inv/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L14 | 몸통 접촉 | 87.5 | 93.8 | 93.8 | 68.8 | 87.5 | 75.0 | 62.5 |
| stairs/pyramid_stairs_wide/up/L9 | 통과 | 96.9 | 100.0 | 96.9 | 100.0 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L9 | 몸통 접촉 | 21.9 | 18.8 | 15.6 | 25.0 | 18.8 | 34.4 | 21.9 |
| stairs/pyramid_stairs_wide/up/L14 | 통과 | 87.5 | 71.9 | 93.8 | 87.5 | 84.4 | 90.6 | 90.6 |
| stairs/pyramid_stairs_wide/up/L14 | 넘어짐 | 3.1 | 15.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L14 | 몸통 접촉 | 37.5 | 40.6 | 46.9 | 46.9 | 40.6 | 53.1 | 46.9 |
| stairs/pyramid_stairs_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L9 | 몸통 접촉 | 62.5 | 90.6 | 68.8 | 78.1 | 78.1 | 50.0 | 71.9 |
| stairs/pyramid_stairs_wide/down/L14 | 통과 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L14 | 넘어짐 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 몸통 접촉 | 50.0 | 43.8 | 53.1 | 31.2 | 28.1 | 28.1 | 28.1 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 통과 | 96.9 | 87.5 | 93.8 | 93.8 | 93.8 | 93.8 | 96.9 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 넘어짐 | 3.1 | 12.5 | 6.2 | 6.2 | 6.2 | 6.2 | 3.1 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 몸통 접촉 | 53.1 | 56.2 | 56.2 | 40.6 | 43.8 | 50.0 | 56.2 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 통과 | 96.9 | 93.8 | 96.9 | 96.9 | 100.0 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 몸통 접촉 | 50.0 | 84.4 | 68.8 | 62.5 | 68.8 | 53.1 | 65.6 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 몸통 접촉 | 93.8 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 통과 | 96.9 | 96.9 | 96.9 | 100.0 | 96.9 | 100.0 | 96.9 |
| stairs/pyramid_stairs_nose/up/L9 | 넘어짐 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/up/L9 | 몸통 접촉 | 25.0 | 53.1 | 43.8 | 28.1 | 18.8 | 25.0 | 37.5 |
| stairs/pyramid_stairs_nose/up/L14 | 통과 | 81.2 | 75.0 | 71.9 | 78.1 | 81.2 | 78.1 | 65.6 |
| stairs/pyramid_stairs_nose/up/L14 | 넘어짐 | 9.4 | 12.5 | 21.9 | 12.5 | 15.6 | 12.5 | 28.1 |
| stairs/pyramid_stairs_nose/up/L14 | 몸통 접촉 | 81.2 | 68.8 | 81.2 | 81.2 | 78.1 | 84.4 | 87.5 |
| stairs/pyramid_stairs_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L9 | 몸통 접촉 | 65.6 | 93.8 | 40.6 | 53.1 | 65.6 | 56.2 | 50.0 |
| stairs/pyramid_stairs_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 몸통 접촉 | 62.5 | 75.0 | 65.6 | 56.2 | 68.8 | 62.5 | 56.2 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 통과 | 100.0 | 84.4 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 넘어짐 | 0.0 | 15.6 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 몸통 접촉 | 71.9 | 78.1 | 78.1 | 84.4 | 75.0 | 75.0 | 84.4 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 몸통 접촉 | 43.8 | 78.1 | 53.1 | 59.4 | 50.0 | 31.2 | 40.6 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 통과 | 96.9 | 96.9 | 93.8 | 93.8 | 93.8 | 93.8 | 93.8 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 | 3.1 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 몸통 접촉 | 68.8 | 87.5 | 93.8 | 68.8 | 90.6 | 71.9 | 84.4 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 통과 | 97.1 | 94.1 ⚠ | 97.0 | 96.7 | 96.7 | 96.9 | 96.5 |
| 평균 | 넘어짐 | 0.7 | 3.0 ⚠ | 1.3 | 0.9 | 0.9 | 0.8 | 1.4 |
| 평균 | 몸통 접촉 | 59.2 | 67.6 | 63.7 | 61.6 | 60.8 | 59.6 | 62.0 |

### 경사·거친 지면

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| terrain/hf_pyramid_slope/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/hf_pyramid_slope/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/hf_pyramid_slope/L14 | 속도 오차 m/s | 0.176 | 0.194 | 0.172 | 0.174 | 0.177 | 0.176 | 0.172 |
| terrain/hf_pyramid_slope_inv/L14 | 통과 | 81.2 | 75.0 | 81.2 | 78.1 | 81.2 | 78.1 | 81.2 |
| terrain/hf_pyramid_slope_inv/L14 | 넘어짐 | 0.0 | 12.5 | 0.0 | 6.2 | 3.1 | 3.1 | 6.2 |
| terrain/hf_pyramid_slope_inv/L14 | 속도 오차 m/s | 0.304 | 0.330 | 0.299 | 0.310 | 0.293 | 0.306 | 0.284 |
| terrain/random_rough/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/random_rough/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/random_rough/L14 | 속도 오차 m/s | 0.162 | 0.150 | 0.161 | 0.158 | 0.157 | 0.169 | 0.162 |
| terrain/ring_fence/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/ring_fence/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/ring_fence/L14 | 속도 오차 m/s | 0.162 | 0.160 | 0.159 | 0.160 | 0.157 | 0.168 | 0.163 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 통과 | 95.3 | 93.8 | 95.3 | 94.5 | 95.3 | 94.5 | 95.3 |
| 평균 | 넘어짐 | 0.0 | 3.1 ⚠ | 0.0 | 1.6 | 0.8 | 0.8 | 1.6 |
| 평균 | 속도 오차 m/s | 0.201 | 0.209 | 0.198 | 0.201 | 0.196 | 0.205 | 0.195 |

## 조건 ouster_only (라이다 ouster, 탑재물 5.0 kg, 밀기 False, 계단 외란 False)

### 좁은 통로

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| narrow/narrow_corridor/pass_wide (폭 0.80) | 통과 | 93.8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 무접촉 | 65.6 | 100.0 | 100.0 | 100.0 | 96.9 | 96.9 | 96.9 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.039 | 0.091 | 0.104 | 0.102 | 0.096 | 0.090 | 0.105 |
| narrow/narrow_corridor/pass_tight (폭 0.66) | 통과 | 78.1 | 68.8 | 68.8 | 15.6 | 28.1 | 43.8 | 37.5 |
| narrow/narrow_corridor/pass_tight (폭 0.66) | 무접촉 | 34.4 | 84.4 | 87.5 | 96.9 | 96.9 | 84.4 | 90.6 |
| narrow/narrow_corridor/pass_tight (폭 0.66) | 벽 거리 중앙값 m | -0.004 | 0.046 | 0.046 | 0.263 | 0.266 | 0.073 | 0.160 |
| narrow/narrow_corridor/block_margin (폭 0.66) | 서기 | 0.0 | 87.5 | 90.6 | 90.6 | 84.4 | 90.6 | 90.6 |
| narrow/narrow_corridor/block_margin (폭 0.66) | 침범 | 100.0 | 6.2 | 9.4 | 9.4 | 12.5 | 9.4 | 9.4 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 서기 | 0.0 | 90.6 | 100.0 | 93.8 | 96.9 | 96.9 | 93.8 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 침범 | 100.0 | 6.2 | 0.0 | 0.0 | 0.0 | 0.0 | 3.1 |
| narrow/narrow_door_pos/pass_wide (폭 0.79) | 통과 | 87.5 | 96.9 | 100.0 | 100.0 | 96.9 | 100.0 | 96.9 |
| narrow/narrow_door_pos/pass_wide (폭 0.79) | 무접촉 | 71.9 | 93.8 | 90.6 | 96.9 | 87.5 | 90.6 | 90.6 |
| narrow/narrow_door_pos/pass_wide (폭 0.79) | 벽 거리 중앙값 m | 0.040 | 0.035 | 0.089 | 0.081 | 0.071 | 0.071 | 0.074 |
| narrow/narrow_door_pos/pass_tight (폭 0.67) | 통과 | 75.0 | 90.6 | 90.6 | 18.8 | 90.6 | 78.1 | 71.9 |
| narrow/narrow_door_pos/pass_tight (폭 0.67) | 무접촉 | 43.8 | 87.5 | 75.0 | 90.6 | 71.9 | 65.6 | 75.0 |
| narrow/narrow_door_pos/pass_tight (폭 0.67) | 벽 거리 중앙값 m | 0.001 | 0.004 | 0.025 | 0.353 | 0.022 | 0.023 | 0.028 |
| narrow/narrow_door_pos/block_margin (폭 0.67) | 서기 | 0.0 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/block_margin (폭 0.67) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_pos/block_narrow (폭 0.52) | 서기 | 0.0 | 96.9 | 100.0 | 96.9 | 100.0 | 96.9 | 96.9 |
| narrow/narrow_door_pos/block_narrow (폭 0.52) | 침범 | 100.0 | 0.0 | 0.0 | 3.1 | 0.0 | 3.1 | 3.1 |
| narrow/narrow_door_neg/pass_wide (폭 0.80) | 통과 | 93.8 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.80) | 무접촉 | 87.5 | 87.5 | 96.9 | 96.9 | 96.9 | 96.9 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.062 | 0.050 | 0.100 | 0.100 | 0.094 | 0.093 | 0.105 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 통과 | 90.6 | 81.2 | 78.1 | 18.8 | 78.1 | 78.1 | 62.5 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 무접촉 | 43.8 | 71.9 | 75.0 | 96.9 | 71.9 | 71.9 | 78.1 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 벽 거리 중앙값 m | -0.000 | -0.001 | 0.023 | 0.448 | 0.021 | 0.025 | 0.047 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 서기 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.53) | 서기 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.53) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 통과 | 90.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 무접촉 | 28.1 | 90.6 | 81.2 | 93.8 | 87.5 | 90.6 | 87.5 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 벽 거리 중앙값 m | -0.007 | 0.033 | 0.038 | 0.030 | 0.044 | 0.035 | 0.042 |
| narrow/narrow_bend_left/pass_tight (폭 0.65) | 통과 | 50.0 | 50.0 | 34.4 | 12.5 | 12.5 | 12.5 | 15.6 |
| narrow/narrow_bend_left/pass_tight (폭 0.65) | 무접촉 | 9.4 | 56.2 | 65.6 | 87.5 | 84.4 | 78.1 | 84.4 |
| narrow/narrow_bend_left/pass_tight (폭 0.65) | 벽 거리 중앙값 m | -0.020 | -0.006 | 0.071 | 0.377 | 0.216 | 0.334 | 0.331 |
| narrow/narrow_bend_left/block_margin (폭 0.65) | 서기 | 0.0 | 84.4 | 93.8 | 90.6 | 87.5 | 96.9 | 96.9 |
| narrow/narrow_bend_left/block_margin (폭 0.65) | 침범 | 100.0 | 15.6 | 3.1 | 6.2 | 12.5 | 3.1 | 3.1 |
| narrow/narrow_bend_left/block_narrow (폭 0.52) | 서기 | 0.0 | 78.1 | 100.0 | 87.5 | 100.0 | 96.9 | 90.6 |
| narrow/narrow_bend_left/block_narrow (폭 0.52) | 침범 | 100.0 | 18.8 | 0.0 | 3.1 | 0.0 | 3.1 | 0.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.81) | 통과 | 87.5 | 93.8 | 96.9 | 93.8 | 93.8 | 93.8 | 93.8 |
| narrow/narrow_bend_right/pass_wide (폭 0.81) | 무접촉 | 37.5 | 87.5 | 84.4 | 93.8 | 84.4 | 84.4 | 87.5 |
| narrow/narrow_bend_right/pass_wide (폭 0.81) | 벽 거리 중앙값 m | -0.005 | 0.042 | 0.045 | 0.034 | 0.044 | 0.044 | 0.037 |
| narrow/narrow_bend_right/pass_tight (폭 0.65) | 통과 | 43.8 | 21.9 | 21.9 | 6.2 | 6.2 | 15.6 | 9.4 |
| narrow/narrow_bend_right/pass_tight (폭 0.65) | 무접촉 | 9.4 | 56.2 | 71.9 | 93.8 | 90.6 | 78.1 | 87.5 |
| narrow/narrow_bend_right/pass_tight (폭 0.65) | 벽 거리 중앙값 m | -0.019 | 0.003 | 0.253 | 0.360 | 0.317 | 0.375 | 0.369 |
| narrow/narrow_bend_right/block_margin (폭 0.65) | 서기 | 0.0 | 87.5 | 93.8 | 93.8 | 90.6 | 100.0 | 96.9 |
| narrow/narrow_bend_right/block_margin (폭 0.65) | 침범 | 100.0 | 12.5 | 6.2 | 3.1 | 9.4 | 0.0 | 3.1 |
| narrow/narrow_bend_right/block_narrow (폭 0.50) | 서기 | 0.0 | 87.5 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/block_narrow (폭 0.50) | 침범 | 100.0 | 9.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 통과 | 79.1 | 80.3 | 79.1 | 56.2 | 70.6 | 72.2 | 68.8 |
| 평균 | 무접촉 | 43.1 | 81.6 | 82.8 | 94.7 | 86.9 | 83.8 | 87.8 |
| 평균 | 벽 거리 중앙값 m | 0.008 | 0.030 | 0.079 | 0.215 | 0.119 | 0.116 | 0.130 |
| 평균 | 서기 | 0.0 | 90.9 | 97.5 | 95.3 | 95.9 | 97.8 | 96.6 |
| 평균 | 침범 | 100.0 | 6.9 | 1.9 | 2.5 | 3.4 | 1.9 | 2.2 |

### 막힌 통로

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| dead_end/narrow_dead_end/L7 | 서기 | 3.1 | 78.1 | 96.9 | 96.9 | 100.0 | 100.0 | 96.9 |
| dead_end/narrow_dead_end/L7 | 침범 | 96.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.1 |
| dead_end/narrow_dead_end/L14 | 서기 | 6.2 | 96.9 | 96.9 | 96.9 | 93.8 | 93.8 | 96.9 |
| dead_end/narrow_dead_end/L14 | 침범 | 90.6 | 0.0 | 0.0 | 0.0 | 3.1 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 서기 | 4.7 | 87.5 | 96.9 | 96.9 | 96.9 | 96.9 | 96.9 |
| 평균 | 침범 | 93.8 | 0.0 | 0.0 | 0.0 | 1.6 | 0.0 | 1.6 |

## 조건 mid360_only (라이다 mid360, 탑재물 5.0 kg, 밀기 False, 계단 외란 False)

### 좁은 통로

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| narrow/narrow_corridor/pass_wide (폭 0.80) | 통과 | 93.8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 무접촉 | 65.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.039 | 0.104 | 0.092 | 0.105 | 0.096 | 0.102 | 0.113 |
| narrow/narrow_corridor/pass_tight (폭 0.66) | 통과 | 78.1 | 84.4 | 75.0 | 25.0 | 46.9 | 59.4 | 37.5 |
| narrow/narrow_corridor/pass_tight (폭 0.66) | 무접촉 | 34.4 | 93.8 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_tight (폭 0.66) | 벽 거리 중앙값 m | -0.004 | 0.057 | 0.054 | 0.351 | 0.154 | 0.066 | 0.288 |
| narrow/narrow_corridor/block_margin (폭 0.66) | 서기 | 0.0 | 93.8 | 96.9 | 96.9 | 90.6 | 96.9 | 96.9 |
| narrow/narrow_corridor/block_margin (폭 0.66) | 침범 | 100.0 | 3.1 | 3.1 | 3.1 | 9.4 | 3.1 | 3.1 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 서기 | 0.0 | 87.5 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 침범 | 100.0 | 6.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.79) | 통과 | 87.5 | 90.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.79) | 무접촉 | 71.9 | 87.5 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.79) | 벽 거리 중앙값 m | 0.040 | 0.015 | 0.097 | 0.105 | 0.103 | 0.095 | 0.096 |
| narrow/narrow_door_pos/pass_tight (폭 0.67) | 통과 | 75.0 | 96.9 | 100.0 | 40.6 | 100.0 | 96.9 | 90.6 |
| narrow/narrow_door_pos/pass_tight (폭 0.67) | 무접촉 | 43.8 | 62.5 | 96.9 | 96.9 | 84.4 | 90.6 | 84.4 |
| narrow/narrow_door_pos/pass_tight (폭 0.67) | 벽 거리 중앙값 m | 0.001 | -0.003 | 0.049 | 0.248 | 0.054 | 0.040 | 0.052 |
| narrow/narrow_door_pos/block_margin (폭 0.67) | 서기 | 0.0 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/block_margin (폭 0.67) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_pos/block_narrow (폭 0.52) | 서기 | 0.0 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 |
| narrow/narrow_door_pos/block_narrow (폭 0.52) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 3.1 | 0.0 | 0.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.80) | 통과 | 93.8 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.80) | 무접촉 | 87.5 | 90.6 | 100.0 | 100.0 | 96.9 | 96.9 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.062 | 0.042 | 0.110 | 0.108 | 0.109 | 0.110 | 0.103 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 통과 | 90.6 | 96.9 | 100.0 | 25.0 | 93.8 | 93.8 | 87.5 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 무접촉 | 43.8 | 75.0 | 90.6 | 100.0 | 90.6 | 84.4 | 96.9 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 벽 거리 중앙값 m | -0.000 | 0.003 | 0.050 | 0.359 | 0.050 | 0.044 | 0.049 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 서기 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.53) | 서기 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.53) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 통과 | 90.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 무접촉 | 28.1 | 96.9 | 100.0 | 100.0 | 100.0 | 93.8 | 100.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 벽 거리 중앙값 m | -0.007 | 0.036 | 0.061 | 0.058 | 0.063 | 0.053 | 0.057 |
| narrow/narrow_bend_left/pass_tight (폭 0.65) | 통과 | 50.0 | 75.0 | 43.8 | 3.1 | 37.5 | 34.4 | 25.0 |
| narrow/narrow_bend_left/pass_tight (폭 0.65) | 무접촉 | 9.4 | 68.8 | 81.2 | 100.0 | 93.8 | 81.2 | 87.5 |
| narrow/narrow_bend_left/pass_tight (폭 0.65) | 벽 거리 중앙값 m | -0.020 | 0.004 | 0.018 | 0.394 | 0.175 | 0.363 | 0.324 |
| narrow/narrow_bend_left/block_margin (폭 0.65) | 서기 | 0.0 | 90.6 | 96.9 | 100.0 | 93.8 | 96.9 | 90.6 |
| narrow/narrow_bend_left/block_margin (폭 0.65) | 침범 | 100.0 | 9.4 | 3.1 | 0.0 | 6.2 | 3.1 | 9.4 |
| narrow/narrow_bend_left/block_narrow (폭 0.52) | 서기 | 0.0 | 84.4 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/block_narrow (폭 0.52) | 침범 | 100.0 | 12.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.81) | 통과 | 87.5 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.81) | 무접촉 | 37.5 | 96.9 | 100.0 | 96.9 | 100.0 | 93.8 | 100.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.81) | 벽 거리 중앙값 m | -0.005 | 0.042 | 0.061 | 0.063 | 0.059 | 0.056 | 0.061 |
| narrow/narrow_bend_right/pass_tight (폭 0.65) | 통과 | 43.8 | 71.9 | 31.2 | 6.2 | 28.1 | 31.2 | 15.6 |
| narrow/narrow_bend_right/pass_tight (폭 0.65) | 무접촉 | 9.4 | 68.8 | 81.2 | 96.9 | 90.6 | 78.1 | 93.8 |
| narrow/narrow_bend_right/pass_tight (폭 0.65) | 벽 거리 중앙값 m | -0.019 | 0.009 | 0.341 | 0.409 | 0.292 | 0.287 | 0.345 |
| narrow/narrow_bend_right/block_margin (폭 0.65) | 서기 | 0.0 | 90.6 | 96.9 | 96.9 | 96.9 | 100.0 | 100.0 |
| narrow/narrow_bend_right/block_margin (폭 0.65) | 침범 | 100.0 | 9.4 | 0.0 | 0.0 | 3.1 | 0.0 | 0.0 |
| narrow/narrow_bend_right/block_narrow (폭 0.50) | 서기 | 0.0 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/block_narrow (폭 0.50) | 침범 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 통과 | 79.1 | 91.2 | 85.0 | 60.0 | 80.6 | 81.6 | 75.6 |
| 평균 | 무접촉 | 43.1 | 84.1 | 94.7 | 99.1 | 95.6 | 91.6 | 96.2 |
| 평균 | 벽 거리 중앙값 m | 0.008 | 0.031 | 0.093 | 0.220 | 0.116 | 0.122 | 0.149 |
| 평균 | 서기 | 0.0 | 94.1 | 98.8 | 99.4 | 97.8 | 99.4 | 98.4 |
| 평균 | 침범 | 100.0 | 4.1 | 0.6 | 0.3 | 2.2 | 0.6 | 1.2 |

### 막힌 통로

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| dead_end/narrow_dead_end/L7 | 서기 | 3.1 | 87.5 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 |
| dead_end/narrow_dead_end/L7 | 침범 | 96.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dead_end/narrow_dead_end/L14 | 서기 | 6.2 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| dead_end/narrow_dead_end/L14 | 침범 | 90.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 서기 | 4.7 | 92.2 | 100.0 | 98.4 | 100.0 | 100.0 | 100.0 |
| 평균 | 침범 | 93.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## 조건 disturbed (라이다 both, 탑재물 5.0 kg, 밀기 True, 계단 외란 True)

### 갭

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| gap/forward/L7/v0.5 | 건넘 | 100.0 | 93.8 | 100.0 | 97.9 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.5 | clean/시도 | 70.8 | 60.0 | 75.0 | 76.6 | 81.2 | 62.5 | 60.4 |
| gap/forward/L7/v0.5 | 넘어짐 | 0.0 | 6.2 | 0.0 | 2.1 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.9 | clean/시도 | 87.5 | 83.3 | 79.2 | 83.3 | 81.2 | 81.2 | 85.4 |
| gap/forward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 건넘 | 97.9 | 93.8 | 91.7 | 93.8 | 87.5 | 95.8 | 95.8 |
| gap/forward/L11/v0.5 | clean/시도 | 76.6 | 53.3 | 77.3 | 82.2 | 69.0 | 69.6 | 76.1 |
| gap/forward/L11/v0.5 | 넘어짐 | 2.1 | 6.2 | 8.3 | 6.2 | 12.5 | 4.2 | 4.2 |
| gap/forward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 95.8 | 97.9 | 100.0 | 100.0 | 100.0 |
| gap/forward/L11/v0.9 | clean/시도 | 85.4 | 85.4 | 76.1 | 74.5 | 87.5 | 85.4 | 79.2 |
| gap/forward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 4.2 | 2.1 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.5 | 건넘 | 85.4 | 75.0 | 87.5 | 85.4 | 87.5 | 83.3 | 85.4 |
| gap/forward/L14/v0.5 | clean/시도 | 73.2 | 83.3 | 81.0 | 73.2 | 76.2 | 80.0 | 87.8 |
| gap/forward/L14/v0.5 | 넘어짐 | 14.6 | 25.0 | 12.5 | 14.6 | 12.5 | 16.7 | 14.6 |
| gap/forward/L14/v0.9 | 건넘 | 91.7 | 89.6 | 95.8 | 91.7 | 87.5 | 93.8 | 89.6 |
| gap/forward/L14/v0.9 | clean/시도 | 61.4 | 81.4 | 76.1 | 61.4 | 61.9 | 51.1 | 74.4 |
| gap/forward/L14/v0.9 | 넘어짐 | 8.3 | 10.4 | 4.2 | 8.3 | 12.5 | 6.2 | 10.4 |
| gap/backward/L7/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.5 | clean/시도 | 79.2 | 79.2 | 77.1 | 75.0 | 81.2 | 70.8 | 75.0 |
| gap/backward/L7/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.9 | clean/시도 | 89.6 | 81.2 | 81.2 | 87.5 | 87.5 | 81.2 | 81.2 |
| gap/backward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 건넘 | 100.0 | 100.0 | 95.8 | 97.9 | 100.0 | 100.0 | 95.8 |
| gap/backward/L11/v0.5 | clean/시도 | 75.0 | 70.8 | 67.4 | 66.0 | 79.2 | 66.7 | 63.0 |
| gap/backward/L11/v0.5 | 넘어짐 | 0.0 | 0.0 | 4.2 | 2.1 | 0.0 | 0.0 | 4.2 |
| gap/backward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.9 | clean/시도 | 83.3 | 79.2 | 83.3 | 83.3 | 87.5 | 83.3 | 87.5 |
| gap/backward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 건넘 | 100.0 | 93.8 | 100.0 | 91.7 | 95.8 | 97.9 | 100.0 |
| gap/backward/L14/v0.5 | clean/시도 | 58.3 | 60.0 | 56.2 | 50.0 | 47.8 | 57.4 | 54.2 |
| gap/backward/L14/v0.5 | 넘어짐 | 0.0 | 6.2 | 0.0 | 8.3 | 4.2 | 2.1 | 0.0 |
| gap/backward/L14/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L14/v0.9 | clean/시도 | 75.0 | 75.0 | 72.9 | 68.8 | 72.9 | 75.0 | 66.7 |
| gap/backward/L14/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 건넘 | 97.9 | 95.5 ⚠ | 97.2 | 96.4 | 96.5 | 97.6 | 97.2 |
| 평균 | clean/시도 | 76.3 | 74.4 | 75.2 | 73.5 ⚠ | 76.1 | 72.0 ⚠ | 74.2 ⚠ |
| 평균 | 넘어짐 | 2.1 | 4.5 ⚠ | 2.8 | 3.6 | 3.5 | 2.4 | 2.8 |

### 계단

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| stairs/pyramid_stairs/up/L9 | 통과 | 100.0 | 96.9 | 96.9 | 96.9 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L9 | 몸통 접촉 | 25.0 | 9.4 | 31.2 | 21.9 | 25.0 | 43.8 | 28.1 |
| stairs/pyramid_stairs/up/L14 | 통과 | 87.5 | 84.4 | 84.4 | 84.4 | 84.4 | 81.2 | 90.6 |
| stairs/pyramid_stairs/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L14 | 몸통 접촉 | 40.6 | 31.2 | 50.0 | 46.9 | 43.8 | 40.6 | 46.9 |
| stairs/pyramid_stairs/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L9 | 몸통 접촉 | 53.1 | 75.0 | 46.9 | 46.9 | 50.0 | 31.2 | 43.8 |
| stairs/pyramid_stairs/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L14 | 몸통 접촉 | 93.8 | 93.8 | 84.4 | 84.4 | 93.8 | 93.8 | 90.6 |
| stairs/pyramid_stairs_inv/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.1 |
| stairs/pyramid_stairs_inv/up/L9 | 몸통 접촉 | 43.8 | 6.2 | 34.4 | 28.1 | 50.0 | 28.1 | 28.1 |
| stairs/pyramid_stairs_inv/up/L14 | 통과 | 96.9 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L14 | 넘어짐 | 3.1 | 0.0 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L14 | 몸통 접촉 | 56.2 | 43.8 | 65.6 | 62.5 | 53.1 | 53.1 | 62.5 |
| stairs/pyramid_stairs_inv/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L9 | 몸통 접촉 | 34.4 | 40.6 | 28.1 | 21.9 | 34.4 | 31.2 | 43.8 |
| stairs/pyramid_stairs_inv/down/L14 | 통과 | 96.9 | 100.0 | 96.9 | 96.9 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L14 | 몸통 접촉 | 87.5 | 81.2 | 84.4 | 81.2 | 81.2 | 65.6 | 71.9 |
| stairs/pyramid_stairs_wide/up/L9 | 통과 | 96.9 | 96.9 | 96.9 | 93.8 | 93.8 | 96.9 | 93.8 |
| stairs/pyramid_stairs_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L9 | 몸통 접촉 | 31.2 | 9.4 | 15.6 | 18.8 | 25.0 | 21.9 | 25.0 |
| stairs/pyramid_stairs_wide/up/L14 | 통과 | 90.6 | 84.4 | 81.2 | 78.1 | 87.5 | 84.4 | 81.2 |
| stairs/pyramid_stairs_wide/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 3.1 | 3.1 | 0.0 | 3.1 |
| stairs/pyramid_stairs_wide/up/L14 | 몸통 접촉 | 46.9 | 59.4 | 50.0 | 59.4 | 40.6 | 43.8 | 56.2 |
| stairs/pyramid_stairs_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L9 | 몸통 접촉 | 50.0 | 84.4 | 50.0 | 62.5 | 62.5 | 50.0 | 59.4 |
| stairs/pyramid_stairs_wide/down/L14 | 통과 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L14 | 넘어짐 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L14 | 몸통 접촉 | 96.9 | 90.6 | 100.0 | 100.0 | 90.6 | 100.0 | 93.8 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 몸통 접촉 | 56.2 | 21.9 | 43.8 | 56.2 | 46.9 | 46.9 | 59.4 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 통과 | 100.0 | 93.8 | 90.6 | 93.8 | 100.0 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 넘어짐 | 0.0 | 6.2 | 9.4 | 6.2 | 0.0 | 0.0 | 3.1 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 몸통 접촉 | 71.9 | 56.2 | 56.2 | 62.5 | 65.6 | 62.5 | 62.5 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 통과 | 96.9 | 93.8 | 96.9 | 96.9 | 96.9 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 몸통 접촉 | 46.9 | 78.1 | 59.4 | 21.9 | 59.4 | 34.4 | 37.5 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 통과 | 100.0 | 100.0 | 93.8 | 96.9 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 몸통 접촉 | 96.9 | 90.6 | 100.0 | 96.9 | 100.0 | 90.6 | 90.6 |
| stairs/pyramid_stairs_nose/up/L9 | 통과 | 96.9 | 90.6 | 93.8 | 84.4 | 93.8 | 87.5 | 93.8 |
| stairs/pyramid_stairs_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/up/L9 | 몸통 접촉 | 53.1 | 31.2 | 43.8 | 46.9 | 46.9 | 37.5 | 37.5 |
| stairs/pyramid_stairs_nose/up/L14 | 통과 | 65.6 | 81.2 | 78.1 | 75.0 | 84.4 | 84.4 | 71.9 |
| stairs/pyramid_stairs_nose/up/L14 | 넘어짐 | 21.9 | 12.5 | 21.9 | 18.8 | 12.5 | 9.4 | 25.0 |
| stairs/pyramid_stairs_nose/up/L14 | 몸통 접촉 | 87.5 | 46.9 | 78.1 | 71.9 | 68.8 | 75.0 | 65.6 |
| stairs/pyramid_stairs_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L9 | 몸통 접촉 | 71.9 | 87.5 | 75.0 | 71.9 | 78.1 | 62.5 | 68.8 |
| stairs/pyramid_stairs_nose/down/L14 | 통과 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L14 | 넘어짐 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 93.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 통과 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 3.1 | 0.0 | 0.0 | 3.1 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 몸통 접촉 | 68.8 | 43.8 | 68.8 | 62.5 | 71.9 | 71.9 | 68.8 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 통과 | 96.9 | 100.0 | 96.9 | 100.0 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 넘어짐 | 3.1 | 0.0 | 3.1 | 0.0 | 3.1 | 3.1 | 3.1 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 몸통 접촉 | 78.1 | 75.0 | 81.2 | 78.1 | 78.1 | 84.4 | 84.4 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 통과 | 100.0 | 96.9 | 96.9 | 100.0 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 몸통 접촉 | 43.8 | 62.5 | 59.4 | 40.6 | 62.5 | 53.1 | 56.2 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 통과 | 96.9 | 100.0 | 100.0 | 96.9 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.1 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 몸통 접촉 | 81.2 | 90.6 | 75.0 | 56.2 | 68.8 | 71.9 | 78.1 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 통과 | 96.7 | 96.4 | 95.8 | 95.3 | 97.1 | 96.6 | 96.4 |
| 평균 | 넘어짐 | 1.2 | 1.0 | 1.7 | 1.3 | 0.8 | 0.7 | 1.7 |
| 평균 | 몸통 접촉 | 63.2 | 58.7 | 61.5 | 58.3 | 62.4 | 58.1 | 60.8 |

## 조건 no_payload (라이다 both, 탑재물 0.0 kg, 밀기 False, 계단 외란 False)

### 갭

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| gap/forward/L7/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.5 | clean/시도 | 66.7 | 68.8 | 64.6 | 68.8 | 70.8 | 72.9 | 66.7 |
| gap/forward/L7/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.9 | clean/시도 | 85.4 | 83.3 | 77.1 | 85.4 | 87.5 | 75.0 | 87.5 |
| gap/forward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 건넘 | 93.8 | 89.6 | 97.9 | 93.8 | 93.8 | 91.7 | 93.8 |
| gap/forward/L11/v0.5 | clean/시도 | 77.8 | 65.1 | 76.6 | 77.8 | 77.8 | 77.3 | 73.3 |
| gap/forward/L11/v0.5 | 넘어짐 | 6.2 | 10.4 | 2.1 | 6.2 | 6.2 | 8.3 | 6.2 |
| gap/forward/L11/v0.9 | 건넘 | 97.9 | 95.8 | 97.9 | 97.9 | 97.9 | 97.9 | 97.9 |
| gap/forward/L11/v0.9 | clean/시도 | 72.3 | 89.1 | 70.2 | 74.5 | 72.3 | 74.5 | 70.2 |
| gap/forward/L11/v0.9 | 넘어짐 | 2.1 | 4.2 | 2.1 | 2.1 | 2.1 | 2.1 | 2.1 |
| gap/forward/L14/v0.5 | 건넘 | 85.4 | 72.9 | 83.3 | 87.5 | 79.2 | 85.4 | 85.4 |
| gap/forward/L14/v0.5 | clean/시도 | 82.9 | 71.4 | 82.5 | 78.6 | 81.6 | 78.0 | 70.7 |
| gap/forward/L14/v0.5 | 넘어짐 | 14.6 | 27.1 | 16.7 | 12.5 | 20.8 | 14.6 | 14.6 |
| gap/forward/L14/v0.9 | 건넘 | 91.7 | 91.7 | 93.8 | 93.8 | 95.8 | 95.8 | 97.9 |
| gap/forward/L14/v0.9 | clean/시도 | 81.8 | 90.9 | 80.0 | 62.2 | 63.0 | 71.7 | 68.1 |
| gap/forward/L14/v0.9 | 넘어짐 | 8.3 | 8.3 | 6.2 | 6.2 | 4.2 | 4.2 | 2.1 |
| gap/backward/L7/v0.5 | 건넘 | 100.0 | 100.0 | 95.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.5 | clean/시도 | 70.8 | 75.0 | 65.2 | 70.8 | 58.3 | 77.1 | 58.3 |
| gap/backward/L7/v0.5 | 넘어짐 | 0.0 | 0.0 | 4.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.9 | clean/시도 | 79.2 | 79.2 | 81.2 | 83.3 | 81.2 | 79.2 | 81.2 |
| gap/backward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 건넘 | 100.0 | 100.0 | 97.9 | 95.8 | 95.8 | 100.0 | 97.9 |
| gap/backward/L11/v0.5 | clean/시도 | 72.9 | 68.8 | 57.4 | 58.7 | 65.2 | 64.6 | 70.2 |
| gap/backward/L11/v0.5 | 넘어짐 | 0.0 | 0.0 | 2.1 | 4.2 | 4.2 | 0.0 | 2.1 |
| gap/backward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.9 | clean/시도 | 83.3 | 77.1 | 79.2 | 81.2 | 77.1 | 79.2 | 83.3 |
| gap/backward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 건넘 | 100.0 | 95.8 | 97.9 | 100.0 | 93.8 | 97.9 | 97.9 |
| gap/backward/L14/v0.5 | clean/시도 | 52.1 | 63.0 | 48.9 | 45.8 | 44.4 | 68.1 | 40.4 |
| gap/backward/L14/v0.5 | 넘어짐 | 0.0 | 4.2 | 2.1 | 0.0 | 6.2 | 2.1 | 2.1 |
| gap/backward/L14/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L14/v0.9 | clean/시도 | 85.4 | 83.3 | 77.1 | 72.9 | 72.9 | 68.8 | 79.2 |
| gap/backward/L14/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 건넘 | 97.4 | 95.5 | 97.0 | 97.4 | 96.4 | 97.4 | 97.6 |
| 평균 | clean/시도 | 75.9 | 76.3 | 71.7 ⚠ | 71.7 ⚠ | 71.0 ⚠ | 73.9 ⚠ | 70.8 ⚠ |
| 평균 | 넘어짐 | 2.6 | 4.5 | 3.0 | 2.6 | 3.6 | 2.6 | 2.4 |

### 계단

| 셀 | 지표 | b6566 | v2u1111 | v21u465 | v21u355 | v21u536 | v21u284 | v21u319 |
|---|---|---|---|---|---|---|---|---|
| stairs/pyramid_stairs/up/L9 | 통과 | 100.0 | 100.0 | 96.9 | 100.0 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L9 | 몸통 접촉 | 25.0 | 12.5 | 25.0 | 31.2 | 15.6 | 21.9 | 31.2 |
| stairs/pyramid_stairs/up/L14 | 통과 | 93.8 | 93.8 | 93.8 | 93.8 | 93.8 | 93.8 | 100.0 |
| stairs/pyramid_stairs/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L14 | 몸통 접촉 | 28.1 | 15.6 | 46.9 | 43.8 | 43.8 | 40.6 | 46.9 |
| stairs/pyramid_stairs/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L9 | 몸통 접촉 | 43.8 | 75.0 | 50.0 | 46.9 | 59.4 | 53.1 | 56.2 |
| stairs/pyramid_stairs/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L14 | 몸통 접촉 | 84.4 | 93.8 | 81.2 | 84.4 | 87.5 | 93.8 | 84.4 |
| stairs/pyramid_stairs_inv/up/L9 | 통과 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L9 | 몸통 접촉 | 18.8 | 25.0 | 37.5 | 34.4 | 31.2 | 46.9 | 50.0 |
| stairs/pyramid_stairs_inv/up/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L14 | 몸통 접촉 | 31.2 | 37.5 | 71.9 | 75.0 | 53.1 | 53.1 | 75.0 |
| stairs/pyramid_stairs_inv/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L9 | 몸통 접촉 | 31.2 | 50.0 | 40.6 | 34.4 | 56.2 | 31.2 | 18.8 |
| stairs/pyramid_stairs_inv/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L14 | 몸통 접촉 | 96.9 | 87.5 | 84.4 | 93.8 | 100.0 | 96.9 | 90.6 |
| stairs/pyramid_stairs_wide/up/L9 | 통과 | 87.5 | 90.6 | 87.5 | 90.6 | 90.6 | 87.5 | 87.5 |
| stairs/pyramid_stairs_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L9 | 몸통 접촉 | 31.2 | 12.5 | 25.0 | 25.0 | 28.1 | 15.6 | 25.0 |
| stairs/pyramid_stairs_wide/up/L14 | 통과 | 90.6 | 93.8 | 90.6 | 90.6 | 90.6 | 90.6 | 93.8 |
| stairs/pyramid_stairs_wide/up/L14 | 넘어짐 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L14 | 몸통 접촉 | 46.9 | 37.5 | 50.0 | 40.6 | 53.1 | 46.9 | 53.1 |
| stairs/pyramid_stairs_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L9 | 몸통 접촉 | 62.5 | 84.4 | 53.1 | 46.9 | 62.5 | 56.2 | 59.4 |
| stairs/pyramid_stairs_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 | 93.8 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 몸통 접촉 | 50.0 | 18.8 | 34.4 | 43.8 | 40.6 | 43.8 | 50.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 통과 | 96.9 | 93.8 | 96.9 | 96.9 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 넘어짐 | 3.1 | 6.2 | 3.1 | 3.1 | 3.1 | 3.1 | 3.1 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 몸통 접촉 | 46.9 | 65.6 | 59.4 | 50.0 | 71.9 | 65.6 | 59.4 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 몸통 접촉 | 59.4 | 84.4 | 62.5 | 62.5 | 71.9 | 56.2 | 59.4 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 통과 | 96.9 | 96.9 | 96.9 | 100.0 | 93.8 | 93.8 | 93.8 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 넘어짐 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 | 3.1 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 93.8 | 93.8 |
| stairs/pyramid_stairs_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/up/L9 | 몸통 접촉 | 15.6 | 43.8 | 34.4 | 37.5 | 40.6 | 34.4 | 59.4 |
| stairs/pyramid_stairs_nose/up/L14 | 통과 | 75.0 | 84.4 | 71.9 | 71.9 | 81.2 | 78.1 | 75.0 |
| stairs/pyramid_stairs_nose/up/L14 | 넘어짐 | 15.6 | 0.0 | 18.8 | 21.9 | 15.6 | 15.6 | 15.6 |
| stairs/pyramid_stairs_nose/up/L14 | 몸통 접촉 | 75.0 | 75.0 | 87.5 | 75.0 | 65.6 | 75.0 | 84.4 |
| stairs/pyramid_stairs_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L9 | 몸통 접촉 | 71.9 | 90.6 | 65.6 | 53.1 | 75.0 | 62.5 | 75.0 |
| stairs/pyramid_stairs_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 몸통 접촉 | 62.5 | 68.8 | 65.6 | 65.6 | 75.0 | 62.5 | 71.9 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 통과 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 넘어짐 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 몸통 접촉 | 78.1 | 75.0 | 78.1 | 81.2 | 81.2 | 78.1 | 81.2 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 통과 | 93.8 | 96.9 | 93.8 | 96.9 | 93.8 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 몸통 접촉 | 46.9 | 84.4 | 65.6 | 50.0 | 53.1 | 53.1 | 53.1 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.1 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 몸통 접촉 | 71.9 | 96.9 | 68.8 | 62.5 | 93.8 | 65.6 | 81.2 |
| **평균** | |  |  |  |  |  |  |  |
| 평균 | 통과 | 97.3 | 97.7 | 97.0 | 97.3 | 97.3 | 97.1 | 97.3 |
| 평균 | 넘어짐 | 0.8 | 0.7 | 0.9 | 1.0 | 0.8 | 1.0 | 0.8 |
| 평균 | 몸통 접촉 | 57.4 | 63.9 | 61.8 | 59.8 | 65.0 | 60.3 | 64.7 |

## 기준 대비 하락 (⚠)

- **v2u1111**: nominal/gap/clean/시도 72.4 vs 75.0; nominal/terrain/넘어짐 2.3 vs 0.0; lidar_off/stairs/통과 94.1 vs 97.1; lidar_off/stairs/넘어짐 3.0 vs 0.7; lidar_off/terrain/넘어짐 3.1 vs 0.0; disturbed/gap/건넘 95.5 vs 97.9; disturbed/gap/넘어짐 4.5 vs 2.1
- **v21u465**: nominal/gap/clean/시도 72.7 vs 75.0; no_payload/gap/clean/시도 71.7 vs 75.9
- **v21u355**: nominal/gap/clean/시도 71.2 vs 75.0; disturbed/gap/clean/시도 73.5 vs 76.3; no_payload/gap/clean/시도 71.7 vs 75.9
- **v21u536**: nominal/gap/건넘 95.0 vs 97.0; nominal/gap/넘어짐 5.0 vs 3.0; no_payload/gap/clean/시도 71.0 vs 75.9
- **v21u284**: nominal/gap/clean/시도 72.1 vs 75.0; disturbed/gap/clean/시도 72.0 vs 76.3; no_payload/gap/clean/시도 73.9 vs 75.9
- **v21u319**: nominal/gap/clean/시도 71.9 vs 75.0; disturbed/gap/clean/시도 74.2 vs 76.3; no_payload/gap/clean/시도 70.8 vs 75.9
