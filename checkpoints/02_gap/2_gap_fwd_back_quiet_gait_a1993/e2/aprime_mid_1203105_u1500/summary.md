# E2 결과: aprime_mid_1203105_u1500

- 기준 모델: b6566, 기존 능력 지표가 기준보다 1%p 넘게 낮으면 ⚠
- seed 42, 에피소드 20 s, env마다 첫 에피소드 1회

## 조건 nominal (라이다 both, 탑재물 5.0 kg, 밀기 False, 계단 외란 False)

### 갭

| 셀 | 지표 | aprime | r3906 | b6566 |
|---|---|---|---|---|
| gap/forward/L7/v0.3 | 건넘 | 100.0 | 100.0 | 95.8 |
| gap/forward/L7/v0.3 | clean/시도 | 75.0 | 60.4 | 73.9 |
| gap/forward/L7/v0.3 | 넘어짐 | 0.0 | 0.0 | 2.1 |
| gap/forward/L7/v0.3 | 머리 들림 최대 평균 | 4.8° | 11.6° | 12.8° |
| gap/forward/L7/v0.3 | 머리 들림 최대 p90 | 8.8° | 17.4° | 17.3° |
| gap/forward/L7/v0.3 | 20° 초과 | 0.0 | 2.1 | 6.2 |
| gap/forward/L7/v0.3 | 15° 초과 시간 | 0.00 s | 0.09 s | 0.35 s |
| gap/forward/L7/v0.5 | 건넘 | 100.0 | 100.0 | 97.9 |
| gap/forward/L7/v0.5 | clean/시도 | 89.6 | 79.2 | 66.0 |
| gap/forward/L7/v0.5 | 넘어짐 | 0.0 | 0.0 | 2.1 |
| gap/forward/L7/v0.5 | 머리 들림 최대 평균 | 4.6° | 12.8° | 13.2° |
| gap/forward/L7/v0.5 | 머리 들림 최대 p90 | 7.9° | 18.9° | 21.8° |
| gap/forward/L7/v0.5 | 20° 초과 | 0.0 | 0.0 | 12.5 |
| gap/forward/L7/v0.5 | 15° 초과 시간 | 0.00 s | 0.12 s | 0.09 s |
| gap/forward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.9 | clean/시도 | 91.7 | 91.7 | 87.5 |
| gap/forward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 머리 들림 최대 평균 | 3.8° | 12.7° | 11.8° |
| gap/forward/L7/v0.9 | 머리 들림 최대 p90 | 8.3° | 17.8° | 16.6° |
| gap/forward/L7/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 15° 초과 시간 | 0.00 s | 0.06 s | 0.02 s |
| gap/forward/L11/v0.3 | 건넘 | 97.9 | 97.9 | 91.7 |
| gap/forward/L11/v0.3 | clean/시도 | 80.9 | 74.5 | 70.5 |
| gap/forward/L11/v0.3 | 넘어짐 | 2.1 | 2.1 | 8.3 |
| gap/forward/L11/v0.3 | 머리 들림 최대 평균 | 8.4° | 17.6° | 16.6° |
| gap/forward/L11/v0.3 | 머리 들림 최대 p90 | 12.8° | 21.9° | 24.2° |
| gap/forward/L11/v0.3 | 20° 초과 | 0.0 | 39.6 | 25.0 |
| gap/forward/L11/v0.3 | 15° 초과 시간 | 0.00 s | 0.50 s | 0.19 s |
| gap/forward/L11/v0.5 | 건넘 | 100.0 | 100.0 | 85.4 |
| gap/forward/L11/v0.5 | clean/시도 | 83.3 | 72.9 | 78.0 |
| gap/forward/L11/v0.5 | 넘어짐 | 0.0 | 0.0 | 14.6 |
| gap/forward/L11/v0.5 | 머리 들림 최대 평균 | 6.4° | 18.2° | 16.1° |
| gap/forward/L11/v0.5 | 머리 들림 최대 p90 | 12.0° | 23.3° | 22.3° |
| gap/forward/L11/v0.5 | 20° 초과 | 0.0 | 50.0 | 35.4 |
| gap/forward/L11/v0.5 | 15° 초과 시간 | 0.00 s | 0.41 s | 0.18 s |
| gap/forward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 97.9 |
| gap/forward/L11/v0.9 | clean/시도 | 85.4 | 81.2 | 76.6 |
| gap/forward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 2.1 |
| gap/forward/L11/v0.9 | 머리 들림 최대 평균 | 5.4° | 16.5° | 14.5° |
| gap/forward/L11/v0.9 | 머리 들림 최대 p90 | 11.6° | 22.0° | 18.2° |
| gap/forward/L11/v0.9 | 20° 초과 | 0.0 | 33.3 | 4.2 |
| gap/forward/L11/v0.9 | 15° 초과 시간 | 0.00 s | 0.19 s | 0.11 s |
| gap/forward/L14/v0.3 | 건넘 | 97.9 | 72.9 | 77.1 |
| gap/forward/L14/v0.3 | clean/시도 | 76.6 | 42.9 | 83.8 |
| gap/forward/L14/v0.3 | 넘어짐 | 2.1 | 27.1 | 22.9 |
| gap/forward/L14/v0.3 | 머리 들림 최대 평균 | 7.8° | 20.7° | 17.7° |
| gap/forward/L14/v0.3 | 머리 들림 최대 p90 | 13.7° | 36.3° | 23.6° |
| gap/forward/L14/v0.3 | 20° 초과 | 2.1 | 66.7 | 52.1 |
| gap/forward/L14/v0.3 | 15° 초과 시간 | 0.00 s | 0.73 s | 0.34 s |
| gap/forward/L14/v0.5 | 건넘 | 93.8 | 89.6 | 85.4 |
| gap/forward/L14/v0.5 | clean/시도 | 87.0 | 60.5 | 68.3 |
| gap/forward/L14/v0.5 | 넘어짐 | 6.2 | 10.4 | 14.6 |
| gap/forward/L14/v0.5 | 머리 들림 최대 평균 | 5.3° | 18.6° | 16.8° |
| gap/forward/L14/v0.5 | 머리 들림 최대 p90 | 12.3° | 24.2° | 22.4° |
| gap/forward/L14/v0.5 | 20° 초과 | 0.0 | 60.4 | 37.5 |
| gap/forward/L14/v0.5 | 15° 초과 시간 | 0.00 s | 0.43 s | 0.24 s |
| gap/forward/L14/v0.9 | 건넘 | 97.9 | 93.8 | 89.6 |
| gap/forward/L14/v0.9 | clean/시도 | 83.0 | 75.6 | 62.8 |
| gap/forward/L14/v0.9 | 넘어짐 | 2.1 | 6.2 | 10.4 |
| gap/forward/L14/v0.9 | 머리 들림 최대 평균 | 4.9° | 17.3° | 15.4° |
| gap/forward/L14/v0.9 | 머리 들림 최대 p90 | 11.9° | 22.9° | 20.3° |
| gap/forward/L14/v0.9 | 20° 초과 | 0.0 | 52.1 | 12.5 |
| gap/forward/L14/v0.9 | 15° 초과 시간 | 0.00 s | 0.26 s | 0.19 s |
| gap/backward/L7/v0.3 | 건넘 | 100.0 | 18.8 | 91.7 |
| gap/backward/L7/v0.3 | clean/시도 | 87.5 | 0.0 | 40.9 |
| gap/backward/L7/v0.3 | 넘어짐 | 0.0 | 75.0 | 8.3 |
| gap/backward/L7/v0.3 | 머리 들림 최대 평균 | 10.3° | 28.7° | 9.5° |
| gap/backward/L7/v0.3 | 머리 들림 최대 p90 | 12.4° | 39.3° | 11.5° |
| gap/backward/L7/v0.3 | 20° 초과 | 0.0 | 81.2 | 6.2 |
| gap/backward/L7/v0.3 | 15° 초과 시간 | 0.00 s | 1.59 s | 0.01 s |
| gap/backward/L7/v0.5 | 건넘 | 100.0 | 29.2 | 100.0 |
| gap/backward/L7/v0.5 | clean/시도 | 85.4 | 0.0 | 70.8 |
| gap/backward/L7/v0.5 | 넘어짐 | 0.0 | 70.8 | 0.0 |
| gap/backward/L7/v0.5 | 머리 들림 최대 평균 | 9.4° | 24.1° | 8.1° |
| gap/backward/L7/v0.5 | 머리 들림 최대 p90 | 11.3° | 38.5° | 10.1° |
| gap/backward/L7/v0.5 | 20° 초과 | 0.0 | 64.6 | 0.0 |
| gap/backward/L7/v0.5 | 15° 초과 시간 | 0.00 s | 0.49 s | 0.00 s |
| gap/backward/L7/v0.9 | 건넘 | 100.0 | 35.4 | 100.0 |
| gap/backward/L7/v0.9 | clean/시도 | 87.5 | 0.0 | 81.2 |
| gap/backward/L7/v0.9 | 넘어짐 | 0.0 | 47.9 | 0.0 |
| gap/backward/L7/v0.9 | 머리 들림 최대 평균 | 6.6° | 17.7° | 6.9° |
| gap/backward/L7/v0.9 | 머리 들림 최대 p90 | 8.7° | 31.7° | 8.6° |
| gap/backward/L7/v0.9 | 20° 초과 | 0.0 | 47.9 | 0.0 |
| gap/backward/L7/v0.9 | 15° 초과 시간 | 0.00 s | 3.63 s | 0.00 s |
| gap/backward/L11/v0.3 | 건넘 | 100.0 | 0.0 | 93.8 |
| gap/backward/L11/v0.3 | clean/시도 | 79.2 | - | 46.7 |
| gap/backward/L11/v0.3 | 넘어짐 | 0.0 | 97.9 | 6.2 |
| gap/backward/L11/v0.3 | 머리 들림 최대 평균 | 9.9° | 33.5° | 9.2° |
| gap/backward/L11/v0.3 | 머리 들림 최대 p90 | 13.2° | 42.2° | 11.5° |
| gap/backward/L11/v0.3 | 20° 초과 | 0.0 | 100.0 | 2.1 |
| gap/backward/L11/v0.3 | 15° 초과 시간 | 0.00 s | 0.71 s | 0.01 s |
| gap/backward/L11/v0.5 | 건넘 | 100.0 | 0.0 | 100.0 |
| gap/backward/L11/v0.5 | clean/시도 | 93.8 | - | 87.5 |
| gap/backward/L11/v0.5 | 넘어짐 | 0.0 | 97.9 | 0.0 |
| gap/backward/L11/v0.5 | 머리 들림 최대 평균 | 8.7° | 29.7° | 8.6° |
| gap/backward/L11/v0.5 | 머리 들림 최대 p90 | 11.7° | 35.1° | 10.9° |
| gap/backward/L11/v0.5 | 20° 초과 | 0.0 | 97.9 | 0.0 |
| gap/backward/L11/v0.5 | 15° 초과 시간 | 0.00 s | 0.26 s | 0.00 s |
| gap/backward/L11/v0.9 | 건넘 | 100.0 | 4.2 | 100.0 |
| gap/backward/L11/v0.9 | clean/시도 | 85.4 | 0.0 | 75.0 |
| gap/backward/L11/v0.9 | 넘어짐 | 0.0 | 93.8 | 0.0 |
| gap/backward/L11/v0.9 | 머리 들림 최대 평균 | 6.5° | 24.5° | 7.3° |
| gap/backward/L11/v0.9 | 머리 들림 최대 p90 | 9.8° | 33.4° | 9.5° |
| gap/backward/L11/v0.9 | 20° 초과 | 0.0 | 77.1 | 0.0 |
| gap/backward/L11/v0.9 | 15° 초과 시간 | 0.00 s | 0.76 s | 0.00 s |
| gap/backward/L14/v0.3 | 건넘 | 100.0 | 0.0 | 87.5 |
| gap/backward/L14/v0.3 | clean/시도 | 50.0 | 0.0 | 21.4 |
| gap/backward/L14/v0.3 | 넘어짐 | 0.0 | 95.8 | 12.5 |
| gap/backward/L14/v0.3 | 머리 들림 최대 평균 | 7.7° | 31.0° | 10.7° |
| gap/backward/L14/v0.3 | 머리 들림 최대 p90 | 11.7° | 35.9° | 20.2° |
| gap/backward/L14/v0.3 | 20° 초과 | 0.0 | 100.0 | 10.4 |
| gap/backward/L14/v0.3 | 15° 초과 시간 | 0.00 s | 0.82 s | 0.04 s |
| gap/backward/L14/v0.5 | 건넘 | 100.0 | 0.0 | 93.8 |
| gap/backward/L14/v0.5 | clean/시도 | 66.7 | - | 62.2 |
| gap/backward/L14/v0.5 | 넘어짐 | 0.0 | 100.0 | 6.2 |
| gap/backward/L14/v0.5 | 머리 들림 최대 평균 | 6.3° | 30.2° | 7.5° |
| gap/backward/L14/v0.5 | 머리 들림 최대 p90 | 8.9° | 33.7° | 9.7° |
| gap/backward/L14/v0.5 | 20° 초과 | 0.0 | 100.0 | 0.0 |
| gap/backward/L14/v0.5 | 15° 초과 시간 | 0.00 s | 0.28 s | 0.00 s |
| gap/backward/L14/v0.9 | 건넘 | 100.0 | 2.1 | 100.0 |
| gap/backward/L14/v0.9 | clean/시도 | 89.6 | 0.0 | 68.8 |
| gap/backward/L14/v0.9 | 넘어짐 | 0.0 | 97.9 | 0.0 |
| gap/backward/L14/v0.9 | 머리 들림 최대 평균 | 5.8° | 28.2° | 6.7° |
| gap/backward/L14/v0.9 | 머리 들림 최대 p90 | 9.0° | 35.3° | 8.8° |
| gap/backward/L14/v0.9 | 20° 초과 | 0.0 | 89.6 | 0.0 |
| gap/backward/L14/v0.9 | 15° 초과 시간 | 0.00 s | 0.48 s | 0.00 s |
| **평균** | |  |  |  |
| 평균 | 건넘 | 99.3 | 52.4 ⚠ | 93.8 |
| 평균 | clean/시도 | 82.1 | 42.6 ⚠ | 67.9 |
| 평균 | 넘어짐 | 0.7 | 45.7 ⚠ | 6.1 |
| 평균 | 머리 들림 최대 평균 | 6.8° | 21.9° | 11.6° |
| 평균 | 머리 들림 최대 p90 | 10.9° | 29.4° | 16.0° |
| 평균 | 20° 초과 | 0.1 | 59.0 | 11.3 |
| 평균 | 15° 초과 시간 | 0.00 s | 0.66 s | 0.10 s |

### 계단

| 셀 | 지표 | aprime | r3906 | b6566 |
|---|---|---|---|---|
| stairs/pyramid_stairs/up/L9 | 통과 | 84.4 | 90.6 | 96.9 |
| stairs/pyramid_stairs/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L9 | 몸통 접촉 | 21.9 | 9.4 | 21.9 |
| stairs/pyramid_stairs/up/L14 | 통과 | 90.6 | 96.9 | 96.9 |
| stairs/pyramid_stairs/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L14 | 몸통 접촉 | 9.4 | 3.1 | 43.8 |
| stairs/pyramid_stairs/down/L9 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L9 | 몸통 접촉 | 71.9 | 40.6 | 53.1 |
| stairs/pyramid_stairs/down/L14 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L14 | 몸통 접촉 | 87.5 | 81.2 | 87.5 |
| stairs/pyramid_stairs_inv/up/L9 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L9 | 몸통 접촉 | 25.0 | 3.1 | 31.2 |
| stairs/pyramid_stairs_inv/up/L14 | 통과 | 100.0 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv/up/L14 | 넘어짐 | 0.0 | 3.1 | 0.0 |
| stairs/pyramid_stairs_inv/up/L14 | 몸통 접촉 | 34.4 | 34.4 | 75.0 |
| stairs/pyramid_stairs_inv/down/L9 | 통과 | 96.9 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L9 | 몸통 접촉 | 40.6 | 25.0 | 34.4 |
| stairs/pyramid_stairs_inv/down/L14 | 통과 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L14 | 몸통 접촉 | 71.9 | 59.4 | 90.6 |
| stairs/pyramid_stairs_wide/up/L9 | 통과 | 90.6 | 93.8 | 93.8 |
| stairs/pyramid_stairs_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L9 | 몸통 접촉 | 15.6 | 9.4 | 18.8 |
| stairs/pyramid_stairs_wide/up/L14 | 통과 | 84.4 | 96.9 | 90.6 |
| stairs/pyramid_stairs_wide/up/L14 | 넘어짐 | 3.1 | 3.1 | 3.1 |
| stairs/pyramid_stairs_wide/up/L14 | 몸통 접촉 | 31.2 | 28.1 | 40.6 |
| stairs/pyramid_stairs_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L9 | 몸통 접촉 | 75.0 | 65.6 | 56.2 |
| stairs/pyramid_stairs_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L14 | 몸통 접촉 | 93.8 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 몸통 접촉 | 21.9 | 25.0 | 31.2 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 몸통 접촉 | 43.8 | 40.6 | 53.1 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 통과 | 96.9 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 몸통 접촉 | 78.1 | 53.1 | 62.5 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 몸통 접촉 | 100.0 | 93.8 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/up/L9 | 몸통 접촉 | 18.8 | 25.0 | 31.2 |
| stairs/pyramid_stairs_nose/up/L14 | 통과 | 84.4 | 53.1 | 68.8 |
| stairs/pyramid_stairs_nose/up/L14 | 넘어짐 | 9.4 | 46.9 | 28.1 |
| stairs/pyramid_stairs_nose/up/L14 | 몸통 접촉 | 18.8 | 78.1 | 75.0 |
| stairs/pyramid_stairs_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L9 | 몸통 접촉 | 71.9 | 50.0 | 62.5 |
| stairs/pyramid_stairs_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 몸통 접촉 | 50.0 | 71.9 | 50.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 통과 | 100.0 | 93.8 | 96.9 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 넘어짐 | 0.0 | 6.2 | 3.1 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 몸통 접촉 | 68.8 | 84.4 | 87.5 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 몸통 접촉 | 71.9 | 18.8 | 59.4 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 몸통 접촉 | 75.0 | 65.6 | 78.1 |
| **평균** | |  |  |  |
| 평균 | 통과 | 96.9 | 96.6 | 97.5 |
| 평균 | 넘어짐 | 0.5 | 2.5 ⚠ | 1.4 |
| 평균 | 몸통 접촉 | 54.0 | 48.6 | 60.2 |

### 경사·거친 지면

| 셀 | 지표 | aprime | r3906 | b6566 |
|---|---|---|---|---|
| terrain/hf_pyramid_slope/L14 | 통과 | 100.0 | 100.0 | 100.0 |
| terrain/hf_pyramid_slope/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| terrain/hf_pyramid_slope/L14 | 속도 오차 m/s | 0.243 | 0.219 | 0.195 |
| terrain/hf_pyramid_slope_inv/L14 | 통과 | 96.9 | 90.6 | 93.8 |
| terrain/hf_pyramid_slope_inv/L14 | 넘어짐 | 0.0 | 0.0 | 3.1 |
| terrain/hf_pyramid_slope_inv/L14 | 속도 오차 m/s | 0.230 | 0.226 | 0.208 |
| terrain/random_rough/L14 | 통과 | 100.0 | 100.0 | 100.0 |
| terrain/random_rough/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| terrain/random_rough/L14 | 속도 오차 m/s | 0.253 | 0.177 | 0.162 |
| terrain/ring_fence/L14 | 통과 | 100.0 | 100.0 | 100.0 |
| terrain/ring_fence/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 |
| terrain/ring_fence/L14 | 속도 오차 m/s | 0.239 | 0.180 | 0.171 |
| **평균** | |  |  |  |
| 평균 | 통과 | 99.2 | 97.7 | 98.4 |
| 평균 | 넘어짐 | 0.0 | 0.0 | 0.8 |
| 평균 | 속도 오차 m/s | 0.241 | 0.201 | 0.184 |

## 기준 대비 하락 (⚠)

- **r3906**: nominal/gap/건넘 52.4 vs 93.8; nominal/gap/clean/시도 42.6 vs 67.9; nominal/gap/넘어짐 45.7 vs 6.1; nominal/stairs/넘어짐 2.5 vs 1.4
