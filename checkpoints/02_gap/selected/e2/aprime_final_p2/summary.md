# E2 결과: aprime_final_p2

- 기준 모델: b6566, 기존 능력 지표가 기준보다 1%p 넘게 낮으면 ⚠
- seed 42, 에피소드 20 s, env마다 첫 에피소드 1회

## 조건 nominal (라이다 both, 탑재물 5.0 kg, 밀기 False, 계단 외란 False)

### 갭

| 셀 | 지표 | b6566 | a1396 | a1283 | a1973 | a1929 |
|---|---|---|---|---|---|---|
| gap/forward/L7/v0.3 | 건넘 | 95.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.3 | clean/시도 | 73.9 | 70.8 | 77.1 | 81.2 | 79.2 |
| gap/forward/L7/v0.3 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 머리 들림 최대 평균 | 12.8° | 4.1° | 4.1° | 3.6° | 3.7° |
| gap/forward/L7/v0.3 | 머리 들림 최대 p90 | 17.3° | 7.7° | 8.0° | 7.9° | 8.7° |
| gap/forward/L7/v0.3 | 20° 초과 | 6.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 15° 초과 시간 | 0.35 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L7/v0.5 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.5 | clean/시도 | 66.0 | 93.8 | 85.4 | 87.5 | 93.8 |
| gap/forward/L7/v0.5 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 머리 들림 최대 평균 | 13.2° | 4.4° | 4.5° | 3.9° | 4.0° |
| gap/forward/L7/v0.5 | 머리 들림 최대 p90 | 21.8° | 7.2° | 7.3° | 7.4° | 7.5° |
| gap/forward/L7/v0.5 | 20° 초과 | 12.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 15° 초과 시간 | 0.09 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.9 | clean/시도 | 87.5 | 89.6 | 91.7 | 95.8 | 91.7 |
| gap/forward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 머리 들림 최대 평균 | 11.8° | 4.3° | 4.0° | 3.8° | 3.9° |
| gap/forward/L7/v0.9 | 머리 들림 최대 p90 | 16.6° | 7.9° | 7.8° | 8.5° | 8.7° |
| gap/forward/L7/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 15° 초과 시간 | 0.02 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.3 | 건넘 | 91.7 | 97.9 | 100.0 | 100.0 | 95.8 |
| gap/forward/L11/v0.3 | clean/시도 | 70.5 | 74.5 | 75.0 | 72.9 | 76.1 |
| gap/forward/L11/v0.3 | 넘어짐 | 8.3 | 2.1 | 0.0 | 0.0 | 4.2 |
| gap/forward/L11/v0.3 | 머리 들림 최대 평균 | 16.6° | 9.0° | 9.2° | 6.6° | 8.2° |
| gap/forward/L11/v0.3 | 머리 들림 최대 p90 | 24.2° | 13.8° | 13.9° | 11.5° | 12.2° |
| gap/forward/L11/v0.3 | 20° 초과 | 25.0 | 0.0 | 0.0 | 0.0 | 2.1 |
| gap/forward/L11/v0.3 | 15° 초과 시간 | 0.19 s | 0.00 s | 0.00 s | 0.00 s | 0.01 s |
| gap/forward/L11/v0.5 | 건넘 | 85.4 | 97.9 | 100.0 | 100.0 | 100.0 |
| gap/forward/L11/v0.5 | clean/시도 | 78.0 | 85.1 | 83.3 | 72.9 | 70.8 |
| gap/forward/L11/v0.5 | 넘어짐 | 14.6 | 2.1 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 머리 들림 최대 평균 | 16.1° | 6.6° | 6.7° | 5.5° | 5.5° |
| gap/forward/L11/v0.5 | 머리 들림 최대 p90 | 22.3° | 12.2° | 12.2° | 11.1° | 11.1° |
| gap/forward/L11/v0.5 | 20° 초과 | 35.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 15° 초과 시간 | 0.18 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.9 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L11/v0.9 | clean/시도 | 76.6 | 87.5 | 83.3 | 91.7 | 87.5 |
| gap/forward/L11/v0.9 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 머리 들림 최대 평균 | 14.5° | 6.9° | 7.4° | 5.9° | 5.8° |
| gap/forward/L11/v0.9 | 머리 들림 최대 p90 | 18.2° | 11.6° | 11.2° | 11.9° | 11.8° |
| gap/forward/L11/v0.9 | 20° 초과 | 4.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 15° 초과 시간 | 0.11 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L14/v0.3 | 건넘 | 77.1 | 93.8 | 97.9 | 100.0 | 97.9 |
| gap/forward/L14/v0.3 | clean/시도 | 83.8 | 71.1 | 79.2 | 68.8 | 85.1 |
| gap/forward/L14/v0.3 | 넘어짐 | 22.9 | 6.2 | 2.1 | 0.0 | 2.1 |
| gap/forward/L14/v0.3 | 머리 들림 최대 평균 | 17.7° | 8.6° | 6.6° | 6.5° | 6.3° |
| gap/forward/L14/v0.3 | 머리 들림 최대 p90 | 23.6° | 16.0° | 11.0° | 12.9° | 10.8° |
| gap/forward/L14/v0.3 | 20° 초과 | 52.1 | 4.2 | 0.0 | 0.0 | 2.1 |
| gap/forward/L14/v0.3 | 15° 초과 시간 | 0.34 s | 0.03 s | 0.00 s | 0.01 s | 0.01 s |
| gap/forward/L14/v0.5 | 건넘 | 85.4 | 95.8 | 93.8 | 97.9 | 93.8 |
| gap/forward/L14/v0.5 | clean/시도 | 68.3 | 76.6 | 72.3 | 80.9 | 88.9 |
| gap/forward/L14/v0.5 | 넘어짐 | 14.6 | 4.2 | 6.2 | 2.1 | 6.2 |
| gap/forward/L14/v0.5 | 머리 들림 최대 평균 | 16.8° | 5.6° | 6.2° | 5.2° | 5.0° |
| gap/forward/L14/v0.5 | 머리 들림 최대 p90 | 22.4° | 11.8° | 12.4° | 12.0° | 11.7° |
| gap/forward/L14/v0.5 | 20° 초과 | 37.5 | 0.0 | 0.0 | 0.0 | 2.1 |
| gap/forward/L14/v0.5 | 15° 초과 시간 | 0.24 s | 0.01 s | 0.01 s | 0.01 s | 0.01 s |
| gap/forward/L14/v0.9 | 건넘 | 89.6 | 95.8 | 95.8 | 97.9 | 91.7 |
| gap/forward/L14/v0.9 | clean/시도 | 62.8 | 80.9 | 79.2 | 80.9 | 77.3 |
| gap/forward/L14/v0.9 | 넘어짐 | 10.4 | 4.2 | 4.2 | 2.1 | 8.3 |
| gap/forward/L14/v0.9 | 머리 들림 최대 평균 | 15.4° | 7.5° | 6.4° | 4.3° | 4.2° |
| gap/forward/L14/v0.9 | 머리 들림 최대 p90 | 20.3° | 15.2° | 9.6° | 12.1° | 12.6° |
| gap/forward/L14/v0.9 | 20° 초과 | 12.5 | 2.1 | 2.1 | 0.0 | 0.0 |
| gap/forward/L14/v0.9 | 15° 초과 시간 | 0.19 s | 0.01 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.3 | 건넘 | 91.7 | 100.0 | 100.0 | 100.0 | 97.9 |
| gap/backward/L7/v0.3 | clean/시도 | 40.9 | 77.1 | 89.6 | 75.0 | 77.1 |
| gap/backward/L7/v0.3 | 넘어짐 | 8.3 | 0.0 | 0.0 | 0.0 | 2.1 |
| gap/backward/L7/v0.3 | 머리 들림 최대 평균 | 9.5° | 10.8° | 10.8° | 9.2° | 8.9° |
| gap/backward/L7/v0.3 | 머리 들림 최대 p90 | 11.5° | 13.7° | 13.8° | 11.5° | 10.6° |
| gap/backward/L7/v0.3 | 20° 초과 | 6.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.3 | 15° 초과 시간 | 0.01 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.5 | clean/시도 | 70.8 | 79.2 | 77.1 | 83.3 | 75.0 |
| gap/backward/L7/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.5 | 머리 들림 최대 평균 | 8.1° | 9.7° | 9.6° | 7.9° | 7.5° |
| gap/backward/L7/v0.5 | 머리 들림 최대 p90 | 10.1° | 12.1° | 11.7° | 9.4° | 9.2° |
| gap/backward/L7/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.9 | clean/시도 | 81.2 | 89.6 | 89.6 | 93.8 | 91.7 |
| gap/backward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 머리 들림 최대 평균 | 6.9° | 7.8° | 7.4° | 5.5° | 5.1° |
| gap/backward/L7/v0.9 | 머리 들림 최대 p90 | 8.6° | 9.0° | 10.2° | 6.8° | 6.7° |
| gap/backward/L7/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.3 | 건넘 | 93.8 | 100.0 | 97.9 | 100.0 | 95.8 |
| gap/backward/L11/v0.3 | clean/시도 | 46.7 | 72.9 | 76.6 | 62.5 | 67.4 |
| gap/backward/L11/v0.3 | 넘어짐 | 6.2 | 0.0 | 2.1 | 0.0 | 4.2 |
| gap/backward/L11/v0.3 | 머리 들림 최대 평균 | 9.2° | 11.1° | 10.7° | 9.7° | 9.3° |
| gap/backward/L11/v0.3 | 머리 들림 최대 p90 | 11.5° | 15.0° | 14.3° | 12.7° | 12.1° |
| gap/backward/L11/v0.3 | 20° 초과 | 2.1 | 0.0 | 0.0 | 0.0 | 2.1 |
| gap/backward/L11/v0.3 | 15° 초과 시간 | 0.01 s | 0.01 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.5 | clean/시도 | 87.5 | 81.2 | 85.4 | 85.4 | 87.5 |
| gap/backward/L11/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 머리 들림 최대 평균 | 8.6° | 10.0° | 9.9° | 8.4° | 7.5° |
| gap/backward/L11/v0.5 | 머리 들림 최대 p90 | 10.9° | 13.3° | 13.1° | 11.1° | 9.9° |
| gap/backward/L11/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.9 | clean/시도 | 75.0 | 81.2 | 83.3 | 85.4 | 79.2 |
| gap/backward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 머리 들림 최대 평균 | 7.3° | 7.6° | 7.5° | 5.9° | 5.9° |
| gap/backward/L11/v0.9 | 머리 들림 최대 p90 | 9.5° | 11.3° | 11.1° | 8.7° | 8.6° |
| gap/backward/L11/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.3 | 건넘 | 87.5 | 97.9 | 100.0 | 93.8 | 97.9 |
| gap/backward/L14/v0.3 | clean/시도 | 21.4 | 75.0 | 75.0 | 59.6 | 63.8 |
| gap/backward/L14/v0.3 | 넘어짐 | 12.5 | 2.1 | 0.0 | 6.2 | 2.1 |
| gap/backward/L14/v0.3 | 머리 들림 최대 평균 | 10.7° | 8.6° | 8.5° | 7.8° | 7.7° |
| gap/backward/L14/v0.3 | 머리 들림 최대 p90 | 20.2° | 13.3° | 13.5° | 11.9° | 11.5° |
| gap/backward/L14/v0.3 | 20° 초과 | 10.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.3 | 15° 초과 시간 | 0.04 s | 0.01 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.5 | 건넘 | 93.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L14/v0.5 | clean/시도 | 62.2 | 75.0 | 75.0 | 83.3 | 75.0 |
| gap/backward/L14/v0.5 | 넘어짐 | 6.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 머리 들림 최대 평균 | 7.5° | 7.8° | 7.1° | 6.5° | 6.2° |
| gap/backward/L14/v0.5 | 머리 들림 최대 p90 | 9.7° | 10.7° | 9.0° | 10.0° | 9.2° |
| gap/backward/L14/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L14/v0.9 | clean/시도 | 68.8 | 79.2 | 68.8 | 81.2 | 87.5 |
| gap/backward/L14/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.9 | 머리 들림 최대 평균 | 6.7° | 6.7° | 6.5° | 5.8° | 5.5° |
| gap/backward/L14/v0.9 | 머리 들림 최대 p90 | 8.8° | 10.6° | 10.1° | 8.8° | 9.4° |
| gap/backward/L14/v0.9 | 20° 초과 | 0.0 | 2.1 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| **평균** | |  |  |  |  |  |
| 평균 | 건넘 | 93.8 | 98.8 | 99.2 | 99.4 | 98.4 |
| 평균 | clean/시도 | 67.9 | 80.0 | 80.4 | 80.1 | 80.8 |
| 평균 | 넘어짐 | 6.1 | 1.2 | 0.8 | 0.6 | 1.6 |
| 평균 | 머리 들림 최대 평균 | 11.6° | 7.6° | 7.4° | 6.2° | 6.1° |
| 평균 | 머리 들림 최대 p90 | 16.0° | 11.8° | 11.1° | 10.3° | 10.1° |
| 평균 | 20° 초과 | 11.3 | 0.5 | 0.1 | 0.0 | 0.5 |
| 평균 | 15° 초과 시간 | 0.10 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |

### 계단

| 셀 | 지표 | b6566 | a1396 | a1283 | a1973 | a1929 |
|---|---|---|---|---|---|---|
| stairs/pyramid_stairs/up/L9 | 통과 | 96.9 | 90.6 | 93.8 | 87.5 | 84.4 |
| stairs/pyramid_stairs/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L9 | 몸통 접촉 | 21.9 | 9.4 | 6.2 | 12.5 | 15.6 |
| stairs/pyramid_stairs/up/L14 | 통과 | 96.9 | 93.8 | 93.8 | 100.0 | 96.9 |
| stairs/pyramid_stairs/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L14 | 몸통 접촉 | 43.8 | 6.2 | 9.4 | 3.1 | 3.1 |
| stairs/pyramid_stairs/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L9 | 몸통 접촉 | 53.1 | 59.4 | 56.2 | 46.9 | 56.2 |
| stairs/pyramid_stairs/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L14 | 몸통 접촉 | 87.5 | 81.2 | 81.2 | 84.4 | 87.5 |
| stairs/pyramid_stairs_inv/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L9 | 몸통 접촉 | 31.2 | 18.8 | 9.4 | 12.5 | 6.2 |
| stairs/pyramid_stairs_inv/up/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L14 | 몸통 접촉 | 75.0 | 25.0 | 31.2 | 28.1 | 25.0 |
| stairs/pyramid_stairs_inv/down/L9 | 통과 | 100.0 | 96.9 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L9 | 몸통 접촉 | 34.4 | 37.5 | 37.5 | 37.5 | 50.0 |
| stairs/pyramid_stairs_inv/down/L14 | 통과 | 96.9 | 100.0 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L14 | 몸통 접촉 | 90.6 | 75.0 | 78.1 | 75.0 | 65.6 |
| stairs/pyramid_stairs_wide/up/L9 | 통과 | 93.8 | 90.6 | 93.8 | 90.6 | 93.8 |
| stairs/pyramid_stairs_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L9 | 몸통 접촉 | 18.8 | 12.5 | 12.5 | 12.5 | 6.2 |
| stairs/pyramid_stairs_wide/up/L14 | 통과 | 90.6 | 93.8 | 90.6 | 87.5 | 93.8 |
| stairs/pyramid_stairs_wide/up/L14 | 넘어짐 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L14 | 몸통 접촉 | 40.6 | 21.9 | 28.1 | 12.5 | 9.4 |
| stairs/pyramid_stairs_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L9 | 몸통 접촉 | 56.2 | 68.8 | 81.2 | 56.2 | 62.5 |
| stairs/pyramid_stairs_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L14 | 몸통 접촉 | 100.0 | 96.9 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 몸통 접촉 | 31.2 | 15.6 | 25.0 | 28.1 | 18.8 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 통과 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 넘어짐 | 0.0 | 0.0 | 3.1 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 몸통 접촉 | 53.1 | 34.4 | 40.6 | 40.6 | 31.2 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 통과 | 100.0 | 96.9 | 100.0 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 몸통 접촉 | 62.5 | 78.1 | 71.9 | 71.9 | 71.9 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/up/L9 | 몸통 접촉 | 31.2 | 21.9 | 43.8 | 28.1 | 21.9 |
| stairs/pyramid_stairs_nose/up/L14 | 통과 | 68.8 | 90.6 | 81.2 | 78.1 | 81.2 |
| stairs/pyramid_stairs_nose/up/L14 | 넘어짐 | 28.1 | 3.1 | 15.6 | 9.4 | 9.4 |
| stairs/pyramid_stairs_nose/up/L14 | 몸통 접촉 | 75.0 | 50.0 | 50.0 | 50.0 | 34.4 |
| stairs/pyramid_stairs_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L9 | 몸통 접촉 | 62.5 | 71.9 | 65.6 | 68.8 | 62.5 |
| stairs/pyramid_stairs_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 몸통 접촉 | 50.0 | 46.9 | 46.9 | 53.1 | 40.6 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 통과 | 96.9 | 100.0 | 96.9 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 넘어짐 | 3.1 | 0.0 | 3.1 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 몸통 접촉 | 87.5 | 59.4 | 62.5 | 62.5 | 62.5 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 몸통 접촉 | 59.4 | 65.6 | 84.4 | 68.8 | 68.8 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 통과 | 100.0 | 100.0 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 몸통 접촉 | 78.1 | 78.1 | 71.9 | 68.8 | 65.6 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 97.5 | 98.0 | 97.5 | 97.3 | 97.5 |
| 평균 | 넘어짐 | 1.4 | 0.1 | 0.9 | 0.4 | 0.4 |
| 평균 | 몸통 접촉 | 60.2 | 51.4 | 53.9 | 50.8 | 48.4 |

### 경사·거친 지면

| 셀 | 지표 | b6566 | a1396 | a1283 | a1973 | a1929 |
|---|---|---|---|---|---|---|
| terrain/hf_pyramid_slope/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/hf_pyramid_slope/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/hf_pyramid_slope/L14 | 속도 오차 m/s | 0.195 | 0.250 | 0.257 | 0.236 | 0.234 |
| terrain/hf_pyramid_slope_inv/L14 | 통과 | 93.8 | 96.9 | 96.9 | 96.9 | 96.9 |
| terrain/hf_pyramid_slope_inv/L14 | 넘어짐 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/hf_pyramid_slope_inv/L14 | 속도 오차 m/s | 0.208 | 0.236 | 0.235 | 0.241 | 0.223 |
| terrain/random_rough/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/random_rough/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/random_rough/L14 | 속도 오차 m/s | 0.162 | 0.257 | 0.263 | 0.236 | 0.246 |
| terrain/ring_fence/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/ring_fence/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/ring_fence/L14 | 속도 오차 m/s | 0.171 | 0.233 | 0.248 | 0.222 | 0.222 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 98.4 | 99.2 | 99.2 | 99.2 | 99.2 |
| 평균 | 넘어짐 | 0.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| 평균 | 속도 오차 m/s | 0.184 | 0.244 | 0.251 | 0.234 | 0.231 |

## 조건 no_payload (라이다 both, 탑재물 0.0 kg, 밀기 False, 계단 외란 False)

### 갭

| 셀 | 지표 | b6566 | a1396 | a1283 | a1973 | a1929 |
|---|---|---|---|---|---|---|
| gap/forward/L7/v0.3 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.3 | clean/시도 | 74.5 | 68.8 | 79.2 | 68.8 | 75.0 |
| gap/forward/L7/v0.3 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 머리 들림 최대 평균 | 14.0° | 4.5° | 4.3° | 3.6° | 3.3° |
| gap/forward/L7/v0.3 | 머리 들림 최대 p90 | 21.1° | 8.8° | 8.7° | 7.0° | 7.5° |
| gap/forward/L7/v0.3 | 20° 초과 | 12.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 15° 초과 시간 | 0.37 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L7/v0.5 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.5 | clean/시도 | 80.9 | 89.6 | 91.7 | 89.6 | 93.8 |
| gap/forward/L7/v0.5 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 머리 들림 최대 평균 | 13.8° | 4.2° | 4.1° | 3.8° | 3.8° |
| gap/forward/L7/v0.5 | 머리 들림 최대 p90 | 18.9° | 7.7° | 7.0° | 8.0° | 7.6° |
| gap/forward/L7/v0.5 | 20° 초과 | 8.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 15° 초과 시간 | 0.17 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.9 | clean/시도 | 89.6 | 89.6 | 85.4 | 91.7 | 95.8 |
| gap/forward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 머리 들림 최대 평균 | 11.5° | 3.8° | 4.0° | 3.6° | 3.8° |
| gap/forward/L7/v0.9 | 머리 들림 최대 p90 | 16.3° | 7.3° | 7.9° | 8.5° | 8.0° |
| gap/forward/L7/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 15° 초과 시간 | 0.02 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.3 | 건넘 | 85.4 | 100.0 | 100.0 | 100.0 | 97.9 |
| gap/forward/L11/v0.3 | clean/시도 | 68.3 | 83.3 | 83.3 | 79.2 | 83.0 |
| gap/forward/L11/v0.3 | 넘어짐 | 14.6 | 0.0 | 0.0 | 0.0 | 2.1 |
| gap/forward/L11/v0.3 | 머리 들림 최대 평균 | 16.1° | 9.4° | 8.7° | 7.3° | 6.6° |
| gap/forward/L11/v0.3 | 머리 들림 최대 p90 | 22.8° | 13.7° | 12.9° | 11.7° | 11.1° |
| gap/forward/L11/v0.3 | 20° 초과 | 33.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.3 | 15° 초과 시간 | 0.17 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.5 | 건넘 | 95.8 | 100.0 | 100.0 | 97.9 | 100.0 |
| gap/forward/L11/v0.5 | clean/시도 | 78.3 | 83.3 | 75.0 | 80.9 | 72.9 |
| gap/forward/L11/v0.5 | 넘어짐 | 4.2 | 0.0 | 0.0 | 2.1 | 0.0 |
| gap/forward/L11/v0.5 | 머리 들림 최대 평균 | 16.7° | 7.2° | 6.3° | 5.4° | 5.9° |
| gap/forward/L11/v0.5 | 머리 들림 최대 p90 | 21.7° | 12.4° | 11.2° | 10.8° | 10.2° |
| gap/forward/L11/v0.5 | 20° 초과 | 18.8 | 2.1 | 2.1 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 15° 초과 시간 | 0.16 s | 0.00 s | 0.01 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L11/v0.9 | clean/시도 | 77.1 | 85.4 | 79.2 | 89.6 | 89.6 |
| gap/forward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 머리 들림 최대 평균 | 14.9° | 6.6° | 7.2° | 4.9° | 4.8° |
| gap/forward/L11/v0.9 | 머리 들림 최대 p90 | 18.8° | 11.4° | 11.6° | 11.3° | 9.6° |
| gap/forward/L11/v0.9 | 20° 초과 | 0.0 | 0.0 | 2.1 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 15° 초과 시간 | 0.10 s | 0.00 s | 0.02 s | 0.00 s | 0.00 s |
| gap/forward/L14/v0.3 | 건넘 | 75.0 | 97.9 | 95.8 | 95.8 | 89.6 |
| gap/forward/L14/v0.3 | clean/시도 | 77.8 | 83.3 | 70.2 | 80.4 | 79.1 |
| gap/forward/L14/v0.3 | 넘어짐 | 25.0 | 2.1 | 4.2 | 4.2 | 10.4 |
| gap/forward/L14/v0.3 | 머리 들림 최대 평균 | 17.4° | 7.6° | 7.2° | 5.6° | 6.9° |
| gap/forward/L14/v0.3 | 머리 들림 최대 p90 | 23.5° | 12.4° | 14.3° | 11.0° | 13.8° |
| gap/forward/L14/v0.3 | 20° 초과 | 35.4 | 2.1 | 2.1 | 2.1 | 4.2 |
| gap/forward/L14/v0.3 | 15° 초과 시간 | 0.31 s | 0.02 s | 0.02 s | 0.03 s | 0.02 s |
| gap/forward/L14/v0.5 | 건넘 | 81.2 | 89.6 | 91.7 | 95.8 | 91.7 |
| gap/forward/L14/v0.5 | clean/시도 | 71.8 | 82.2 | 89.1 | 82.6 | 81.8 |
| gap/forward/L14/v0.5 | 넘어짐 | 18.8 | 10.4 | 8.3 | 4.2 | 8.3 |
| gap/forward/L14/v0.5 | 머리 들림 최대 평균 | 17.1° | 5.6° | 5.4° | 4.1° | 4.3° |
| gap/forward/L14/v0.5 | 머리 들림 최대 p90 | 23.1° | 11.6° | 11.5° | 11.0° | 9.4° |
| gap/forward/L14/v0.5 | 20° 초과 | 35.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.5 | 15° 초과 시간 | 0.23 s | 0.00 s | 0.01 s | 0.00 s | 0.00 s |
| gap/forward/L14/v0.9 | 건넘 | 93.8 | 95.8 | 91.7 | 97.9 | 100.0 |
| gap/forward/L14/v0.9 | clean/시도 | 71.1 | 77.1 | 76.6 | 87.2 | 83.3 |
| gap/forward/L14/v0.9 | 넘어짐 | 6.2 | 4.2 | 8.3 | 2.1 | 0.0 |
| gap/forward/L14/v0.9 | 머리 들림 최대 평균 | 16.0° | 7.1° | 7.3° | 3.8° | 3.9° |
| gap/forward/L14/v0.9 | 머리 들림 최대 p90 | 19.5° | 14.4° | 12.0° | 10.9° | 8.5° |
| gap/forward/L14/v0.9 | 20° 초과 | 8.3 | 2.1 | 6.2 | 0.0 | 0.0 |
| gap/forward/L14/v0.9 | 15° 초과 시간 | 0.20 s | 0.02 s | 0.03 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.3 | 건넘 | 95.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.3 | clean/시도 | 39.1 | 83.3 | 85.4 | 79.2 | 79.2 |
| gap/backward/L7/v0.3 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.3 | 머리 들림 최대 평균 | 9.0° | 11.5° | 10.9° | 9.3° | 9.1° |
| gap/backward/L7/v0.3 | 머리 들림 최대 p90 | 10.4° | 13.9° | 13.6° | 11.0° | 10.8° |
| gap/backward/L7/v0.3 | 20° 초과 | 4.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.3 | 15° 초과 시간 | 0.34 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.5 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.5 | clean/시도 | 61.7 | 81.2 | 85.4 | 85.4 | 83.3 |
| gap/backward/L7/v0.5 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.5 | 머리 들림 최대 평균 | 8.5° | 10.0° | 9.8° | 8.2° | 7.5° |
| gap/backward/L7/v0.5 | 머리 들림 최대 p90 | 10.0° | 12.2° | 11.8° | 10.3° | 8.9° |
| gap/backward/L7/v0.5 | 20° 초과 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.9 | clean/시도 | 87.5 | 93.8 | 89.6 | 89.6 | 91.7 |
| gap/backward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 머리 들림 최대 평균 | 6.8° | 7.7° | 7.6° | 5.8° | 5.2° |
| gap/backward/L7/v0.9 | 머리 들림 최대 p90 | 8.4° | 9.5° | 9.6° | 8.1° | 6.3° |
| gap/backward/L7/v0.9 | 20° 초과 | 0.0 | 0.0 | 2.1 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.01 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.3 | 건넘 | 91.7 | 97.9 | 97.9 | 100.0 | 100.0 |
| gap/backward/L11/v0.3 | clean/시도 | 31.8 | 74.5 | 74.5 | 68.8 | 81.2 |
| gap/backward/L11/v0.3 | 넘어짐 | 8.3 | 2.1 | 2.1 | 0.0 | 0.0 |
| gap/backward/L11/v0.3 | 머리 들림 최대 평균 | 10.3° | 10.9° | 10.7° | 9.2° | 9.3° |
| gap/backward/L11/v0.3 | 머리 들림 최대 p90 | 14.6° | 14.3° | 13.8° | 12.2° | 11.6° |
| gap/backward/L11/v0.3 | 20° 초과 | 8.3 | 0.0 | 2.1 | 0.0 | 0.0 |
| gap/backward/L11/v0.3 | 15° 초과 시간 | 0.04 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.5 | clean/시도 | 62.5 | 85.4 | 91.7 | 81.2 | 81.2 |
| gap/backward/L11/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 머리 들림 최대 평균 | 8.4° | 10.1° | 9.8° | 8.0° | 7.8° |
| gap/backward/L11/v0.5 | 머리 들림 최대 p90 | 10.3° | 13.1° | 12.9° | 11.3° | 10.5° |
| gap/backward/L11/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.9 | clean/시도 | 81.2 | 81.2 | 79.2 | 83.3 | 81.2 |
| gap/backward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 머리 들림 최대 평균 | 6.9° | 7.4° | 7.4° | 5.7° | 5.6° |
| gap/backward/L11/v0.9 | 머리 들림 최대 p90 | 9.3° | 11.5° | 10.6° | 8.5° | 8.2° |
| gap/backward/L11/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.3 | 건넘 | 81.2 | 95.8 | 95.8 | 97.9 | 95.8 |
| gap/backward/L14/v0.3 | clean/시도 | 17.9 | 67.4 | 71.7 | 59.6 | 65.2 |
| gap/backward/L14/v0.3 | 넘어짐 | 18.8 | 4.2 | 4.2 | 2.1 | 4.2 |
| gap/backward/L14/v0.3 | 머리 들림 최대 평균 | 12.0° | 8.9° | 8.0° | 8.1° | 7.8° |
| gap/backward/L14/v0.3 | 머리 들림 최대 p90 | 25.8° | 13.7° | 13.6° | 11.9° | 10.9° |
| gap/backward/L14/v0.3 | 20° 초과 | 20.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.3 | 15° 초과 시간 | 0.06 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.5 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L14/v0.5 | clean/시도 | 57.4 | 75.0 | 72.9 | 70.8 | 75.0 |
| gap/backward/L14/v0.5 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 머리 들림 최대 평균 | 8.5° | 7.7° | 7.1° | 6.1° | 6.2° |
| gap/backward/L14/v0.5 | 머리 들림 최대 p90 | 11.1° | 10.7° | 10.5° | 9.4° | 8.7° |
| gap/backward/L14/v0.5 | 20° 초과 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L14/v0.9 | clean/시도 | 72.9 | 75.0 | 79.2 | 83.3 | 87.5 |
| gap/backward/L14/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.9 | 머리 들림 최대 평균 | 6.7° | 7.0° | 6.6° | 5.7° | 5.6° |
| gap/backward/L14/v0.9 | 머리 들림 최대 p90 | 8.9° | 11.3° | 9.9° | 9.3° | 8.7° |
| gap/backward/L14/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| **평균** | |  |  |  |  |  |
| 평균 | 건넘 | 94.0 | 98.7 | 98.5 | 99.2 | 98.6 |
| 평균 | clean/시도 | 66.7 | 81.1 | 81.1 | 80.6 | 82.2 |
| 평균 | 넘어짐 | 5.8 | 1.3 | 1.5 | 0.8 | 1.4 |
| 평균 | 머리 들림 최대 평균 | 11.9° | 7.6° | 7.4° | 6.0° | 5.9° |
| 평균 | 머리 들림 최대 p90 | 16.4° | 11.7° | 11.3° | 10.1° | 9.4° |
| 평균 | 20° 초과 | 10.5 | 0.3 | 0.9 | 0.1 | 0.2 |
| 평균 | 15° 초과 시간 | 0.12 s | 0.00 s | 0.01 s | 0.00 s | 0.00 s |

### 계단

| 셀 | 지표 | b6566 | a1396 | a1283 | a1973 | a1929 |
|---|---|---|---|---|---|---|
| stairs/pyramid_stairs/up/L9 | 통과 | 96.9 | 84.4 | 87.5 | 87.5 | 90.6 |
| stairs/pyramid_stairs/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L9 | 몸통 접촉 | 28.1 | 15.6 | 15.6 | 9.4 | 15.6 |
| stairs/pyramid_stairs/up/L14 | 통과 | 96.9 | 90.6 | 93.8 | 96.9 | 96.9 |
| stairs/pyramid_stairs/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L14 | 몸통 접촉 | 50.0 | 12.5 | 18.8 | 9.4 | 6.2 |
| stairs/pyramid_stairs/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L9 | 몸통 접촉 | 46.9 | 78.1 | 56.2 | 56.2 | 59.4 |
| stairs/pyramid_stairs/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L14 | 몸통 접촉 | 87.5 | 81.2 | 93.8 | 81.2 | 87.5 |
| stairs/pyramid_stairs_inv/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L9 | 몸통 접촉 | 21.9 | 15.6 | 12.5 | 3.1 | 18.8 |
| stairs/pyramid_stairs_inv/up/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L14 | 몸통 접촉 | 50.0 | 34.4 | 12.5 | 34.4 | 21.9 |
| stairs/pyramid_stairs_inv/down/L9 | 통과 | 100.0 | 96.9 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L9 | 몸통 접촉 | 40.6 | 31.2 | 37.5 | 34.4 | 43.8 |
| stairs/pyramid_stairs_inv/down/L14 | 통과 | 100.0 | 100.0 | 96.9 | 96.9 | 90.6 |
| stairs/pyramid_stairs_inv/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L14 | 몸통 접촉 | 96.9 | 78.1 | 68.8 | 75.0 | 90.6 |
| stairs/pyramid_stairs_wide/up/L9 | 통과 | 96.9 | 93.8 | 93.8 | 93.8 | 90.6 |
| stairs/pyramid_stairs_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L9 | 몸통 접촉 | 25.0 | 6.2 | 9.4 | 9.4 | 12.5 |
| stairs/pyramid_stairs_wide/up/L14 | 통과 | 90.6 | 93.8 | 87.5 | 87.5 | 90.6 |
| stairs/pyramid_stairs_wide/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L14 | 몸통 접촉 | 37.5 | 40.6 | 25.0 | 15.6 | 9.4 |
| stairs/pyramid_stairs_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L9 | 몸통 접촉 | 59.4 | 71.9 | 78.1 | 56.2 | 59.4 |
| stairs/pyramid_stairs_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 몸통 접촉 | 21.9 | 6.2 | 21.9 | 15.6 | 12.5 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 통과 | 100.0 | 100.0 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 넘어짐 | 0.0 | 0.0 | 3.1 | 3.1 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 몸통 접촉 | 59.4 | 31.2 | 37.5 | 18.8 | 25.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 통과 | 100.0 | 100.0 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 몸통 접촉 | 65.6 | 87.5 | 78.1 | 71.9 | 62.5 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 통과 | 96.9 | 100.0 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 93.8 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 통과 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/up/L9 | 몸통 접촉 | 31.2 | 6.2 | 28.1 | 6.2 | 25.0 |
| stairs/pyramid_stairs_nose/up/L14 | 통과 | 71.9 | 87.5 | 75.0 | 78.1 | 81.2 |
| stairs/pyramid_stairs_nose/up/L14 | 넘어짐 | 18.8 | 9.4 | 12.5 | 9.4 | 9.4 |
| stairs/pyramid_stairs_nose/up/L14 | 몸통 접촉 | 71.9 | 31.2 | 56.2 | 40.6 | 34.4 |
| stairs/pyramid_stairs_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L9 | 몸통 접촉 | 68.8 | 78.1 | 71.9 | 62.5 | 75.0 |
| stairs/pyramid_stairs_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 통과 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 넘어짐 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 몸통 접촉 | 65.6 | 50.0 | 53.1 | 40.6 | 50.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 통과 | 93.8 | 96.9 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 넘어짐 | 6.2 | 3.1 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 몸통 접촉 | 81.2 | 50.0 | 56.2 | 62.5 | 53.1 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 통과 | 100.0 | 100.0 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 몸통 접촉 | 59.4 | 78.1 | 68.8 | 62.5 | 59.4 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 통과 | 100.0 | 96.9 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 몸통 접촉 | 81.2 | 81.2 | 84.4 | 71.9 | 75.0 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 97.5 | 97.5 | 96.6 | 97.0 | 97.3 |
| 평균 | 넘어짐 | 1.2 | 0.5 | 0.7 | 0.5 | 0.4 |
| 평균 | 몸통 접촉 | 60.4 | 52.7 | 53.3 | 47.4 | 49.6 |

### 경사·거친 지면

| 셀 | 지표 | b6566 | a1396 | a1283 | a1973 | a1929 |
|---|---|---|---|---|---|---|
| terrain/hf_pyramid_slope/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/hf_pyramid_slope/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/hf_pyramid_slope/L14 | 속도 오차 m/s | 0.193 | 0.251 | 0.257 | 0.230 | 0.229 |
| terrain/hf_pyramid_slope_inv/L14 | 통과 | 93.8 | 96.9 | 96.9 | 96.9 | 96.9 |
| terrain/hf_pyramid_slope_inv/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/hf_pyramid_slope_inv/L14 | 속도 오차 m/s | 0.207 | 0.246 | 0.225 | 0.227 | 0.217 |
| terrain/random_rough/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/random_rough/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/random_rough/L14 | 속도 오차 m/s | 0.158 | 0.264 | 0.261 | 0.237 | 0.249 |
| terrain/ring_fence/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/ring_fence/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/ring_fence/L14 | 속도 오차 m/s | 0.169 | 0.247 | 0.255 | 0.224 | 0.225 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 98.4 | 99.2 | 99.2 | 99.2 | 99.2 |
| 평균 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 평균 | 속도 오차 m/s | 0.182 | 0.252 | 0.249 | 0.229 | 0.230 |

## 기준 대비 하락 (⚠)

- 없음
