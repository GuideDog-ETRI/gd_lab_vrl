# E2 결과: aprime_final_p3

- 기준 모델: b6566, 기존 능력 지표가 기준보다 1%p 넘게 낮으면 ⚠
- seed 42, 에피소드 20 s, env마다 첫 에피소드 1회

## 조건 nominal (라이다 both, 탑재물 5.0 kg, 밀기 False, 계단 외란 False)

### 갭

| 셀 | 지표 | b6566 | a1993 | a1941 | a1991 | a2000_final |
|---|---|---|---|---|---|---|
| gap/forward/L7/v0.3 | 건넘 | 95.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.3 | clean/시도 | 73.9 | 72.9 | 77.1 | 58.3 | 66.7 |
| gap/forward/L7/v0.3 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 머리 들림 최대 평균 | 12.8° | 3.6° | 4.3° | 3.6° | 3.6° |
| gap/forward/L7/v0.3 | 머리 들림 최대 p90 | 17.3° | 7.7° | 8.5° | 8.3° | 7.7° |
| gap/forward/L7/v0.3 | 20° 초과 | 6.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 15° 초과 시간 | 0.35 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L7/v0.5 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.5 | clean/시도 | 66.0 | 83.3 | 89.6 | 85.4 | 79.2 |
| gap/forward/L7/v0.5 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 머리 들림 최대 평균 | 13.2° | 3.9° | 4.4° | 3.6° | 3.7° |
| gap/forward/L7/v0.5 | 머리 들림 최대 p90 | 21.8° | 7.3° | 7.7° | 7.5° | 6.7° |
| gap/forward/L7/v0.5 | 20° 초과 | 12.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 15° 초과 시간 | 0.09 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.9 | clean/시도 | 87.5 | 93.8 | 89.6 | 91.7 | 95.8 |
| gap/forward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 머리 들림 최대 평균 | 11.8° | 3.8° | 4.3° | 3.7° | 3.8° |
| gap/forward/L7/v0.9 | 머리 들림 최대 p90 | 16.6° | 8.4° | 8.7° | 8.2° | 8.4° |
| gap/forward/L7/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 15° 초과 시간 | 0.02 s | 0.00 s | 0.01 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.3 | 건넘 | 91.7 | 100.0 | 100.0 | 97.9 | 100.0 |
| gap/forward/L11/v0.3 | clean/시도 | 70.5 | 83.3 | 72.9 | 72.3 | 81.2 |
| gap/forward/L11/v0.3 | 넘어짐 | 8.3 | 0.0 | 0.0 | 2.1 | 0.0 |
| gap/forward/L11/v0.3 | 머리 들림 최대 평균 | 16.6° | 8.0° | 8.4° | 7.2° | 7.3° |
| gap/forward/L11/v0.3 | 머리 들림 최대 p90 | 24.2° | 11.4° | 13.1° | 11.3° | 11.8° |
| gap/forward/L11/v0.3 | 20° 초과 | 25.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.3 | 15° 초과 시간 | 0.19 s | 0.01 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.5 | 건넘 | 85.4 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L11/v0.5 | clean/시도 | 78.0 | 75.0 | 70.8 | 77.1 | 77.1 |
| gap/forward/L11/v0.5 | 넘어짐 | 14.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 머리 들림 최대 평균 | 16.1° | 5.0° | 6.2° | 5.5° | 5.4° |
| gap/forward/L11/v0.5 | 머리 들림 최대 p90 | 22.3° | 10.3° | 12.3° | 11.1° | 10.4° |
| gap/forward/L11/v0.5 | 20° 초과 | 35.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 15° 초과 시간 | 0.18 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.9 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L11/v0.9 | clean/시도 | 76.6 | 87.5 | 89.6 | 85.4 | 89.6 |
| gap/forward/L11/v0.9 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 머리 들림 최대 평균 | 14.5° | 5.7° | 6.1° | 5.2° | 5.1° |
| gap/forward/L11/v0.9 | 머리 들림 최대 p90 | 18.2° | 12.0° | 12.1° | 11.5° | 11.9° |
| gap/forward/L11/v0.9 | 20° 초과 | 4.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 15° 초과 시간 | 0.11 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L14/v0.3 | 건넘 | 77.1 | 95.8 | 95.8 | 97.9 | 100.0 |
| gap/forward/L14/v0.3 | clean/시도 | 83.8 | 76.1 | 78.3 | 70.2 | 81.2 |
| gap/forward/L14/v0.3 | 넘어짐 | 22.9 | 4.2 | 4.2 | 2.1 | 0.0 |
| gap/forward/L14/v0.3 | 머리 들림 최대 평균 | 17.7° | 6.6° | 5.9° | 6.7° | 6.4° |
| gap/forward/L14/v0.3 | 머리 들림 최대 p90 | 23.6° | 14.3° | 13.4° | 13.3° | 13.1° |
| gap/forward/L14/v0.3 | 20° 초과 | 52.1 | 2.1 | 0.0 | 2.1 | 2.1 |
| gap/forward/L14/v0.3 | 15° 초과 시간 | 0.34 s | 0.02 s | 0.01 s | 0.00 s | 0.02 s |
| gap/forward/L14/v0.5 | 건넘 | 85.4 | 95.8 | 95.8 | 97.9 | 97.9 |
| gap/forward/L14/v0.5 | clean/시도 | 68.3 | 84.8 | 80.4 | 76.6 | 78.7 |
| gap/forward/L14/v0.5 | 넘어짐 | 14.6 | 4.2 | 4.2 | 2.1 | 2.1 |
| gap/forward/L14/v0.5 | 머리 들림 최대 평균 | 16.8° | 4.3° | 5.0° | 4.3° | 4.8° |
| gap/forward/L14/v0.5 | 머리 들림 최대 p90 | 22.4° | 13.2° | 14.1° | 12.5° | 12.5° |
| gap/forward/L14/v0.5 | 20° 초과 | 37.5 | 0.0 | 2.1 | 0.0 | 0.0 |
| gap/forward/L14/v0.5 | 15° 초과 시간 | 0.24 s | 0.00 s | 0.01 s | 0.00 s | 0.00 s |
| gap/forward/L14/v0.9 | 건넘 | 89.6 | 97.9 | 97.9 | 97.9 | 100.0 |
| gap/forward/L14/v0.9 | clean/시도 | 62.8 | 78.7 | 78.7 | 80.9 | 79.2 |
| gap/forward/L14/v0.9 | 넘어짐 | 10.4 | 2.1 | 2.1 | 2.1 | 0.0 |
| gap/forward/L14/v0.9 | 머리 들림 최대 평균 | 15.4° | 4.8° | 4.6° | 4.4° | 4.8° |
| gap/forward/L14/v0.9 | 머리 들림 최대 p90 | 20.3° | 13.1° | 12.9° | 12.0° | 12.6° |
| gap/forward/L14/v0.9 | 20° 초과 | 12.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.9 | 15° 초과 시간 | 0.19 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.3 | 건넘 | 91.7 | 100.0 | 100.0 | 100.0 | 97.9 |
| gap/backward/L7/v0.3 | clean/시도 | 40.9 | 77.1 | 83.3 | 75.0 | 78.7 |
| gap/backward/L7/v0.3 | 넘어짐 | 8.3 | 0.0 | 0.0 | 0.0 | 2.1 |
| gap/backward/L7/v0.3 | 머리 들림 최대 평균 | 9.5° | 9.2° | 9.1° | 9.1° | 9.4° |
| gap/backward/L7/v0.3 | 머리 들림 최대 p90 | 11.5° | 11.1° | 10.9° | 11.5° | 11.6° |
| gap/backward/L7/v0.3 | 20° 초과 | 6.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.3 | 15° 초과 시간 | 0.01 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.5 | clean/시도 | 70.8 | 81.2 | 89.6 | 79.2 | 87.5 |
| gap/backward/L7/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.5 | 머리 들림 최대 평균 | 8.1° | 7.4° | 7.6° | 7.9° | 7.8° |
| gap/backward/L7/v0.5 | 머리 들림 최대 p90 | 10.1° | 9.1° | 8.9° | 9.4° | 9.2° |
| gap/backward/L7/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.9 | clean/시도 | 81.2 | 89.6 | 89.6 | 87.5 | 93.8 |
| gap/backward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 머리 들림 최대 평균 | 6.9° | 5.5° | 5.3° | 5.3° | 5.4° |
| gap/backward/L7/v0.9 | 머리 들림 최대 p90 | 8.6° | 7.0° | 6.3° | 6.7° | 6.7° |
| gap/backward/L7/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.3 | 건넘 | 93.8 | 100.0 | 100.0 | 100.0 | 95.8 |
| gap/backward/L11/v0.3 | clean/시도 | 46.7 | 66.7 | 87.5 | 75.0 | 76.1 |
| gap/backward/L11/v0.3 | 넘어짐 | 6.2 | 0.0 | 0.0 | 0.0 | 4.2 |
| gap/backward/L11/v0.3 | 머리 들림 최대 평균 | 9.2° | 8.8° | 9.1° | 8.7° | 9.4° |
| gap/backward/L11/v0.3 | 머리 들림 최대 p90 | 11.5° | 12.2° | 11.8° | 11.6° | 12.5° |
| gap/backward/L11/v0.3 | 20° 초과 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.3 | 15° 초과 시간 | 0.01 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.5 | clean/시도 | 87.5 | 85.4 | 83.3 | 85.4 | 79.2 |
| gap/backward/L11/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 머리 들림 최대 평균 | 8.6° | 8.0° | 7.6° | 8.2° | 7.7° |
| gap/backward/L11/v0.5 | 머리 들림 최대 p90 | 10.9° | 11.1° | 10.1° | 10.7° | 11.2° |
| gap/backward/L11/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.9 | clean/시도 | 75.0 | 85.4 | 85.4 | 77.1 | 83.3 |
| gap/backward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 머리 들림 최대 평균 | 7.3° | 5.4° | 5.8° | 5.6° | 5.7° |
| gap/backward/L11/v0.9 | 머리 들림 최대 p90 | 9.5° | 7.6° | 8.1° | 7.5° | 7.9° |
| gap/backward/L11/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.3 | 건넘 | 87.5 | 100.0 | 100.0 | 97.9 | 95.8 |
| gap/backward/L14/v0.3 | clean/시도 | 21.4 | 60.4 | 70.8 | 66.0 | 66.0 |
| gap/backward/L14/v0.3 | 넘어짐 | 12.5 | 0.0 | 0.0 | 2.1 | 4.2 |
| gap/backward/L14/v0.3 | 머리 들림 최대 평균 | 10.7° | 7.7° | 7.8° | 7.5° | 7.7° |
| gap/backward/L14/v0.3 | 머리 들림 최대 p90 | 20.2° | 11.3° | 11.3° | 11.4° | 10.9° |
| gap/backward/L14/v0.3 | 20° 초과 | 10.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.3 | 15° 초과 시간 | 0.04 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.5 | 건넘 | 93.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L14/v0.5 | clean/시도 | 62.2 | 75.0 | 83.3 | 75.0 | 75.0 |
| gap/backward/L14/v0.5 | 넘어짐 | 6.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 머리 들림 최대 평균 | 7.5° | 6.2° | 6.0° | 6.0° | 6.4° |
| gap/backward/L14/v0.5 | 머리 들림 최대 p90 | 9.7° | 9.2° | 8.9° | 9.1° | 9.3° |
| gap/backward/L14/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L14/v0.9 | clean/시도 | 68.8 | 85.4 | 81.2 | 85.4 | 85.4 |
| gap/backward/L14/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.9 | 머리 들림 최대 평균 | 6.7° | 5.6° | 5.7° | 5.8° | 5.6° |
| gap/backward/L14/v0.9 | 머리 들림 최대 p90 | 8.8° | 8.5° | 8.8° | 8.3° | 9.0° |
| gap/backward/L14/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| **평균** | |  |  |  |  |  |
| 평균 | 건넘 | 93.8 | 99.4 | 99.4 | 99.4 | 99.3 |
| 평균 | clean/시도 | 67.9 | 80.1 | 82.3 | 78.0 | 80.8 |
| 평균 | 넘어짐 | 6.1 | 0.6 | 0.6 | 0.6 | 0.7 |
| 평균 | 머리 들림 최대 평균 | 11.6° | 6.1° | 6.3° | 6.0° | 6.1° |
| 평균 | 머리 들림 최대 p90 | 16.0° | 10.3° | 10.4° | 10.1° | 10.2° |
| 평균 | 20° 초과 | 11.3 | 0.1 | 0.1 | 0.1 | 0.1 |
| 평균 | 15° 초과 시간 | 0.10 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |

### 계단

| 셀 | 지표 | b6566 | a1993 | a1941 | a1991 | a2000_final |
|---|---|---|---|---|---|---|
| stairs/pyramid_stairs/up/L9 | 통과 | 96.9 | 87.5 | 90.6 | 87.5 | 90.6 |
| stairs/pyramid_stairs/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L9 | 몸통 접촉 | 21.9 | 9.4 | 9.4 | 12.5 | 15.6 |
| stairs/pyramid_stairs/up/L14 | 통과 | 96.9 | 100.0 | 100.0 | 93.8 | 96.9 |
| stairs/pyramid_stairs/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L14 | 몸통 접촉 | 43.8 | 6.2 | 3.1 | 12.5 | 12.5 |
| stairs/pyramid_stairs/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L9 | 몸통 접촉 | 53.1 | 59.4 | 53.1 | 53.1 | 65.6 |
| stairs/pyramid_stairs/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L14 | 몸통 접촉 | 87.5 | 81.2 | 90.6 | 81.2 | 78.1 |
| stairs/pyramid_stairs_inv/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L9 | 몸통 접촉 | 31.2 | 12.5 | 12.5 | 15.6 | 21.9 |
| stairs/pyramid_stairs_inv/up/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L14 | 몸통 접촉 | 75.0 | 37.5 | 18.8 | 37.5 | 28.1 |
| stairs/pyramid_stairs_inv/down/L9 | 통과 | 100.0 | 96.9 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L9 | 몸통 접촉 | 34.4 | 46.9 | 46.9 | 34.4 | 46.9 |
| stairs/pyramid_stairs_inv/down/L14 | 통과 | 96.9 | 96.9 | 93.8 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L14 | 몸통 접촉 | 90.6 | 75.0 | 71.9 | 75.0 | 75.0 |
| stairs/pyramid_stairs_wide/up/L9 | 통과 | 93.8 | 90.6 | 93.8 | 87.5 | 90.6 |
| stairs/pyramid_stairs_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L9 | 몸통 접촉 | 18.8 | 6.2 | 6.2 | 12.5 | 15.6 |
| stairs/pyramid_stairs_wide/up/L14 | 통과 | 90.6 | 84.4 | 90.6 | 87.5 | 90.6 |
| stairs/pyramid_stairs_wide/up/L14 | 넘어짐 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L14 | 몸통 접촉 | 40.6 | 18.8 | 12.5 | 15.6 | 9.4 |
| stairs/pyramid_stairs_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L9 | 몸통 접촉 | 56.2 | 53.1 | 53.1 | 56.2 | 56.2 |
| stairs/pyramid_stairs_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L14 | 몸통 접촉 | 100.0 | 93.8 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 몸통 접촉 | 31.2 | 28.1 | 28.1 | 25.0 | 28.1 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 3.1 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 몸통 접촉 | 53.1 | 15.6 | 28.1 | 40.6 | 28.1 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 통과 | 100.0 | 96.9 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 몸통 접촉 | 62.5 | 68.8 | 84.4 | 78.1 | 56.2 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/up/L9 | 몸통 접촉 | 31.2 | 21.9 | 28.1 | 25.0 | 9.4 |
| stairs/pyramid_stairs_nose/up/L14 | 통과 | 68.8 | 90.6 | 81.2 | 78.1 | 87.5 |
| stairs/pyramid_stairs_nose/up/L14 | 넘어짐 | 28.1 | 3.1 | 6.2 | 6.2 | 3.1 |
| stairs/pyramid_stairs_nose/up/L14 | 몸통 접촉 | 75.0 | 43.8 | 46.9 | 53.1 | 34.4 |
| stairs/pyramid_stairs_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L9 | 몸통 접촉 | 62.5 | 65.6 | 68.8 | 78.1 | 81.2 |
| stairs/pyramid_stairs_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 몸통 접촉 | 50.0 | 46.9 | 34.4 | 46.9 | 40.6 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 통과 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 넘어짐 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 몸통 접촉 | 87.5 | 62.5 | 53.1 | 62.5 | 53.1 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 몸통 접촉 | 59.4 | 68.8 | 62.5 | 68.8 | 62.5 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 통과 | 100.0 | 96.9 | 96.9 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 몸통 접촉 | 78.1 | 75.0 | 71.9 | 75.0 | 78.1 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 97.5 | 97.5 | 97.5 | 96.9 | 97.7 |
| 평균 | 넘어짐 | 1.4 | 0.1 | 0.3 | 0.3 | 0.3 |
| 평균 | 몸통 접촉 | 60.2 | 49.9 | 49.3 | 52.5 | 49.9 |

### 경사·거친 지면

| 셀 | 지표 | b6566 | a1993 | a1941 | a1991 | a2000_final |
|---|---|---|---|---|---|---|
| terrain/hf_pyramid_slope/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/hf_pyramid_slope/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/hf_pyramid_slope/L14 | 속도 오차 m/s | 0.195 | 0.234 | 0.233 | 0.237 | 0.235 |
| terrain/hf_pyramid_slope_inv/L14 | 통과 | 93.8 | 96.9 | 96.9 | 96.9 | 96.9 |
| terrain/hf_pyramid_slope_inv/L14 | 넘어짐 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/hf_pyramid_slope_inv/L14 | 속도 오차 m/s | 0.208 | 0.237 | 0.231 | 0.231 | 0.230 |
| terrain/random_rough/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/random_rough/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/random_rough/L14 | 속도 오차 m/s | 0.162 | 0.243 | 0.255 | 0.245 | 0.242 |
| terrain/ring_fence/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/ring_fence/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/ring_fence/L14 | 속도 오차 m/s | 0.171 | 0.230 | 0.228 | 0.231 | 0.227 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 98.4 | 99.2 | 99.2 | 99.2 | 99.2 |
| 평균 | 넘어짐 | 0.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| 평균 | 속도 오차 m/s | 0.184 | 0.236 | 0.237 | 0.236 | 0.233 |

## 조건 no_payload (라이다 both, 탑재물 0.0 kg, 밀기 False, 계단 외란 False)

### 갭

| 셀 | 지표 | b6566 | a1993 | a1941 | a1991 | a2000_final |
|---|---|---|---|---|---|---|
| gap/forward/L7/v0.3 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.3 | clean/시도 | 74.5 | 66.7 | 62.5 | 70.8 | 77.1 |
| gap/forward/L7/v0.3 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 머리 들림 최대 평균 | 14.0° | 3.7° | 3.9° | 3.4° | 3.0° |
| gap/forward/L7/v0.3 | 머리 들림 최대 p90 | 21.1° | 7.8° | 8.0° | 7.3° | 6.7° |
| gap/forward/L7/v0.3 | 20° 초과 | 12.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 15° 초과 시간 | 0.37 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L7/v0.5 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.5 | clean/시도 | 80.9 | 81.2 | 85.4 | 83.3 | 83.3 |
| gap/forward/L7/v0.5 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 머리 들림 최대 평균 | 13.8° | 4.0° | 3.9° | 3.8° | 3.8° |
| gap/forward/L7/v0.5 | 머리 들림 최대 p90 | 18.9° | 7.8° | 7.6° | 7.9° | 7.3° |
| gap/forward/L7/v0.5 | 20° 초과 | 8.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 15° 초과 시간 | 0.17 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.9 | clean/시도 | 89.6 | 95.8 | 91.7 | 93.8 | 93.8 |
| gap/forward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 머리 들림 최대 평균 | 11.5° | 3.5° | 3.5° | 3.5° | 3.4° |
| gap/forward/L7/v0.9 | 머리 들림 최대 p90 | 16.3° | 8.2° | 7.6° | 7.7° | 8.5° |
| gap/forward/L7/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 15° 초과 시간 | 0.02 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.3 | 건넘 | 85.4 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L11/v0.3 | clean/시도 | 68.3 | 83.3 | 85.4 | 83.3 | 70.8 |
| gap/forward/L11/v0.3 | 넘어짐 | 14.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.3 | 머리 들림 최대 평균 | 16.1° | 7.4° | 6.9° | 7.2° | 7.3° |
| gap/forward/L11/v0.3 | 머리 들림 최대 p90 | 22.8° | 11.5° | 11.4° | 10.7° | 11.7° |
| gap/forward/L11/v0.3 | 20° 초과 | 33.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.3 | 15° 초과 시간 | 0.17 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.5 | 건넘 | 95.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L11/v0.5 | clean/시도 | 78.3 | 77.1 | 83.3 | 72.9 | 72.9 |
| gap/forward/L11/v0.5 | 넘어짐 | 4.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 머리 들림 최대 평균 | 16.7° | 5.0° | 5.9° | 5.5° | 5.3° |
| gap/forward/L11/v0.5 | 머리 들림 최대 p90 | 21.7° | 10.8° | 11.6° | 11.1° | 10.8° |
| gap/forward/L11/v0.5 | 20° 초과 | 18.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 15° 초과 시간 | 0.16 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L11/v0.9 | clean/시도 | 77.1 | 89.6 | 89.6 | 85.4 | 87.5 |
| gap/forward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 머리 들림 최대 평균 | 14.9° | 5.4° | 5.5° | 5.1° | 5.5° |
| gap/forward/L11/v0.9 | 머리 들림 최대 p90 | 18.8° | 11.8° | 11.7° | 11.1° | 11.6° |
| gap/forward/L11/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 15° 초과 시간 | 0.10 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L14/v0.3 | 건넘 | 75.0 | 100.0 | 97.9 | 89.6 | 97.9 |
| gap/forward/L14/v0.3 | clean/시도 | 77.8 | 77.1 | 78.7 | 79.1 | 83.0 |
| gap/forward/L14/v0.3 | 넘어짐 | 25.0 | 0.0 | 2.1 | 10.4 | 2.1 |
| gap/forward/L14/v0.3 | 머리 들림 최대 평균 | 17.4° | 6.4° | 7.6° | 5.5° | 5.9° |
| gap/forward/L14/v0.3 | 머리 들림 최대 p90 | 23.5° | 13.2° | 11.3° | 10.2° | 10.5° |
| gap/forward/L14/v0.3 | 20° 초과 | 35.4 | 0.0 | 4.2 | 2.1 | 2.1 |
| gap/forward/L14/v0.3 | 15° 초과 시간 | 0.31 s | 0.01 s | 0.01 s | 0.01 s | 0.01 s |
| gap/forward/L14/v0.5 | 건넘 | 81.2 | 95.8 | 93.8 | 95.8 | 95.8 |
| gap/forward/L14/v0.5 | clean/시도 | 71.8 | 80.4 | 84.4 | 82.6 | 78.3 |
| gap/forward/L14/v0.5 | 넘어짐 | 18.8 | 4.2 | 6.2 | 4.2 | 4.2 |
| gap/forward/L14/v0.5 | 머리 들림 최대 평균 | 17.1° | 4.4° | 4.8° | 4.2° | 5.4° |
| gap/forward/L14/v0.5 | 머리 들림 최대 p90 | 23.1° | 10.8° | 11.2° | 11.8° | 13.0° |
| gap/forward/L14/v0.5 | 20° 초과 | 35.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.5 | 15° 초과 시간 | 0.23 s | 0.00 s | 0.00 s | 0.01 s | 0.00 s |
| gap/forward/L14/v0.9 | 건넘 | 93.8 | 100.0 | 97.9 | 97.9 | 97.9 |
| gap/forward/L14/v0.9 | clean/시도 | 71.1 | 77.1 | 87.2 | 83.0 | 78.7 |
| gap/forward/L14/v0.9 | 넘어짐 | 6.2 | 0.0 | 2.1 | 2.1 | 2.1 |
| gap/forward/L14/v0.9 | 머리 들림 최대 평균 | 16.0° | 4.0° | 4.7° | 4.5° | 4.5° |
| gap/forward/L14/v0.9 | 머리 들림 최대 p90 | 19.5° | 12.2° | 12.1° | 12.7° | 12.2° |
| gap/forward/L14/v0.9 | 20° 초과 | 8.3 | 0.0 | 0.0 | 0.0 | 2.1 |
| gap/forward/L14/v0.9 | 15° 초과 시간 | 0.20 s | 0.00 s | 0.00 s | 0.00 s | 0.02 s |
| gap/backward/L7/v0.3 | 건넘 | 95.8 | 100.0 | 100.0 | 97.9 | 100.0 |
| gap/backward/L7/v0.3 | clean/시도 | 39.1 | 83.3 | 85.4 | 77.1 | 83.3 |
| gap/backward/L7/v0.3 | 넘어짐 | 2.1 | 0.0 | 0.0 | 2.1 | 0.0 |
| gap/backward/L7/v0.3 | 머리 들림 최대 평균 | 9.0° | 9.1° | 9.2° | 9.2° | 9.2° |
| gap/backward/L7/v0.3 | 머리 들림 최대 p90 | 10.4° | 10.8° | 10.7° | 11.4° | 10.9° |
| gap/backward/L7/v0.3 | 20° 초과 | 4.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.3 | 15° 초과 시간 | 0.34 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.5 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.5 | clean/시도 | 61.7 | 85.4 | 81.2 | 70.8 | 81.2 |
| gap/backward/L7/v0.5 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.5 | 머리 들림 최대 평균 | 8.5° | 7.9° | 7.6° | 7.7° | 7.8° |
| gap/backward/L7/v0.5 | 머리 들림 최대 p90 | 10.0° | 9.8° | 9.3° | 9.7° | 9.5° |
| gap/backward/L7/v0.5 | 20° 초과 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.9 | clean/시도 | 87.5 | 91.7 | 89.6 | 91.7 | 91.7 |
| gap/backward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 머리 들림 최대 평균 | 6.8° | 5.7° | 5.2° | 5.7° | 5.7° |
| gap/backward/L7/v0.9 | 머리 들림 최대 p90 | 8.4° | 7.3° | 6.6° | 6.7° | 6.6° |
| gap/backward/L7/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 2.1 |
| gap/backward/L7/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.01 s |
| gap/backward/L11/v0.3 | 건넘 | 91.7 | 100.0 | 97.9 | 97.9 | 97.9 |
| gap/backward/L11/v0.3 | clean/시도 | 31.8 | 68.8 | 76.6 | 72.3 | 68.1 |
| gap/backward/L11/v0.3 | 넘어짐 | 8.3 | 0.0 | 2.1 | 2.1 | 2.1 |
| gap/backward/L11/v0.3 | 머리 들림 최대 평균 | 10.3° | 9.0° | 9.0° | 8.8° | 8.9° |
| gap/backward/L11/v0.3 | 머리 들림 최대 p90 | 14.6° | 12.0° | 11.4° | 11.6° | 11.7° |
| gap/backward/L11/v0.3 | 20° 초과 | 8.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.3 | 15° 초과 시간 | 0.04 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.5 | clean/시도 | 62.5 | 83.3 | 85.4 | 89.6 | 79.2 |
| gap/backward/L11/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 머리 들림 최대 평균 | 8.4° | 8.0° | 8.0° | 7.9° | 8.3° |
| gap/backward/L11/v0.5 | 머리 들림 최대 p90 | 10.3° | 11.1° | 10.7° | 10.9° | 11.6° |
| gap/backward/L11/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.9 | clean/시도 | 81.2 | 81.2 | 81.2 | 83.3 | 79.2 |
| gap/backward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 머리 들림 최대 평균 | 6.9° | 5.8° | 5.7° | 5.5° | 5.7° |
| gap/backward/L11/v0.9 | 머리 들림 최대 p90 | 9.3° | 8.8° | 8.2° | 8.3° | 8.4° |
| gap/backward/L11/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.3 | 건넘 | 81.2 | 95.8 | 100.0 | 95.8 | 89.6 |
| gap/backward/L14/v0.3 | clean/시도 | 17.9 | 80.9 | 56.2 | 67.4 | 70.5 |
| gap/backward/L14/v0.3 | 넘어짐 | 18.8 | 4.2 | 0.0 | 4.2 | 10.4 |
| gap/backward/L14/v0.3 | 머리 들림 최대 평균 | 12.0° | 7.4° | 8.1° | 7.9° | 7.4° |
| gap/backward/L14/v0.3 | 머리 들림 최대 p90 | 25.8° | 11.2° | 11.7° | 12.7° | 11.6° |
| gap/backward/L14/v0.3 | 20° 초과 | 20.8 | 0.0 | 0.0 | 2.1 | 0.0 |
| gap/backward/L14/v0.3 | 15° 초과 시간 | 0.06 s | 0.00 s | 0.00 s | 0.01 s | 0.00 s |
| gap/backward/L14/v0.5 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L14/v0.5 | clean/시도 | 57.4 | 72.9 | 77.1 | 83.3 | 79.2 |
| gap/backward/L14/v0.5 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 머리 들림 최대 평균 | 8.5° | 6.2° | 6.4° | 5.9° | 6.2° |
| gap/backward/L14/v0.5 | 머리 들림 최대 p90 | 11.1° | 9.3° | 9.6° | 9.2° | 10.2° |
| gap/backward/L14/v0.5 | 20° 초과 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L14/v0.9 | clean/시도 | 72.9 | 83.3 | 87.5 | 85.4 | 81.2 |
| gap/backward/L14/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.9 | 머리 들림 최대 평균 | 6.7° | 5.6° | 5.5° | 5.3° | 5.3° |
| gap/backward/L14/v0.9 | 머리 들림 최대 p90 | 8.9° | 9.1° | 8.7° | 8.1° | 8.8° |
| gap/backward/L14/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| **평균** | |  |  |  |  |  |
| 평균 | 건넘 | 94.0 | 99.5 | 99.3 | 98.6 | 98.8 |
| 평균 | clean/시도 | 66.7 | 81.1 | 81.6 | 80.8 | 79.9 |
| 평균 | 넘어짐 | 5.8 | 0.5 | 0.7 | 1.4 | 1.2 |
| 평균 | 머리 들림 최대 평균 | 11.9° | 6.0° | 6.2° | 5.9° | 6.0° |
| 평균 | 머리 들림 최대 p90 | 16.4° | 10.2° | 10.0° | 9.9° | 10.1° |
| 평균 | 20° 초과 | 10.5 | 0.0 | 0.2 | 0.2 | 0.3 |
| 평균 | 15° 초과 시간 | 0.12 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |

### 계단

| 셀 | 지표 | b6566 | a1993 | a1941 | a1991 | a2000_final |
|---|---|---|---|---|---|---|
| stairs/pyramid_stairs/up/L9 | 통과 | 96.9 | 93.8 | 87.5 | 87.5 | 84.4 |
| stairs/pyramid_stairs/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L9 | 몸통 접촉 | 28.1 | 6.2 | 12.5 | 15.6 | 9.4 |
| stairs/pyramid_stairs/up/L14 | 통과 | 96.9 | 100.0 | 96.9 | 100.0 | 96.9 |
| stairs/pyramid_stairs/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L14 | 몸통 접촉 | 50.0 | 6.2 | 6.2 | 6.2 | 6.2 |
| stairs/pyramid_stairs/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L9 | 몸통 접촉 | 46.9 | 43.8 | 62.5 | 40.6 | 50.0 |
| stairs/pyramid_stairs/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L14 | 몸통 접촉 | 87.5 | 90.6 | 93.8 | 87.5 | 87.5 |
| stairs/pyramid_stairs_inv/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L9 | 몸통 접촉 | 21.9 | 12.5 | 12.5 | 6.2 | 12.5 |
| stairs/pyramid_stairs_inv/up/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L14 | 몸통 접촉 | 50.0 | 43.8 | 34.4 | 53.1 | 40.6 |
| stairs/pyramid_stairs_inv/down/L9 | 통과 | 100.0 | 96.9 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L9 | 몸통 접촉 | 40.6 | 50.0 | 34.4 | 37.5 | 34.4 |
| stairs/pyramid_stairs_inv/down/L14 | 통과 | 100.0 | 96.9 | 96.9 | 93.8 | 93.8 |
| stairs/pyramid_stairs_inv/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L14 | 몸통 접촉 | 96.9 | 71.9 | 90.6 | 87.5 | 68.8 |
| stairs/pyramid_stairs_wide/up/L9 | 통과 | 96.9 | 90.6 | 93.8 | 90.6 | 93.8 |
| stairs/pyramid_stairs_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L9 | 몸통 접촉 | 25.0 | 6.2 | 9.4 | 9.4 | 6.2 |
| stairs/pyramid_stairs_wide/up/L14 | 통과 | 90.6 | 84.4 | 93.8 | 87.5 | 87.5 |
| stairs/pyramid_stairs_wide/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L14 | 몸통 접촉 | 37.5 | 21.9 | 6.2 | 12.5 | 21.9 |
| stairs/pyramid_stairs_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L9 | 몸통 접촉 | 59.4 | 71.9 | 75.0 | 56.2 | 75.0 |
| stairs/pyramid_stairs_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L14 | 몸통 접촉 | 100.0 | 96.9 | 100.0 | 100.0 | 93.8 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 몸통 접촉 | 21.9 | 34.4 | 15.6 | 28.1 | 18.8 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 통과 | 100.0 | 100.0 | 93.8 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 넘어짐 | 0.0 | 0.0 | 6.2 | 3.1 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 몸통 접촉 | 59.4 | 21.9 | 18.8 | 40.6 | 34.4 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 통과 | 100.0 | 96.9 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 몸통 접촉 | 65.6 | 71.9 | 81.2 | 75.0 | 71.9 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 통과 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/up/L9 | 몸통 접촉 | 31.2 | 15.6 | 18.8 | 18.8 | 25.0 |
| stairs/pyramid_stairs_nose/up/L14 | 통과 | 71.9 | 78.1 | 84.4 | 84.4 | 84.4 |
| stairs/pyramid_stairs_nose/up/L14 | 넘어짐 | 18.8 | 9.4 | 9.4 | 3.1 | 3.1 |
| stairs/pyramid_stairs_nose/up/L14 | 몸통 접촉 | 71.9 | 40.6 | 46.9 | 37.5 | 46.9 |
| stairs/pyramid_stairs_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L9 | 몸통 접촉 | 68.8 | 68.8 | 71.9 | 87.5 | 84.4 |
| stairs/pyramid_stairs_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 통과 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 넘어짐 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 몸통 접촉 | 65.6 | 43.8 | 43.8 | 62.5 | 50.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 통과 | 93.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 넘어짐 | 6.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 몸통 접촉 | 81.2 | 62.5 | 56.2 | 59.4 | 46.9 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 통과 | 100.0 | 100.0 | 96.9 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 몸통 접촉 | 59.4 | 46.9 | 59.4 | 65.6 | 62.5 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 통과 | 100.0 | 96.9 | 100.0 | 96.9 | 93.8 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 몸통 접촉 | 81.2 | 75.0 | 84.4 | 68.8 | 75.0 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 97.5 | 97.3 | 97.4 | 97.0 | 97.1 |
| 평균 | 넘어짐 | 1.2 | 0.4 | 0.7 | 0.3 | 0.1 |
| 평균 | 몸통 접촉 | 60.4 | 50.1 | 51.4 | 52.3 | 50.9 |

### 경사·거친 지면

| 셀 | 지표 | b6566 | a1993 | a1941 | a1991 | a2000_final |
|---|---|---|---|---|---|---|
| terrain/hf_pyramid_slope/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/hf_pyramid_slope/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/hf_pyramid_slope/L14 | 속도 오차 m/s | 0.193 | 0.229 | 0.238 | 0.237 | 0.226 |
| terrain/hf_pyramid_slope_inv/L14 | 통과 | 93.8 | 96.9 | 96.9 | 96.9 | 96.9 |
| terrain/hf_pyramid_slope_inv/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/hf_pyramid_slope_inv/L14 | 속도 오차 m/s | 0.207 | 0.221 | 0.228 | 0.224 | 0.219 |
| terrain/random_rough/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/random_rough/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/random_rough/L14 | 속도 오차 m/s | 0.158 | 0.249 | 0.245 | 0.251 | 0.243 |
| terrain/ring_fence/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| terrain/ring_fence/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| terrain/ring_fence/L14 | 속도 오차 m/s | 0.169 | 0.230 | 0.228 | 0.226 | 0.226 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 98.4 | 99.2 | 99.2 | 99.2 | 99.2 |
| 평균 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 평균 | 속도 오차 m/s | 0.182 | 0.233 | 0.235 | 0.234 | 0.228 |

## 기준 대비 하락 (⚠)

- 없음
