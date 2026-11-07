# E2 결과: bprime_stride_s2_hangang_final_a6000

- 기준 모델: a18l, 기존 능력 지표가 기준보다 1%p 넘게 낮으면 ⚠
- seed 42, 에피소드 20 s, env마다 첫 에피소드 1회

## 조건 nominal (라이다 both, 후방 Mid-360 True, 탑재물 5.0 kg, 밀기 False, 계단 외란 False)

### 갭

| 셀 | 지표 | hg999 | hg846 | hg927 | hg861 | hg519 |
|---|---|---|---|---|---|---|
| gap/forward/L7/v0.3 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.3 | clean/시도 | 79.2 | 77.1 | 87.5 | 89.6 | 89.6 |
| gap/forward/L7/v0.3 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 머리 들림 최대 평균 | 4.9° | 5.5° | 5.7° | 5.6° | 5.2° |
| gap/forward/L7/v0.3 | 머리 들림 최대 p90 | 8.4° | 8.9° | 9.7° | 9.2° | 8.9° |
| gap/forward/L7/v0.3 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L7/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.5 | clean/시도 | 87.5 | 87.5 | 85.4 | 95.8 | 85.4 |
| gap/forward/L7/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 머리 들림 최대 평균 | 5.2° | 5.0° | 5.3° | 5.0° | 5.0° |
| gap/forward/L7/v0.5 | 머리 들림 최대 p90 | 8.9° | 8.5° | 8.5° | 8.9° | 9.7° |
| gap/forward/L7/v0.5 | 20° 초과 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.01 s |
| gap/forward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.9 | clean/시도 | 77.1 | 81.2 | 85.4 | 81.2 | 85.4 |
| gap/forward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 머리 들림 최대 평균 | 3.9° | 4.2° | 4.4° | 4.2° | 3.5° |
| gap/forward/L7/v0.9 | 머리 들림 최대 p90 | 7.8° | 7.8° | 7.4° | 7.6° | 6.4° |
| gap/forward/L7/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.3 | 건넘 | 97.9 | 100.0 | 100.0 | 97.9 | 100.0 |
| gap/forward/L11/v0.3 | clean/시도 | 78.7 | 79.2 | 85.4 | 83.0 | 81.2 |
| gap/forward/L11/v0.3 | 넘어짐 | 2.1 | 0.0 | 0.0 | 2.1 | 0.0 |
| gap/forward/L11/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.3 | 머리 들림 최대 평균 | 6.4° | 6.5° | 6.9° | 6.2° | 6.0° |
| gap/forward/L11/v0.3 | 머리 들림 최대 p90 | 10.8° | 10.6° | 11.6° | 10.4° | 9.8° |
| gap/forward/L11/v0.3 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.3 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 97.9 |
| gap/forward/L11/v0.5 | clean/시도 | 75.0 | 81.2 | 77.1 | 66.7 | 80.9 |
| gap/forward/L11/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 2.1 |
| gap/forward/L11/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 머리 들림 최대 평균 | 5.3° | 5.3° | 5.8° | 5.6° | 5.4° |
| gap/forward/L11/v0.5 | 머리 들림 최대 p90 | 9.5° | 10.0° | 10.3° | 10.0° | 10.4° |
| gap/forward/L11/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L11/v0.9 | clean/시도 | 87.5 | 79.2 | 89.6 | 81.2 | 83.3 |
| gap/forward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 머리 들림 최대 평균 | 4.8° | 5.4° | 5.5° | 5.8° | 5.2° |
| gap/forward/L11/v0.9 | 머리 들림 최대 p90 | 8.8° | 8.9° | 9.1° | 10.5° | 9.0° |
| gap/forward/L11/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 2.1 | 0.0 |
| gap/forward/L11/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.01 s | 0.00 s |
| gap/forward/L14/v0.3 | 건넘 | 95.8 | 95.8 | 97.9 | 100.0 | 91.7 |
| gap/forward/L14/v0.3 | clean/시도 | 76.1 | 89.1 | 87.2 | 83.3 | 84.1 |
| gap/forward/L14/v0.3 | 넘어짐 | 4.2 | 4.2 | 2.1 | 0.0 | 8.3 |
| gap/forward/L14/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.3 | 머리 들림 최대 평균 | 7.6° | 7.7° | 7.6° | 7.6° | 8.5° |
| gap/forward/L14/v0.3 | 머리 들림 최대 p90 | 14.3° | 12.1° | 10.8° | 14.4° | 12.3° |
| gap/forward/L14/v0.3 | 20° 초과 | 4.2 | 2.1 | 0.0 | 0.0 | 6.2 |
| gap/forward/L14/v0.3 | 15° 초과 시간 | 0.01 s | 0.01 s | 0.00 s | 0.01 s | 0.02 s |
| gap/forward/L14/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 97.9 |
| gap/forward/L14/v0.5 | clean/시도 | 85.4 | 75.0 | 79.2 | 83.3 | 76.6 |
| gap/forward/L14/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 2.1 |
| gap/forward/L14/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.5 | 머리 들림 최대 평균 | 6.7° | 7.0° | 7.6° | 6.7° | 6.6° |
| gap/forward/L14/v0.5 | 머리 들림 최대 p90 | 10.8° | 10.8° | 11.7° | 10.3° | 10.4° |
| gap/forward/L14/v0.5 | 20° 초과 | 0.0 | 0.0 | 2.1 | 0.0 | 0.0 |
| gap/forward/L14/v0.5 | 15° 초과 시간 | 0.01 s | 0.00 s | 0.01 s | 0.00 s | 0.00 s |
| gap/forward/L14/v0.9 | 건넘 | 95.8 | 100.0 | 100.0 | 97.9 | 100.0 |
| gap/forward/L14/v0.9 | clean/시도 | 78.3 | 83.3 | 81.2 | 78.7 | 81.2 |
| gap/forward/L14/v0.9 | 넘어짐 | 4.2 | 0.0 | 0.0 | 2.1 | 0.0 |
| gap/forward/L14/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.9 | 머리 들림 최대 평균 | 6.9° | 7.8° | 6.9° | 7.2° | 7.4° |
| gap/forward/L14/v0.9 | 머리 들림 최대 p90 | 11.5° | 12.9° | 10.2° | 10.8° | 12.1° |
| gap/forward/L14/v0.9 | 20° 초과 | 0.0 | 2.1 | 0.0 | 0.0 | 4.2 |
| gap/forward/L14/v0.9 | 15° 초과 시간 | 0.01 s | 0.01 s | 0.00 s | 0.00 s | 0.01 s |
| gap/backward/L7/v0.3 | 건넘 | 95.8 | 97.9 | 95.8 | 100.0 | 100.0 |
| gap/backward/L7/v0.3 | clean/시도 | 76.1 | 78.7 | 82.6 | 83.3 | 70.8 |
| gap/backward/L7/v0.3 | 넘어짐 | 4.2 | 2.1 | 4.2 | 0.0 | 0.0 |
| gap/backward/L7/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.3 | 머리 들림 최대 평균 | 7.8° | 7.1° | 7.9° | 6.7° | 5.9° |
| gap/backward/L7/v0.3 | 머리 들림 최대 p90 | 11.0° | 9.9° | 10.2° | 9.7° | 9.3° |
| gap/backward/L7/v0.3 | 20° 초과 | 4.2 | 2.1 | 4.2 | 0.0 | 0.0 |
| gap/backward/L7/v0.3 | 15° 초과 시간 | 0.02 s | 0.00 s | 0.01 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.5 | 건넘 | 97.9 | 97.9 | 97.9 | 95.8 | 97.9 |
| gap/backward/L7/v0.5 | clean/시도 | 89.4 | 87.2 | 83.0 | 84.8 | 83.0 |
| gap/backward/L7/v0.5 | 넘어짐 | 2.1 | 2.1 | 2.1 | 4.2 | 2.1 |
| gap/backward/L7/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.5 | 머리 들림 최대 평균 | 7.1° | 7.5° | 7.2° | 7.2° | 6.8° |
| gap/backward/L7/v0.5 | 머리 들림 최대 p90 | 9.1° | 10.4° | 10.1° | 8.9° | 10.4° |
| gap/backward/L7/v0.5 | 20° 초과 | 2.1 | 2.1 | 2.1 | 4.2 | 2.1 |
| gap/backward/L7/v0.5 | 15° 초과 시간 | 0.01 s | 0.00 s | 0.00 s | 0.01 s | 0.00 s |
| gap/backward/L7/v0.9 | 건넘 | 100.0 | 97.9 | 100.0 | 100.0 | 97.9 |
| gap/backward/L7/v0.9 | clean/시도 | 72.9 | 87.2 | 77.1 | 81.2 | 76.6 |
| gap/backward/L7/v0.9 | 넘어짐 | 0.0 | 2.1 | 0.0 | 0.0 | 2.1 |
| gap/backward/L7/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 머리 들림 최대 평균 | 6.1° | 6.5° | 6.4° | 6.1° | 6.1° |
| gap/backward/L7/v0.9 | 머리 들림 최대 p90 | 9.9° | 11.1° | 10.6° | 10.1° | 9.9° |
| gap/backward/L7/v0.9 | 20° 초과 | 0.0 | 2.1 | 0.0 | 0.0 | 2.1 |
| gap/backward/L7/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.3 | 건넘 | 85.4 | 87.5 | 89.6 | 91.7 | 83.3 |
| gap/backward/L11/v0.3 | clean/시도 | 80.5 | 52.4 | 65.1 | 72.7 | 65.0 |
| gap/backward/L11/v0.3 | 넘어짐 | 14.6 | 12.5 | 10.4 | 8.3 | 16.7 |
| gap/backward/L11/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.3 | 머리 들림 최대 평균 | 11.3° | 11.2° | 10.6° | 9.9° | 11.5° |
| gap/backward/L11/v0.3 | 머리 들림 최대 p90 | 31.2° | 34.4° | 28.5° | 14.3° | 34.0° |
| gap/backward/L11/v0.3 | 20° 초과 | 14.6 | 12.5 | 10.4 | 8.3 | 16.7 |
| gap/backward/L11/v0.3 | 15° 초과 시간 | 0.01 s | 0.02 s | 0.01 s | 0.01 s | 0.07 s |
| gap/backward/L11/v0.5 | 건넘 | 95.8 | 91.7 | 95.8 | 97.9 | 93.8 |
| gap/backward/L11/v0.5 | clean/시도 | 82.6 | 88.6 | 80.4 | 78.7 | 77.8 |
| gap/backward/L11/v0.5 | 넘어짐 | 4.2 | 8.3 | 4.2 | 2.1 | 6.2 |
| gap/backward/L11/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 머리 들림 최대 평균 | 8.8° | 9.5° | 10.0° | 8.3° | 8.9° |
| gap/backward/L11/v0.5 | 머리 들림 최대 p90 | 14.1° | 12.4° | 14.7° | 12.8° | 13.0° |
| gap/backward/L11/v0.5 | 20° 초과 | 4.2 | 8.3 | 6.2 | 4.2 | 6.2 |
| gap/backward/L11/v0.5 | 15° 초과 시간 | 0.01 s | 0.01 s | 0.01 s | 0.01 s | 0.01 s |
| gap/backward/L11/v0.9 | 건넘 | 97.9 | 97.9 | 95.8 | 93.8 | 91.7 |
| gap/backward/L11/v0.9 | clean/시도 | 80.9 | 83.0 | 73.9 | 75.6 | 81.8 |
| gap/backward/L11/v0.9 | 넘어짐 | 2.1 | 2.1 | 4.2 | 6.2 | 8.3 |
| gap/backward/L11/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 머리 들림 최대 평균 | 8.3° | 7.9° | 9.0° | 9.1° | 9.3° |
| gap/backward/L11/v0.9 | 머리 들림 최대 p90 | 12.8° | 11.0° | 11.8° | 10.6° | 14.9° |
| gap/backward/L11/v0.9 | 20° 초과 | 2.1 | 2.1 | 4.2 | 6.2 | 8.3 |
| gap/backward/L11/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.01 s | 0.01 s | 0.01 s |
| gap/backward/L14/v0.3 | 건넘 | 95.8 | 79.2 | 89.6 | 81.2 | 91.7 |
| gap/backward/L14/v0.3 | clean/시도 | 65.2 | 55.3 | 51.2 | 61.5 | 72.7 |
| gap/backward/L14/v0.3 | 넘어짐 | 4.2 | 20.8 | 10.4 | 18.8 | 8.3 |
| gap/backward/L14/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.3 | 머리 들림 최대 평균 | 9.6° | 14.9° | 11.9° | 12.8° | 10.5° |
| gap/backward/L14/v0.3 | 머리 들림 최대 p90 | 12.3° | 36.5° | 32.8° | 33.6° | 14.4° |
| gap/backward/L14/v0.3 | 20° 초과 | 4.2 | 20.8 | 10.4 | 16.7 | 8.3 |
| gap/backward/L14/v0.3 | 15° 초과 시간 | 0.02 s | 0.03 s | 0.03 s | 0.02 s | 0.01 s |
| gap/backward/L14/v0.5 | 건넘 | 91.7 | 91.7 | 91.7 | 89.6 | 93.8 |
| gap/backward/L14/v0.5 | clean/시도 | 47.7 | 54.5 | 56.8 | 46.5 | 62.2 |
| gap/backward/L14/v0.5 | 넘어짐 | 8.3 | 8.3 | 8.3 | 10.4 | 6.2 |
| gap/backward/L14/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 머리 들림 최대 평균 | 10.5° | 11.0° | 10.2° | 11.3° | 9.5° |
| gap/backward/L14/v0.5 | 머리 들림 최대 p90 | 19.3° | 21.6° | 12.3° | 33.2° | 14.1° |
| gap/backward/L14/v0.5 | 20° 초과 | 8.3 | 10.4 | 8.3 | 10.4 | 6.2 |
| gap/backward/L14/v0.5 | 15° 초과 시간 | 0.01 s | 0.02 s | 0.01 s | 0.01 s | 0.01 s |
| gap/backward/L14/v0.9 | 건넘 | 97.9 | 97.9 | 95.8 | 95.8 | 97.9 |
| gap/backward/L14/v0.9 | clean/시도 | 61.7 | 53.2 | 60.9 | 50.0 | 44.7 |
| gap/backward/L14/v0.9 | 넘어짐 | 2.1 | 2.1 | 4.2 | 4.2 | 2.1 |
| gap/backward/L14/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.9 | 머리 들림 최대 평균 | 8.6° | 8.1° | 8.6° | 8.6° | 7.9° |
| gap/backward/L14/v0.9 | 머리 들림 최대 p90 | 10.8° | 12.0° | 10.4° | 11.3° | 10.6° |
| gap/backward/L14/v0.9 | 20° 초과 | 2.1 | 2.1 | 4.2 | 4.2 | 2.1 |
| gap/backward/L14/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.01 s | 0.01 s | 0.00 s |
| **평균** | |  |  |  |  |  |
| 평균 | 건넘 | 97.1 | 96.4 | 97.2 | 96.8 | 96.4 |
| 평균 | clean/시도 | 76.8 | 76.3 | 77.2 | 76.5 | 76.8 |
| 평균 | 넘어짐 | 2.9 | 3.6 | 2.8 | 3.2 | 3.6 |
| 평균 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 평균 | 머리 들림 최대 평균 | 7.2° | 7.7° | 7.6° | 7.4° | 7.2° |
| 평균 | 머리 들림 최대 p90 | 12.3° | 13.9° | 12.8° | 13.1° | 12.2° |
| 평균 | 20° 초과 | 2.7 | 3.7 | 2.9 | 3.1 | 3.5 |
| 평균 | 15° 초과 시간 | 0.01 s | 0.01 s | 0.01 s | 0.01 s | 0.01 s |

### 계단

| 셀 | 지표 | hg999 | hg846 | hg927 | hg861 | hg519 |
|---|---|---|---|---|---|---|
| stairs/pyramid_stairs/up/L9 | 통과 | 90.6 | 90.6 | 90.6 | 90.6 | 87.5 |
| stairs/pyramid_stairs/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L9 | 몸통 접촉 | 12.5 | 6.2 | 12.5 | 9.4 | 9.4 |
| stairs/pyramid_stairs/up/L14 | 통과 | 93.8 | 93.8 | 96.9 | 90.6 | 93.8 |
| stairs/pyramid_stairs/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L14 | 몸통 접촉 | 12.5 | 21.9 | 18.8 | 15.6 | 21.9 |
| stairs/pyramid_stairs/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L9 | 몸통 접촉 | 59.4 | 62.5 | 46.9 | 62.5 | 50.0 |
| stairs/pyramid_stairs/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L14 | 몸통 접촉 | 68.8 | 81.2 | 59.4 | 71.9 | 78.1 |
| stairs/pyramid_stairs_inv/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L9 | 몸통 접촉 | 6.2 | 28.1 | 6.2 | 18.8 | 18.8 |
| stairs/pyramid_stairs_inv/up/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L14 | 몸통 접촉 | 37.5 | 43.8 | 37.5 | 40.6 | 37.5 |
| stairs/pyramid_stairs_inv/down/L9 | 통과 | 93.8 | 90.6 | 90.6 | 96.9 | 90.6 |
| stairs/pyramid_stairs_inv/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L9 | 몸통 접촉 | 28.1 | 31.2 | 31.2 | 34.4 | 31.2 |
| stairs/pyramid_stairs_inv/down/L14 | 통과 | 90.6 | 93.8 | 93.8 | 96.9 | 93.8 |
| stairs/pyramid_stairs_inv/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L14 | 몸통 접촉 | 84.4 | 68.8 | 71.9 | 65.6 | 81.2 |
| stairs/pyramid_stairs_wide/up/L9 | 통과 | 90.6 | 93.8 | 90.6 | 93.8 | 93.8 |
| stairs/pyramid_stairs_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L9 | 몸통 접촉 | 9.4 | 12.5 | 9.4 | 6.2 | 9.4 |
| stairs/pyramid_stairs_wide/up/L14 | 통과 | 90.6 | 90.6 | 90.6 | 87.5 | 90.6 |
| stairs/pyramid_stairs_wide/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 3.1 | 0.0 |
| stairs/pyramid_stairs_wide/up/L14 | 몸통 접촉 | 18.8 | 15.6 | 21.9 | 18.8 | 18.8 |
| stairs/pyramid_stairs_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L9 | 몸통 접촉 | 84.4 | 56.2 | 62.5 | 56.2 | 53.1 |
| stairs/pyramid_stairs_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L14 | 몸통 접촉 | 100.0 | 93.8 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 몸통 접촉 | 37.5 | 40.6 | 43.8 | 28.1 | 31.2 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 통과 | 96.9 | 87.5 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 넘어짐 | 3.1 | 12.5 | 3.1 | 3.1 | 3.1 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 몸통 접촉 | 56.2 | 53.1 | 37.5 | 50.0 | 37.5 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 통과 | 90.6 | 87.5 | 90.6 | 90.6 | 96.9 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 몸통 접촉 | 59.4 | 56.2 | 62.5 | 62.5 | 46.9 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 통과 | 100.0 | 93.8 | 93.8 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 몸통 접촉 | 100.0 | 96.9 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/up/L9 | 통과 | 84.4 | 78.1 | 84.4 | 84.4 | 84.4 |
| stairs/pyramid_stairs_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/up/L9 | 몸통 접촉 | 31.2 | 37.5 | 37.5 | 43.8 | 46.9 |
| stairs/pyramid_stairs_nose/up/L14 | 통과 | 81.2 | 81.2 | 81.2 | 81.2 | 78.1 |
| stairs/pyramid_stairs_nose/up/L14 | 넘어짐 | 3.1 | 3.1 | 6.2 | 6.2 | 9.4 |
| stairs/pyramid_stairs_nose/up/L14 | 몸통 접촉 | 71.9 | 81.2 | 75.0 | 65.6 | 71.9 |
| stairs/pyramid_stairs_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L9 | 몸통 접촉 | 87.5 | 75.0 | 71.9 | 78.1 | 71.9 |
| stairs/pyramid_stairs_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L14 | 몸통 접촉 | 96.9 | 96.9 | 100.0 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 몸통 접촉 | 65.6 | 59.4 | 59.4 | 56.2 | 56.2 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 몸통 접촉 | 53.1 | 50.0 | 56.2 | 43.8 | 53.1 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 통과 | 96.9 | 93.8 | 93.8 | 93.8 | 93.8 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 몸통 접촉 | 68.8 | 50.0 | 56.2 | 59.4 | 62.5 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 통과 | 96.9 | 96.9 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 몸통 접촉 | 81.2 | 75.0 | 71.9 | 90.6 | 71.9 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 95.7 | 94.7 | 95.4 | 95.7 | 95.6 |
| 평균 | 넘어짐 | 0.3 | 0.7 | 0.4 | 0.5 | 0.5 |
| 평균 | 몸통 접촉 | 55.5 | 53.9 | 52.1 | 53.1 | 52.2 |

### 좁은 통로

| 셀 | 지표 | hg999 | hg846 | hg927 | hg861 | hg519 |
|---|---|---|---|---|---|---|
| narrow/narrow_corridor/pass_wide (폭 0.80) | 통과 | 96.9 | 96.9 | 100.0 | 96.9 | 96.9 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.095 | 0.097 | 0.099 | 0.108 | 0.091 |
| narrow/narrow_corridor/pass_wide/back (폭 0.80) | 통과 | 96.9 | 93.8 | 93.8 | 90.6 | 93.8 |
| narrow/narrow_corridor/pass_wide/back (폭 0.80) | 무접촉 | 84.4 | 90.6 | 90.6 | 93.8 | 90.6 |
| narrow/narrow_corridor/pass_wide/back (폭 0.80) | 벽 거리 중앙값 m | 0.093 | 0.086 | 0.091 | 0.089 | 0.083 |
| narrow/narrow_corridor/pass_tight (폭 0.65) | 통과 | 100.0 | 93.8 | 100.0 | 96.9 | 100.0 |
| narrow/narrow_corridor/pass_tight (폭 0.65) | 무접촉 | 100.0 | 100.0 | 96.9 | 93.8 | 96.9 |
| narrow/narrow_corridor/pass_tight (폭 0.65) | 벽 거리 중앙값 m | 0.054 | 0.053 | 0.057 | 0.059 | 0.053 |
| narrow/narrow_corridor/pass_tight/back (폭 0.65) | 통과 | 96.9 | 100.0 | 100.0 | 96.9 | 96.9 |
| narrow/narrow_corridor/pass_tight/back (폭 0.65) | 무접촉 | 90.6 | 100.0 | 100.0 | 93.8 | 93.8 |
| narrow/narrow_corridor/pass_tight/back (폭 0.65) | 벽 거리 중앙값 m | 0.040 | 0.041 | 0.037 | 0.038 | 0.036 |
| narrow/narrow_corridor/block_margin (폭 0.65) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/block_margin (폭 0.65) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.81) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.81) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.81) | 벽 거리 중앙값 m | 0.100 | 0.102 | 0.105 | 0.106 | 0.101 |
| narrow/narrow_door_pos/pass_wide/back (폭 0.81) | 통과 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide/back (폭 0.81) | 무접촉 | 96.9 | 100.0 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_door_pos/pass_wide/back (폭 0.81) | 벽 거리 중앙값 m | 0.098 | 0.099 | 0.089 | 0.104 | 0.097 |
| narrow/narrow_door_pos/pass_tight (폭 0.65) | 통과 | 100.0 | 100.0 | 96.9 | 96.9 | 93.8 |
| narrow/narrow_door_pos/pass_tight (폭 0.65) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_door_pos/pass_tight (폭 0.65) | 벽 거리 중앙값 m | 0.053 | 0.048 | 0.051 | 0.054 | 0.050 |
| narrow/narrow_door_pos/pass_tight/back (폭 0.65) | 통과 | 93.8 | 93.8 | 90.6 | 93.8 | 90.6 |
| narrow/narrow_door_pos/pass_tight/back (폭 0.65) | 무접촉 | 87.5 | 81.2 | 84.4 | 84.4 | 78.1 |
| narrow/narrow_door_pos/pass_tight/back (폭 0.65) | 벽 거리 중앙값 m | 0.030 | 0.029 | 0.028 | 0.037 | 0.029 |
| narrow/narrow_door_pos/block_margin (폭 0.65) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/block_margin (폭 0.65) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_pos/block_narrow (폭 0.50) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/block_narrow (폭 0.50) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.79) | 통과 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.79) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.79) | 벽 거리 중앙값 m | 0.103 | 0.093 | 0.098 | 0.098 | 0.082 |
| narrow/narrow_door_neg/pass_wide/back (폭 0.79) | 통과 | 96.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide/back (폭 0.79) | 무접촉 | 100.0 | 100.0 | 93.8 | 93.8 | 90.6 |
| narrow/narrow_door_neg/pass_wide/back (폭 0.79) | 벽 거리 중앙값 m | 0.092 | 0.097 | 0.086 | 0.089 | 0.083 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 통과 | 93.8 | 96.9 | 96.9 | 96.9 | 84.4 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 벽 거리 중앙값 m | 0.053 | 0.060 | 0.059 | 0.056 | 0.051 |
| narrow/narrow_door_neg/pass_tight/back (폭 0.67) | 통과 | 96.9 | 93.8 | 93.8 | 90.6 | 90.6 |
| narrow/narrow_door_neg/pass_tight/back (폭 0.67) | 무접촉 | 81.2 | 78.1 | 81.2 | 84.4 | 84.4 |
| narrow/narrow_door_neg/pass_tight/back (폭 0.67) | 벽 거리 중앙값 m | 0.029 | 0.028 | 0.029 | 0.017 | 0.026 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.52) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.52) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.073 | 0.064 | 0.073 | 0.071 | 0.057 |
| narrow/narrow_bend_left/pass_wide/back (폭 0.80) | 통과 | 93.8 | 96.9 | 96.9 | 93.8 | 96.9 |
| narrow/narrow_bend_left/pass_wide/back (폭 0.80) | 무접촉 | 90.6 | 93.8 | 96.9 | 84.4 | 71.9 |
| narrow/narrow_bend_left/pass_wide/back (폭 0.80) | 벽 거리 중앙값 m | 0.023 | 0.026 | 0.025 | 0.015 | 0.006 |
| narrow/narrow_bend_left/pass_tight (폭 0.66) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/pass_tight (폭 0.66) | 무접촉 | 90.6 | 93.8 | 90.6 | 96.9 | 87.5 |
| narrow/narrow_bend_left/pass_tight (폭 0.66) | 벽 거리 중앙값 m | 0.029 | 0.028 | 0.028 | 0.034 | 0.021 |
| narrow/narrow_bend_left/pass_tight/back (폭 0.66) | 통과 | 84.4 | 87.5 | 84.4 | 84.4 | 87.5 |
| narrow/narrow_bend_left/pass_tight/back (폭 0.66) | 무접촉 | 56.2 | 56.2 | 43.8 | 50.0 | 34.4 |
| narrow/narrow_bend_left/pass_tight/back (폭 0.66) | 벽 거리 중앙값 m | 0.001 | 0.000 | -0.001 | -0.001 | -0.001 |
| narrow/narrow_bend_left/block_margin (폭 0.66) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/block_margin (폭 0.66) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_left/block_narrow (폭 0.51) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/block_narrow (폭 0.51) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.80) | 통과 | 96.9 | 96.9 | 96.9 | 96.9 | 96.9 |
| narrow/narrow_bend_right/pass_wide (폭 0.80) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.071 | 0.073 | 0.075 | 0.076 | 0.067 |
| narrow/narrow_bend_right/pass_wide/back (폭 0.80) | 통과 | 90.6 | 90.6 | 90.6 | 90.6 | 90.6 |
| narrow/narrow_bend_right/pass_wide/back (폭 0.80) | 무접촉 | 90.6 | 87.5 | 90.6 | 84.4 | 81.2 |
| narrow/narrow_bend_right/pass_wide/back (폭 0.80) | 벽 거리 중앙값 m | 0.027 | 0.028 | 0.033 | 0.021 | 0.009 |
| narrow/narrow_bend_right/pass_tight (폭 0.66) | 통과 | 96.9 | 96.9 | 100.0 | 96.9 | 96.9 |
| narrow/narrow_bend_right/pass_tight (폭 0.66) | 무접촉 | 87.5 | 90.6 | 87.5 | 93.8 | 81.2 |
| narrow/narrow_bend_right/pass_tight (폭 0.66) | 벽 거리 중앙값 m | 0.021 | 0.018 | 0.020 | 0.022 | 0.016 |
| narrow/narrow_bend_right/pass_tight/back (폭 0.66) | 통과 | 93.8 | 93.8 | 93.8 | 93.8 | 90.6 |
| narrow/narrow_bend_right/pass_tight/back (폭 0.66) | 무접촉 | 62.5 | 62.5 | 62.5 | 75.0 | 31.2 |
| narrow/narrow_bend_right/pass_tight/back (폭 0.66) | 벽 거리 중앙값 m | 0.005 | 0.003 | 0.003 | 0.005 | -0.002 |
| narrow/narrow_bend_right/block_margin (폭 0.66) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/block_margin (폭 0.66) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_right/block_narrow (폭 0.52) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/block_narrow (폭 0.52) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 96.2 | 96.6 | 96.6 | 95.8 | 95.3 |
| 평균 | 무접촉 | 90.9 | 91.7 | 90.9 | 91.4 | 85.6 |
| 평균 | 벽 거리 중앙값 m | 0.055 | 0.054 | 0.054 | 0.055 | 0.048 |
| 평균 | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 평균 | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### 막힌 통로

| 셀 | 지표 | hg999 | hg846 | hg927 | hg861 | hg519 |
|---|---|---|---|---|---|---|
| dead_end/narrow_dead_end/L7 | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| dead_end/narrow_dead_end/L7 | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dead_end/narrow_dead_end/L14 | 서기 | 93.8 | 96.9 | 96.9 | 96.9 | 93.8 |
| dead_end/narrow_dead_end/L14 | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |
| 평균 | 서기 | 96.9 | 98.4 | 98.4 | 98.4 | 96.9 |
| 평균 | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## 조건 mid360_rear (라이다 mid360, 후방 Mid-360 True, 탑재물 5.0 kg, 밀기 False, 계단 외란 False)

### 좁은 통로

| 셀 | 지표 | hg999 | hg846 | hg927 | hg861 | hg519 |
|---|---|---|---|---|---|---|
| narrow/narrow_corridor/pass_wide (폭 0.80) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.096 | 0.101 | 0.098 | 0.099 | 0.095 |
| narrow/narrow_corridor/pass_wide/back (폭 0.80) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide/back (폭 0.80) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide/back (폭 0.80) | 벽 거리 중앙값 m | 0.074 | 0.078 | 0.078 | 0.085 | 0.086 |
| narrow/narrow_corridor/pass_tight (폭 0.65) | 통과 | 96.9 | 96.9 | 96.9 | 96.9 | 96.9 |
| narrow/narrow_corridor/pass_tight (폭 0.65) | 무접촉 | 100.0 | 100.0 | 100.0 | 96.9 | 96.9 |
| narrow/narrow_corridor/pass_tight (폭 0.65) | 벽 거리 중앙값 m | 0.056 | 0.053 | 0.054 | 0.052 | 0.048 |
| narrow/narrow_corridor/pass_tight/back (폭 0.65) | 통과 | 87.5 | 87.5 | 87.5 | 87.5 | 84.4 |
| narrow/narrow_corridor/pass_tight/back (폭 0.65) | 무접촉 | 90.6 | 93.8 | 96.9 | 96.9 | 93.8 |
| narrow/narrow_corridor/pass_tight/back (폭 0.65) | 벽 거리 중앙값 m | 0.030 | 0.035 | 0.032 | 0.036 | 0.036 |
| narrow/narrow_corridor/block_margin (폭 0.65) | 서기 | 96.9 | 96.9 | 96.9 | 96.9 | 96.9 |
| narrow/narrow_corridor/block_margin (폭 0.65) | 침범 | 3.1 | 3.1 | 3.1 | 3.1 | 3.1 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.81) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.81) | 무접촉 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.81) | 벽 거리 중앙값 m | 0.103 | 0.094 | 0.099 | 0.108 | 0.096 |
| narrow/narrow_door_pos/pass_wide/back (폭 0.81) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide/back (폭 0.81) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide/back (폭 0.81) | 벽 거리 중앙값 m | 0.103 | 0.099 | 0.094 | 0.101 | 0.095 |
| narrow/narrow_door_pos/pass_tight (폭 0.65) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_door_pos/pass_tight (폭 0.65) | 무접촉 | 100.0 | 96.9 | 96.9 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_tight (폭 0.65) | 벽 거리 중앙값 m | 0.046 | 0.046 | 0.041 | 0.047 | 0.042 |
| narrow/narrow_door_pos/pass_tight/back (폭 0.65) | 통과 | 96.9 | 100.0 | 96.9 | 93.8 | 100.0 |
| narrow/narrow_door_pos/pass_tight/back (폭 0.65) | 무접촉 | 84.4 | 87.5 | 87.5 | 81.2 | 90.6 |
| narrow/narrow_door_pos/pass_tight/back (폭 0.65) | 벽 거리 중앙값 m | 0.033 | 0.033 | 0.038 | 0.036 | 0.037 |
| narrow/narrow_door_pos/block_margin (폭 0.65) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/block_margin (폭 0.65) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_pos/block_narrow (폭 0.50) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/block_narrow (폭 0.50) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.79) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.79) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.79) | 벽 거리 중앙값 m | 0.105 | 0.105 | 0.105 | 0.107 | 0.100 |
| narrow/narrow_door_neg/pass_wide/back (폭 0.79) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide/back (폭 0.79) | 무접촉 | 96.9 | 96.9 | 100.0 | 96.9 | 100.0 |
| narrow/narrow_door_neg/pass_wide/back (폭 0.79) | 벽 거리 중앙값 m | 0.101 | 0.091 | 0.096 | 0.100 | 0.090 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 통과 | 96.9 | 96.9 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 무접촉 | 100.0 | 93.8 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 벽 거리 중앙값 m | 0.051 | 0.060 | 0.057 | 0.056 | 0.055 |
| narrow/narrow_door_neg/pass_tight/back (폭 0.67) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_door_neg/pass_tight/back (폭 0.67) | 무접촉 | 78.1 | 84.4 | 75.0 | 78.1 | 78.1 |
| narrow/narrow_door_neg/pass_tight/back (폭 0.67) | 벽 거리 중앙값 m | 0.025 | 0.029 | 0.029 | 0.029 | 0.027 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.52) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.52) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.068 | 0.067 | 0.071 | 0.074 | 0.063 |
| narrow/narrow_bend_left/pass_wide/back (폭 0.80) | 통과 | 93.8 | 93.8 | 93.8 | 90.6 | 93.8 |
| narrow/narrow_bend_left/pass_wide/back (폭 0.80) | 무접촉 | 93.8 | 78.1 | 93.8 | 87.5 | 71.9 |
| narrow/narrow_bend_left/pass_wide/back (폭 0.80) | 벽 거리 중앙값 m | 0.025 | 0.023 | 0.024 | 0.013 | 0.006 |
| narrow/narrow_bend_left/pass_tight (폭 0.66) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_bend_left/pass_tight (폭 0.66) | 무접촉 | 96.9 | 100.0 | 90.6 | 96.9 | 93.8 |
| narrow/narrow_bend_left/pass_tight (폭 0.66) | 벽 거리 중앙값 m | 0.021 | 0.028 | 0.028 | 0.027 | 0.022 |
| narrow/narrow_bend_left/pass_tight/back (폭 0.66) | 통과 | 100.0 | 100.0 | 100.0 | 96.9 | 96.9 |
| narrow/narrow_bend_left/pass_tight/back (폭 0.66) | 무접촉 | 50.0 | 53.1 | 62.5 | 62.5 | 25.0 |
| narrow/narrow_bend_left/pass_tight/back (폭 0.66) | 벽 거리 중앙값 m | -0.002 | -0.000 | 0.005 | 0.003 | -0.004 |
| narrow/narrow_bend_left/block_margin (폭 0.66) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/block_margin (폭 0.66) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_left/block_narrow (폭 0.51) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/block_narrow (폭 0.51) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.80) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.80) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.073 | 0.074 | 0.075 | 0.083 | 0.070 |
| narrow/narrow_bend_right/pass_wide/back (폭 0.80) | 통과 | 90.6 | 90.6 | 90.6 | 90.6 | 90.6 |
| narrow/narrow_bend_right/pass_wide/back (폭 0.80) | 무접촉 | 93.8 | 100.0 | 93.8 | 96.9 | 75.0 |
| narrow/narrow_bend_right/pass_wide/back (폭 0.80) | 벽 거리 중앙값 m | 0.028 | 0.025 | 0.028 | 0.022 | 0.010 |
| narrow/narrow_bend_right/pass_tight (폭 0.66) | 통과 | 93.8 | 93.8 | 93.8 | 93.8 | 93.8 |
| narrow/narrow_bend_right/pass_tight (폭 0.66) | 무접촉 | 93.8 | 90.6 | 84.4 | 90.6 | 90.6 |
| narrow/narrow_bend_right/pass_tight (폭 0.66) | 벽 거리 중앙값 m | 0.021 | 0.020 | 0.015 | 0.018 | 0.013 |
| narrow/narrow_bend_right/pass_tight/back (폭 0.66) | 통과 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 |
| narrow/narrow_bend_right/pass_tight/back (폭 0.66) | 무접촉 | 78.1 | 71.9 | 59.4 | 56.2 | 31.2 |
| narrow/narrow_bend_right/pass_tight/back (폭 0.66) | 벽 거리 중앙값 m | 0.004 | 0.004 | 0.006 | 0.000 | -0.001 |
| narrow/narrow_bend_right/block_margin (폭 0.66) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/block_margin (폭 0.66) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_right/block_narrow (폭 0.52) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/block_narrow (폭 0.52) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 97.8 | 98.0 | 97.8 | 97.5 | 97.3 |
| 평균 | 무접촉 | 92.8 | 92.3 | 92.0 | 91.9 | 87.2 |
| 평균 | 벽 거리 중앙값 m | 0.053 | 0.053 | 0.054 | 0.055 | 0.049 |
| 평균 | 서기 | 99.7 | 99.7 | 99.7 | 99.7 | 99.7 |
| 평균 | 침범 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |

### 막힌 통로

| 셀 | 지표 | hg999 | hg846 | hg927 | hg861 | hg519 |
|---|---|---|---|---|---|---|
| dead_end/narrow_dead_end/L7 | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| dead_end/narrow_dead_end/L7 | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dead_end/narrow_dead_end/L14 | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| dead_end/narrow_dead_end/L14 | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |
| 평균 | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 평균 | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## 조건 mid360_only (라이다 mid360, 후방 Mid-360 False, 탑재물 5.0 kg, 밀기 False, 계단 외란 False)

### 좁은 통로

| 셀 | 지표 | hg999 | hg846 | hg927 | hg861 | hg519 |
|---|---|---|---|---|---|---|
| narrow/narrow_corridor/pass_wide (폭 0.80) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.094 | 0.098 | 0.096 | 0.105 | 0.087 |
| narrow/narrow_corridor/pass_wide/back (폭 0.80) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide/back (폭 0.80) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/pass_wide/back (폭 0.80) | 벽 거리 중앙값 m | 0.078 | 0.082 | 0.084 | 0.079 | 0.087 |
| narrow/narrow_corridor/pass_tight (폭 0.65) | 통과 | 96.9 | 93.8 | 96.9 | 96.9 | 93.8 |
| narrow/narrow_corridor/pass_tight (폭 0.65) | 무접촉 | 96.9 | 100.0 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_corridor/pass_tight (폭 0.65) | 벽 거리 중앙값 m | 0.055 | 0.055 | 0.056 | 0.053 | 0.052 |
| narrow/narrow_corridor/pass_tight/back (폭 0.65) | 통과 | 87.5 | 87.5 | 87.5 | 87.5 | 87.5 |
| narrow/narrow_corridor/pass_tight/back (폭 0.65) | 무접촉 | 96.9 | 87.5 | 90.6 | 90.6 | 87.5 |
| narrow/narrow_corridor/pass_tight/back (폭 0.65) | 벽 거리 중앙값 m | 0.032 | 0.036 | 0.035 | 0.029 | 0.034 |
| narrow/narrow_corridor/block_margin (폭 0.65) | 서기 | 96.9 | 96.9 | 96.9 | 96.9 | 96.9 |
| narrow/narrow_corridor/block_margin (폭 0.65) | 침범 | 3.1 | 3.1 | 3.1 | 3.1 | 3.1 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_corridor/block_narrow (폭 0.52) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.81) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.81) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide (폭 0.81) | 벽 거리 중앙값 m | 0.095 | 0.104 | 0.099 | 0.096 | 0.098 |
| narrow/narrow_door_pos/pass_wide/back (폭 0.81) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/pass_wide/back (폭 0.81) | 무접촉 | 100.0 | 96.9 | 100.0 | 96.9 | 96.9 |
| narrow/narrow_door_pos/pass_wide/back (폭 0.81) | 벽 거리 중앙값 m | 0.091 | 0.095 | 0.096 | 0.105 | 0.099 |
| narrow/narrow_door_pos/pass_tight (폭 0.65) | 통과 | 100.0 | 100.0 | 96.9 | 100.0 | 93.8 |
| narrow/narrow_door_pos/pass_tight (폭 0.65) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_door_pos/pass_tight (폭 0.65) | 벽 거리 중앙값 m | 0.045 | 0.042 | 0.042 | 0.049 | 0.039 |
| narrow/narrow_door_pos/pass_tight/back (폭 0.65) | 통과 | 93.8 | 96.9 | 96.9 | 93.8 | 96.9 |
| narrow/narrow_door_pos/pass_tight/back (폭 0.65) | 무접촉 | 81.2 | 81.2 | 90.6 | 81.2 | 81.2 |
| narrow/narrow_door_pos/pass_tight/back (폭 0.65) | 벽 거리 중앙값 m | 0.028 | 0.032 | 0.036 | 0.040 | 0.030 |
| narrow/narrow_door_pos/block_margin (폭 0.65) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/block_margin (폭 0.65) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_pos/block_narrow (폭 0.50) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_pos/block_narrow (폭 0.50) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.79) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.79) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide (폭 0.79) | 벽 거리 중앙값 m | 0.099 | 0.097 | 0.103 | 0.102 | 0.100 |
| narrow/narrow_door_neg/pass_wide/back (폭 0.79) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_wide/back (폭 0.79) | 무접촉 | 96.9 | 96.9 | 100.0 | 100.0 | 96.9 |
| narrow/narrow_door_neg/pass_wide/back (폭 0.79) | 벽 거리 중앙값 m | 0.096 | 0.092 | 0.100 | 0.095 | 0.085 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 통과 | 93.8 | 96.9 | 100.0 | 96.9 | 96.9 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 무접촉 | 100.0 | 100.0 | 93.8 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_tight (폭 0.67) | 벽 거리 중앙값 m | 0.059 | 0.058 | 0.053 | 0.063 | 0.051 |
| narrow/narrow_door_neg/pass_tight/back (폭 0.67) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/pass_tight/back (폭 0.67) | 무접촉 | 78.1 | 81.2 | 75.0 | 62.5 | 78.1 |
| narrow/narrow_door_neg/pass_tight/back (폭 0.67) | 벽 거리 중앙값 m | 0.026 | 0.032 | 0.015 | 0.031 | 0.019 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/block_margin (폭 0.67) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.52) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_door_neg/block_narrow (폭 0.52) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.074 | 0.067 | 0.069 | 0.071 | 0.059 |
| narrow/narrow_bend_left/pass_wide/back (폭 0.80) | 통과 | 90.6 | 90.6 | 90.6 | 87.5 | 93.8 |
| narrow/narrow_bend_left/pass_wide/back (폭 0.80) | 무접촉 | 84.4 | 87.5 | 87.5 | 78.1 | 75.0 |
| narrow/narrow_bend_left/pass_wide/back (폭 0.80) | 벽 거리 중앙값 m | 0.017 | 0.017 | 0.009 | 0.010 | 0.005 |
| narrow/narrow_bend_left/pass_tight (폭 0.66) | 통과 | 96.9 | 100.0 | 100.0 | 100.0 | 93.8 |
| narrow/narrow_bend_left/pass_tight (폭 0.66) | 무접촉 | 96.9 | 87.5 | 93.8 | 96.9 | 75.0 |
| narrow/narrow_bend_left/pass_tight (폭 0.66) | 벽 거리 중앙값 m | 0.021 | 0.020 | 0.020 | 0.022 | 0.022 |
| narrow/narrow_bend_left/pass_tight/back (폭 0.66) | 통과 | 93.8 | 93.8 | 100.0 | 93.8 | 93.8 |
| narrow/narrow_bend_left/pass_tight/back (폭 0.66) | 무접촉 | 50.0 | 37.5 | 53.1 | 46.9 | 25.0 |
| narrow/narrow_bend_left/pass_tight/back (폭 0.66) | 벽 거리 중앙값 m | 0.001 | -0.002 | 0.002 | 0.001 | -0.002 |
| narrow/narrow_bend_left/block_margin (폭 0.66) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/block_margin (폭 0.66) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_left/block_narrow (폭 0.51) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_left/block_narrow (폭 0.51) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.80) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.80) | 무접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/pass_wide (폭 0.80) | 벽 거리 중앙값 m | 0.074 | 0.074 | 0.066 | 0.076 | 0.066 |
| narrow/narrow_bend_right/pass_wide/back (폭 0.80) | 통과 | 90.6 | 90.6 | 90.6 | 90.6 | 90.6 |
| narrow/narrow_bend_right/pass_wide/back (폭 0.80) | 무접촉 | 93.8 | 87.5 | 93.8 | 84.4 | 75.0 |
| narrow/narrow_bend_right/pass_wide/back (폭 0.80) | 벽 거리 중앙값 m | 0.010 | 0.014 | 0.024 | 0.010 | 0.007 |
| narrow/narrow_bend_right/pass_tight (폭 0.66) | 통과 | 93.8 | 93.8 | 93.8 | 93.8 | 90.6 |
| narrow/narrow_bend_right/pass_tight (폭 0.66) | 무접촉 | 87.5 | 93.8 | 75.0 | 87.5 | 81.2 |
| narrow/narrow_bend_right/pass_tight (폭 0.66) | 벽 거리 중앙값 m | 0.011 | 0.017 | 0.012 | 0.015 | 0.012 |
| narrow/narrow_bend_right/pass_tight/back (폭 0.66) | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/pass_tight/back (폭 0.66) | 무접촉 | 46.9 | 59.4 | 50.0 | 50.0 | 34.4 |
| narrow/narrow_bend_right/pass_tight/back (폭 0.66) | 벽 거리 중앙값 m | -0.000 | 0.004 | 0.002 | 0.001 | -0.001 |
| narrow/narrow_bend_right/block_margin (폭 0.66) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/block_margin (폭 0.66) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| narrow/narrow_bend_right/block_narrow (폭 0.52) | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| narrow/narrow_bend_right/block_narrow (폭 0.52) | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 96.9 | 97.2 | 97.7 | 97.0 | 96.6 |
| 평균 | 무접촉 | 90.5 | 89.8 | 90.2 | 88.8 | 85.0 |
| 평균 | 벽 거리 중앙값 m | 0.050 | 0.052 | 0.051 | 0.053 | 0.047 |
| 평균 | 서기 | 99.7 | 99.7 | 99.7 | 99.7 | 99.7 |
| 평균 | 침범 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |

### 막힌 통로

| 셀 | 지표 | hg999 | hg846 | hg927 | hg861 | hg519 |
|---|---|---|---|---|---|---|
| dead_end/narrow_dead_end/L7 | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| dead_end/narrow_dead_end/L7 | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dead_end/narrow_dead_end/L14 | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| dead_end/narrow_dead_end/L14 | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **평균** | |  |  |  |  |  |
| 평균 | 서기 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 평균 | 침범 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## 조건 lidar_off (라이다 off, 후방 Mid-360 False, 탑재물 5.0 kg, 밀기 False, 계단 외란 False)

### 갭

| 셀 | 지표 | hg999 | hg846 | hg927 | hg861 | hg519 |
|---|---|---|---|---|---|---|
| gap/forward/L7/v0.3 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.3 | clean/시도 | 89.6 | 70.8 | 72.9 | 85.4 | 77.1 |
| gap/forward/L7/v0.3 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 머리 들림 최대 평균 | 5.0° | 5.8° | 5.9° | 5.4° | 5.5° |
| gap/forward/L7/v0.3 | 머리 들림 최대 p90 | 8.7° | 9.9° | 8.8° | 8.0° | 8.9° |
| gap/forward/L7/v0.3 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.3 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L7/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.5 | clean/시도 | 72.9 | 81.2 | 77.1 | 77.1 | 77.1 |
| gap/forward/L7/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 머리 들림 최대 평균 | 4.3° | 4.8° | 4.9° | 4.5° | 4.6° |
| gap/forward/L7/v0.5 | 머리 들림 최대 p90 | 8.7° | 8.1° | 8.1° | 8.4° | 8.7° |
| gap/forward/L7/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L7/v0.9 | 건넘 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/forward/L7/v0.9 | clean/시도 | 87.5 | 85.4 | 89.6 | 85.4 | 87.5 |
| gap/forward/L7/v0.9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 머리 들림 최대 평균 | 4.6° | 5.0° | 5.2° | 4.6° | 4.2° |
| gap/forward/L7/v0.9 | 머리 들림 최대 p90 | 7.9° | 8.6° | 8.8° | 8.1° | 7.4° |
| gap/forward/L7/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L7/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L11/v0.3 | 건넘 | 100.0 | 97.9 | 97.9 | 97.9 | 97.9 |
| gap/forward/L11/v0.3 | clean/시도 | 79.2 | 74.5 | 72.3 | 74.5 | 66.0 |
| gap/forward/L11/v0.3 | 넘어짐 | 0.0 | 2.1 | 2.1 | 2.1 | 2.1 |
| gap/forward/L11/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.3 | 머리 들림 최대 평균 | 6.5° | 7.1° | 7.5° | 7.1° | 7.7° |
| gap/forward/L11/v0.3 | 머리 들림 최대 p90 | 11.4° | 11.3° | 11.6° | 11.4° | 13.4° |
| gap/forward/L11/v0.3 | 20° 초과 | 0.0 | 0.0 | 2.1 | 2.1 | 4.2 |
| gap/forward/L11/v0.3 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.01 s | 0.03 s |
| gap/forward/L11/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 97.9 | 100.0 |
| gap/forward/L11/v0.5 | clean/시도 | 89.6 | 79.2 | 77.1 | 76.6 | 68.8 |
| gap/forward/L11/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 2.1 | 0.0 |
| gap/forward/L11/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.5 | 머리 들림 최대 평균 | 5.1° | 5.3° | 6.1° | 6.0° | 5.0° |
| gap/forward/L11/v0.5 | 머리 들림 최대 p90 | 10.2° | 10.6° | 11.3° | 11.4° | 9.3° |
| gap/forward/L11/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 2.1 | 0.0 |
| gap/forward/L11/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.01 s | 0.00 s |
| gap/forward/L11/v0.9 | 건넘 | 100.0 | 100.0 | 97.9 | 100.0 | 100.0 |
| gap/forward/L11/v0.9 | clean/시도 | 83.3 | 79.2 | 80.9 | 79.2 | 85.4 |
| gap/forward/L11/v0.9 | 넘어짐 | 0.0 | 0.0 | 2.1 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 머리 들림 최대 평균 | 5.2° | 5.4° | 5.5° | 4.9° | 5.0° |
| gap/forward/L11/v0.9 | 머리 들림 최대 p90 | 10.6° | 10.0° | 9.8° | 9.6° | 10.3° |
| gap/forward/L11/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L11/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L14/v0.3 | 건넘 | 95.8 | 100.0 | 97.9 | 95.8 | 97.9 |
| gap/forward/L14/v0.3 | clean/시도 | 69.6 | 77.1 | 78.7 | 78.3 | 72.3 |
| gap/forward/L14/v0.3 | 넘어짐 | 4.2 | 0.0 | 2.1 | 4.2 | 2.1 |
| gap/forward/L14/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.3 | 머리 들림 최대 평균 | 7.1° | 7.5° | 8.5° | 7.0° | 7.0° |
| gap/forward/L14/v0.3 | 머리 들림 최대 p90 | 12.9° | 11.5° | 13.1° | 11.9° | 12.1° |
| gap/forward/L14/v0.3 | 20° 초과 | 0.0 | 0.0 | 2.1 | 0.0 | 0.0 |
| gap/forward/L14/v0.3 | 15° 초과 시간 | 0.01 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/forward/L14/v0.5 | 건넘 | 97.9 | 95.8 | 100.0 | 100.0 | 100.0 |
| gap/forward/L14/v0.5 | clean/시도 | 85.1 | 89.1 | 81.2 | 89.6 | 87.5 |
| gap/forward/L14/v0.5 | 넘어짐 | 2.1 | 4.2 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.5 | 머리 들림 최대 평균 | 6.6° | 6.1° | 7.0° | 6.7° | 6.4° |
| gap/forward/L14/v0.5 | 머리 들림 최대 p90 | 11.0° | 9.8° | 11.9° | 10.8° | 11.1° |
| gap/forward/L14/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 2.1 | 0.0 |
| gap/forward/L14/v0.5 | 15° 초과 시간 | 0.01 s | 0.00 s | 0.00 s | 0.01 s | 0.00 s |
| gap/forward/L14/v0.9 | 건넘 | 100.0 | 97.9 | 97.9 | 97.9 | 97.9 |
| gap/forward/L14/v0.9 | clean/시도 | 83.3 | 78.7 | 85.1 | 83.0 | 87.2 |
| gap/forward/L14/v0.9 | 넘어짐 | 0.0 | 2.1 | 2.1 | 2.1 | 2.1 |
| gap/forward/L14/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.9 | 머리 들림 최대 평균 | 6.6° | 6.3° | 6.5° | 6.0° | 6.2° |
| gap/forward/L14/v0.9 | 머리 들림 최대 p90 | 11.1° | 10.5° | 10.1° | 9.4° | 9.6° |
| gap/forward/L14/v0.9 | 20° 초과 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/forward/L14/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L7/v0.3 | 건넘 | 93.8 | 85.4 | 95.8 | 95.8 | 95.8 |
| gap/backward/L7/v0.3 | clean/시도 | 68.9 | 78.0 | 67.4 | 76.1 | 65.2 |
| gap/backward/L7/v0.3 | 넘어짐 | 6.2 | 14.6 | 4.2 | 4.2 | 4.2 |
| gap/backward/L7/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.3 | 머리 들림 최대 평균 | 8.7° | 10.7° | 8.1° | 8.2° | 6.8° |
| gap/backward/L7/v0.3 | 머리 들림 최대 p90 | 10.1° | 32.5° | 10.5° | 9.8° | 8.2° |
| gap/backward/L7/v0.3 | 20° 초과 | 8.3 | 14.6 | 4.2 | 6.2 | 4.2 |
| gap/backward/L7/v0.3 | 15° 초과 시간 | 0.02 s | 0.05 s | 0.00 s | 0.06 s | 0.00 s |
| gap/backward/L7/v0.5 | 건넘 | 100.0 | 100.0 | 100.0 | 97.9 | 97.9 |
| gap/backward/L7/v0.5 | clean/시도 | 85.4 | 89.6 | 83.3 | 87.2 | 91.5 |
| gap/backward/L7/v0.5 | 넘어짐 | 0.0 | 0.0 | 0.0 | 2.1 | 2.1 |
| gap/backward/L7/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.5 | 머리 들림 최대 평균 | 6.3° | 6.3° | 6.5° | 6.9° | 6.1° |
| gap/backward/L7/v0.5 | 머리 들림 최대 p90 | 9.3° | 8.8° | 9.0° | 10.2° | 9.3° |
| gap/backward/L7/v0.5 | 20° 초과 | 0.0 | 0.0 | 0.0 | 2.1 | 2.1 |
| gap/backward/L7/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.18 s |
| gap/backward/L7/v0.9 | 건넘 | 100.0 | 97.9 | 100.0 | 100.0 | 100.0 |
| gap/backward/L7/v0.9 | clean/시도 | 79.2 | 83.0 | 75.0 | 79.2 | 81.2 |
| gap/backward/L7/v0.9 | 넘어짐 | 0.0 | 2.1 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 머리 들림 최대 평균 | 6.1° | 6.7° | 6.3° | 6.2° | 5.6° |
| gap/backward/L7/v0.9 | 머리 들림 최대 p90 | 10.6° | 10.2° | 10.3° | 9.7° | 8.8° |
| gap/backward/L7/v0.9 | 20° 초과 | 0.0 | 2.1 | 0.0 | 0.0 | 0.0 |
| gap/backward/L7/v0.9 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.3 | 건넘 | 93.8 | 91.7 | 87.5 | 87.5 | 91.7 |
| gap/backward/L11/v0.3 | clean/시도 | 75.6 | 77.3 | 73.8 | 69.0 | 72.7 |
| gap/backward/L11/v0.3 | 넘어짐 | 6.2 | 8.3 | 12.5 | 12.5 | 8.3 |
| gap/backward/L11/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.3 | 머리 들림 최대 평균 | 10.4° | 10.9° | 11.4° | 11.0° | 10.2° |
| gap/backward/L11/v0.3 | 머리 들림 최대 p90 | 14.5° | 16.3° | 31.2° | 31.1° | 14.9° |
| gap/backward/L11/v0.3 | 20° 초과 | 6.2 | 8.3 | 12.5 | 10.4 | 8.3 |
| gap/backward/L11/v0.3 | 15° 초과 시간 | 0.01 s | 0.01 s | 0.01 s | 0.01 s | 0.03 s |
| gap/backward/L11/v0.5 | 건넘 | 97.9 | 97.9 | 93.8 | 97.9 | 100.0 |
| gap/backward/L11/v0.5 | clean/시도 | 80.9 | 83.0 | 80.0 | 83.0 | 85.4 |
| gap/backward/L11/v0.5 | 넘어짐 | 2.1 | 2.1 | 6.2 | 2.1 | 0.0 |
| gap/backward/L11/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.5 | 머리 들림 최대 평균 | 8.8° | 8.6° | 9.5° | 8.6° | 7.1° |
| gap/backward/L11/v0.5 | 머리 들림 최대 p90 | 12.7° | 14.1° | 13.2° | 12.0° | 11.5° |
| gap/backward/L11/v0.5 | 20° 초과 | 2.1 | 2.1 | 6.2 | 2.1 | 0.0 |
| gap/backward/L11/v0.5 | 15° 초과 시간 | 0.00 s | 0.00 s | 0.01 s | 0.00 s | 0.00 s |
| gap/backward/L11/v0.9 | 건넘 | 97.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| gap/backward/L11/v0.9 | clean/시도 | 76.6 | 75.0 | 79.2 | 72.9 | 68.8 |
| gap/backward/L11/v0.9 | 넘어짐 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 머리 들림 최대 평균 | 7.3° | 6.8° | 7.2° | 7.4° | 6.1° |
| gap/backward/L11/v0.9 | 머리 들림 최대 p90 | 10.0° | 9.7° | 10.5° | 10.5° | 9.0° |
| gap/backward/L11/v0.9 | 20° 초과 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L11/v0.9 | 15° 초과 시간 | 0.01 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s |
| gap/backward/L14/v0.3 | 건넘 | 85.4 | 87.5 | 81.2 | 77.1 | 97.9 |
| gap/backward/L14/v0.3 | clean/시도 | 56.1 | 64.3 | 48.7 | 45.9 | 53.2 |
| gap/backward/L14/v0.3 | 넘어짐 | 14.6 | 12.5 | 18.8 | 22.9 | 2.1 |
| gap/backward/L14/v0.3 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.3 | 머리 들림 최대 평균 | 12.6° | 12.3° | 13.3° | 15.6° | 9.0° |
| gap/backward/L14/v0.3 | 머리 들림 최대 p90 | 31.0° | 31.7° | 34.8° | 35.9° | 12.0° |
| gap/backward/L14/v0.3 | 20° 초과 | 12.5 | 12.5 | 16.7 | 22.9 | 2.1 |
| gap/backward/L14/v0.3 | 15° 초과 시간 | 0.02 s | 0.03 s | 0.04 s | 0.03 s | 0.00 s |
| gap/backward/L14/v0.5 | 건넘 | 87.5 | 91.7 | 93.8 | 91.7 | 95.8 |
| gap/backward/L14/v0.5 | clean/시도 | 57.1 | 63.6 | 71.1 | 75.0 | 87.0 |
| gap/backward/L14/v0.5 | 넘어짐 | 12.5 | 8.3 | 6.2 | 8.3 | 4.2 |
| gap/backward/L14/v0.5 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.5 | 머리 들림 최대 평균 | 11.1° | 9.7° | 10.0° | 9.8° | 8.3° |
| gap/backward/L14/v0.5 | 머리 들림 최대 p90 | 35.9° | 14.5° | 16.3° | 16.9° | 12.4° |
| gap/backward/L14/v0.5 | 20° 초과 | 12.5 | 6.2 | 6.2 | 8.3 | 4.2 |
| gap/backward/L14/v0.5 | 15° 초과 시간 | 0.02 s | 0.01 s | 0.01 s | 0.02 s | 0.01 s |
| gap/backward/L14/v0.9 | 건넘 | 97.9 | 91.7 | 93.8 | 97.9 | 97.9 |
| gap/backward/L14/v0.9 | clean/시도 | 66.0 | 61.4 | 57.8 | 57.4 | 55.3 |
| gap/backward/L14/v0.9 | 넘어짐 | 2.1 | 8.3 | 6.2 | 2.1 | 2.1 |
| gap/backward/L14/v0.9 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gap/backward/L14/v0.9 | 머리 들림 최대 평균 | 7.8° | 10.0° | 9.4° | 8.1° | 7.7° |
| gap/backward/L14/v0.9 | 머리 들림 최대 p90 | 10.4° | 11.3° | 12.2° | 11.2° | 10.3° |
| gap/backward/L14/v0.9 | 20° 초과 | 2.1 | 8.3 | 6.2 | 2.1 | 2.1 |
| gap/backward/L14/v0.9 | 15° 초과 시간 | 0.00 s | 0.01 s | 0.01 s | 0.00 s | 0.00 s |
| **평균** | |  |  |  |  |  |
| 평균 | 건넘 | 97.1 | 96.4 | 96.5 | 96.4 | 98.4 |
| 평균 | clean/시도 | 77.0 | 77.2 | 75.1 | 76.4 | 76.1 |
| 평균 | 넘어짐 | 2.9 | 3.6 | 3.5 | 3.6 | 1.6 |
| 평균 | 미도달(정지) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 평균 | 머리 들림 최대 평균 | 7.2° | 7.5° | 7.7° | 7.4° | 6.6° |
| 평균 | 머리 들림 최대 p90 | 13.2° | 13.3° | 13.4° | 13.1° | 10.4° |
| 평균 | 20° 초과 | 2.5 | 3.0 | 3.1 | 3.4 | 1.5 |
| 평균 | 15° 초과 시간 | 0.01 s | 0.01 s | 0.01 s | 0.01 s | 0.01 s |

### 계단

| 셀 | 지표 | hg999 | hg846 | hg927 | hg861 | hg519 |
|---|---|---|---|---|---|---|
| stairs/pyramid_stairs/up/L9 | 통과 | 96.9 | 93.8 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L9 | 몸통 접촉 | 15.6 | 15.6 | 12.5 | 12.5 | 12.5 |
| stairs/pyramid_stairs/up/L14 | 통과 | 93.8 | 93.8 | 93.8 | 93.8 | 93.8 |
| stairs/pyramid_stairs/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/up/L14 | 몸통 접촉 | 15.6 | 12.5 | 12.5 | 15.6 | 6.2 |
| stairs/pyramid_stairs/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L9 | 몸통 접촉 | 43.8 | 56.2 | 40.6 | 59.4 | 53.1 |
| stairs/pyramid_stairs/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs/down/L14 | 몸통 접촉 | 90.6 | 78.1 | 90.6 | 78.1 | 78.1 |
| stairs/pyramid_stairs_inv/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L9 | 몸통 접촉 | 25.0 | 21.9 | 25.0 | 25.0 | 28.1 |
| stairs/pyramid_stairs_inv/up/L14 | 통과 | 96.9 | 100.0 | 96.9 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv/up/L14 | 넘어짐 | 3.1 | 0.0 | 3.1 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/up/L14 | 몸통 접촉 | 46.9 | 50.0 | 43.8 | 43.8 | 43.8 |
| stairs/pyramid_stairs_inv/down/L9 | 통과 | 90.6 | 90.6 | 87.5 | 90.6 | 87.5 |
| stairs/pyramid_stairs_inv/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L9 | 몸통 접촉 | 34.4 | 37.5 | 50.0 | 43.8 | 50.0 |
| stairs/pyramid_stairs_inv/down/L14 | 통과 | 96.9 | 96.9 | 96.9 | 93.8 | 96.9 |
| stairs/pyramid_stairs_inv/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv/down/L14 | 몸통 접촉 | 75.0 | 75.0 | 75.0 | 81.2 | 65.6 |
| stairs/pyramid_stairs_wide/up/L9 | 통과 | 93.8 | 93.8 | 93.8 | 93.8 | 93.8 |
| stairs/pyramid_stairs_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/up/L9 | 몸통 접촉 | 9.4 | 9.4 | 3.1 | 6.2 | 3.1 |
| stairs/pyramid_stairs_wide/up/L14 | 통과 | 81.2 | 78.1 | 78.1 | 78.1 | 78.1 |
| stairs/pyramid_stairs_wide/up/L14 | 넘어짐 | 0.0 | 3.1 | 3.1 | 3.1 | 3.1 |
| stairs/pyramid_stairs_wide/up/L14 | 몸통 접촉 | 28.1 | 34.4 | 21.9 | 40.6 | 31.2 |
| stairs/pyramid_stairs_wide/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L9 | 몸통 접촉 | 78.1 | 65.6 | 65.6 | 71.9 | 68.8 |
| stairs/pyramid_stairs_wide/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_wide/down/L14 | 몸통 접촉 | 96.9 | 96.9 | 100.0 | 96.9 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/up/L9 | 몸통 접촉 | 25.0 | 28.1 | 34.4 | 34.4 | 34.4 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 통과 | 100.0 | 93.8 | 96.9 | 96.9 | 93.8 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 넘어짐 | 0.0 | 6.2 | 3.1 | 3.1 | 6.2 |
| stairs/pyramid_stairs_inv_wide/up/L14 | 몸통 접촉 | 40.6 | 34.4 | 34.4 | 34.4 | 37.5 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 통과 | 96.9 | 93.8 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L9 | 몸통 접촉 | 43.8 | 53.1 | 40.6 | 40.6 | 34.4 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 통과 | 96.9 | 93.8 | 93.8 | 93.8 | 90.6 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_wide/down/L14 | 몸통 접촉 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_nose/up/L9 | 통과 | 96.9 | 93.8 | 96.9 | 96.9 | 96.9 |
| stairs/pyramid_stairs_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/up/L9 | 몸통 접촉 | 40.6 | 34.4 | 31.2 | 28.1 | 25.0 |
| stairs/pyramid_stairs_nose/up/L14 | 통과 | 93.8 | 84.4 | 93.8 | 90.6 | 90.6 |
| stairs/pyramid_stairs_nose/up/L14 | 넘어짐 | 6.2 | 12.5 | 6.2 | 6.2 | 9.4 |
| stairs/pyramid_stairs_nose/up/L14 | 몸통 접촉 | 65.6 | 65.6 | 59.4 | 56.2 | 78.1 |
| stairs/pyramid_stairs_nose/down/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L9 | 몸통 접촉 | 62.5 | 68.8 | 71.9 | 71.9 | 59.4 |
| stairs/pyramid_stairs_nose/down/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_nose/down/L14 | 몸통 접촉 | 96.9 | 100.0 | 100.0 | 100.0 | 96.9 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L9 | 몸통 접촉 | 62.5 | 62.5 | 59.4 | 59.4 | 56.2 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 통과 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/up/L14 | 몸통 접촉 | 75.0 | 68.8 | 65.6 | 78.1 | 78.1 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 통과 | 96.9 | 100.0 | 93.8 | 96.9 | 90.6 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L9 | 몸통 접촉 | 59.4 | 46.9 | 68.8 | 50.0 | 53.1 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 통과 | 93.8 | 96.9 | 93.8 | 90.6 | 90.6 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 넘어짐 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stairs/pyramid_stairs_inv_nose/down/L14 | 몸통 접촉 | 84.4 | 71.9 | 78.1 | 68.8 | 81.2 |
| **평균** | |  |  |  |  |  |
| 평균 | 통과 | 96.9 | 96.0 | 96.2 | 96.2 | 95.7 |
| 평균 | 넘어짐 | 0.4 | 0.9 | 0.7 | 0.5 | 0.8 |
| 평균 | 몸통 접촉 | 54.8 | 53.6 | 53.5 | 54.0 | 53.0 |

## 기준 대비 하락 (⚠)

- 없음
