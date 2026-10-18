# gd_lab.gast_nav — GAST-Nav v1

좁은 통로 통과, 시계방향 제자리 회전, 계단 정렬을 학습하는 교사(백본: Ray+IR 3906)의 학습 코드다.
설계: `docs/design/GAST_NAV_V1_DESIGN.md` (v0.3.1, 부록 C 구현 계약). 공통 형상·코드: `schema.py`.

| 디렉토리 | 내용 |
|---|---|
| `src/gd_lab/gast_nav/` | 지도(지지면, 장애물 지도, 근거리 clearance, warp, OBS1), 정책(migration, adapter, trainable, PPO), 명령·FSM·감독기 |
| `scripts/gast_nav/` | 학습·평가 실행기 (pilot 이후) |
| `tests/gast_nav/` | CPU 단위 시험, `golden/` 기준 입출력 |
| `configs/gast_nav/` | 수치 설정 (trainable manifest, 감독기·FSM 설정) |

상태: 구현 중 (UPDATE159). 학습은 아직 하지 않는다.
