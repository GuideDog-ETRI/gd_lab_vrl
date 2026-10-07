# RL rollout replay — 설계 문서

정책 하나를 짧게(기본 4초, 100 Hz면 400 스텝) 돌려 매 스텝을 기록합니다. 브라우저에서 로봇 움직임과 보상 구성을 같이 재생합니다.
보상 함수를 검토하기 위한 오프라인 분석 도구입니다. 기록 중인 시뮬레이터를 실시간으로 따라가는 모드도 있습니다.

## 1. 목적

PPO 정책이 어떤 상태에서 왜 그 행동을 냈는지를 보상 관점에서 보이게 합니다.

- 그 스텝의 보상이 어떤 항목의 합으로 만들어졌는가 (항목별 막대)
- critic이 예상한 미래 보상 V(s)와 실제로 받은 할인 미래 보상 G_t는 얼마인가, 어떤 항목이 G_t를 움직였는가
- advantage A_t의 부호: PPO가 이 상태에서 이 행동의 확률을 올리는가, 내리는가
- 정책이 받은 지형 입력과 실제 지형, 학생 카메라가 본 것의 차이

비목표: 학습 중 실시간 모니터링(TensorBoard 영역), 대량 통계(MuJoCo 평가 `gd_rbq10_deploy_vrl/evaluation/` 영역).

## 2. 구성

| 파일 | 역할 |
|---|---|
| `rollout_log.py` | 기록 핵심. `RolloutLog.before_step` / `after_step` / `finish`. 미래 보상·TD 오차·GAE 계산, JSON 쓰기, 실시간 NDJSON 스트림 |
| `record_rollout.py` | 교사가 운전하는 기록 (Isaac, rsl_rl runner로 체크포인트 로드) |
| `scripts/gast/train_student_live.py --replay_out` | 학생이 운전하는 기록 (증류와 같은 루프, 학습·저장 없음) |
| `server.py` | 로컬 HTTP 서버(127.0.0.1). 저장소 파일 서빙, 기록 실행, 목록, 실시간 스트림 API |
| `viewer.html` | 뷰어. three.js r128(`vendor/`, MIT)로 RBQ10 URDF 메시를 그림 |
| `make_demo.py` | 시뮬레이터 없이 뷰어를 시험하는 합성 기록 |
| `serve.sh` | 서버 실행 |

흐름: 기록기(Isaac) → `logs/rl_replay/<name>.json` (+ 실시간이면 `<name>.live.ndjson`) → `server.py` → `viewer.html`

## 3. 실행

```bash
python3 tools/rl_replay/server.py --port 8767     # → http://127.0.0.1:8767/tools/rl_replay/viewer.html
```

뷰어 메뉴: ● 기록 · 📂 불러오기 · ▶ 재생 · ⏸ 일시정지 · ■ 정지 · ◀1 / 1▶ · 속도 · 환경 · 지형 레이어 3종. 키보드: Space, ←/→.

교사 운전 기록 (명령줄):
```bash
PYTHONPATH=src apptainer exec --nv ~/workspace/gd_lab_isaaclab.sif ~/workspace/venv_apptainer/bin/python \
  tools/rl_replay/record_rollout.py --headless --task Gd-GastGapCleanV21-Rbq10-Dreamwaq-v0 \
  --checkpoint checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt \
  --num_envs 8 --seconds 4 --warmup_seconds 1.5 --vx 1.0 --student_view \
  --out logs/rl_replay/gast2000.json --live logs/rl_replay/gast2000.live.ndjson
```

학생 운전 기록 (예: `/home/user/gd_project/claude_handoff/run_rl_replay_records_20261008.sh`):
```bash
... scripts/gast/train_student_live.py --headless --task Gd-GastTeacherGastStudent-Rbq10-Dreamwaq-Vision-v0 --v21_env \
  --teacher_checkpoint <교사 model_2000.pt> --student_resume <student_top5_iter_17200.pt> --num_envs 8 \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
  --replay_out logs/rl_replay/student17200.json --replay_live logs/rl_replay/student17200.live.ndjson --replay_vx 1.0
```

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--seconds` / `--replay_seconds` | 4 | 기록 길이 |
| `--warmup_seconds` / `--replay_warmup_seconds` | 1.0 / 1.5 | 기록 전에 걷는 시간 (초기 자세 영향 제거) |
| `--vx` / `--replay_vx` | 없음 | 전진 명령 고정. 없으면 task의 명령 분포 |
| `--stochastic` | 끔 | 교사: 학습처럼 행동 샘플. 기본은 배포처럼 평균 |
| `--student_view` | 끔 | 벨리 카메라를 렌더하고 깊이를 점구름으로 기록 (학생 기록은 항상 켬) |
| `--cloud_envs` / `--cloud_stride` | 2 / 4 | 점구름을 남길 env 수, 픽셀 간격 (용량) |
| `--live` / `--replay_live` | 없음 | 매 스텝 NDJSON 한 줄씩 스트림 |

## 4. 운전자 두 종류

| | 교사 운전 (`record_rollout.py`) | 학생 운전 (`--replay_out`) |
|---|---|---|
| 행동 | 교사 actor(교사 지형 latent) 평균, `--stochastic`이면 샘플 | 교사 actor + **학생 지형 latent**. 학생 프레임이 0.3초보다 오래되면 latent 0 |
| 카메라 | `--student_view`일 때만 렌더 (행동에는 안 쓰임) | 증류와 같은 전송: 70–100 ms 간격, 0–150 ms 지연, 5% 드롭 |
| V(s) | 교사 critic | **교사 critic** (학생 전용 critic 없음) |
| `action_std` | 정책 std | 0 (결정적) |
| 추가 기록 | 없음 | `latent_err`, `student_fresh`, `teacher_action` |
| meta.driver | `"teacher"` | `"student"` (+ `teacher_checkpoint`, `student_iteration`) |

학생 운전은 증류 루프를 그대로 씁니다. `student_warmup=0`, `student_ramp=1`로 학생이 100% 운전하게 하고, backward와 optimizer step을 건너뛰어 가중치는 바뀌지 않습니다. Top-5와 저장 간격은 10^9로 막습니다. 기록이 차면 JSON을 쓰고 바로 종료합니다.

## 5. 데이터 형식 (`<name>.json`)

```
{ "meta": {...}, "envs": [ {env 0}, {env 1}, ... ] }
```

**meta**: `task`, `checkpoint`, `checkpoint_sha256`, `dt`(0.01), `gamma`, `lam`, `steps`(T), `num_envs`(N), `stochastic`, `vx`,
`term_names`(K개 보상 항목), `body_names`, `joint_names`(12), `foot_names`(4), `driver`, `notes`.

**env별 배열** (모두 길이 T, 소수 4자리 반올림):

| 키 | 스텝당 모양 | 내용 |
|---|---|---|
| `base_pos`, `base_quat` | 3, 4 | 몸체 월드 위치, 자세 (Isaac wxyz) |
| `base_lin_vel_b` | 3 | 몸체 좌표계 선속도 |
| `command` | 3 | 명령 (vx, vy, ωz) |
| `bodies` | B×3 | 모든 링크 월드 위치 (뷰어 FK 검증용) |
| `joint_pos`, `joint_vel` | 12 | 관절 |
| `action_mean`, `action_std`, `action` | 12 | 정책 평균, std, 실제 출력 |
| `value` | 스칼라 | critic V(s_t) |
| `latent` | 32 | 행동에 들어간 지형 latent (교사 운전이면 교사, 학생 운전이면 학생 것) |
| `terms` | K | 보상 항목별 이번 스텝 값 = `RewardManager._step_reward × dt` |
| `reward` | 스칼라 | `env.step`이 돌려준 보상 |
| `done` | 0/1 | 이 스텝에서 에피소드 종료 (넘어짐 또는 시간 초과) |
| `foot_contact` | 4 (0/1) | 발 접촉력 > 1 N |
| `scan` | 187×3 | 높이 스캐너 적중점 (11×17, 정답 지형) |
| `scan_z` | 스칼라 | 스캐너 원점 높이 |
| `terrain_obs` | 374 | 교사 인코더의 지형 입력: 높이 187 + 유효 187 |
| `cloud` | 가변 (3의 배수) | 학생 카메라 점구름 [x,y,z,...] (앞쪽 `cloud_envs`개 env만) |
| `return`, `advantage`, `delta` | 스칼라 | G_t, A_t, δ_t (6절) |
| `term_returns` | K | G_t^k |
| 학생 운전만: `latent_err`, `student_fresh`, `teacher_action` | 스칼라, 0/1, 12 | 학생·교사 latent MSE, 프레임 신선도, 교사였다면 낼 행동 |

**실시간 스트림** (`<name>.live.ndjson`), 줄마다 JSON 하나:
1. `{"meta": {..., "steps_planned": T, "live": true}}`
2. 스텝마다 `{"t": i, "envs": [{위 키들의 스텝 i 값}, ...]}` (`return`·`advantage`·`delta`·`term_returns` 없음)
3. 끝 `{"end": true, "out": "<최종 json 경로>"}`

## 6. 수식

보상 r_t는 그 스텝에 받은 값입니다. 각 항목 k의 몫 r^k_t = `_step_reward[k] × dt`이고, Σ_k r^k_t = r_t가 되어야 합니다(7절 확인 필요 항목).

finish에서 env마다 뒤에서부터 계산합니다. alive_t = 1 − done_t이고, 부트스트랩은 마지막 스텝 다음 상태의 critic 값 V(s_T)입니다.

```latex
\delta_t = r_t + \gamma\, V(s_{t+1})\,(1-d_t) - V(s_t)
```
```latex
A_t = \delta_t + \gamma\lambda\,(1-d_t)\,A_{t+1}, \qquad A_T = 0
```
```latex
G_t = r_t + \gamma\,(1-d_t)\,G_{t+1}, \qquad G_T = V(s_T)
```
```latex
G^k_t = r^k_t + \gamma\,(1-d_t)\,G^k_{t+1}, \qquad G^k_T = 0
```

- γ, λ는 agent cfg 값입니다(현재 0.99, 0.95). 100 Hz에서 γ=0.99의 유효 지평은 약 1초(100 스텝)입니다.
- Σ_k G^k_t + γ^{T−t}·V(s_T)(중간 종료가 없을 때) = G_t입니다. 부트스트랩 V(s_T)는 항목으로 나눌 수 없어 G_t에만 들어갑니다.
- 실시간 모드의 G·A는 뷰어가 같은 식으로 계산하되, 그때까지 받은 마지막 스텝의 V를 부트스트랩으로 씁니다(잠정값). 최종 JSON이 오면 교체합니다.

**"왜 이 행동인가"** 문구: A_t > 0.02면 기대보다 좋았음(PPO가 확률을 올림), A_t < −0.02면 나빴음(내림), 그 사이는 영향 작음.

**교사 입력 지형 복원**: 교사 인코더 입력 높이 h = clip(scan_z − z − 0.5, −1, 1) × 5 (`CameraVisibleTerrain`). 뷰어는 유효 칸만 z = scan_z − 0.5 − h/5로 되돌려 그립니다.

## 7. 알려진 한계

**검증 상태 (2026-10-08)**
- 실제 Isaac 기록은 아직 한 번도 성공 확인하지 않았습니다. 지금까지 확인한 것은 세 가지입니다.
  - 합성 데모와 가짜 스트림으로 뷰어 확인
  - 가짜 환경으로 `RolloutLog` CPU 점검
  - 문법 검사
- 첫 실제 기록은 `run_rl_replay_records_20261008.sh`에서 돕니다.

**확인이 필요한 가정**
- Σ_k terms = reward: RewardManager가 `_step_reward`를 가중치 적용 rate로 두고 reward = Σ rate × dt라는 가정입니다. wrapper가 보상을 바꾸면 깨집니다.
- `gast_history` 마지막 프레임의 높이 인코딩이 `terrain`과 같다는 가정입니다(`GastTeacherView`가 같은 것으로 취급).

**수식상의 차이**
- 시간 초과(time-out)도 넘어짐과 똑같이 부트스트랩을 끊습니다. rsl_rl 학습은 time-out에 γV를 더하므로, 기록 창 안에서 시간 초과가 나면 그 근처 G·A가 학습 때와 다릅니다. 4초 창이라 드물지만, done 스텝 주변은 주의해야 합니다.
- 4초 창 끝은 V(s_T)로 부트스트랩합니다. 창 끝 근처의 G_t는 critic 추정에 크게 의존합니다.
- 창 안에서 에피소드가 끝나면 다음 스텝은 새 에피소드입니다(로봇이 순간이동). 뷰어 타임라인에 빨간 세로선으로 표시됩니다.

**학생 운전의 해석**
- V(s)와 A_t는 교사 critic 기준입니다. A_t는 "교사 정책 기준 기대값보다 학생 행동의 결과가 좋았나"로 읽어야 합니다. 학생 정책 자신의 가치 추정이 아닙니다.
- 학생 프레임에 학습용 증강이 그대로 적용됩니다. `GastDistillation.update`의 증강은 프레임 누락 10%, 블러 15%, 카메라별 가림 8%입니다. 배포 때보다 입력이 나쁜 조건이라, 학생 성능이 낮게 보일 수 있습니다.
- `action_std`는 0이라 std 막대가 없습니다.

**지형 레이어**
- 학생 카메라 점구름은 매 스텝 가장 최근 렌더입니다. 학생이 실제로 받은 프레임은 전송 지연(최대 150 ms)이 있어 그보다 늦은 장면이고, 드롭된 프레임은 받지 못했습니다. 그래서 "학생이 볼 수 있는 것"이지 "그 순간 학생이 가진 것"은 아닙니다.
- 점구름은 앞쪽 `cloud_envs`개 env만, 픽셀 4칸마다, 유효 깊이 범위만 남깁니다.
- 교사 입력 높이는 인코딩 전 (scan_z − z − 0.5)를 ±1 m로 자릅니다. 그래서 스캐너보다 1.5 m 넘게 낮거나 0.5 m 넘게 높은 칸은 경계값으로 그려집니다.

**기타**
- 용량: 합성 데모(1 env, 400 스텝, 점구름 없음)가 3 MB입니다. 8 env + 점구름 2 env는 수십 MB로 예상하지만 실측하지 않았습니다.
- `file://`로 열면 URDF를 못 읽어 2D 막대 인형으로 그립니다. 3D 메시는 `server.py`로 열어야 합니다.
- `server.py`는 127.0.0.1에만 바인딩하고 인증이 없습니다. 기록은 한 번에 하나이고, 학습·증류 프로세스가 있으면 `force` 없이는 거부합니다.
- 기록은 GPU에서 학습과 함께 돌면 메모리가 부족할 수 있습니다(RTX 5090 32 GB에서 증류 512 env가 약 30 GB 사용).
