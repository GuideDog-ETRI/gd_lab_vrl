# RL rollout replay — 설계 문서

정책 하나를 짧게(기본 4초, 100 Hz면 400 스텝) 돌려 매 스텝을 기록합니다. 브라우저에서 로봇 움직임과 보상 구성을 같이 재생합니다.
보상 함수를 검토하기 위한 오프라인 분석 도구입니다. 기록 중인 시뮬레이터를 실시간으로 따라가는 모드도 있습니다.

## 1. 목적

PPO 정책이 어떤 상태에서 왜 그 행동을 냈는지를 보상 관점에서 보이게 합니다.

- 그 스텝의 보상이 어떤 항목의 합으로 만들어졌는가 (항목별 막대)
- critic이 예상한 미래 보상 V(s)와 실제로 받은 할인 미래 보상 G_t는 얼마인가, 어떤 항목이 G_t를 움직였는가
- advantage A_t의 부호: 이 기록 구간에서 교사 critic 기준 결과가 기대보다 큰가, 작은가 (실제 PPO 업데이트 방향 아님)
- 정책이 받은 지형 입력과 실제 지형, 학생 카메라가 본 것의 차이

비목표: 학습 중 실시간 모니터링(TensorBoard 영역), 대량 통계(MuJoCo 평가 `gd_rbq10_deploy_vrl/evaluation/` 영역).

## 2. 구성

| 파일 | 역할 |
|---|---|
| `rollout_log.py` | 기록 핵심. `RolloutLog.before_step` / `after_step` / `finish`. 미래 보상·TD 오차·GAE 계산, JSON 쓰기, 실시간 NDJSON 스트림 |
| `record_rollout.py` | 교사가 운전하는 기록 (Isaac, rsl_rl runner로 체크포인트 로드) |
| `gast/scripts/train_student_live.py --replay_out` | 학생이 운전하는 기록 (증류와 같은 루프, 학습·저장 없음) |
| `server.py` | 로컬 HTTP 서버(127.0.0.1). 저장소 파일 서빙, 기록 실행, 목록, 실시간 스트림 API |
| `viewer.html` | 뷰어. three.js r128(`vendor/`, MIT)로 RBQ10 URDF 메시를 그림 |
| `make_demo.py` | 시뮬레이터 없이 뷰어를 시험하는 합성 기록 |
| `serve.sh` | 서버 실행 |

흐름: 기록기(Isaac) → `logs/rl_replay/<name>.json` (+ 실시간이면 `<name>.live.ndjson`) → `server.py` → `viewer.html`

## 3. 실행

```bash
TRAIN_ARM=4 python3 tools/rl_replay/server.py --port 8767     # → http://127.0.0.1:8767/tools/rl_replay/viewer.html
```

뷰어 메뉴: ● 기록 · 📂 불러오기 · ▶ 재생 · ⏸ 일시정지 · ■ 정지 · ◀1 / 1▶ · 속도 · 환경 · 지형 레이어 3종. 키보드: Space, ←/→.

교사 운전 기록 (명령줄):
```bash
TRAIN_ARM=4 PYTHONPATH=gast/src apptainer exec --nv ~/workspace/gd_lab_isaaclab.sif ~/workspace/venv_apptainer/bin/python \
  tools/rl_replay/record_rollout.py --headless --task Gd-GastGapCleanV21-Rbq10-Dreamwaq-v0 \
  --checkpoint checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt \
  --num_envs 8 --seconds 4 --warmup_seconds 1.5 --vx 1.0 --student_view \
  --out logs/rl_replay/gast2000.json --live logs/rl_replay/gast2000.live.ndjson
```

학생 운전 기록 (예: `/home/user/gd_project/claude_handoff/run_rl_replay_records_20261008.sh`):
```bash
... gast/scripts/train_student_live.py --headless --task Gd-GastTeacherGastStudent-Rbq10-Dreamwaq-Vision-v0 --v21_env \
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

학생 운전은 `student_replay.py`의 별도 `torch.no_grad()` 경로다. student.eval(), actor에 학생 latent 직접 주입, 학습 증강 없음. interval/delay/drop 전송은 유지하며 optimizer·TensorBoard·Top-5·live-config를 만들지 않는다. 과거 run 이름 재사용과 출력 충돌을 거부한다. merged layout에서는 스크립트 경로만 `scripts/gast/train_student_live.py`로 바뀐다.

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
| `foot_contact` | 4 (0/1) | before_step의 발 접촉력 > 1 N |
| `foot_contact_after` | 4 (0/1/null) | after_step; 종료 행은 pre-reset hook, 미취득은 null |
| `terrain_scan`, `terrain_scan_z`, `terrain_capture_step` | 187×3, scalar, scalar | 실제 term capture clock과 대응한 좌표. 미취득 행은 null |
| `terrain_history` | 8×375 | GAST 운전일 때 실제 8시점 ego-warped 입력 (과거 높이의 현재 월드 z라는 뜻 아님) |
| `cloud_capture_step` | scalar | 표시된 카메라 snapshot의 capture step; 전달된 프레임과 구분 |
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

보상 r_t는 그 스텝에 받은 값입니다. 각 항목 k의 몫 r^k_t = `_step_reward[k] × dt`이고, Σ_k r^k_t = r_t가 되어야 합니다(RewardManager 소스 확인; PPO 내부 timeout/RND 추가 보상과는 별개).

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

**해석:** A의 부호는 이 기록 구간의 교사 critic 기준 GAE다. advantage 정규화·clipping·time-out 보정을 재현하지 않으므로 실제 PPO 확률 변경을 뜻하지 않는다.

**교사 입력 지형 복원**: 교사 인코더 입력 높이 h = clip(scan_z − z − 0.5, −1, 1) × 5 (`CameraVisibleTerrain`). 뷰어는 유효 칸만 z = scan_z − 0.5 − h/5로 되돌려 그립니다.

## 7. 수정 및 검증 상태 (2026-10-08)
- 격리 worktree에서만 구현. 실행 중 기본 checkout/merge_gd_lab에는 적용하지 않았다.
- Arm override(play=True), BIVT 등록, main GAST PYTHONPATH, 명령 term 고정(관측 중복 계산 없음).
- TRAIN_ARM을 반드시 명시. 실제 dt/decimation/actuator gains는 meta에 기록.
- NaN/Inf는 null, allow_nan=False. JSON은 같은 파일시스템에서 atomic no-clobber 게시.
- CLI/server 공유 advisory lock. 서버는 GPU compute PID/메모리 사용 확인 실패 시 거부한다.
  force는 busy 우회하지 않는다. 체크포인트는 이 저장소 내부 .pt만 허용.
- /api/live의 offset/next는 byte offset, 완성 줄 경계만 허용하며 최대 32줄씩 반환한다.
- localhost Host/Origin 검증, 엄격한 타입/범위/출력 이름 검증. 메타데이터는 textContent로 표시.
- CPU 수식/표준 JSON/명령/교사 입력/서버/공유 잠금/학생 frozen loop 테스트와 Node fake-live 테스트를 둔다.
- Isaac 실기록 및 실제 브라우저 WebGL/대용량 성능 측정은 미실시. GPU가 실험 중이므로 성공으로 간주하지 않는다.

### 남은 해석상 한계
- terminal과 time-out 모두 bootstrap 차단. rsl_rl timeout reward 보정은 재현하지 않는다.
- 항목별 G에는 bootstrap 몫 없음. live의 마지막 V(s_t)는 아직 없는 V(s_next)의 잠정 근사.
- GAST history의 과거 프레임은 xy ego-warped이나 높이 기준 z의 과거 이동 보정까지 검증된 것이 아니다.
  뷰어는 최신 프레임만 overlay하고 실제 history는 별도 배열로 보존한다.
- Ray 입력 좌표는 실제 observation term last_step과 일치할 때만 캐시한다. 기록 시작 이전 capture는
  다음 capture까지 null로 표시한다. 모르는 좌표를 현재 scanner에 붙이지 않는다.
- 카메라 점구름은 캡처 snapshot이며 student transport delivery 프레임 자체가 아니다.
  교사 student_view는 정책 tick마다 렌더하도록 설정한다. 실제 렌더/pose 정렬과 reset hook는 Isaac 검증 대기.
- 높이 입력 clip ±1m를 역변환하므로 범위 밖 실제 높이는 복원 불가.
- 학생 replay는 학습용 증강을 제거했으므로 과거 replay 결과와 직접 동일 조건 비교하면 안 된다.
- advisory lock는 갱신된 recorder끼리만 공유한다. 기존 training/sim 외부 작업은 이 잠금을 사용하지 않는다.
  GPU 검사는 순간 snapshot이며 이후 외부 작업이 시작되는 경쟁을 원천 차단하지 않는다. CLI도 실행 전에 GPU를 직접 확인해야 한다.
- server/GUI를 실행하는 것과 Isaac recording을 실행하는 것은 별개다. 현재 실험이 끝나기 전에는 recording 금지.
- CPU 검증 명령:
  `python -B -m unittest discover -s tools/rl_replay -p 'test_*.py' -v`
  `node tools/rl_replay/test_viewer.cjs`
