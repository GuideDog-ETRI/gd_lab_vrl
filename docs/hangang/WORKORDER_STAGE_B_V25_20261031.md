# 작업서: 한강 B단계 (v2.5 레시피: 통로·회전·낭떠러지 정지), 16노드

2026-10-31 · B.S.Seo (ETRI) · 브랜치 `gh200` · 설계 `docs/design/GAST_NARROW_V25_DESIGN.md` · 앞 작업서 `WORKORDER_LINEAGE16_20261030.md`

이 작업서는 **a6000의 v2.5 검증 런(`narrow_v25a_653_…`)이 it100 관문을 넘은 뒤**에만 실행한다. 관문 결과와 실행 허가는 사용자가 준다.

## 1. 목적과 판정

A단계(3906 → 후진 갭) 최고 모델에서 v2.5 레시피로 1,200 update. 결과는 v2.3(602)·v24h(653)과 같은 입력·출력 구조의 정책 하나.

| 기능 | 판정 (E2 기본 조건, `configs/gast_narrow/e2/v25_turn.yaml`) |
|---|---|
| 전·후진 갭, 계단 | 건넘·통과 ≥ A 모델 − 1 %p, 갭 전진 머리 들림 평균 ≤ 7°, p90 ≤ 10° |
| 좁은 통로(전·후진), 막히면 서기, 막다른 길 서기 | 602 − 1 %p 이내, 막다른 길 서기 ≥ 90 %, 후진 통로 ≥ 전진 − 5 %p |
| 제자리 회전 (v2.5 성공 정의) | 회전 가능 성공 ≥ 90 %, 깨끗한 회전(≤ 10 cm) 보고, 회전 불가 서기 ≥ 90 %, 침범 ≤ 5 % |
| 낭떠러지 정지 (새) | 네 방향 서기 ≥ 90 %, 끝선 침범 ≤ 5 % (옆 방향은 따로 표기; 70 % 미만이면 v2.6 과제) |

## 2. 48시간 제한 없음 (사용자 결정 10-31)

A·B 모두 계획한 update 수를 **재제출로 완주**한다. 시간 때문에 update 수를 줄이지 않는다. 24 h 제한마다 sbatch 스크립트가 마지막 저장본에서 스스로 재제출한다(§5에서 확인).

## 3. 실행 전 점검 (한 줄씩, 결과를 사용자에게 보고)

```bash
G=/scratch/t515a05/gd_lab
# 1) A 결과: STAGE_A 선정 근거(E2 요약 경로)와 sha256
timeout -k 10 60 sha256sum $STAGE_A
# 2) 코드: gh200을 번들로 올린 뒤(GitHub fetch 금지) 헤드가 지시한 커밋인지
cd $G/gd_lab_vrl && git log --oneline -1 && git status --short | head
# 3) 지형 65열·v25 과제가 로드되는지 — 1노드 smoke (아래 §4의 smoke 명령), 10 iteration 안에 끝나는지
# 4) 대기열: natStr 노드 수와 제외 노드
timeout -k 10 120 sinfo -p natStr -h -o "%T %D" | sort
```

## 4. 실행 순서

```bash
cd /scratch/t515a05/gd_lab/jobs
# smoke (1노드, 5 update): v25 로더가 A 모델(backgap)에서 LiDAR 인코더 새로 + 첫 층 0 열 확장을 하는지,
# [NARROW_ANCHOR] stand_target 가 STAGE_A 인지, gen.num_cols 65 인지 로그에서 확인
STAGE_A=<A 최고 .pt> STAGE_A_SHA256=<sha> NARROW_UPDATES=5 PER_GPU=256 NO_RESUBMIT=1 \
  sbatch -N 1 --time=00:40:00 lineage16_b.sbatch

# 본 런 (16노드, 1,200 update, LINEAGE_RECIPE 기본 v25)
STAGE_A=<A 최고 .pt> STAGE_A_SHA256=<sha> sbatch lineage16_b.sbatch
```

- 재제출(24 h 제한·STOP·크래시 뒤)은 스크립트가 한다. 수동으로 할 때만: `STAGE_A/STAGE_A_SHA256 그대로, NARROW_SOURCE=<model_N.pt> NARROW_SOURCE_SHA256=<sha> NARROW_UPDATES=<남은 수> sbatch lineage16_b.sbatch`.
- 정지 anchor 목표는 스크립트가 `NARROW_STAND_ANCHOR_CKPT=$STAGE_A`로 고정한다(재제출에서 NARROW_SOURCE가 자기 저장본으로 바뀌어도 목표는 A 모델). 이 줄이 로그 `[NARROW_ANCHOR] … stand_target=` 에 보여야 한다.
- Top-5 기준: `configs/gast_narrow/online_top5_lineage16_b_v25.json` (막힌 통로 서기 ≥ 0.85, 침범 ≤ 0.05, 막다른 길 ≥ 0.85, 낭떠러지 서기 ≥ 0.85·침범 ≤ 0.05, 정지 중 10° 초과 시간 ≤ 6 s, 갭 레벨 ≥ 8.5). **it200에 E2 1회(노드 1대)로 기준을 재조정**한다(파일은 실행 중 다시 읽힘).

## 5. 감시 규칙 — 어제(10-30 21:14 ~ 10-31 11:28) 같은 멈춤이 다시 없도록

어제 멈춘 원인: 끝난 job(1182064)에 `srun --overlap --jobid=… nvidia-smi`를 걸어 21시간 대기했고, 앞의 `timeout 120`은 `-k`가 없어 srun을 죽이지 못했다. 아래 규칙을 지키면 같은 일이 생기지 않는다.

| 규칙 | 내용 |
|---|---|
| 모든 Slurm·ssh·git 원격 명령 | `timeout -k 10 120 <명령>` 으로 감싼다(120 s 뒤 TERM, 10 s 뒤 KILL). 예: `timeout -k 10 120 squeue -u t515a05` |
| srun 진단 | `timeout -k 10 60 squeue -j <id> -h -o %T` 가 `RUNNING`을 돌려준 **직후**에만, 그리고 `timeout -k 10 90 srun --overlap --jobid=<id> …`로. 끝난 job ID에는 절대 걸지 않는다 |
| GPU 사용률 확인 | 필요 없다. rank 0 로그의 iteration 시간으로 충분하다. 꼭 보려면 위 규칙으로 1회만 |
| 백그라운드 대기 루프 | 최대 시간을 둔다: `timeout -k 10 1800 bash -c 'until …; do sleep 60; done'` (30분). 끝나지 않는 `tail -f`·`while true`는 쓰지 않는다 |
| 명령 묶기 | `&&`·`;`로 긴 체인을 만들지 않는다. GitHub fetch는 실행 명령과 같은 줄에 두지 않는다(onvlm2에서 1시간 멈춘 적 있음). 코드는 번들로 받는다 |
| 응답 없는 명령 | 2분 넘게 결과가 없으면 그 명령을 중단하고 상황을 먼저 보고한다 |
| 보고 주기 | 10분마다: 현재 시각, job ID·상태(`squeue`), 최근 Learning iteration, iteration 시간, Top-5 Top-1 점수, 주요 지표(아래) |

**보고할 지표**(rank 0 로그 / top5_decisions.jsonl): `narrow_passage_rate`, `narrow_hold_rate`, `dead_end_hold_rate`, `turn_stop_rate`, `turn_success_rate`(R), `turn_clean_rate`, `turn_leg_impulse_p90`, `cliff_hold_rate`, `cliff_hold_by_direction`, `cliff_breach_rate`, `stand_nose_up_t10_mean`, `platform_gap_termination_rate`, `gap_clean_crossing_rate`, `stairs_termination_rate`, 커리큘럼 레벨(narrow_cliff, narrow_turn_*, narrow_dead_end), `stand_anchor_mse`, value loss.

**중단·이상 판단** (보고만 하고 중단은 사용자·BSSEO 세션 결정):
- 발산: `stand_anchor_mse` 3배 이상 급등 후 3.0 초과, value loss > 1.0, base_contact 종료 > 0.25, nonfinite.
- 회귀: it100 이후 `narrow_passage_rate` < 0.8 또는 `narrow_hold_rate` < 0.8 이 20 it 연속(v24i 패턴).
- 멈춤: 10분 넘게 새 iteration 줄이 없으면 노드 문제. `timeout -k 10 60 scontrol show job <id>`로 상태를 보고, `watchdog_v23.sh` 방식(노드 제외 → 마지막 저장본에서 재시작)을 쓰되 재시작 명령은 §4의 수동 재제출 명령.
- STOP: `touch logs/launches/$GAST_RUN_ID/STOP` (저장 후 종료). kill은 쓰지 않는다.

## 6. 재제출 체인 확인

24 h 제한 15분 전 USR1 → STOP → 저장 → `TRAIN_DONE` → `resubmit:` 줄 → 새 job ID. 각 재제출 뒤 10분 안에 확인할 것: `timeout -k 10 60 squeue -u t515a05`에 새 lin16_b가 보이는지, 새 로그의 warm start 줄에 `source_kind narrow_v25_continue`, `ppo_optimizer restored`, `stand_target=<STAGE_A>`가 있는지. 없으면 §4의 수동 재제출 명령으로 다시 올리고 보고한다.

## 7. E2

학습 종료 후(또는 it300·600·900 중간 점검): `configs/gast_narrow/e2/v25_turn.yaml`을 복사해 models를 {STAGE_A, v23u602, Top-5}로 바꾸고 `e2_multi.sbatch` 방식으로 노드 5대 분할. `run_e2.py`는 부분 묶음(`--only`) 실행을 `__only-…` 디렉터리로 분리하므로 어제 같은 retention 누락은 없다. 결과: `summary.md` 전문과 §1 판정표를 사용자에게 보고. 모델·요약은 번들 또는 tar로 5090 PC에 전달(커밋에 Claude 관련 줄 금지).

## 8. 바뀐 코드 (이 작업서와 함께)

| 파일 | 내용 |
|---|---|
| `scripts/gast_narrow/hangang/lineage16_b.sbatch` | `LINEAGE_RECIPE` 기본 v25(트레이너·과제·Top-5 기준 파일), `NARROW_STAND_ANCHOR_CKPT=$STAGE_A` 고정 |
| `configs/gast_narrow/online_top5_lineage16_b_v25.json` | B단계 v2.5 Top-5 기준 |
| v2.5 코드 일체 | `GAST_NARROW_V25_DESIGN.md` §10·§11 |
