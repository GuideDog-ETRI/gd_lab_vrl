# ARM4 BIVT-Ray Clean 갭 파인튜닝: 3-GPU 30,000 업데이트 학습 조건

2026-10-05 사용자 결정에 따른 학습 조건이다. 학습 서버 `10.77.32.231`(OnVLM2-A)에서
실행한다. 학습 중에는 코드를 바꾸지 않는다. 실제 실행 결과(run id, 시작 시각,
정지 파일)는 문서 끝 "실행 기록"에 덧붙인다.

## 목적

BIVT-Ray 선생 17206이 갭 위에서 발을 내렸다가 올리는 동작을, 발을 앞뒤로 뻗어
빠뜨리지 않고 건너는 동작으로 바꾼다.

## 기준

| 항목 | 값 |
|---|---|
| 코드 | `vrl_models` `a8003b8` 이후(이 문서를 포함한 커밋). 작업 폴더 `/data/users/bsseo/gd_lab_vrl_worktrees/bivt-ray-ddp-newcam-17206` |
| 태스크 | `Gd-VrlGapFinetuneCleanRaycast-Rbq10-Dreamwaq-v0` (렌더링 없는 raycast 가시성) |
| 로봇 | ARM4 (`TRAIN_ARM=4`): 100 Hz(dt 0.005 × decimation 2), Kp hip 123.39 / knee 127.77, Kd 2.4 |
| 카메라 | `vendor_new` (RBQ SDK 보정값) |
| 시작 체크포인트 | `checkpoints/teachers/bivt/ray_top1_17206_20261003/teacher/17206_top1.pt` (sha256 `a70adbda5b73925f5936eaef27ee3b33686d9f7489486afdda9bf40ddb0c326d`) |
| 업데이트 | 17207부터 **30,000회** → `GAP_TARGET_ITERATIONS=47207` (17207–47206, 마지막 `model_47206.pt`) |
| GPU / env | GPU 0·1·2, 총 4096 env (1366/1365/1365). GPU 3(VLLM PID 329833)은 사용·정지 금지 |
| seed | 42 (rank마다 +local_rank) |

`--target_iterations`는 "총 완료 업데이트 번호"다. `load()`가 저장 iteration+1(17207)에서
재개하므로 `max_iterations = target − 17207`이다. smoke(target 17210)는 17207–17209의 3회였다.

## PPO

| 항목 | 값 |
|---|---|
| PPO learning rate | 1e-4 고정 (`--force_ppo_lr 1e-4`, `schedule=fixed`) |
| CENet learning rate | 1e-3 |
| rollout / 학습 | 100 step/env, epoch 5, mini-batch 4 |
| 기타 | gamma 0.99, lambda 0.95, clip 0.2, entropy 0.01, max_grad_norm 1.0 |
| 다중 GPU | gradient·empirical normalizer(two-pass)·CENet·AdaBoot rank 동기화, 재개 직후와 100 update마다 상태 해시 비교, 비유한 gradient는 모든 rank가 함께 실패 |

## 보상 (갭 지형에서만 계산)

| 항목 | 가중치 | 정의 |
|---|---|---|
| `platform_gap_intrusion` | −3.0 /s | 갭 슬롯 위 발이 아래 발판보다 3 cm 넘게 내려가면 시작, 13 cm에서 최대(0–1), 네 발 평균(뒷발 포함) |
| `platform_gap_clean` | +1.5 /회 | 건너는 동안 어느 발도 3 cm 넘게 빠지지 않은 통과, 방향·episode당 1회 |
| `platform_gap_foot_drop` | −2.0 /s | 기존 선생 항목. 발판 대비 22 cm 넘게 떨어질 때(47 cm에서 최대) |
| `platform_gap_crossing` | +0.5 /회 | 기존 선생 항목. 네 발 모두 건너 0.1 s 안정. 갭 curriculum과 clean 판정이 이 기록을 사용하므로 끄지 않는다 |
| `platform_gap_monitor` | 1.0 | 항상 0. intrusion/clean 계산과 진단 통계 기록 |

- 깨끗한 통과 0.5+1.5=2.0, 발을 빠뜨린 통과 0.5.
- 갭 보상은 `platform_gap` 지형 칸에서만 계산된다(`in_slot & family`). 계단·경사 등 다른 지형의
  보상은 17206 선생과 같다.
- 속도 명령 범위는 처음부터 최종 범위(갭 명령 상한 0.6 m/s 유지), 지형 curriculum은 17206의
  지형별 floor(mean)로 복원한다(약 8–10). 지형 15열 중 5열이 갭이다.

## 저장과 Top-5

- 체크포인트 100 iteration마다. 새 run이므로 리더보드는 비어서 시작한다.
- Top-5는 매 iteration rollout의 완료 episode로 계산하는 online proxy다(`configs/online_top5.json`).
- 점수: 0.40·G + 0.25·B(1−몸통 접촉) + 0.15·P(전진) + 0.10·T(추종) + 0.05·E(에너지) + 0.05·Q(총 보상),
  (family, level) macro 평균.
- **G = 깨끗한 통과율.** 갭 모니터가 있는 갭 파인튜닝 태스크에서 G는 `gap_clean_success`
  (통과했고, clean 보너스를 받았고, 옆으로 돌아가거나 넘어지지 않은 episode)의 비율이다.
  모니터가 없는 태스크는 기존 통과율(`platform_gap_crossing`)을 쓴다. 리더보드 항목에
  `gap_success_definition`(`clean_crossing`/`crossing`), `criteria_metrics.gap_crossing_rate`(통과율),
  `criteria_metrics.gap_clean_crossing_rate`(깨끗한 통과율)를 기록한다.
- 통과 조건: 갭 평균 레벨 ≥ 8, 전체 평균 지형 레벨 ≥ 9, 몸통 접촉·갭 종료·계단 종료율 각각 ≤ 9%,
  1위 대비 G(깨끗한 통과율) 하락 ≤ 2%p, B 하락 ≤ 1%p, 후보 간격 100.
- gate C는 면제로 manifest에 기록한다:
  "user waived gate C on 2026-10-05: multi-GPU ARM4 Clean gap fine-tuning from 17206 with vendor_new cameras".

## 실행

```bash
cd /data/users/bsseo/gd_lab_vrl_worktrees/bivt-ray-ddp-newcam-17206
tmux new-session -d -s bivt_gap_clean_ddp3 \
  'GAP_GPUS=0,1,2 GAP_TARGET_ITERATIONS=47207 GAP_GATE_WAIVER="user waived gate C on 2026-10-05: multi-GPU ARM4 Clean gap fine-tuning from 17206 with vendor_new cameras" bash scripts/run_bivt_gap_finetune_ddp.sh clean train'
```

- 예상 시간: 약 39–42시간(3-GPU 4096 env smoke 약 4.7 s/iter 기준).
- 정지: 런처가 출력하는 `stop_file=<작업 폴더>/logs/<run_id>.stop`을 만들면 현재 업데이트 후
  체크포인트를 저장하고 종료한다.
- 결과: `logs/vision_rbq10_dreamwaq/arm_4/<시각>_<run_id>/` (manifest `params/gap_finetune_manifest.yaml`,
  `gap_attempts.jsonl`, `best_top5/`). 런처는 `logs/launches/<run_id>/`에 commit, patch, 체크포인트 sha,
  source tarball을 남긴다.

## 감시 항목

- 계단 6종 레벨·종료율을 17206 시작값(레벨 약 9.1–9.6, 종료율 약 5–6%)과 비교한다.
  종료율 2배 이상 또는 레벨 1 이상 하락이면 보고한다(갭 보상은 계단에 직접 적용되지 않지만
  하나의 정책이라 간접 영향 가능).
- 갭: `Curriculum/platform_gap_diagnostics/clean_rate`, `fail_retreat_rate`(갭 회피 여부),
  갭 레벨, 갭 종료율, Top-5의 `gap_clean_crossing_rate`.

## 사전 검증 (2026-10-05)

- 컨테이너 테스트(`PYTHONPATH=src ... pytest -q -p no:cacheprovider tests`): 297 passed / 1 failed
  (`test_gavd::test_onnx_export`, 컨테이너에 onnxruntime 미설치). 이 테스트는 GAVD 학생의 ONNX
  export 결과를 onnxruntime으로 실행해 PyTorch 출력과 비교하는 검사로, 선생 학습 경로와는 무관하다.
- `tools/check_conventions.py`는 기존부터 실패: `src/gd_lab/mdp/platform_gap_attempts.py` 319줄(상한 300),
  `tests/test_platform_gap_finetune.py`의 홈 디렉터리 절대경로 문자열. 이 학습과 무관해 수정하지 않음.
- Clean 3-GPU smoke(192 env)
  - `2026-10-05_22-29-04_gap_clean_ddp3_s42_smoke`(코드 `b9b5d3a`, 3 update) EXIT=0: 3 rank 모두
    17207 재개, manifest(`vendor_new`, distributed, world_size 3, 보상 1/−3/1.5, PPO lr 1e-4 fixed), loss 유한.
  - `2026-10-05_22-48-57_gap_clean_ddp3_s42_smoke`(Top-5 clean G 코드, 6 update) EXIT=0.
  - 두 번 모두 GPU 3 메모리 90,800 MiB 불변.
- 3-GPU 4096 env 동기화 smoke(BIVT-Ray 17206 재개 32 update, 저장본 재개 12 update·4 update마다 상태 검사) EXIT=0.
- 서버 측 준비 보고: `docs/coordination/server_claude_ddp_readiness_20261005.md`.

## 실행 기록

- 시작: 2026-10-05 22:54 KST, tmux `bivt_gap_clean_ddp3`, 코드 `60455f8`
- run id: `2026-10-05_22-54-39_gap_clean_ddp3_s42_train`
- run 폴더: `logs/vision_rbq10_dreamwaq/arm_4/2026-10-05_22-54-55_2026-10-05_22-54-39_gap_clean_ddp3_s42_train/`
- 콘솔 로그: `logs/bivt_gap_clean_ddp3_train_console_20261005_225439.log`
- 정지 파일: `logs/2026-10-05_22-54-39_gap_clean_ddp3_s42_train.stop` (작업 폴더 기준)
- 초기 속도: 약 5.45 s/iter(17233 시점) → 예상 종료 2026-10-07 20시 무렵(약 45시간)
- GPU 메모리: GPU 0·1·2 각 약 16 GB, GPU 3 VLLM 90,800 MiB 불변
