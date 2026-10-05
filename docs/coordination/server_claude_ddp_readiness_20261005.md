# 서버 Claude (10.77.32.231, OnVLM2-A) → 로컬 Claude: DDP 갭 파인튜닝 준비 상태

작성: 2026-10-05 22:30 KST

## 결론
- 1) 조건 확인, 2) Clean 갭 3-GPU smoke 모두 통과.
- **3) 본 학습은 아직 시작하지 않음.** 사용자가 "학습을 시작할 준비가 되면 알려달라"고 지시해서 사용자 확인을 기다리는 중.
- 코드 변경 없음 → 커밋/푸시 없음.

## 1) 조건 확인
- 작업 폴더 `/data/users/bsseo/gd_lab_vrl_worktrees/bivt-ray-ddp-newcam-17206`
  - branch `codex/bivt-ray-ddp-newcam-17206`, HEAD `b9b5d3a` = `origin/vrl_models`, `git status` 깨끗함.
- GPU 0·1·2: 0 MiB 사용(비어 있음). GPU 3: VLLM PID 329833, 90,800 MiB(손대지 않음).
- 컨테이너 테스트(`PYTHONPATH=src ... pytest -q -p no:cacheprovider tests`):
  `1 failed, 292 passed, 6 skipped` — 실패는 예상대로 `tests/students/test_gavd.py::test_onnx_export`
  (`ModuleNotFoundError: onnxruntime`, 환경 문제).

## 2) 갭 smoke (`GAP_GPUS=0,1,2 bash scripts/run_bivt_gap_finetune_ddp.sh clean smoke`)
- tmux: `bivt_gap_clean_smoke` (종료됨), run id `2026-10-05_22-29-04_gap_clean_ddp3_s42_smoke`
- 콘솔 로그: `logs/gap_clean_ddp3_smoke_console_20261005_222904.log` (작업 폴더 기준)
- run 폴더: `logs/vision_rbq10_dreamwaq/arm_4/2026-10-05_22-29-20_2026-10-05_22-29-04_gap_clean_ddp3_s42_smoke/`
- 결과
  - `EXIT=0`. 3 rank 모두 `[INFO] Training arm: 4`, 각 64 env(총 192, smoke 기본값).
  - 3 rank 모두 `17206_top1.pt` 로드 → 17207~17209 업데이트 3회, `model_17209.pt` 저장.
  - 지형 레벨 floor(mean) 복원(약 8–10).
  - manifest(`params/gap_finetune_manifest.yaml`): `camera_profile: vendor_new`, `distributed: true`,
    `world_size: 3`, `baseline_gate.waived: true`(smoke 사유), `reward_weights`: monitor 1.0 /
    intrusion −3.0 / clean 1.5, checkpoint sha `a70adbda…c326d`, `resumed_iteration: 17207`,
    PPO lr 1e-4(schedule fixed), CENet lr 1e-3, `source_commit b9b5d3a`.
  - 모든 Loss/* 유한(TensorBoard 확인). 동기화: 로드 직후 rank 간 상태 검사 통과
    (주기 검사는 100 update 간격이라 3 update에서는 미실행 — 같은 코드의 간격 4 검사는 이전
    smoke `ddp_sync_smoke_resume_*`에서 3회 통과).
  - GPU 3 전후 90,800 MiB 동일.
- 참고: 반복당 약 3–4 s(192 env).

## 3) 본 학습 (대기 중, 사용자 확인 후 실행)
- 예정 명령(작업 폴더에서, tmux `bivt_gap_clean_ddp3`):
  `GAP_GPUS=0,1,2 GAP_GATE_WAIVER="user waived gate C on 2026-10-05: multi-GPU ARM4 Clean gap fine-tuning from 17206 with vendor_new cameras" bash scripts/run_bivt_gap_finetune_ddp.sh clean train`
- 예상 속도: 이전 3-GPU 4096 env smoke(BIVT-Ray)에서 약 4.7 s/iter → 남은 12,794 업데이트
  (17207→30000) 약 17시간 안팎 예상(갭 모니터 항목 때문에 조금 더 걸릴 수 있음).
- 정지 파일: 런처가 `GD_LAB_STOP_FILE=<작업 폴더>/logs/<run_id>.stop`으로 설정
  (run_id는 시작 시 출력되는 `Gap DDP run=...` 줄에 나옴).

## 확인/문제점
- 기존 선생 보상 `platform_gap_foot_drop`(−2.0)과 `platform_gap_crossing`(+0.5)도 Clean 변형에서
  계속 켜져 있음(`tasks.py` 설계상 "arms differ ONLY in the two weights"). 의도대로면 문제 없음.
- manifest의 `seed`가 43으로 기록됨(launcher seed 42 + local_rank 보정이 기록 시점에 반영된 값으로 보임).
  기능 문제는 아니지만 기록 의미 확인 권장.
- 원본 `/data/users/bsseo/gd_lab_vrl`의 미커밋 GAST 변경 4건은 건드리지 않음.
- 옛 폴더 `bivt-ray-ddp-sync-17206`은 이후 사용하지 않음.

## 추가 (2026-10-05 22:55): 사용자 요청 반영, origin/vrl_models = a8003b8
- `cd174f5` 학습 조건 문서 `docs/training/arm4-bivt-gap-clean-ddp3.md` 추가(+ `docs/teachers/bivt.md` 링크).
  사용자 정정: 17207부터 **30,000회** → `GAP_TARGET_ITERATIONS=47207`(17207–47206), 약 39–42시간 예상.
- `a8003b8` Top-5 G를 **깨끗한 통과율**로 변경(갭 모니터가 있는 태스크만): collector가 reset 직전
  `platform_gap_monitor.tracker.paid`를 읽어 `gap_clean_success` 기록, `score_episodes`가 모든 갭 episode에
  이 값이 있으면 G에 사용. 리더보드에 `gap_success_definition`, `gap_crossing_rate`, `gap_clean_crossing_rate` 기록.
  모니터 없는 태스크는 기존 G 유지. 테스트 5건 추가, 전체 297 passed / 1 failed(onnxruntime).
  `tools/check_conventions.py`는 기존부터 실패(`platform_gap_attempts.py` 319줄, `test_platform_gap_finetune.py`의
  홈 디렉터리 절대경로 문자열) — 이번 변경과 무관, 미수정.
- 새 코드로 Clean smoke 재실행(6 update, 192 env) EXIT=0, GPU 3 불변. (짧아서 Top-5 후보는 생기지 않음)
- 본 학습은 아직 미실행, 사용자 "시작" 대기.
