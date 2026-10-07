# gd_lab_vrl

RBQ10 비전 보행 학습 저장소다. 블라인드 정책에서 지형 교사(BIVT-Ray, GAST)를 PPO로 학습하고,
카메라 학생(GAST, RVLD, GAVD)으로 증류한다. 블라인드 학습은 `gd_lab`, 배포는 `gd_rbq10_deploy_vrl`에 있다.

| 약어 | 전체 이름 | 역할 |
|---|---|---|
| BIVT-Ray | Blind-Initialized Visual Teacher, Raycast Visibility | 블라인드 정책에서 시작한 카메라 가시 높이맵 교사 |
| GAST | Gap-Aware Spatiotemporal Teacher-Student | 전체 높이맵 + 8시점 기억 교사(GAST-T)와 시공간 기억 학생(GAST-S) |
| RVLD | Recurrent Visual Latent Distillation | 카메라별 CNN + GRU 학생 |
| GAVD | Grid-Attention Visual Distillation | 지형 격자가 영상에 attention하는 학생 |

현재 계보: DWB `d_v3.6.21_b1_18` → BIVT-Ray 17206 → BIVT-Ray Clean 21068 → GAST 교사 v2.1(학습 중).
현재 학생: GAST 19840, RVLD 19008, GAVD 19008 (모두 교사 21068).

## 실행

사용자가 실행하는 것은 `experiments/<방법>/`의 셸 스크립트다. Python 진입점은 `scripts/`(BIVT-Ray, RVLD, GAVD)와
`gast/scripts/`(GAST)에 있다. 목록과 용도는 [experiments/README.md](experiments/README.md).

```bash
bash experiments/bivt/run_bivt_gap_v2_ddp.sh <clean_checkpoint.pt> <sha256> smoke|train     # BIVT-Ray v2 (3 GPU)
bash experiments/gast/train_teacher_gapclean_from_bivt_3gpu.sh smoke|train <bivt_teacher.pt>  # GAST 교사 (21068 warm start)
bash experiments/gast/train_bivt21068_gapclean_env512_bptt16.sh                               # GAST 학생 (교사 21068)
```

## 구조

```
experiments/{bivt,gast,rvld,gavd}/   실행 셸 스크립트
scripts/                             Python 진입점 (train_bivt.py, distill_student.py, export_*, play_*)
src/gd_lab/                          BIVT-Ray · RVLD · GAVD 코드 (core, rl, mdp, teachers, students, deploy)
gast/                                GAST 코드: src/gd_lab (gd_lab 사본 + gast 패키지), scripts/, tests/, configs/
configs/  tests/  tools/  assets/  containers/  docs/
checkpoints/                         교사·학생 묶음: 메타데이터(README, params, SHA256)만 git, 가중치는 git 밖
logs/                                학습 로그 (git 밖)
archive/                             CVTT, BAVRL, vendor_legacy 실험 스크립트와 예전 문서
```

**`gd_lab` 패키지가 두 벌이다.** 루트 `src/gd_lab`(BIVT-Ray, RVLD, GAVD)과 `gast/src/gd_lab`(GAST)이 갈라져 있다.
PPO의 다중 GPU 동기화 방식과 Top-5 기본값 등이 서로 다르다(13개 파일, 약 540줄). 합치면 학습 동작이 바뀌므로
별도 작업으로 검증하며 합친다. 그 전까지 GAST 실행은 `gast/`를 작업 폴더로 쓴다(`experiments/gast/*.sh`가 처리).

## 체크포인트

`checkpoints/<teachers|students>/<방법>/<묶음>/` 아래 메타데이터만 git에 있다. 가중치(`*.pt`, `*.onnx`)는
SHA256SUMS.txt로 확인하며 서버 간에 직접 복사한다. 전체 가중치가 있는 10/07 사본: `~/gd_project/archive/gd_lab_vrl_20261007/`.

## 문서

- [RBQ10 교사·학생 정책 계보](docs/lineage/RBQ10_policy_lineage.md)
- [개 보행 인사이트로 본 교사 보상 — Claude·Codex 교차검토 결과 (2026-10-07)](docs/teachers/gast_v21_reward_dog_insights_20261007.md)
- [RL 롤아웃 분석 도구 설계](tools/rl_replay/README.md)

## 설치

Apptainer 이미지(IsaacSim 5.1.0 + IsaacLab 2.3.1)와 호스트 venv 기준이다. 절차는 [docs/history/README_before_reorg_20261007.md](docs/history/README_before_reorg_20261007.md)의 Quick start를 따른다.

```bash
run() { OMNI_KIT_ACCEPT_EULA=YES apptainer exec --writable-tmpfs \
        ~/workspace/gd_lab_isaaclab.sif ~/workspace/venv/bin/python "$@"; }
run -m pytest tests -q            # 루트 코드 단위 테스트
cd gast && run -m pytest tests -q # GAST 단위 테스트
```

10/07 재편 이전 README와 구조 문서는 `docs/history/`에 있다.
