# 실행 스크립트

저장소 어디서 실행해도 된다(각 스크립트가 저장소 루트를 작업 폴더로 쓴다).

| 방법 | 스크립트 | 용도 |
|---|---|---|
| BIVT-Ray | `bivt/run_bivt.sh <gpu> <log> <args...>` | `scripts/train_bivt.py` 범용 실행 |
| BIVT-Ray | `bivt/run_bivt_onnx.sh`, `bivt/run_bivt_ray_enhanced.sh` | 블라인드 ONNX에서 BIVT-Ray 시작 |
| BIVT-Ray | `bivt/run_bivt_gap_finetune.sh`, `bivt/run_bivt_gap_finetune_ddp.sh` | 갭 파인튜닝 (Clean run, 단일/3 GPU) |
| BIVT-Ray | `bivt/run_bivt_gap_v2_ddp.sh` | v2 보상 검증 학습 (31625 → 36201, 10/07 중단) |
| GAST 교사 | `gast/train_teacher_gapclean_from_bivt_3gpu.sh` | BIVT-Ray 21068 warm start + v2.1 (현재 서버 학습) |
| GAST 교사 | `gast/train_teacher_scratch_v2_3gpu.sh`, `gast/train_teacher_3gpu.sh` | 처음부터 학습 (실험 3) |
| GAST 학생 | `gast/train_bivt21068_gapclean_env512_bptt16.sh` | 교사 21068 → GAST 학생 (19840) |
| GAST 학생 | `gast/train_gast19840_gapfocus.sh` | 19840 갭 집중 파인튜닝 (24960) |
| GAST 학생 | `gast/train_17206_gast_stage1.sh` | 교사 17206 (vendor_legacy) |
| RVLD | `rvld/train_bivt_ray17206_rvld.sh` | 교사 17206 템플릿. 21068 학생(19008)은 90.111에서 학습 |
| GAVD | `gavd/train_bivt_ray17206_gavd.sh` | 교사 17206 템플릿. 21068 학생(19008)은 90.111에서 학습 |
| GAST 학생 | `gast/train_gastv21_student_live.sh <teacher_package>` | GAST 교사(v2.1) → GAST 학생 |
| RVLD | `rvld/train_gast_teacher_rvld.sh <teacher_package>` | GAST 교사 → RVLD 학생 (`common/distill_from_gast_teacher.sh`) |
| GAVD | `gavd/train_gast_teacher_gavd.sh <teacher_package>` | GAST 교사 → GAVD 학생 |
