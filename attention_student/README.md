# Attention 학생 증류 인계본

Top-1 6987 교사를 고정한 `grid_attention_v1` 학생 증류의 소스 스냅샷입니다.
기존 루트 소스를 변경하지 않고 공유하기 위해 신규 소스와 기존 파일 수정본을 별도로 보관합니다.
이 디렉터리는 **독립 실행 패키지가 아닙니다**. 실행 전에 아래 적용 절차를 따르세요.

## 구조와 적용 경로

| 인계본 경로 | 기존 리포 기준 적용 경로 |
| --- | --- |
| `src/gd_lab/rl/perception_attention.py` | `src/gd_lab/rl/perception_attention.py` |
| `src/gd_lab/rl/attention_distillation.py` | `src/gd_lab/rl/attention_distillation.py` |
| `scripts/train_attention_5090.sh` | `scripts/train_attention_5090.sh` (5090 로컬 전용) |
| `tests/test_perception_attention.py` | `tests/test_perception_attention.py` |
| `docs/training/attention-student-5090.md` | 참고 문서 |
| `integration/scripts/train_perception.py` | `scripts/train_perception.py` |
| `integration/scripts/export_student_vrl.py` | `scripts/export_student_vrl.py` |
| `integration/src/gd_lab/deploy/export_student_vrl.py` | `src/gd_lab/deploy/export_student_vrl.py` |

`integration/`은 기존 소스의 수정된 전체 복사본입니다. 자동 덮어쓰기하지 말고 대상 리포와 diff를 검토하여 병합하세요.
복사 기준 HEAD는 `e6b8f00`입니다. 독립 실험 `blind_start/`와는 다른 방식이며 그 코드를 수정하지 않습니다.

## 적용과 실행

1. 대상 리포의 사용자 변경을 보존하고 별도 작업 브랜치에서 시작합니다.
2. 위 표에 따라 신규 소스를 반영하고 integration 수정본을 비교·병합합니다.
3. 원본 리포 루트에서 관련 pytest 및 ruff, checkpoint/export 호환성을 검증합니다.
4. 교사 체크포인트와 환경을 준비한 뒤 학습합니다. 이 폴더 안에서 실행 스크립트를 직접 실행하지 마세요.

`train_attention_5090.sh`는 `/home/user`의 Apptainer 환경을 사용하는 단일 GPU 실행용입니다.
학습 main 서버 `10.77.32.231`의 다중 GPU 실행 스크립트에 합치거나 그대로 실행하지 마세요.
배포 main 서버는 `10.254.90.20`이며 배포·MuJoCo 실험은 배포 리포에서 수행합니다.

## 모델 계약

- 학생 구조: 공간 CNN + 11×17 격자 cross-attention + GRU → latent 32.
- 입력 frames `[1,4,2,45,80]`, hidden `[1,64]`.
- hidden[63]은 프레임 나이(초, 0~1 제한), hidden[0:63]만 recurrent memory입니다.
- 교사 6987 SHA256: `c153e0d3d18ce023b6b1dba4651cf9e01d3eb38b62a651c8ebeb2bb5f3236edd`.
- 학생과 대응 교사 actor를 반드시 함께 사용하고 배포 frame-age 계약을 검증하세요.
- 가중치·로그·영상·빌드 산출물은 포함하지 않습니다.

기존 학습 문서는 당시 실행 기록을 보존한 문서이므로 launch/production-runtime 상태 문장은 현재 상태를 보장하지 않습니다.
이 소스 공유는 보행 성능 검증 완료나 학습 실행을 의미하지 않습니다.
