# 학습 구조 — 2026-09-30

실행 소스는 `src/gd_lab` 한 곳에서 관리합니다. 인계용 복사본은
`docs/archive/attention_student_handoff`에 보존하며 실행하거나 수정하지 않습니다.

- `rl/`: 공통 PPO, Runner, CENet, blind actor.
- `teachers/cvtt/`: 기존 비전 교사 actor, 환경, agent 설정.
- `teachers/bivt/`: 블라인드에서 초기화하는 비전 교사 및 가시성 변형.
- `students/rvld/`: 기존 학생.
- `students/gavd/`: 격자 Attention 학생, 행동 및 공간 보조 지도.
- `core/`: 카메라 전송·시간·관측 계약 등 교사/학생 공통 기능.
- `deploy/`: 공통 export와 메타데이터.
- `scripts/train_cvtt.py`: 기존 비전 교사 학습.
- `scripts/train_bivt.py`: blind-start 교사 학습.
- `scripts/distill_student.py`: CNN-GRU/Attention 공통 증류 진입점.
- `scripts/export_student.py`: 학생 export.
- `tests/teachers/`, `tests/students/`, `tests/integration/`: 방법별 검사.
- `checkpoints/teachers/blind/arm4_38000/`: 받은 블라인드 모델과 원본 params/SHA/diff.
- `checkpoints/teachers/cvtt/arm4_top5_20260930/`: 기존 비전 Top-5.
- `checkpoints/teachers/bivt/render_v1_final_20260930/`: BIVT-Render v1 최종 `teacher/model_1299.pt` 및 원본 설정·학습 로그 (`c61cd08`에서 수신).

기존 체크포인트의 params와 metadata는 학습 당시 기록이므로 재작성하지 않습니다.
기존 run 로그 안의 절대 경로도 역사적 기록입니다. checkpoint를 불러올 때는 새 경로를 명시하세요.
이번 변경은 가중치와 학습 알고리즘을 바꾸는 작업이 아닙니다.
고정 블라인드 + 비전 행동 보정은 아직 구현하지 않았습니다.
실제 파일 이동 목록과 이동 전후 SHA는 `layout-migration.json`에 있습니다.

## 서버 역할

- 10.254.90.20 (5090): 학습 및 배포 겸용. 학생 및 신규 실험, MuJoCo 검증.
- 10.77.32.231 (다중 GPU): 기존 4카메라 렌더링을 포함하는 비전 교사 학습.
- 머신별 실행 설정은 섞지 않습니다. 구조 변경은 `vrl_models` 브랜치로 공유하며 기존 `main`은 유지합니다.

## 모델 파일 공유 범위

이번 구조 변경 커밋은 모델 가중치를 추가·이동·삭제하지 않습니다. 기존 GitHub 모델은 기존 경로에 유지하며, 새 `checkpoints/teachers/` 경로의 가중치는 로컬 배치입니다. 다른 서버에서는 필요한 가중치를 기존 경로에서 복사하거나 별도로 전달한 뒤 스크립트에 지정하세요. 설정과 메타데이터만으로는 학습·증류를 실행할 수 없습니다.

## 검증

리포 루트를 작업 디렉터리로 하고 `PYTHONPATH=$PWD/src`를 사용합니다.
`python -m pytest tests/students tests/teachers tests/integration tests/test_camera_transport.py -q`.
새 위치의 소스로 기존 checkpoint를 strict load하고 export parity를 확인한 뒤 학습하세요.
