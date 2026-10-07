# 보관

현재 학습 경로에서 빠진 실험 스크립트와 문서다. 경로는 재편 전 기준이라 그대로는 실행되지 않는다.
다시 돌려야 하면 태그 `before-reorg-20261007`을 `git worktree`로 꺼내거나 `~/gd_project/archive/gd_lab_vrl_20261007/`을 쓴다.

- `cvtt/`: CVTT 교사와 그 학생(RVLD/GAVD/GAST) 학습·복구 스크립트
- `bavrl/`: BAVRL 학습·export
- `legacy/`: vendor_legacy 카메라 BIVT-Ray 4500/7986 학생 스크립트
- `docs/`: 9/30 attention 학생 인계 문서

`src/gd_lab`의 CVTT·BAVRL 모듈은 BIVT-Ray가 같은 클래스(`teachers/cvtt/actor_critic.py` 등)를 쓰므로 그대로 둔다.
