# gd_lab

RBQ10 사족보행 로봇 강화학습 프레임워크. IsaacSim 5.1.0 + IsaacLab 2.3.1 위에서
blind DreamWaQ(고유수용 감각 기반 rough-terrain 보행)를 학습한다.

## Quick start

Apptainer 컨테이너 기준 워크플로우다. 호스트에는 NVIDIA 드라이버(CUDA 12.8 지원,
570대 이상)와 apptainer만 있으면 되고, IsaacSim을 호스트에 설치하지 않는다.
이미지는 GPU 모델과 무관하다 (드라이버는 실행 시 호스트에서 주입).

이미 IsaacSim 5.1.0 + IsaacLab 2.3.1 conda env가 있는 머신이라면 1~3단계를
건너뛰고 그 env에서 `pip install -e .`만 하면 된다 — 이후 명령은 `run` 접두어
없이 동일하다.

### 1. 클론 + 이미지 빌드 (머신당 한 번)

```bash
git clone <remote> gd_lab && cd gd_lab
apptainer build ~/workspace/gd_lab_isaaclab.sif containers/gd_lab_isaaclab.def
```

이미지에는 버전이 고정된 베이스만 들어간다: CUDA 12.8 + Python 3.11 +
torch 2.7.0(cu128) + IsaacSim 5.1.0. 설계 근거와 상세는
[containers/](containers/README.md). 비루트 빌드는 apptainer 설정에 따라
`--fakeroot`가 필요할 수 있다.

### 2. venv 구성 + 설치 (머신당 한 번)

IsaacLab과 gd_lab은 이미지 밖 호스트 venv에 editable로 산다 — 코드를 고쳐도
이미지 리빌드가 없다. venv는 컨테이너의 python으로 만들어 이미지의
torch/isaacsim을 그대로 본다:

```bash
sif=~/workspace/gd_lab_isaaclab.sif
apptainer exec $sif python3.11 -m venv --system-site-packages ~/workspace/venv

git clone -b v2.3.1 https://github.com/isaac-sim/IsaacLab ~/workspace/IsaacLab
apptainer exec $sif ~/workspace/venv/bin/pip install \
    -e ~/workspace/IsaacLab/source/isaaclab \
    -e ~/workspace/IsaacLab/source/isaaclab_assets \
    -e "$HOME/workspace/IsaacLab/source/isaaclab_rl[rsl-rl]" \
    -e ~/workspace/IsaacLab/source/isaaclab_tasks
# [rsl-rl] extra가 rsl-rl-lib==3.0.1을 핀 버전으로 설치한다 (없으면 학습이 import에서 실패).

# --no-deps: env의 torch/gymnasium/rsl-rl 핀이 움직이면 안 된다.
apptainer exec $sif ~/workspace/venv/bin/pip install -e . --no-deps
```

부가 패키지는 `--no-deps`에 걸리지 않게 필요한 것만 개별 설치한다:

```bash
apptainer exec $sif ~/workspace/venv/bin/pip install wandb   # wandb 로거 사용 시
apptainer exec $sif ~/workspace/venv/bin/pip install pytest              # 4단계 검증에 필요
apptainer exec $sif ~/workspace/venv/bin/pip install ruff import-linter    # 기여자 체크용
```

### 3. 실행 셸 함수

이후 모든 명령은 "컨테이너 안의 venv python"으로 실행한다. 셸에 한 줄 정의:

```bash
run() { OMNI_KIT_ACCEPT_EULA=YES apptainer exec --writable-tmpfs \
        ~/workspace/gd_lab_isaaclab.sif ~/workspace/venv/bin/python "$@"; }
```

`--writable-tmpfs`는 Omniverse 캐시용 임시 오버레이, `OMNI_KIT_ACCEPT_EULA`는
대화형 EULA 프롬프트 생략(비대화형 실행에 필수)이다. 호스트 환경변수는
컨테이너로 그대로 전달되므로 `CUDA_VISIBLE_DEVICES` 등은 앞에 붙이면 된다.

### 4. 설치 검증

```bash
run -m pytest tests -q                                        # 시뮬레이터 불필요, 수 초
GD_LAB_ISAAC_TESTS=1 run -m pytest tests/test_smoke_isaac.py  # 태스크 생성 + 1 iter 학습
```

첫 줄은 RL 스택·관측 계약·export parity를, 둘째 줄은 IsaacSim 포함 전체 경로를
확인한다. 스모크가 green이면 학습 준비 완료.

주의: 셸에 ROS가 source되어 있으면(`PYTHONPATH`에 `/opt/ros/...`) pytest 수집이
깨질 수 있다 → `env -u PYTHONPATH ...`로 실행.

### 5. 학습 → 확인 → 배포

```bash
# GPU 선택 + 학습 (로그: <repo>/logs/blind_rbq10_dreamwaq/<타임스탬프>/)
CUDA_VISIBLE_DEVICES=0 run scripts/train.py \
    --task Gd-Blind-Rbq10-Dreamwaq-v0 --num_envs 4096 --headless --logger tensorboard

# 장기 학습: nohup은 셸 함수를 못 받으므로 bash -c로 감싼다 (ssh 끊겨도 유지)
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES \
    apptainer exec --writable-tmpfs ~/workspace/gd_lab_isaaclab.sif \
    ~/workspace/venv/bin/python scripts/train.py \
    --task Gd-Blind-Rbq10-Dreamwaq-v0 --num_envs 4096 --headless --logger tensorboard \
    --run_name a1' > logs/nohup_a1.log 2>&1 &

# 튜닝 오버라이드는 hydra CLI로 (코드 값 수정 금지 — 규칙 6)
run scripts/train.py env.rewards.base_height.weight=-10.0 agent.max_iterations=20000

# 재개
run scripts/train.py --resume --load_run <run폴더명>

# 학습된 정책 확인 — 실행 시 exported/policy.{pt,onnx} 자동 생성
run scripts/play.py --task Gd-Blind-Rbq10-Dreamwaq-Play-v0

# Xbox 패드 free-play (호스트 패드 장치 + inputs 패키지 필요)
run scripts/play.py --task Gd-Blind-Rbq10-Dreamwaq-Gamepad-v0

# 시뮬레이터 없이 체크포인트만으로 export
run scripts/export.py logs/blind_rbq10_dreamwaq/<run>/model_50000.pt
```

### 6. 기여자 체크 (PR 전)

```bash
ruff check src scripts tests tools
python tools/check_conventions.py
lint-imports
pytest tests -q
```

컨테이너 경로에서는 각 명령을 `apptainer exec <sif> ~/workspace/venv/bin/<명령>`
으로 실행한다 (venv 스크립트의 shebang이 컨테이너 python을 가리키므로 호스트에서
직접 실행되지 않는다).

CI가 같은 검사를 강제한다. 규칙 11에 따라 rsl_rl 코드를 저장소에 복사하면 CI가
실패한다 — RSL-RL은 `rsl-rl-lib==3.0.1`(pip, 무수정)을 그대로 사용하고, 확장은
`src/gd_lab/rl/`의 서브클래스로만 한다.

## 구조

```
src/gd_lab/
  core/      L0  ObsSpec(관측 계약 타입), registry(gym ID 생성), paths
  robots/    L1  RBQ10 정의 (URDF + DelayedDCMotor). task 지식 없음
  mdp/       L1  reward/obs/command/action/curriculum/terrain/symmetry 순수 항들
  managers/  L1  IsaacLab 매니저 확장 (action history)
  devices/   L1  gamepad (play 전용, lazy import)
  rl/        L2  rsl_rl 상속 확장: DreamwaqActorCritic/PPO/Runner + CENet
  methods/   L2  dreamwaq recipe — spec.py가 관측 계약의 유일한 선언
  agents/    L2  runner cfg (레이아웃 값은 전부 spec에서 유도)
  tasks/     L2  blind_rough 환경 조립 (+PLAY/GAMEPAD 변형)
  deploy/    L3  학습 forward를 재사용하는 단일 export 경로 (JIT + ONNX)
scripts/     L4  얇은 엔트리 (train/play/export, cli_args 1벌)
```

의존 방향은 위에서 아래로만(단방향), `import-linter`가 CI에서 강제한다.

## 규칙

1. 의존 방향 단방향. `core`는 아무것도 import하지 않는다.
2. task는 task를, method는 method를 상속하지 않는다. 공유는 조립으로.
3. 한 이름 = 한 정의 (CI가 AST 스캔으로 강제).
4. 관측 레이아웃은 `methods/*/spec.py` 한 곳에서 선언하고 나머지는 유도한다.
   네트워크 생성자·mirror 증강·export가 전부 spec과 대조 검증한다.
5. 모듈 경계는 타입으로 (NamedTuple/dataclass; tuple 인덱스 금지).
6. 튜닝 값 in-place 수정 금지. hydra CLI 오버라이드, 반복되면 `configs/experiment/`.
7. gym ID는 `core/registry.py`로만 생성한다 (CI 강제).
8. 파일 300줄 상한 (CI 강제).
9. export는 학습 forward를 재사용한다. 차원 표 재기술 금지.
10. 주석은 코드가 표현 못 하는 제약·유도만, 최소한으로. 이력·출처는 git과 `docs/`에.
11. rsl-rl-lib 버전 고정, 저장소 내 사본·패치 금지 (CI 강제).

## 테스트

```bash
pytest tests -q                                  # 시뮬레이터 불필요 (CPU)
GD_LAB_ISAAC_TESTS=1 pytest tests/test_smoke_isaac.py   # 학습 머신 (IsaacSim 필요)
```

CPU 테스트만으로 ObsSpec 계약, mirror 순열(involution), CENet/AdaBoot/PPO 학습 루프,
저장/재개, export parity(JIT == act_inference)까지 검증된다. 시뮬레이터 스모크
테스트는 게임패드를 제외한 등록 태스크를 2-env로 생성해 1 iteration 학습을 돌린다.
