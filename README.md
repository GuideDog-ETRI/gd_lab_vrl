# gd_lab_vrl

RBQ10 사족보행 로봇 강화학습 프레임워크. IsaacSim 5.1.0 + IsaacLab 2.3.1 위에서
DreamWaQ 및 카메라 기반 VRL 선생·학생을 학습한다.

## Arm4 VRL: 선생 PT → 학생 PT → ONNX → MuJoCo

배포 저장소는 [gd_rbq10_deploy_vrl](https://github.com/GuideDog-ETRI/gd_rbq10_deploy_vrl/tree/vrl)이다.
이 절의 명령은 이 저장소 루트에서 실행한다. 아래 Quick start는 환경 설치와 blind 경로를 설명한다.
진행 중인 선생/학생 작업이 있으면 새 학습을 중복 실행하지 않는다.

| 파일 | 용도 |
|---|---|
| `model_3700.pt` | 선생 PPO 체크포인트. 증류의 고정된 정답 및 배포 actor의 출처 |
| `perception_N.pt` | 학생 CNN·GRU 체크포인트. 카메라 입력을 선생 지형 latent로 변환 |
| `policy_vrl.onnx` | 선생에서 내보낸 actor·CENet. 학생 latent를 입력으로 받음 |
| `policy_vrl_student.onnx` | 선택한 학생에서 내보낸 카메라 인코더 |
| export 폴더의 `policy_vrl*.pt` | TorchScript 추론 파일. 원래 학습 체크포인트와 구별 |

### 1. 선생 학습과 체크포인트 선택

```bash
./scripts/start_vrl_3gpu_tmux.sh 4 40000
```

GPU 0·1·2가 총 4096 env로 하나의 정책을 학습하며 100 iteration마다 저장한다.
Arm4는 제어 100 Hz, Kp hip/thigh 123.39·knee 127.77, Kd 2.4이다.
선생의 원래 실행 폴더와 `params/agent.yaml`, `params/env.yaml`을 함께 보관한다.
Top5는 해당 실행의 `best_top5/`와 `leaderboard.json`에 저장되며, gap 평균 난이도 8 이상을
포함한 온라인 선정 조건을 적용한다. 별도 평가로 선정한 최적 모델이라는 의미는 아니다.

이후 예시는 아래 선생을 고정해서 사용한다. 다른 선생으로 바꿀 때는 증류·평가·export 모두 같이 바꾼다.

```bash
teacher_run=2026-09-26_17-03-13_arm4_3gpu_top5_resume_model3000
teacher_dir="$PWD/logs/vision_rbq10_dreamwaq/arm_4/$teacher_run"
```

### 2. 학생 증류

아래 전용 실행기는 위 `model_3700.pt`를 고정해 사용한다. 인자는 GPU 번호·캡처 횟수·env 수이다.
이 서버에서 64 env 검증 시 학생 프로세스는 GPU 약 7.5 GiB, RAM 약 15.7 GiB를 사용했다.
다른 머신의 용량·속도는 짧은 실행으로 먼저 측정한다.

```bash
./scripts/start_arm4_student_tmux.sh 0 20000 64
tmux attach -t vrl_arm4_student3700
```

Ctrl-b 다음 d로 화면에서 나가도 학습은 계속된다. 학생은 200회 캡처마다 저장한다.
`--iterations`는 누락을 포함한 카메라 캡처 시도 횟수이며 PPO iteration과 다르다.
기본 증류 조건은 70~100 ms 캡처 간격, 0~50 ms 전송 지연, 5% 프레임 세트 누락이다.
영상과 선생 정답은 촬영 시점으로 짝지으며, 리셋 이전/역순 도착 패킷은 폐기한다.
명목 카메라 계약은 Arm4의 8 제어 스텝·80 ms이며, 시간 변동 설정은 학생 PT에 별도로 기록한다.

### 3. 학생으로 보행 평가

다음 함수는 이 서버와 같은 Apptainer 설치를 사용한다. 다른 PC에서는 두 경로를 조정한다.

```bash
gd_sif="${GD_LAB_SIF:-$HOME/workspace/gd_lab_isaaclab.sif}"
gd_python="${GD_LAB_PYTHON:-$HOME/workspace/venv/bin/python}"
mkdir -p logs/usd_tmp/student_readme
vrl_run() {
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES="${VRL_GPU:-0}" TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES \
    apptainer exec --nv --writable-tmpfs --bind "$PWD/logs/usd_tmp/student_readme:/tmp/IsaacLab" \
    "$gd_sif" "$gd_python" "$@"
}
student_run="$(cat logs/arm4_student3700_latest_run.txt)"
student_pt="$PWD/logs/vision_rbq10_dreamwaq/arm_4/$student_run/perception_20000.pt"
vrl_run scripts/play_student.py --headless --device cuda:0 --num_envs 20 \
  --load_run "$teacher_run" --checkpoint model_3700.pt --student_checkpoint "$student_pt" \
  --max_steps 20000 --diag_every 25 \
  --camera_interval_ms 70 100 --camera_delay_ms 0 50 --camera_drop_prob 0.05
```

`perception_20000.pt`는 완료 후 생기는 예시다. 같은 선생에 대해 여러 학생 저장본을 비교해 선택한다.
기본 주기 비교는 `--camera_interval_ms 80 80 --camera_delay_ms 0 0 --camera_drop_prob 0`으로 실행한다.
증류 MSE와 함께 지형별 낙상·gap 통과·속도 추종을 평가한다. 지연된 latent를 쓰는 actor의 성능은
이 보행 평가로 확인해야 하며, 증류 MSE 감소만으로 배포 성능을 확정하지 않는다.

### 4. 선생 actor와 학생을 함께 ONNX로 내보내기

앞 단계의 변수/함수를 정의한 셸에서 실행한다. export에는 시뮬레이터 실행이 필요 없다.

```bash
export_dir="$PWD/exported/arm4_teacher3700"
vrl_run scripts/export_vrl.py "$teacher_dir/model_3700.pt" --out "$export_dir"
vrl_run scripts/export_student_vrl.py "$student_pt" --actor-onnx "$export_dir/policy_vrl.onnx"
sha256sum "$export_dir/policy_vrl.onnx" "$export_dir/policy_vrl_student.onnx" > "$export_dir/SHA256SUMS.txt"
```

학생 ONNX는 배우는 동안 사용한 선생 actor와 한 쌍이다. 다른 선생의 actor와 섞지 않는다.
현재 VRL exporter는 완전한 주기·게인·보정 계약을 ONNX에 내장하지 않으므로
선생/학생 PT, 선생 params, 코드 커밋, 카메라 계약과 실행 조건도 함께 보관한다.

### 5. 다른 PC로 전달하고 MuJoCo 실행

대상 PC의 배포 저장소에 `resources/policy/vrl/arm4_teacher3700/` 폴더를 만든 뒤 복사한다.
아래 사용자·주소·경로는 대상 PC에 맞춰 바꾼다.

```bash
scp "$export_dir/policy_vrl.onnx" "$export_dir/policy_vrl_student.onnx" \
  user@target-host:/absolute/path/gd_rbq10_deploy_vrl/resources/policy/vrl/arm4_teacher3700/
```

대상 PC에서는 배포 README의 의존성 설치·빌드를 마친 뒤 다음을 실행한다.

```bash
RBQ_WALK=ours RBQ_POLICY_FILE=vrl/arm4_teacher3700/policy_vrl.onnx \
  RBQ_SIM_VISION=1 bash scripts/run_sim_vrl.sh
```

원본 PT를 Pilot에 직접 넣지 않는다. 두 ONNX의 sibling 파일명을 유지하고,
Pilot에서 학생 로드·카메라 수신·latent 갱신을 확인한다. 배포 코드 수정 후에는 재빌드가 필요하다.
다른 PC에서 **증류도** 하려면 ONNX 두 개 대신 학습 저장소·assets·선생 PT/params·Isaac Lab 환경이 필요하다.
설정, 검증 범위와 리소스 측정은 [Arm4 학생 증류 안내](docs/training/arm4-student-deployment.md)를 참고한다.

## Quick start

Apptainer 컨테이너 기준 워크플로우다. 호스트에는 NVIDIA 드라이버(CUDA 12.8 지원,
570대 이상)와 apptainer만 있으면 되고, IsaacSim을 호스트에 설치하지 않는다.
이미지는 GPU 모델과 무관하다 (드라이버는 실행 시 호스트에서 주입).

이미 IsaacSim 5.1.0 + IsaacLab 2.3.1 conda env가 있는 머신이라면 1–3단계를
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
    --task Gd-Blind-Rbq10-Dreamwaq-v0 --num_envs 4096 --max_iterations 40000 --headless --logger tensorboard

# 장기 학습: nohup은 셸 함수를 못 받으므로 bash -c로 감싼다 (ssh 끊겨도 유지)
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES \
    apptainer exec --writable-tmpfs ~/workspace/gd_lab_isaaclab.sif \
    ~/workspace/venv/bin/python scripts/train.py \
    --task Gd-Blind-Rbq10-Dreamwaq-v0 --num_envs 4096 --max_iterations 40000 --headless --logger tensorboard \
    --run_name a1' > logs/nohup_a1.log 2>&1 &

# 튜닝 오버라이드는 hydra CLI로 (코드 값 수정 금지 — 규칙 6)
run scripts/train.py env.rewards.base_height.weight=-10.0 agent.max_iterations=20000

# Pulse 없는 4개 학습 arm: 1/3은 50 Hz, 2/4는 100 Hz, 모두 payload 포함
# 1/2는 현재 gain, 3/4는 kp=123.39/127.77, kd=2.4
TRAIN_ARM=1 run scripts/train.py --max_iterations 40000 --headless --logger tensorboard

# 재개: --max_iterations는 추가 실행 횟수 (계산법은 아래 학습 전달서 참고)
run scripts/train.py --resume --load_run <run폴더명> --max_iterations <남은횟수>

# 학습된 정책 확인 — 실행 시 exported/policy.{pt,onnx} 자동 생성
run scripts/play.py --task Gd-Blind-Rbq10-Dreamwaq-Play-v0

# Xbox 패드 free-play (호스트 패드 장치 + inputs 패키지 필요)
run scripts/play.py --task Gd-Blind-Rbq10-Dreamwaq-Gamepad-v0

# 시뮬레이터 없이 체크포인트 + 같은 run의 params/agent.yaml로 export
run scripts/export.py logs/blind_rbq10_dreamwaq/<run>/model_39999.pt
```

Arm 설정과 재개/play 방법은 [configs/experiment/](configs/experiment/README.md).
Arm을 선택한 실행은 `logs/blind_rbq10_dreamwaq/arm_N/` 아래에 기록된다.
원격 머신에서 네 arm을 각각 40,000 iteration 실행하는 절차는
[학습 전달서](docs/training/four-arms-40000.md)에 정리되어 있다.

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
