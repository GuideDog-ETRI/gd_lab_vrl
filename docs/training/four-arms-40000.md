# ARM 1~4: 각 40,000 iteration 학습 전달서

## 목표와 전달 범위

동일한 코드 버전으로 ARM 1~4를 각각 처음부터 **40,000 iteration** 학습한다.
완료 후 각 arm의 체크포인트, 최종 설정, 학습 로그, ONNX export를 전달한다.
이 문서는 원격 학습 담당자 또는 학습을 실행하는 에이전트가 따라 할 실행 절차다.

기능 구현 기준 커밋은 `25161ba`다. 전달자는 **이 문서까지 커밋하고 원격에
반영한 최종 commit SHA**와 이 문서 경로를 함께 전달한다. 학습 담당자는
학습 중 코드를 변경하지 않고, 실제 사용한 SHA를 결과에 기록한다.

## 실험 조건

| TRAIN_ARM | 정책 주기 | HIP/THIGH kp | KNEE kp | HIP/THIGH kd | KNEE kd |
| --- | --- | --- | --- | --- | --- |
| 1 | 50 Hz | 88.1367 | 102.2177 | 1.9919 | 1.9932 |
| 2 | 100 Hz | 88.1367 | 102.2177 | 1.9919 | 1.9932 |
| 3 | 50 Hz | 123.39 | 127.77 | 2.4 | 2.4 |
| 4 | 100 Hz | 123.39 | 127.77 | 2.4 | 2.4 |

설정 원본은 [`configs/experiment/arm_N.yaml`](../../configs/experiment/)이다.
arm 1·2는 로봇 기본 gain을 상속하므로, 전달받은 코드 버전과 위 표를 함께 확인한다.

네 arm의 공통 조건:

- Command pulse OFF, `pulse_prob_schedule` OFF. 나머지 커리큘럼은 유지한다.
- Payload 이벤트 ON: 에피소드마다 **0 kg 20% / 6 kg 80%**.
- Payload 장착 위치: trunk 좌표계에서 `x ∈ [-0.05, 0.05] m`, `y=0`, `z=0.07 m`.
- 표의 gain을 기준으로 기존 reset 시 gain 랜덤화 **0.85~1.15배**를 유지한다.
- 일반·역계단은 각각 기존 폭 25 cm / 높이 5~20 cm를 비율 0.1로 유지하고,
  폭 30 cm / 높이 5~25 cm인 `pyramid_stairs_wide`·`pyramid_stairs_inv_wide`를
  비율 0.1씩 추가한다. 전체 지형은 10종, proportion 합은 1.0이다.
- `feet_cadence_overrun`의 상한은 **swing 0.4 s / stance 0.6 s**다.
  Weight -1.0과 발별 overrun cap 0.5를 유지한다.
- 물리 `dt=0.005 s`. 정책 decimation은 50 Hz에서 4, 100 Hz에서 2다.
- PPO rollout 100 steps/env, gamma, 관측 history 등 나머지 학습 설정은 그대로다.
  따라서 같은 iteration 수여도 100 Hz arm의 시뮬레이션 시간은 50 Hz arm의 절반이다.

아래 명령의 운용 기본값은 **4096 env, seed 42, TensorBoard, 저장 간격 1000**이다.
GPU 메모리에 맞춰 env 수를 조정해야 한다면 네 arm에 같은 값을 적용하고 기록한다.
임의의 reward/gain/커리큘럼 오버라이드는 추가하지 않는다.

계단 분할과 cadence 변경은 `25161ba` 이후의 변경이다. 학습 머신에는 이 변경과
문서가 모두 포함된 최종 코드 버전을 전달해야 한다.

## 1. 원격 학습 머신 준비

전달자가 원격 저장소에 코드를 올린 다음, 학습 머신의 `gd_lab` 루트에서 실행한다.
아래는 `main` 기준이며, 전달받은 최종 SHA와 `git rev-parse HEAD`가 같은지 확인한다.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git merge-base --is-ancestor 25161ba HEAD
git status --short
git rev-parse HEAD
nvidia-smi
```

작업 트리가 깨끗하고 코드 버전이 일치한 상태에서 진행한다. 설치 방법은
[README](../../README.md)의 Quick start를 따른다. 필요한 버전은
IsaacSim 5.1.0, IsaacLab 2.3.1, `rsl-rl-lib==3.0.1`이다.

실행 환경에 맞는 `train_python` 정의 **하나**를 선택한다.

**기존 conda 환경:** IsaacLab 환경을 활성화한 뒤 정의한다.

```bash
train_python() { env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES python "$@"; }
```

**Apptainer:** README의 이미지·venv 경로를 사용하는 경우다. 실제 경로가 다르면 수정한다.

```bash
train_python() {
    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
        apptainer exec --nv --writable-tmpfs \
        "$HOME/workspace/gd_lab_isaaclab.sif" \
        "$HOME/workspace/venv/bin/python" "$@"
}
```

이하 명령은 같은 셸에서 실행한다. 셸을 다시 열면 함수와 `GD_LAB_LOG_ROOT`를 다시 설정한다.

```bash
train_python -m pip install -e . --no-deps
train_python -c 'import yaml, tensorboard; from importlib.metadata import version; print(version("rsl-rl-lib"))'
train_python -m pytest tests -q
GD_LAB_ISAAC_TESTS=1 train_python -m pytest tests/test_smoke_isaac.py -q
```

## 2. 실행 기록과 원격 GPU 점검

본학습 결과를 저장할 경로를 만들고 출력된 경로를 기록한다.

```bash
export GD_LAB_LOG_ROOT="$PWD/logs/arms_40k_seed42_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$GD_LAB_LOG_ROOT"
git rev-parse HEAD > "$GD_LAB_LOG_ROOT/source_commit.txt"
nvidia-smi > "$GD_LAB_LOG_ROOT/gpu.txt"
printf '%s\n' "$GD_LAB_LOG_ROOT"
```

기존 모델의 학습·export·배포 검증은 32 env × 3 iteration이었다. 계단·cadence
변경 후에는 네 arm 각각 2 env × 1 iteration을 포함한 IsaacLab 검사 19개를
통과했다. 본학습 전에 사용할 GPU에서 4096 env로 짧게 실행해 메모리와 처리량을
확인한다. 아래 로그는 본학습과 분리된다.

```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_ARM=1 \
    GD_LAB_LOG_ROOT="$GD_LAB_LOG_ROOT/preflight" \
    train_python scripts/train.py --headless --device cuda:0 \
    --num_envs 4096 --seed 42 --max_iterations 3 --logger tensorboard \
    --run_name preflight
```

프로세스 정상 종료, 체크포인트 생성, NaN/Inf 및
`Deployment metadata unavailable` 경고가 없는지 확인한다.
이 preflight 체크포인트는 본학습의 초기 가중치로 사용하지 않는다.

## 3. 본학습 실행

공통 실행 함수를 정의한다. 각 arm은 독립적인 초기 가중치로 시작한다.

```bash
train_one_arm() {
    local arm="$1"
    local gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" TRAIN_ARM="$arm" \
        train_python -u scripts/train.py \
        --task Gd-Blind-Rbq10-Dreamwaq-v0 \
        --headless --device cuda:0 --num_envs 4096 --seed 42 \
        --max_iterations 40000 --logger tensorboard \
        --run_name seed42_40k agent.save_interval=1000
}
export -f train_python train_one_arm
```

`CUDA_VISIBLE_DEVICES`로 물리 GPU를 선택한 뒤 프로세스 내부에서는 `cuda:0`을 쓴다.
GPU 배치에 따라 아래 **A 또는 B 중 하나만** 실행한다.

### A. GPU 한 장에서 순차 실행

GPU 0에서 arm 1 → 2 → 3 → 4 순으로 실행한다. 한 arm이 실패하면 다음 arm으로
넘어가지 않고 중단한다. SSH가 끊겨도 실행은 유지된다.

```bash
nohup bash -c '
    set -euo pipefail
    for arm in 1 2 3 4; do
        train_one_arm "$arm" 0 > "$GD_LAB_LOG_ROOT/arm_${arm}.console.log" 2>&1
    done
' > "$GD_LAB_LOG_ROOT/launcher.log" 2>&1 &
echo "$!" > "$GD_LAB_LOG_ROOT/launcher.pid"
```

### B. GPU 네 장에서 병렬 실행

사용 가능한 GPU 0~3에 하나씩 배치한다. GPU 번호는 머신 상황에 맞게 바꾼다.

```bash
for arm in 1 2 3 4; do
    gpu=$((arm - 1))
    nohup bash -c 'train_one_arm "$1" "$2"' _ "$arm" "$gpu" \
        > "$GD_LAB_LOG_ROOT/arm_${arm}.console.log" 2>&1 &
    echo "$!" > "$GD_LAB_LOG_ROOT/arm_${arm}.pid"
done
```

두 방식 모두 학습 결과는 다음 구조다.

```text
<GD_LAB_LOG_ROOT>/
├── source_commit.txt
├── gpu.txt
├── arm_1.console.log ... arm_4.console.log
└── blind_rbq10_dreamwaq/
    ├── arm_1/<timestamp>_seed42_40k/
    ├── arm_2/<timestamp>_seed42_40k/
    ├── arm_3/<timestamp>_seed42_40k/
    └── arm_4/<timestamp>_seed42_40k/
        ├── params/env.yaml
        ├── params/agent.yaml
        ├── events.out.tfevents.*
        ├── model_0.pt, model_1000.pt, ...
        └── model_39999.pt
```

## 4. 모니터링과 재개

```bash
tail -f "$GD_LAB_LOG_ROOT/arm_1.console.log"
nvidia-smi
```

Iteration 증가, 처리량, reward/episode length와 loss 추이를 확인한다. 저장된
`params/env.yaml`의 decimation, gain, pulse 설정, payload 이벤트가 위 조건과
일치하는지도 확인한다. 프로세스가 실행 중이라는 사실만으로 완료 처리하지 않는다.

**재개 시 `--max_iterations`는 목표 총량이 아니라 추가 학습 횟수다.**
체크포인트의 `iter`는 0부터 시작하므로 남은 횟수는 `40000 - (iter + 1)`이다.
다음은 arm 2를 재개하는 예이며, 실제 run 폴더와 체크포인트를 넣는다.
같은 arm의 학습 프로세스가 종료된 것을 확인한 다음 실행한다.

```bash
arm=2
run_dir="$GD_LAB_LOG_ROOT/blind_rbq10_dreamwaq/arm_2/실제_run_폴더명"
checkpoint="$run_dir/model_12000.pt"
remaining=$(train_python - "$checkpoint" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
remaining = 40000 - (int(checkpoint["iter"]) + 1)
if remaining <= 0:
    raise SystemExit("이미 40,000 iteration을 완료한 체크포인트입니다.")
print(remaining)
PY
) || exit 1

CUDA_VISIBLE_DEVICES=0 TRAIN_ARM="$arm" \
    train_python -u scripts/train.py --headless --device cuda:0 \
    --num_envs 4096 --seed 42 --logger tensorboard \
    --resume --load_run "$(basename "$run_dir")" --checkpoint "$(basename "$checkpoint")" \
    --max_iterations "$remaining" --run_name seed42_40k_resume agent.save_interval=1000
```

예를 들어 `iter=12000`이면 추가 횟수는 **27,999**다. 재개 시 새 run 폴더가
생성되므로, 원래 폴더와 재개 폴더를 모두 기록한다. 위 재개 명령은 전경 실행이므로
장시간 실행할 때는 tmux 안에서 실행하거나 본학습과 같이 nohup으로 감싼다.

## 5. 완료 확인과 export

새 학습의 마지막 iteration은 `39999`이며 최종 파일은 **`model_39999.pt`**다.
정상 종료와 함께 체크포인트의 `iter == 39999`, 학습 loss의 유한성,
배포 메타데이터 포함 여부를 확인한다. 아래의 `run_dir`를 arm별 최종 폴더로 바꾼다.

```bash
run_dir="$GD_LAB_LOG_ROOT/blind_rbq10_dreamwaq/arm_1/실제_최종_run_폴더명"
train_python - "$run_dir/model_39999.pt" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert checkpoint["iter"] == 39999
assert all(torch.isfinite(v).all() for v in checkpoint["model_state_dict"].values())
context = checkpoint["infos"]["gd_lab"]["deploy_context"]
assert any(term["source"] == "payload" for term in context["terms"])
print("complete:", checkpoint["iter"] + 1, "iterations; policy_dt:", context["policy_dt"])
PY
train_python scripts/export.py "$run_dir/model_39999.pt"
```

Export에는 같은 run의 `params/agent.yaml`이 필요하다. 체크포인트를 다른 위치로
옮겼다면 `--agent-config /실제/경로/agent.yaml`을 지정한다. 결과는
`exported/policy.pt`, `exported/policy.onnx`, `exported/deploy.json`이다.

배포 검증 환경에서는 `gd_rbq10_deploy`의 `policy-check`에 export 폴더를 전달해
각 arm의 주기·gain·관측 규격과 추론을 확인한다. 현재 번들 SDK/DDS의 검증은
Ubuntu 24.04에서 수행했다. 배포 코드 `a1b4f6f`에서는 `configs/walk.env`의
`RBQ_PAYLOAD_KG`가 정책으로 전달되며, 환경변수가 파일 값보다 우선한다.
기존 3-iteration 모델로 0/6 kg 입력과 ONNX 계약의 `kg × 0.2` 변환을 확인했다.
새 본학습 결과도 같은 검사를 다시 수행한다. 신규 arm은 6 kg으로 학습하므로
배포 관측 1.2가 정상이며, 이전 정책의 5 kg 학습 조건과 구분한다.

검사를 통과한 arm별 export를 `gd_rbq10_deploy/resources/policy/arm_N/`처럼
따로 배치하고, `walk.env`에 `RBQ_WALK=ours`, `RBQ_POLICY_OURS=arm_N`,
`RBQ_PAYLOAD_KG=실제_적재량`을 지정한다. 런타임은 ONNX 내부 계약을 읽고,
`deploy.json`은 함께 보관한다. 설정 변경 후 Pilot을 다시 시작해야 반영된다.

## 전달할 결과

- 각 arm의 완료 여부, commit SHA, seed, env 수, GPU와 최종 run 경로.
- `model_39999.pt`, `params/env.yaml`, `params/agent.yaml`, TensorBoard 및 콘솔 로그.
- `exported/` 전체와 배포 검사 결과. 중단·재개했다면 이전 run 경로도 포함한다.
- 같은 평가 조건에서 확인한 보행 결과와 비교 메모. 40,000회 완료만으로 성능이
  검증된 것으로 간주하지 않는다.

체크포인트와 로그는 Git에 추가하지 않고 파일 전송이나 별도 결과 저장소로 전달한다.
