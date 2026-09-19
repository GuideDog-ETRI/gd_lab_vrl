# experiment overrides

반복되는 hydra 오버라이드를 yaml로 고정하는 곳. 일회성 조정은 CLI 오버라이드
(`env.rewards.X.weight=...`)로 한다.

## 학습 arm

`TRAIN_ARM=1`, `2`, `3`, `4`로 선택한다. `arm_N.yaml`은 Hydra 오버라이드
목록이며, `train.py`와 `play.py`가 명시적인 CLI 오버라이드보다 먼저 적용한다.
환경 변수를 지정하지 않으면 기존 기본 설정으로 실행한다. 다른 값은 오류다.

| TRAIN_ARM | 정책 제어 주기 | HIP/THIGH kp | KNEE kp | HIP/THIGH kd | KNEE kd |
| --- | --- | --- | --- | --- | --- |
| 1 | 50 Hz | 현재 기본값 (88.1367) | 현재 기본값 (102.2177) | 현재 기본값 (1.9919) | 현재 기본값 (1.9932) |
| 2 | 100 Hz | 현재 기본값 (88.1367) | 현재 기본값 (102.2177) | 현재 기본값 (1.9919) | 현재 기본값 (1.9932) |
| 3 | 50 Hz | 123.39 | 127.77 | 2.4 | 2.4 |
| 4 | 100 Hz | 123.39 | 127.77 | 2.4 | 2.4 |

- 네 arm 모두 command pulse와 그 확률 커리큘럼을 끈다
  (`pulse_prob=0`, `pulse_prob_schedule=null`). 나머지 커리큘럼은 유지한다.
- Payload reset 이벤트를 그대로 포함한다. 현재 설정은 에피소드마다
  **0 kg 20% / 6 kg 80%**, 장착 위치는 trunk 기준
  `x ∈ [-0.05, 0.05] m`, `y=0`, `z=0.07 m`이다.
- 표는 nominal gain이다. 기존 reset 시 gain 랜덤화(0.85~1.15배)는 유지한다.
  arm 1·2는 로봇 기본 gain을 상속하고, arm 3·4는 표의 값을 정확히 적용한다.
- 물리 시뮬레이션은 모두 `dt=0.005 s` (200 Hz)다. 제어 decimation만
  50 Hz에서 4, 100 Hz에서 2로 바꾸고, height scanner와 렌더링 주기를
  정책 주기에 맞춘다. Contact sensor는 물리 주기로 유지한다.
- PPO rollout 길이, gamma, 관측 history 길이는 기존 step 수를 유지한다.
  따라서 100 Hz에서 같은 step 수가 나타내는 시뮬레이션 시간은 절반이다.

IsaacLab 환경에서 실행:

```bash
TRAIN_ARM=1 python scripts/train.py --headless --logger tensorboard
TRAIN_ARM=2 python scripts/train.py --headless --logger tensorboard
TRAIN_ARM=3 python scripts/train.py --headless --logger tensorboard
TRAIN_ARM=4 python scripts/train.py --headless --logger tensorboard
```

각 명령은 독립적인 학습 실행이다. 컨테이너에서는 README의 `run` 함수를
`python` 대신 쓴다. 로그와 체크포인트는 기본적으로
`logs/blind_rbq10_dreamwaq/arm_N/<타임스탬프>/`에 저장되며,
`params/env.yaml`, `params/agent.yaml`에 최종 적용 설정이 기록된다.
`--experiment_name` 또는 `agent.experiment_name=...`을 명시하면 저장 경로를
직접 바꿀 수 있다.

재개와 play에도 같은 arm을 지정한다. Play/Gamepad는 자체 command와
curriculum 설정을 유지하면서 해당 arm의 주기, gain, 체크포인트 경로를 쓴다.

```bash
TRAIN_ARM=2 python scripts/train.py --headless --resume --load_run <run폴더명>
TRAIN_ARM=2 python scripts/play.py --load_run <run폴더명>
TRAIN_ARM=4 python scripts/play.py --task Gd-Blind-Rbq10-Dreamwaq-Gamepad-v0
```
