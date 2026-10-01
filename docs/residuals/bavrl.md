# BAVRL — Blind-Anchored Visual Residual Learning

고정 `DWB-38000`의 CENet·Actor·관측 정규화를 유지하고, 영상으로 작은 행동 보정을 학습한다.
이는 교사 latent 증류가 아니다. 원본 교사의 지형 latent는 필요하지 않다.

## 구현

- `src/gd_lab/residuals/bavrl/model.py`: 체크포인트 strict load, 고정 교사, 시각 잔차, 저장/복원.
- `src/gd_lab/residuals/bavrl/ppo.py`: 잔차 PPO, GAE, 공간 보조 손실.
- `scripts/train_bavrl.py`: 단일 GPU 학습, 촬영 시점 정답·지연·누락·에피소드 격리.
- `tests/test_bavrl.py`: CPU 회귀 검사. 실제 보행 성능 검증은 별도다.

4카메라 IR(렌더 grayscale proxy)+Depth → 공간 CNN → 11×17 grid cross-attention → GRU →
32차원 시각 특징을 블라인드 Actor 입력·이전 보정·영상 나이와 결합한다.
잔차 MLP는 128→64→12이고 마지막 층은 0으로 초기화한다.
가치망은 교사 critic의 독립 복사본을 학습한다. 교사 자체와 정규화는 갱신하지 않는다.

초기 **결정론적** 행동은 블라인드와 같다. 유효 영상이 있는 학습 중에는 Gaussian 탐색이 있으므로
샘플 행동까지 항상 같다는 뜻은 아니다. PPO는 tanh·slew 이전 원시 샘플의 log-probability를 사용한다.
현재 GRU 학습은 촬영 경계에서 기억을 detach하는 단일 촬영 truncated 방식이며, 긴 sequence BPTT가 아니다.
이미지·입력 기억·나이·이전 잔차를 그대로 PPO에 재생하며 minibatch마다 augmentation하지 않는다.

## 영상 미수신 계약

- 첫 영상이 없거나 수신 불가가 명시되면 시각망을 우회하고 **잔차=0**.
- NaN/Inf 영상, 잘못된 나이, 마지막 촬영 후 250ms 초과도 동일하다.
- 이때 탐색 노이즈 및 이전 잔차를 모두 제거하여 `action = frozen_blind_actor(obs)`를 그대로 사용한다.
- 누락된 한 프레임과 완전한 센서 장애를 구분한다. 정상 캐시 영상은 250ms 이내에 한해 재사용한다.
  즉, 연결 끊김 신호가 없으면 마지막 프레임이 만료되는 시점에 블라인드로 복귀한다.
- 유효하지 않은 행의 다음 시각 기억은 0이다. 에피소드 리셋 시 기억·잔차·영상 유효성도 초기화한다.
- 블라인드 복귀는 즉시 수행하여 아래 slew 제한보다 우선한다. 전환 충격은 추후 시뮬레이션 평가 대상이다.

같은 관측에서 같은 행동을 출력한다는 의미다. 이미 비전 보행으로 달라진 물리 상태까지 되돌리거나,
블라인드 정책의 갭 통과·전도 방지를 보장하지 않는다.

기본 잔차 한계 ±0.2, tick당 변화 ±0.02는 **정규화된 policy action 단위**다.
실제 관절각 보정은 여기에 원본 action scale을 곱한다. radians로 혼동하면 안 된다.

## 학습 목표와 범위

PPO 환경 보상 + value loss + 작은 entropy/잔차 크기/변화량 항 + 공간 보조 손실을 사용한다.
공간 목표는 가시 높이·가시성·상승/하강 경계·평탄 지지면 proxy다. 안전 착지 정답이 아니다.
GAVD의 광선·카메라 위치 embedding을 재사용하며, 실시간 자세를 사용하는 정확한 BEV 투영은 아니다.
지연은 0~100ms, packet drop은 기본 5%다. 카메라별 고장 augmentation과 긴 BPTT는 아직 없다.
Isaac 자동 reset의 timeout은 RSL과 같은 pre-step V 근사로 bootstrap한다.

## 실행

레포 루트에서 기존 Isaac Apptainer GPU 런타임/환경변수 설정을 사용해 다음 Python 진입점을 실행한다.
먼저 `--num_envs 2 --iterations 1 --horizon 2 --epochs 1 --batch-size 2`로 시뮬레이터 smoke가 필요하다.

```bash
TRAIN_ARM=4 python scripts/train_bavrl.py --headless \
  --teacher checkpoints/teachers/blind/arm4_38000/teacher/model_38000.pt \
  --teacher-agent checkpoints/teachers/blind/arm4_38000/teacher/params/agent.yaml \
  --output logs/bavrl/first_run --num_envs 32 --iterations 1000
```

`--output`은 새 디렉토리여야 한다. 관측 순서·action scale/offset·PD gain·dt가 교사와 다르면 중단한다.
`--resume`은 BAVRL 모델/optimizer만 복구하며 물리 상태·curriculum·센서 queue/RNG는 재개하지 않는다.
교사 hash, 전체 동결 상태, 잔차 설정이 다르면 복원을 거부한다. 50회마다 및 마지막에 저장한다.
`scripts/export_bavrl.py`는 actor/vision ONNX를 분리 export하고 parity를 검사한다.
배포 리포의 BAVRL-sim backend만 지원하며, 일반 VRL actor와 혼용하지 않는다.

## 2026-09-30 실행 기록

- Isaac 2env, 1iteration smoke 저장 성공.
- tmux `bavrl-5090-20260930`: 32env / horizon16 / PPO4epochs / 1000iteration 실행 시작.
- 실행 디렉토리 `logs/bavrl/dwb38000_bavrl_1000_20260930/`.
- `bavrl_150.pt`를 배포 리포 `resources/policy/bavrl/dwb38000_bavrl150_20260930/`에 배치 (기존 vrl 경로에서 분리).
- actor ONNX parity, 영상 없음/만료 시 잔차 0, C++ 로더의 무영상 추론 확인.
- 배포 리포 `bavrl/run_sim.sh`의 기본값은 이 **중간 150회 모델**. 전체 학습 완료/보행 성능 검증을 뜻하지 않는다.
- 실제 MuJoCo 보행/실물 구동은 실행하지 않았다. 최종 체크포인트 자동 교체도 하지 않는다.
