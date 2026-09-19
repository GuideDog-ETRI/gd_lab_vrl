# DreamWaQ 학습 구조 결정

코드에서 읽히지 않는 근거만 남긴다. 차원·입출력 규격은 `methods/dreamwaq/spec.py`와
`tests/`가 유일한 선언이므로 여기에 다시 쓰지 않는다.

## CENet: 분리(decoupled) 학습

- CENet은 자체 Adam(lr 1e-3)으로 학습되며 PPO loss에 합류하지 않는다.
  code는 actor에 **detach**되어 들어간다 (PPO gradient가 encoder에 닿지 않음).
- 손실: `MSE(v_hat, v_gt) + MSE(decode, o_{t+1}) + beta * KL`.
  - KL은 batch·latent 양쪽 **mean** reduction, `beta = 1.0`.
  - velocity head는 decoder에 detach되어 들어가므로 reconstruction은
    latent/decoder만 학습시킨다.
  - o_{t+1} 재구성 타깃은 terminal 전이(done)를 마스크한다.
- 입력·타깃 모두 actor 관측 normalizer를 통과한다. 따라서
  `actor_obs_normalization=True`가 필수다 (정책 생성자가 강제).
- v_gt는 critic 관측의 `base_lin_vel` 항을 이름으로 슬라이스한 값이다.
  관측 매니저의 scale·clip이 적용된 값 기준이므로 v_hat도 같은 스케일을
  추정한다. 배포에서 실속도가 필요하면 그 항의 scale로 되나눈다.

## CENet 패스의 위치: PPO update 이후 (인터리브 아님)

CENet 학습 패스는 `super().update()`(PPO minibatch 루프) **완료 후** 같은 rollout
데이터에 대해 epochs x minibatches 스텝으로 실행한다. PPO 업데이트 forward가
rollout 시점과 동일한 CENet 가중치의 code를 보므로 재계산 log-prob의 일관성이
인터리브 방식보다 오히려 좋다. CENet은 좌우대칭 증강 **이전**의 원본 배치로
학습한다.

## AdaBoot: iteration당 코인 1개

- `p_boot = 1 - tanh(CV(에피소드 리턴))`, 완료 에피소드 리턴 rolling 100개 기준,
  EMA 0.8, warmup 10 스케줄 업데이트.
- GT 주입 결정은 **learning iteration당 코인 1개**이며, 해당 iteration의 rollout
  act와 PPO 업데이트 forward **양쪽에 동일하게** 적용된다. per-step 또는
  rollout 한쪽만 주입하면 저장된 log-prob과 재계산 log-prob이 어긋나
  adaptive LR이 바닥으로 붕괴한다.
- 코인 재추첨은 업데이트 종료 후 첫 rollout act(grad 비활성 시점 감지)에서
  1회 수행된다. 에피소드 리턴 추적은 `DreamwaqPPO.process_env_step`이 자체
  버퍼로 수행한다(러너 로거에 비의존).
- play/deploy(`act_inference`)는 절대 주입하지 않는다.

## 관측 레이아웃 규약

- IsaacLab 히스토리 평탄화는 term-major(항별로 oldest->newest 연속 블록)다.
  actor가 보는 K개 프레임의 추출 인덱스는 이 규약에서 계산된다.
- policy one-step 레이아웃은 원 논문의 proprio 블록 뒤에 `payload` 1열이 붙은
  형태다. payload는 DR이 아니라 **태스크 변수**다 — 배포 시 운용자가 실제
  적재량을 입력하므로, 정책이 하중을 CENet으로 추정하게 두는 대신 명시적으로
  조건화한다. 그래서 노이즈를 싣지 않는다.

## Actor 입력: K=4 프레임

actor는 최신 one-step 대신 **최신 K=4 프레임**(newest first)과 code를 받는다.
K는 저장된 히스토리(H=5) 안에 들어가므로 관측·ONNX 계약은 그대로다. 단
actor 1층 폭이 `K * one_step + code_dim`이므로 **K를 바꾸면 체크포인트가
resume되지 않는다**.

## 배포 메타데이터

ONNX 그래프는 입출력 float 개수만 말한다. 어느 열이 어느 관절인지, 어떤 단위
인지, 커맨드에 이미 어떤 scale이 적용됐는지, 어떤 PD 게인 기준으로 학습됐는지는
그래프에 없고, 배포 측이 하나라도 틀리면 로봇은 자신 있게 틀리게 움직인다.
관절 순서는 파싱된 articulation에서만 알 수 있고 export는 시뮬레이터 없이
돌므로, 학습 시점에 `deploy/metadata.py`가 해석된 환경에서 스냅샷을 떠
체크포인트에 싣고 export가 그것을 ONNX metadata(`gd_lab.policy.v1`)와
`.deploy.json`에 적는다. 계약이 그래프를 설명하지 못하면 export는 실패한다
(조용히 라벨 없는 그래프를 내보내지 않는다).
