# RSL-RL 확장 방식: pip 고정 + 상속

## 결정

- `rsl-rl-lib==3.0.1`을 pip 의존성으로 고정하고 **무수정**으로 사용한다.
  저장소 내 사본·패치·훅 추가 금지 (CI가 검사).
- DreamWaQ 확장은 `src/gd_lab/rl/`의 서브클래스 3개로만 구현한다:
  `DreamwaqActorCritic(ActorCritic)`, `DreamwaqPPO(PPO)`, `DreamwaqRunner(OnPolicyRunner)`.

## upstream 결합 지점

아래가 upstream 구현에 의존하는 지점의 전부다.

| 위치 | 오버라이드 | upstream 결합도 |
|---|---|---|
| `rl/runner.py::_construct_algorithm` | upstream 본문을 따르되 `eval` 클래스 해석 2곳을 importlib(`"<module>:<attr>"`)로 대체 | **높음** — upstream의 동명 메서드 흐름(resolve_rnd/symmetry → policy → storage → alg)과 동기화 필요. deprecated `empirical_normalization` 경로는 다루지 않으므로 정규화 플래그는 `policy` cfg에서 명시한다 |
| `rl/runner.py::save/load` | 저장 dict의 `infos["gd_lab"]`에 adaptive LR·CENet 옵티마이저 상태를 추가(additive) | 낮음 |
| `rl/runner.py::load` | 체크포인트는 해당 iteration 완료 후 저장되므로 재개 시 `current_learning_iteration += 1` | 중간 — upstream 저장 시점 규약에 의존 |
| `rl/actor_critic.py` | `super().__init__` 후 actor MLP를 재생성(입력 = latest one-step + CENet code). `act`/`act_inference` 오버라이드 | 중간 — 베이스의 normalizer/std 처리 재사용 |
| `rl/ppo.py::process_env_step` | 보조 버퍼(next-obs latest, dones, 에피소드 리턴) 축적 후 `super()` 호출 | 낮음 |
| `rl/ppo.py::update` | AdaBoot 스케줄 갱신 → `super().update()` → CENet 보조 학습 패스 | 낮음 — storage 텐서가 `clear()` 후에도 유지된다는 사실에 의존 |

## 그대로 두는 upstream 특성

- minibatch 순열이 전 epoch에 대해 1회만 추출됨.
- `num_envs * num_steps_per_env`가 `num_mini_batches`로 나누어떨어지지 않으면
  rollout 꼬리가 학습에서 제외됨 (기본 설정 4096 x 48 / 4는 정확히 나누어떨어진다).
- timeout bootstrap이 V(s_t)를 사용.
- `log_dir=None`이면 iteration 0 저장에서 크래시 — 러너는 항상 log_dir와 함께 생성한다.
