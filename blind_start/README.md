# Blind-start VRL (2단계 학습 실험)

gd_lab의 잘 걷는 **블라인드 DreamWaQ 체크포인트에서 출발**해 VRL 선생을 만드는 실험 코드다.
기존 `gd_lab_vrl` 코드(`src/gd_lab`, `scripts/train_vrl.py`)는 수정하지 않고, 이 디렉토리에서
상속·조립만 한다. `gd_lab_blind_start`를 import하면 태스크 2개가 추가 등록된다.

| 단계 | 태스크 | 지형 관측 | 카메라 렌더링 |
|---|---|---|---|
| 2-A | `Gd-VrlBlindStart-Rbq10-Dreamwaq-v0` | 전체 높이맵(모두 보임) + blackout | 없음 (빠름) |
| 2-B | `Gd-VrlBlindStart-Rbq10-Dreamwaq-Vision-v0` | 카메라 가시 높이맵(기존 VRL 선생과 동일) + blackout | 있음 |

## 구성

- `gd_lab_blind_start/blind_init.py`: 블라인드 가중치를 VRL actor-critic에 복사한다. actor 첫 층의
  지형 latent 32열만 0으로 시작하므로 iteration 0에서는 블라인드와 같은 행동을 낸다.
- `gd_lab_blind_start/blackout.py`, `terrain_dropout.py`: 지형 입력 blackout. 0.5–3 s 구간과
  5% 전체 blind episode로 약 15–20% 스텝에서 지형 그룹 전체를 0(모두 무효)으로 만든다.
- `gd_lab_blind_start/actor_critic.py`: 유효 셀이 하나도 없으면 지형 latent를 0으로 만든다.
  카메라가 끊긴 상태는 항상 "latent = 0"으로 actor에 들어가고, blackout 학습으로 이 입력에서
  블라인드 보행을 유지한다. **배포에서도 카메라 미수신 시 0 latent를 보내야 한다.**
- `scripts/train_blind_start.py`: `scripts/train_vrl.py` 사본. 위 태스크 등록, 2-A 카메라 끄기,
  `--blind_init` 추가만 다르다.

## 실행 (저장소 루트에서, Arm4)

```bash
run_bs() {
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES="${GPU:-0}" TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES \
    apptainer exec --nv --writable-tmpfs --bind "$PWD/logs/usd_tmp/blindstart:/tmp/IsaacLab" \
    "$GD_LAB_SIF" "$GD_LAB_PYTHON" blind_start/scripts/train_blind_start.py \
    --headless --device cuda:0 --seed 42 --logger tensorboard "$@"
}
# 2-A: 블라인드에서 출발 (카메라 없음)
run_bs --task Gd-VrlBlindStart-Rbq10-Dreamwaq-v0 --num_envs 4096 --max_iterations 1000 \
  --run_name arm4_blindstart_2a --blind_init <gd_lab>/logs/blind_rbq10_dreamwaq/arm_4/<run>/model_38000.pt \
  agent.save_interval=100
# 2-B: 2-A 결과에서 카메라 가시성 조건으로 fine-tuning
run_bs --task Gd-VrlBlindStart-Rbq10-Dreamwaq-Vision-v0 --num_envs 1365 --max_iterations 300 \
  --run_name arm4_blindstart_2b --resume --load_run <2-A run> --checkpoint model_999.pt \
  agent.save_interval=50
```

로그는 기존과 같이 `logs/vision_rbq10_dreamwaq/arm_4/` 아래에 쌓인다.

`scripts/run_blind_start.sh <gpu> <log> <train_blind_start.py 인자...>`는 위 실행을 감싼 것이다.
Isaac(carb)이 PID 이름으로 만드는 `/dev/shm` 파일이 다른 사용자의 오래된 파일과 우연히 겹치면
시작 직후 abort(134)가 나는데, 그 경우에만 최대 3회 재시도한다.

## 테스트

```bash
python -m pytest blind_start/tests -q
(cd blind_start && ruff check .)
```
