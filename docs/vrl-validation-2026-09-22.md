# VRL 실행 검증 및 센서 수정 — 2026-09-22

범위는 이 저장소뿐이다. 형제 `gd_lab`, 배포 저장소, 공유 SIF/venv 및 기존
블라인드 학습은 변경하지 않는다. Git push는 하지 않는다.

## 수정

- Depth ghost는 깨끗한 depth 기준으로 방향을 보장한다. false hole은 더 멀게,
  false return은 더 가깝게 생성하며 센서 clip 범위로 제한한다. far sentinel은
  실제 표면으로 취급하지 않는다. IR과 teacher 원본, 비갭 영역은 보존한다.
- `--no_camera_noise`는 일반 노이즈와 gap ghost를 모두 끈다.
- 카메라 관측은 전역 4 정책스텝(80ms) 주기로 갱신한다. 부분 리셋은 해당
  행만 즉시 다시 캡처하되 다음 전역 주기에 재합류한다. 센서의 lazy read 주기를
  0으로 하여 리셋 때문에 센서 내부 주기가 밀리지 않도록 했다. 렌더링 자체는
  여전히 정규 4스텝과 reset 시에만 일어난다.
- Fabric 사용 시 카메라의 USD Xform 기반 pose가 초기 스폰 위치에 남는 문제를
  발견했다. 이미지 투영과 ghost 위치 계산은 이제 현재 PhysX trunk 위치·자세에
  벤더 장착 extrinsic을 합성한다. 벤더 보정값이나 FoV를 임의로 넓히지 않았다.
- Student play는 부분 리셋 직후 hidden을 지우고 새 프레임에서 latent를 계산한다.
- 체크포인트에 카메라 보정 계약, `ir_proxy` 채널 표기, 노이즈 활성 여부와
  teacher 체크포인트 경로를 저장한다. play는 다른 보정 계약을 거부한다.
  예전 metadata 없는 체크포인트는 경고를 출력하고 읽을 수 있다.
- 가시 셀 비율과 hazard 감독 가능 행 비율을 TensorBoard에 기록한다.
  `hazard_mse=0`만으로 학습이 정상이라고 판단하지 않는다.
- Kit cache/data/logs는 `logs/kit_runtime/<pid>/`에 격리한다. 실제 cache 토큰도
  실행 전에 확인한다. material cache의 작은 컨테이너 overlay 공간 부족을 막는다.
- 중복 `VrlSceneCfg`를 제거하고 gallery도 teacher 카메라를 항상 활성화한다.
- `src/gd_lab/rl/cenet.py`는 수정하지 않았다. 모든 이상치 가드는 그대로다.

## 관측 계약

| 경로 | 텐서 |
|---|---|
| Policy / CENet history | `[N,230]` |
| Critic | 전체 scan 187을 포함한 `[N,298]` |
| Teacher terrain encoder | 가시 높이 187 + 가시 mask 187 = `[N,374]` → `[N,32]` |
| Actor | proprio history 184 + CENet code 19 + terrain latent 32 = 235 → action 12 |
| Student | `[N,4,2,45,80]` + hidden `[N,64]` → latent `[N,32]` + hidden `[N,64]` |

## 짧은 실환경 검사

최종 CPU 회귀 검사는 **89 passed, 1 skipped**(78.41초)이며 Ruff와 프로젝트
convention 검사도 통과했다. skip은 별도 opt-in Isaac smoke 모듈이다.
본 작업의 실제 Isaac 검증은 아래 별도 명령으로 수행했다.

최종 student 증류는 8환경에서 10 camera ticks(BPTT 8+2, optimizer update 2회)로
제한했다. 마지막 window의 latent MSE는 0.01242, hazard MSE는 0.02244,
가시 셀 비율은 0.0762, hazard 감독 가능 행 비율은 1.0이었다.
체크포인트는 `logs/vision_smoke_100/student_pose_fixed_smoke_10/perception_10.pt`에
저장됐다. 이 수치는 입력과 역전파 경로를 확인할 뿐 수렴이나 안전 성능을 뜻하지 않는다.

동일 체크포인트를 `play_student.py`로 재로딩하여 40 정책스텝/10 카메라 갱신의
student→Actor 추론을 완료했다(exit 0). action 비유한 값 오류 및 material cache
공간 부족 오류는 없었다. 최종 추론 시 teacher latent와의 MSE는 0.01126이었다.

`scripts/check_vrl_camera_sync.py --headless --device cuda:0`는 optimizer 없이
10 정책스텝만 실행한다. 1·6·7스텝에 서로 다른 환경을 강제 리셋하여 다음을 검사한다.

- 프레임 획득 시점과 실제 render의 physics-step counter가 일치한다.
- lazy 센서 버퍼의 갱신 시각이 최신이며 카메라 보정값이 현재 trunk에 합성된다.
- 리셋되지 않은 행은 버퍼를 유지하고 4·8스텝에서 모두 같은 주기로 돌아온다.
- terrain label은 유한하며 전체 visibility가 0이 아니다.

실제 검증에서 가시 셀 비율은 수정 전 0에서 수정 후 약 4.75–9.49%가 됐다.
이 비율은 8환경 짧은 PLAY 장면의 수치이지, 모든 지형/자세의 보장값이 아니다.
해당 실행의 Kit cache 저장 오류는 재현되지 않았고 material_cache.json 생성도 확인했다.

## 남은 한계

- IR은 RGB luminance 기반 대용 영상이다. 이번 수정은 실제 active-IR의 재질 응답,
  projector, stereo matching을 구현하지 않는다. 실기 IR로 depth 오류를 보완하는
  성능은 별도 센서 모델과 실측 검증이 필요하다.
- 실환경 검사는 기본 vendor_legacy 보정으로 수행했다. vendor_new는 이번에
  좌표 변환 단위 테스트만 수행했으며 실제 렌더/배포 검증은 별도로 필요하다.
- 이전의 visibility 0 상태에서 학습한 VRL 체크포인트를 올바른 지형 teacher로
  간주해서는 안 된다. 다음 본 학습은 수정된 teacher로 새로 시작하는 것이 필요하다.
  이번 student 실행의 teacher는 실행 확인용 model_0.pt이며 성능 평가용이 아니다.
- 짧은 실행/회귀 테스트는 갭 통과 성공률·장기 수렴·실기 안전을 보장하지 않는다.
  MuJoCo/실기 배포 코드는 이번 범위에 포함하지 않는다.
- Kit NGX 초기화 및 URDF foot visual reference 경고는 별개로 남는다. 학습 실행을
  막지는 않았지만 전체 렌더 자산 품질 검사를 대신하지 않는다.

재현은 동일 SIF/venv에 이 저장소의 `src`를 PYTHONPATH로 지정한다. 공유 venv의
editable 설치를 덮어써 다른 블라인드 학습의 import 경로를 바꾸지 않는다.
