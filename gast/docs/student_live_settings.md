# GAST 학생 증류: 온라인 설정

원본 `gast/scripts/train_student.py`는 그대로 두고, 온라인 설정 적용은 `gast/scripts/train_student_live.py`에 분리했다. 런처는 `gast/scripts/run_student_live.sh`이며, 17206 교사 첫 단계 예시는 `gast/scripts/train_17206_gast_stage1.sh`이다.

## 런타임 JSON

학습 초기화 시 `--live_config` 경로에 JSON을 원자적으로 생성한다. 옵션을 생략하면 해당 run 폴더의 `student_live.json`을 쓴다. 실행 로그에 경로가 출력된다. JSON은 캡처 iteration마다 확인하며, 적용된 변경은 `[LIVE_CONFIG]`로 기록한다. 편집기는 임시 파일에 쓴 뒤 원래 JSON으로 rename하는 원자적 저장을 권장한다.

핫리로드 대상:

- `learning_rate`: optimizer param group에 즉시 반영
- `student_warmup`, `student_ramp`: 학생 rollout 스케줄에 즉시 반영
- `checkpoint_interval`: 다음 저장 판단부터 반영; 저장은 BPTT 창 경계에서 수행
- `bptt_steps`: 현재 창 길이는 유지하고 다음 BPTT 창부터 반영

잘못된 JSON/값 또는 미지원 키는 마지막 정상 설정을 유지하며 경고한다. `hazard_loss_coef`는 Top-5 점수의 비교 가능성을 위해 실행 중 변경할 수 없다.

## 재시작이 필요한 설정

`num_envs`, 총 `iterations`, seed, 교사 체크포인트, 학생 아키텍처/hidden 크기, 카메라 보정·주기·지연·drop 조건, 센서 노이즈 모드, Top-5 시작/보관 수/스무딩/gate, hazard loss 계수는 실행 계약이므로 재시작 후 지정한다. 특히 env 수 전환은 현재 체크포인트에서 optimizer/model을 복원해 새 프로세스로 재개해야 한다.

## 저장 및 Top-5 해석

기본 전체 체크포인트 간격은 1000 capture iteration이다. BPTT16에서는 저장 시점이 창 경계에 맞춰질 수 있고, 단계 종료 시 최종 체크포인트를 저장한다. Top-5는 5000 iteration부터 optimizer/BPTT 창마다 후보를 평가한다. 순위 점수는 스무딩된 학습 손실 proxy이며 visible fraction 및 hazard-supervised fraction 각 0.95 gate를 사용한다. 이는 held-out 평가나 갭/계단 보행 성능 검증이 아니다.

## 검증 상태 및 첫 단계

문법 검사는 `py_compile`, 실행 스크립트는 `bash -n`, 설정 동작은 Isaac Lab 컨테이너 smoke test로 통과했다. `gast/tests/test_live_config.py`도 컨테이너에서 6개 테스트 모두 통과했다. 원본 `train_student.py`는 수정하지 않았다.

`train_17206_gast_stage1.sh`는 17206 교사 Top-1(SHA256 `a70adbda5b73925f5936eaef27ee3b33686d9f7489486afdda9bf40ddb0c326d`)을 고정하고 256 env, BPTT16, 총 500 iteration으로 시작한다. 첫 패킷 검증에서 hazard 라벨·visible fraction·Attention 보조출력의 1D batch 텐서를 2D `flatten(1)`으로 검사하던 오류를 세 번의 실행에서 차례로 발견했다. 별도 `finite_batch_rows` 유틸리티로 모든 batch 텐서 rank를 안전하게 처리하도록 수정했고 1D/2D 및 NaN 사례를 회귀 테스트에 추가했다. 실패한 실행들은 optimizer update 전에 종료했으며 학습 체크포인트는 없었다. 컨테이너 테스트 9개와 문법 검사가 통과했다. 수정된 `gast17206-live-256-retry4`는 확인 시점에 160/500 capture iteration까지 진행했다. 최근 관측 `visible_fraction≈0.1235`인데 Top-5 gate는 0.95로 설정되어 있어, 현재 정의가 그대로라면 향후 후보가 gate에서 탈락할 수 있다. 이 gate는 임의 변경하지 않았으며 후보 선정 전에 metric 의미/기준을 검토해야 한다. 단계 목표를 마친 뒤 정상 저장·손실·자원 상태를 확인하고 512 env 누적 1500, 이후 1024 env 총 20000으로 재개할 계획이며 승격은 별도 확인 후 진행한다.
