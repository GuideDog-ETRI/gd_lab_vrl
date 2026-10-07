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

기본 전체 체크포인트 간격은 1000 capture iteration이다. BPTT16에서는 저장 시점이 창 경계에 맞춰질 수 있고, 단계 종료 시 최종 체크포인트를 저장한다. Top-5는 5000 iteration부터 optimizer/BPTT 창마다 후보를 평가한다. 순위 점수는 스무딩된 학습 손실 proxy다. `visible_sample_fraction`은 유효 student row 중 교사 가시 셀이 하나 이상 있는 비율이며, `--top5_min_visible_sample_fraction`(구 alias `--top5_min_visible_fraction`)으로 gate한다. 다른 서버에서 사용 중인 0.85는 이 샘플 커버리지 기준으로 적용해야 한다. raw 셀 밀도 `visible_fraction`은 진단 통계로만 기록한다. `hazard_supervised_fraction` gate는 별도다. 이는 held-out 평가나 갭/계단 보행 성능 검증이 아니다.

## 검증 상태 및 첫 단계

문법 검사는 `py_compile`, 실행 스크립트는 `bash -n`으로 통과했다. Isaac Lab 컨테이너에서 `test_live_config.py`, `test_student_utils.py`, `test_student_top5.py` 테스트를 실행했다(최종 결과는 배포 커밋에서 확인). 원본 `train_student.py`는 수정하지 않았다.

`train_17206_gast_stage1.sh`는 17206 교사 Top-1(SHA256 `a70adbda5b73925f5936eaef27ee3b33686d9f7489486afdda9bf40ddb0c326d`)을 고정하고 256 env, BPTT16, 총 500 iteration으로 실행했다. 첫 패킷 검증에서 hazard 라벨·visible fraction·Attention 보조출력의 1D batch 텐서를 2D `flatten(1)`으로 검사하던 오류를 세 번의 실행에서 발견했다. 전용 `finite_batch_rows` 유틸리티로 수정했고 회귀 테스트를 추가했다. 이어 품질 gate도 점검해, 기존 0.95가 전체 셀 가시 밀도(실측 약 0.12)에 적용되어 후보가 탈락할 수 있음을 확인했다. 새 `train_student_live.py`는 raw `visible_fraction`을 진단값으로 보존하고, Top-5 gate에는 유효 student row 중 교사 가시 셀이 하나 이상 있는 row 비율(`visible_sample_fraction`)을 적용한다. 다른 서버의 0.85는 이 샘플 커버리지 기준으로 해석해야 하며 실행 시 `--top5_min_visible_sample_fraction 0.85`를 전달해야 한다. 시작 시 고정되는 설정으로 live JSON 핫리로드 대상이 아니다. hazard supervision gate는 별도다. 이전 `train_student.py`는 보존했고 수정 적용에는 `run_student_live.sh`/`train_student_live.py`가 필요하다. retry4는 256 env/BPTT16에서 500/500을 완료해 `gast/logs/gast/arm4/gast17206_env256_stage500_retry4_20261004/perception_500.pt`를 저장했다. 최종 로그는 latent MSE 0.00373, hazard MSE 0.01034, 가시 셀 비율 0.1232, hazard-supervised fraction 1.0, dropped packet 25/500이다. 500회 구간은 Top-5 시작점 5000 이전이라 Top-5 후보는 아직 없다. 다음 512-env 단계로 넘어갈지는 기존 계획에 따른 별도 실행이며 자동 시작하지 않았다.
