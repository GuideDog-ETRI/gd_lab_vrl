# 한강 GH200 대규모 다중 노드 학습 사용 경험과 제안

2026-10-29 · B.S.Seo (ETRI)

## 요약

natStr 파티션에서 자체 빌드 aarch64 컨테이너로 Isaac Lab 강화학습(PyTorch DDP)을 **최대 236노드(GPU 944장)까지 돌렸고, 128노드(GPU 512장) 학습을 15시간 7분 동안 중단 없이 끝냈습니다** (10/29 07:27~22:35). 장애로 볼 만한 것은 특정 노드 3대뿐이었고, 나머지는 한강 환경 특성과 가이드 보완 제안, 문의 사항입니다. (최종 정리 10/29)

- **대규모 성능:** 236노드에서 초당 약 252만 step, 128노드(GPU당 환경 3,584개)에서 초당 약 210만 step (아래 규모별 측정)
- **환경 특성·문제 노드: 22건** (계산 노드 HTTPS 전용, `--nv`의 Vulkan ICD, srun 기본 CPU 1개, 자체 SIF의 aws-ofi-nccl 의존 라이브러리, 대규모 출력·rendezvous, 시작 실패 노드, 사용자당 작업 5개 제한, ISAM 반영 지연 등)
- **제안·문의:** 사용자 가이드 보완 8건, 운영 정책 제안 6건, 문의 5건 (마지막 절)

## 작업 환경

노드당 srun task 1개가 컨테이너 안에서 torchrun으로 GPU 4개를 띄우는 구조입니다.

| 항목 | 값 |
|---|---|
| 계정 / 그룹 | t515a05 / bs0232 |
| 파티션 / QOS | natStr / natstr (작업당 최대 1일) |
| 작업 요청 | `--exclusive --gres=gpu:4 --ntasks-per-node=1`, `--comment="field=robotics;appl=pytorch-ddp"` |
| 실행 | `srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none` → `apptainer exec --nv` → `torch.distributed.run --nnodes=N --nproc_per_node=4` |
| 컨테이너 | 자체 빌드 aarch64 SIF (`nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` 기반), apptainer 1.4.5 |
| 소프트웨어 | Python 3.11, PyTorch 2.7.0+cu128 (NCCL 2.26.2), Isaac Sim 5.1.0 (pip), Isaac Lab 2.3.1, rsl-rl 3.0.1 |
| GPU 드라이버 | 570.124.06 (CUDA 12.8) |
| 노드 간 통신 | 처음에는 NCCL TCP 소켓(hsn0), 이후 한강 가이드대로 aws-ofi-nccl 1.20.0 + libfabric 2.3.1 (cxi). `NCCL_PROTO=^LL128` |
| 학습 규모 | GPU당 환경 2,048~3,584개, iteration당 100 step. GPU당 프로세스 1개(Isaac Sim + PyTorch) |

작업 파일은 모두 `/scratch/t515a05/gd_lab/` 아래에 있습니다.

## 규모별 성능 측정

노드를 118배로 늘리는 동안 iteration당 시간은 약 1.7배만 늘었고 처리량은 약 98배가 됐습니다. 236노드와 128노드(환경 3,584개)는 실제 학습 전체 구간의 평균입니다. iteration = 환경마다 100 step 수집 + PPO 학습 1회입니다.

| 노드 (GPU) | 노드 간 통신 | GPU당 환경 | iteration당 시간 | 처리량 |
|---|---|---|---|---|
| 236 (944) | AWS-OFI cxi | 2,048 | 약 77 s | 약 2,520,000 step/s |
| 128 (512) | AWS-OFI cxi | 3,584 | 87~89 s | 약 2,100,000 step/s (15시간 연속) |
| 118 (472) | AWS-OFI cxi | 2,048 | 59~64 s | 약 1,520,000 step/s |
| 112 (448) | AWS-OFI cxi | 2,048 | 57~60 s | 약 1,500,000 step/s |
| 64 (256) | TCP hsn0 | 2,048 | 65~72 s | 730,000~810,000 step/s |
| 32 (128) | TCP hsn0 | 1,366 | 57.1 s | 301,000 step/s |
| 8 (32) | TCP hsn0 | 1,366 | 52.2 s | 83,000 step/s |
| 2 (8) | TCP hsn0 | 1,366 | 45.3 s | 24,000 step/s |
| 1 (4) | 노드 내부 | 1,366 / 2,048 / 4,096 | 35.1 / 46.5 / 74.4 s | 15,500 step/s (1,366개) |

## 한강 환경 특성과 우회 방법

자체 빌드 컨테이너로 다중 노드 학습을 할 때 확인한 10건입니다. 모두 우회했으며, 가이드에 있으면 다른 사용자의 시행착오를 줄일 수 있는 내용입니다.

| # | 항목 | 증상 | 우회 방법 |
|---|---|---|---|
| 1 | 계산 노드 외부 네트워크 | HTTP(80) 차단, HTTPS만 허용. 컨테이너 빌드 중 `apt-get update` 실패 | apt 저장소를 https로 변경 |
| 2 | 계산 노드에서 Omniverse S3 에셋 요청 | 요청이 응답 없이 10분간 멈춤 | 원격 에셋을 쓰는 기능 끄기 |
| 3 | `apptainer --nv`와 Vulkan | 호스트 ICD 파일명이 `nvidia_icd.aarch64.json`(절대 경로 `/usr/lib64/libGLX_nvidia.so.0`)이라 컨테이너에 연결되지 않음. Isaac Sim이 GPU를 찾지 못함 | `library_path: libGLX_nvidia.so.0`인 ICD json을 `/usr/share/vulkan/icd.d/nvidia_icd.json`에 바인드 |
| 4 | srun step CPU 배정 | `--cpus-per-task` 없이 srun으로 노드당 1 task를 띄우면 CPU 1개만 배정. 4 GPU 프로세스가 코어 1개를 공유해 약 24배 느려짐 | `srun --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none` |
| 5 | 자체 SIF에서 aws-ofi-nccl | 플러그인이 libhwloc.so.15, libnl-3.so.200(libhwloc는 libpciaccess.so.0)을 필요로 하는데 Ubuntu 기반 SIF에 없음. `network AWS Libfabric not found` | kisti-container `--nccl ofi --dry-run`의 바인드에 세 라이브러리를 추가로 `/host/usr/lib64`에 바인드 |
| 6 | NCCL LL128 | 가이드의 노드 간 AllReduce 오류 안내 | `NCCL_PROTO=^LL128` |
| 7 | 대규모 작업의 표준 출력 | 128노드(512 프로세스)에서 모든 프로세스 출력을 srun으로 모으면 출력 전달이 밀려 프로세스가 쓰기에서 멈추고, 다른 프로세스는 집합 통신에서 무한 대기 | rank별 파일로 출력 (`torchrun --log_dir ... --redirects 3`) |
| 8 | torchrun c10d rendezvous | 128노드에서 간헐적으로 `RendezvousConnectionError` / `failed to recv, got 0 bytes` | 고정 rendezvous(`--node_rank=$SLURM_NODEID --master_addr --master_port`) + 작업별 포트(`20000 + JOB_ID % 20000`) |
| 9 | 멈춘 프로세스 진단 | 계산 노드에서 ptrace가 막혀 py-spy 사용 불가 | Python `faulthandler.dump_traceback_later`로 주기적 스택 출력 |
| 10 | GPU 드라이버 | 570.124.06. NGC `isaac-sim:5.1.0` 이미지는 570.169 이상 요구. 최신 PyTorch cu130 휠은 580 이상 요구 | pip 휠(Isaac Sim 5.1.0)과 PyTorch 2.7.0+cu128으로 자체 빌드 |

**학습 중 추가로 확인한 사항 (10/28~29)**

| # | 항목 | 증상 | 근거 | 우회 방법 |
|---|---|---|---|---|
| 11 | node1730 작업 시작 실패 | `srun: error: task 225 launch failed: Error configuring interconnect` | 작업 1171289, 10/28 21:56 (237노드) | `--exclude=node1730` |
| 12 | node1395 작업 시작 실패 | `task 66 launch failed: Error configuring interconnect` | 작업 1173432, 10/29 06:32 (236노드) | `--exclude=node1395` |
| 13 | node1402 GPU 2번 멈춤 (추정) | 작업은 시작되지만 GPU 2번 프로세스가 첫 시뮬레이션 step을 끝내지 못함. 오류 메시지 없음, ECC 오류 0, 클럭 제한 없음 | 작업 1173416(10/29 06:09), 1173460(06:56)에서 동일 증상. 같은 노드의 다른 GPU 3장과 다른 노드는 정상. 10/28 17노드 시험(1170171)에서는 이 노드도 정상 | `--exclude=node1402`, 점검 요청 |
| 14 | 노드 하나 실패에 전체 대기 | 위 11~13의 경우 나머지 수백 노드가 오류 없이 무한 대기 (작업은 RUNNING 상태 유지) | 매번 15~20분간 236노드 또는 128노드 점유 | 사용자 감시 스크립트로 탐지 후 제외·재제출 |
| 15 | /scratch(Lustre)의 rename | `renameat2(RENAME_EXCHANGE)` 미지원 → `[Errno 22] Invalid argument` (디렉터리 원자적 교체 불가) | 작업 1171339, 10/29 06:09 체크포인트 저장 중 학습 중단 | 일반 rename 3회로 대체 |
| 16 | 계산 노드 GPU 메모리 | 로그인 노드는 GH200 144 GB로 표시, 계산 노드는 96 GB (Slurm GRES 이름은 `nvidia_gh200_120gb`) | `nvidia-smi` 비교 | GPU당 작업 크기를 96 GB 기준으로 설계 |
| 17 | GPU 활용 실측 | 236노드 학습(GPU당 환경 2,048개): 사용률 평균 66%, 전력 평균 180 W(최대 290 W), 메모리 48.5/96 GB | 10/28 22:10, 노드 5개 표본 | (참고용 성능 자료) |
| 18 | 대기열 비공개 | 다른 사용자의 실행·대기 작업을 볼 수 없어, 대규모 작업이 다른 사용자를 기다리게 하는지 판단 불가 | `squeue`에 자기 작업만 표시 | (운영 정책 제안 참고) |
| 19 | 사용자당 작업 5개 제한 | 평가 작업 15개(노드 1대씩)를 제출하자 5개만 접수되고 나머지는 QOSMaxSubmitJobPerUserLimit으로 거부 | 10/29 22:39, 작업 1179861~1179865 접수 후 거부 | 작업 하나(노드 10대) 안에서 srun step을 노드별로 나눠 실행 |
| 20 | /scratch 밖에서 srun 거부 | 홈 디렉터리에서 실행 중인 작업에 붙어 nvidia-smi만 보려 해도(`srun --overlap --jobid`) cli_filter가 "jobs must run under /scratch or /appsdata"로 거부 | 10/29 17:53, 작업 1173553 GPU 상태 확인 중 | /scratch로 이동한 뒤 실행 |
| 21 | ISAM 사용량 반영 지연 | 10/29 17:55 조회 시 10/28까지 끝난 작업만 반영(707,073 SRU-초). 실제 사용량은 작업 기록으로 계산하면 약 497만 SRU-초 | isam 메뉴 (1)과 sacct 노드·시간 합계(노드 1시간 = 1,200 SRU-초) 비교 | sacct로 직접 계산 |
| 22 | 128노드 장시간 GPU 활용 실측 | GPU당 환경 3,584개: 사용률 평균 62~67%(90% 이상 47~54% 시간, 20% 미만 22~25% 시간), 전력 평균 224~239 W(최대 302 W), 메모리 80/96 GB. 15시간 동안 성능 저하·GPU 오류 없음 | 10/29 12:32, 17:53, 20:42, 21:23, 노드 8개 표본, 1초 간격 90초 | (참고용 성능 자료) |

## 제안과 문의

가이드 8건을 보완하면 자체 컨테이너 사용자가 대규모 작업을 훨씬 빨리 안정화할 수 있습니다.

**사용자 가이드 보완 제안**

1. **자체 컨테이너용 NCCL-OFI 연결 목록:** kisti-container `--nccl ofi`가 바인드하는 경로·라이브러리·환경변수를 문서로 공개하고, 이미지에 필요한 라이브러리(libhwloc, libnl-3, libpciaccess)를 명시
2. **대규모 작업의 출력 처리:** 수백 프로세스 이상에서는 표준 출력을 srun으로 모으지 말고 rank별 파일(`srun --output=%x_%j_%t.out`, `torchrun --redirects`)로 받도록 권장
3. **srun CPU 배정:** 노드당 1 task로 여러 GPU 프로세스를 띄울 때 `--cpus-per-task`를 반드시 지정해야 한다는 주의 문구
4. **대규모 torchrun rendezvous:** 100노드 이상에서는 고정 rendezvous(`--node_rank`, `--master_addr`)와 작업별 포트 사용 권장
5. **Vulkan을 쓰는 컨테이너:** Isaac Sim, Omniverse 등은 `--nv`만으로 GPU를 찾지 못함. ICD 바인드 방법 안내, 또는 apptainer 설정에서 `nvidia_icd.aarch64.json` 자동 연결
6. **계산 노드 네트워크 정책:** HTTPS만 허용됨(HTTP 차단)과 외부 에셋 서버 접근 제한을 컨테이너 빌드 안내에 명시
7. **작업 수 제한과 다수 소규모 작업 실행법:** natStr의 사용자당 작업 5개 제한을 QOS 표와 함께 명시하고, 모델 평가처럼 노드 1대짜리 작업을 여러 개 돌릴 때 한 작업 안에서 srun step으로 나누는 예제 제공
8. **/scratch 실행 제약의 범위:** 작업 제출뿐 아니라 실행 중 작업에 붙는 `srun --overlap`(모니터링용)도 /scratch에서만 허용된다는 점과 모니터링 권장 방법 안내

**운영 정책 제안**

현재 natStr QOS에는 사용자별 노드 상한이 없어, 빈 노드만 있으면 한 사용자가 파티션 대부분을 잡을 수 있습니다. 이번 시험에서도 작업 하나가 natStr 522노드 중 236노드(45%)를 사용했습니다.

1. **사용자별·과제별 노드 상한:** QOS의 `MaxTRESPerUser`, 계정의 `GrpTRES` 등으로 동시에 쓸 수 있는 노드·GPU 수를 제한 (현재는 작업당 522노드, 사용자당 동시 작업 5개만 제한)
2. **대규모 작업의 예약·시간대 배정:** 수백 노드 규모 작업은 Slurm 예약(reservation)이나 정해진 시간대로 운영해 다른 사용자의 대기를 예측 가능하게
3. **대규모 사전 시험의 사전 협의 절차:** 이번처럼 할당받은 서버를 대규모로 미리 써 보는 사용은 운영자와 일정·규모를 미리 협의하는 절차 마련
4. **시작 실패 노드의 자동 처리:** srun 작업 시작에 실패한 노드(`launch failed`)가 있으면 작업을 즉시 실패 처리하거나 해당 노드를 자동 drain. 지금은 노드 하나 때문에 수백 노드가 RUNNING 상태로 무한 대기함
5. **파티션 대기 현황 공개:** 파티션별 대기 작업 수·대기 노드 수 정도는 사용자가 볼 수 있게 해, 대규모 작업 규모를 스스로 조절할 수 있게
6. **실시간 사용량 조회:** ISAM 사용량이 하루 이상 늦게 반영되어 대규모 사용 중 잔여 시간을 알기 어렵습니다. 갱신 주기를 명시하거나 실행 중 작업까지 포함한 사용량 조회 명령 제공

**문의**

1. 계산 노드에서 멈춘 프로세스를 진단할 방법(py-spy, gdb 등 ptrace 허용 여부 또는 권장 도구)이 있는지
2. PyTorch 2.7에 들어 있는 NCCL 2.26.2와 aws-ofi-nccl 1.20.0 조합이 지원 범위인지, 권장 NCCL 버전이 있는지
3. GPU 드라이버(현재 570.124.06) 업데이트 계획이 있는지 (NGC Isaac Sim 5.1은 570.169 이상, CUDA 13 휠은 580 이상 필요)
4. natStr에서 100노드 이상 작업을 예약하거나 우선 배정받는 방법이 있는지 (빈 노드가 있어도 우선순위로 15~30분씩 대기)
5. 작업당 24시간 제한을 넘는 장기 학습에 권장하는 방식(작업 연결, 예약 등)이 있는지

재현용 스크립트는 요청하시면 공유하겠습니다.
