# Apptainer 학습 컨테이너

빌드 → venv 구성 → 학습까지의 단계별 워크플로우는 [루트 README의 Quick
start](../README.md#quick-start)에 있다. 이 문서는 이미지의 설계와 세부 사항만
다룬다.

## 설계

`gd_lab_isaaclab.def`는 무겁고 버전이 고정된 베이스만 이미지에 굽는다:
CUDA 12.8 + Python 3.11 + torch 2.7.0(cu128) + IsaacSim 5.1.0. 개발 중에
바뀌는 것들 — IsaacLab, rsl-rl, gd_lab 자체 — 은 **컨테이너의 python으로 만든
호스트 쪽 venv**(`--system-site-packages`)에 editable로 두므로, 코드를 고쳐도
이미지 리빌드가 필요 없다. 실제 다중 GPU 학습 서버에서 검증된 레시피 그대로다.

이미지는 GPU 모델과 무관하다: pip 휠 설치뿐이라 빌드 시 GPU별 컴파일이 없고,
torch cu128 휠이 fat binary로 여러 아키텍처(sm_70~sm_120, Blackwell 포함)의
커널을 담는다. 호스트에 필요한 것은 CUDA 12.8을 지원하는 NVIDIA 드라이버뿐이며,
드라이버는 실행 시 apptainer가 호스트에서 주입한다. `.def`를 고쳐 리빌드하는
경우는 GPU가 아니라 스택이 바뀔 때다 (torch/IsaacSim 버전 업, cu128 휠에 커널이
없는 구형 아키텍처 지원).

## 빌드 변형

```bash
apptainer build gd_lab_isaaclab.sif containers/gd_lab_isaaclab.def
# 이미지 내부를 들여다보며 쓰고 싶으면 sandbox로:
apptainer build --sandbox gd_lab_isaaclab containers/gd_lab_isaaclab.def
```

빌드는 `.def` 레시피가 있을 때 하는 것이고, 이미 빌드된 `.sif`(또는 sandbox
디렉토리)를 받았다면 빌드 없이 바로 실행하면 된다. 비루트 빌드는 apptainer
설정에 따라 `--fakeroot`가 필요할 수 있다.

## 실행 노트

- `--writable-tmpfs`: Omniverse 캐시용 임시 오버레이. 없으면 캐시 쓰기 경고가
  나거나 시작이 느려진다.
- `OMNI_KIT_ACCEPT_EULA=YES`: 대화형 EULA 프롬프트 생략 (nohup 실행에 필수).
- GPU는 `CUDA_VISIBLE_DEVICES`로 고르며, 프로세스 안에서는 `cuda:0`이 된다.
- apptainer가 NVIDIA 드라이버를 자동 주입하지 않는 호스트에서는 `--nv` 추가.
- venv 스크립트(`ruff`, `pytest` 등)의 shebang은 컨테이너 python을 가리키므로
  호스트에서 직접 실행되지 않는다 — 항상 `apptainer exec` 안에서 실행한다.
