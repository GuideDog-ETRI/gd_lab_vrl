# RL rollout replay

정책 하나를 짧게(기본 4초, 100 Hz면 400 스텝) 돌려 매 스텝을 기록하고, 브라우저에서 로봇 움직임과 보상 구성을 같이 재생한다.
실시간이 아니라 기록 파일을 재생하는 오프라인 분석 도구다.

1. 기록 (Isaac, GPU 필요):
   ```bash
   PYTHONPATH=src apptainer exec --nv ~/workspace/gd_lab_isaaclab.sif ~/workspace/venv_apptainer/bin/python \
     tools/rl_replay/record_rollout.py --headless --task Gd-GastGapCleanV21-Rbq10-Dreamwaq-v0 \
     --checkpoint checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt \
     --num_envs 8 --seconds 4 --out logs/rl_replay/gast2000.json
   ```
   `--vx 0.8`로 전진 명령 고정, `--stochastic`이면 학습 때처럼 행동을 샘플한다(기본은 배포처럼 평균).
2. 재생: `tools/rl_replay/viewer.html`을 브라우저로 열고 기록 파일을 고르거나 끌어다 놓는다(오프라인 동작).

화면 구성
- 로봇: 몸체·다리 관절 위치, 발 접지(초록/빨강), 11×17 높이 스캔(색 = 높이), 지나온 경로
- 이번 스텝 보상: 보상 항목별 값(가중치 적용, 스텝당), 크기순
- 미래 보상의 항목별 몫: G_t^k = Σ γ^i r^k_{t+i} — 가치 V(s)를 만드는 항목이 무엇인지
- 타임라인: critic 가치 V(s), 실제 미래 보상 G_t, advantage A_t(GAE), 보상 r_t
- 행동: 관절별 정책 평균 ± std와 실제 출력
- "왜 이 행동인가": A_t > 0이면 기대보다 결과가 좋아 PPO가 그 행동의 확률을 올리고, A_t < 0이면 내린다.

`make_demo.py`는 시뮬레이터 없이 뷰어를 시험하는 합성 기록을 만든다.

## Terrain layers and live view

The robot window overlays three terrain layers, each switched on/off in the header:

- **정답 지형** (blue): the 11x17 height-scanner hits, i.e. the privileged ground truth (critic).
- **교사 입력** (orange): the cells the teacher policy was actually given (`obs["terrain"]`, or the newest
  `gast_history` frame for GAST), put back at the height it received. BIVT-Ray shows only camera-visible cells.
- **학생 카메라** (green): the four belly depth cameras' pixels as world points (`--student_view`), i.e. what a
  student can see. Kept for the first `--cloud_envs` envs only (size).

**Live:** with "시뮬레이터에서 바로 받아 보기" ticked, the recorder also writes `<name>.live.ndjson`, one line per
step, and the viewer follows it while Isaac runs (G_t/A_t provisional, bootstrapped with the last V). When the
run ends the viewer swaps in the final `.json`. A stream started from the command line (`--live <file>` under
`logs/rl_replay/`) shows up in 📂 불러오기 as "● 실시간", or open `viewer.html?live=logs/rl_replay/<name>.live.ndjson`.
