# 100 Hz 보행정책 백본 체크포인트

100 Hz·실기 게인(ARM4) 계보의 백본 모델이다. 각 모델은 바로 위 모델에서 이어 학습했다(후진 갭 v2만 3906에서 다시 시작).
`scripts/gast_narrow/train_narrow_*.sh`는 아래 경로와 SHA256을 기본값으로 쓴다.

| 모델 | 파일 | SHA256 | 출발 → 학습 |
|---|---|---|---|
| DWB-38000 | `checkpoints/teachers/blind/arm4_38000/teacher/model_38000.pt` | `c6cfe178cc069766f9543fb5fba983eb05a322d507e4302ee2495d4d26d2d46b` | 처음부터, 4096 env, 38,000 update |
| Ray+IR gap50 3906 | `checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt` | `f3e7afe53ea8be0f2dd777f331af417aba20b13b5e5be5470ec3d939470f4d98` | DWB-38000 → 1024 env, 5,000 update |
| 후진 갭 v2 6566 | `checkpoints/gast_narrow/backgapv2_6566.pt` | `22cf918c5029e0b6f55a51b50739b40958d41244514262a525750b088611ba78` | 3906 → 1024 env, 8,000 update |
| Narrow v1 1748 | `checkpoints/gast_narrow/narrowv1_1748.pt` | `e404fb41df686233b1d2909470dc685068c655c2295f69eb87b4a3478eeb7dc0` | 6566 → 1024 env, 3,000 update |
| Narrow v2 1111 | `checkpoints/gast_narrow/narrowv2_1111.pt` | `34bb627e6c4b29c847178bd1f7a1a54f060db25c77f88d120a956183815783ea` | 1748 → 4096 env (GPU 3장), 1,500 update |
| Narrow v2.1 465 | `checkpoints/gast_narrow/narrowv21_465.pt` | `d39c1294206121ba5eab85a83df07ab8cd7d0e484f10ffeb24768fb030a9f6ea` | 1111 → 4096 env (GPU 3장), 700 update |
| Narrow v2.2 371 | `checkpoints/gast_narrow/narrowv22_371.pt` | `00ce70515294ea9033505c40c1e295205e9070856c12c4a0d950dfffe693e677` | 465 → 4096 env (GPU 3장), 400 update |

- v2.1·v2.2 학습은 통로가 아닌 타일에서 6566 행동을 앵커로 쓴다(`NARROW_ANCHOR_CKPT`, 기본 `backgapv2_6566.pt`).
- 각 run 폴더(`narrow_v2_1748_*`, `narrow_v21_1111_*`, `narrow_v22_465_*`)에는 학습 설정(`params/`), 완료 기록, 온라인 Top-5 순위표, E2 요약만 둔다. 그 밖의 Top-5 체크포인트는 저장소에 넣지 않는다.
- 확인: `sha256sum -c` 대신 위 표의 값과 `sha256sum <파일>` 결과를 비교한다.
