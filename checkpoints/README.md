# 체크포인트

현재 계보의 묶음만 둔다. 메타데이터(README, `params/`, `SHA256SUMS.txt`, `metadata.json`, leaderboard)는 git에 있고,
가중치(`*.pt`, `*.onnx`)는 git 밖이다(.gitignore). 받은 가중치는 같은 경로에 두고 `sha256sum -c SHA256SUMS.txt`로 확인한다.

| 묶음 | 역할 |
|---|---|
| `teachers/bivt/ray_top1_17206_20261003` | BIVT-Ray 17206 (블라인드 d_v3.6.21_b1_18에서 시작, vendor_legacy 카메라) |
| `teachers/bivt/ray_gap_clean_vendor_new_top1_21068_20261005` | BIVT-Ray Clean 21068 (vendor_new). GAST v2.1 교사의 출발점, 현재 학생들의 교사 |
| `students/rvld/bivt_ray21068_student19008_20261006` | RVLD 학생 19008 (교사 21068) |
| `students/gavd/bivt_ray21068_student19008_20261007` | GAVD 학생 19008 (교사 21068) |

GAST v2.1 교사와 GAST 학생 19840 묶음은 확정되면 같은 형식으로 추가한다.
계보에서 빠진 예전 묶음(CVTT, BAVRL용 DWB-38000, BIVT-Ray 4500/7986/12123, Render v1, GAST 7901 등)은 10/07에 git에서 뺐다.
5090 PC의 `~/gd_project/archive/gd_lab_vrl_checkpoints_20261007/`(메타데이터)와 `~/gd_project/archive/gd_lab_vrl_20261007/`(가중치 포함)에 있다.
